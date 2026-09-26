# Reverie — High-Level Design

**The memory layer that learns which memories are worth having.**

*Status: v0.3 — partially implemented and measured · Owner: TBD · Last updated: 2026-08-12*

> ## Read this first
>
> **This document is a hypothesis. `experiments/README.md` is the evidence.** Where the
> two disagree, the experiments are right and this file is stale — say so rather than
> implementing what is written here.
>
> Substantial parts of the original design have been **overturned by measurement**, not
> by argument. The sections below are annotated inline, but the load-bearing reversals
> are:
>
> | Was | Now | Found by |
> |---|---|---|
> | Recall is one pipeline: seed → spread → rank (§10.1) | **Two channels.** Precision (exact precedent, bypasses the traversal) and association (spreading). The traversal *destroys* precedent lookup | E6c |
> | Salience gates consolidation cost (§8.3) | Salience gates **distillation only, never episodic retention**. Gating retention pinned the graph at ~10 nodes forever | E6b |
> | Dedup by embedding similarity (§8.5) | **Episodic nodes are never deduplicated.** They are events | E5 |
> | No-LLM path is "reduced quality" (§8.9) | Structurally different, and the defaults were tuned for a distiller it does not run | E5 |
> | `used_memories` is the highest-value integration ask (§9.4) | Citation credit is *worse* than activation credit | E2 |
> | Spreading activation beats top-k (§10.3) | True for multi-hop chains, **false and harmful for precedent lookup** | E3, E6c |
> | Retrieval beats recency "at scale" | Only where situations **recur**. Below ~2 recurrences per context, recency wins | E10 |
> | Attribution is the differentiator (§9) | **Refuted three times.** Human-taught procedural memory is the lever | E7, E9 |
> | Bad memory removes itself (G8, §8.8) | **Quarantine does not fire.** `forget` is the only reliable removal | E4 re-run |
>
> **Benchmark regime matters more than anything else in this document.** Every early
> negative result came from measuring in a task landscape smaller than a context
> window, where naive history-stuffing structurally cannot lose. The design's claim only
> becomes testable when the landscape exceeds what fits in a prompt.

---

## 1. Summary

Reverie is a memory service that sits alongside any agentic system. Agents write raw
experience to it during a task. Reverie turns that experience into retrievable memory
**out of band**. Agents recall relevant memory at the start of their next task.

> **The thesis changed in v0.4, and it changed because of measurement.**
>
> v0.1–v0.3 claimed the differentiator was *outcome attribution* — memories accumulating
> an empirical track record, with the good ones reinforced and the bad ones quarantined.
> That claim has now failed three independent tests (E7, E8, E9) and the constraint is
> arithmetic rather than fixable: total credit mass equals the number of outcomes, so a
> graph with more memories than outcomes cannot give any single memory enough evidence
> to rank on. See §9.
>
> What *does* work, measured: **precise retrieval, and letting a human teach.**

Two claims, both measured on `reverie-bench` at 600 tasks over a 72-context on-call
landscape:

| | |
|---|---|
| **Retrieval beats recency where situations recur** | 0.459 ±0.020 vs 0.402 ±0.022 for budget-matched history-stuffing, at 775 tokens against 1,146; paired over 16 seeds the difference is +0.056 ±0.032, significant. Reverie's curve rises with history; a recency window's is flat, because it cannot reach past itself. **The governing variable is repetition density, not scale** — measured at 8.3 tasks per context Reverie wins, at 2.0 it loses, at 1.0 it is at chance (E10). A landscape of genuine one-offs is not a fit and the README must say so |
| **Human review is the strongest lever in the system** | 0.640 ±0.019 at 50% review — paired lift +0.181 ±0.026 over Reverie alone, and *fewer* tokens (708). Monotonic in both how often you ask and how right the human is; +0.227 at 100% review |

The design principle that survived intact is the one about *when* work happens. Recall
is synchronous and fast because an agent is waiting. **Everything else is out of band.**
Consolidation, linking, community detection, and human review all happen between
sessions. Nothing Reverie does should make the current task slower; it exists to make
the next one better.

The name is the sleep metaphor and the metaphor is load-bearing: experience is cheap and
constant, consolidation is expensive and happens offline, and what survives the night is
a compressed, associative structure — not a transcript.

Reverie runs as a single process against a single SQLite file on a laptop, and as a
horizontally scaled service against Postgres in a cluster, with no code change in the
agent.

## 2. Prior art and positioning

This is a crowded space and the draft has to survive contact with it. An honest map, because anyone evaluating the project will make one anyway.

| System | Core idea | What it does not do |
|---|---|---|
| **Mem0** | Drop-in memory layer, hybrid vector + graph + KV, automatic fact extraction | No outcome feedback. Memories accumulate with equal weight forever; nothing demotes a memory that makes agents worse |
| **Zep / Graphiti** | Temporal knowledge graph with fact-validity windows; strong on "what was true when" | Tracks *truth over time*, not *usefulness*. A fact can be perfectly current and consistently unhelpful |
| **Letta (MemGPT)** | OS-style memory hierarchy; the agent self-manages its own context via tools | Agent-controlled, so quality is bounded by the agent's judgment about its own memory — the thing least worth trusting |
| **Cognee** | Graph+vector with a configurable "cognify" pipeline and many retrieval modes | Rich pipeline, no closed loop from task outcome back to memory weight |
| **Vestige** | Local-first MCP cognitive memory: FSRS decay, spreading activation, suppression | Closest neighbour on *mechanism*. Decay is time-and-use-based, not outcome-based — a memory used often and wrongly stays strong |
| **Hindsight** (vectorize-io) | "Agent memory that learns"; entities, temporal reasoning, SOTA on LongMemEval | Optimises conversational recall. LongMemEval measures answering questions about a transcript, not doing a job better the second time |
| **Research: ExpeL, Reflexion, MemRL, AttriMem** | Exactly this thesis — learn from outcomes, reinforce what worked | Papers and reference implementations, not deployable infrastructure. No storage layer, no MCP server, no ops story |

The two-sentence version for the README:

> Existing agent memory optimises **recall fidelity** — can the system retrieve the thing you told it. Reverie optimises **precedent precision**: given what this agent is doing right now, surface the specific past case that applies, and let a human correct the record when it is wrong.

> **v0.4 note.** The earlier version of this paragraph claimed the differentiator was
> *recall utility* — "did retrieving that thing make the next task go better", learned
> from outcomes. That is the attribution claim, and it did not survive measurement (§9).
> The honest differentiator is narrower and still real: **conjunctive precedent
> retrieval over a landscape larger than a context window, plus a human-teachable
> procedural layer.** Nothing in the comparison table does both.

**Where we are genuinely weaker.** Zep's temporal model is more rigorous than ours and we should say so. Mem0's integration surface is far broader. Letta's agent-controlled memory is better when the agent genuinely knows what it needs. Reverie is the right choice when an agent does *repeated work with checkable outcomes* — deploys, migrations, bug fixes, integrations, data pipelines. It is the wrong choice for a chatbot that needs to remember your dog's name, and the README should say that out loud. Claiming the whole space is how a project loses credibility in the first comment thread.

---

## 3. Goals and non-goals

### Goals

| # | Goal |
|---|---|
| G1 | Agent-agnostic. Any framework (LangGraph, CrewAI, Claude Code, a bare `while` loop) integrates in under 20 lines |
| G2 | Writes are cheap and non-blocking. An agent never waits on consolidation |
| G3 | Recall is budgeted. A recall returns a bounded, compressed brief — never a document dump |
| G4 | Measurably improves task performance over repeated runs. Ships with the harness that proves it |
| G5 | Runs fully local: no API key, no external service, no container orchestration |
| G6 | Same binary deploys to cloud with a swapped storage and model backend |
| G7 | Memory is inspectable, traceable, and correctable. A human can read, trace, edit, and delete any memory |
| G8 | Memory can be shown to be *wrong* and removed automatically, without a human noticing first. ❌ **Not satisfied** — quarantine does not fire (E4). A human `forget` is currently the only reliable removal path |

G8 is new in v0.2 and is the goal that distinguishes the project. Everything from §9 exists to serve it.

### Non-goals

- **Not a RAG system over documents.** Reverie stores what an agent *did* and what happened, not a corpus. Point it at your docs and you will be disappointed.
- **Not a conversation-history store.** Short-term context management is the agent framework's job.
- **Not a vector database.** Embeddings are how you enter the graph, not how the graph is represented.
- **Not a prompt manager, eval platform, or observability tool.** Adjacent to all three; none of them.
- **Not a general knowledge graph.** We model experience, not the world.
- **v1 is single-tenant per deployment.** No org/billing/RBAC layer.

---

## 4. Design principles

1. **Write cheap, think expensive, think later.** The hot path is an append. All
   reasoning happens in a background worker. This is the principle that has survived
   every revision, and it now carries more weight than it did: **nothing Reverie does
   may slow the session it is observing.** Consolidation, linking, abstraction, and
   human review are all out of band. The product is a better *next* run, never a
   modified current one. The single exception is recall itself, which an agent waits
   on — hence the latency budget in §10.2 and the rule that no model call may appear
   on that path.
2. **Provenance over confidence.** Every stored claim carries how it was derived — observed, inferred, asserted, or ambiguous. Users can always separate what was seen from what was guessed.
3. **Precision over recall.** Injecting 3k tokens of marginally relevant history makes an agent worse. A recall that returns nothing is a valid and frequently correct outcome.
4. **Never trust self-report.** An LLM asked whether it succeeded says yes far too often. Ground truth first, human signal second, behavioural proxy third, LLM judgment last.
5. **Memory must be able to be wrong.** Contradiction, decay, quarantine, and deletion are first-class operations, not cleanup scripts.
6. **Nothing is believed because it is stored.** A memory earns its rank from its track record, not from having been written down confidently.
7. **One file, no daemon, for the default case.** Adoption dies at `docker-compose up`.

---

## 5. System architecture

### 5.1 Components

| Component | Responsibility | Runs |
|---|---|---|
| **Ingest API** | Accepts episodes and outcome reports. Validates, assigns IDs, appends to buffer. Returns immediately | Hot path, in-process or HTTP |
| **Episode buffer** | Append-only, unprocessed experience. Durable. Cheap to write, never read by agents | Storage layer |
| **Consolidation worker** | The dream cycle. Batch job: segment → score → distill → link → reconcile → abstract → prune | Background, scheduled or triggered |
| **Memory graph** | Typed nodes, weighted edges, communities. The durable store | Storage layer |
| **Recall engine** | Cue → seed → spreading activation → rank → compress → brief. Synchronous, latency-sensitive | Hot path |
| **Attribution engine** | Maps outcome reports back onto the memories that participated in a recall. Updates utility posteriors | Background, cheap |
| **Model gateway** | Abstraction over the LLM used for distillation and judgment. Ollama, Anthropic, OpenAI, or none | Consolidation only — never the hot path |
| **Storage adapter** | `MemoryStore` interface. SQLite and Postgres implementations | — |
| **Interfaces** | MCP server, Python/TS SDK, REST API, CLI | — |

### 5.2 Process topology

**Local (default).** One process. Embedded SQLite in WAL mode. Consolidation runs as an in-process scheduled task or via `reverie consolidate`. Model gateway points at Ollama or is disabled entirely (see §8.8 for the no-LLM path).

**Cloud.** Stateless API pods behind a load balancer, Postgres + pgvector, consolidation workers as a separate deployment consuming a work queue, model gateway pointing at a hosted API. The `MemoryStore` interface is the only seam that changes.

### 5.3 Data flow

```
  Agent task starts
      │
      ├──▶ recall(cue, budget) ──▶ Recall engine ──▶ brief + recall_id
      │                                (reads memory graph, writes recall log)
      │
      ├──▶ [agent works, emits observations]
      │         └──▶ remember(episode) ──▶ Ingest API ──▶ episode buffer
      │
      └──▶ report_outcome(recall_id, signal) ──▶ Ingest API ──▶ outcome buffer

  ... later, offline ...

  Consolidation worker            Attribution engine
      ├── reads episode buffer        ├── reads outcome buffer + recall log
      ├── writes memory graph         └── updates utility posteriors on nodes
      └── marks episodes consolidated
```

Attribution runs independently of consolidation and far more often — it is a cheap arithmetic update with no model call, and outcomes should move rankings within minutes, not overnight.

---

## 6. Data model

### 6.1 Memory types

Four types, one graph, different write paths and lifetimes.

| Type | Content | Volume | Decay | Recall trigger |
|---|---|---|---|---|
| **Episodic** | What happened, timestamped, near-raw | High | Fast (τ ≈ 14d) | Rarely direct; substrate for distillation and for "have I done this before" |
| **Semantic** | Facts about the environment. "Staging Postgres is a read replica" | Low | Slow (τ ≈ 180d) | Entity match in the cue |
| **Procedural** | Conditioned actions. "When X, do Y." Carries a success rate | Medium | Medium, reinforced on use | Situation match; the primary payload of a recall |
| **Valence** | Scalar affect attached to an entity or approach | Low | Slow, asymmetric (negative decays slower) | Any activation of the attached node |

Valence is the least conventional piece. It is a single float in `[-1, 1]` on a node, not a memory record — cheap to store, cheap to retrieve, and functioning as a prior that biases search before any structured lesson is read. The behavioural target: an agent that has burned hours on a library twice should feel drag on that node immediately, ahead of retrieving the reasons. It is also the piece most likely to be cut; see §17.3.

### 6.2 Graph schema

```sql
-- Schema version ---------------------------------------------------------
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- rows: schema_version, embedding_model, embedding_dim, created_at

-- Nodes ------------------------------------------------------------------
CREATE TABLE nodes (
  id            TEXT PRIMARY KEY,          -- uuidv7, time-sortable
  scope_id      TEXT NOT NULL,             -- namespace: agent, team, or global
  type          TEXT NOT NULL,             -- entity|episode|lesson|failure_mode|
                                           -- tool|artifact|task_type|concept|theme
  memory_class  TEXT NOT NULL,             -- episodic|semantic|procedural
  label         TEXT NOT NULL,             -- short canonical name
  body          TEXT,                      -- the memory content
  valence       REAL NOT NULL DEFAULT 0.0, -- [-1, 1]

  -- provenance
  provenance    TEXT NOT NULL,             -- observed|inferred|asserted|ambiguous
  source_ids    TEXT,                      -- JSON array of episode ids

  -- lifecycle
  state         TEXT NOT NULL DEFAULT 'active',
                -- active|superseded|contested|stale|quarantined|merged|deleted
  superseded_by TEXT REFERENCES nodes(id),
  merged_into   TEXT REFERENCES nodes(id), -- set when dedup folds this into another

  -- context binding: memory learned under one model may not transfer to another
  learned_under TEXT,                      -- JSON {model, agent_version, env_hash}

  -- community (see §8.7)
  community_id  TEXT,
  centrality    REAL NOT NULL DEFAULT 0.0,

  -- utility posterior (Beta), time-decayed
  alpha         REAL NOT NULL DEFAULT 1.0,
  beta          REAL NOT NULL DEFAULT 1.0,
  attributions  INTEGER NOT NULL DEFAULT 0,
  utility_at    INTEGER NOT NULL,          -- last decay application

  -- activation stats
  strength      REAL NOT NULL DEFAULT 1.0,
  activations   INTEGER NOT NULL DEFAULT 0,
  last_activated INTEGER,

  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

CREATE INDEX idx_nodes_scope_state ON nodes(scope_id, state, type);
CREATE INDEX idx_nodes_label       ON nodes(scope_id, label);
CREATE INDEX idx_nodes_community   ON nodes(scope_id, community_id, centrality DESC);

-- Edges ------------------------------------------------------------------
CREATE TABLE edges (
  id           TEXT PRIMARY KEY,
  scope_id     TEXT NOT NULL,
  src          TEXT NOT NULL REFERENCES nodes(id),
  dst          TEXT NOT NULL REFERENCES nodes(id),
  relation     TEXT NOT NULL,   -- caused|preceded|applies_to|contradicts|
                                -- resolved_by|part_of|similar_to|learned_from|
                                -- summarizes
  weight       REAL NOT NULL DEFAULT 0.5,   -- [0,1], traversal conductance
  provenance   TEXT NOT NULL,               -- observed|inferred|asserted|ambiguous
  confidence   REAL NOT NULL DEFAULT 0.5,
  traversals   INTEGER NOT NULL DEFAULT 0,
  last_traversed INTEGER,
  created_at   INTEGER NOT NULL
);

CREATE INDEX idx_edges_src ON edges(scope_id, src, weight DESC);
CREATE INDEX idx_edges_dst ON edges(scope_id, dst, weight DESC);

-- Communities (see §8.7) --------------------------------------------------
CREATE TABLE communities (
  id           TEXT PRIMARY KEY,
  scope_id     TEXT NOT NULL,
  label        TEXT,             -- LLM-generated, 2-5 words
  summary      TEXT,             -- compressed brief of the whole community
  cohesion     REAL,
  node_count   INTEGER,
  prev_id      TEXT,             -- lineage across re-clustering runs
  updated_at   INTEGER NOT NULL
);

-- Embeddings (entry points only, not the graph representation) ------------
CREATE VIRTUAL TABLE node_vec USING vec0(
  node_id   TEXT PRIMARY KEY,
  embedding FLOAT[768]
);
-- The model that produced these lives in meta.embedding_model. Changing it
-- invalidates every row; see §15.3.

-- Full-text (the other entry point) ---------------------------------------
CREATE VIRTUAL TABLE node_fts USING fts5(node_id, label, body);

-- Recall log — the spine of attribution -----------------------------------
CREATE TABLE recalls (
  id            TEXT PRIMARY KEY,     -- returned to the agent as recall_id
  scope_id      TEXT NOT NULL,
  cue           TEXT NOT NULL,
  node_ids      TEXT NOT NULL,        -- JSON array, ordered by rank
  activations   TEXT NOT NULL,        -- JSON map node_id -> activation score
  withheld_ids  TEXT,                 -- JSON array; ablation arm (§9.4)
  is_ablation   INTEGER NOT NULL DEFAULT 0,
  token_cost    INTEGER,
  latency_ms    INTEGER,
  created_at    INTEGER NOT NULL
);

CREATE INDEX idx_recalls_created ON recalls(scope_id, created_at);

-- Outcomes ----------------------------------------------------------------
CREATE TABLE outcomes (
  id           TEXT PRIMARY KEY,
  recall_id    TEXT REFERENCES recalls(id),
  episode_id   TEXT,
  idem_key     TEXT UNIQUE,          -- caller-supplied; makes retries safe
  signal_tier  INTEGER NOT NULL,     -- 1=ground truth .. 4=llm judgment
  outcome      TEXT NOT NULL,        -- success|failure|partial|abandoned
  detail       TEXT,
  attributed   INTEGER NOT NULL DEFAULT 0,
  created_at   INTEGER NOT NULL
);

-- Episode buffer ----------------------------------------------------------
CREATE TABLE episode_buffer (
  id            TEXT PRIMARY KEY,
  scope_id      TEXT NOT NULL,
  client_key    TEXT,                -- caller idempotency key
  payload       TEXT NOT NULL,       -- validated episode JSON
  consolidated  INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL,
  UNIQUE(scope_id, client_key)
);
```

The graph is deliberately stored as plain relational tables. Traversal happens in application code over the indexed edge table. This is fast well past several million edges, avoids a Neo4j-class dependency, and — more importantly — lets us express decay, weighting, and quarantine logic that no graph query language handles cleanly.

### 6.3 Episode contract

The most important schema in the system, because everything downstream is bounded by what the agent can tell us.

```json
{
  "schema_version": 1,
  "episode_id": "uuidv7",
  "client_key": "task-8891:attempt-1",
  "scope_id": "agent:deploy-bot",
  "agent": {
    "model": "claude-opus-5",
    "version": "deploy-bot@2.3.1",
    "env_hash": "sha256:9f1c..."
  },
  "task": {
    "id": "task-8891",
    "description": "Migrate the orders table to add a shipping_status column",
    "type": "db_migration",
    "recall_id": "rc_01J..."
  },
  "steps": [
    {
      "kind": "tool_call",
      "name": "bash",
      "input": "alembic upgrade head",
      "output_summary": "hung for 300s, killed",
      "error": "TimeoutError",
      "ts": 1754820000
    }
  ],
  "entities": ["alembic", "orders_table", "staging_db"],
  "used_memories": ["mem_01J8...", "mem_01J9..."],
  "outcome": {
    "status": "failure",
    "signal_tier": 1,
    "evidence": "exit_code=124",
    "detail": "migration timed out"
  },
  "duration_ms": 341000,
  "ts": 1754820341
}
```

Notes on the contract:

- `client_key` makes ingest idempotent. Agents retry; without this, one flaky network call becomes two episodes and double-counted evidence.
- `agent.model` and `agent.env_hash` bind a memory to the conditions it was learned under. A procedural memory that works for one model may be useless or harmful for another, and an environment fingerprint is what makes staleness detectable rather than guessable.
- `entities` is optional; consolidation extracts them if absent. Supplying them raises quality substantially and costs the agent nothing.
- `used_memories` is optional and is the single highest-value field an integrator can populate — it converts attribution from correlational to near-exact (§9.4).
- `outcome` is optional at episode time; it can arrive later via `report_outcome`, which is the common case for async signals like CI results.
- `signal_tier` is required whenever an outcome is present. See §9.1.
- `output_summary` rather than `output`. We truncate hard at ingest; storing full tool output is the fastest route to a 40GB SQLite file.

---

## 7. Ingestion

Hot path. Target p99 under 5ms local, under 30ms over HTTP.

```
remember(episode) →
  1. validate against schema (reject, don't coerce)
  2. redact (§14.1) — before anything is written to disk
  3. truncate oversized fields (output_summary → 2KB, steps → 200)
  4. assign uuidv7, stamp scope
  5. INSERT INTO episode_buffer  (ON CONFLICT client_key DO NOTHING)
  6. return {episode_id, deduplicated: bool}
```

No embedding, no LLM call, no graph write. If the consolidation worker is down, ingestion continues to succeed and the backlog drains later. This is intentional: memory degrading to "nothing was learned today" is acceptable; memory blocking an agent mid-task is not.

**Backpressure.** If the buffer exceeds a configured high-water mark, ingestion begins sampling — keeping all failures and a decreasing fraction of successes. Failures are worth more than successes for distillation, and dropping them is never correct.

**Sampling bias, and the correction.** That policy deliberately skews the buffer toward failure, which would corrupt any base rate computed from it — including the tier-3 behavioural baselines in §9.1 and the "N of M runs succeeded" figures in briefs. So ingest maintains an unsampled counter table of `(scope, task_type, outcome) → count`, incremented for *every* episode including dropped ones. All base rates are read from the counters, never from the buffer. This is a small table and a large correctness win.

---

## 8. Consolidation — the dream cycle

Background. Triggered by schedule, by buffer depth, by explicit CLI invocation, or by a session-end hook. Idempotent and resumable; episodes are marked consolidated only on commit.

### 8.1 Pipeline

```
segment → score → distill → link → reconcile → abstract → prune
```

`abstract` is new in v0.2 and is where the associative structure actually forms. See §8.7.

### 8.2 Segment

Group buffered episodes into consolidation units. Default unit is the task. Long tasks are split at natural boundaries: outcome transitions, tool-family changes, or long time gaps. Related episodes across tasks that share entities are batched together so distillation sees repetition — repetition is what turns an anecdote into a lesson.

### 8.3 Score

Assign each unit a **salience** score determining how much distillation budget it gets. Cheap, deterministic, no LLM:

```
salience = w1·outcome_extremity      # clear failures and clean successes both score high
         + w2·novelty                # cosine distance to nearest existing episodic node
         + w3·cost                   # duration, token spend, retry count
         + w4·entity_centrality      # touches high-degree nodes in the graph
         - w5·redundancy             # near-duplicate of something already consolidated
```

Units below threshold are archived without distillation. On a typical day most episodes are boring and should cost nothing. This is the primary lever on consolidation cost.

> **Implemented, with a hard constraint discovered by E6b.** Salience gates the *model
> call* and nothing else. Low-salience units still run the free structured path
> (`NullDistiller` → `link`) and still record their episodes.
>
> Letting salience gate retention was catastrophic. Salience is scored per *unit* —
> everything buffered since the last cycle — and `novelty` is cosine distance over the
> unit's concatenated text. Ten on-call incidents covering ten service×symptom pairs
> *never seen before* still look near-identical in aggregate to the previous ten.
> Novelty collapsed, every batch after the first fell below threshold, and the graph sat
> at ~10 episode nodes no matter how much experience accumulated. Retrieval was then
> searching a store that did not contain the answer, which accounted for every negative
> retrieval result recorded before it was found.
>
> `units_distilled` counts only units that spent model tokens, so it stays a cost
> metric. Regression test: `test_low_salience_still_records_episodes`.

### 8.4 Distill

The one LLM-dependent stage. For each surviving unit, extract candidate memories under a strict output schema:

```json
{
  "semantic": [{"label": "...", "body": "...", "entities": [...],
                "cites": [step_idx]}],
  "procedural": [{"condition": "...", "action": "...", "rationale": "...",
                  "entities": [...], "cites": [step_idx]}],
  "failure_modes": [{"label": "...", "symptom": "...", "root_cause": "...",
                     "resolution": "...", "cites": [step_idx]}],
  "valence": [{"entity": "...", "delta": -0.3, "reason": "..."}],
  "entities": [{"label": "...", "type": "...", "aliases": [...]}]
}
```

Constraints enforced in the prompt and validated after:

- **Every candidate must cite the step indices it derives from.** Uncited candidates are dropped. This is the cheapest hallucination filter available.
- **No candidate may generalise beyond a single observation** unless the unit contains ≥2 corroborating instances. Single observations become episodic nodes with `provenance: observed`, not lessons.
- **Procedural memories need a machine-checkable-ish condition.** "When things go wrong, be careful" is rejected by a heuristic filter on condition specificity.
- **Nothing imperative aimed at the agent.** See §14.2 — this is a security control, not a style rule.

### 8.5 Link

Insert candidates into the graph:

1. **Resolve entities.** Exact label match → alias match → embedding match above threshold → create new. Entity resolution errors are the main source of graph fragmentation; when in doubt, create separately and let a later merge pass join them.
2. **Deduplicate memories — semantic and procedural only.** Cosine similarity above threshold plus same entity set → reinforce the existing node (bump `alpha`, `strength`, `activations`) rather than insert.

   > **Episodic nodes are never deduplicated** (E5). An episode is an *event*, and two
   > things that happened are two things that happened. Dedup is embedding-based, and
   > the discriminative part of an episode is often a two-token difference inside an
   > otherwise identical sentence — `strategy=direct → failure` versus
   > `strategy=sql_first → success` — which scores far above the 0.88 threshold. Forty
   > episodes were collapsing into a 20-node graph, averaging away precisely the
   > distinction the agent needed. Episodic volume is bounded by decay and pruning
   > (§8.8), which is the mechanism designed for it.
 Reinforcement, not duplication, is how repeated experience registers. The absorbed node is not deleted — it is marked `merged` with `merged_into` set, because live recall logs still point at it (§9.5).
3. **Create edges.** `caused`, `preceded`, `resolved_by` come from step ordering and are `observed`. `applies_to`, `similar_to` come from the model or embeddings and are `inferred`. Where the model itself is unsure, `ambiguous`. Never blur the three.
4. **Apply valence deltas** to entity nodes, clamped, with a per-episode cap so a single catastrophic run cannot permanently poison a node.

### 8.6 Reconcile

Contradiction handling. For each new candidate, search for existing active memories with conflicting content over the same entity set.

| Situation | Action |
|---|---|
| New memory has strictly better evidence (higher signal tier, more recent, more corroboration) | Mark old `superseded`, link `superseded_by` |
| Evidence is comparable | Mark both `contested`. Contested memories are still recalled, but the brief presents both and flags the conflict |
| New memory conflicts with a human-`asserted` memory | New memory is rejected and logged. Humans win |
| Old memory's `env_hash` no longer matches current | Old marked `stale`, flagged "re-verify" in any recall |

Contested state is a feature. An agent told "these two things conflict, here's both" behaves better than one told a confident falsehood.

### 8.7 Abstract — schema formation

**New in v0.2, and the stage that makes this an associative memory rather than a tagged store.**

Distillation produces lessons about individual episodes. What it cannot produce is the observation that thirty scattered memories are all really about one thing. Human consolidation does exactly that — repeated episodes collapse into schemas, and the schema is what gets recalled. This stage is the mechanical analogue, and it borrows its shape from graph-analysis tooling like [graphify](https://github.com/Graphify-Labs/graphify).

Runs on a slower cadence than the rest of the pipeline (default: every 20 consolidation cycles, or when node count grows 15% since the last run).

1. **Community detection.** Louvain over the weighted edge graph, per scope. Produces clusters of densely interconnected memories — in practice these come out as coherent themes: "staging DB quirks", "CI flakiness", "auth service integration".
2. **Stable community identity.** Naive re-clustering renumbers everything each run, which would make community IDs useless for anything persistent. Match each new community to the previous run's by maximum node overlap (Jaccard above threshold), inherit its ID, and record `prev_id`. Only genuinely new clusters get new IDs.
3. **Community summary.** One cheap LLM call per changed community: a 2–5 word label and a compressed summary of what the cluster collectively knows. Stored on the `communities` row.
4. **Theme nodes.** Each community gets a node of `type: theme`, `provenance: inferred`, linked to its members with `summarizes` edges. Theme nodes are recallable in their own right.
5. **Hub identification.** Compute betweenness centrality; store on the node. High-centrality nodes are the ones everything routes through — the `staging_db` entity that half the graph touches.

Three things this buys, in descending order of confidence that they are worth the cost:

- **Budget-efficient broad recall.** When a cue is broad and activation spreads thin across many low-scoring nodes in one community, returning the theme summary costs ~80 tokens instead of ~600 for the members. Direct win on the §10 token budget.
- **Human legibility.** `reverie stats` can say "this agent has 7 knowledge communities, largest is *Staging Database Quirks* (34 memories, cohesion 0.71)" instead of "4,812 nodes." This is what makes the project screenshot-able, which matters more for adoption than it should.
- **Traversal shortcuts.** Intra-community hops can carry lower decay than cross-community hops, on the theory that a same-theme association is more likely relevant. Speculative; gate it behind an ablation before believing it.

**Costs, stated plainly.** Clustering is O(E log V) and runs over the whole scope, so it is the most expensive non-LLM operation in the system. Summaries cost one LLM call per changed community. Theme nodes are `inferred` and will sometimes be wrong, so they must never outrank an `observed` memory — enforced by the provenance factor in ranking (§10.1).

### 8.8 Prune

- **Decay strength**: `strength ← strength · exp(-Δt / τ_class)`.
- **Decay utility counts**: `alpha ← 1 + (alpha−1)·λ`, same for beta. Old evidence stops dominating.
- **Drop** episodic nodes below the strength floor with no incoming edges from semantic/procedural nodes.
- **Quarantine** nodes whose utility lower bound falls below threshold after ≥N attributions. Excluded from recall, retained for audit, revivable by human assertion.
- **Reap recall logs** past the retention window with no unattributed outcomes outstanding (§15.4).
- **Compact**: merge converged entity nodes, rebuild FTS, vacuum.

### 8.9 The degraded no-LLM path

If no model is configured, the pipeline runs segment → score → link → reconcile → abstract → prune, skipping distill and community summarisation (communities still form; they just get IDs instead of names). Episodic nodes and the entity graph are still built from structured episode fields, and valence still accrues from outcomes. Recall works, at reduced quality.

> **"Reduced quality" understated this (E5).** The no-LLM path produces *no* procedural
> or semantic lessons at all — only entities, failure modes, and episodes. The recall
> defaults reserved 50% of the token budget for procedural memories and capped episodic
> at 20%, so half the budget sat held for classes that could not exist while the only
> real evidence starved. It delivered 0.2 evidence items per task against naive
> replay's 19.5.
>
> Fixed by `adaptive_quotas`: configured fractions are renormalised over the classes
> actually present in the candidate pool. Any future stage that changes *what kinds* of
> node get created must check the quota interaction — this is the second time a default
> tuned for one configuration silently crippled another.

Reverie being useful with zero API key and no local model matters enormously for adoption, and this path must be tested in CI rather than assumed to work.

### 8.10 Human review — the strongest lever measured

**Added in v0.4.** At a session boundary the agent may ask its operator how the session
went and, more importantly, *what should have been done*. The answer is written as a
**procedural memory** (`assert_fact(..., memory_class="procedural")`, provenance
`asserted`).

This is the substitute for the LLM distiller (M2), and on one axis it is strictly
better: a model reading the transcript can only summarise what happened, whereas a human
supplies the **counterfactual** — the thing that is by construction absent from every
episode. Measured at 600 tasks, 8 seeds:

| review frequency | accuracy | vs no review |
|---|---|---|
| 5% of sessions | 0.492 | +0.034 |
| 10% | 0.503 | +0.045 |
| 25% | 0.541 | +0.083 |
| 50% | 0.629 | +0.171 |
| 100% | **0.685** | **+0.227** |

Behaviour is also monotonic in how *right* the human is (0.462 → 0.616 as lesson
accuracy goes 0 → 1.0), which is the check that the content is genuinely being used
rather than the node count perturbing something else. An earlier build passed the
frequency test and failed the accuracy test — lessons were reaching the brief without
being applied — and that was a retrieval bug, not a feature (§10.1).

Three constraints, all of which follow from §4.1:

1. **Review never blocks the agent.** Asking happens between sessions and the answer
   improves the next run. A reviewer's correction that stalls the current task has
   already cost more than it can return.
2. **Ask rarely, and ask well.** Benefit is monotonic in frequency and has not saturated
   at 100%, but real reviewers stop answering. Gate on salience (§8.3) — ask after
   novel, expensive, or failed sessions — which also sidesteps the unreliable
   session-end signal in §19.
3. **A human comment is the most dangerous input in the system.** §8.6 lets asserted
   memories win all conflicts and §13 treats brief content as untrusted data; a
   free-text lesson bypasses both. Human input is trusted about *facts* and must not be
   licensed to issue *instructions* — same schema constraints and imperative-content
   filter as distiller output.

---

## 9. Attribution — how memory learns

This section is the project. Everything else is table stakes.

> **Scope correction from E7 — attribution does not apply to all memory classes.**
>
> Measured lift under working retrieval, 600 tasks, 8 seeds: **+0.003 static, +0.009
> under environment drift**, against a difference band of ±0.049. Zero both times.
>
> The cause is structural, not a tuning failure. Over 600 episodic nodes the median node
> receives **one** attribution, mean evidence mass is 0.23, and the utility LCB at p25,
> median and p75 is *exactly* 0.250 — `utility_lcb(1, 1, 0.25)`, the untouched prior.
> Three quarters of the graph never leaves it.
>
> A Beta posterior needs repeated participation. Attribution assumes **few nodes,
> recalled many times each**; episodic memory is **many nodes, recalled once each**.
> Independently: for an episode the outcome *is* the content — a node reading
> `strategy=rollback — failure` states its own result, so the posterior is a weaker
> second encoding of what the agent already reads from the brief.
>
> Everything below therefore applies to **semantic and procedural memories only** —
> lessons that are recalled repeatedly and assert something whose outcome is not in the
> text. Episodic nodes should be excluded from the posterior machinery: it cannot pay
> off for them, and it pollutes the peer distribution that quarantine reads from.
>
> **The claim in §1 remains untested**, and the blocking dependency is the LLM distiller
> (M2), not more attribution work. Do not claim attribution lift in either direction
> until real procedural memories exist to rank.

### 9.1 Signal tiers

Ranked. Consolidation and attribution both weight by tier.

| Tier | Source | Examples | Weight |
|---|---|---|---|
| 1 | Ground truth | tests pass/fail, exit code, CI status, exception thrown, HTTP status | 1.0 |
| 2 | Human signal | user rejected output, user retried, thumbs-down, PR closed unmerged | 0.8 |
| 3 | Behavioural proxy | task took 4× the median for its type, agent looped, retry-count spike | 0.4 |
| 4 | LLM judgment | a judge model's assessment | 0.15 |

Tier 4 is a fallback and is weighted low on purpose. Self-assessed success rates from LLM agents are inflated to the point of being nearly uninformative, and a system that reinforces on them will confidently learn nonsense. The SDK must make tier-1 signals trivially easy to attach — `report_outcome(rid, exit_code=p.returncode)` — because the quality of the entire system is bounded here.

Tier 3 requires a per-`task_type` baseline distribution (median duration, median step count) to compare against. These are computed from the unsampled counter table (§7), maintained incrementally, and are unavailable until a task type has ≥20 completed instances. Before that, tier 3 is not emitted at all rather than emitted against a garbage baseline.

### 9.2 Utility update

Each memory node holds a Beta posterior over "did participating in a recall correlate with success."

When an outcome arrives for `recall_id`:

```python
for node_id, activation in recall.activations.items():
    node_id = resolve_merges(node_id)          # follow merged_into (§9.5)
    credit  = activation / sum(recall.activations.values())
    w       = TIER_WEIGHT[outcome.signal_tier] * credit

    if outcome.outcome == "success":  node.alpha += w
    elif outcome.outcome == "failure": node.beta  += w
    elif outcome.outcome == "partial": node.alpha += w/2; node.beta += w/2
    # abandoned → no update; we learn nothing from a task nobody finished
    node.attributions += 1
```

Ranking uses the **lower confidence bound** of the posterior, not the mean:

```
utility = beta_ppf(0.25, alpha, beta)
```

This keeps a memory that succeeded once from outranking one that succeeded 40 times out of 45, which the mean would allow.

### 9.3 Outcome idempotency and conflicts

Outcomes arrive over unreliable channels and sometimes more than once.

- Duplicate outcomes are rejected on `idem_key`. Callers that omit it get a synthesised key of `(recall_id, outcome, signal_tier)`.
- Multiple *distinct* outcomes for one `recall_id` are legal and expected — a fast tier-3 proxy followed by a slow tier-1 CI result. Both are applied; the tier weighting means the ground-truth signal dominates naturally, with no special-casing.
- If a later, higher-tier outcome **contradicts** an earlier lower-tier one, the earlier update is reversed before the new one is applied. The recall log retains both for audit. Without this, a wrong tier-3 guess is permanently baked into the posterior.

### 9.4 The known weakness: correlation is not causation

A memory can be recalled, be entirely irrelevant, and collect credit for a success it had nothing to do with. This is the central epistemic problem and it should be stated in the README, not buried.

Mitigations, in order of cost:

- **Credit weighting by activation share** (§9.2). Cheap, partial. A barely-activated memory earns barely any credit.
- **Explicit agent citation.** The `used_memories` field on the episode. ~~The single highest-value field an integrator can populate.~~
  > **Overturned by E2.** Citation credit scores ρ = 0.614 against activation credit's
  > 0.644 — measurably *worse*, not better. Zeroing credit for memories the agent did
  > not cite means those memories never accumulate evidence at all, so they sit at the
  > prior forever and can never be ranked. Diffuse, slightly-wrong credit is more
  > informative than sharp, narrow credit. Keep the field for offline analysis; do not
  > build the SDK around it and do not make it the primary integration ask.
- **Recall ablation sampling.** On a configurable fraction of recalls (default 5%), withhold the lowest-ranked memories, record them in `withheld_ids`, and compare outcome rates between arms. This is the only mechanism here that yields a genuinely causal signal, at the cost of slightly degraded recalls on a small slice of traffic. It is also, not coincidentally, the mechanism that produces the headline benchmark number.

Ablation deserves more design attention than it got in v0.1. Withholding the *lowest-ranked* memories tests the least interesting hypothesis — those memories probably do not matter and the ablation arm will look identical. A stratified variant that occasionally withholds a *top-ranked* memory produces far more information per ablated recall, at higher cost to that individual task. Default to lowest-rank withholding; expose top-rank ablation as an opt-in for users running the benchmark.

### 9.5 Attribution across a mutating graph

Consolidation merges, supersedes, and prunes nodes. Recall logs written before a merge point at node IDs that no longer independently exist, and an outcome can easily arrive after the node it refers to has been folded into another. Silently dropping those updates would bias attribution toward whatever happened not to get merged.

The rules:

- Merged nodes are tombstoned (`state='merged'`, `merged_into` set), never deleted. Attribution follows the pointer, transitively, with a cycle guard.
- Superseded nodes still accept attribution. A memory that was recalled and then replaced should still record how it performed, because that record is the evidence for whether superseding it was right.
- Quarantined nodes still accept attribution — that is how a quarantined memory can eventually be exonerated.
- Deleted nodes (human `forget`) drop the update on the floor and log it. Deletion means deletion.
- Nodes are only hard-deleted after the recall-log retention window has passed, so the tombstone table cannot grow without bound.

---

## 10. Recall

Hot path. Target p99 under 100ms local. The latency budget shapes the whole design.

### 10.1 Algorithm — two channels

> **This section was rewritten after E6c.** v0.2 described a single pipeline
> (seed → spread → rank). That pipeline is correct for *association* and actively
> destroys *precedent lookup*, so recall now runs two independent channels.
>
> The evidence: on a 72-context on-call landscape the correct precedent was present in
> the graph, the seed stage ranked it **first at score 1.000**, and it reached the brief
> **4% of the time**. Activation accumulates on hub nodes over many paths until it
> outranks a correctly-seeded leaf — the traversal reliably discarded the one node that
> already had the answer. Adding the precision channel took that to **77%**, against
> naive recency's 42%.

```python
def recall(cue, scope, budget_tokens=1200):
    entities = resolve_cue_entities(cue, scope)   # exact label / alias hits

    # ---- CHANNEL A: PRECISION -------------------------------------------
    # Exact precedents, admitted on their own authority. Never traversed.
    direct = {}
    for node, matched in conjunctive_neighbours(entities, scope).items():
        if len(matched) < 2:
            continue                              # single-entity is the flood case
        coverage    = idf_weighted(matched) / idf_weighted(entities)
        specificity = len(matched) / entity_degree(node)   # hub penalty
        if coverage >= DIRECT_MIN_COVERAGE:       # ADMISSION: cue coverage only
            direct[node] = coverage**2 * specificity       # RANKING: composite

    # ---- CHANNEL B: ASSOCIATION -----------------------------------------
    # Unchanged. Finds the load-bearing fact N hops from a dissimilar cue.
    seeds      = fuse(fts(cue), vector(embed(cue)), entities,
                      conjunctive_boost=direct)   # also nudges the seed set
    activation = spread(seeds, max_hops=3, fanout=12, visit_cap=400)
    scored     = [rank(n, activation[n]) for n in load(activation)]
    scored     = collapse_communities(scored)

    # ---- SELECT ----------------------------------------------------------
    # Precision gets a reserved slice FIRST. Without the reservation these lose
    # to whatever the traversal inflated, which is the failure being routed around.
    selected  = take(direct, budget=budget_tokens * DIRECT_RESERVE)
    selected += knapsack(scored, budget_tokens - spent(selected),
                         quotas=adaptive(scored), max_contested=2)

    return render(selected), log_recall(cue, selected, activation)
```

**Admission and ranking are different questions, and conflating them empties the
channel.** Three failed attempts, all worth recording because each looked correct:

| Attempt | Result | Why |
|---|---|---|
| Gate on the composite score | **0 admissions** | Specificity divides by entity degree, so a perfect match scored 0.375 against a 0.5 threshold purely for being attached to eight entities rather than three |
| Specificity over *total* degree | worse than no channel | The `preceded` temporal spine counted against episodes — they were penalised for having a past and a future |
| No specificity term | exact match ties with junk | Generic `failure_mode` hubs ("restart did not mitigate", linked to everything) reach full cue coverage trivially and tied at 1.000 |

The rule that works: **admit on coverage of the cue, rank by coverage × specificity.**
Whether a memory is *about what was asked* is a property of the cue; how distinctive it
is among such memories is a ranking concern.

`max_nodes` is derived from `budget_tokens` rather than fixed. A hard cap of 12 at ~25
tokens/node bound at ~300 tokens and made `budget_tokens` completely inert — capture was
flat from 200 to 1200 tokens (E5).

### 10.2 Latency, honestly

Three hops at fanout 12 is up to 1,728 edge expansions, and a naive implementation issues a query per node. That does not hit 100ms.

What makes the budget achievable:

- `ACTIVATION_FLOOR` prunes aggressively — in practice most hop-2 nodes fall below it, and effective fanout collapses fast.
- Edges are fetched per *frontier*, not per node: one `WHERE src IN (...)` query per hop, three queries total.
- `VISIT_CAP` (default 400) is a hard ceiling that guarantees termination regardless of graph shape. Hitting it is logged as a recall-quality warning, not swallowed.
- Node bodies are loaded once, at rank time, only for nodes that survived the floor.
- Embedding the cue is often the single largest cost. A local ONNX model runs ~5–15ms; a remote embedding API blows the budget entirely and must not be the default.

These numbers are guesses until M0 measures them. If p99 turns out to be 300ms, the honest fix is reducing `MAX_HOPS` to 2 and saying so, not quietly redefining the target.

### 10.3 When spreading activation helps, and when it hurts

The original claim: the useful memory is frequently *not* similar to the cue. Vector
search over "should I use this ORM for the migration" retrieves memories that talk about
ORMs and migrations. Spreading activation retrieves *this ORM* → *the deadlock incident*
→ *root cause was the connection pool* → *and pool config is shared with the job
runner*. Three hops out, textually dissimilar, and the actually load-bearing fact.

**That claim is half true, and the half that fails is the half used to sell the
project.**

*Supported (E3).* Spreading retrieves **3.2× more of a planted causal chain** than top-k
(0.68 vs 0.21) for 40% more tokens.

*Not supported (E3).* It does not reliably surface the **terminal** fact three hops out —
hit rate 2.5% against top-k's 0%. Attenuation reaches the node and the ranker then
discards it.

*Actively harmful (E6c).* For precedent lookup — "what happened last time on this
service with this symptom" — the traversal is the problem, not the solution. It buries a
correctly-seeded exact match under hub-accumulated activation. This is why the precision
channel exists and why it bypasses spreading entirely.

**The two use cases want opposite retrieval behaviour**, which is the single most
important thing to understand before changing this code. Association wants to travel
away from the cue; precedent lookup wants to stay exactly on it. Any change that "unifies
the retrieval path" will silently break one of them.

For the README: `reverie path A B` will find a chain because BFS is exhaustive, but
ordinary `recall` will not reliably surface its endpoint. Shipping that as the headline
demo would be the overclaim §18 warns against.

### 10.4 Brief rendering

Templated, deterministic, no LLM in the hot path:

```
## Relevant memory (7 items, 840 tokens)

### What worked before
- When migrating large tables on staging, run with `--sql` first and apply
  manually. [12/13 successes · observed · 3d ago]

### Watch out
- `alembic upgrade head` against staging times out when the job runner holds
  an open transaction. Check pg_stat_activity first.
  [failure mode · seen 3× · last 3d ago]
- ⚠ CONTESTED: two memories disagree on whether staging allows DDL during
  business hours. [reverie inspect mem_01J8..]

### Context
- staging_db is a read replica; migrations target primary. [asserted by human]
- Theme — Staging Database Quirks: 34 related memories, mostly about
  connection-pool contention and replica lag. [inferred summary]

### Stale
- ⚠ The deploy runbook memory was learned against env 9f1c..; current env is
  a41b... Re-verify before relying on it.
```

Structure and honesty markers do real work. The agent sees the track record, the provenance, and the conflicts, and can weight accordingly. Note that every claim carries either a success ratio or an explicit provenance tag — nothing appears as bare assertion.

---

## 11. Interfaces

### 11.1 MCP server (primary)

The distribution strategy. One install, works in Claude Code, Cursor, Codex, and everything else that speaks MCP.

| Tool | Signature |
|---|---|
| `reverie_recall` | `(cue: str, budget?: int) -> {brief, recall_id}` |
| `reverie_remember` | `(episode: Episode) -> {episode_id}` |
| `reverie_report_outcome` | `(recall_id: str, outcome: str, signal_tier: int, evidence?: str) -> ok` |
| `reverie_inspect` | `(node_id: str) -> {node, edges, utility_history}` |
| `reverie_forget` | `(node_id: str, reason: str) -> ok` |

The tool *descriptions* are part of the design, not documentation. `reverie_report_outcome`'s description must push hard toward tier-1 evidence, and `reverie_remember`'s must ask for `used_memories`, because in MCP deployments the tool description is the only integration guidance the agent ever sees.

### 11.2 SDK

```python
from reverie import Reverie

mem = Reverie(scope="agent:deploy-bot")     # defaults to ~/.reverie/memory.db

brief, rid = mem.recall("migrate orders table, add shipping_status")

with mem.episode(task="migrate orders", recall_id=rid) as ep:
    result = run_migration()
    ep.step("bash", "alembic upgrade head", exit_code=result.code)
    ep.outcome("failure" if result.code else "success", tier=1,
               evidence=f"exit_code={result.code}")
```

The context manager matters: it makes the correct integration the easiest one, and it captures duration, exceptions, and the `recall_id` linkage without the agent author thinking about any of it. If an exception escapes the block, it records an outcome automatically rather than losing the episode.

### 11.3 REST

`POST /v1/recall`, `POST /v1/episodes`, `POST /v1/outcomes`, `GET /v1/nodes/{id}`, `DELETE /v1/nodes/{id}`, `POST /v1/consolidate`. Bearer auth. For non-Python, non-MCP agents.

### 11.4 CLI

```
reverie init                      # create db, print integration snippet
reverie consolidate [--since 1d]  # run the dream cycle now
reverie recall "<cue>"            # what would the agent see?
reverie inspect <node_id>         # node, edges, utility history, sources
reverie why <node_id>             # trace back to originating episodes
reverie path <node_a> <node_b>    # shortest association path between two memories
reverie explain <node_id>         # plain-language account of what this is and why
reverie forget <node_id>
reverie assert "<fact>" --entities a,b   # human-authored memory, wins conflicts
reverie stats                     # communities, utility distribution, backlog
reverie doctor                    # health check: backlog, orphans, dupes, latency
reverie serve --mcp | --http
```

`why`, `path`, and `explain` are the trust surface and should be built early. `why` traces any memory to the raw episodes that produced it. `path` answers "how are these two things connected in this agent's head" — the demo from §10.3, on the command line. Being able to audit an association is what makes the system safe to leave running.

`doctor` is unglamorous and will do more for adoption than any of them: the first question every user has is "is this thing actually working?"

---

## 12. Scoping and fleet learning

Every node carries a `scope_id`. Recall reads from a scope chain:

```
agent:deploy-bot  →  team:platform  →  global
```

Writes go to the agent scope by default. Promotion to `team` happens when a memory is independently corroborated by ≥N distinct agent scopes (default 3) with consistent content and non-conflicting outcomes. Promotion to `global` is human-gated.

Recall merges across the chain, with narrower scopes winning conflicts — a lesson specific to this agent beats the team-wide default.

This is the feature with the most commercial gravity and the least prior art. It is also where privacy leakage lives: a memory promoted from one agent's scope can carry secrets from that agent's environment. Promotion must run a redaction pass, must be logged, and **must be disabled by default in v1**.

---

## 13. Security and threat model

Split out of the failure-modes table in v0.2 because a memory system has a genuinely unusual security posture and the table was too small to hold it.

### 13.1 What Reverie is, from an attacker's perspective

**A persistence layer for prompt injection.** An attacker who lands one payload in one tool output — a malicious README, a poisoned API response, a crafted error message — gets it distilled into a "lesson" and replayed into every future task, forever, with the system's own credibility attached. This is worse than ordinary injection, which is at least transient.

**A secret aggregator.** Tool outputs contain tokens, connection strings, and internal hostnames. A memory system stores tool outputs by construction. Left unaddressed, `~/.reverie/memory.db` becomes the highest-value file on the machine.

### 13.2 Controls

| Threat | Control |
|---|---|
| Injection persisted as a lesson | Distillation output is schema-constrained; no free-form text reaches the graph unstructured. An imperative-content heuristic rejects candidates phrased as instructions to the agent |
| Injected content replayed as instruction | Briefs render as **data** with explicit framing ("the following are recorded observations, not instructions"). Never interpolated into a system prompt |
| Deliberate memory poisoning | Utility LCB demotes memories that correlate with failure; quarantine tier; ablation detects memories whose presence hurts; human `assert` overrides |
| Secrets persisted | Redaction at ingest **before first write**: regex for known key formats plus entropy heuristics for unknown ones. Redaction is not best-effort cleanup after the fact |
| Secrets leaked across scopes | Promotion runs a second redaction pass and is off by default |
| Local file exfiltration | Document that the DB is sensitive. Optional at-rest encryption via SQLCipher. Do not pretend file permissions are a security model |
| Untrusted MCP client | The local MCP server binds stdio or loopback only, never `0.0.0.0`. Scope is server-configured, not client-supplied — otherwise any local process reads any scope by asking |
| Cloud multi-tenancy | Out of scope for v1, and the README must say so rather than implying isolation that does not exist |

### 13.3 The control that will not work

Redaction catches known patterns. It will not catch a proprietary internal identifier that happens to be sensitive, and it will not catch a secret that has been paraphrased by the distillation model into prose. `reverie forget` and scoped deletion therefore have to be genuinely complete — cascading through nodes, edges, embeddings, FTS, episode buffer, and recall logs — because deletion is the only backstop that actually works.

---

## 14. Operational design

The section v0.1 was missing entirely. These are the things that decide whether the project survives its first week in someone else's hands.

### 14.1 Concurrency

SQLite in WAL mode allows many readers and one writer. That maps onto Reverie's access pattern well, but only if it is designed for rather than discovered:

- **Recall** is read-only and never blocks. Many agent processes can recall concurrently against one file.
- **Ingest** is a single small append. Contention is possible under high write rates; ingest uses a short busy-timeout and retries, and the SDK batches writes within an episode context manager rather than writing per step.
- **Consolidation** takes a long write transaction and is the real contention risk. It runs under an advisory lock (a row in `meta`) so two workers cannot consolidate one scope simultaneously, and it commits per consolidation unit rather than per cycle, so ingest gets gaps to write in.
- **Attribution** is a short read-modify-write per node, guarded by the same busy-timeout retry.
- **Postgres** removes all of this and is the answer above roughly one write per second sustained. Say so in the docs rather than letting people discover SQLite's limits in production.

### 14.2 Consolidation is resumable

Long cycles get interrupted — laptop sleeps, container gets evicted. Episodes are marked consolidated only on commit of the unit that contained them; an interrupted cycle re-processes at most one unit, and the `client_key` dedup plus similarity-based reinforcement makes that re-processing harmless rather than duplicative.

### 14.3 Schema and embedding migration

Two independent versioned things, and conflating them is a classic mistake:

- **DB schema**: `meta.schema_version`, forward-only numbered migrations run on open. Refuse to open a database newer than the binary.
- **Embedding model**: `meta.embedding_model` and `meta.embedding_dim`. If the configured model differs from the stored one, Reverie **refuses to start** rather than silently mixing incompatible vector spaces, and prints the one command that fixes it (`reverie reembed`). Re-embedding is a background batch that rebuilds `node_vec` and swaps it atomically. The FTS path stays functional throughout, so the system degrades rather than stops.
- **Episode contract**: `schema_version` in the payload. Old buffered episodes must remain consolidatable after an upgrade.

### 14.4 Retention

Everything in this system grows. Each table needs a stated policy, or the answer to "how big does this get" is "unbounded," which is not an answer:

| Table | Policy |
|---|---|
| `episode_buffer` | Deleted after consolidation + grace period (default 7d) |
| `recalls` | Retained 90d, or until all linked outcomes are attributed plus 7d, whichever is longer. Then aggregated into per-node counters and dropped |
| `outcomes` | Retained with their recall |
| `nodes` (episodic) | Decay + prune per §8.8; hard cap with LRU eviction |
| `nodes` (semantic/procedural) | Not size-capped. If these grow unbounded, distillation is over-generating and that is the bug to fix |
| tombstones (`merged`) | Hard-deleted once no recall log references them |

### 14.5 Observing Reverie itself

Reverie is explicitly not an observability tool, which makes it easy to forget it needs to be observable. Minimum: recall latency histogram, recall-empty rate, buffer depth and drain rate, consolidation duration and LLM token spend per cycle, distillation rejection rate by reason, attribution lag, quarantine events. Exposed via `reverie stats`, `reverie doctor`, and an optional Prometheus endpoint.

The distillation rejection rate by reason is the most diagnostic number in the system. If 90% of candidates are dropped for missing citations, the distill prompt is broken, and nothing else in the pipeline will tell you.

### 14.6 Cold start

A fresh install returns nothing from every recall, which reads as "broken" to a new user long before it reads as "correct." Three mitigations, in order of honesty:

1. **Say so.** `reverie recall` on an empty graph returns an explicit "no memories yet — Reverie needs ~10 completed tasks before recall becomes useful," not an empty string.
2. **Import.** `reverie import` from existing sources: CI logs, shell history, git history, incident postmortems, an agent framework's own trace files. These produce low-confidence `inferred` memories that get corrected by real experience.
3. **Assert.** `reverie assert` lets a human seed the things they already know are true about the environment. This is the fastest path to a useful first recall and should be step 3 of the quickstart.

---

## 15. Failure modes and mitigations

| Risk | Mechanism | Mitigation |
|---|---|---|
| **Memory poisoning** | Wrong lesson recalled, task coincidentally succeeds, lesson reinforced | Contradiction detection, utility LCB not mean, ablation sampling, quarantine tier, human `assert` overrides |
| **Context bloat** | Recall injects marginal history; agent gets worse and slower | Hard token budget, type quotas, activation floor, community collapse, measured in eval as recall precision |
| **Staleness** | Environment moves, memory doesn't | `env_hash` fingerprinting, `stale` flag surfaced in briefs, time-based confidence decay |
| **Graph fragmentation** | Entity resolution failures split one concept into six nodes | Alias tables, periodic merge pass, `reverie doctor` surfaces suspected duplicates |
| **Community instability** | Re-clustering renumbers everything; theme nodes churn | Overlap-matched stable IDs with `prev_id` lineage; re-cluster on a slow cadence |
| **Consolidation cost** | LLM bill grows with agent activity | Salience gating (most units never distilled), local-model default, batch distillation, cost surfaced in `stats` |
| **Runaway growth** | DB grows without bound | Per-table retention (§14.4), episodic decay and prune, buffer sampling under backpressure |
| **Attribution bias** | Backpressure sampling skews base rates toward failure | Unsampled counter table; all base rates read from counters, never the buffer |
| **Lost attribution** | Node merged or pruned before its outcome arrives | Tombstones with `merged_into`; attribution follows pointers transitively |
| **Valence runaway** | One bad week permanently poisons a useful tool | Per-episode delta cap, clamping, asymmetric decay that still decays |
| **Privacy leakage** | Secrets in tool output persisted forever; scope promotion crosses boundaries | See §13 |
| **Prompt injection via memory** | Malicious tool output becomes a "lesson" instructing the agent | See §13 |
| **The whole thesis is wrong** | Attribution signal is too noisy to rank memories usefully | This is what M0 and M3 exist to find out. If the learning curve is flat with attribution on and off, the honest move is to publish that and cut §9 |

That last row is not filler. A memory system whose attribution signal is pure noise is strictly worse than one without it, because it will confidently demote good memories. The eval harness has to be able to detect that outcome, and the project has to be willing to report it.

---

## 16. Evaluation

The project's credibility rests here. Architecture does not earn stars; a reproducible number does.

### 16.1 What existing benchmarks do not measure

LOCOMO and LongMemEval measure conversational recall — can the system answer a question about something said earlier. Useful as a sanity check, and worth running so the comparison table has a row people recognise. But orthogonal to the actual claim.

Reverie's claim is that **an agent gets better at doing a job by having done it before**, which nothing in the standard suite measures. If we adopt someone else's benchmark as the headline, we are optimising for someone else's objective and competing on their terms — against systems that have been tuned for it for two years.

### 16.2 The harness — built, in `reverie/bench/`

`reverie-bench` exists. Four families, four arms, an objective function, and a scalar
`capture` metric. Runs with no API key; `LLMAgent` is the adapter for a real model.

**The baseline is not "memory off."** Without memory an agent cannot learn across
episodes at all, so that arm is pinned at chance and the on arm wins by construction.
The honest baseline is what practitioners actually do — paste recent history into the
context window:

| Arm | Evidence the agent sees |
|---|---|
| `none` | nothing — the floor |
| `replay` | last N episodes verbatim, optionally capped by token budget |
| `reverie` | a budgeted recall brief |
| `reverie_noattr` | same, attribution disabled — isolates **attribution lift** |

One agent policy across all arms, byte-identical seeded task sequences. Only the
evidence differs.

**The metric is `capture`, not success rate:**

```
capture = (accuracy − floor) / (ceiling − floor)
```

0.0 = memory added nothing over having none; 1.0 = matched naive history-stuffing;
>1.0 = beat what practitioners do. Comparable across families, which raw accuracy is
not. The efficiency claim is a *constraint*, not a second thing to average in:

```
maximise capture  s.t.  token_ratio ≤ 0.5,  quarantine_rate ≤ 0.10
```

folded into a scalar as one-sided penalties. Otherwise an optimiser "discovers" that the
best configuration is an enormous token budget.

#### Three methodology traps, each of which produced a wrong conclusion

**1. Benchmark regime decides the answer before you measure anything.** The first three
families have ≤8 distinct contexts. A 31-episode recency window then holds several
examples of *every* context, so recency is guaranteed to have the precedent and
selective retrieval cannot beat it — replay dominated at every token budget. Those
families structurally cannot test the design's claim. `IncidentFamily` (12 services × 6
symptoms = 72 contexts) is the one that can, and it is the only one where the crossover
appears. **Check the context-space size before believing any retrieval result.**

**2. Horizon must exceed the window.** At 40 tasks with a 40-episode window, replay never
forgets and never gets expensive. The regime the design targets — history far larger
than any context window — only starts around a few hundred tasks. Measured at 600 tasks,
unbounded replay costs 8,908 tokens/prompt and is no longer a real option.

**3. Standard errors must be cluster-robust.** `sqrt(p(1-p)/n_tasks)` assumes independent
Bernoulli trials. Outcomes *within a run* are strongly correlated — a run either
discovers the rule or it does not, and everything after follows. Per-run accuracies on
one family span 0.10 to 0.65. The effective sample size is the number of **runs**, not
tasks, and the binomial form understates error by ~1.8×. This produced a phantom
"generalisation gap" (+0.70 vs +0.27 at 4 seeds) that vanished at 16 seeds (+0.55 vs
+0.54). Compare against `Score.capture_se`, and **use ≥16 seeds before believing any
comparison** — a 4-seed band is roughly ±0.35 capture, most of the interesting range.

**Primary metrics:**

| Metric | Definition |
|---|---|
| Learning curve | Success rate vs run index — the shape is the story |
| Capture | Normalised lift between the `none` floor and the `replay` ceiling |
| Token ratio | Brief tokens ÷ replay tokens; the lift must survive the cost |
| Exact-precedent rate | Fraction of briefs containing a precedent for the current context |
| **Attribution lift** | Capture with attribution on vs. memory-on-attribution-off. **Measured zero three times; see §9** |
| **Review lift** | Capture with human review on vs. off, swept over frequency and reviewer accuracy. Currently the largest effect in the system (+0.227) |
| **Repetition density** | Tasks per distinct context. Report it with every retrieval number — it predicts the result better than any tuning parameter (E10) |
| **Reviewer-error robustness** | Capture under random / systematic / adversarial / over-general reviewer error. Over-general is the one that goes below baseline |
| Consolidation cost | LLM tokens per episode consolidated |

**Attribution lift was the row that was supposed to justify the project, and it does
not.** It has measured zero three times: on episodic memory (E7), under environment
drift (E7), and finally under the exact condition E7 predicted would rescue it — few
nodes, recalled often, outcome not in the text, some of them wrong (E9). That hypothesis
is now refuted rather than untested.

The binding constraint is arithmetic. Total credit mass equals the number of outcomes,
so a graph with more memories than outcomes cannot give any single memory enough
evidence to rank on:

```
credit mass == outcomes == 600 over a 600-task run
nodes sharing it == ~628
→ ~1.0 unit of evidence per node, before splitting across a 9-item brief
```

Report it anyway, as a negative. "We built it, measured it three ways, and it did not
work" is more useful to a reader than silence — and it is the honest counterweight to a
space full of systems that assert the same idea without testing it.

### 16.3 Build it first

The eval harness should be built before the consolidation logic, not after. Nearly every parameter in this document — decay constants, activation floor, fanout, salience weights, budget defaults, the collapse threshold — is a guess. The harness is what turns guesses into measurements, and without it the sophisticated parts of this design are unfalsifiable decoration.

---

## 17. Milestones

| Milestone | Scope | Proves |
|---|---|---|
| Milestone | Scope | Status |
|---|---|---|
| **M0 — Harness** | `reverie-bench`: 4 families, 4 arms, objective function, human oracle | ✅ built (`reverie/bench/`) |
| **M1 — Walking skeleton** | SQLite store, ingest, no-LLM consolidation, hybrid seed, recall, SDK, CLI | ✅ built |
| **M2 — Distillation** | Distiller seam, valence, dedup/reinforce | ⚠️ seam + `NullDistiller` only. **No LLM distiller exists.** Human review (§8.10) now fills this role and measures better than an LLM distiller plausibly would |
| **M3 — Attribution** | Recall log, outcome API, Beta utility, LCB ranking, ablation, decay/prune | ❌ **built and refuted.** Zero lift in three independent tests; constraint is arithmetic (§9). Candidate for removal |
| **M4 — Trust** | Reconcile, quarantine, provenance, `why`/`path`/`explain`/`doctor` | ❌ **quarantine does not fire.** The E4 scenario that once removed a poison in 23 recalls now never removes it; held-out detection is 5/20 even tuned. **Goal G8 is not satisfied** |
| **M5 — Association** | Communities, theme nodes, collapse, stable IDs | ⚠️ built; association channel is dominated by generic failure-mode hubs |
| **M6 — Precision retrieval** | Conjunctive matching, two-channel recall, containment admission | ✅ built. **This is what actually works** — 4% → 78% exact-precedent retrieval |
| **M7 — Human review** | Session-boundary review, procedural lessons, out-of-band write | ✅ built. **+0.227**, the strongest lever measured |
| **M8 — Over-generalisation guard** | Validate a lesson's scope against episodic evidence at consolidation | ❌ **highest-value unbuilt item.** The one reviewer error that drives performance *below* the no-human baseline (E10) |
| **M9 — Distribution** | MCP server, `reverie init`, README with benchmark table | ❌ not started |
| **M10 — Scale** | Postgres adapter, worker queue, REST auth, scope chain | ❌ not started |

**Where the project actually is.** The loop runs end to end. On a 72-context on-call
landscape at 600 tasks it beats budget-matched history-stuffing (0.496 vs 0.409) at 29%
fewer tokens, and human review lifts that to 0.685. Those are real, reproducible, and
measured against an honest baseline.

Two things a reader should not be allowed to miss:

1. **The original thesis is dead.** Outcome attribution — the thing this document was
   built around for three revisions — does not work, and M3 should probably be deleted
   rather than defended. What replaced it is narrower and better evidenced: precise
   precedent retrieval plus a human-teachable procedural layer.
2. **The win is conditional on repetition.** Below ~2 recurrences per context, plain
   recency beats us (E10). That belongs in the README next to the headline number, not
   discovered by a user three weeks in.

M8 is the next thing to build. It is small, runs out of band, needs no posterior, and
closes the only measured failure mode where the system is worse than doing nothing.

## 18. Launch strategy

Stated explicitly because "get traction on GitHub" is a project goal, and goals that stay implicit do not get designed for.

**License: Apache-2.0.** Permissive enough for company adoption without a legal review, and the patent grant matters to exactly the enterprise users who might later pay for the cloud version. MIT is fine too. AGPL would protect a future commercial offering but suppresses precisely the adoption this project needs first.

**The README leads with the number, not the architecture.** A benchmark table and the learning-curve chart above the fold. The architecture diagram goes in `docs/`. Every competitor's README opens with an architecture diagram; none of them open with a reproducible result.

**Ship the `path` demo as an animation.** `reverie path "postgres-orm" "job-runner"` printing the four-hop chain from §10.3 is the thing that makes the idea click in five seconds. That animation is worth more than three sections of prose.

**Be conspicuously honest about limitations.** §9.4 (correlation is not causation), §2 (where competitors are better), and the possibility that the thesis is wrong (§15, last row) all go in the README, not just in this document. In a space full of overclaiming, the credible project stands out — and the technically serious people who give a project its first real traction are specifically looking for whether you know your own weaknesses.

**Sequence:** working benchmark → MCP one-liner install → one genuinely good demo → *then* announce. A repo that gets attention before `reverie init` works cleanly gets one look and no second one.

**Do not name a competitor in a marketing claim.** The comparison table in §2 is for the docs, phrased as "different objective," not "better." The maintainers of these projects are the audience most likely to amplify a good new idea, and least likely to amplify one that opened by dunking on them.

---

## 19. Questions, answered and open

### 19.1 Answered by the experiments

Four of these are no longer open. Measurements come from `experiments/`; the full
write-up including what did not work is in `experiments/README.md`.

**Q5 (was the highest-priority unknown): how much attribution signal actually exists?**

Measured. Recalls needed to reach Spearman ρ = 0.5 against ground truth, as a
function of how often a recalled memory actually influenced the task:

| Recall precision | Recalls to ρ=0.5 |
|---|---|
| 20% | not reached within 10,000 |
| 50% | 3,200 |
| 100% | 800 |

Two consequences, one of them a correction to this document:

- **Sample complexity is governed by recall precision, not by graph size or agent
  count.** Recall precision is therefore an *attribution* concern, not only a
  token-budget concern — a link §10 does not currently draw.
- **Co-occurrence is not the bottleneck.** §15 lists locked-together recall sets as
  the identifiability risk. Sweeping co-occurrence from 0.0 to 1.0 moved the
  crossover not at all: variation in activation *shares* carries enough signal by
  itself. That row of §15 is wrong and should be rewritten.

The honest reading: attribution works, and it needs thousands of attributed recalls
at realistic precision. That is fine for a busy agent over weeks and hopeless for an
afternoon trial. §14.6 and the README must set that expectation explicitly.

**Q on credit assignment (§9.2, §9.4): does activation weighting beat uniform, and
is `used_memories` worth the integration burden?**

Activation 0.644 · citation 0.614 · rank 0.600 · uniform 0.518 (ρ at 6,400 recalls).

- Activation weighting is worth its complexity: +20% over the null. §9.2 stands.
- **`used_memories` is not the high-value field §9.4 claims.** Citation-based credit
  scored *below* the activation heuristic, because zeroing out credit for irrelevant
  memories means they never accumulate evidence and can never be ranked. Diffuse,
  slightly-wrong credit beats sharp, narrow credit. §9.4's recommendation to build
  the SDK around this field is withdrawn; keep the field for offline analysis, demote
  it from the primary integration ask.

**Q on the retrieval mechanism (§10.3): does spreading activation beat top-k?**

Half-supported, and the failing half is the half used to sell the project.
Spreading activation retrieves **3.2× more of the causal chain** than hybrid top-k
(0.68 vs 0.21) for 40% more tokens — the graph genuinely surfaces material vector
search misses. But the specific claim that it surfaces the *terminal* load-bearing
fact three hops out is not supported: hit rate 2.5% against a 0% baseline. With
`hop_decay = [1.0, 0.6, 0.35]` and `max_nodes = 12`, a 3-hop node arrives at ~20% of
seed activation and loses the budget competition. **The mechanism reaches the fact
and the ranker then discards it.**

Until a fix is measured (flatter decay, terminal-node bonus, or a reserved budget
slot beyond hop 1), the `reverie path` demo must not be the headline in §18. `path`
finds the chain because BFS is exhaustive; ordinary `recall` does not surface its
endpoint, and shipping a demo that misrepresents default behaviour is precisely the
overclaim §18 warns against.

**Q on self-correction (G8, §13.2): does bad memory remove itself?**

Yes, under ground truth. Recalls to quarantine a memory that causes failure 85% of
the time: tier 1 → 23, tier 2 → 29, tier 3 → 60, tier 4 → 162. Control arm with
attribution disabled: never demoted, served in 50/50 subsequent recalls.

The 7× spread between tier 1 and tier 4 is the empirical justification for §9.1.
**Ablation contributed nothing** (23.4 recalls at 0% vs 24.7 at 30%), which together
with the co-occurrence result means the ablation machinery of §9.4 is not currently
load-bearing. Keep it — it is cheap and may matter under heavier confounding — but
drop the claim that it is what makes attribution work.

### 19.2 Corrections to this document forced by implementation

1. **§8.3 and §8.5 contradict each other.** Salience gating archived redundant units
   before they reached `link`, so repeated experience registered as nothing — while
   §8.5 says reinforcement is how repetition registers. Resolved with a cheap
   reinforce-only path for low-salience units: no model call, no new nodes, just
   strength on what they already match.
2. **Appendix A's `quarantine_threshold: 0.25` was inert.** `utility_lcb(1,1,0.25)`
   is exactly 0.25, so the uninformative prior sat on the threshold and everything
   with any blame was quarantined. Replaced with a peer-relative MAD rule; a
   base-rate-relative rule was tried first and is self-defeating, because a harmful
   memory present in most recalls drags down the very base rate meant to catch it.
3. **Quarantine must gate on evidence mass, not participation count.** Credit is
   split across a recall, so eight participations is under one unit of evidence.
4. **Quarantine must never touch `entity`/`theme` nodes or `asserted` memories.**
   Quarantining a hub silently disconnects everything routed through it — the
   memories stay `active` but become unreachable, which is worse than demotion
   because nothing reports it. And auto-demoting a human assertion is §8.6's
   "humans win" with the sign flipped.

### 19.3 Still open

1. **Consolidation trigger.** Wall-clock schedule, session end, or buffer depth? Session end is the truest analogue to sleep but agents do not reliably signal it. Probably all three, configurable, defaulting to buffer depth with an idle timer.
2. **Embedding model default.** A local model that must be downloaded is friction; an API-based one breaks the no-key promise. Leaning toward a small ONNX model fetched on `init`, with the FTS path fully functional if it is absent. The shipped hashing embedder is a placeholder that captures lexical, not semantic, similarity.
3. **Does valence earn its keep?** The most novel piece and the least proven. Still untested — it needs an ablation of its own before anything is allowed to depend on it. Cut it without ceremony if it does not show.
4. **Does the community layer earn its keep?** Implemented and tested for stability, but its *value* is unmeasured. Strong demo story, weak evidence story — exactly the combination that survives longer than it should.
5. **Can the 3-hop retrieval gap be closed?** See above. This is now the
   highest-priority unknown, having displaced Q5.
6. **Cross-agent identity.** Is `scope_id` sufficient, or is there a distinct concept of "the same agent across deployments"? Affects fleet learning.
7. **How much should an agent steer its own recall?** A `focus="failure_modes"` parameter gives useful control but also lets a confused agent steer away from the memory it most needs.
8. **Multi-modal episodes.** Screenshots from browser agents are high-signal and expensive. Out of scope for v1, but the episode schema should not preclude them.

---

## Appendix A — Default parameters

All guesses pending M0 measurement, all configurable.

```yaml
recall:
  max_hops: 3
  fanout: 12
  visit_cap: 400
  activation_floor: 0.05
  hop_decay: [1.0, 0.6, 0.35]
  intra_community_bonus: 1.15
  budget_tokens: 1200
  max_nodes: null            # derive from budget; 12 made budget_tokens inert (E5)
  tokens_per_node_estimate: 25
  max_nodes_floor: 6
  max_nodes_ceiling: 40
  adaptive_quotas: true      # renormalise quotas over classes actually present (E5)
  max_contested: 2
  type_quotas: {procedural: 0.5, semantic: 0.3, episodic: 0.2}
  valence_gain: 0.4
  collapse_min_members: 4
  provenance_factor: {observed: 1.0, asserted: 1.0, inferred: 0.7, ambiguous: 0.4}
  recency_halflife_days: 30.0
  context_mismatch_penalty: 0.8

  # Conjunctive entity matching (E6b) — scores candidates by how much of the
  # cue they match, before spreading dilutes it.
  conjunctive: true
  conjunctive_weight: 0.10       # additive on RRF, whose terms are ~0.016
  conjunctive_exponent: 2.0      # >1 rewards matching many cue entities
  conjunctive_specificity: 1.0   # hub penalty, over ENTITY degree only
  conjunctive_k: 24

  # Precision channel (E6c) — exact precedents bypass spreading entirely.
  direct_match: true
  direct_reserve: 0.5            # share of budget the channel may claim
  direct_min_coverage: 0.95      # ADMISSION is on cue coverage, never the composite

consolidation:
  salience_threshold: 0.35
  min_corroboration_for_lesson: 2
  dedup_similarity: 0.88
  max_valence_delta_per_episode: 0.25
  abstract_every_n_cycles: 20
  abstract_on_growth_pct: 15
  community_match_jaccard: 0.5

attribution:
  ablation_rate: 0.05
  ablation_mode: lowest_rank      # or stratified
  tier_weights: {1: 1.0, 2: 0.8, 3: 0.4, 4: 0.15}
  utility_lcb_quantile: 0.25
  min_task_instances_for_tier3: 20

decay:
  tau_days: {episodic: 14, semantic: 180, procedural: 60}
  utility_lambda: 0.98          # per consolidation cycle
  quarantine_threshold: 0.25    # utility LCB
  min_attributions_for_quarantine: 5

ingest:
  max_output_summary_bytes: 2048
  max_steps_per_episode: 200
  buffer_high_water: 50000

retention:
  episode_buffer_grace_days: 7
  recall_log_days: 90
  outcome_attribution_grace_days: 7
```

---

## Appendix B — Naming and namespaces

`Engram` was the working title through v0.1 and is not viable: at least seven active GitHub projects use the exact name in the agent-memory space, several with near-identical positioning, and one (`engram.to`) commercially. The v0.1 default path `~/.engram/engram.db` is already in use by one of them.

**Reverie** is clear as of 2026-08-10:

| Namespace | Status |
|---|---|
| PyPI `reverie` | Available |
| npm `reverie` | Taken — a single empty 1.0.0 published in 2017. Use `@reverie/sdk` |
| GitHub `reverie` (org) | Verify before announcing |
| `reverie.dev` / `.sh` | Verify before announcing |

Claim PyPI, the GitHub org, and the domain in one sitting, before the first public commit. Names in this space are being taken weekly.
