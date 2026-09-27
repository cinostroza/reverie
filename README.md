# Reverie

**Precise precedent retrieval for AI agents, with a human in the loop.**

> ⚠️ **Pre-alpha, and the thesis changed.** This project started out claiming that
> *outcome attribution* — memories earning a track record and demoting themselves when
> they hurt — was the differentiator. **We built it, measured it three ways, and it does
> not work.** See [E7/E9](https://github.com/cinostroza/reverie/blob/main/experiments/README.md). What does work is narrower and better
> evidenced. Numbers below are from synthetic benchmarks against an honest baseline, not
> from production agents. Read [Known limitations](#known-limitations) first.

---

Most agent memory optimises **recall fidelity**: can the system retrieve the thing you
told it. Reverie optimises **precedent precision**: given what this agent is doing right
now, surface the specific past case that applies — and let a human correct the record
when it is wrong.

Two claims, both measured on `reverie-bench` over a 72-context on-call landscape at 600
tasks, against budget-matched history-stuffing (what people actually do today):

| | Reverie | Naive replay | Reverie + human review |
|---|---|---|---|
| success rate | 0.459 ±0.020 | 0.402 ±0.022 | **0.640 ±0.019** |
| tokens per prompt | 775 | 1,146 | **708** |

Paired over 16 identical seeds, so seed variance cancels:
**Reverie − replay = +0.056 ±0.032 (significant)**;
**human review − Reverie = +0.181 ±0.026 (significant)**.

The learning curves are the real story: Reverie **rises** with accumulated history
(0.300 → 0.439 from 200 to 600 tasks in the committed E6 run) while a recency window is
**flat** (0.402 → 0.381), because a window cannot reach past itself however much history
piles up behind it.

**Everything except recall happens out of band.** Consolidation, linking, and human
review run between sessions. Nothing Reverie does makes the current task slower; it
exists to make the next one better.

## Install

```bash
pip install reverie-memory            # core: zero dependencies, stdlib sqlite3 only
```

The distribution is `reverie-memory`; the import and the CLI are both `reverie`.
To reproduce the experiments, work from a clone:

```bash
pip install -e ".[experiments,dev]"   # + numpy/scipy/pandas/matplotlib, pytest
```

## Use

```python
from reverie import Reverie

mem = Reverie(scope="agent:deploy-bot")        # defaults to ~/.reverie/memory.db

result = mem.recall("migrate orders table, add shipping_status")
print(result.brief)                            # budgeted, templated, no LLM call

with mem.episode("migrate orders", recall_id=result.recall_id) as ep:
    code = run_migration()
    ep.entity("alembic", "staging_db")
    ep.step("bash", "alembic upgrade head", exit_code=code)
    ep.outcome("failure" if code else "success", tier=1, evidence=f"exit={code}")

mem.dream()    # attribute outcomes, then consolidate
```

The context manager captures duration, exceptions, and the recall linkage without you
thinking about it. An exception escaping the block is recorded as a tier-1 failure
rather than losing the episode.

```bash
reverie init                     # create db, print an integration snippet
reverie recall "<cue>"           # what would the agent see?
reverie consolidate --abstract   # run the dream cycle now
reverie why <node_id>            # trace a memory back to its source episodes
reverie path <a> <b>             # shortest association path between two memories
reverie doctor                   # is this thing actually working?
reverie forget <node_id>         # cascades through edges, embeddings, FTS
```

## What makes it different

| | Reverie | Mem0 / Zep / Cognee / Letta |
|---|---|---|
| Retrieval | **conjunctive** — scores candidates by how much of the cue they match, before spreading dilutes it | similarity + recency |
| Precedent lookup | dedicated channel that **bypasses** graph traversal | single ranked pipeline |
| Correcting memory | a human teaches a procedural lesson; it is retrievable next session | edit or delete a record |
| Requires | situations that **recur** | nothing |

Reverie is the right choice when an agent does **repeated work over a landscape larger
than a context window** — on-call response, deploys, migrations, integrations. It is the
wrong choice for a chatbot remembering your dog's name, and the wrong choice for
genuinely one-off work: below ~2 recurrences per distinct situation, plain recency beats
us (E10). Zep's temporal model is more rigorous than ours; Mem0's integration surface is
far broader.

## Design

- **Write cheap, think expensive, think later.** Ingest is an append: no embedding,
  no LLM call, no graph write. p99 target 5 ms local.
- **Consolidation is the sleep cycle.** `segment → score → distill → link →
  reconcile → abstract → prune`, offline. Community detection folds scattered
  memories into themes.
- **Recall is associative, not top-k.** Hybrid seeding (BM25 + vector + exact
  entity), then spreading activation across the graph, then a hard token budget.
- **Provenance over confidence.** Every claim is tagged `observed`, `inferred`,
  `asserted`, or `ambiguous`. Briefs render as *data*, explicitly framed as "not
  instructions" — a memory store is a persistence layer for prompt injection, and
  that has to be designed for on day one.
- **Runs with no API key.** The no-LLM consolidation path builds the entity graph and
  episodic spine from structured episode fields alone, and is tested in CI.

Full design: [`reverie_hld.md`](https://github.com/cinostroza/reverie/blob/main/reverie_hld.md).

## Experiments

Ten experiments in [`experiments/`](https://github.com/cinostroza/reverie/tree/main/experiments), six as reproducible notebooks. They
changed the design repeatedly, including three reversals of decisions in the design doc.

| | Question | Headline result |
|---|---|---|
| **E1** | How much evidence does attribution need? | ~3,200 recalls to ρ=0.5 |
| **E2** | Which credit rule works? | Activation beats uniform by 20%; **citation is worse** |
| **E3** | Does the graph beat top-k? | 3.9× more causal chain — but rarely the terminal fact |
| **E4** | Does bad memory remove itself? | **No, not currently.** Quarantine does not fire |
| **E5** | Does an agent get better at its job? | Found 2 structural bugs. Wrong benchmark regime throughout |
| **E6** | Does retrieval beat recency? | Yes — after a two-channel recall rewrite. 4% → 78% precedent |
| **E7** | Does attribution lift performance? | **No.** Posterior never leaves the prior |
| **E8** | Does human review help? | Not as first built — a retrieval bug, not a bad idea |
| **E9** | …after fixing the plumbing? | **+0.227.** Attribution still zero |
| **E10** | Where does it break? | Repetition density is the governing variable |

## Known limitations

Stated up front, because they are the first things a careful reader will look for.

1. **Outcome attribution does not work.** Zero lift in three independent tests, and the
   constraint is arithmetic rather than a tuning failure: total credit mass equals the
   number of outcomes, so a graph with more memories than outcomes cannot give any one
   memory enough evidence to rank on. The code ships, disabled-by-default is under
   consideration, and it is documented as a negative result rather than removed.
2. **The win is conditional on repetition.** At 8.3 tasks per distinct context we beat
   recency; at 2.0 we lose; at 1.0 we are at chance. Measure your repetition density
   before adopting this.
3. **An over-general human lesson is worse than no human at all.** A reviewer who drops
   a feature that mattered drives performance *below* the no-review baseline
   (0.411 vs 0.456). Guarding against this is the top unbuilt item. Reviewers who are
   simply *wrong* — even adversarially, even 50% of the time — are handled fine.
4. **No LLM distiller.** `NullDistiller` produces no procedural or semantic memories, so
   everything measured rests on episodic recall plus human lessons.
5. **Recall does not reliably surface 3-hop facts** (E3). The associative demo in the
   design doc overclaims relative to what default `recall` does today.
6. **Quarantine does not work.** The E4 scenario that once removed a poisoned memory in
   23 recalls now never fires, and held-out seeds put detection at 5/20 even when tuned.
   Design goal G8 — "memory can be shown to be wrong and removed automatically" — is not
   currently satisfied.
7. **Single-tenant.** No org/RBAC layer. The SQLite file is sensitive — it holds
   redacted tool output, and redaction is best-effort.

## Development

```bash
pytest -q                              # 76 tests
python experiments/_build_notebooks.py # regenerate E1-E4
python experiments/_build_e5.py        # E5
python experiments/_build_e6.py        # E6
```

The core has **no runtime dependencies**. numpy/scipy are experiment-side only, which
is why the Beta quantile in `reverie/_math.py` is implemented from scratch and
verified against `scipy.stats.beta` to 1e-9 in the test suite.

## License

Apache-2.0.
