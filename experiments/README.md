# Reverie experiments

Thirteen experiments, all reproducible from a clean checkout. E1–E6 are notebooks;
E7–E10, E12 and E13 are scripted sweeps. E11 is the only one that calls a real model.

```bash
pip install -e ".[experiments,dev]"

# E1-E6 -- notebooks. Regenerate, then execute; the builders wipe outputs.
python experiments/_build_notebooks.py     # E1-E4 (regenerates ALL FOUR)
python experiments/_build_e5.py            # E5
python experiments/_build_e6.py            # E6
jupyter nbconvert --to notebook --execute --inplace experiments/E*.ipynb
python experiments/_notebook_digest.py     # dump the printed tables to diff

# E7-E12 -- sweeps. ~25 min on 10 cores; results cached as JSON.
python experiments/_run_sweeps.py                    # all, or name one
python experiments/_analyse.py                       # tables from cached results
python experiments/_attribution_diagnostic.py        # why attribution is zero
python experiments/_figure1.py                       # paper/figures/fig1.pdf
```

Notebooks are build artifacts generated from the `_build_*.py` scripts, so shared setup
(styling, seeds, palette) lives in one place instead of drifting across files.
**Edit the builder, not the `.ipynb`.**

Sweeps write raw per-run rows to `results/*.json`, so re-analysis never re-runs compute.
`_analyse.py` is the only place statistics live; `results/REPORT.txt` is its committed
output and is what the numbers below are copied from.

## Re-run of 2026-09-04: what changed

Everything was re-measured against the current engine. Three corrections came out of it,
and all three were invisible before the re-run:

1. **The benchmark was not reproducible at all.** `IncidentFamily` derived its hidden
   rules from the built-in `hash()`, which is salted per process, so the same seed drew a
   different landscape in every interpreter — measured 0.368 / 0.400 / 0.408 for one
   nominally fixed configuration. Fixed by `bench.core.stable_hash`. **Every number
   recorded before this date came from an unknown landscape draw.**
2. **The canonical numbers were second-half success, and never said so.** Scored over the
   whole run the retrieval effect is *negative* (−0.044); scored over the second half it
   is positive (+0.026 on one landscape, +0.041 across eight). Both are now reported
   everywhere, always labelled.
3. **The landscape was an uncontrolled constant.** Varying only the task seed re-measures
   one arbitrary rule set. Crossing 8 landscapes × 6 seeds, the retrieval effect is
   +0.041 ±0.016 but ranges from −0.007 to +0.061 per landscape — a single-landscape
   study could have reported either sign.

Also fixed: the 1200-context row was labelled 1.0 tasks/context, which is arithmetically
impossible at a fixed 600-task budget. It is 0.5.

**What survived unchanged:** the attribution null (three ablations, all n.s.), the
review-frequency and review-quality monotonicity, the error-mode taxonomy, the density
crossover, E12's four verdicts, E4's dead quarantine, and E3's 3.9× chain recall.

| # | Question | Method | Verdict |
|---|---|---|---|
| **E1** | How many attributed recalls before utility ranking is trustworthy? | Simulation, ground truth known | Thousands. Bound by recall precision, not volume |
| **E2** | Which credit rule works? Is `used_memories` worth asking for? | Simulation, identical traces | Activation wins; citation is *worse*. Demote `used_memories` |
| **E3** | Does spreading activation beat top-k retrieval? | Real engine, planted chains | Half-true: 3.9× more chain, but rarely the terminal fact |
| **E4** | Does bad memory remove itself? | Real engine, poisoned memory | **No longer.** Once 23 recalls; now never fires |
| **E5** | Does an agent actually get better at its job? | `reverie-bench`, 4 arms | Found 2 structural bugs; wrong benchmark regime throughout |
| **E6** | Does retrieval beat recency at scale? | On-call landscape, horizon sweep | Yes at 600 tasks (0.453 vs 0.411) — after a two-channel recall rewrite |
| **E7** | Does outcome attribution lift task performance? | Attribution ablation + drift | **No.** Posterior never leaves the prior on episodic memory |
| **E8** | Does human review help? | Simulated reviewer | Not as first built — a retrieval bug, not a bad idea |
| **E9** | …and after fixing the plumbing? | Admission-rule fix, re-run | **+0.217 at full review.** Strongest lever in the project. Attribution still zero |
| **E10** | Where does it break, and how does it handle a wrong human? | Scale + error-mode stress | Robust to wrong; **fragile to over-general** |
| **E11** | Does the cross-check property survive a real model? | `claude-opus-5`, 40 headless calls | Yes. ≥2 contradicting precedents override a wrong lesson 10/10 |
| **E12** | Can a scope guard repair over-generalisation? | 4 locked predictions | **No.** Fires on ~tens of nodes/run, moves performance 0 |
| **E13** | Is any of this reproducible? | 8 landscapes × 6 seeds | Only after fixing `hash()`; retrieval effect +0.041 ±0.016 |

## Why two methodologies

E1 and E2 need **ground truth about causal effect**, which is unobservable in any
real deployment — you cannot measure how well an estimator recovers a quantity you
cannot independently see. So they use a generative world model
(`reverie.sim`) where each memory's true effect is a parameter. The estimator under
test is the shipped one; `tests/test_core.py::TestSim::test_matches_engine_lcb`
asserts the simulation's LCB agrees with `reverie._math.utility_lcb` to 1e-9, so a
sweep result is a statement about the real code and not a lookalike.

E3 onwards test **mechanism**, not estimation, so they run the actual `RecallEngine`,
`Consolidator` and `AttributionEngine` against a real SQLite store.

## How to read this directory

**Three results overturned earlier conclusions in this same file.** Read it in order;
each experiment is annotated with what it refuted. In particular:

* E5's conclusion ("replay dominates at every token budget") was an artifact of a
  benchmark whose context space was smaller than a recency window. E6 corrects it.
* E6's conclusion ("retrieval beats recency at scale") is corrected by E10 — the
  governing variable is **repetition density**, not scale.
* E7/E9 refute the original project thesis. Outcome attribution does not work, and the
  constraint is arithmetic rather than a tuning failure.
* E13 qualifies **every** number that predates it: the benchmark was not reproducible,
  and the headline retrieval effect depends on two protocol choices nobody had stated.

## Reproducibility

Every number here is from a committed notebook run or a scripted sweep against the
shipped code. Two habits are load-bearing and were both learned the hard way:

1. **Cluster-robust error bars.** `sqrt(p(1-p)/n_tasks)` assumes independent trials;
   outcomes within a run are strongly correlated and it understates error ~1.8×. That
   produced one phantom finding (a "generalisation gap" that vanished at 16 seeds). Use
   ≥16 seeds before believing a comparison; a 4-seed band is ~±0.35 capture.
2. **Re-run notebooks after touching the engine.** Committed outputs that predate a code
   change are worse than no outputs. Check with:
   `python -c "import os,glob;print(max(os.path.getmtime(f) for f in glob.glob('reverie/**/*.py',recursive=True)))"`
   against the notebook mtimes.
3. **Vary the landscape, not just the seed.** `seed` controls the task *sequence*; the
   hidden rules come from `landscape_seed`. Sweeping seeds alone re-measures one rule set
   with tight-looking error bars that exclude the dominant source of variation. Use the
   `landscape` sweep for any headline comparison.
4. **Never derive an environment from `hash()`.** It is salted per process. This silently
   made every number here irreproducible until 2026-09-04. Use
   `reverie.bench.core.stable_hash`.
5. **Say which half you scored.** Second-half and whole-run success disagree by enough to
   flip the retrieval result's sign. `_run_sweeps.py` records both on every run.

---

## The canonical numbers

Everything else in this file is diagnostic. These are the two claims the project
actually rests on — 600 tasks, 72-context on-call landscape, 16 seeds, **second-half
success** (see the metric note below):

| | Reverie | Naive replay | Reverie + human review |
|---|---|---|---|
| success (2nd half) | 0.450 ±0.025 | 0.424 ±0.017 | **0.635 ±0.020** |
| success (whole run) | 0.363 | 0.408 | 0.509 |
| tokens per prompt | 794 | 1,146 | **704** |

Paired over 16 identical seeds, so seed variance cancels:
**Reverie − replay = +0.026 ±0.027 (not significant on one landscape)**;
**human review − Reverie = +0.185 ±0.033 (significant)**.

Across **8 landscapes × 6 seeds**, which is the estimate to quote because it does not
condition on one draw of the hidden rules:

| | effect | landscapes positive | per-landscape range |
|---|---|---|---|
| retrieval (Reverie − replay) | **+0.041 ±0.016** | 7/8 | −0.007 to +0.061 |
| human review | **+0.174 ±0.021** | 8/8 | +0.126 to +0.232 |

Human review is worth roughly **4×** the entire retrieval machinery.

**Two metric warnings, both load-bearing.**

*Which half you score flips the sign.* Reverie starts with an empty graph; replay is
useful from task two. Whole-run scoring charges Reverie a cold start it pays once per
run, and that is enough to turn +0.041 into −0.044. Second-half is the honest steady-state
claim and is what the pre-2026-09-04 numbers used, but they never said so — which is how
"Reverie beats replay" and "Reverie loses to replay" were both true of the same data.

*Which landscape you draw flips the significance.* One landscape gives +0.026 ±0.027
(n.s.); eight give +0.041 ±0.016 (significant). Report the landscape-averaged figure.

Reported as *paired* differences because every arm sees byte-identical task sequences;
pairing removes seed variance and is the reason a +0.041 effect is resolvable at all.

## Results

Numbers below are from the committed notebook runs and scripted sweeps.

### E1 — Attribution sample complexity

**This answers HLD open question #5, and the answer is uncomfortable.**

Recalls needed to reach Spearman ρ = 0.5 between estimated utility and true effect:

| P(recalled memory actually mattered) | Recalls to ρ=0.5 |
|---|---|
| 20% | **not reached within 10,000** |
| 35% | 6,400 |
| 50% | 3,200 |
| 75% | 1,600 |
| 100% | 800 |

Two findings, one of them the opposite of what the HLD assumed:

1. **Sample complexity is dominated by recall precision, not volume.** The
   binding constraint is how often a recalled memory actually influenced the
   task. This makes recall precision an *attribution* concern, not merely a
   token-budget concern — a connection HLD §10 does not draw.

2. **Co-occurrence barely matters.** Sweeping `cooccurrence` from 0.0 to 1.0 left
   the crossover flat at ~3,200 recalls. The design's assumption that locked-together
   recall sets are the identifiability bottleneck is **not supported**. Variation in
   activation *shares* between recalls carries enough signal on its own. This also
   weakens the case for ablation (see E4, which finds the same thing independently).

**Implication.** At a realistic relevance of ~50%, useful attribution needs a few
thousand attributed recalls. That is plausible for a busy agent over weeks, and
implausible for a hobbyist trying Reverie for an afternoon. The README must set that
expectation honestly, and `reverie doctor` should report progress toward it rather
than letting users conclude the system is broken.

### E2 — Credit assignment

Spearman ρ at 6,400 recalls:

| Rule | ρ | vs default |
|---|---|---|
| **activation** (HLD §9.2 default) | **0.644** | — |
| citation (`used_memories`) | 0.614 | −5% |
| rank | 0.600 | −7% |
| uniform | 0.518 | −20% |

1. **Activation weighting earns its complexity.** +20% over the uniform null. HLD
   §9.2 is validated; keep it.

2. **Citation is *worse* than activation, which was not expected.** The notebook
   originally described `citation` as an upper bound, on the reasoning that perfect
   relevance information must beat a heuristic. It does not. Zeroing out credit for
   irrelevant memories means those memories never accumulate evidence at all, so
   they sit at the prior forever and cannot be ranked. Diffuse, slightly-wrong credit
   turns out to be more informative than sharp, narrow credit.

   **This changes a design decision.** HLD §9.4 calls `used_memories` "the single
   highest-value field an integrator can populate" and recommends building the SDK
   around it. On this evidence it is not worth the integration burden as a *credit
   rule*. It may still be worth collecting for offline analysis, but it should be
   demoted from the primary integration ask.

### E3 — Retrieval mechanism

Planted chains, load-bearing fact *k* hops from a vocabulary-disjoint cue, buried in
distractors that do share cue vocabulary. 500 nodes, 40 probes.

| Mode | Chain recall | Target hit rate | Tokens | p50 latency |
|---|---|---|---|---|
| fts + vector top-k | 0.21 | 0.0% | 189 | 4.9 ms |
| 1-hop | 0.42 | 2.5% | 240 | 5.1 ms |
| spreading (3-hop) | **0.81** | 5.0% | 313 | 5.3 ms |

*Re-measured under two-channel recall (E6c).* Chain recall improved 0.68 → 0.81 and the
terminal-fact hit rate doubled, 2.5% → 5.0%, for ~18% more tokens. The conjunctive seed
boost helps the traversal as well as the precision channel.

**The HLD §10.3 claim is half-supported, and the half that fails is the half that
was used to sell the project.**

- *Supported:* spreading activation retrieves **3.9× more of the causal chain** than
  top-k (0.81 vs 0.21) for 66% more tokens. The graph genuinely surfaces associated
  material that vector search misses.
- *Not supported:* the specific claim that it surfaces the **terminal load-bearing
  fact** three hops out. Hit rate is 2.5%, barely above the top-k baseline of 0%.

The cause is attenuation: with `hop_decay = [1.0, 0.6, 0.35]` and `max_nodes = 12`,
a 3-hop node's activation is roughly 20% of a seed's before ranking, and it loses the
budget competition to nearer, better-scoring nodes. **The mechanism reaches the fact
and then the ranker discards it.**

Candidate fixes, none yet tested:
- Flatter `hop_decay`, at the cost of precision.
- A terminal-node bonus for nodes with high in-degree from the activated set.
- A reserved budget slot for the highest-activation node beyond hop 1.

Until one of these is measured, **the `path` demo in HLD §18 should not be the
headline**: `reverie path A B` will find the chain because BFS is exhaustive, but
ordinary `recall` will not surface its endpoint. Shipping a demo that does not
reflect default recall behaviour is exactly the kind of overclaim §18 warns against.

Latency at 500 nodes is a non-issue (p50 ~5 ms). The scaling sweep in the notebook is
less comfortable: p50 rises to **69.8 ms at 8,100 nodes**, approaching the 100 ms budget
in §10.2. The `visited` cap holds at 10 throughout, so the cost is per-visit work rather
than traversal breadth — but the budget is not the non-issue the earlier run suggested.

### E4 — Poisoning and self-correction

A memory that is confidently wrong and causes failure 85% of the time.

> ⚠️ **This result no longer reproduces.** The table below was measured against the
> pre-two-channel engine. Re-executed against current code, the poisoned memory is
> **never quarantined** in this scenario. Treat the tier gradient as a historical
> observation about the old ranking path, not a current property of the system. See
> *Quarantine reliability is an open problem* below.

Original measurement (superseded):

| Outcome signal tier | Recalls to quarantine |
|---|---|
| 1 — ground truth | 23 |
| 2 — human signal | 29 |
| 3 — behavioural proxy | 60 |
| 4 — LLM judgment | 162 |

Control arm (attribution disabled): poisoned memory **never** demoted, still served
in 50/50 subsequent recalls.

1. ~~**G8 holds under ground truth.**~~ **Withdrawn.** It held on the engine as it stood
   at the time. Under current code the same scenario never quarantines, and held-out
   seeds put detection at 5/20 even in the tuned configuration. G8 — "memory can be
   shown to be wrong and removed automatically" — is **not currently satisfied**.

2. **The tier system is empirically justified.** A 7× spread between tier 1 and
   tier 4 is the concrete argument for HLD §9.1's insistence on real evidence.
   `report_outcome(..., tier=4)` is not useless, but it is 7× slower to protect you.

3. **The control arm still behaves as designed** — with attribution disabled the poison
   stays `active` and is served in 50/50 subsequent recalls. But since the treatment arm
   no longer removes it either, this is no longer a comparison-table row. Do not ship
   the claim "accumulate-only memory systems cannot do this" until quarantine works.

4. **Ablation contributed nothing** (23.4 recalls at 0% vs 24.7 at 30%). Combined
   with E1's co-occurrence result, the ablation machinery in HLD §9.4 is not paying
   for itself in these regimes. Keep the code — it is cheap and may matter under
   heavier confounding — but drop the claim that it is load-bearing.

---

### E5 — reverie-bench: does the agent actually improve?

`reverie/bench/` runs four arms over byte-identical seeded task sequences. The agent
policy is identical in every arm; only the evidence it receives differs. The baseline
is **naive history-stuffing**, not "memory off" — without memory an agent cannot learn
across episodes at all, so that comparison is rigged and its number worthless.

db_migration, 40 tasks × 8 seeds, success over the second half (chance = 0.25):

| Arm | Before restructure | **After** | Tokens | Evidence |
|---|---|---|---|---|
| `none` | 0.188 | 0.181 | 0 | 0.0 |
| `replay` (last 40 verbatim) | 0.738 | 0.719 | 506 | 19.5 |
| `reverie` | 0.181 | **0.525** | 419 | 7.4 |
| `reverie_noattr` | 0.188 | 0.537 | 421 | 7.4 |

Initially **Reverie performed no better than having no memory at all.** Three
structural bugs, all found within an hour of the harness existing:

1. **Episodic dedup destroyed the signal.** `_upsert_memory` deduplicated by embedding
   similarity at 0.88, but the discriminative part of an episode is often a two-token
   difference inside an identical sentence (`strategy=direct → failure` vs
   `strategy=sql_first → success`). Forty episodes collapsed into 20 nodes. Fixed:
   episodic nodes are events and are never deduplicated; volume is controlled by decay
   and pruning, the mechanism designed for it.

2. **Two budget knobs, one resource — and the wrong one bound.** `max_nodes=12` at
   ~25 tokens/node caps a brief at ~300 tokens, so `budget_tokens` was completely
   inert (capture flat across 200–1200). `max_nodes` now defaults to `None` and derives
   from the budget. **This was the fix that mattered.**

3. **Quotas reserved budget for classes that do not exist.** `type_quotas` held 50% for
   procedural memories, but the no-LLM path (HLD §8.9) produces none. `adaptive_quotas`
   renormalises over the classes actually present in the candidate pool.

Ablation, seeds 0–5, noise band ±0.120 capture:

| Config | capture |
|---|---|
| old: fixed quotas, `max_nodes=12` | −0.016 |
| adaptive quotas only | −0.005 |
| derived `max_nodes` only | +0.209 |
| both, budget 1200 (new defaults) | **+0.621** |

**Adaptive quotas alone did nothing** — inside the noise band. The fix that sounded
principled was not the fix that mattered; `max_nodes` was the binding constraint and
quotas only begin to matter once it stops capping. Recorded because the hypothesis
going in was the opposite.

Confirmation in the sensitivity sweep: `quota_episodic` is now **flat** across
0.2–0.8 (0.690 → 0.659, all inside noise) — the adaptive rule made the constant
irrelevant, which is what replacing a constant with a rule should look like.
`budget_tokens` went from inert to the dominant lever (0.008 → 0.682).

**Still not a win.** Capture is ~0.59 at a token ratio of 0.84 — two thirds of
replay's lift for 84% of its tokens. On the current efficiency frontier replay still
dominates, and the objective reflects that: capture +0.593, objective +0.254 after the
token penalty.

**Attribution lift remains zero** (`reverie` 0.525 vs `reverie_noattr` 0.537, well
inside noise). This is uninformative rather than negative — attribution can only
re-rank what recall surfaces, and until recall is competitive the measurement has
little to bite on.

**Generalisation holds, and checking it exposed a statistics bug.** An earlier 4-seed
run put `db_migration` at +0.70 and held-out `data_pipeline` at +0.27 against a
reported band of ±0.12, which read as a real failure to transfer. At 16 seeds:

| Family | capture @4 seeds | capture @16 seeds |
|---|---|---|
| db_migration | +0.772 ± 0.275 | **+0.547 ± 0.169** |
| api_integration | +0.708 ± 0.221 | **+0.529 ± 0.142** |
| data_pipeline *(held out)* | +0.492 ± 0.183 | **+0.539 ± 0.158** |

No gap. The held-out family matches the two used during development, so the defaults
are not a lookup table for `db_migration` and `api_integration`.

The reason noise read as a finding: **the standard error was computed wrongly.**
`sqrt(p(1-p)/n_tasks)` assumes independent Bernoulli trials, but outcomes within a run
are strongly correlated — a run either discovers the feature→strategy rule or it does
not, and everything after follows from that. Per-run accuracies on one family span
0.10 to 0.65. The effective sample size is the number of **runs**, not tasks, and the
binomial form understates the error by ~1.8×. `Score.noise_se` is now cluster-robust,
computed across runs. Under the corrected band the original comparison was 0.367 apart
against a combined ±0.485 — never significant.

**Rule of thumb:** a 4-seed comparison carries a band of roughly ±0.35 capture, which
is most of the interesting range. Use ≥16 seeds before believing any comparison, and
treat published numbers from fewer as provisional.

### E6 — The on-call regime: no crossover, and the reason why

The families in E5 have tiny context spaces (`db_migration` has 8). A 31-episode
recency window holds several examples of *every* context, so recency is guaranteed to
contain the relevant precedent and selective retrieval cannot beat it. Those families
structurally cannot test the design's claim.

`IncidentFamily` targets the intended regime: 12 services × 6 symptoms = **72 distinct
incident types**, so a recency window usually holds no matching precedent while a graph
of hundreds of episodes does. Second-half accuracy, 6 seeds (uniform chance 0.167,
majority-class 0.32):

| Arm | 50 tasks | 200 tasks | 600 tasks | Tokens @600 |
|---|---|---|---|---|
| `reverie` | 0.233 | 0.218 | **0.238** | 718 |
| `replay` (1200-token cap) | 0.353 | 0.370 | 0.397 | 1,147 |
| `replay` (unbounded) | 0.373 | 0.473 | **0.552** | **9,069** |

**Half the thesis is confirmed, the important half is not.**

*Confirmed:* unbounded replay becomes unaffordable exactly as predicted — 9,069 tokens
at 600 tasks, extrapolating to ~60k at 4,000. History-stuffing does not survive to the
scale the design targets.

*Not confirmed:* **Reverie is flat.** 0.233 → 0.218 → 0.238 across a 12× increase in
history. It does not exploit accumulated experience at all, and at 0.238 it sits below
the 0.32 a constant "always scale_out" policy would achieve. There is no crossover, and
on current evidence there is no trajectory toward one.

**Root cause — retrieval, measured directly.** Fraction of briefs containing at least
one precedent for the exact `(service, symptom)` pair being handled, over tasks 200–400:

| | exact precedents per brief | briefs with ≥1 |
|---|---|---|
| `reverie` | 0.13 | **13%** |
| `replay` (1200-token cap) | 0.56 | **42%** |

Reverie retrieves the relevant precedent **three times less often than taking the most
recent 37 episodes**, from a graph that provably contains it.

**Why: spreading activation is disjunctive, and precedent lookup needs conjunction.**
The cue names two entities (`service=svc-03`, `symptom=sym-2`). Activation spreads from
each into its own neighbourhood and the result is closer to a *union* — episodes about
that service, plus episodes about that symptom — when the thing needed is the
*intersection*, the episodes about both. With 72 pairs, matching on service alone
leaves a 1-in-6 chance of the right symptom. HLD §10.3 argues associative spreading
beats top-k because the useful memory is textually dissimilar to the cue; that argument
holds for the multi-hop chains in E3 and works against us here, where the useful memory
is the one matching the cue most precisely on *every* dimension at once.

Secondary: token efficiency. Reverie spends **93 tokens per usable evidence item**
against replay's 31 — roughly two thirds of the brief goes to entity, failure-mode and
theme nodes that carry no precedent the agent can act on.

### E6b — Two fixes, one real, and the crossover still does not exist

**Fix 1: episodic retention (severe, now fixed).** The graph was pinned at ~10 episode
nodes no matter how much experience accumulated. `nodes_created` was 0 on every
consolidation cycle after the first.

Cause: **salience gated episodic retention, when HLD §8.3 only ever intended it to gate
distillation cost.** Salience is scored per *unit* — everything buffered since the last
cycle — and novelty is cosine distance over the unit's concatenated text. Ten on-call
incidents covering ten service×symptom pairs *never seen before* still look
near-identical in aggregate to the previous ten. Novelty collapsed, salience fell below
threshold, and every batch after the first was archived without recording anything.

Low-salience units now run the free structured path (`NullDistiller` → `link`) and skip
only the model call. Verified: 200 tasks → 200 episode nodes, previously 10.
`units_distilled` still counts only units that spent model tokens, so it remains a cost
metric. Regression test: `test_low_salience_still_records_episodes`.

**Fix 2: conjunctive entity matching (implemented, ineffective).** `store.conjunctive_
neighbors` / `entity_degrees` / `node_degrees` plus `RecallEngine._conjunctive` score
candidates by IDF-weighted cue coverage × candidate specificity. A second genuine bug
was fixed on the way: `_candidate_entities` stripped `=`, so `service=svc-03` never
resolved to its entity node and structured cues had no entity signal at all.

The scoring is demonstrably correct at the seed stage — the exact precedent scores
1.000. It does not survive into the brief.

**Post-fix horizon sweep, 6 seeds, second-half accuracy:**

| Arm | 50 tasks | 200 | 600 | Tokens @600 |
|---|---|---|---|---|
| `reverie` | 0.240 | 0.178 | **0.234** | 751 |
| `replay` (1200-tok cap) | 0.300 | 0.358 | 0.372 | 1,147 |
| `replay` (unbounded) | 0.320 | 0.473 | **0.551** | 8,933 |

**Still flat.** Fixing retention did not change the outcome. Exact-precedent retrieval:

| | conjunctive off | conj, no specificity | conj + specificity | replay |
|---|---|---|---|---|
| precedent in brief | 6% | 6% | 4% | **42%** |

Specificity weighting made it slightly *worse*, which rules out the hub-penalty
hypothesis.

**Where the signal is lost is now precisely bounded.** The precedent is in the graph
(≥4 exact-pair episodes at 300 tasks). The seed stage ranks it top with score 1.000. The
brief contains it 4–6% of the time. So the loss is in **spreading activation and
ranking, between seeding and selection** — most likely activation accumulating on hub
nodes via many paths until it exceeds a correctly-seeded leaf. That is a specific,
testable next step: instrument `_spread` and `_score_node` on a single known-good seed
and find where it falls out of the top-k.

### E6c — Two-channel recall, and the crossover appears

The fix is structural, and it contradicts HLD §10.1. That section treats seeding as a
way to *enter* the graph, with spreading and ranking deciding what matters. For
precedent lookup the ordering is backwards: the seed already has the answer and the
traversal dilutes it.

Recall now runs **two independent channels**:

* **Association** — seed → spread → rank, unchanged. Finds the load-bearing fact three
  hops from a textually dissimilar cue (E3). This is what §10.1 describes.
* **Precision** — candidates covering essentially all of the cue's IDF-weighted
  entities are admitted on their own authority, ranked among themselves by coverage ×
  specificity, and given a reserved slice of the budget. They never enter the
  traversal, so hub-inflated activation cannot outrank them.

Getting the admission rule right took three attempts, and the failures are informative:

1. **Gating on the composite score** (coverage × specificity) admitted nothing. A
   perfect match scored 0.375 against a 0.5 threshold purely because it was attached to
   eight entities rather than three. *Whether a memory is about what was asked* is a
   property of cue coverage; *how distinctive it is* is a ranking question. Conflating
   admission with ranking silently emptied the channel.
2. **Specificity over total degree** penalised episodes for having a past and a future
   — the `preceded` temporal spine counted against them. Restricted to entity edges.
3. **No specificity at all** let generic `failure_mode` hubs ("restart did not
   mitigate", linked to everything) tie with the exact precedent at 1.000.

**Retrieval, 3 seeds, 400 tasks:**

| | exact precedent in brief | accuracy | tokens |
|---|---|---|---|
| association only | 4% | 0.222 | 797 |
| **+ precision channel** | **77%** | 0.353 | 768 |
| replay (1200-tok cap) | 42% | 0.380 | 1,185 |

**Horizon sweep, 6 seeds, second-half accuracy:**

| Arm | 50 tasks | 200 | 600 | Tokens @600 |
|---|---|---|---|---|
| **`reverie`** | 0.227 | 0.293 | **0.479** | **812** |
| `replay` (1200-tok cap) | 0.387 | 0.397 | 0.409 | 1,147 |
| `replay` (unbounded) | 0.387 | 0.475 | 0.517 | 8,908 |

**The crossover is between 200 and 600 tasks.** Reverie loses badly at 50, closes at
200, and at 600 beats budget-matched replay on accuracy (0.479 vs 0.409) using 29%
fewer tokens. Against *unbounded* replay it reaches 93% of the accuracy for 9% of the
token cost.

The shapes are the finding. Reverie rises — 0.227 → 0.293 → 0.479 — because more
history means a better chance the precedent exists. Budget-capped replay is flat —
0.387 → 0.397 → 0.409 — because a recency window cannot reach past itself no matter how
much history accumulates behind it. Unbounded replay keeps climbing but its cost grows
linearly and is already 8,908 tokens at 600 tasks.

**This is the first evidence for the project's central claim**, and it only appears in
the regime the design was actually aimed at: a landscape larger than a context window,
over thousands of invocations. Every earlier negative result came from benchmarking in
the one regime where recency cannot lose.

Still open: attribution lift is untested under the new retrieval path, and the
association channel remains dominated by generic failure-mode hubs — the precision
channel routes around that problem rather than solving it.

### E7 — Attribution lift is zero, and the reason is structural

The project's differentiating claim, measured under working retrieval for the first
time. 600 tasks, 8 seeds, second-half accuracy, cluster-robust bands.

Two blockers had to be cleared before the measurement could mean anything:

1. **The precision channel ignored the posterior.** It ranked by
   `coverage² × specificity`; utility never entered, so attribution had no path to
   influence the brief and lift would have been zero by construction. Utility LCB now
   weights ranking *within* the channel (`direct_utility_weight`), leaving admission on
   cue coverage alone.
2. **The benchmark had no signal to detect.** In a static landscape every precedent is
   simply true, so ranking by track record cannot beat ranking by recency.
   `IncidentFamily(drift_at=300)` rewrites half the landscape mid-run — a config change
   that makes previously-correct fixes wrong — so stale precedents become actively
   harmful.

| | attr + utility | attr, no utility | no attribution | replay (1200 tok) |
|---|---|---|---|---|
| **STATIC** | 0.439 ±0.029 | 0.425 ±0.039 | 0.436 ±0.028 | 0.390 ±0.022 |
| **DRIFT @300** | 0.362 ±0.036 | 0.365 ±0.035 | 0.353 ±0.033 | 0.320 ±0.030 |

Independently replicated in `E6_scale_and_retrieval.ipynb` at 6 seeds: static
0.466 / 0.473 / 0.471, drift 0.343 / 0.339 / 0.327. Same conclusion, and the drift lift
(+0.016 there, +0.009 here) is positive in both runs but inside the band in both — worth
re-checking at higher seed counts if procedural memories ever make it non-trivial, not
worth claiming now.

**Lift is +0.003 static and +0.009 under drift, against a difference band of ±0.049.**
Zero in both. The utility-aware ranking contributes nothing (0.362 vs 0.365). Reverie
still beats budget-matched replay in both conditions, so the retrieval win of E6c holds
— but it is *not* attribution doing the work.

**Why: attribution is aimed at the wrong memory class.** Measured over 600 episodic
nodes after 600 drifting tasks:

| | |
|---|---|
| episodic nodes | 600 |
| mean attributions per node | 9.4 (**median 1**, max 471) |
| mean evidence mass `(α−1)+(β−1)` | 0.23 |
| utility LCB spread | min 0.196 · **p25 = median = p75 = 0.250** · max 0.312 |

0.250 is exactly `utility_lcb(1, 1, 0.25)` — the untouched prior. **Three quarters of
the graph never leaves it.** A Beta posterior needs repeated participation to say
anything, and episodic memory is structurally the wrong shape for that:

* Attribution assumes **few nodes, many recalls each** — distilled lessons ("when X, do
  Y") recalled hundreds of times, accumulating a track record.
* Episodic memory is **many nodes, few recalls each** — one node per event, most seen
  once. E1 already found ranking needs thousands of attributed recalls; here the median
  node gets one.

A second, independent reason points the same way: **for an episode, the outcome is
already the content.** A node body reading `strategy=rollback — failure` states its own
result, and the agent reads it directly from the brief. The posterior is a second,
weaker encoding of information the content already carries. Even a perfectly estimated
utility would be redundant for episodic memory.

**This is a design correction, not just a null result.** HLD §9 treats attribution as
applying uniformly to all memories. It cannot. Attribution earns its complexity only for
memories that are (a) recalled repeatedly and (b) assert something whose outcome is not
already in the text — that is, **distilled semantic and procedural lessons**. Those come
from the LLM distiller, and there is no LLM distiller (M2 is a seam plus
`NullDistiller`).

So the honest status of the central claim: **still untested, and now for a specific
reason.** It was measured on the one memory class it cannot work on. The blocking
dependency is M2, not more attribution tuning. Two follow-ups worth doing:

- Exclude episodic nodes from the posterior machinery entirely — it is doing work that
  cannot pay off and adds noise to the peer distribution that quarantine reads from.
- Re-run E7 once real procedural memories exist. Until then, do not claim attribution
  lift in either direction.

### E8 — Human-in-the-loop review: no measurable knowledge transfer

A simulated reviewer (`reverie/bench/human.py`) is asked at session boundaries and
returns a rating plus up to two **procedural lessons** drawn from that session's
failures — the counterfactual ("for this service and symptom, the fix is X") that no
episodic record can contain. Lessons are written via `assert_fact(...,
memory_class="procedural")`, so this is also the first configuration in which the
memory class E7 identified as attribution's *only* viable target actually exists.

Two knobs: `frequency` (share of sessions reviewed) and `accuracy` (share of lessons
that are correct). 600 tasks, 6 seeds, second-half accuracy.

| Frequency (accuracy 1.0) | | Accuracy (frequency 0.25) | |
|---|---|---|---|
| no human | 0.449 ±0.051 | 1.0 | 0.448 ±0.042 |
| 0.10 | 0.458 ±0.022 | 0.8 | 0.479 ±0.056 |
| 0.25 | 0.452 ±0.045 | 0.6 | 0.464 ±0.043 |
| 0.50 | 0.463 ±0.039 | 0.4 | 0.462 ±0.033 |
| 1.00 | 0.449 ±0.040 | | |

**The decisive row is accuracy 0.4** — 60% of lessons confidently wrong — which performs
*no worse* than accuracy 1.0. If lessons were transmitting knowledge, wrong ones would
hurt. **The system is insensitive to lesson correctness**, so whatever the lessons are
doing, it is not conveying what the human knows.

Rating-only review (no lesson) was 0.440 against a 0.449 baseline — nothing, as
predicted.

Attribution lift with fallible lessons present (freq 0.5, accuracy 0.6), the condition
E7 said was needed — few nodes, recalled often, outcome not in the text, and some of
them wrong:

| attr + utility | attr, no utility | no attribution |
|---|---|---|
| 0.443 ±0.050 | 0.444 ±0.031 | 0.446 ±0.035 |

**Still zero.** Attribution did not demote the wrong lessons.

#### The horizon picture, and a refuted hypothesis

The obvious explanation — accumulated episodic precedent drowns out one or two lessons —
is **wrong**. Measured evidence composition in the brief:

| | lessons | episodes |
|---|---|---|
| 50 tasks | 57.8% of evidence weight | 42.2% |
| 600 tasks | **87.6%** | 12.4% |

Lessons *dominate* the brief. They are not drowned; they are crowding episodes out. And
accuracy still does not matter, which means the lessons being retrieved are largely for
**other contexts** — they occupy budget without matching the task at hand. (The
composition figure sums raw `Evidence.weight`, not weight × similarity², so it measures
what is *in* the brief rather than what the agent actually acts on. That gap is the
finding.)

Across horizons, with human review at frequency 0.5:

| tasks | no human | human | replay (1200 tok) |
|---|---|---|---|
| 50 | 0.267 ±0.122 | 0.240 ±0.069 | 0.420 ±0.158 |
| 150 | 0.269 ±0.027 | 0.316 ±0.071 | 0.422 ±0.088 |
| 300 | 0.313 ±0.032 | 0.380 ±0.062 | 0.378 ±0.066 |
| 600 | 0.434 ±0.026 | 0.478 ±0.039 | 0.409 ±0.033 |

A consistent **+0.04 to +0.07** at 150–600, same sign at three independent horizons, but
inside the band at every one individually — and the two 600-task runs disagree (+0.014
in the sweep above, +0.044 here), which is itself a noise estimate. Not claimable at 6
seeds.

#### What to conclude

**Human review is not currently transmitting knowledge**, and the accuracy-insensitivity
is the proof — not the flat frequency curve, which could have been a power problem. Any
small benefit that exists is more likely a side effect of adding `procedural`-class
nodes (which changes the adaptive-quota mix, and therefore which episodes get budget)
than of the lessons' content.

Three concrete follow-ups, in order:

1. **Lesson retrieval precision.** Lessons take 87.6% of brief weight while accuracy is
   irrelevant, so most retrieved lessons do not apply to the current task. Lessons name
   two features where episodes name three, so they under-cover a three-entity cue and
   are admitted to the precision channel on looser terms. Fix retrieval before
   concluding anything about human feedback.
2. **Re-run at ≥16 seeds.** The E5 rule was set for exactly this situation; a +0.05
   effect needs the seeds.
3. **Test the quota side-effect directly** — inject *inert* procedural nodes with no
   content and see whether the same benefit appears. If it does, the lessons are doing
   nothing at all.

The idea remains sound: a human knows the counterfactual and a transcript does not.
But on current evidence the plumbing does not deliver it to the agent, and
**human-in-the-loop should not be claimed as a feature yet.**

### E9 — Fixing the plumbing: human review works, attribution still does not

E8 found human lessons had no effect *and* that the system was insensitive to whether
they were correct. The cause was retrieval, not the idea.

**Admission was the bug.** Lessons name `{service, symptom}`; the cue names three
entities. IDF-weighted coverage came to ~0.78 against a 0.95 threshold, so lessons
**never entered the precision channel**. They reached the brief only through the
association channel, where `adaptive_quotas` hands the procedural class 50% of the
budget and there were only ~30 lessons to fill it — relevance-blind crowding, which is
exactly the 87.6%-of-weight-yet-accuracy-irrelevant signature E8 recorded.

The fix is conceptual: **a memory is an exact precedent if everything it claims to be
about is present in the cue**, even when the cue names more. Admission is now either

* `coverage ≥ 0.95` — the candidate accounts for the whole cue (an exact episode), or
* `containment ≥ 0.95` — the cue accounts for the whole candidate (a lesson that
  generalises over a feature the cue happens to specify).

A generalising lesson is *more* useful for dropping a decoy, not less. Also fixed:
`node_degrees` counted edges rather than distinct entity neighbours, double-counting
bidirectional relations and halving every containment ratio.

**Result — the falsifiable test passes.** Lesson quality now changes behaviour, which it
provably did not before (600 tasks, frequency 0.5):

*Re-measured 2026-09-04; second-half success, 6 seeds.*

| accuracy | success | paired vs no human |
|---|---|---|
| no human | 0.446 ±0.018 | — |
| 0.4 | 0.495 ±0.024 | +0.049 ±0.030 |
| 0.6 | 0.513 ±0.057 | +0.067 ±0.047 |
| 0.8 | 0.592 ±0.037 | +0.147 ±0.035 |
| 1.0 | **0.610 ±0.026** | **+0.164 ±0.022** |

Monotonic. Compare E8, where accuracy 0.4 and 1.0 were indistinguishable.

**How often must you ask?** (accuracy 1.0, 600 tasks, 8 seeds, second-half success)

| frequency | success | paired vs baseline |
|---|---|---|
| no human | 0.450 ±0.025 | — |
| 0.05 | 0.482 ±0.029 | +0.032 ±0.036 (n.s.) |
| 0.10 | 0.487 ±0.037 | +0.037 ±0.042 (n.s.) |
| 0.25 | 0.502 ±0.032 | +0.051 ±0.041 |
| 0.50 | 0.614 ±0.031 | +0.163 ±0.047 |
| 1.00 | **0.668 ±0.025** | **+0.217 ±0.034** |

**It does not saturate.** I predicted the curve would flatten by 10–20%; it climbs all
the way. Two lessons per reviewed session against 72 incident types means coverage is
still filling in even at frequency 1.0 — the ceiling is landscape size, not diminishing
returns. Review is the strongest single lever measured anywhere in this project.

#### Attribution: still zero, and now the reason is arithmetic

E7 predicted attribution would work once lessons existed — few nodes, recalled often,
outcome not in the text, some of them wrong. That condition now holds exactly, and
(freq 0.5, accuracy 0.6, 8 seeds):

| attr + utility | attr, no utility | no attribution | paired lift |
|---|---|---|---|
| 0.525 ±0.024 | 0.526 ±0.036 | 0.532 ±0.047 | **−0.008 ±0.058** (n.s.) |

**Zero.** E7's hypothesis is no longer untested — it is refuted. Re-measured 2026-09-04
across all three landscapes, every paired interval contains zero:

| landscape | attr + utility | attr, no utility | no attribution | paired lift |
|---|---|---|---|---|
| static | 0.460 ±0.024 | 0.438 ±0.012 | 0.433 ±0.021 | +0.027 ±0.034 (n.s.) |
| drift @300 | 0.377 ±0.028 | 0.386 ±0.027 | 0.351 ±0.023 | +0.026 ±0.035 (n.s.) |
| fallible lessons | 0.525 ±0.024 | 0.526 ±0.036 | 0.532 ±0.047 | −0.008 ±0.058 (n.s.) |

The pre-registered condition (row 3) is the one whose point estimate is *negative*.

One real bug was found and fixed along the way. Credit share was apportioned from raw
activation, and **precision-channel nodes bypass spreading, so their activation is 0** —
the memories doing most of the work in the brief were receiving no credit at all.
Lessons participated in a median of 50 recalls each and accumulated a median evidence
mass of 0.81. Two repair attempts, both measured:

| credit weighting | median evidence mass | utility LCB range |
|---|---|---|
| raw activation (broken) | 0.81 | 0.245 – 0.386 |
| raw score | 0.47 | 0.236 – 0.328 |
| **per-channel normalised** | 0.54 | 0.234 – **0.403** |

Raw score is *worse* because the channels' scores are deliberately incomparable —
precision is a confidence in [0,1], association is an activation product, and summing
them lets one scale swamp the other. Per-channel normalisation is the shipped fix.

It is still not enough, and the reason is a hard bound rather than a tuning failure:

```
total credit mass in the system == number of outcomes
                                == 600 over this run
nodes sharing it                == ~700 (650 episodes + ~50 lessons)
→ 0.86 units of evidence per node, before splitting across a 9-item brief
```

Measured on a live graph by `_attribution_diagnostic.py` (600 tasks, 4 seeds, reviewer at
accuracy 0.6):

| | |
|---|---|
| median attributions / node | **1.0** |
| mean attributions / node | 33.5 — the distribution is skewed, not merely sparse |
| nodes still exactly at the Beta(1,1) prior | **31%** |
| utility LCB quartiles | p25 **0.249** · p50 **0.250** · p75 0.276 |

Half the graph has not moved off the prior at all. (An earlier note here claimed 75% at
the prior with all three quartiles at 0.250; re-measurement puts it at 31% and p75 at
0.276. The conclusion is unchanged — a posterior sitting on its prior at the median
cannot rank anything — but the stronger phrasing was not supported.)

A Beta posterior cannot separate anything on one unit of evidence. **Attribution's
sample complexity is bounded by outcomes ÷ nodes**, which is E1's finding restated as a
budget: it needs either orders of magnitude more tasks, or a graph small enough that
each memory is recalled constantly. Episodic-heavy memory is structurally the wrong
shape, and per-episode retention makes it worse the longer the system runs.

Headline numbers were re-verified after the credit change: no-human 0.458 → 0.458,
human f=0.5 0.629 → 0.649. Unaffected, as expected given attribution does nothing.

#### Out-of-band review costs nothing

The design principle in HLD §4.1 is that nothing Reverie does may slow the session it is
observing. The first human-review build violated it — `_review()` consolidated
synchronously at the session boundary, on the reasoning that a correction which only
takes effect next cycle is worthless. That reasoning was wrong, and testable:

| | accuracy | wall clock |
|---|---|---|
| sync consolidate on review | 0.638 ±0.031 | 27.8 s/run |
| **async, deferred to schedule** | **0.629 ±0.028** | 28.5 s/run |

Indistinguishable. `assert_fact` already writes the lesson node and its entity edges
directly, so the lesson is live either way; the synchronous cycle was only draining
buffered *episodes* on the agent's critical path for no benefit. Async is now the
default.

This matters more at scale than the numbers suggest: consolidation cost grows with graph
size, so a synchronous review that is free at 600 episodes is not free at 60,000.

#### What this means for the project

The evidence now points somewhere specific and away from the original thesis:

* **Human-authored procedural memory + precision retrieval is the product.** +0.227 at
  full review, monotonic in both frequency and quality, and it beats every other lever
  measured.
* **Outcome attribution is not earning its complexity.** Three independent tests (E4
  poisoning aside), and the binding constraint is arithmetic rather than fixable by
  better credit assignment. HLD §9 should be treated as unproven and a candidate for
  removal, not defended.
* The honest framing for a README is *"memory that a human can teach, retrieved
  precisely"* — not *"memory that learns which memories are worth having."*

### E10 — Stress tests: where it breaks, and how it handles a wrong human

#### 1. Scale: the governing variable is repetition, not size

Landscape size swept against a fixed task budget. Chance is 0.167.

| contexts | tasks/context | no human | human f=0.5 | replay (1200 tok) | Reverie − replay |
|---|---|---|---|---|---|
| 72 (12×6) | 8.3 | **0.449** | **0.586** | 0.411 | **+0.038 ±0.014** |
| 300 (30×10) | 2.0 | 0.224 | 0.351 | **0.316** | −0.092 ±0.020 |
| 1200 (60×20) | 0.5 | 0.197 | 0.231 | **0.226** | −0.029 ±0.027 |

*The last row was previously labelled 1.0 tasks/context. At a fixed 600-task budget over
1200 contexts it is 0.5; the label was wrong, the measurement was not.*

**This qualifies E6c's headline and the qualification matters.** Reverie beats
budget-matched replay at 8.3 tasks per context, *loses* at 2.0, and loses at 0.5. The
advantage is not "at scale" — it is **at repetition density**. Retrieval can only
surface a precedent that exists; below ~2 recurrences per context there is usually
nothing to find, and replay's 38 recent items beat Reverie's 9 precise ones because
neither has the answer and breadth wins.

The honest claim is therefore: *Reverie wins where the same situations recur.* For
on-call that is the realistic case — services fail in the same handful of ways — but a
landscape of one-off incidents is not a fit, and the README must say so.

Human review lifts every density and is the **only** thing that helps at 0.5
tasks/context (0.197 → 0.231), because a lesson supplies knowledge that repetition
otherwise has to earn.

Latency stayed flat (~700 tokens, no growth) and wall-clock scaled roughly linearly
with tasks, so nothing broke structurally at 1200 contexts.

#### 2. Human mistakes: robust to being wrong, *fragile to being over-general*

600 tasks, frequency 0.5, 8 seeds, second-half success. `no human` baseline = **0.440**.

| error mode | accuracy 0.8 | accuracy 0.5 |
|---|---|---|
| random (independent slips) | 0.555 | 0.511 |
| systematic (a stable misconception) | 0.563 | 0.473 |
| adversarial (always the worst answer) | 0.561 | 0.487 |

**Structured error is no worse than random error**, which was not the expected result —
systematic and adversarial mistakes repeat rather than cancelling, so they should
entrench. They do not, and the reason is architectural worth naming: **episodes
cross-check lessons.** A wrong lesson for context X coexists with episodes recording
that strategy failing at X, and the agent's evidence vote sees both. Even adversarial
review at 50% accuracy lands at 0.487 against a 0.440 baseline — bad advice costs you
part of the benefit, not the system.

*Re-run note:* the 2026-08 numbers put adversarial-at-50% marginally **below** baseline
(0.450 vs 0.456) and the claim was "at or above". Re-measured, every error mode at every
accuracy is *above* baseline, and all but systematic-at-50% significantly so. The
taxonomy is unchanged and slightly stronger than first recorded.

#### 3. Over-generalisation is the failure mode that actually hurts

The reviewer keeps only `service` when the true rule also depends on `symptom` — a
confident, correct-sounding lesson that is wrong five times in six:

16 seeds, second-half success:

| | success | paired vs no human |
|---|---|---|
| correct abstraction (`service`+`symptom`) | 0.614 ±0.020 | +0.164 ±0.021 |
| **over-general (`service` only)** | **0.435 ±0.017** | **−0.016 ±0.015** |
| no human at all | 0.451 ±0.014 | — |

**Below baseline**, significantly, though by less than first recorded (−0.016 rather than
−0.045). This is the only condition measured anywhere in which human review
makes the agent worse than silence, and it is a direct consequence of the containment
admission rule from E9: a lesson naming a subset of the cue's entities is admitted to
the precision channel at full confidence, because subset-matching is exactly what lets
legitimate generalisations in. The system trusts the human's abstraction and has no way
to check it.

This is precisely what attribution was supposed to catch — an over-general lesson is
recalled constantly and fails most of the time — and attribution does not work (E9).

**Proposed fix, not yet built:** validate a lesson's generalisation against episodic
evidence at consolidation time. If a lesson claims `service=X → Y` but episodes for
`service=X` show `Y` failing under some symptoms, the lesson is over-general: narrow it
to the contexts it actually holds for, or demote its confidence. This is cheap, runs out
of band, needs no posterior, and does not depend on attribution's sample complexity. It
is the highest-value unbuilt item.

#### 4. Stale human knowledge degrades gracefully

Drift at task 300; reviewer stops being asked at the drift point, so every lesson they
gave is about the old world:

8 seeds, second-half success:

| | success | paired vs drift/no human |
|---|---|---|
| drift, no human | 0.369 ±0.017 | — |
| drift, human throughout | 0.475 ±0.026 | +0.107 ±0.031 |
| **drift, human stops at drift (stale)** | **0.432 ±0.029** | **+0.063 ±0.022** |

Stale lessons remain net positive (0.432 > 0.369). Half the landscape did not drift, so
half the lessons stay valid, and accumulating episodes erode the rest. Staleness costs
about half the benefit rather than inverting it — the failure mode is graceful, unlike
over-generalisation.

### E11 — LLM-agent probe: does the cross-check property survive a real model?

**The load-bearing threat to validity.** Every number in E1–E10 comes from `EvidenceAgent`,
a scripted similarity-weighted voter. The E10 robustness finding — *robust to a reviewer
being wrong, fragile to one being vague* — rests entirely on **episodes cross-checking
lessons**: a bad lesson sits beside episodes showing that action failing, and the agent's
vote sees both. A real LLM might instead follow the confident-sounding imperative and
ignore the contradicting evidence, which would invert the result.

Cheap decisive test: build Reverie-shaped briefs containing one confident **wrong** human
lesson (`strategy=restart`) plus *N* pairs of contradicting episodes (restart → failure,
failover → success), and ask a real model to choose. Model: **`claude-opus-5`**, run
headless via `claude -p`, 10 trials per condition, 40 calls total. Reproducible from
`experiments/probes/`.

| contradicting episode pairs | Opus 5 → failover (evidence) | Opus 5 → restart (wrong lesson) | scripted agent |
|---|---|---|---|
| 0 | 0/10 | **10/10** | follows lesson |
| 1 | 4/10 | 6/10 | follows evidence |
| 2 | **10/10** | 0/10 | follows evidence |
| 4 | **10/10** | 0/10 | follows evidence |

**The property holds.** With two or more contradicting precedents, Opus 5 overrides a
confident human lesson unanimously. It is marginally more lesson-loyal than the scripted
voter — which flips at one pair, where the model is split 4/6 — but the qualitative
finding is the same, and the scripted agent is if anything the *less* lesson-loyal
instrument. E10's error-mode taxonomy is not an artifact of the agent policy.

**And it explains the one failure mode mechanistically.** At **zero** contradicting
episodes both agents follow the wrong lesson unanimously — there is nothing to
cross-check against. That is precisely what over-generalisation creates: a lesson that
claims authority over contexts where no local episodes exist yet. The guard in
`refined-approach.md` is therefore aimed at the right thing, and the reason it cannot be
replaced by "just let episodes outvote it" is now measured rather than argued.

**Caveats.** One model, one landscape, single-turn choices with a rendered brief rather
than a live agent loop; 10 trials per cell gives wide bands on the N=1 split
specifically. The N=0 and N≥2 endpoints are unambiguous.

### E12 — The over-generalisation guard: built, tested, and it does not work

`Consolidator.validate_scopes` (HLD 8.7b) implements the mechanism designed in
`refined-approach.md`. It is deliberately **action-agnostic**: it never parses a lesson
body, so it does not care what action was recommended or how a domain spells it. It tests
only the *uniformity* a scope claim implies — if episodes inside a lesson's claimed scope
split cleanly into succeeding and failing groups along some entity the lesson does not
name, the scope is too coarse. Candidate splits are binary "linked to entity e / not",
which needs no notion of a feature at all. It narrows rather than deletes, linking the
old lesson via `superseded_by`.

Four predictions were locked before running. **The primary one failed.**

Re-measured 2026-09-04, 6 seeds, second-half success, guard off → guard on:

| | prediction | measured | verdict |
|---|---|---|---|
| **H1** | over-general recovers toward the correct-scope arm (0.614) | 0.435 → 0.461, **+0.026 ±0.031 (n.s.)** — still 0.15 short of correct-scope | ❌ **refuted** |
| **H2** | guard does *not* help action errors | random −0.011 ±0.057, adversarial +0.010 ±0.034 — both n.s. | ✅ confirmed |
| **H3** | small net negative under correct-scope review | −0.012 ±0.047 — zero within noise | ✅ confirmed (cheaper than predicted) |
| **H4** | benefit tracks repetition density | no benefit to track | ⚪ vacuous |

**And a case E12 never tested:** the guard with *no reviewer at all* is also inert
(+0.006 ±0.020, 16 seeds). This needed checking because consolidation promotes
corroborated episodes into lessons on its own, so the guard has candidates to act on even
when no human ever speaks — it was a live hypothesis for the retrieval-arm regression
until measured.

**The guard fires and does nothing.** Under over-general review it narrows or demotes
tens of nodes per run against 0 with the guard off, so the mechanism is active and
finding the over-general lessons. Task performance does not move.

**Why, structurally.** The guard can *remove* harm but cannot *create* coverage. An
over-general lesson claiming `service=X` is right for one symptom in six; narrowing it
yields a correct lesson for that one symptom and leaves the other five with **no lesson
at all**. Those contexts fall back to episodic recall — the no-human baseline. So the
guard's ceiling is ~0.451 (no review), not 0.614 (correct review), and measured 0.461 it
reaches roughly that ceiling and no further. H1 was structurally impossible as written,
and locking it before running is what made that visible instead of arguable.

H2 and H3 passing matter more than they look: together they say the guard is *correctly
scoped and nearly free* — it ignores action errors, and it costs nothing when the human
was right. It is a well-behaved mechanism aimed at a problem it cannot solve.

**What this implies.** The measured failure mode is not "bad lessons are present" but
"good lessons are absent for most of the claimed scope". Removing the bad rule does not
fill that gap. Two directions the evidence actually supports, neither of them this guard:

1. **Ask the reviewer to narrow, rather than narrowing for them.** The guard detects
   over-generalisation reliably (~49 detections/run). Surfacing that as a targeted
   follow-up question — "your `service=X` rule fails on sym-2 and sym-4; what should
   happen there?" — converts detection into coverage, which is the thing that moves the
   number. This makes review *adaptive* instead of random-sampled.
2. **Accept over-generalisation as bounded.** It costs 0.016 against no review (0.435 vs
   0.451) — even smaller than first measured. That is the smallest problem currently on
   the board, and E9's frequency curve says a percentage point of extra review buys more
   than fixing it.

The guard stays in the tree, default on, as a detector — its output is the input to (1).
Do not claim it as a performance feature.

E4's tier gradient replicates on held-out seeds (24 / 30 / 61 / 213 recalls, seeds
100–129, never used during tuning). The false-positive rate did not.

The peer-relative MAD rule as first written destroyed **4.7 of 12 healthy memories**
via a cascade: as healthy memories converge, MAD collapses, the cutoff
`median − k·1.4826·MAD` creeps up toward the median, ordinary lower-tail variation
reads as outlier, and each false quarantine shrinks the peer set and shifts the
median further. Adding a severity cap (`quarantine_max_fraction`) cut false positives
to 0.30/12 but dropped detection from 27/30 runs to 5/20.

Swept on held-out seeds, tier 1:

| `quarantine_max_fraction` | Detected | False positives /12 |
|---|---|---|
| 0.5 | 4/20 | 0.35 |
| **0.6 (current default)** | **5/20** | **0.30** |
| 0.7 | 6/20 | 0.70 |
| 0.9 | 6/20 | 2.85 |

Detection never exceeds 6/20 anywhere in the sweep, so the cap is not the binding
constraint — something else limits sensitivity (likely `quarantine_min_peers` falling
back to the absolute floor once peers thin out). **This rule is not solved.** The unit
test `test_bad_memory_is_quarantined` passes because it uses a single memory and is
far easier than the multi-memory held-out scenario; that is a testing gap, not
evidence of correctness.

### E13 — Is any of this reproducible? Two protocol choices decide the headline

Run last, and it should have been run first. Everything above was re-measured against the
current engine; three things came out that qualify every earlier number in this file.

**1. The benchmark was not reproducible.** `IncidentFamily._correct` derived the hidden
feature→strategy rules from the built-in `hash()` over strings, which Python salts per
process. The same seed therefore drew a *different landscape in every interpreter*:

| run | seed | replay accuracy |
|---|---|---|
| process 1 | 0 | 0.368 |
| process 2 | 0 | 0.400 |
| process 3 | 0 | 0.408 |

That spread is as large as several of the effects reported here. Fixed by
`reverie.bench.core.stable_hash` (blake2b over a repr); all 76 tests still pass. **No
number recorded before 2026-09-04 came from a known landscape**, which is why the re-run
was not optional.

**2. Which half of the run you score flips the sign of the retrieval result.**

| metric | Reverie | replay (1200 tok) | paired difference |
|---|---|---|---|
| whole run | 0.363 ±0.020 | 0.408 ±0.010 | **−0.044 ±0.021** |
| second half | 0.450 ±0.025 | 0.424 ±0.017 | **+0.026 ±0.027** |

Reverie begins each run with an empty graph; replay is useful from task two. Whole-run
scoring charges Reverie a cold start that a real deployment pays once and a benchmark
re-pays every run. Second half is the defensible steady-state claim and is what the
canonical numbers always used — but they never said so, which is how the same data
supported both "beats replay" and "loses to replay". Both are now reported everywhere.

**3. The landscape was an uncontrolled constant, not a controlled variable.** `seed`
varies the task *sequence*; `landscape_seed` varies the rules. Every earlier sweep held
the latter fixed, so 16 "seeds" gave a tight error bar around one arbitrary rule set.
Crossing 8 landscapes × 6 seeds (second-half success, paired within each landscape):

| | mean effect | landscapes positive | per-landscape range |
|---|---|---|---|
| retrieval (Reverie − replay) | **+0.041 ±0.016** | 7/8 | −0.007 to +0.061 |
| human review | **+0.174 ±0.021** | 8/8 | +0.126 to +0.232 |

The retrieval effect is real but small, and a single-landscape study could land anywhere
in that range — including on a null. Human review is positive on all 8 and is not
sensitive to the choice. **Quote the landscape-averaged figures.**

**What this cost, and what it bought.** Three recorded claims moved: the retrieval effect
weakened from +0.056 ±0.032 to +0.041 ±0.016, over-generalisation's penalty shrank from
−0.045 to −0.016, and the share of nodes pinned at the attribution prior fell from 75% to
31%. Nothing reversed. The attribution null, the review curves, the error taxonomy, the
density crossover and all four E12 verdicts reproduced.

## Design changes these experiments forced

Three bugs and two design reversals, all found by running the experiments rather than
by reading the code:

1. **The quarantine threshold did nothing.** HLD Appendix A sets
   `quarantine_threshold = 0.25` and `utility_lcb_quantile = 0.25`. But
   `utility_lcb(1, 1, 0.25)` is *exactly* 0.25 — the uninformative Beta(1,1) prior
   sits precisely on the threshold. Every memory with any blame at all was
   quarantined, good and bad at identical rates, which is why E4's first run showed
   a perfectly flat tier response. Replaced with a peer-relative MAD rule.

2. **Base-rate-relative quarantine is self-defeating.** The first fix judged
   memories against the scope's base success rate. A harmful memory present in most
   recalls drags that base rate down with it, lowering the cutoff meant to catch it —
   the worse the poison, the better it hides. Replaced with peer-relative.

3. **Quarantine must gate on evidence mass, not participation count.** Credit is
   split across a recall, so eight participations is under one unit of real evidence,
   and an LCB on a near-prior posterior always looks bad. Added
   `min_evidence_for_quarantine`.

4. **Salience gating silently discarded repeated experience.** Redundant units fell
   below the salience threshold and were archived before reaching `link`, so
   repetition registered as nothing — directly contradicting HLD §8.5, where
   reinforcement is how repetition is supposed to register. Added a cheap
   `reinforce_only` path.

5. **`used_memories` is not the high-value field the HLD claims** (E2, above).

---

## Adding an experiment

Add a notebook spec to `_build_notebooks.py` and rerun it. Conventions:

- Seed everything; `SEEDS = 24` unless there is a reason.
- Plot mean with a bootstrap CI ribbon (`band()`), never bare point estimates.
- Use the Okabe-Ito `PALETTE`; it is colourblind-safe and consistent across figures.
- `save(fig, name)` writes both PNG (300 dpi) and PDF into `figures/`.
- End with a **Findings** cell that states what the numbers mean, including when they
  contradict the design. Every reversal above came from writing that cell honestly.
