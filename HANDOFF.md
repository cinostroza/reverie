# Reverie — handoff

*Written 2026-09-04, revised the same day after a full re-run. Read this first, then
`findings.md`, then `experiments/README.md`.*

This document assumes no memory of the work. It states where the project is, what is
proven, what is not, and what a future session would most easily get wrong.

---

## 1. What this is, in one paragraph

Reverie is a long-term memory service for AI agents: agents write experience to it,
it consolidates that experience out of band, and agents recall from it at the start of
their next task. It began as a system whose differentiator was **outcome attribution**
(memories earn a track record and demote themselves when they hurt). **That thesis is
dead — measured, three ways.** What survives and works is *precise precedent retrieval
plus a channel for humans to teach*, on workloads where situations recur.

---

## 2. The state in one table

| Area | Status |
|---|---|
| Core engine (store, ingest, consolidation, recall, CLI, SDK) | ✅ built, 76 tests pass |
| `reverie-bench` (4 families, 4 arms, objective fn, human oracle) | ✅ built, `reverie/bench/` |
| Two-channel recall (precision + association) | ✅ built, this is what works |
| Human review loop | ✅ built, strongest measured lever |
| Outcome attribution | ❌ **built and refuted** — do not "fix" it, see §5 |
| Quarantine (design goal G8) | ❌ does not fire; G8 unsatisfied |
| Over-generalisation guard | ⚠️ built, fires correctly, changes nothing (E12) |
| LLM distiller | ❌ never built — **the biggest untested lever** |
| Paper | ✅ compiles clean (8 pp), arXiv-ready; front matter is the non-anonymous preprint |
| Figure 1 | ✅ `paper/figures/fig1.pdf`, built by `experiments/_figure1.py` |
| Notebooks E1–E6 | ✅ regenerated and re-executed 2026-09-04 against current engine |
| E7–E13 | ✅ now real drivers: `_run_sweeps.py` + `_analyse.py`, results cached as JSON |
| Benchmark reproducibility | ✅ fixed — was salted by `hash()`, see §5.6 |
| NeurIPS checklist | ❌ not started |

---

## 3. What is actually proven

All on `reverie-bench`, 600 tasks, on-call landscape (12 services × 6 symptoms = 72
contexts), **paired over identical seeds**, **second-half success** (metric matters — see
§5.7). Re-measured 2026-09-04; `experiments/results/REPORT.txt` is the machine output.

| | success | tokens/prompt |
|---|---|---|
| budget-matched history-stuffing (the honest baseline) | 0.424 ±0.017 | 1,146 |
| Reverie, two-channel retrieval | 0.450 ±0.025 | 794 |
| + human-taught procedural memory | **0.635 ±0.020** | **704** |

Averaged over **8 independently drawn landscapes** — the figures to quote, because a
single landscape does not identify the retrieval effect:

- retrieval **+0.041 ±0.016**, positive on 7/8 (per-landscape range −0.007 to +0.061)
- human review **+0.174 ±0.021**, positive on 8/8

Human teaching is worth ~**4×** the entire retrieval machinery.

**Conditional on repetition density** (this is the most transferable finding):
8.3 tasks/context → we win (+0.038); 2.0 → we lose (−0.092); 0.5 → both near chance. Any
agent-memory number reported without its repetition density is uninterpretable.

**Reviewer-error taxonomy:** robust to reviewers being *wrong* (random, systematic and
adversarial error at 50% accuracy all land **above** the no-review baseline of 0.440 —
0.511 / 0.473 / 0.487 — because episodes cross-check lessons) but fragile to reviewers
being *vague* (over-generalisation → 0.435 against a 0.451 baseline, the only condition
below doing nothing). Validated on a real model: `claude-opus-5` overrides a confident
wrong lesson 10/10 given ≥2 contradicting precedents, and follows it 10/10 given zero —
which is exactly why over-generalisation is the dangerous mode. (E11's stored results in
`experiments/probes/results.tsv` were verified against the claims, not re-run; they cost
real API calls and the harness is intact.)

---

## 4. Repo map

```
reverie/            engine. bench/ holds the benchmark harness.
  bench/core.py     stable_hash lives here — every environment draw must go through it.
experiments/        README.md is the lab notebook (E1-E13) — the real record.
  _build_*.py       notebooks are BUILD ARTIFACTS. Edit the builder, not the .ipynb.
  _run_sweeps.py    E7-E13 drivers. One Cell per (condition, seed); parallel; JSON out.
  _analyse.py       the ONLY place statistics live. Paired + cluster-robust.
  _attribution_diagnostic.py   why attribution is zero, measured on a live graph.
  _figure1.py       builds paper/figures/fig1.pdf from cached results.
  _notebook_digest.py          dumps notebook stdout so a re-run can be diffed.
  results/          raw per-run rows + REPORT.txt (committed machine output).
  probes/           E11 LLM probe (mkbrief.py, run.sh, results.tsv)
paper/              main.tex, refs.bib (14 verified citations), check_refs.py
  figures/fig1.pdf  Figure 1: density crossover + lever sizes.
literature/         survey.md + gap-analysis.md — positioning, verified
findings.md         project memory / synthesis. Read at the start of any session.
refined-approach.md the "trust the advice, check the scope" design
reverie_hld.md      v0.4 design doc. Carries a "Read this first" reversal table.
```

**`reverie_hld.md` is a hypothesis, `experiments/README.md` is the evidence.** Where they
disagree, the experiments win and the HLD is stale. It says so at the top.

---

## 5. Seven things a future session will get wrong

**1. Do not try to fix attribution.** It measured zero lift three times: on episodic
memory, under environment drift, and under the exact conditions pre-registered as
necessary for it to work. The cause is arithmetic, not a bug — total credit mass equals
the number of outcomes, so 600 outcomes spread over ~700 nodes gives 0.86 units of
evidence each, before splitting across a 9-item brief. Measured on a live graph: median **1**
attribution/node, and a utility LCB whose p25/p50 are 0.249/0.250 — exactly the untouched
Beta(1,1) prior — with 31% of nodes never having moved. (An earlier draft said 75% at the
prior with all three quartiles on it; that was too strong. Conclusion unchanged.)
Concurrent work
(RoMeRL, arXiv 2608.02508) names this the "Memory-Reward Trap" independently. One real
bug *was* found and fixed here (credit was apportioned by activation, and precision-channel
nodes bypass activation, so they got no credit) — fixing it did not change the outcome.

**2. Use cluster-robust error bars, always.** `sqrt(p(1-p)/n_tasks)` assumes independent
trials. Outcomes *within a run* are strongly correlated — a run either discovers the rule
or doesn't — so it understates error ~1.8×. This manufactured one phantom finding (a
"generalisation gap" that vanished at 16 seeds). Compare `Score.capture_se`, not
`noise_se`. **Use ≥16 seeds**; a 4-seed band is ~±0.35 capture, most of the interesting
range. Prefer *paired* differences across identical seeds — that's what makes a +0.041
effect resolvable at all.

**3. Benchmark regime decides the answer before you measure.** The first three task
families have ≤8 contexts, so a 31-episode recency window holds every context and
retrieval *cannot* win. E5's conclusion ("replay dominates at every token budget") was an
artifact of that. Always report tasks-per-context.

**4. Re-run notebooks after touching the engine.** Committed outputs that predate a code
change are worse than none — E4's headline silently stopped reproducing. Check with:
`python -c "import os,glob;print(max(os.path.getmtime(f) for f in glob.glob('reverie/**/*.py',recursive=True)))"`
against notebook mtimes. All six were re-executed on 2026-09-04 and are current.

**5. Never write BibTeX from memory.** AI citations have ~40% error rate. All 14 in
`paper/refs.bib` were fetched from the arXiv API; three unverifiable ones are explicit
`PLACEHOLDER_*_VERIFY` entries and are *not* cited in the text. `python paper/check_refs.py`
enforces this. Note: Python's `urllib` fails SSL here — use `curl -sL`.

**6. Never derive the environment from `hash()`.** Python salts it per process, so
`IncidentFamily` drew a different hidden landscape in every interpreter and *no number was
reproducible* — one fixed configuration gave 0.368 / 0.400 / 0.408. Fixed by
`reverie.bench.core.stable_hash`. If you add a task family, route every environment
decision through it.

**7. Two protocol choices decide the headline; state both.**
*Which half you score:* whole-run charges Reverie a cold start and gives retrieval
**−0.044**; second-half gives **+0.041**. Second-half is the steady-state claim and is
what every canonical number here uses.
*Which landscape you draw:* `seed` sets the task order, `landscape_seed` sets the hidden
rules. Sweeping seeds alone yields a confident interval around one arbitrary rule set.
Use the `landscape` sweep (8 × 6) for anything you intend to publish.

---

## 6. Building the paper

There is still no system TeX install. The paper is built with **tectonic**, a single
self-contained binary (no install, no PATH change): download the Windows zip from
github.com/tectonic-typesetting/tectonic/releases, then

    tectonic -X compile paper/main.tex --keep-intermediates

It fetches the packages it needs on first run. `--keep-intermediates` keeps `main.bbl`,
which arXiv needs because it does **not** run BibTeX. The arXiv upload is exactly:
`main.tex`, `neurips.sty`, `main.bbl`, `figures/fig1.pdf`. Always test-compile that set in
an empty directory: that check caught a natbib author-year/numeric clash that the full
build silently tolerated and arXiv would have rejected.

Front matter is currently the **arXiv preprint** (author named, venue notice blanked).
Do not use `[final]`: it prints "NeurIPS 2025" in the footer, which reads as a false
acceptance claim. For an anonymous venue submission, drop the option and restore the
anonymous `uthor`.

---

## 7. Do these first, in order

*Items 1 and 3 of the previous list are done: all six notebooks were re-executed against
the current engine, E7–E13 now have real drivers, and Figure 1 is built. What remains:*

1. **Submit to arXiv** (author's account). Compiled and read end to end on 2026-09-26.
   Before submitting, make the repo public if the paper is to claim a code release: it
   says "we release reverie-bench", and github.com/cinostroza/reverie is private.

2. **Fill the NeurIPS checklist.** Required for D&B, not started.

3. **Consider promoting E13 to a contribution rather than a caveat.** The paper currently
   carries it as §"Two protocol choices that decide the answer". For a Datasets &
   Benchmarks venue that section may be the most useful thing in the paper — a worked
   demonstration that a plausible agent-memory evaluation flips its own sign under two
   choices nobody states. It could carry more weight than it currently does.

4. **The whole-run vs second-half gap is a finding, not just a caveat.** Reverie's cold
   start costs it the entire retrieval advantage over a 600-task run. Nothing measures how
   long the cold start actually is. A learning-curve figure (success vs task index, all
   arms) would answer "how many tasks before memory pays for itself" — a question a
   practitioner has and this benchmark can answer cheaply, since `_run_sweeps.py` already
   stores per-run records.

---

## 8. The one big open bet

**Build an LLM distiller.** It is the only untested capability that could plausibly move
the headline number, and the logic is direct: lessons are worth ~4× the retrieval
machinery, human time is the constraint on lessons, and a distiller is *lessons without
the human*.

An earlier note in `findings.md` argued a model can only summarise what happened while
only a human supplies the counterfactual. **That was too strong.** The episodes already
contain counterfactuals — `strategy=restart → failure` sitting next to
`strategy=failover → success` for the same context *is* the counterfactual, in the data.
A model reading those pairs can write the rule. `reverie/bench/agents.py` has an
`LLMAgent` adapter and `experiments/probes/` shows how to drive a real model headlessly
via `claude -p`, so the harness already exists.

Second bet, cheaper: **directed review**. E12's guard detects over-generalisation
reliably (~49 detections/run) but can't act usefully — it removes harm without creating
coverage. Turn it into a *question generator*: instead of asking the reviewer about a
random session, ask about the specific gap ("your `service=X` rule fails on sym-2 and
sym-4 — what should happen there?"). Same reviewer minutes, aimed at the binding
constraint. Directed review at 10% vs random review at 10% is a clean experiment.

---

## 9. Honest strategic read

The project as originally conceived — *"memory that learns which memories are worth
having"* — is dead, and the field already named the wall.

What is left is real but narrower, and it splits in two:

- **As a system**, it is a workflow product: precise precedent retrieval over recurring
  work, plus a human teaching channel. Good, differentiated (nobody does conjunctive
  precedent retrieval with a scope-aware review loop), but it needs the distiller before
  it is worth shipping, and it only works where situations recur.
- **As research**, the contribution is the *paper*, and it is nearly done. A rigorous
  negative result with a diagnosed mechanism, a benchmark whose controlled variable is
  repetition density, and a pre-registered refutation of our own proposed fix. That is
  genuinely useful to the 47 credit-assignment methods currently climbing the same wall.

**Do not let the paper wait on the system.** The results are in hand; the system needs
another build cycle.
