# Reverie — findings

*Project memory. Read this first on any new session.*

## Current understanding

Reverie began as an **outcome-attribution** system: memories earn a track record, and
ones that correlate with failure demote themselves. Ten experiments later, that thesis is
dead and a narrower one is well supported.

**What works, measured (600 tasks, 72-context on-call landscape, 16 seeds, paired,
second-half success — re-run 2026-09-04):**

| | success | tokens |
|---|---|---|
| naive history-stuffing (budget-matched) | 0.424 ±0.017 | 1,146 |
| Reverie, two-channel retrieval | 0.450 ±0.025 | 794 |
| + human-taught procedural memory | **0.635 ±0.020** | **704** |

Averaged over **8 independently drawn landscapes** (the estimate to quote, because one
landscape does not identify the retrieval effect):

- retrieval **+0.041 ±0.016**, positive on 7/8 landscapes (range −0.007 to +0.061)
- human review **+0.174 ±0.021**, positive on 8/8

Human review is worth ~**4×** the entire retrieval machinery. Pairing is load-bearing —
the same effects are only marginal unpaired.

**Two protocol facts that decide the headline, both discovered by re-running:**

1. **Score the second half.** Whole-run scoring charges Reverie a cold start it pays once
   per run, and flips retrieval to **−0.044 ±0.021**. Second half is the steady-state
   claim; always say which you used.
2. **Vary `landscape_seed`, not just `seed`.** The former picks the hidden rules, the
   latter only the task order. Sweeping seeds alone gives a tight error bar around one
   arbitrary rule set.

**What does not work:** outcome attribution. Zero lift in three independent tests,
including under environment drift and under the exact conditions predicted to rescue it.

## Patterns and insights

1. **Retrieval, not attribution, was always the binding constraint.** Every mechanism
   that measured zero (attribution, drift-sensitivity, human lessons) turned out to be
   downstream of a retrieval defect. Fix retrieval and the effect appears.
2. **Association and precedent lookup want opposite behaviour.** Spreading activation
   travels *away* from the cue — right for multi-hop chains (E3: 3.9× more chain than
   top-k), catastrophic for "what happened last time on this exact thing" (4% → 78%
   once precedent lookup bypassed the traversal). A unified retrieval path silently
   breaks one of them.
3. **Admission and ranking are different questions.** Conflating them emptied the
   precision channel three separate times. Admit on coverage of the cue; rank by
   coverage × specificity.
4. **The system is robust to a human being wrong, fragile to a human being vague.**
   Random, systematic, and even adversarial reviewer error all stay *above* the no-review
   baseline of 0.440 (0.511 / 0.473 / 0.487 at 50% accuracy), because episodes cross-check
   lessons. But **over-generalisation drives performance below baseline** (0.435 vs
   0.451, −0.016 ±0.015) — the only measured condition where the system is worse than
   doing nothing.
5. **Repetition density, not scale, governs whether retrieval wins.** 8.3 tasks per
   context → we win (+0.038); 2.0 → we lose (−0.092); 0.5 → both near chance. "Beats
   recency at scale" is wrong as stated.
6. **The environment must never be derived from `hash()`.** It is salted per process, so
   the benchmark drew a different landscape in every interpreter and nothing was
   reproducible. One nominally fixed configuration gave 0.368 / 0.400 / 0.408. Use
   `reverie.bench.core.stable_hash`. This invalidated the *reproducibility* of every
   pre-2026-09-04 number, though re-running changed no conclusion.

## Lessons and constraints

- **Cluster-robust error bars, always.** `sqrt(p(1-p)/n_tasks)` understates error ~1.8×
  because outcomes within a run are correlated. It manufactured one phantom finding.
  Use ≥16 seeds; prefer *paired* differences across identical seeds.
- **Seeds are not landscapes.** Sweeping `seed` alone produces a confident interval
  around one draw of the hidden rules. Cross ≥8 `landscape_seed` values for any headline.
- **State the metric.** Second-half vs whole-run success disagree by enough to flip the
  retrieval result's sign. `_run_sweeps.py` records both on every run; `_analyse.py`
  prints both.
- **Benchmark regime decides the answer before you measure.** Three families with ≤8
  contexts made recency unbeatable by construction and produced a wholly misleading
  conclusion (E5). Always report tasks-per-context.
- **Re-run notebooks after touching the engine.** Committed outputs that predate a code
  change are worse than none. E4's headline silently stopped reproducing.
- **Salience must never gate retention** — only distillation cost. Gating retention
  pinned the graph at 10 nodes forever.
- **Episodic nodes are events; never deduplicate them.**
- **Credit share cannot be computed from raw scores across channels** — they are
  deliberately incomparable. Normalise within channel.

## Open questions

1. **Over-generalisation guard** (highest value, now designed — see
   `refined-approach.md`). Version-space specialisation: find episodes within the
   lesson's claimed scope, and if the action underperforms there, split on the
   highest-information-gain feature and narrow rather than delete. Four locked
   predictions, H2/H3 being the discriminating ones.
2. **Quarantine is broken.** The E4 scenario that once removed a poison in 23 recalls
   now never fires; held-out detection 5/20. **Design goal G8 is unsatisfied.**
3. **No LLM distiller.** Everything rests on episodic recall plus human lessons.
4. **Should attribution be cut entirely?** Leaning: keep as a documented negative,
   default it off. See `literature/survey.md` — the field calls this the Memory-Reward
   Trap and is actively engineering around it; we have evidence you can sidestep it.

## Round-2 literature outcome (2026-08-24)

Searched to check whether our contribution is unclaimed. Two corrections:

- **"Humans teach the agent" is prior art** — Memp, EvoSkill, Hermes, Memento-Skills, and
  a human-in-the-loop lifelong skill framework all do learned/fed-back procedural memory.
  Do not claim it.
- **Over-generalisation is named but unguarded.** The closest paper (arXiv 2606.23127,
  fetched and verified) observes that skills "become specialized to role-specific
  workflows and lose effectiveness under transfer" and proposes no mechanism. Survey
  literature calls quality gates "necessary but still underdeveloped."

So the contribution is the *guard*, plus the head-to-head, plus the benchmark — not the
human loop itself. Full analysis in `literature/gap-analysis.md`; design in
`refined-approach.md`.

## Paper thesis (current draft framing)

> Outcome attribution is the expensive way to obtain what a human can supply directly.
> We measure both arms in one system: attribution contributes nothing across three
> ablations, while ~2 human-authored lessons per reviewed session deliver +0.174 ±0.021 —
> roughly 4× the retrieval machinery. The enabling condition is a retrieval path that
> admits generalising lessons. The scope guard we designed to catch the one dangerous
> reviewer failure mode (over-generalisation) *fires correctly and does not help*, because
> it removes harm without creating coverage — that is E12, and it is reported as a
> negative result rather than a feature.

**Trust the advice; verify the scope.** A reviewer makes two claims — an *action* claim
and a *scope* claim. Only the second is checkable from data, and only the second has ever
hurt us: random, systematic and adversarial reviewer error all stay at or above baseline,
while over-generalisation alone goes below it.

Related work to position against: RoMeRL (Memory-Reward Trap), Mem-T, Memory-R2, HiMPO.
All engineer denser reward; none consider substituting human teaching.
