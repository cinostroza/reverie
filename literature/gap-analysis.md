# Gap analysis — where Reverie's contribution actually sits

Round 2, 2026-08-24. Two questions: is our claimed contribution unclaimed, and does
prior art already solve the over-generalisation guard?

## Human-taught procedural memory is NOT novel

Procedural memory learned from feedback is a crowded 2025–26 area:

| System | Approach |
|---|---|
| **Memp** (arXiv 2508.06433) | distils trajectories into stepwise procedures; add/modify/delete on execution feedback |
| **EvoSkill** (2026) | generate–verify–refine loop over skills |
| **Hermes** | YAML blueprints via Designer/Coder pipeline with evaluator critique |
| **Memento-Skills** | procedural memory as markdown skills in stateful prompts |
| **Growing with Your Embodied Agent** (OpenReview 1su9RkTVT9) | human-in-the-loop lifelong skill learning; encodes feedback into reusable skills |

**Do not claim "human teaches the agent" as the contribution.** It is prior art.

## What IS unclaimed — verified against the closest paper

**Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation**
Belikova, Parchiev, Egorov, Davydenko, Gusev, Savchenko, Makarenko. arXiv:2606.23127
(fetched and checked, appeared in both searches)

Three gaps confirmed against it:

1. **It studies *learned* skills from execution traces, not human-supplied ones.**
2. **It observes over-generalisation but does not guard against it.** Its central tension
   is that "some skills generalize broadly across tasks and models, whereas others become
   specialized to role-specific workflows and lose effectiveness under transfer" — an
   empirical observation with no proposed mechanism.
3. **No comparison of outcome-attribution against alternatives.**

The broader survey literature agrees the guard is missing: over-generalisation is named
as "the sibling risk — a lesson learned in one context applied blindly in another," and
quality gates (confidence scores, contradiction checking, expiration) are described as
**"necessary but still underdeveloped."**

Also relevant: *Useful Memories Become Faulty When Continuously Updated by LLMs*
(arXiv 2605.12978) — memory degradation under repeated self-update, the failure our
guard is meant to prevent on the human-input path.

## Contribution, restated honestly

Not "humans can teach agents." Rather:

1. **Head-to-head, one system, same benchmark:** outcome attribution contributes **zero**
   across three ablations; human teaching contributes **+0.181 ±0.026**. Nobody has run
   this comparison — the field is busy engineering better attribution (RoMeRL, Mem-T,
   Memory-R2, HiMPO; 47 methods catalogued 2024–26).
2. **Evidence-scoped trust** — a concrete, cheap guard for the over-generalisation
   failure everyone names and nobody fixes. See `refined-approach.md`.
3. **Repetition density as the governing variable** for when memory retrieval beats
   recency at all. We measure the crossover; "at scale" is the wrong axis.
4. **reverie-bench** — a benchmark whose controlled variable is repetition density, with
   an honest baseline (budget-matched history-stuffing, not "memory off").
