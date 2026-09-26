# Refined approach — evidence-scoped trust

*Round-2 synthesis, 2026-08-24. Supersedes the attribution-centred design in
`reverie_hld.md` §9.*

## The one-sentence version

**Trust the human's advice; verify the human's scope.**

A reviewer who says *"for `service=X`, do `Y`"* is making two separate claims: an
**action** claim (`Y` is the right fix) and a **scope** claim (this holds for all of
`service=X`). Reverie currently accepts both on the same authority. Every measured
failure of human input comes from the second claim, never the first.

## Why this is the right lever

The evidence is unusually clean about where the problem is:

| reviewer error | result | vs no-review baseline (0.456) |
|---|---|---|
| random, 50% wrong | 0.483 | above |
| systematic (stable misconception), 50% wrong | 0.501 | above |
| adversarial (worst answer), 50% wrong | 0.450 | at |
| **over-general (correct action, wrong scope)** | **0.411** | **below** |

Being *wrong* is survivable because episodes cross-check lessons — a bad action for
context C sits beside episodes showing that action failing at C, and the agent's vote
sees both. Being *vague* is not, because an over-general rule has **perfect containment**
(its entities are a subset of the cue's), so it is admitted to the precision channel at
full confidence for every context it nominally covers, and there is no counter-evidence
mechanism at the scope level.

This also explains why attribution cannot rescue it: attribution operates per-node, and
an over-general lesson is a *single* node that is right sometimes and wrong other times.
Its posterior converges to "mediocre" rather than "wrong," which is exactly the value
that neither promotes nor demotes.

## The mechanism

Runs at consolidation, out of band, no model call, no posterior.

```
for each asserted procedural lesson L: scope C_L, action A
    E ← episodes whose context ⊇ C_L and whose action == A
    if |E| < min_scope_evidence:  continue        # innocent until proven over-general
    p ← success rate over E
    if p ≥ scope_keep:            continue        # the scope claim holds

    # The action may still be right — the scope is too wide. Find where it holds.
    f* ← argmax over features f ∉ C_L of information_gain(E, split_on=f)
    if gain(f*) < min_split_gain: demote L        # no clean sub-scope; action is wrong
    else:
        for each value v of f* with success rate ≥ scope_keep:
            emit narrowed lesson (C_L ∪ {f*=v}) → A
        retire L, linked via superseded_by
```

Four properties that matter:

- **Needs no new signal.** Episodes already record `(context, action, outcome)`. This is
  data we throw away today.
- **Narrows rather than deletes.** A rule right for 2 of 6 symptoms becomes two rules,
  not zero. Deleting would discard a correct action claim over a wrong scope claim.
- **Cheap and out of band.** One grouped query per lesson at consolidation. Fits §4.1 —
  nothing on the agent's critical path.
- **Auditable and reversible.** Narrowing is a `superseded_by` edge, visible to
  `reverie why`. The system is overriding a human, which must never be silent.

Classically this is **specialisation toward the most specific consistent hypothesis** —
the S-boundary of a version space — with a decision-stump split to choose the
discriminating feature. That grounding is worth stating in the paper; it is not a new
learning algorithm, it is a known one applied where the field has only named the problem.

## The conceptual claim underneath

HLD §8.6 says human-asserted memories win all conflicts. That is the wrong default, and
so is treating them as ordinary evidence. The right relationship is **evidence-scoped
trust**: accept what the human uniquely knows (the counterfactual — what *should* have
been done, which no transcript contains), and check what the data can check (how widely
it applies).

This also resolves a tension we introduced in E9. Containment-based admission is what
lets a legitimate generalisation into the precision channel at all — and it is exactly
what lets a bad one dominate. The guard is the missing half of that design, not a patch.

## Predictions (lock before running)

1. **H1** — over-general review recovers from 0.411 to within noise of the correct-scope
   arm (0.621). *Confirmatory.*
2. **H2** — the guard does **not** measurably help under random/systematic/adversarial
   error, because those are action errors and the guard only touches scope. If it does
   help there, the mechanism is doing something other than advertised.
3. **H3** — the guard is a small net negative under correct-scope review, from
   occasionally narrowing a rule that was fine. Magnitude is the cost of the safety.
4. **H4** — benefit grows with repetition density, since narrowing needs within-scope
   episodes. At 1.0 tasks/context it should do nothing at all.

H2 and H3 are the ones worth caring about — H1 succeeding alone would be consistent with
a guard that simply suppresses lessons.

## Paper shape

**Framing:** *Outcome attribution is the expensive way to obtain what a human can supply
directly — provided you check the scope of what they supply.*

Contributions, in defensible order:
1. Head-to-head attribution vs human teaching in one system on one benchmark. Attribution
   zero across three ablations, with an arithmetic account (credit mass = outcomes ÷
   nodes) that independently reproduces RoMeRL's Memory-Reward Trap.
2. Evidence-scoped trust: the guard above, with the error-mode taxonomy showing precisely
   which reviewer failures need guarding.
3. `reverie-bench`: repetition density as the controlled variable; budget-matched
   history-stuffing as the honest baseline.

**Venue:** COLM or NeurIPS Datasets & Benchmarks. The benchmark is a real artifact and
D&B rewards honest negative results; a pure systems venue would want deployment scale we
do not have.

**Threat to validity to state plainly:** every number is from a synthetic contextual
bandit with a scripted agent, not an LLM agent on real incidents. The `LLMAgent` adapter
exists and is unused. Reviewers will ask, and the answer should be in the paper rather
than in rebuttal.
