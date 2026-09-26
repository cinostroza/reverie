# Literature survey — credit assignment for agent memory

Bootstrapped 2026-08-23. Question driving the search: **our outcome-attribution result
measured zero lift three times with an arithmetic cause (credit mass = outcomes ÷ nodes).
Has anyone else hit this wall, and did they report it?**

Answer: **yes, and it has a name.** The finding is independently corroborated by
concurrent 2026 work. That is good news — it means our result is a property of the
approach, not a bug in our implementation.

## Primary hit — RoMeRL (verified against the abstract)

**RoMeRL: Balancing Feedback Coverage and the Memory-Reward Trap in Self-Evolving Agent
Memory via Reduced-Order Utility States**
Yang, Chen, Zhuang, Fan, Chen, Li, Yang, Tai. arXiv:2608.02508

Names our failure mode the **"Memory-Reward Trap"** and identifies the same two causes we
measured, in the same order:

1. *Dimensionality growth* — "trajectory-indexed utilities grow with the interaction
   history, thereby dispersing limited feedback over an ever-expanding state space."
   This is our arithmetic bound restated: total credit mass equals the number of
   outcomes, so more memories means less evidence per memory. We measured the endpoint —
   median 1 attribution per episodic node, 75% of the graph pinned at the untouched
   0.250 prior.
2. *Credit dilution across co-retrieved memories* — "irrelevant experiences may receive
   misleading utility updates." This is our observation that credit divides across a
   9-item brief, so a memory that did the work and a memory that rode along score alike.

**Their fix:** a fixed-dimensional per-task memory state factorised by outcome polarity,
replacing per-trajectory tracking with "a fixed set of semantic coordinates" —
i.e. *reduce the dimensionality so the feedback lands somewhere bounded.*

## The field's response, and where ours differs

Every method found responds to sparse memory credit by **engineering denser or
better-targeted reward**:

| Work | arXiv | Approach |
|---|---|---|
| Mem-T: Densifying Rewards for Long-Horizon Memory Agents | 2601.23014 | densify the reward signal |
| Tree-based Credit Assignment for Multi-Agent Memory | 2605.04811 | backpropagate through a memory-operation tree |
| Memory-R2: Fair Credit Assignment for Memory-Augmented Agents | 2605.21768 | fairer apportionment |
| HiMPO: Hindsight-Informed Memory Policy Optimization | 2606.16285 | disentangle tool/reasoning/memory error sources |
| Meta-Cognitive Memory Policy Optimization | 2605.30159 | belief-entropy self-supervision |
| AttriMem (earlier, cited in our HLD) | 2607.21106 | token-level attribution for local rewards |

**None of them consider giving up on attribution.** Our E9 result — human-taught
procedural memory at +0.181 ±0.026 paired, versus attribution at zero — suggests a
cheaper path that sidesteps the trap rather than engineering around it. That contrast is
the paper.

## Also relevant

- **Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures**
  (2606.08275) — notes that agent-removal counterfactuals carry a bias that *more
  samples cannot remove*. Relevant to our ablation arm, which E1/E4 independently found
  contributed nothing.
- **When to Forget: A Memory Governance Primitive** (2604.12007) — relevant to our
  broken quarantine (G8 unsatisfied).
- A 2026 survey catalogues **47 credit-assignment methods** published 2024–early 2026,
  confirming this is an active and crowded problem area.

## Implications for our positioning

1. **The negative result is corroborated, not novel as a problem statement.** Do not
   claim discovery of the memory-reward trap. RoMeRL names it and we should cite it.
2. **Our contribution is the clean quantitative characterisation plus the alternative.**
   We have an end-to-end measurement in a running system with an explicit bound
   (credit mass ÷ nodes), three independent ablations, and a demonstrated substitute
   that does not require solving credit assignment at all.
3. **Reframe the paper** away from "attribution fails" (known) toward "**attribution is
   the expensive way to get what a human can supply directly**" — with numbers on both
   arms and a measured cost curve for how often you must ask.
