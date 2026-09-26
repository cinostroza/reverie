"""A simulated human reviewer, for measuring human-in-the-loop feedback.

## Why this is the interesting experiment

E7 found attribution lift is zero, and the diagnosis was structural: a Beta
posterior needs repeated participation, but episodic memory is *many nodes
recalled once each* (median 1 attribution per node, 75% of the graph sitting
at the untouched prior). Attribution can only work on memories that are
recalled repeatedly **and** assert something whose outcome is not already in
the text. Nothing in the system produces those, because there is no LLM
distiller.

A human reviewer produces exactly that memory class, and produces it better
than an LLM distiller could. A model reading the transcript can only summarise
what happened; a human knows the **counterfactual** -- what should have been
done instead. That is the content an episode can never contain.

So this module tests two things at once:

1. Does human feedback improve task performance, and how often do you have to
   ask before it pays for itself?
2. Now that procedural memories exist and can be *wrong*, does attribution
   finally have something it can usefully demote?

The second question is why `accuracy` is a knob. A confidently wrong lesson is
recalled often, matches the cue well, and causes failures -- precisely the
shape attribution was designed for and has never yet been given.

## What is deliberately modelled

* **Humans are asked rarely.** `frequency` is the share of sessions that get a
  review at all. Ask after every session and real reviewers stop answering.
* **Humans give one or two lessons, not twenty.** `lessons_per_session`.
* **Humans comment on what went wrong.** Lessons are drawn from the session's
  failures, which is where the counterfactual lives.
* **Humans generalise, and sometimes over-generalise.** Lessons name only the
  features the reviewer believes matter, dropping decoys -- which is the whole
  value, and also the failure mode when they drop the wrong one.
* **Humans are sometimes wrong.** `accuracy`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .core import Attempt, stable_hash

__all__ = ["HumanLesson", "HumanFeedback", "HumanOracle"]


@dataclass(slots=True)
class HumanLesson:
    """A procedural lesson: under this condition, do this."""

    context: str          # canonical "k=v,k=v" over the features the human kept
    strategy: str
    correct: bool         # ground truth, for analysis only -- never shown to the agent
    entities: list[str] = field(default_factory=list)

    def render(self) -> str:
        # Canonical, parseable, and visibly distinct from an episode so the
        # brief reader can weight it differently.
        return f"LESSON {self.context} strategy={self.strategy}"


@dataclass(slots=True)
class HumanFeedback:
    reviewed: bool
    rating: float | None = None            # [0,1] share of the session that went well
    lessons: list[HumanLesson] = field(default_factory=list)


@dataclass(slots=True)
class HumanOracle:
    """Simulated reviewer over one session's attempts.

    `truth_fn` maps a task to the strategy that would have worked. In a real
    deployment this is the engineer's knowledge; here it is the family's hidden
    rule, which is what makes the experiment measurable.
    """

    frequency: float = 0.2          # share of sessions reviewed
    accuracy: float = 0.9           # share of lessons that are correct
    lessons_per_session: int = 2
    keep_features: tuple[str, ...] = ("service", "symptom")  # the human's abstraction
    rate_only: bool = False         # ablation: a rating with no lesson

    # How the human is wrong when they are wrong. Real reviewers do not err
    # uniformly at random, and the structured failures are worse because they
    # do not average out across repetitions.
    #
    #   random      -- independent mistakes; the benign case, and the only one
    #                  earlier sweeps measured
    #   systematic  -- a consistent misconception. The same context always draws
    #                  the same wrong answer, so repetition entrenches it instead
    #                  of cancelling it
    #   adversarial -- deliberately the worst available answer, for the §13
    #                  threat model: an insider or a compromised review channel
    error_mode: str = "random"
    # Sessions after which the reviewer stops being asked. Models a human whose
    # knowledge goes stale: they were right about the world as it was, and the
    # world moved. Their lessons stay in the graph, asserted and confident.
    stop_after_session: int | None = None
    _sessions: int = 0

    def review(
        self,
        attempts: Sequence[Attempt],
        truth_fn,
        rng: random.Random,
    ) -> HumanFeedback:
        self._sessions += 1
        if self.stop_after_session is not None and self._sessions > self.stop_after_session:
            return HumanFeedback(reviewed=False)
        if not attempts or rng.random() >= self.frequency:
            return HumanFeedback(reviewed=False)

        rating = sum(1.0 for a in attempts if a.success) / len(attempts)
        if self.rate_only:
            return HumanFeedback(reviewed=True, rating=rating)

        # Reviewers comment on what went wrong; a session that went fine
        # produces a rating and nothing else.
        failures = [a for a in attempts if not a.success]
        if not failures:
            return HumanFeedback(reviewed=True, rating=rating)

        picked = rng.sample(failures, min(self.lessons_per_session, len(failures)))
        lessons: list[HumanLesson] = []
        for attempt in picked:
            task = attempt.task
            kept = {k: v for k, v in task.features.items() if k in self.keep_features}
            context = ",".join(f"{k}={v}" for k, v in sorted(kept.items()))

            right = truth_fn(task)
            wrong = [s for s in task.strategies if s != right]

            if self.error_mode == "systematic":
                # Deterministic per context: the reviewer holds a stable
                # misconception rather than making independent slips, so
                # asking twice yields the same wrong answer twice.
                errs = (stable_hash(("misconception", context)) % 1000) / 1000.0
                mistaken = errs >= self.accuracy
                pick = wrong[stable_hash(("choice", context)) % len(wrong)]
            elif self.error_mode == "adversarial":
                mistaken = rng.random() >= self.accuracy
                pick = wrong[0]          # stable worst-case answer
            else:
                mistaken = rng.random() >= self.accuracy
                pick = rng.choice(wrong)

            strategy, correct = (pick, False) if mistaken else (right, True)

            lessons.append(
                HumanLesson(
                    context=context,
                    strategy=strategy,
                    correct=correct,
                    entities=[f"{k}={v}" for k, v in sorted(kept.items())],
                )
            )
        return HumanFeedback(reviewed=True, rating=rating, lessons=lessons)
