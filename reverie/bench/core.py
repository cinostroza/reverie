"""reverie-bench — does an agent actually get better at its job?

This is HLD M0, the harness the design says should exist before anything else.
Everything in ``experiments/E1..E4`` measures the *mechanism*: can attribution
rank memories, does the graph retrieve, does a poisoned memory get removed.
None of it measures the claim the project is actually built on, which is that
**an agent gets better at doing a task by having done it before**.

## The baseline problem

"Memory on vs memory off" is a rigged comparison. Without memory an agent
cannot learn across episodes at all, so the off arm is pinned at chance and the
on arm wins by construction. That number would be worthless.

The honest comparison is against what practitioners actually do today — paste
recent history into the context window. So the harness runs four arms over
identical task sequences:

| Arm | Evidence the agent sees | What it isolates |
|---|---|---|
| ``none`` | nothing | the floor: what pure per-episode reasoning achieves |
| ``replay`` | last N raw episodes, verbatim | the real competitor: naive history-stuffing |
| ``reverie`` | a budgeted recall brief | the full system |
| ``reverie_noattr`` | same, attribution disabled | the **attribution lift** (HLD 16.2) |

The claim worth publishing is not "memory helps". It is *"Reverie matches
history-stuffing accuracy at a fraction of the tokens, and attribution
accounts for X points of it"* — and either of those can come out negative,
which is the point of measuring.

## Why the environment is a contextual bandit

Each task exposes observable **features** and admits several **strategies**.
Hidden rules map features to the strategy that works. An agent that memorises
one instance learns nothing, because features are resampled every episode; only
the feature→strategy *rule* generalises. That is the smallest environment where
"having done it before" is worth anything, and it is programmatically checkable,
which HLD 16.2 requires.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

__all__ = [
    "Task",
    "Attempt",
    "TaskFamily",
    "Agent",
    "Evidence",
    "RunResult",
    "run_arm",
    "compare_arms",
    "stable_hash",
]


def stable_hash(*parts: object) -> int:
    """Deterministic stand-in for the builtin ``hash``.

    ``hash()`` over strings is salted per interpreter (PYTHONHASHSEED), so a
    benchmark that derives its hidden landscape from it draws a *different*
    landscape in every process and no published number reproduces. Measured
    before this was fixed: one identical seed gave 0.368 / 0.400 / 0.408 across
    three interpreters -- a spread as large as several of the effects we report.

    Anything that fixes the environment (rule mapping, drift set, flake draw,
    a reviewer's stable misconception) must go through this instead.
    """
    digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(slots=True)
class Task:
    """One instance of a task family."""

    family: str
    index: int
    features: dict[str, str]
    strategies: tuple[str, ...]

    @property
    def context(self) -> str:
        """Canonical, parseable rendering of the observable situation.

        Both the memory writer and the evidence reader go through this, so a
        lesson recorded under one wording is findable under the same wording.
        """
        return ",".join(f"{k}={v}" for k, v in sorted(self.features.items()))

    def describe(self, strategy: str) -> str:
        return f"{self.family}[{self.context}] strategy={strategy}"


@dataclass(slots=True)
class Attempt:
    task: Task
    strategy: str
    success: bool
    detail: str
    steps: int = 1


@dataclass(slots=True)
class Evidence:
    """One observation available to the agent: a context, an action, a result.

    Every arm reduces to a list of these. That is what makes the comparison
    fair -- the arms differ in *what evidence reaches the agent*, never in how
    the agent reasons over it.
    """

    context: str
    strategy: str
    success: bool
    weight: float = 1.0


class TaskFamily(Protocol):
    name: str
    strategies: tuple[str, ...]

    def sample(self, rng: random.Random, index: int) -> Task: ...
    def evaluate(self, task: Task, strategy: str) -> Attempt: ...


class Agent(Protocol):
    def choose(
        self, task: Task, evidence: Sequence[Evidence], rng: random.Random
    ) -> str: ...


@dataclass(slots=True)
class RunResult:
    arm: str
    family: str
    seed: int
    records: list[dict[str, Any]] = field(default_factory=list)

    def rows(self) -> list[dict[str, Any]]:
        return [{"arm": self.arm, "family": self.family, "seed": self.seed, **r}
                for r in self.records]


def run_arm(
    family: TaskFamily,
    agent: Agent,
    backend: Any,
    *,
    n_tasks: int = 40,
    seed: int = 0,
    arm: str = "none",
) -> RunResult:
    """Run one arm over one seeded task sequence.

    The task sequence depends only on ``seed`` and ``family``, so every arm
    faces byte-identical tasks in an identical order. Any difference between
    arms is therefore attributable to the evidence backend and nothing else.
    """
    task_rng = random.Random(seed * 7919 + 13)
    agent_rng = random.Random(seed * 104729 + 7)

    result = RunResult(arm=arm, family=family.name, seed=seed)
    backend.reset()

    # Human-in-the-loop backends need the family's ground truth to simulate a
    # reviewer who knows what should have been done. Set after reset(), which
    # clears per-run state.
    if hasattr(backend, "_truth_fn") and hasattr(family, "_correct"):
        backend._truth_fn = family._correct

    for i in range(n_tasks):
        task = family.sample(task_rng, i)

        evidence, tokens, meta = backend.retrieve(task)
        strategy = agent.choose(task, evidence, agent_rng)
        attempt = family.evaluate(task, strategy)

        backend.record(task, attempt, meta)

        result.records.append({
            "task_index": i,
            "context": task.context,
            "strategy": strategy,
            "success": int(attempt.success),
            "evidence_items": len(evidence),
            "tokens": tokens,
            "steps": attempt.steps,
        })

    backend.finish()
    return result


def compare_arms(
    family_factory: Callable[[], TaskFamily],
    agent_factory: Callable[[], Agent],
    backends: dict[str, Callable[[], Any]],
    *,
    n_tasks: int = 40,
    seeds: Iterable[int] = range(20),
) -> list[dict[str, Any]]:
    """Cross every arm with every seed. Returns tidy rows for pandas."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm, make_backend in backends.items():
            res = run_arm(
                family_factory(), agent_factory(), make_backend(),
                n_tasks=n_tasks, seed=seed, arm=arm,
            )
            rows.extend(res.rows())
    return rows
