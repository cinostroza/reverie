"""Evidence backends -- the four arms.

Each backend answers one question: given the task about to be attempted, what
does the agent get to see, and what did it cost in tokens? Token accounting is
the point of the comparison. `replay` will always have *access* to more
information than `reverie`; the interesting question is whether Reverie
matches its accuracy for materially fewer tokens, and whether attribution
contributes anything on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..api import Reverie
from ..config import Config
from ..render import estimate_tokens
from .core import Attempt, Evidence, Task

__all__ = ["NoMemoryBackend", "ReplayBackend", "ReverieBackend"]

# Episodic bodies are written as "<family>[<ctx>] strategy=<s> — <outcome>",
# so the same canonical form that goes in comes back out.
_BODY = re.compile(r"^(?P<family>\w+)\[(?P<ctx>[^\]]*)\]\s*strategy=(?P<strategy>\w+)")
# Human lessons carry their own form so the brief reader can weight them apart
# from episodes: they are asserted counterfactuals, not observed events.
_LESSON = re.compile(r"^LESSON\s+(?P<ctx>[^\s]+)\s+strategy=(?P<strategy>\w+)")


@dataclass(slots=True)
class NoMemoryBackend:
    """The floor. Each episode is faced fresh."""

    name: str = "none"

    def reset(self) -> None: ...
    def finish(self) -> None: ...

    def retrieve(self, task: Task) -> tuple[list[Evidence], int, dict]:
        return [], 0, {}

    def record(self, task: Task, attempt: Attempt, meta: dict) -> None: ...


@dataclass(slots=True)
class ReplayBackend:
    """Naive history-stuffing: paste the last N episodes into context.

    This is what practitioners actually do, and it is the arm Reverie has to
    beat on cost rather than on accuracy. With a large enough window it sees
    strictly more than Reverie does.
    """

    window: int = 40
    # Cap by tokens rather than episode count. At long horizons the episode
    # count is not the real constraint -- the context window is. A budget of
    # None keeps the pure count-based window (an upper bound that a real
    # deployment could not afford once history runs to thousands of episodes).
    budget_tokens: int | None = None
    name: str = "replay"
    _log: list[tuple[Task, Attempt]] = field(default_factory=list)

    def reset(self) -> None:
        self._log = []

    def finish(self) -> None: ...

    @staticmethod
    def _cost(t: Task, a: Attempt) -> int:
        # Cost the same text an LLM would actually be shown.
        return estimate_tokens(
            f"{t.describe(a.strategy)} -> "
            f"{'success' if a.success else 'failure'}: {a.detail}"
        )

    def retrieve(self, task: Task) -> tuple[list[Evidence], int, dict]:
        recent = self._log[-self.window:] if self.budget_tokens is None else self._log

        selected: list[tuple[Task, Attempt]] = []
        tokens = 0
        # Walk backwards from the present: recency is the whole policy, so
        # under a budget the newest episodes are the ones kept.
        for t, a in reversed(recent):
            c = self._cost(t, a)
            if self.budget_tokens is not None and tokens + c > self.budget_tokens:
                break
            selected.append((t, a))
            tokens += c
            if self.budget_tokens is None and len(selected) >= self.window:
                break
        selected.reverse()

        evidence = [
            Evidence(context=t.context, strategy=a.strategy, success=a.success)
            for t, a in selected
        ]
        return evidence, tokens, {}

    def record(self, task: Task, attempt: Attempt, meta: dict) -> None:
        self._log.append((task, attempt))


@dataclass(slots=True)
class ReverieBackend:
    """The real system, end to end.

    Writes go through `remember`, outcomes through `report_outcome`, and
    recall through the actual `RecallEngine`. Nothing is short-circuited: the
    evidence handed to the agent is parsed back out of the memory nodes that
    recall actually surfaced.
    """

    scope: str = "agent:bench"
    attribution: bool = True
    # Separates the two halves of attribution so the ablation can isolate them:
    # `attribution` controls whether credit is *recorded*, this controls whether
    # the resulting utility posterior is allowed to *weight* the evidence vote.
    # "attribution on, utility off" is the arm that shows recording credit is
    # not merely cheap but inert.
    use_utility_weight: bool = True
    consolidate_every: int = 10
    budget_tokens: int = 1200   # matches RecallConfig default
    name: str = "reverie"
    config: Config | None = None
    # Human-in-the-loop review. `oracle` is asked at each session boundary.
    oracle: Any = None
    session_len: int = 20
    lesson_weight: float = 3.0
    # Whether the reviewer's answer is folded into the graph synchronously.
    # Defaults to False: nothing a human says may block the agent. Review is
    # out of band and improves the *next* run, not this one.
    #
    # Measured, 600 tasks / 6 seeds / freq 0.5: sync 0.638 +/-0.031 vs async
    # 0.629 +/-0.028. Indistinguishable, so the design principle costs nothing
    # and there is no reason to put consolidation on the critical path.
    sync_consolidate_on_review: bool = False
    final_quarantine_rate: float = 0.0
    lessons_written: int = 0
    _mem: Reverie | None = None
    _n: int = 0
    _session: list = field(default_factory=list)
    _truth_fn: Any = None
    _rng: Any = None

    def reset(self) -> None:
        cfg = self.config or Config()
        if self._mem is not None:
            self._mem.close()
        self._mem = Reverie(scope=self.scope, db=":memory:", config=cfg)
        self._n = 0
        self._session = []
        self.lessons_written = 0
        self._rng = __import__("random").Random(0xBEEF)

    def finish(self) -> None:
        if self._mem is None:
            return
        # Snapshot before closing: the objective uses this as an over-pruning
        # guardrail. Note it is NOT a false-positive rate -- the bench injects
        # no poisoned memories, so there is no ground truth about which
        # memories deserved removal. E4 is where that is measured.
        rows = self._mem.store.conn.execute(
            "SELECT state, COUNT(*) FROM nodes WHERE scope_id=? GROUP BY state",
            (self.scope,),
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        total = sum(counts.values())
        self.final_quarantine_rate = (
            counts.get("quarantined", 0) / total if total else 0.0
        )
        self._mem.close()
        self._mem = None

    def retrieve(self, task: Task) -> tuple[list[Evidence], int, dict]:
        assert self._mem is not None
        result = self._mem.recall(task.context, budget_tokens=self.budget_tokens)

        evidence: list[Evidence] = []
        for scored in result.nodes:
            node = scored.node
            body = node.body or ""
            lesson = _LESSON.match(body)
            if lesson:
                # A lesson states what *should* be done, so it enters as
                # positive evidence carrying more weight than a single event.
                evidence.append(
                    Evidence(context=lesson.group("ctx"),
                             strategy=lesson.group("strategy"),
                             success=True, weight=self.lesson_weight)
                )
                continue
            m = _BODY.match(body)
            if not m:
                continue
            success = "success" in body.lower()
            # Utility-weighted: a memory with a good track record votes
            # harder. With attribution off, alpha/beta stay at the prior and
            # every memory carries equal weight -- which is exactly the
            # contrast the `reverie_noattr` arm is there to isolate.
            weight = 1.0
            if self.attribution and self.use_utility_weight and node.attributions > 0:
                weight = max(0.1, self._mem.attributor.utility(node.id) * 2.0)
            evidence.append(
                Evidence(context=m.group("ctx"), strategy=m.group("strategy"),
                         success=success, weight=weight)
            )

        return evidence, result.token_cost, {"recall_id": result.recall_id}

    def record(self, task: Task, attempt: Attempt, meta: dict) -> None:
        assert self._mem is not None
        mem = self._mem
        recall_id = meta.get("recall_id") or None

        with mem.episode(
            task.describe(attempt.strategy),
            task_type=task.family,
            recall_id=recall_id,
        ) as ep:
            ep.entity(*[f"{k}={v}" for k, v in task.features.items()])
            ep.entity(f"strategy:{attempt.strategy}")
            ep.step("action", attempt.strategy, output=attempt.detail,
                    error=None if attempt.success else attempt.detail)
            ep.outcome("success" if attempt.success else "failure", tier=1,
                       evidence=attempt.detail, detail=attempt.detail)

        self._n += 1
        self._session.append(attempt)
        if self.attribution:
            mem.attribute()
        if self._n % self.consolidate_every == 0:
            mem.consolidate()
        if self.oracle is not None and self._n % self.session_len == 0:
            self._review()

    def _review(self) -> None:
        """Session boundary: ask the human, write what they say."""
        assert self._mem is not None
        feedback = self.oracle.review(self._session, self._truth_fn, self._rng)
        self._session = []
        if not feedback.reviewed:
            return
        for lesson in feedback.lessons:
            self._mem.assert_fact(
                lesson.render(),
                entities=lesson.entities,
                memory_class="procedural",
            )
            self.lessons_written += 1
        # `assert_fact` already writes the lesson node and its entity edges
        # directly, so the lesson is live regardless. This extra cycle only
        # drains buffered *episodes*, and doing it here puts consolidation on
        # the agent's critical path for no stated reason.
        if self.sync_consolidate_on_review:
            self._mem.consolidate()
