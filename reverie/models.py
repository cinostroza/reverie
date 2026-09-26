"""Typed records exchanged across the API boundary."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Literal

__all__ = [
    "SignalTier",
    "Provenance",
    "NodeState",
    "MemoryClass",
    "Step",
    "TaskRef",
    "AgentRef",
    "OutcomeReport",
    "Episode",
    "Node",
    "Edge",
    "RecallResult",
    "ScoredNode",
    "new_id",
    "now_ms",
    "ValidationError",
]

PROVENANCE = ("observed", "inferred", "asserted", "ambiguous")
NODE_STATES = (
    "active",
    "superseded",
    "contested",
    "stale",
    "quarantined",
    "merged",
    "deleted",
)
MEMORY_CLASSES = ("episodic", "semantic", "procedural")
OUTCOMES = ("success", "failure", "partial", "abandoned")

Provenance = Literal["observed", "inferred", "asserted", "ambiguous"]
NodeState = Literal[
    "active", "superseded", "contested", "stale", "quarantined", "merged", "deleted"
]
MemoryClass = Literal["episodic", "semantic", "procedural"]


class ValidationError(ValueError):
    """Raised at ingest. HLD 7: reject, don't coerce."""


class SignalTier(IntEnum):
    """HLD 9.1. Lower is more trustworthy."""

    GROUND_TRUTH = 1
    HUMAN = 2
    BEHAVIORAL = 3
    LLM_JUDGMENT = 4


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str = "") -> str:
    """uuid7-ish: time-ordered so ids sort by creation and index well."""
    ts = now_ms()
    rand = uuid.uuid4().int & ((1 << 74) - 1)
    raw = (ts << 74) | rand
    body = f"{raw:032x}"
    return f"{prefix}_{body}" if prefix else body


@dataclass(slots=True)
class Step:
    kind: str  # tool_call | observation | decision | message
    name: str = ""
    input: str = ""
    output_summary: str = ""
    error: str | None = None
    exit_code: int | None = None
    ts: int = field(default_factory=now_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskRef:
    id: str = ""
    description: str = ""
    type: str = "unknown"
    recall_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRef:
    """Binds a memory to the conditions it was learned under (HLD 6.3).

    A procedural memory learned by one model may not transfer to another, and
    ``env_hash`` is what makes staleness detectable rather than guessable.
    """

    model: str = ""
    version: str = ""
    env_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeReport:
    status: str  # success | failure | partial | abandoned
    signal_tier: int
    evidence: str = ""
    detail: str = ""

    def validate(self) -> None:
        if self.status not in OUTCOMES:
            raise ValidationError(
                f"outcome.status must be one of {OUTCOMES}, got {self.status!r}"
            )
        if self.signal_tier not in (1, 2, 3, 4):
            raise ValidationError(
                f"outcome.signal_tier must be 1-4, got {self.signal_tier!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Episode:
    """HLD 6.3. The most important schema in the system: everything downstream
    is bounded by what the agent can tell us."""

    scope_id: str
    task: TaskRef = field(default_factory=TaskRef)
    agent: AgentRef = field(default_factory=AgentRef)
    steps: list[Step] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    used_memories: list[str] = field(default_factory=list)
    outcome: OutcomeReport | None = None
    duration_ms: int = 0
    ts: int = field(default_factory=now_ms)
    episode_id: str = ""
    client_key: str | None = None
    schema_version: int = 1

    def validate(self) -> None:
        if not self.scope_id or not isinstance(self.scope_id, str):
            raise ValidationError("scope_id is required and must be a non-empty string")
        if ":" not in self.scope_id:
            raise ValidationError(
                f"scope_id must be namespaced like 'agent:name', got {self.scope_id!r}"
            )
        if self.schema_version != 1:
            raise ValidationError(
                f"unsupported episode schema_version {self.schema_version}"
            )
        if not isinstance(self.steps, list):
            raise ValidationError("steps must be a list")
        for i, s in enumerate(self.steps):
            if not isinstance(s, Step):
                raise ValidationError(f"steps[{i}] must be a Step")
            if not s.kind:
                raise ValidationError(f"steps[{i}].kind is required")
        if self.outcome is not None:
            self.outcome.validate()
        if self.duration_ms < 0:
            raise ValidationError("duration_ms must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "client_key": self.client_key,
            "scope_id": self.scope_id,
            "agent": self.agent.to_dict(),
            "task": self.task.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "entities": list(self.entities),
            "used_memories": list(self.used_memories),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "duration_ms": self.duration_ms,
            "ts": self.ts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        outcome = d.get("outcome")
        return cls(
            scope_id=d["scope_id"],
            task=TaskRef(**d.get("task", {})),
            agent=AgentRef(**d.get("agent", {})),
            steps=[Step(**s) for s in d.get("steps", [])],
            entities=list(d.get("entities", [])),
            used_memories=list(d.get("used_memories", [])),
            outcome=OutcomeReport(**outcome) if outcome else None,
            duration_ms=d.get("duration_ms", 0),
            ts=d.get("ts", now_ms()),
            episode_id=d.get("episode_id", ""),
            client_key=d.get("client_key"),
            schema_version=d.get("schema_version", 1),
        )


@dataclass(slots=True)
class Node:
    id: str
    scope_id: str
    type: str
    memory_class: str
    label: str
    body: str = ""
    valence: float = 0.0
    provenance: str = "observed"
    source_ids: list[str] = field(default_factory=list)
    state: str = "active"
    superseded_by: str | None = None
    merged_into: str | None = None
    learned_under: dict[str, Any] = field(default_factory=dict)
    community_id: str | None = None
    centrality: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    attributions: int = 0
    utility_at: int = field(default_factory=now_ms)
    strength: float = 1.0
    activations: int = 0
    last_activated: int | None = None
    created_at: int = field(default_factory=now_ms)
    updated_at: int = field(default_factory=now_ms)

    @property
    def is_recallable(self) -> bool:
        return self.state in ("active", "contested", "stale")


@dataclass(slots=True)
class Edge:
    id: str
    scope_id: str
    src: str
    dst: str
    relation: str
    weight: float = 0.5
    provenance: str = "inferred"
    confidence: float = 0.5
    traversals: int = 0
    last_traversed: int | None = None
    created_at: int = field(default_factory=now_ms)


@dataclass(slots=True)
class ScoredNode:
    node: Node
    activation: float
    score: float
    tokens: int


@dataclass(slots=True)
class RecallResult:
    brief: str
    recall_id: str
    nodes: list[ScoredNode] = field(default_factory=list)
    token_cost: int = 0
    latency_ms: float = 0.0
    visited: int = 0
    hit_visit_cap: bool = False
    is_ablation: bool = False
    withheld_ids: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.nodes)
