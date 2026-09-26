"""Task families (HLD 16.2).

Each family has a deterministic environment, a programmatic success check, and
hidden rules that only generalise as a *feature → strategy mapping*. Features
are resampled every episode, so memorising one instance is worth nothing and
the agent has to learn the rule.

Every family also carries irreducible noise (`flake_rate`). Without it the
learning curve saturates at exactly 1.0, which looks impressive and tells you
nothing about whether the estimator is calibrated -- a memory system should be
able to tolerate a task that occasionally fails for reasons nobody can learn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .core import Attempt, Task, stable_hash

__all__ = ["MigrationFamily", "ApiIntegrationFamily", "PipelineFamily",
           "IncidentFamily", "ALL_FAMILIES"]


@dataclass(slots=True)
class MigrationFamily:
    """Database migrations with environment-specific gotchas.

    Hidden rules, in priority order:
      1. business hours + DDL     -> only `off_hours` succeeds (staging blocks DDL)
      2. large table              -> only `sql_first` succeeds (direct times out)
      3. otherwise                -> `direct` is fine and fastest
    """

    name: str = "db_migration"
    strategies: tuple[str, ...] = ("direct", "sql_first", "lock_timeout", "off_hours")
    flake_rate: float = 0.05

    def sample(self, rng: random.Random, index: int) -> Task:
        return Task(
            family=self.name,
            index=index,
            features={
                "size": rng.choice(["small", "large"]),
                "hours": rng.choice(["business", "off"]),
                "indexed": rng.choice(["yes", "no"]),  # a decoy feature
            },
            strategies=self.strategies,
        )

    def _correct(self, task: Task) -> str:
        if task.features["hours"] == "business":
            return "off_hours"
        if task.features["size"] == "large":
            return "sql_first"
        return "direct"

    def evaluate(self, task: Task, strategy: str) -> Attempt:
        want = self._correct(task)
        ok = strategy == want
        # Deterministic per (task, strategy) so re-running is reproducible and
        # an agent cannot farm retries for a luckier draw.
        rng = random.Random(stable_hash((task.family, task.index, task.context, strategy)) & 0xFFFFFFFF)
        if ok and rng.random() < self.flake_rate:
            ok = False
            detail = "flaky test suite"
        elif ok:
            detail = "migration applied"
        else:
            detail = {
                "off_hours": "DDL blocked during business hours",
                "sql_first": "statement timeout on large table",
                "direct": "unnecessary overhead but succeeded",
            }.get(want, "wrong strategy")
        return Attempt(task=task, strategy=strategy, success=ok, detail=detail,
                       steps=1 if ok else 2)


@dataclass(slots=True)
class ApiIntegrationFamily:
    """A mock third-party API with undocumented quirks."""

    name: str = "api_integration"
    strategies: tuple[str, ...] = ("plain", "paginate", "chunk", "with_header")
    flake_rate: float = 0.05

    def sample(self, rng: random.Random, index: int) -> Task:
        return Task(
            family=self.name,
            index=index,
            features={
                "endpoint": rng.choice(["orders", "users", "events"]),
                "payload": rng.choice(["small", "big"]),
            },
            strategies=self.strategies,
        )

    def _correct(self, task: Task) -> str:
        if task.features["endpoint"] == "events":
            return "paginate"          # undocumented: events is always paged
        if task.features["payload"] == "big":
            return "chunk"             # undocumented: 1MB body limit
        return "with_header"           # undocumented: requires X-Api-Version

    def evaluate(self, task: Task, strategy: str) -> Attempt:
        want = self._correct(task)
        ok = strategy == want
        rng = random.Random(stable_hash((task.family, task.index, task.context, strategy)) & 0xFFFFFFFF)
        if ok and rng.random() < self.flake_rate:
            ok, detail = False, "upstream 503"
        elif ok:
            detail = "200 OK"
        else:
            detail = {"paginate": "truncated at 100 records", "chunk": "413 payload too large",
                      "with_header": "400 missing X-Api-Version"}.get(want, "request failed")
        return Attempt(task=task, strategy=strategy, success=ok, detail=detail,
                       steps=1 if ok else 2)


@dataclass(slots=True)
class PipelineFamily:
    """A data pipeline with schema drift and timezone traps."""

    name: str = "data_pipeline"
    strategies: tuple[str, ...] = ("as_is", "normalize_tz", "map_schema", "backfill")
    flake_rate: float = 0.08

    def sample(self, rng: random.Random, index: int) -> Task:
        return Task(
            family=self.name,
            index=index,
            features={
                "source": rng.choice(["eu_west", "us_east", "internal"]),
                "schema": rng.choice(["v1", "v2"]),
            },
            strategies=self.strategies,
        )

    def _correct(self, task: Task) -> str:
        if task.features["schema"] == "v2":
            return "map_schema"
        if task.features["source"] == "eu_west":
            return "normalize_tz"
        return "as_is"

    def evaluate(self, task: Task, strategy: str) -> Attempt:
        want = self._correct(task)
        ok = strategy == want
        rng = random.Random(stable_hash((task.family, task.index, task.context, strategy)) & 0xFFFFFFFF)
        if ok and rng.random() < self.flake_rate:
            ok, detail = False, "transient source unavailability"
        elif ok:
            detail = "rows loaded"
        else:
            detail = {"map_schema": "unknown column in v2 payload",
                      "normalize_tz": "timestamps off by 2h",
                      "as_is": "over-processed"}.get(want, "load failed")
        return Attempt(task=task, strategy=strategy, success=ok, detail=detail,
                       steps=1 if ok else 3)


ALL_FAMILIES = {
    "db_migration": MigrationFamily,
    "api_integration": ApiIntegrationFamily,
    "data_pipeline": PipelineFamily,
}


@dataclass(slots=True)
class IncidentFamily:
    """On-call incident response over a large service landscape.

    The families above all have tiny context spaces -- `MigrationFamily` has
    8 distinct feature combinations. With a 31-episode recency window an agent
    holds several examples of *every* context, so recency is guaranteed to
    contain the relevant precedent and selective retrieval cannot beat it.
    That makes those families structurally unable to test Reverie's actual
    claim, and it is why replay dominates on them at every token budget.

    This one is built for the regime the design targets: thousands of
    invocations across a landscape far larger than any context window. With 12
    services x 6 symptoms there are 72 distinct incident types, so a recency
    window of ~30 episodes usually holds *no* matching precedent while a graph
    of 800 episodes does. Retrieval either finds it or it does not -- which is
    the thing worth measuring.

    Two kinds of hidden structure, because a pure lookup table would be
    unfair in the other direction:

    * **Global symptom rules.** Some symptoms have the same remediation on
      every service. Generalisable; similarity-weighted voting can learn them
      from any service's history.
    * **Service-specific rules.** The rest depend on the exact
      (service, symptom) pair. Only an actual precedent for that pair helps,
      which is what recency reliably fails to supply at scale.
    """

    name: str = "oncall_incident"
    strategies: tuple[str, ...] = (
        "restart", "rollback", "scale_out", "clear_cache", "failover", "page_owner",
    )
    n_services: int = 12
    n_symptoms: int = 6
    global_symptom_frac: float = 0.34   # share of symptoms with a service-independent fix
    flake_rate: float = 0.05
    landscape_seed: int = 20260812

    # Environment drift. After `drift_at` tasks, `drift_frac` of the landscape
    # changes its correct remediation -- a config change, a version bump, a
    # failover that makes the old fix wrong.
    #
    # This is what makes attribution testable. Without drift every precedent is
    # simply true, so a system that ranks by track record cannot beat one that
    # ranks by recency, and "attribution lift" has nothing to measure. With
    # drift, stale precedents are actively harmful and the utility posterior is
    # the only mechanism that can notice.
    drift_at: int | None = None
    drift_frac: float = 0.5

    def _services(self) -> list[str]:
        return [f"svc-{i:02d}" for i in range(self.n_services)]

    def _symptoms(self) -> list[str]:
        return [f"sym-{i}" for i in range(self.n_symptoms)]

    def sample(self, rng: random.Random, index: int) -> Task:
        return Task(
            family=self.name,
            index=index,
            features={
                "service": rng.choice(self._services()),
                "symptom": rng.choice(self._symptoms()),
                "severity": rng.choice(["sev1", "sev2", "sev3"]),  # decoy
            },
            strategies=self.strategies,
        )

    def _correct(self, task: Task) -> str:
        symptom = task.features["symptom"]
        service = task.features["service"]

        # The landscape is fixed for the life of the family, not resampled per
        # task, so it is genuinely learnable rather than noise.
        n_global = max(1, round(self.n_symptoms * self.global_symptom_frac))
        global_syms = set(self._symptoms()[:n_global])

        key = (self.landscape_seed, symptom) if symptom in global_syms else (
            self.landscape_seed, service, symptom
        )

        era = self.landscape_seed
        if self.drift_at is not None and task.index >= self.drift_at:
            # Deterministic per-pair: the same half of the landscape drifts on
            # every seed, so the difficulty is comparable across runs.
            drifted = (stable_hash((self.landscape_seed, "drift", service, symptom))
                       % 1000) / 1000.0 < self.drift_frac
            if drifted:
                era = self.landscape_seed + 1

        return self.strategies[stable_hash((era, *key[1:])) % len(self.strategies)]

    def evaluate(self, task: Task, strategy: str) -> Attempt:
        want = self._correct(task)
        ok = strategy == want
        rng = random.Random(stable_hash((task.family, task.index, task.context, strategy)) & 0xFFFFFFFF)
        if ok and rng.random() < self.flake_rate:
            ok, detail = False, "remediation applied but incident recurred"
        elif ok:
            detail = "incident mitigated"
        else:
            detail = f"{strategy} did not mitigate; escalated"
        return Attempt(task=task, strategy=strategy, success=ok, detail=detail,
                       steps=1 if ok else 3)


ALL_FAMILIES["oncall_incident"] = IncidentFamily
