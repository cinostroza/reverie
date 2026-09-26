"""A scalar objective over Reverie's configuration.

## Why not just maximise success rate

Three reasons, each of which would make a tuned number meaningless.

**Task difficulty confounds it.** `db_migration` and `api_integration` have
different chance rates and different achievable ceilings. A config that scores
0.55 on one and 0.55 on the other is not performing equally well on both, and
averaging raw rates silently weights the easier family higher.

**It ignores cost.** `replay` reaches 0.738 by pasting 506 tokens into every
prompt. Reverie's entire argument is that it gets comparable results for
materially fewer tokens. An objective blind to tokens would "discover" that the
best configuration is an enormous budget -- which is true, useless, and exactly
the opposite of the design goal.

**It has no floor.** Success rate at chance (0.25 with four strategies) means
the memory contributed nothing, but 0.25 is not zero and an optimiser will
happily climb noise around it.

## What is measured instead

Everything is expressed relative to the two arms that bracket the problem, both
of which are independent of Reverie's config and therefore computed once and
cached:

    floor    = accuracy of the `none` arm      (no memory at all)
    ceiling  = accuracy of the `replay` arm    (naive history-stuffing)

    capture  = (accuracy - floor) / (ceiling - floor)

`capture` is scale-free and reads directly: 0.0 means memory contributed
nothing over having none, 1.0 means it matched naive history-stuffing, and >1.0
means it beat the thing practitioners actually do. It is comparable across task
families, which raw accuracy is not.

The efficiency claim then becomes a *constraint* rather than a second thing to
average in:

    maximise   capture
    subject to token_ratio <= max_token_ratio     (default 0.5)
               quarantine_rate <= max_quarantine_rate

Constraints are folded into the scalar as one-sided penalties, so the optimiser
sees a smooth surface but cannot buy accuracy with unlimited tokens.

## Read the warnings in `Score` before believing a number

`Score.noise_se` is a **cluster-robust** standard error, computed across runs
rather than across tasks, because outcomes within a run are strongly
correlated -- a run either discovers the rule or it does not. Compare capture
values against `Score.capture_se`, not `noise_se`; the former is in the right
units.

Differences smaller than roughly `2 * capture_se` are not real, and with few
seeds that band is wide. A 4-seed run has a band of roughly +/-0.35 capture,
which is most of the interesting range -- use 16 seeds or more before
believing any comparison.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..config import Config
from .agents import EvidenceAgent
from .backends import NoMemoryBackend, ReplayBackend, ReverieBackend
from .core import run_arm
from .families import ALL_FAMILIES

__all__ = ["Score", "Baselines", "baselines_for", "evaluate", "make_objective",
           "SEARCH_SPACE", "config_from_params"]


@dataclass(slots=True)
class Baselines:
    floor: float
    ceiling: float
    ceiling_tokens: float

    @property
    def headroom(self) -> float:
        return self.ceiling - self.floor


_BASELINE_CACHE: dict[tuple, Baselines] = {}


def baselines_for(
    families: Sequence[str],
    seeds: Sequence[int],
    n_tasks: int,
    replay_window: int = 40,
) -> Baselines:
    """`none` and `replay` do not depend on Reverie's config, so cache them."""
    key = (tuple(families), tuple(seeds), n_tasks, replay_window)
    if key in _BASELINE_CACHE:
        return _BASELINE_CACHE[key]

    floor_hits: list[int] = []
    ceil_hits: list[int] = []
    ceil_tokens: list[int] = []

    for fname in families:
        F = ALL_FAMILIES[fname]
        for seed in seeds:
            r = run_arm(F(), EvidenceAgent(), NoMemoryBackend(),
                        n_tasks=n_tasks, seed=seed, arm="none")
            floor_hits += [x["success"] for x in r.records]

            r = run_arm(F(), EvidenceAgent(), ReplayBackend(window=replay_window),
                        n_tasks=n_tasks, seed=seed, arm="replay")
            ceil_hits += [x["success"] for x in r.records]
            ceil_tokens += [x["tokens"] for x in r.records]

    b = Baselines(
        floor=statistics.mean(floor_hits),
        ceiling=statistics.mean(ceil_hits),
        ceiling_tokens=statistics.mean(ceil_tokens),
    )
    _BASELINE_CACHE[key] = b
    return b


@dataclass(slots=True)
class Score:
    capture: float
    accuracy: float
    floor: float
    ceiling: float
    tokens: float
    token_ratio: float
    quarantine_rate: float
    evidence_items: float
    latency_ms: float
    noise_se: float
    n_runs: int = 0
    penalties: dict[str, float] = field(default_factory=dict)
    objective: float = 0.0
    n_trials: int = 0

    @property
    def capture_se(self) -> float:
        """`noise_se` converted into capture units.

        `noise_se` is the standard error of *accuracy*. Comparing two capture
        values against it directly understates the band by a factor of
        1/headroom -- roughly 2.5x with the default families, since the
        headroom between the `none` and `replay` arms is only ~0.40 accuracy.
        Always compare capture differences against this, not `noise_se`.
        """
        headroom = self.ceiling - self.floor
        return self.noise_se / headroom if abs(headroom) > 1e-9 else float("inf")

    def explain(self) -> str:
        lines = [
            f"objective        {self.objective:+.4f}",
            f"  capture        {self.capture:+.4f}   "
            f"(accuracy {self.accuracy:.3f} between floor {self.floor:.3f} "
            f"and ceiling {self.ceiling:.3f})",
            f"  token ratio    {self.token_ratio:.3f}   "
            f"({self.tokens:.0f} vs replay's {self.tokens / max(self.token_ratio, 1e-9):.0f})",
            f"  quarantined    {self.quarantine_rate:.3f}",
            f"  evidence/task  {self.evidence_items:.1f}",
            f"  recall latency {self.latency_ms:.1f} ms",
        ]
        for name, value in self.penalties.items():
            if value:
                lines.append(f"  penalty:{name:<7} {-value:+.4f}")
        lines.append(
            f"  noise SE       {self.noise_se:.4f} accuracy "
            f"= {self.capture_se:.4f} capture  "
            f"(differences below {2 * self.capture_se:.3f} capture are not real; "
            f"{self.n_runs} runs / {self.n_trials} tasks)"
        )
        return "\n".join(lines)


def evaluate(
    config: Config | None = None,
    *,
    families: Sequence[str] = ("db_migration", "api_integration"),
    seeds: Sequence[int] = tuple(range(8)),
    n_tasks: int = 40,
    budget_tokens: int = 1200,
    attribution: bool = True,
    consolidate_every: int = 10,
    max_token_ratio: float = 0.5,
    max_quarantine_rate: float = 0.10,
    token_penalty: float = 1.0,
    quarantine_penalty: float = 2.0,
    replay_window: int = 40,
) -> Score:
    """Run every family/seed under one config and score it."""
    base = baselines_for(families, seeds, n_tasks, replay_window)

    hits: list[int] = []
    tokens: list[int] = []
    evidence: list[int] = []
    quarantined: list[float] = []
    run_means: list[float] = []
    t0 = time.perf_counter()
    n_recalls = 0

    for fname in families:
        F = ALL_FAMILIES[fname]
        for seed in seeds:
            backend = ReverieBackend(
                attribution=attribution,
                consolidate_every=consolidate_every,
                budget_tokens=budget_tokens,
                config=config or Config(),
            )
            r = run_arm(F(), EvidenceAgent(), backend,
                        n_tasks=n_tasks, seed=seed, arm="reverie")
            run_hits = [x["success"] for x in r.records]
            hits += run_hits
            run_means.append(statistics.mean(run_hits))
            tokens += [x["tokens"] for x in r.records]
            evidence += [x["evidence_items"] for x in r.records]
            n_recalls += len(r.records)
            quarantined.append(getattr(backend, "final_quarantine_rate", 0.0))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    accuracy = statistics.mean(hits)
    mean_tokens = statistics.mean(tokens)
    token_ratio = mean_tokens / max(base.ceiling_tokens, 1e-9)
    q_rate = statistics.mean(quarantined) if quarantined else 0.0

    headroom = base.headroom
    capture = (accuracy - base.floor) / headroom if abs(headroom) > 1e-9 else 0.0

    penalties = {
        "tokens": token_penalty * max(0.0, token_ratio - max_token_ratio),
        "quarant": quarantine_penalty * max(0.0, q_rate - max_quarantine_rate),
    }

    # Cluster-robust standard error, computed across *runs* rather than
    # across tasks.
    #
    # The binomial form sqrt(p(1-p)/n_tasks) assumes independent Bernoulli
    # trials and is badly wrong here: outcomes within a run are strongly
    # correlated, because a run either discovers the feature->strategy rule or
    # it does not, and everything after that point follows. Measured per-run
    # accuracies on one family span 0.10 to 0.65. The effective sample size is
    # the number of runs, not the number of tasks, and the binomial estimate
    # understates the true error by ~1.8x.
    #
    # This is not academic. A 4-seed comparison of two families reported
    # captures of +0.70 and +0.27 against a binomial band of +/-0.12, which
    # read as a large, real generalisation gap. At 16 seeds the two families
    # sit at +0.51 and +0.58 -- there was no gap, and the wrong band is what
    # made noise look like a finding.
    if len(run_means) > 1:
        se = statistics.stdev(run_means) / math.sqrt(len(run_means))
    else:
        se = math.sqrt(max(accuracy * (1 - accuracy), 1e-9) / max(len(hits), 1))

    return Score(
        capture=capture,
        accuracy=accuracy,
        floor=base.floor,
        ceiling=base.ceiling,
        tokens=mean_tokens,
        token_ratio=token_ratio,
        quarantine_rate=q_rate,
        evidence_items=statistics.mean(evidence),
        latency_ms=elapsed_ms / max(n_recalls, 1),
        noise_se=se,
        n_runs=len(run_means),
        penalties=penalties,
        objective=capture - sum(penalties.values()),
        n_trials=len(hits),
    )


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------
#
# Deliberately not "every field in Config". With ~40 tunables and an evaluation
# that costs minutes, a full-dimensional search spends its whole budget on
# noise. These are the parameters with a plausible mechanism for affecting
# task performance; run the one-at-a-time sensitivity sweep in E5 first and
# narrow further before letting an optimiser loose.

SEARCH_SPACE: dict[str, tuple[str, Any]] = {
    "recall.budget_tokens":            ("int", (200, 1200)),
    "recall.max_nodes":                ("int", (6, 32)),
    "recall.max_hops":                 ("int", (1, 3)),
    "recall.fanout":                   ("int", (4, 24)),
    "recall.activation_floor":         ("float", (0.005, 0.20)),
    "recall.quota_episodic":           ("float", (0.1, 0.9)),
    "recall.recency_halflife_days":    ("float", (1.0, 180.0)),
    "consolidation.salience_threshold": ("float", (0.0, 0.8)),
    "consolidation.dedup_similarity":  ("float", (0.80, 0.99)),
    "attribution.utility_lcb_quantile": ("float", (0.05, 0.5)),
    "decay.utility_lambda":            ("float", (0.90, 1.0)),
    "decay.tau_episodic":              ("float", (3.0, 120.0)),
}


def config_from_params(params: dict[str, Any]) -> Config:
    """Materialise a `Config` from a flat parameter dict.

    Quota handling is the one non-obvious bit: `quota_episodic` sets the
    episodic share and the remainder is split between procedural and semantic
    in the default 5:3 ratio. Optimising three quotas independently would let
    the optimiser pick values that do not sum to 1, which the selector then
    renormalises -- making two of the three dimensions redundant and wasting
    search budget on a degenerate parameterisation.
    """
    cfg = Config()

    for key, value in params.items():
        section, _, name = key.partition(".")

        if name == "quota_episodic":
            ep = float(value)
            rest = max(0.0, 1.0 - ep)
            cfg.recall.type_quotas = {
                "episodic": ep,
                "procedural": rest * 0.625,
                "semantic": rest * 0.375,
            }
            continue

        if name == "tau_episodic":
            cfg.decay.tau_days = dict(cfg.decay.tau_days)
            cfg.decay.tau_days["episodic"] = float(value)
            continue

        target = getattr(cfg, section, None)
        if target is not None and hasattr(target, name):
            setattr(target, name, value)

    return cfg


def make_objective(
    **eval_kwargs: Any,
) -> Callable[[dict[str, Any]], float]:
    """Return `params -> objective`, ready for Optuna/scipy/CMA-ES.

    Usage with Optuna::

        import optuna
        from reverie.bench.objective import SEARCH_SPACE, make_objective

        fn = make_objective(seeds=range(12))

        def trial_fn(trial):
            params = {}
            for key, (kind, (lo, hi)) in SEARCH_SPACE.items():
                params[key] = (trial.suggest_int(key, lo, hi) if kind == "int"
                               else trial.suggest_float(key, lo, hi))
            return fn(params)

        optuna.create_study(direction="maximize").optimize(trial_fn, n_trials=60)
    """

    def objective(params: dict[str, Any]) -> float:
        cfg = config_from_params(params)
        budget = params.get("recall.budget_tokens", 400)
        return evaluate(cfg, budget_tokens=int(budget), **eval_kwargs).objective

    return objective
