"""Generative simulation harness for the attribution experiments.

Why simulate at all, when there is a real engine three modules over? Because
the central open question -- *how much outcome evidence does it take before a
Beta posterior separates a useful memory from a useless one* (HLD 19.5) --
needs ground truth. In a live deployment the true causal effect of a memory is
unobservable, so there is nothing to validate the estimate against. Here it is
a parameter.

The world model:

    logit P(success | R) = b0 + sum_{i in R} share_i * effect_i * relevance_i

where ``R`` is the recalled set, ``share_i`` is the memory's activation share,
``effect_i in [-1, 1]`` is its true causal effect, and ``relevance_i`` is a
Bernoulli draw for whether the memory actually mattered on this task. That last
term is the "recalled but irrelevant" problem from HLD 9.4 made explicit and
tunable, and it turns out to dominate the sample complexity.

Requires the ``experiments`` extra (numpy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

__all__ = [
    "WorldConfig",
    "World",
    "SimResult",
    "run_trial",
    "sweep",
    "spearman",
    "rank_metrics",
]


# --------------------------------------------------------------------------
# statistics helpers (kept local so the module has one dependency, not four)
# --------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _beta_lcb(alpha: np.ndarray, beta: np.ndarray, q: float = 0.25) -> np.ndarray:
    """Vectorised Beta quantile by bisection.

    The same estimator as ``reverie._math.utility_lcb``; the test suite
    asserts the two agree, so a sweep result is a statement about the shipped
    engine and not about a lookalike.
    """
    from scipy.stats import beta as _sbeta  # experiments extra

    return _sbeta.ppf(q, alpha, beta)


def rank_metrics(
    estimate: np.ndarray, truth: np.ndarray, *, harmful_threshold: float = -0.2
) -> dict[str, float]:
    """How well does the estimate order the memories?

    ``spearman`` is the headline. ``worst_k_recall`` is the operationally
    important one: quarantine only needs to identify the harmful tail, and a
    system can be mediocre at global ordering while still being good at that.
    """
    harmful = truth <= harmful_threshold
    k = int(harmful.sum())
    out = {
        "spearman": spearman(estimate, truth),
        "n_harmful": float(k),
    }
    if k:
        worst_predicted = set(np.argsort(estimate)[:k].tolist())
        worst_true = set(np.flatnonzero(harmful).tolist())
        out["worst_k_recall"] = len(worst_predicted & worst_true) / k
    else:
        out["worst_k_recall"] = float("nan")
    return out


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

@dataclass(slots=True)
class WorldConfig:
    n_memories: int = 40
    recall_size: int = 6

    # Distribution of true causal effects. A few genuinely harmful memories,
    # a few genuinely good ones, and a large indifferent middle -- which is
    # what a real memory store looks like after distillation.
    frac_harmful: float = 0.15
    frac_helpful: float = 0.25
    effect_scale: float = 1.6

    # P(a recalled memory actually influenced this task). The HLD 9.4 problem.
    relevance_p: float = 0.5

    # Co-occurrence structure. 0.0 = memories are selected independently;
    # 1.0 = memories are locked into fixed groups that always appear together.
    # This is the identifiability knob: perfectly co-occurring memories cannot
    # be told apart by any amount of outcome data.
    cooccurrence: float = 0.3
    n_clusters: int = 6

    base_logit: float = 0.4  # baseline success odds with no memory effect

    # Observation noise: the signal tier mix. Tier 1 is ground truth, tier 4
    # is an LLM's opinion. Weights are the HLD 9.1 values.
    tier_mix: dict[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}
    )
    tier_flip_prob: dict[int, float] = field(
        default_factory=lambda: {1: 0.0, 2: 0.05, 3: 0.20, 4: 0.35}
    )
    tier_weights: dict[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 0.8, 3: 0.4, 4: 0.15}
    )

    # Activation shares. Higher concentration = a few memories dominate the
    # recall, which sharpens credit assignment.
    activation_concentration: float = 1.0

    # Ablation: fraction of recalls that withhold part of the set.
    ablation_rate: float = 0.0
    ablation_mode: Literal["lowest_rank", "stratified", "random"] = "random"


class World:
    """A synthetic agent-and-memory world with known ground truth."""

    def __init__(self, cfg: WorldConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.effects = self._draw_effects()
        self.clusters = self._draw_clusters()

    def _draw_effects(self) -> np.ndarray:
        cfg = self.cfg
        n = cfg.n_memories
        effects = self.rng.normal(0.0, 0.15, size=n)

        n_harm = int(round(cfg.frac_harmful * n))
        n_help = int(round(cfg.frac_helpful * n))
        idx = self.rng.permutation(n)
        effects[idx[:n_harm]] = -np.abs(
            self.rng.normal(cfg.effect_scale * 0.6, cfg.effect_scale * 0.25, n_harm)
        )
        effects[idx[n_harm : n_harm + n_help]] = np.abs(
            self.rng.normal(cfg.effect_scale * 0.6, cfg.effect_scale * 0.25, n_help)
        )
        return np.clip(effects, -3.0, 3.0)

    def _draw_clusters(self) -> np.ndarray:
        return self.rng.integers(0, self.cfg.n_clusters, size=self.cfg.n_memories)

    def draw_recall(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (member indices, activation shares summing to 1)."""
        cfg = self.cfg
        k = min(cfg.recall_size, cfg.n_memories)

        if self.rng.random() < cfg.cooccurrence:
            # Cluster-driven: draw from one theme, so the same memories keep
            # appearing together.
            cluster = self.rng.integers(0, cfg.n_clusters)
            pool = np.flatnonzero(self.clusters == cluster)
            if len(pool) < k:
                extra = self.rng.choice(
                    np.setdiff1d(np.arange(cfg.n_memories), pool),
                    size=k - len(pool), replace=False,
                )
                members = np.concatenate([pool, extra])
            else:
                members = self.rng.choice(pool, size=k, replace=False)
        else:
            members = self.rng.choice(cfg.n_memories, size=k, replace=False)

        raw = self.rng.dirichlet(np.full(k, cfg.activation_concentration))
        order = np.argsort(-raw)
        return members[order], raw[order]

    def outcome(
        self, members: np.ndarray, shares: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Sample a task outcome. Returns (success, relevance mask)."""
        cfg = self.cfg
        relevance = self.rng.random(len(members)) < cfg.relevance_p
        logit = cfg.base_logit + float(
            np.sum(shares * self.effects[members] * relevance)
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        return int(self.rng.random() < p), relevance

    def observe(self, success: int) -> tuple[int, int]:
        """Apply signal-tier noise. Returns (reported_success, tier)."""
        cfg = self.cfg
        tiers = list(cfg.tier_mix)
        probs = np.array([cfg.tier_mix[t] for t in tiers], dtype=float)
        probs = probs / probs.sum()
        tier = int(self.rng.choice(tiers, p=probs))
        if self.rng.random() < cfg.tier_flip_prob.get(tier, 0.0):
            success = 1 - success
        return success, tier


# --------------------------------------------------------------------------
# trials
# --------------------------------------------------------------------------

@dataclass(slots=True)
class SimResult:
    n_recalls: int
    alpha: np.ndarray
    beta: np.ndarray
    attributions: np.ndarray
    effects: np.ndarray
    lcb: np.ndarray
    mean: np.ndarray
    metrics: dict[str, float]
    credit_mode: str
    config: WorldConfig

    def as_row(self, **extra: Any) -> dict[str, Any]:
        return {
            "n_recalls": self.n_recalls,
            "credit_mode": self.credit_mode,
            **self.metrics,
            **extra,
        }


def _shares_for_mode(
    shares: np.ndarray, relevance: np.ndarray, mode: str
) -> np.ndarray:
    """Credit assignment, matching ``reverie.attribution.credit_shares``."""
    k = len(shares)
    if mode == "uniform":
        return np.full(k, 1.0 / k)
    if mode == "rank":
        raw = 1.0 / np.log2(np.arange(k) + 2)
        return raw / raw.sum()
    if mode == "citation":
        # The agent reports which memories it actually used. Perfect
        # information about relevance, which is the upper bound on what any
        # credit rule can achieve.
        if relevance.sum() == 0:
            return np.zeros(k)
        out = np.zeros(k)
        out[relevance] = 1.0 / relevance.sum()
        return out
    return shares / shares.sum() if shares.sum() > 0 else np.full(k, 1.0 / k)


def run_trial(
    cfg: WorldConfig,
    n_recalls: int,
    *,
    seed: int = 0,
    credit_mode: str = "activation",
    lcb_quantile: float = 0.25,
    checkpoints: Sequence[int] | None = None,
) -> SimResult | list[SimResult]:
    """Run one world for ``n_recalls`` and estimate every memory's utility.

    With ``checkpoints``, returns a list of snapshots so a single expensive
    trial yields a whole learning curve instead of one point.
    """
    world = World(cfg, seed=seed)
    n = cfg.n_memories
    alpha = np.ones(n)
    beta = np.ones(n)
    attributions = np.zeros(n, dtype=int)

    marks = sorted(set(checkpoints or [n_recalls]))
    snapshots: list[SimResult] = []
    next_mark = 0

    for step in range(1, n_recalls + 1):
        members, shares = world.draw_recall()

        # Ablation withholds part of the recalled set. This is the only
        # mechanism that breaks co-occurrence deadlock: without it, memories
        # that always appear together are mathematically indistinguishable.
        if cfg.ablation_rate > 0 and world.rng.random() < cfg.ablation_rate and len(members) > 1:
            n_keep = max(1, len(members) - 3)
            if cfg.ablation_mode == "lowest_rank":
                keep = np.arange(n_keep)
            elif cfg.ablation_mode == "stratified":
                keep = (
                    np.arange(len(members) - n_keep, len(members))
                    if world.rng.random() < 0.5
                    else np.arange(n_keep)
                )
            else:
                keep = world.rng.choice(len(members), size=n_keep, replace=False)
            members, shares = members[keep], shares[keep]
            shares = shares / shares.sum()

        success, relevance = world.outcome(members, shares)
        reported, tier = world.observe(success)

        credit = _shares_for_mode(shares, relevance, credit_mode)
        weight = cfg.tier_weights.get(tier, 0.1) * credit

        if reported:
            alpha[members] += weight
        else:
            beta[members] += weight
        attributions[members] += 1

        if next_mark < len(marks) and step == marks[next_mark]:
            lcb = _beta_lcb(alpha, beta, lcb_quantile)
            snapshots.append(
                SimResult(
                    n_recalls=step,
                    alpha=alpha.copy(), beta=beta.copy(),
                    attributions=attributions.copy(),
                    effects=world.effects.copy(),
                    lcb=lcb, mean=alpha / (alpha + beta),
                    metrics=rank_metrics(lcb, world.effects),
                    credit_mode=credit_mode, config=cfg,
                )
            )
            next_mark += 1

    return snapshots if checkpoints else snapshots[-1]


def sweep(
    base: WorldConfig,
    axis: str,
    values: Sequence[Any],
    *,
    checkpoints: Sequence[int],
    n_seeds: int = 20,
    credit_mode: str = "activation",
) -> list[dict[str, Any]]:
    """Vary one parameter, average over seeds, return tidy rows for pandas."""
    from dataclasses import replace

    rows: list[dict[str, Any]] = []
    for value in values:
        cfg = replace(base, **{axis: value})
        for seed in range(n_seeds):
            for snap in run_trial(
                cfg, max(checkpoints), seed=seed,
                credit_mode=credit_mode, checkpoints=checkpoints,
            ):
                rows.append(snap.as_row(**{axis: value, "seed": seed}))
    return rows
