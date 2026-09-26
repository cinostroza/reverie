"""Configuration. Defaults mirror HLD Appendix A.

Every value here is a guess pending measurement. The experiment notebooks in
``experiments/`` exist to replace them with numbers; where one has been
measured, the docstring says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

__all__ = [
    "RecallConfig",
    "ConsolidationConfig",
    "AttributionConfig",
    "DecayConfig",
    "IngestConfig",
    "RetentionConfig",
    "Config",
]


@dataclass(slots=True)
class RecallConfig:
    max_hops: int = 3
    fanout: int = 12
    visit_cap: int = 400
    activation_floor: float = 0.05
    hop_decay: tuple[float, ...] = (1.0, 0.6, 0.35)
    intra_community_bonus: float = 1.15
    contradiction_damping: float = -0.5
    budget_tokens: int = 1200

    # `max_nodes` and `budget_tokens` are two knobs on one resource, and a
    # fixed 12 made the token budget inert: at ~25 tokens per node the node cap
    # binds at ~300 tokens, so raising `budget_tokens` from 400 to 1200 changed
    # nothing measurable (E5 sensitivity sweep: capture -0.02 at every value).
    #
    # None means "derive it from the budget", which keeps the two consistent by
    # construction. An explicit int still overrides, because a caller who wants
    # a short brief for readability rather than cost should be able to say so.
    max_nodes: int | None = None
    tokens_per_node_estimate: int = 25
    max_nodes_floor: int = 6
    max_nodes_ceiling: int = 40

    max_contested: int = 2
    seed_k: int = 20
    seed_top: int = 8

    # Conjunctive entity matching (E6). Spreading activation unions the
    # neighbourhoods of the cue's entities; precedent lookup needs their
    # intersection. These score candidates by how much of the cue they match
    # *before* activation is allowed to dilute the signal.
    #
    # `conjunctive_weight` is additive on top of the RRF score, whose per-
    # ranking contributions are ~1/(rrf_k+rank) ~ 0.016. A weight of 0.10 lets
    # a full-coverage match outrank anything the three fuzzy rankings can
    # produce, which is the intent: an exact precedent should win.
    conjunctive: bool = True
    conjunctive_weight: float = 0.10
    conjunctive_exponent: float = 2.0   # >1 rewards matching many cue entities
    conjunctive_k: int = 24
    # Hub penalty: how hard to discount candidates linked to far more
    # than the cue's entities. 0 disables it and lets generic hubs tie
    # with exact precedents.
    conjunctive_specificity: float = 1.0

    # Precision channel (E6b). Exact precedents bypass spreading activation
    # entirely and get a reserved slice of the budget.
    #
    # The traversal is built for association -- finding the load-bearing fact
    # three hops from a textually dissimilar cue (E3). Precedent lookup is the
    # opposite problem and the traversal destroys it: the correct precedent is
    # seeded first at score 1.000 and reaches the brief 4-6% of the time,
    # because activation piles onto hubs until they outrank it.
    direct_match: bool = True
    direct_reserve: float = 0.5        # share of the budget the channel may claim
    # Fraction of the cue's (IDF-weighted) entities a candidate must cover
    # to enter the precision channel at all.
    direct_min_coverage: float = 0.95
    # Alternative admission route: everything the candidate is about is named
    # by the cue. Admits generalising lessons that legitimately omit features.
    direct_min_containment: float = 0.95
    # Exponent on utility LCB when ranking within the precision channel.
    # 0 disables it, which is the ablation arm for attribution lift.
    direct_utility_weight: float = 1.0
    rrf_k: int = 60  # reciprocal-rank-fusion constant
    # Quotas express a *preference* between memory classes, not a reservation.
    # As fixed fractions they silently became a reservation: the no-LLM path
    # (HLD 8.9) produces no procedural or semantic lessons at all, so half the
    # budget sat held for classes with zero candidates while the episodic
    # evidence that did exist was capped at 20%. Measured in E5: 0.2 evidence
    # items delivered per task against replay's 19.5, and capture of ~0.
    #
    # With `adaptive_quotas`, the configured fractions are renormalised over
    # only the classes actually present in the ranked candidate pool, so a
    # missing class redistributes its share instead of wasting it.
    type_quotas: dict[str, float] = field(
        default_factory=lambda: {"procedural": 0.5, "semantic": 0.3, "episodic": 0.2}
    )
    adaptive_quotas: bool = True
    valence_gain: float = 0.4
    collapse_min_members: int = 4
    provenance_factor: dict[str, float] = field(
        default_factory=lambda: {
            "observed": 1.0,
            "asserted": 1.0,
            "inferred": 0.7,
            "ambiguous": 0.4,
        }
    )
    recency_halflife_days: float = 30.0
    context_mismatch_penalty: float = 0.8


@dataclass(slots=True)
class ConsolidationConfig:
    salience_threshold: float = 0.35
    salience_weights: dict[str, float] = field(
        default_factory=lambda: {
            "outcome_extremity": 0.30,
            "novelty": 0.25,
            "cost": 0.15,
            "entity_centrality": 0.20,
            "redundancy": 0.25,  # subtracted
        }
    )
    min_corroboration_for_lesson: int = 2
    dedup_similarity: float = 0.88
    max_valence_delta_per_episode: float = 0.25
    abstract_every_n_cycles: int = 20
    abstract_on_growth_pct: float = 15.0
    community_match_jaccard: float = 0.5
    max_units_per_cycle: int = 500

    # Scope guard (8.7b). Validates an asserted lesson's *scope* claim against
    # episodic evidence, narrowing an over-general rule rather than deleting it.
    # Over-generalisation is the only reviewer error measured to drive the system
    # below the no-review baseline (E10: 0.411 vs 0.456).
    scope_guard: bool = True
    scope_min_evidence: int = 8      # in-scope episodes before judging a lesson
    scope_keep_success: float = 0.6  # success rate at which the scope claim holds
    scope_min_gain: float = 0.05     # information gain needed to justify a split


@dataclass(slots=True)
class AttributionConfig:
    ablation_rate: float = 0.05
    ablation_mode: str = "lowest_rank"  # or "stratified"
    ablation_withhold_n: int = 3
    tier_weights: dict[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 0.8, 3: 0.4, 4: 0.15}
    )
    utility_lcb_quantile: float = 0.25
    min_task_instances_for_tier3: int = 20
    # Credit assignment: how a recall's outcome is divided among its members.
    # "activation" is the HLD 9.2 default; the alternatives exist so the
    # E2 notebook can compare them on identical traces.
    credit_mode: str = "activation"  # activation | uniform | citation | rank


@dataclass(slots=True)
class DecayConfig:
    tau_days: dict[str, float] = field(
        default_factory=lambda: {"episodic": 14.0, "semantic": 180.0, "procedural": 60.0}
    )
    utility_lambda: float = 0.98

    # Quarantine is judged *relative to the scope's own base success rate*.
    #
    # The absolute threshold of 0.25 that this replaced was unusable, and the
    # reason is worth recording: utility_lcb(1, 1, 0.25) is exactly 0.25, so
    # the uninformative Beta(1,1) prior sat precisely on the threshold. Any
    # memory that reached min_attributions with even a trace of blame fell
    # below it, so the threshold did no filtering at all -- good and bad
    # memories were quarantined at identical rates, and the signal-tier
    # weighting had no visible effect. Found by E4; see experiments/README.md.
    quarantine_mode: str = "relative"      # relative (peer MAD) | absolute
    quarantine_mad_k: float = 3.0          # deviations below the peer median
    quarantine_min_peers: int = 5          # need a distribution to compare against
    # Severity cap: the cutoff can never exceed this fraction of the peer
    # median, so a memory must be substantially worse than typical rather
    # than merely below-median. Swept on held-out seeds; 0.6 is the best
    # available balance (5/20 detection, 0.30 false positives per 12) but the
    # detection rate is poor and this rule is NOT considered solved -- see
    # experiments/README.md, "Quarantine reliability is an open problem".
    quarantine_max_fraction: float = 0.6
    quarantine_threshold: float = 0.12       # absolute floor, also used if mode='absolute'

    # Quarantine gates on *evidence mass*, (alpha-1)+(beta-1), not on the
    # number of recalls a memory appeared in. Credit is divided among
    # everything in a recall, so with a dozen memories in play a
    # "participation" is worth ~1/12 of a unit -- and the LCB of a posterior
    # that thin is dominated by the prior, which is pessimistic by
    # construction. Gating on participations therefore quarantined whichever
    # memories happened to collect slightly more blame, independent of
    # whether they were actually bad.
    min_evidence_for_quarantine: float = 3.0
    min_attributions_for_quarantine: int = 8
    strength_floor: float = 0.05
    valence_tau_days_positive: float = 90.0
    valence_tau_days_negative: float = 180.0  # negative affect decays slower


@dataclass(slots=True)
class IngestConfig:
    max_output_summary_bytes: int = 2048
    max_steps_per_episode: int = 200
    buffer_high_water: int = 50_000
    redact: bool = True
    max_body_bytes: int = 4096


@dataclass(slots=True)
class RetentionConfig:
    episode_buffer_grace_days: int = 7
    recall_log_days: int = 90
    outcome_attribution_grace_days: int = 7


@dataclass(slots=True)
class Config:
    recall: RecallConfig = field(default_factory=RecallConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    embedding_model: str = "hash-64"
    embedding_dim: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        kwargs: dict[str, Any] = {}
        section_types = {f.name: f.type for f in fields(cls)}
        for key, value in data.items():
            if key not in section_types:
                continue
            if isinstance(value, dict) and key in {
                "recall",
                "consolidation",
                "attribution",
                "decay",
                "ingest",
                "retention",
            }:
                section_cls = {
                    "recall": RecallConfig,
                    "consolidation": ConsolidationConfig,
                    "attribution": AttributionConfig,
                    "decay": DecayConfig,
                    "ingest": IngestConfig,
                    "retention": RetentionConfig,
                }[key]
                # tier_weights round-trips through JSON with string keys.
                if key == "attribution" and "tier_weights" in value:
                    value = dict(value)
                    value["tier_weights"] = {
                        int(k): v for k, v in value["tier_weights"].items()
                    }
                if key == "recall" and "hop_decay" in value:
                    value = dict(value)
                    value["hop_decay"] = tuple(value["hop_decay"])
                kwargs[key] = section_cls(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
