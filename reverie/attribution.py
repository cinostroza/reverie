"""Attribution: how memory learns (HLD 9).

Every recall is logged with the exact set of memories it surfaced. Every
outcome is attributed back onto that set, updating a Beta posterior over
"did participating in a recall correlate with success". Ranking then uses the
lower confidence bound of that posterior.

Three properties this module is responsible for getting right, all of which
are easy to get subtly wrong:

1. A merged or superseded node still receives its attribution (HLD 9.5).
   Dropping those updates biases the whole system toward whichever memories
   happened not to get merged.
2. A later higher-tier outcome *reverses* an earlier lower-tier one rather
   than stacking on top of it (HLD 9.3). Otherwise a wrong behavioural guess
   is permanently baked into the posterior.
3. Credit is divided by activation share, so a barely-activated memory earns
   barely any credit for an outcome it barely influenced (HLD 9.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from ._math import beta_mean, utility_lcb
from .config import AttributionConfig, DecayConfig
from .models import now_ms
from .store import SQLiteStore

__all__ = ["AttributionEngine", "AttributionStats", "credit_shares"]

_DAY_MS = 86_400_000


def _median(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


@dataclass(slots=True)
class AttributionStats:
    outcomes_processed: int = 0
    nodes_updated: int = 0
    reversals: int = 0
    dropped_deleted: int = 0
    dropped_no_recall: int = 0
    quarantined: list[str] = field(default_factory=list)


def credit_shares(
    activations: dict[str, float],
    *,
    mode: str = "activation",
    cited: Iterable[str] | None = None,
    ranked_ids: list[str] | None = None,
) -> dict[str, float]:
    """Divide one unit of credit among the memories in a recall.

    The estimator comparison in ``experiments/E2`` swaps between these on
    identical traces, which is the only way to tell whether activation
    weighting actually beats the uniform baseline.

    - ``activation``: proportional to activation share (HLD 9.2 default).
    - ``uniform``: every recalled memory gets 1/n. The null hypothesis.
    - ``citation``: all credit to memories the agent said it used. Near-exact
      when available, but requires agent cooperation.
    - ``rank``: 1/log2(rank+2) weighting, ignoring raw activation magnitude.
    """
    if not activations:
        return {}

    if mode == "uniform":
        share = 1.0 / len(activations)
        return dict.fromkeys(activations, share)

    if mode == "citation":
        cited_set = {c for c in (cited or ()) if c in activations}
        if not cited_set:
            # No citation supplied: fall back rather than throwing the outcome
            # away, since a missing field is the common case, not an error.
            return credit_shares(activations, mode="activation")
        share = 1.0 / len(cited_set)
        return {nid: (share if nid in cited_set else 0.0) for nid in activations}

    if mode == "rank":
        order = ranked_ids or sorted(
            activations, key=lambda k: activations[k], reverse=True
        )
        raw = {nid: 1.0 / math.log2(i + 2) for i, nid in enumerate(order)}
        total = sum(raw.values()) or 1.0
        return {nid: v / total for nid, v in raw.items()}

    # activation (default)
    total = sum(max(0.0, v) for v in activations.values())
    if total <= 0.0:
        share = 1.0 / len(activations)
        return dict.fromkeys(activations, share)
    return {nid: max(0.0, v) / total for nid, v in activations.items()}


class AttributionEngine:
    def __init__(
        self,
        store: SQLiteStore,
        config: AttributionConfig | None = None,
        decay: DecayConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or AttributionConfig()
        self.decay = decay or DecayConfig()

    # -- core update -------------------------------------------------------

    def _deltas_for(
        self, outcome: str, weight: float
    ) -> tuple[float, float]:
        """Map an outcome and its weight onto (d_alpha, d_beta)."""
        if outcome == "success":
            return weight, 0.0
        if outcome == "failure":
            return 0.0, weight
        if outcome == "partial":
            return weight / 2.0, weight / 2.0
        return 0.0, 0.0  # abandoned: we learn nothing from an unfinished task

    def _reverse_prior_attributions(
        self, recall_id: str, current_tier: int, current_outcome: str
    ) -> int:
        """HLD 9.3.

        When a higher-tier outcome contradicts an already-applied lower-tier
        one for the same recall, undo the earlier update first. Agreement
        needs no reversal -- two independent signals pointing the same way is
        genuine corroboration and should compound.
        """
        reversals = 0
        for prior in self.store.outcomes_for_recall(recall_id):
            if not prior["attributed"]:
                continue
            if prior["signal_tier"] <= current_tier:
                continue  # equal or better evidence; leave it standing
            if prior["outcome"] == current_outcome:
                continue  # agrees

            for entry in self.store.attributions_for_outcome(prior["id"]):
                self.store.apply_utility_delta(
                    entry["node_id"], -entry["d_alpha"], -entry["d_beta"], d_attr=-1
                )
                self.store.mark_reversed(entry["id"])
                reversals += 1
        return reversals

    def attribute_one(self, outcome_row: dict[str, Any]) -> tuple[int, int]:
        """Apply a single outcome. Returns (nodes_updated, reversals)."""
        recall = self.store.get_recall(outcome_row["recall_id"])
        if recall is None:
            self.store.mark_attributed(outcome_row["id"])
            return 0, 0

        tier = int(outcome_row["signal_tier"])
        result = outcome_row["outcome"]
        tier_weight = self.config.tier_weights.get(tier, 0.1)

        reversals = self._reverse_prior_attributions(recall["id"], tier, result)

        activations: dict[str, float] = recall["activations"]
        shares = credit_shares(
            activations,
            mode=self.config.credit_mode,
            ranked_ids=recall["node_ids"],
        )

        updated = 0
        for node_id, share in shares.items():
            if share <= 0.0:
                continue
            # Follow merges; a node absorbed by consolidation still earns its
            # record, credited to whatever survived (HLD 9.5).
            target = self.store.resolve_merges(node_id)
            if target is None:
                continue  # deleted: the update is dropped on the floor

            weight = tier_weight * share
            d_alpha, d_beta = self._deltas_for(result, weight)
            if d_alpha == 0.0 and d_beta == 0.0:
                continue

            self.store.apply_utility_delta(target, d_alpha, d_beta)
            self.store.record_attribution(outcome_row["id"], target, d_alpha, d_beta)
            updated += 1

        self.store.mark_attributed(outcome_row["id"])
        return updated, reversals

    def run(self, limit: int = 1000) -> AttributionStats:
        """Drain the pending-outcome queue.

        Cheap arithmetic, no model call -- this should run far more often than
        consolidation so outcomes move rankings within minutes (HLD 5.3).
        """
        stats = AttributionStats()
        for row in self.store.pending_outcomes(limit):
            updated, reversals = self.attribute_one(row)
            stats.outcomes_processed += 1
            stats.nodes_updated += updated
            stats.reversals += reversals
        stats.quarantined = self.quarantine_sweep()
        return stats

    # -- utility surface ---------------------------------------------------

    def utility(self, node_id: str) -> float:
        node = self.store.get_node(node_id)
        if node is None:
            return 0.0
        return utility_lcb(node.alpha, node.beta, self.config.utility_lcb_quantile)

    def utility_report(self, node_id: str) -> dict[str, Any]:
        node = self.store.get_node(node_id)
        if node is None:
            return {}
        return {
            "node_id": node_id,
            "alpha": node.alpha,
            "beta": node.beta,
            "attributions": node.attributions,
            "mean": beta_mean(node.alpha, node.beta),
            "lcb": utility_lcb(node.alpha, node.beta, self.config.utility_lcb_quantile),
            "state": node.state,
        }

    # -- decay and quarantine ---------------------------------------------

    def decay_utility(self, scope_id: str, now: int | None = None) -> int:
        """``alpha <- 1 + (alpha-1)*lambda``. Old evidence stops dominating."""
        now = now or now_ms()
        lam = self.decay.utility_lambda
        touched = 0
        for node in self.store.iter_nodes(
            [scope_id], states=("active", "contested", "stale", "quarantined")
        ):
            new_alpha = 1.0 + (node.alpha - 1.0) * lam
            new_beta = 1.0 + (node.beta - 1.0) * lam
            self.store.update_node(
                node.id, alpha=new_alpha, beta=new_beta, utility_at=now
            )
            touched += 1
        return touched

    def quarantine_sweep(self, scope_id: str | None = None) -> list[str]:
        """Demote memories whose track record has gone bad (HLD 8.8, G8).

        Requires a minimum number of attributions first: quarantining on the
        strength of two unlucky recalls would be worse than not quarantining
        at all.
        """
        quarantined: list[str] = []
        scopes = [scope_id] if scope_id else self._all_scopes()
        for scope in scopes:
            cutoff = self.quarantine_cutoff(scope)
            for node in self.store.iter_nodes([scope], states=("active", "contested")):
                if not self.is_quarantinable(node):
                    continue
                if node.attributions < self.decay.min_attributions_for_quarantine:
                    continue
                evidence = (node.alpha - 1.0) + (node.beta - 1.0)
                if evidence < self.decay.min_evidence_for_quarantine:
                    continue
                lcb = utility_lcb(node.alpha, node.beta, self.config.utility_lcb_quantile)
                if lcb < cutoff:
                    self.store.update_node(node.id, state="quarantined")
                    quarantined.append(node.id)
        return quarantined

    def quarantine_cutoff(self, scope_id: str) -> float:
        """The LCB below which a memory is demoted.

        **Peer-relative, via a robust outlier rule.** A memory is quarantined
        when its LCB sits far below the LCB of its peers in the same scope:

            cutoff = median(peer LCB) - k * 1.4826 * MAD(peer LCB)

        Two earlier rules were tried and both failed for instructive reasons,
        recorded here because the failure modes are not obvious:

        1. **Absolute constant (the HLD Appendix A default, 0.25).**
           ``utility_lcb(1, 1, 0.25)`` is exactly 0.25, so the uninformative
           prior sat precisely on the threshold and everything with any blame
           at all was quarantined.

        2. **Fraction of the scope's base success rate.** Self-defeating: a
           genuinely harmful memory that appears in most recalls drags the
           scope's base rate down with it, which lowers the very cutoff meant
           to catch it. The worse the poison, the more it hides.

        The peer-relative rule is immune to both. It does not care about the
        absolute difficulty of the agent's work, and a memory cannot escape by
        dragging everyone down -- it is measured against the distribution it
        is a member of. The 1.4826 factor rescales MAD to be a consistent
        estimator of sigma for normal data, so ``k`` reads as "standard
        deviations below the median".
        """
        floor = self.decay.quarantine_threshold
        if self.decay.quarantine_mode == "absolute":
            return floor

        peers = [
            utility_lcb(n.alpha, n.beta, self.config.utility_lcb_quantile)
            for n in self.store.iter_nodes([scope_id], states=("active", "contested"))
            if self.is_quarantinable(n)
            and (n.alpha - 1.0) + (n.beta - 1.0) >= self.decay.min_evidence_for_quarantine
        ]
        if len(peers) < self.decay.quarantine_min_peers:
            return floor  # no distribution to compare against yet

        peers.sort()
        median = _median(peers)
        mad = _median([abs(p - median) for p in peers])

        # Outlier test: how far below the peer median is unusual?
        mad_cutoff = (
            median - self.decay.quarantine_mad_k * 1.4826 * mad
            if mad > 1e-9
            else floor
        )

        # Severity cap: a memory must also be *substantially* worse than
        # typical, not merely in the lower tail.
        #
        # Without this cap the rule cascades and eats the scope. Healthy
        # memories converge to a tight LCB distribution, MAD collapses toward
        # zero, and `median - k*MAD` creeps up to just under the median --
        # at which point ordinary lower-tail variation is "an outlier". Each
        # false quarantine then shrinks the peer set and shifts the median,
        # quarantining more. Measured on held-out seeds before this cap: 4.7
        # of 12 healthy memories destroyed, cascading to 9 of 12 by recall 80.
        cap = median * self.decay.quarantine_max_fraction

        return min(mad_cutoff, cap)

    @staticmethod
    def is_quarantinable(node) -> bool:
        """Only claim-bearing memories can be quarantined.

        Two exclusions, both of which caused real damage before being added:

        - **Structural nodes.** An `entity` is the graph's skeleton and a
          `theme` is a view over a community, not an assertion about the
          world. Quarantining either removes a hub from traversal and
          silently disconnects every memory that routed through it -- the
          memories themselves stay 'active' but become unreachable, which is
          strictly worse than demoting them, because nothing reports it.
        - **Human assertions.** HLD 8.6 says humans win conflicts. A rule
          that lets a noisy correlational signal overrule a human is the same
          rule with the sign flipped. An `asserted` memory can only be
          removed by `forget`.
        """
        if node.provenance == "asserted":
            return False
        return node.type not in ("entity", "theme")

    def revive(self, node_id: str) -> None:
        """Bring a quarantined memory back, resetting its posterior to the
        prior. Used by human `assert` and by the exoneration path."""
        self.store.update_node(
            node_id, state="active", alpha=1.0, beta=1.0, attributions=0
        )

    def _all_scopes(self) -> list[str]:
        rows = self.store.conn.execute("SELECT DISTINCT scope_id FROM nodes").fetchall()
        return [r[0] for r in rows]

    # -- ablation ----------------------------------------------------------

    def ablation_summary(self, scope_id: str) -> dict[str, Any]:
        """Compare outcome rates between ablated and normal recalls.

        This is the only mechanism in the system that yields a genuinely
        causal signal about whether recalled memories help (HLD 9.4).
        """
        rows = self.store.conn.execute(
            """SELECT r.is_ablation AS ablated, o.outcome AS outcome, COUNT(*) AS n
               FROM recalls r JOIN outcomes o ON o.recall_id = r.id
               WHERE r.scope_id = ? AND o.outcome IN ('success','failure')
               GROUP BY r.is_ablation, o.outcome""",
            (scope_id,),
        ).fetchall()

        tally = {0: {"success": 0, "failure": 0}, 1: {"success": 0, "failure": 0}}
        for r in rows:
            tally[int(r["ablated"])][r["outcome"]] = int(r["n"])

        def rate(d: dict[str, int]) -> float | None:
            n = d["success"] + d["failure"]
            return d["success"] / n if n else None

        full, ablated = rate(tally[0]), rate(tally[1])
        return {
            "full_recall_success_rate": full,
            "ablated_success_rate": ablated,
            "lift": (full - ablated) if (full is not None and ablated is not None) else None,
            "n_full": sum(tally[0].values()),
            "n_ablated": sum(tally[1].values()),
        }
