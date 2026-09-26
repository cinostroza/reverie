"""Recall (HLD 10). The hot read path.

Cue -> hybrid seed -> spreading activation -> rank -> collapse -> budget -> brief.

The latency budget shapes everything. Three guards keep the traversal bounded
regardless of graph shape: an activation floor that prunes most of hop 2, a
hard visit cap, and one edge query per *frontier* rather than per node.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from ._math import utility_lcb
from .config import Config
from .embed import Embedder, get_embedder
from .models import Node, RecallResult, ScoredNode, now_ms
from .render import estimate_tokens, render_brief
from .store import SQLiteStore

__all__ = ["RecallEngine"]

_DAY_MS = 86_400_000


@dataclass(slots=True)
class _Seed:
    node_id: str
    score: float
    source: str


class RecallEngine:
    def __init__(
        self,
        store: SQLiteStore,
        config: Config | None = None,
        embedder: Embedder | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.store = store
        self.config = config or Config()
        self.embedder = embedder or get_embedder(self.config.embedding_model)
        self._rng = rng or random.Random()

    # -- 1. seed -----------------------------------------------------------

    def _seed(self, cue: str, scopes: list[str]) -> list[_Seed]:
        """Reciprocal rank fusion over BM25, vector, and exact entity match."""
        cfg = self.config.recall
        rankings: list[tuple[list[str], float]] = []

        fts = self.store.fts_search(cue, scopes, k=cfg.seed_k)
        rankings.append(([nid for nid, _ in fts], 1.0))

        vec = self.store.vector_search(self.embedder.embed(cue), scopes, k=cfg.seed_k)
        rankings.append(([nid for nid, _ in vec], 1.0))

        # Exact entity match carries the highest weight: if the cue literally
        # names a thing we have a node for, that is the strongest entry point
        # available and no fuzzy method should outrank it.
        entity_hits = self._resolve_cue_entities(cue, scopes)
        rankings.append((entity_hits, 2.0))

        fused: dict[str, float] = {}
        for ids, weight in rankings:
            for rank, nid in enumerate(ids):
                fused[nid] = fused.get(nid, 0.0) + weight / (cfg.rrf_k + rank + 1)

        # Conjunctive boost: reward candidates that match *many* cue entities.
        for nid, (score, _cov, _con) in self._conjunctive(entity_hits, scopes).items():
            fused[nid] = fused.get(nid, 0.0) + cfg.conjunctive_weight * score

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: cfg.seed_top]
        if not top:
            return []

        # Normalise so initial activation is scale-free across cue types.
        peak = top[0][1] or 1.0
        return [_Seed(nid, score / peak, "fused") for nid, score in top]


    def _conjunctive(
        self, entity_ids: list[str], scopes: list[str]
    ) -> dict[str, tuple[float, float, float]]:
        """IDF-weighted coverage of the cue's entities, per candidate node.

        Activation spreading treats "touches one cue entity" and "touches all
        of them" almost alike, because each entity floods its own
        neighbourhood independently. On a landscape of 72 service x symptom
        pairs that leaves a 1-in-6 chance of the right symptom once the
        service matches, and measured exact-precedent recall was 13% against
        naive recency's 42%.

        Coverage is IDF-weighted so a decoy entity attached to most of the
        graph (`severity=sev1`) cannot stand in for a discriminative one, and
        raised to `conjunctive_exponent` so matching two entities is worth
        more than twice matching one -- the intersection is the whole point.
        """
        cfg = self.config.recall
        if not cfg.conjunctive or len(entity_ids) < 2:
            return {}

        degrees = self.store.entity_degrees(entity_ids, scopes)
        total_nodes = max(sum(degrees.values()), 1)

        # Rarer entity -> more evidence that a shared neighbour is relevant.
        idf = {
            eid: math.log(1.0 + total_nodes / (1.0 + degrees.get(eid, 0)))
            for eid in entity_ids
        }
        denom = sum(idf.values())
        if denom <= 0.0:
            return {}

        hits = self.store.conjunctive_neighbors(entity_ids, scopes)
        candidates = [nid for nid, matched in hits.items() if len(matched) >= 2]
        if not candidates:
            return {}

        # Coverage alone is not enough. Generic hub nodes -- a `failure_mode`
        # like "restart did not mitigate", linked to every entity that ever
        # co-occurred with it -- achieve full cue coverage trivially and tie
        # with the one exact precedent at 1.0. Measured: seven such hubs scored
        # 1.000 alongside the correct episode, which then ranked 8th and was
        # cut by `seed_top`.
        #
        # So weight coverage (how much of the *cue* the candidate matches) by
        # specificity (how much of the *candidate* is the cue). An episode
        # about exactly this service and symptom has degree 3 and specificity
        # 1.0; a hub touching 300 entities has specificity 0.01. This is the
        # same intuition as IDF, applied to the candidate rather than the term.
        degrees = self.store.node_degrees(candidates, scopes)

        scored: dict[str, tuple[float, float, float]] = {}
        for nid in candidates:
            matched = hits[nid]
            # coverage:    how much of the CUE this candidate accounts for
            # containment: how much of the CANDIDATE the cue accounts for
            #
            # Both are needed, because generality and specificity are
            # different virtues. An episode about (service, severity, symptom)
            # is a precedent when all three match -- high coverage. A human
            # lesson about (service, symptom) deliberately drops the decoy and
            # applies to *any* severity -- lower coverage, but total
            # containment, and it is more useful rather than less.
            #
            # Admitting on coverage alone excluded every generalising lesson:
            # weighted coverage came to ~0.78 against a 0.95 threshold, so
            # lessons never entered the precision channel and reached the
            # brief only through the association channel's procedural quota,
            # relevance-blind.
            coverage = min(1.0, sum(idf.get(e, 0.0) for e in matched) / denom)
            containment = len(matched) / max(degrees.get(nid, len(matched)), 1)
            score = (coverage ** cfg.conjunctive_exponent) * (
                containment ** cfg.conjunctive_specificity
            )
            scored[nid] = (score, coverage, containment)

        top = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
        return dict(top[: cfg.conjunctive_k])


    def _resolve_cue_entities(self, cue: str, scopes: list[str]) -> list[str]:
        hits: list[str] = []
        for token in _candidate_entities(cue):
            node = self.store.find_by_label(token, scopes) or self.store.find_by_alias(
                token, scopes
            )
            if node and node.id not in hits:
                hits.append(node.id)
        return hits

    def _direct_matches(self, cue: str, scopes: list[str]) -> dict[str, float]:
        """Precision channel: exact precedents, never routed through spreading.

        The association channel (seed -> spread -> rank) is built on the
        premise that the useful memory is *not* the one most similar to the
        cue, and E3 confirms that for multi-hop causal chains. Precedent
        lookup is the opposite problem, and the same machinery actively
        destroys it.

        Measured on the on-call benchmark: the correct precedent is present in
        the graph, the seed stage ranks it first with a conjunctive score of
        1.000, and it reaches the brief 4-6% of the time. Activation
        accumulates on hub nodes over many paths until it outscores a
        correctly-seeded leaf, so the traversal reliably discards the one node
        that already had the answer.

        So this channel skips the traversal entirely. A candidate matching
        enough of the cue's entities is returned on its own authority, with a
        reserved slice of the budget, and competes with nothing.
        """
        cfg = self.config.recall
        if not cfg.direct_match:
            return {}
        entity_ids = self._resolve_cue_entities(cue, scopes)
        if len(entity_ids) < 2:
            return {}
        # Gate on *coverage of the cue*, rank by the composite score.
        #
        # Gating on the composite was wrong and produced zero direct matches.
        # The composite folds in specificity, which divides by the candidate's
        # entity degree -- so a perfect match scored 0.375 against a 0.5
        # threshold purely because it was attached to eight entities rather
        # than three. Whether a memory is *about what was asked* is a property
        # of how much of the cue it covers; how distinctive it is among such
        # memories is a ranking question, not an admission one.
        # Admit on EITHER: the candidate accounts for the whole cue (an exact
        # episode), or the cue accounts for the whole candidate (a lesson that
        # generalises over features the cue also happens to specify).
        return {
            nid: score
            for nid, (score, coverage, containment) in
            self._conjunctive(entity_ids, scopes).items()
            if coverage >= cfg.direct_min_coverage
            or containment >= cfg.direct_min_containment
        }

    # -- 2. spread ---------------------------------------------------------

    def _spread(
        self, seeds: list[_Seed], scopes: list[str]
    ) -> tuple[dict[str, float], int, bool]:
        cfg = self.config.recall
        activation: dict[str, float] = {s.node_id: s.score for s in seeds}
        frontier = [s.node_id for s in seeds]
        visited: set[str] = set()
        hit_cap = False

        # Community membership drives the intra-community traversal bonus.
        communities = self._community_map(scopes)

        for hop in range(cfg.max_hops):
            if not frontier or hit_cap:
                break

            live = [
                n
                for n in frontier
                if n not in visited and activation.get(n, 0.0) >= cfg.activation_floor
            ]
            if not live:
                break

            if len(visited) + len(live) > cfg.visit_cap:
                live = live[: max(0, cfg.visit_cap - len(visited))]
                hit_cap = True
            visited.update(live)

            edge_map = self.store.edges_from(live, scopes, limit_per_node=cfg.fanout)
            decay = cfg.hop_decay[hop] if hop < len(cfg.hop_decay) else cfg.hop_decay[-1]

            next_frontier: list[str] = []
            for src in live:
                a = activation.get(src, 0.0)
                for edge in edge_map.get(src, ()):
                    hop_decay = decay
                    if (
                        communities.get(src) is not None
                        and communities.get(src) == communities.get(edge.dst)
                    ):
                        hop_decay *= cfg.intra_community_bonus

                    delta = a * edge.weight * hop_decay * edge.confidence
                    if edge.relation == "contradicts":
                        # Contradiction dampens rather than propagating: a
                        # memory that conflicts with an activated one should
                        # be suppressed, not amplified by association.
                        delta *= cfg.contradiction_damping

                    activation[edge.dst] = activation.get(edge.dst, 0.0) + delta
                    if edge.dst not in visited:
                        next_frontier.append(edge.dst)

            frontier = next_frontier

        return activation, len(visited), hit_cap

    def _community_map(self, scopes: list[str]) -> dict[str, str | None]:
        q = (
            f"SELECT id, community_id FROM nodes "
            f"WHERE scope_id IN ({','.join('?' * len(scopes))}) AND community_id IS NOT NULL"
        )
        return {r["id"]: r["community_id"] for r in self.store.conn.execute(q, scopes)}

    # -- 3. rank -----------------------------------------------------------

    def _score_node(
        self, node: Node, activation: float, now: int, context: dict[str, str] | None
    ) -> float:
        cfg = self.config.recall

        recency = 1.0
        if node.last_activated:
            age_days = max(0.0, (now - node.last_activated) / _DAY_MS)
            recency = 0.5 + 0.5 * math.exp(-age_days / cfg.recency_halflife_days)

        utility = utility_lcb(
            node.alpha, node.beta, self.config.attribution.utility_lcb_quantile
        )
        provenance = cfg.provenance_factor.get(node.provenance, 0.5)

        context_factor = 1.0
        if context and node.learned_under:
            env = node.learned_under.get("env_hash")
            if env and context.get("env_hash") and env != context["env_hash"]:
                context_factor = cfg.context_mismatch_penalty

        valence_gain = 1.0 + cfg.valence_gain * abs(node.valence)

        return activation * recency * utility * provenance * context_factor * valence_gain

    # -- 4. collapse -------------------------------------------------------

    def _collapse_communities(self, scored: list[ScoredNode]) -> list[ScoredNode]:
        """Swap a dense cluster of low-scoring members for its theme summary.

        When activation spreads thin across many members of one community,
        the summary carries most of the information at a fraction of the
        tokens (HLD 8.7).
        """
        cfg = self.config.recall
        by_community: dict[str, list[ScoredNode]] = {}
        loose: list[ScoredNode] = []

        for sn in scored:
            cid = sn.node.community_id
            if cid and sn.node.type != "theme":
                by_community.setdefault(cid, []).append(sn)
            else:
                loose.append(sn)

        out = list(loose)
        for cid, members in by_community.items():
            if len(members) < cfg.collapse_min_members:
                out.extend(members)
                continue

            theme = self._theme_node(cid)
            if theme is None:
                out.extend(members)
                continue

            # Only collapse when no single member is carrying the recall. A
            # strong individual memory must never be hidden behind a summary.
            members.sort(key=lambda s: s.score, reverse=True)
            best = members[0].score
            mean = sum(m.score for m in members) / len(members)
            if best > 2.0 * mean:
                out.extend(members)
                continue

            out.append(
                ScoredNode(
                    node=theme,
                    activation=sum(m.activation for m in members),
                    score=best,
                    tokens=estimate_tokens(theme.label) + estimate_tokens(theme.body),
                )
            )
        return out

    def _theme_node(self, community_id: str) -> Node | None:
        r = self.store.conn.execute(
            "SELECT * FROM nodes WHERE community_id=? AND type='theme' "
            "AND state='active' LIMIT 1",
            (community_id,),
        ).fetchone()
        if r is None:
            return None
        from .store import _row_to_node

        return _row_to_node(r)

    # -- 5. select ---------------------------------------------------------

    def _select(self, scored: list[ScoredNode], cfg=None,
                direct: list[ScoredNode] | None = None) -> list[ScoredNode]:
        """Greedy knapsack under a token budget with per-class quotas.

        Greedy by score-per-token is not optimal, but the budget is small and
        the items are homogeneous in size; the optimal solve is not worth the
        latency.

        ``cfg`` is threaded in rather than read off ``self`` so a per-call
        ``budget_tokens`` override is actually honoured -- reading the
        instance config here silently ignored the caller's budget.
        """
        cfg = cfg or self.config.recall
        scored.sort(key=lambda s: s.score / max(1, s.tokens), reverse=True)

        max_nodes = _resolve_max_nodes(cfg)

        # Precision channel first, on a reserved slice of the budget. Without
        # a reservation these lose to whatever the traversal inflated, which
        # is the whole failure this channel exists to route around.
        selected: list[ScoredNode] = []
        total = 0
        taken: set[str] = set()
        if direct:
            reserve = int(cfg.budget_tokens * cfg.direct_reserve)
            for sn in sorted(direct, key=lambda s: s.score, reverse=True):
                if len(selected) >= max_nodes or total + sn.tokens > reserve:
                    break
                selected.append(sn)
                taken.add(sn.node.id)
                total += sn.tokens
        quota_tokens = {
            cls: int(cfg.budget_tokens * frac)
            for cls, frac in _effective_quotas(cfg, scored).items()
        }
        spent = dict.fromkeys(quota_tokens, 0)
        contested = 0

        for sn in scored:
            if len(selected) >= max_nodes or total >= cfg.budget_tokens:
                break
            if sn.score <= 0.0 or sn.node.id in taken:
                continue
            if sn.node.state == "contested":
                if contested >= cfg.max_contested:
                    continue
                contested += 1

            cls = sn.node.memory_class
            if cls in quota_tokens and spent[cls] + sn.tokens > quota_tokens[cls]:
                continue
            if total + sn.tokens > cfg.budget_tokens:
                continue

            selected.append(sn)
            total += sn.tokens
            if cls in spent:
                spent[cls] += sn.tokens

        # Any budget left over after quotas is offered back, best-first. A
        # memory-rich agent with only procedural memories should still fill
        # its brief rather than returning a third of one.
        if total < cfg.budget_tokens and len(selected) < max_nodes:
            chosen = {s.node.id for s in selected}
            for sn in scored:
                if len(selected) >= max_nodes or total + sn.tokens > cfg.budget_tokens:
                    break
                if sn.node.id in chosen or sn.score <= 0.0:
                    continue
                selected.append(sn)
                total += sn.tokens

        selected.sort(key=lambda s: s.score, reverse=True)
        return selected

    # -- public ------------------------------------------------------------

    def recall(
        self,
        cue: str,
        scope_id: str,
        *,
        scope_chain: list[str] | None = None,
        budget_tokens: int | None = None,
        context: dict[str, str] | None = None,
        log: bool = True,
        allow_ablation: bool = True,
    ) -> RecallResult:
        started = time.perf_counter()
        cfg = self.config.recall
        if budget_tokens is not None:
            cfg = _with_budget(cfg, budget_tokens)

        scopes = scope_chain or [scope_id]
        now = now_ms()

        direct_hits = self._direct_matches(cue, scopes)

        seeds = self._seed(cue, scopes)
        if not seeds:
            return RecallResult(
                brief=render_brief([], empty_reason="no_seed"),
                recall_id="",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        activation, visited, hit_cap = self._spread(seeds, scopes)

        nodes = self.store.get_nodes(activation.keys())
        scored: list[ScoredNode] = []
        for nid, act in activation.items():
            node = nodes.get(nid)
            if node is None or not node.is_recallable:
                continue
            score = self._score_node(node, act, now, context)
            if score <= 0.0:
                continue
            scored.append(
                ScoredNode(
                    node=node,
                    activation=act,
                    score=score,
                    tokens=estimate_tokens(node.label) + estimate_tokens(node.body),
                )
            )

        # Precision channel: exact precedents, assembled independently of the
        # traversal so they cannot be outscored by inflated hubs.
        direct_scored: list[ScoredNode] = []
        if direct_hits:
            missing = [nid for nid in direct_hits if nid not in nodes]
            if missing:
                nodes.update(self.store.get_nodes(missing))
            for nid, confidence in direct_hits.items():
                node = nodes.get(nid)
                if node is None or not node.is_recallable:
                    continue
                # Track record breaks ties between precedents. Admission is
                # already settled by cue coverage, so utility only decides
                # *which* of several matching precedents to spend budget on --
                # which is the question attribution exists to answer. Without
                # this the precision channel ignores the posterior entirely and
                # attribution lift is zero by construction.
                rank_score = confidence
                if cfg.direct_utility_weight > 0.0:
                    u = utility_lcb(
                        node.alpha, node.beta,
                        self.config.attribution.utility_lcb_quantile,
                    )
                    rank_score *= u ** cfg.direct_utility_weight

                direct_scored.append(
                    ScoredNode(
                        node=node,
                        activation=activation.get(nid, 0.0),
                        # A match score in [0,1], deliberately not comparable
                        # to an activation score -- these are ranked among
                        # themselves and given their own budget, never pooled
                        # with the traversal.
                        score=rank_score,
                        tokens=estimate_tokens(node.label) + estimate_tokens(node.body),
                    )
                )

        direct_ids = {s.node.id for s in direct_scored}
        scored = self._collapse_communities(scored)
        selected = self._select(scored, cfg, direct=direct_scored)

        # Ablation: withhold the lowest-ranked members on a small fraction of
        # recalls so the outcome difference gives a causal read (HLD 9.4).
        withheld: list[str] = []
        is_ablation = False
        acfg = self.config.attribution
        if allow_ablation and self._rng.random() < acfg.ablation_rate and len(selected) > 1:
            is_ablation = True
            n = min(acfg.ablation_withhold_n, len(selected) - 1)
            if acfg.ablation_mode == "stratified" and self._rng.random() < 0.5:
                dropped = selected[:n]          # withhold the *best*
                selected = selected[n:]
            else:
                dropped = selected[-n:]         # withhold the worst
                selected = selected[:-n]
            withheld = [s.node.id for s in dropped]

        latency = (time.perf_counter() - started) * 1000
        token_cost = sum(s.tokens for s in selected)
        brief = render_brief(selected, base_rate_lookup=self._success_ratio)

        recall_id = ""
        if log and selected:
            recall_id = self.store.log_recall(
                scope_id,
                cue,
                [s.node.id for s in selected],
                _credit_weights(selected, direct_ids),
                withheld_ids=withheld,
                is_ablation=is_ablation,
                token_cost=token_cost,
                latency_ms=latency,
            )
            for s in selected:
                self.store.update_node(
                    s.node.id,
                    activations=s.node.activations + 1,
                    last_activated=now,
                )

        return RecallResult(
            brief=brief,
            recall_id=recall_id,
            nodes=selected,
            token_cost=token_cost,
            latency_ms=latency,
            visited=visited,
            hit_visit_cap=hit_cap,
            is_ablation=is_ablation,
            withheld_ids=withheld,
        )

    def _success_ratio(self, node: Node) -> tuple[int, int] | None:
        """Successes / attributions, for the track record shown in a brief."""
        if node.attributions <= 0:
            return None
        successes = round(node.alpha - 1.0)
        total = round(node.alpha + node.beta - 2.0)
        if total <= 0:
            return None
        return int(successes), int(total)




def _credit_weights(
    selected: list[ScoredNode], direct_ids: set[str]
) -> dict[str, float]:
    """Attribution credit share per selected node (HLD 9.2).

    Credit cannot be apportioned from raw scores, because the two recall
    channels produce scores on deliberately incomparable scales: precision is
    a match confidence in [0,1], association is an activation product. Two
    wrong ways to do this were measured before landing on the third:

    * **Raw activation.** Precision-channel nodes bypass spreading, so their
      activation is 0 and they received *no* credit at all -- lessons
      participated in a median of 50 recalls each and accumulated a median
      evidence mass of 0.81, pinning every utility LCB near the 0.25 prior.
    * **Raw score.** Whichever channel happens to produce larger numbers
      swamps the other. Measured worse than activation: median mass fell to
      0.47.

    So normalise *within* each channel, then let the channels contribute on
    equal footing. A top-ranked precision node and a top-ranked association
    node both carry weight 1.0, which is the honest statement: we know each was
    the most prominent thing its channel contributed, and we cannot say more.
    """
    out: dict[str, float] = {}
    for group in (
        [s for s in selected if s.node.id in direct_ids],
        [s for s in selected if s.node.id not in direct_ids],
    ):
        if not group:
            continue
        peak = max(s.score for s in group) or 1.0
        for s in group:
            out[s.node.id] = max(s.score / peak, 1e-6)
    return out

def _resolve_max_nodes(cfg) -> int:
    """Node cap, derived from the token budget unless explicitly set.

    Keeps the two budget knobs consistent. A hard-coded cap that binds before
    the token budget makes `budget_tokens` inert, which is what E5's
    sensitivity sweep found: capture was flat across 200..1200 tokens because
    the node cap never let the extra budget be spent.
    """
    if cfg.max_nodes is not None:
        return int(cfg.max_nodes)
    derived = cfg.budget_tokens // max(1, cfg.tokens_per_node_estimate)
    return max(cfg.max_nodes_floor, min(cfg.max_nodes_ceiling, derived))


def _effective_quotas(cfg, scored) -> dict[str, float]:
    """Renormalise class quotas over the classes actually available.

    A quota for a class with no candidates is not a reservation for later --
    it is budget that cannot be spent at all. Redistributing it in proportion
    to the configured preferences keeps the intended ordering (procedural is
    still favoured over episodic where both exist) while letting a graph that
    only holds episodes fill its brief with episodes.
    """
    quotas = cfg.type_quotas
    if not getattr(cfg, "adaptive_quotas", False):
        return quotas

    present = {s.node.memory_class for s in scored if s.score > 0.0}
    available = {c: f for c, f in quotas.items() if c in present and f > 0.0}
    total = sum(available.values())
    if not available or total <= 0.0:
        return {}
    return {c: f / total for c, f in available.items()}


def _with_budget(cfg, budget: int):
    from copy import replace

    return replace(cfg, budget_tokens=budget)


def _candidate_entities(cue: str) -> list[str]:
    """Cheap entity candidates from a cue: no NER, no model call.

    Deliberately crude. Real entity extraction happens during consolidation
    where there is a budget for it; here the only job is to notice when the
    cue literally names a node we already have.
    """
    # `=` is preserved so a structured cue like "service=svc-03,symptom=sym-2"
    # yields the entity labels that were actually stored. Splitting on it
    # produced "service" and "svc-03" separately, neither of which matches the
    # node labelled "service=svc-03" -- so entity resolution silently failed
    # on every structured cue and the conjunctive signal had nothing to work
    # with. Both the whole token and its parts are emitted, so looser cues
    # still resolve.
    tokens = "".join(ch if (ch.isalnum() or ch in "_-.=") else " " for ch in cue).split()
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        parts = [t, *(t.split("=") if "=" in t else [])]
        for p in parts:
            if len(p) < 3 or p.lower() in _CUE_STOP or p in seen:
                continue
            seen.add(p)
            out.append(p)
    for a, b in zip(tokens, tokens[1:]):
        if len(a) > 2 and len(b) > 2:
            out.append(f"{a} {b}")
    return out


_CUE_STOP = frozenset(
    "the and for with from that this what how why when where should could would "
    "can will does did are was were has have had not but you your our its".split()
)
