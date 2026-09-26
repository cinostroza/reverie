"""Consolidation -- the dream cycle (HLD 8).

    segment -> score -> distill -> link -> reconcile -> abstract -> prune

Idempotent and resumable: episodes are marked consolidated only on commit of
the unit that contained them, so an interrupted cycle re-processes at most one
unit. Re-processing is harmless because `link` reinforces matching nodes
rather than inserting duplicates.

The `distill` stage is the only one that needs a model. Everything else runs
on structured episode fields, which is what makes the no-LLM path (HLD 8.9)
real rather than aspirational.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .community import cohesion, louvain, match_communities, modularity
from .config import Config
from .embed import Embedder, cosine, get_embedder
from .models import Edge, Episode, Node, new_id, now_ms
from .store import SQLiteStore

__all__ = ["Consolidator", "ConsolidationStats", "Distiller", "NullDistiller", "Unit"]

_DAY_MS = 86_400_000


@dataclass(slots=True)
class Unit:
    """A consolidation unit: the episodes distillation sees together."""

    episodes: list[Episode]
    buffer_ids: list[str]
    task_type: str
    salience: float = 0.0

    @property
    def entities(self) -> list[str]:
        seen: dict[str, None] = {}
        for ep in self.episodes:
            for e in ep.entities:
                seen.setdefault(e, None)
        return list(seen)

    @property
    def outcomes(self) -> list[str]:
        return [ep.outcome.status for ep in self.episodes if ep.outcome]


@dataclass(slots=True)
class ConsolidationStats:
    units_seen: int = 0
    units_distilled: int = 0
    units_archived: int = 0
    nodes_created: int = 0
    nodes_reinforced: int = 0
    edges_created: int = 0
    contradictions: int = 0
    candidates_rejected: dict[str, int] = field(default_factory=dict)
    communities: int = 0
    pruned: int = 0
    quarantined: int = 0
    scopes_narrowed: int = 0
    scopes_demoted: int = 0
    episodes_consolidated: int = 0

    def reject(self, reason: str) -> None:
        self.candidates_rejected[reason] = self.candidates_rejected.get(reason, 0) + 1


class Distiller(Protocol):
    """The LLM seam. Returns candidate memories under the HLD 8.4 schema."""

    def distill(self, unit: Unit) -> dict[str, list[dict[str, Any]]]: ...


class NullDistiller:
    """The no-LLM path (HLD 8.9).

    Builds episodic nodes and the entity graph from structured episode fields
    only. No semantic or procedural memories -- those genuinely require a
    model -- but recall still works and valence still accrues from outcomes.
    """

    def distill(self, unit: Unit) -> dict[str, list[dict[str, Any]]]:
        entities = [{"label": e, "type": "entity", "aliases": []} for e in unit.entities]

        # Outcome-derived valence needs no model: a failed task touching an
        # entity is evidence about that entity.
        valence: list[dict[str, Any]] = []
        for ep in unit.episodes:
            if not ep.outcome:
                continue
            delta = {"failure": -0.15, "partial": -0.05, "success": 0.08}.get(
                ep.outcome.status, 0.0
            )
            if delta:
                for e in ep.entities:
                    valence.append(
                        {"entity": e, "delta": delta, "reason": f"task {ep.outcome.status}"}
                    )

        # A recurring, named error is a failure mode observable from structure
        # alone -- no interpretation required.
        failure_modes: list[dict[str, Any]] = []
        errors: dict[str, list[int]] = defaultdict(list)
        for ep in unit.episodes:
            for i, step in enumerate(ep.steps):
                if step.error:
                    errors[step.error].append(i)
        for error, indices in errors.items():
            if len(indices) >= 1:
                failure_modes.append(
                    {
                        "label": error,
                        "symptom": error,
                        "root_cause": "",
                        "resolution": "",
                        "cites": indices,
                        "provenance": "observed",
                    }
                )

        return {
            "semantic": [],
            "procedural": [],
            "failure_modes": failure_modes,
            "valence": valence,
            "entities": entities,
        }


class Consolidator:
    def __init__(
        self,
        store: SQLiteStore,
        config: Config | None = None,
        distiller: Distiller | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = store
        self.config = config or Config()
        self.distiller = distiller or NullDistiller()
        self.embedder = embedder or get_embedder(self.config.embedding_model)
        # Per-node valence spend for the current cycle; reset in run().
        self._valence_spent: dict[str, float] = {}

    # -- 8.2 segment -------------------------------------------------------

    def segment(self, scope_id: str, limit: int) -> list[Unit]:
        rows = self.store.pending_episodes(scope_id, limit)
        by_task: dict[str, Unit] = {}

        for row in rows:
            episode = Episode.from_dict(json.loads(row["payload"]))
            # Default unit is the task; episodes sharing a task id batch so
            # distillation sees repetition, which is what turns an anecdote
            # into a lesson.
            key = episode.task.id or row["id"]
            unit = by_task.get(key)
            if unit is None:
                unit = Unit(episodes=[], buffer_ids=[], task_type=row["task_type"])
                by_task[key] = unit
            unit.episodes.append(episode)
            unit.buffer_ids.append(row["id"])

        # Cross-task batching: units sharing entities are merged so repetition
        # across tasks is visible too.
        return self._batch_by_entity(list(by_task.values()))

    def _batch_by_entity(self, units: list[Unit]) -> list[Unit]:
        if len(units) < 2:
            return units

        parent = list(range(len(units)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        by_entity: dict[str, list[int]] = defaultdict(list)
        for i, unit in enumerate(units):
            for entity in unit.entities:
                by_entity[entity].append(i)

        for indices in by_entity.values():
            # Only batch small groups; merging everything that touches a hub
            # entity would produce one enormous unit.
            if 2 <= len(indices) <= 8:
                for j in indices[1:]:
                    union(indices[0], j)

        merged: dict[int, Unit] = {}
        for i, unit in enumerate(units):
            root = find(i)
            if root not in merged:
                merged[root] = Unit(
                    episodes=[], buffer_ids=[], task_type=unit.task_type
                )
            merged[root].episodes.extend(unit.episodes)
            merged[root].buffer_ids.extend(unit.buffer_ids)
        return list(merged.values())

    # -- 8.3 score ---------------------------------------------------------

    def score(self, unit: Unit, scope_id: str) -> float:
        w = self.config.consolidation.salience_weights

        outcomes = unit.outcomes
        if outcomes:
            extremity = sum(
                1.0 if o in ("failure", "success") else 0.3 for o in outcomes
            ) / len(outcomes)
        else:
            extremity = 0.2

        novelty = self._novelty(unit, scope_id)

        durations = [ep.duration_ms for ep in unit.episodes]
        steps = sum(len(ep.steps) for ep in unit.episodes)
        cost = min(1.0, (sum(durations) / 600_000.0) * 0.5 + (steps / 100.0) * 0.5)

        centrality = self._entity_centrality(unit, scope_id)
        redundancy = 1.0 - novelty

        return (
            w["outcome_extremity"] * extremity
            + w["novelty"] * novelty
            + w["cost"] * cost
            + w["entity_centrality"] * centrality
            - w["redundancy"] * redundancy
        )

    def _unit_text(self, unit: Unit) -> str:
        parts: list[str] = []
        for ep in unit.episodes:
            parts.append(ep.task.description)
            for step in ep.steps:
                parts.append(f"{step.name} {step.input} {step.output_summary} {step.error or ''}")
        return " ".join(parts)

    def _novelty(self, unit: Unit, scope_id: str) -> float:
        vec = self.embedder.embed(self._unit_text(unit))
        if not any(vec):
            return 0.5
        hits = self.store.vector_search(vec, [scope_id], k=1)
        if not hits:
            return 1.0
        return max(0.0, min(1.0, 1.0 - hits[0][1]))

    def _entity_centrality(self, unit: Unit, scope_id: str) -> float:
        entities = unit.entities
        if not entities:
            return 0.0
        total = 0.0
        for label in entities:
            node = self.store.find_by_label(label, [scope_id])
            if node:
                total += min(1.0, node.centrality)
        return min(1.0, total / len(entities))

    # -- 8.5 link ----------------------------------------------------------

    def _resolve_entity(self, label: str, scope_id: str) -> Node:
        node = self.store.find_by_label(label, [scope_id])
        if node:
            return node
        node = self.store.find_by_alias(label, [scope_id])
        if node:
            return node

        vec = self.embedder.embed(label)
        hits = self.store.vector_search(vec, [scope_id], k=1)
        if hits and hits[0][1] >= 0.97:
            candidate = self.store.get_node(hits[0][0])
            # Only fold into another *entity*; a high-similarity lesson is not
            # the same thing as the entity it mentions.
            if candidate and candidate.type == "entity":
                return candidate

        node = Node(
            id=new_id("ent"),
            scope_id=scope_id,
            type="entity",
            memory_class="semantic",
            label=label,
            body=label,
            provenance="observed",
        )
        self.store.add_node(node, self.embedder.embed(label))
        return node

    def _upsert_memory(
        self,
        *,
        scope_id: str,
        node_type: str,
        memory_class: str,
        label: str,
        body: str,
        provenance: str,
        source_ids: list[str],
        learned_under: dict[str, Any],
        stats: ConsolidationStats,
    ) -> tuple[Node, bool]:
        """Insert, or reinforce an existing near-duplicate.

        Reinforcement rather than duplication is how repeated experience
        registers (HLD 8.5 step 2).
        """
        vec = self.embedder.embed(f"{label} {body}")
        threshold = self.config.consolidation.dedup_similarity

        # Episodic nodes are never deduplicated. An episode is an *event*, and
        # two things that happened are two things that happened -- collapsing
        # them destroys the frequency information that "12/13 successes" is
        # computed from.
        #
        # This is not hypothetical. Dedup is embedding-similarity based, and
        # the discriminative part of an episode is often a two-token
        # difference ("strategy=direct -> failure" vs "strategy=sql_first ->
        # success") inside an otherwise identical sentence. Those score well
        # above the 0.88 threshold and were being merged, so the graph held
        # one node where it should have held forty, and the benchmark agent
        # received no usable evidence at all. Episodic volume is controlled by
        # decay and pruning (8.8), which is the mechanism designed for it.
        if memory_class == "episodic":
            node = Node(
                id=new_id("mem"), scope_id=scope_id, type=node_type,
                memory_class=memory_class, label=label, body=body,
                provenance=provenance, source_ids=source_ids,
                learned_under=learned_under,
            )
            self.store.add_node(node, vec)
            stats.nodes_created += 1
            return node, True

        for cand_id, sim in self.store.vector_search(vec, [scope_id], k=5):
            if sim < threshold:
                break
            candidate = self.store.get_node(cand_id)
            if candidate is None or candidate.type != node_type:
                continue
            if candidate.state in ("deleted", "merged"):
                continue

            merged_sources = list(dict.fromkeys(candidate.source_ids + source_ids))
            self.store.update_node(
                candidate.id,
                strength=min(3.0, candidate.strength + 0.25),
                activations=candidate.activations + 1,
                source_ids=merged_sources,
                state="active" if candidate.state == "stale" else candidate.state,
            )
            stats.nodes_reinforced += 1
            return candidate, False

        node = Node(
            id=new_id("mem"),
            scope_id=scope_id,
            type=node_type,
            memory_class=memory_class,
            label=label,
            body=body,
            provenance=provenance,
            source_ids=source_ids,
            learned_under=learned_under,
        )
        self.store.add_node(node, vec)
        stats.nodes_created += 1
        return node, True

    def _validate_candidate(
        self, kind: str, cand: dict[str, Any], unit: Unit, stats: ConsolidationStats
    ) -> bool:
        """HLD 8.4 constraints, enforced after generation.

        The citation check is the cheapest hallucination filter available and
        catches the largest share of bad candidates.
        """
        if kind in ("semantic", "procedural", "failure_modes"):
            cites = cand.get("cites")
            if not cites:
                stats.reject("uncited")
                return False

        if kind == "procedural":
            condition = (cand.get("condition") or "").strip()
            if len(condition) < 12 or _is_vague(condition):
                stats.reject("vague_condition")
                return False
            n_instances = len(unit.episodes)
            if n_instances < self.config.consolidation.min_corroboration_for_lesson:
                stats.reject("insufficient_corroboration")
                return False

        text = " ".join(str(v) for v in cand.values() if isinstance(v, str))
        if _is_imperative_to_agent(text):
            # Security control, not style (HLD 13.2): a memory phrased as an
            # instruction is how an injected payload becomes a standing order.
            stats.reject("imperative_content")
            return False

        return True

    def link(self, unit: Unit, candidates: dict[str, list[dict[str, Any]]],
             scope_id: str, stats: ConsolidationStats) -> list[Node]:
        source_ids = [ep.episode_id for ep in unit.episodes if ep.episode_id]
        learned_under = {}
        if unit.episodes and unit.episodes[0].agent:
            agent = unit.episodes[0].agent
            learned_under = {
                "model": agent.model,
                "version": agent.version,
                "env_hash": agent.env_hash,
            }

        entity_nodes: dict[str, Node] = {}
        for cand in candidates.get("entities", []):
            label = (cand.get("label") or "").strip()
            if not label:
                continue
            node = self._resolve_entity(label, scope_id)
            entity_nodes[label] = node
            for alias in cand.get("aliases", []):
                self.store.add_alias(scope_id, alias, node.id)

        created: list[Node] = []

        # Episodic spine: one node per episode, always observed.
        for ep in unit.episodes:
            label, body = _episodic_text(ep)
            node, is_new = self._upsert_memory(
                scope_id=scope_id, node_type="episode", memory_class="episodic",
                label=label, body=body, provenance="observed",
                source_ids=[ep.episode_id] if ep.episode_id else [],
                learned_under=learned_under, stats=stats,
            )
            if is_new:
                created.append(node)
            for entity in ep.entities:
                target = entity_nodes.get(entity) or self._resolve_entity(entity, scope_id)
                stats.edges_created += self._edge(
                    scope_id, node.id, target.id, "applies_to", 0.6, "observed", 0.9
                )

        spec = [
            ("failure_modes", "failure_mode", "semantic", "observed"),
            ("semantic", "lesson", "semantic", "inferred"),
            ("procedural", "lesson", "procedural", "inferred"),
        ]
        for key, node_type, memory_class, provenance in spec:
            for cand in candidates.get(key, []):
                if not self._validate_candidate(key, cand, unit, stats):
                    continue
                label, body = _candidate_text(key, cand)
                if not label:
                    continue
                node, is_new = self._upsert_memory(
                    scope_id=scope_id, node_type=node_type, memory_class=memory_class,
                    label=label, body=body,
                    provenance=cand.get("provenance", provenance),
                    source_ids=source_ids, learned_under=learned_under, stats=stats,
                )
                if is_new:
                    created.append(node)
                for entity in cand.get("entities", []) or unit.entities:
                    target = entity_nodes.get(entity) or self._resolve_entity(entity, scope_id)
                    stats.edges_created += self._edge(
                        scope_id, target.id, node.id, "applies_to", 0.7,
                        "inferred", 0.6
                    )

        # Temporal spine over episodic nodes, from step ordering. Observed.
        episodic = [n for n in created if n.memory_class == "episodic"]
        for a, b in zip(episodic, episodic[1:]):
            stats.edges_created += self._edge(
                scope_id, a.id, b.id, "preceded", 0.4, "observed", 0.95
            )

        self._apply_valence(candidates.get("valence", []), entity_nodes, scope_id)
        return created

    def _edge(
        self, scope_id: str, src: str, dst: str, relation: str,
        weight: float, provenance: str, confidence: float,
    ) -> int:
        if src == dst:
            return 0
        self.store.add_edge(
            Edge(id=new_id("e"), scope_id=scope_id, src=src, dst=dst,
                 relation=relation, weight=weight, provenance=provenance,
                 confidence=confidence)
        )
        # Association is symmetric for traversal; direction is carried by the
        # relation name, and a reverse edge is what lets activation reach a
        # cause from its effect.
        self.store.add_edge(
            Edge(id=new_id("e"), scope_id=scope_id, src=dst, dst=src,
                 relation=relation, weight=weight * 0.8, provenance=provenance,
                 confidence=confidence)
        )
        return 1

    def _apply_valence(
        self, deltas: list[dict[str, Any]], entities: dict[str, Node], scope_id: str
    ) -> None:
        """Apply valence deltas under a per-node, per-*cycle* budget.

        The cap has to span the whole cycle, not each unit. A scope with
        thirty failing episodes segments into many units, and a per-unit cap
        would let each one spend the full budget -- so one bad afternoon could
        drive an entity to -1.0 and permanently poison it, which is exactly
        what the cap exists to prevent (HLD 8.5 step 4).
        """
        cap = self.config.consolidation.max_valence_delta_per_episode
        accumulated: dict[str, float] = defaultdict(float)
        for d in deltas:
            label = d.get("entity")
            if not label:
                continue
            node = entities.get(label) or self._resolve_entity(label, scope_id)
            accumulated[node.id] += float(d.get("delta", 0.0))

        for node_id, raw in accumulated.items():
            node = self.store.get_node(node_id)
            if node is None:
                continue
            spent = self._valence_spent.get(node_id, 0.0)
            remaining = cap - abs(spent)
            if remaining <= 0.0:
                continue
            delta = max(-remaining, min(remaining, raw))
            self._valence_spent[node_id] = spent + abs(delta)
            self.store.update_node(
                node_id, valence=max(-1.0, min(1.0, node.valence + delta))
            )

    def reinforce_only(
        self, unit: Unit, scope_id: str, stats: ConsolidationStats
    ) -> int:
        """Strengthen existing memories a redundant unit corroborates.

        The cheap half of `link`: embedding lookup and a counter bump, no
        model call and no inserts. This is what makes "I have now seen this
        happen nine times" show up in a brief's track record even though only
        the first occurrence was ever worth distilling.
        """
        reinforced = 0
        threshold = self.config.consolidation.dedup_similarity
        seen: set[str] = set()

        # Match per *episode*, not per unit. A unit's concatenated text is
        # several times longer than any single node's, so its embedding never
        # clears the dedup threshold against one -- matching at unit scale
        # silently reinforced nothing at all.
        for episode in unit.episodes:
            # Must use the *same* representation `link` embedded the node
            # with. Querying with raw step text instead scored ~0.66 against
            # a byte-identical episode -- nowhere near the 0.88 dedup
            # threshold -- so reinforcement silently never fired.
            label, body = _episodic_text(episode)
            text = f"{label} {body}".strip()
            if not text:
                continue
            vec = self.embedder.embed(text)
            if not any(vec):
                continue

            for node_id, similarity in self.store.vector_search(vec, [scope_id], k=3):
                if similarity < threshold:
                    break
                if node_id in seen:
                    continue
                node = self.store.get_node(node_id)
                if node is None or node.state not in ("active", "contested"):
                    continue
                seen.add(node_id)
                extra = [episode.episode_id] if episode.episode_id else []
                self.store.update_node(
                    node.id,
                    strength=min(3.0, node.strength + 0.15),
                    activations=node.activations + 1,
                    source_ids=list(dict.fromkeys(node.source_ids + extra))[:64],
                )
                stats.nodes_reinforced += 1
                reinforced += 1

        # Entities named by a redundant unit are still corroborated, and
        # valence still accrues from its outcomes.
        self._apply_valence(
            NullDistiller().distill(unit).get("valence", []), {}, scope_id
        )
        return reinforced

    # -- 8.6 reconcile -----------------------------------------------------

    def reconcile(self, new_nodes: list[Node], scope_id: str, stats: ConsolidationStats) -> None:
        for node in new_nodes:
            if node.memory_class == "episodic":
                continue  # episodes are events; they cannot contradict
            vec = self.store.get_embedding(node.id)
            if vec is None:
                continue

            for cand_id, sim in self.store.vector_search(vec, [scope_id], k=6):
                if cand_id == node.id or sim < 0.75:
                    continue
                other = self.store.get_node(cand_id)
                if other is None or other.state not in ("active", "contested"):
                    continue
                if other.type != node.type:
                    continue
                if not _looks_contradictory(node.body, other.body):
                    continue

                stats.contradictions += 1
                self._edge(scope_id, node.id, other.id, "contradicts", 0.5,
                           "inferred", 0.5)

                # Humans win, unconditionally.
                if other.provenance == "asserted":
                    self.store.update_node(node.id, state="superseded",
                                           superseded_by=other.id)
                    break

                if _evidence_rank(node) > _evidence_rank(other):
                    self.store.update_node(other.id, state="superseded",
                                           superseded_by=node.id)
                else:
                    # Comparable evidence: flag both rather than picking. An
                    # agent told "these conflict, here's both" behaves better
                    # than one told a confident falsehood.
                    self.store.update_node(node.id, state="contested")
                    self.store.update_node(other.id, state="contested")
                break

    # -- 8.7 abstract ------------------------------------------------------

    def abstract(self, scope_id: str, stats: ConsolidationStats,
                 summarizer: Callable[[list[Node]], tuple[str, str]] | None = None) -> int:
        edges = self.store.neighbors_undirected([scope_id])
        if len(edges) < 4:
            return 0

        partition = louvain(edges, seed=0)
        if not partition:
            return 0

        previous: dict[str, list[str]] = defaultdict(list)
        for node in self.store.iter_nodes([scope_id], states=("active", "contested")):
            if node.community_id:
                previous[node.community_id].append(node.id)

        mapping = match_communities(
            partition, dict(previous),
            jaccard_threshold=self.config.consolidation.community_match_jaccard,
        )

        groups: dict[int, list[str]] = defaultdict(list)
        for node_id, cid in partition.items():
            groups[cid].append(node_id)

        centrality = _approx_betweenness(edges)
        n_communities = 0

        for cid, member_ids in groups.items():
            if len(member_ids) < 3:
                continue
            community_id = mapping.get(cid) or new_id("cm")
            members = [
                n for n in (self.store.get_node(m) for m in member_ids)
                if n is not None and n.state in ("active", "contested")
            ]
            if not members:
                continue

            for node in members:
                self.store.update_node(
                    node.id, community_id=community_id,
                    centrality=centrality.get(node.id, 0.0),
                )

            if summarizer is not None:
                label, summary = summarizer(members)
            else:
                # No-LLM path: name the community after its most central
                # member rather than inventing prose.
                hub = max(members, key=lambda n: centrality.get(n.id, 0.0))
                label = hub.label[:60]
                summary = (
                    f"{len(members)} related memories centred on {hub.label[:60]}."
                )

            coh = cohesion(edges, member_ids)
            self.store.upsert_community(
                community_id, scope_id, label=label, summary=summary,
                cohesion=coh, node_count=len(members),
                prev_id=mapping.get(cid),
            )
            self._upsert_theme_node(community_id, scope_id, label, summary, members)
            n_communities += 1

        stats.communities = n_communities
        self.store.set_meta(f"last_abstract:{scope_id}", str(now_ms()))
        self.store.set_meta(
            f"modularity:{scope_id}", f"{modularity(edges, partition):.4f}"
        )
        return n_communities

    def _upsert_theme_node(
        self, community_id: str, scope_id: str, label: str, summary: str,
        members: list[Node],
    ) -> None:
        row = self.store.conn.execute(
            "SELECT id FROM nodes WHERE community_id=? AND type='theme' LIMIT 1",
            (community_id,),
        ).fetchone()

        if row:
            self.store.update_node(row["id"], label=label, body=summary)
            theme_id = row["id"]
        else:
            theme = Node(
                id=new_id("thm"), scope_id=scope_id, type="theme",
                memory_class="semantic", label=label, body=summary,
                provenance="inferred", community_id=community_id,
            )
            self.store.add_node(theme, self.embedder.embed(f"{label} {summary}"))
            theme_id = theme.id

        for member in members[:32]:
            self._edge(scope_id, theme_id, member.id, "summarizes", 0.3,
                       "inferred", 0.5)

    # -- 8.7b scope guard --------------------------------------------------

    def validate_scopes(self, scope_id: str, stats: ConsolidationStats) -> None:
        """Check each asserted lesson's *scope* claim against episodic evidence.

        A reviewer who says "for service X, do Y" makes two claims: an **action**
        claim (Y is the right fix) and a **scope** claim (this holds across all of
        X). Only the second is checkable from data, and only the second has ever
        hurt us -- E10 measured random, systematic and adversarial reviewer error
        all landing at or above the no-review baseline, while over-generalisation
        alone drove performance *below* it (0.411 vs 0.456).

        E11 explains why. A wrong lesson sitting beside contradicting episodes is
        overruled -- by the scripted agent, and on Claude Opus 5 by a real model
        too (10/10 at two contradicting precedents). But at *zero* contradicting
        episodes both follow the lesson unanimously. An over-general lesson is
        dangerous precisely because it claims authority over contexts that have no
        local episodes to cross-check it.

        So this stage tests the uniformity the scope claim implies: if episodes
        inside the claimed scope split cleanly into succeeding and failing groups
        along some feature the lesson does not name, the scope is too coarse.

        Deliberately action-agnostic -- it never parses the lesson body, so it does
        not care what the recommended action is or how a domain spells it. It asks
        only whether the region the lesson claims is actually uniform, and its
        candidate splits are "linked to entity e / not linked to e", which needs no
        notion of a feature at all.

        Narrows rather than deletes: a rule right for two of six symptoms should
        become two rules, not zero. Discarding a correct action claim because the
        scope claim was wrong throws away the part the human got right.
        """
        cfg = self.config.consolidation
        if not cfg.scope_guard:
            return

        lessons = self.store.nodes_of(
            scope_id, memory_class="procedural", provenance="asserted"
        )
        if not lessons:
            return

        episodes = self.store.nodes_of(scope_id, node_type="episode")
        if len(episodes) < cfg.scope_min_evidence:
            return

        ep_entities = self.store.entity_neighbors([e.id for e in episodes], [scope_id])
        outcomes = {e.id: _episode_succeeded(e) for e in episodes}
        lesson_entities = self.store.entity_neighbors([n.id for n in lessons], [scope_id])

        for lesson in lessons:
            claimed = lesson_entities.get(lesson.id, set())
            if not claimed:
                continue

            in_scope = [
                e for e in episodes
                if outcomes[e.id] is not None
                and claimed <= ep_entities.get(e.id, set())
            ]
            if len(in_scope) < cfg.scope_min_evidence:
                continue  # innocent until proven over-general

            rate = sum(1 for e in in_scope if outcomes[e.id]) / len(in_scope)
            if rate >= cfg.scope_keep_success:
                continue  # the scope claim holds

            split = _best_entity_split(in_scope, ep_entities, outcomes, claimed)
            if (
                split is None
                or split[1] < cfg.scope_min_gain
                or split[2] < cfg.scope_keep_success
            ):
                # No clean sub-scope: the action looks wrong, not merely
                # over-broad. Demote rather than delete -- a human asserted it,
                # and `reverie why` should still explain what happened.
                self.store.update_node(lesson.id, state="contested")
                stats.scopes_demoted += 1
                continue

            entity_id = split[0]
            narrowed = Node(
                id=new_id("mem"), scope_id=scope_id, type=lesson.type,
                memory_class=lesson.memory_class, label=lesson.label,
                body=lesson.body, provenance=lesson.provenance,
                source_ids=list(lesson.source_ids or []),
            )
            self.store.add_node(narrowed, self.store.get_embedding(lesson.id))
            for ent in claimed | {entity_id}:
                self._edge(scope_id, ent, narrowed.id, "applies_to", 0.8, "inferred", 1.0)
            self.store.update_node(
                lesson.id, state="superseded", superseded_by=narrowed.id
            )
            stats.scopes_narrowed += 1

    # -- 8.8 prune ---------------------------------------------------------

    def prune(self, scope_id: str, stats: ConsolidationStats, now: int | None = None) -> None:
        now = now or now_ms()
        decay = self.config.decay

        for node in self.store.iter_nodes(
            [scope_id], states=("active", "contested", "stale")
        ):
            tau_days = decay.tau_days.get(node.memory_class, 60.0)
            age_days = max(0.0, (now - node.updated_at) / _DAY_MS)
            strength = node.strength * math.exp(-age_days / tau_days)

            # Valence decays too, negative more slowly than positive.
            tau_v = (
                decay.valence_tau_days_negative if node.valence < 0
                else decay.valence_tau_days_positive
            )
            valence = node.valence * math.exp(-age_days / tau_v)

            if (
                node.memory_class == "episodic"
                and strength < decay.strength_floor
                and not self._has_derived_edges(node.id)
            ):
                self.store.delete_node(node.id, reason="decayed")
                stats.pruned += 1
                continue

            self.store.update_node(node.id, strength=strength, valence=valence)

    def _has_derived_edges(self, node_id: str) -> bool:
        r = self.store.conn.execute(
            """SELECT 1 FROM edges e JOIN nodes n ON n.id = e.src
               WHERE e.dst = ? AND n.memory_class IN ('semantic','procedural')
               AND n.state = 'active' LIMIT 1""",
            (node_id,),
        ).fetchone()
        return r is not None

    # -- orchestration -----------------------------------------------------

    def run(
        self, scope_id: str, *, force_abstract: bool = False,
        summarizer: Callable[[list[Node]], tuple[str, str]] | None = None,
    ) -> ConsolidationStats:
        stats = ConsolidationStats()
        cfg = self.config.consolidation
        self._valence_spent = {}

        with self.store.consolidation_lock(scope_id):
            units = self.segment(scope_id, cfg.max_units_per_cycle)
            stats.units_seen = len(units)

            for unit in units:
                unit.salience = self.score(unit, scope_id)

                low_salience = unit.salience < cfg.salience_threshold
                if low_salience:
                    # Salience gates *distillation*, never episodic retention.
                    #
                    # HLD 8.3 introduces salience as the lever on consolidation
                    # *cost* -- "how much distillation budget it gets" -- and
                    # distillation is the only stage that spends model tokens.
                    # Skipping the whole link stage as well meant low-salience
                    # experience was never recorded at all, and that was
                    # catastrophic rather than merely lossy.
                    #
                    # The reason is a granularity mismatch. Salience is scored
                    # per *unit*, and a unit is everything buffered since the
                    # last cycle. Novelty is cosine distance over the unit's
                    # concatenated text, so ten on-call incidents covering ten
                    # service x symptom pairs *never seen before* still look
                    # near-identical in aggregate to the previous ten. Novelty
                    # collapsed, salience fell under threshold, and every batch
                    # after the first was archived.
                    #
                    # Measured before this change: 200 episodes ingested and
                    # consolidated produced a graph pinned at 10 episode nodes,
                    # with `nodes_created=0` on every cycle after the first.
                    # That single fact accounts for the flat learning curves in
                    # E6, the 13% exact-precedent retrieval, and the zero
                    # attribution lift -- retrieval was searching a graph that
                    # did not contain the experience.
                    #
                    # Episodic volume is bounded by decay and pruning (8.8),
                    # which is the mechanism designed for it.
                    stats.units_archived += 1

                candidates = (
                    NullDistiller().distill(unit)
                    if low_salience
                    else self.distiller.distill(unit)
                )
                created = self.link(unit, candidates, scope_id, stats)
                self.reconcile(created, scope_id, stats)

                # Commit per unit, not per cycle: an interrupted run loses at
                # most one unit, and ingest gets gaps to write in.
                self.store.mark_consolidated(unit.buffer_ids)
                stats.episodes_consolidated += len(unit.buffer_ids)
                # Counts units that actually spent model tokens, so it stays a
                # cost metric. Archived units run the free structured path and
                # are counted by `units_archived` instead.
                if not low_salience:
                    stats.units_distilled += 1

            self.validate_scopes(scope_id, stats)

            if force_abstract or self._should_abstract(scope_id):
                self.abstract(scope_id, stats, summarizer)

            self.prune(scope_id, stats)

        return stats

    def _should_abstract(self, scope_id: str) -> bool:
        cfg = self.config.consolidation
        cycles = int(self.store.get_meta(f"cycles:{scope_id}") or 0) + 1
        self.store.set_meta(f"cycles:{scope_id}", str(cycles))

        if cycles % cfg.abstract_every_n_cycles == 0:
            return True

        last_count = int(self.store.get_meta(f"abstract_nodes:{scope_id}") or 0)
        current = self.store.count_nodes(scope_id)
        if last_count and current > last_count * (1 + cfg.abstract_on_growth_pct / 100):
            self.store.set_meta(f"abstract_nodes:{scope_id}", str(current))
            return True
        if not last_count:
            self.store.set_meta(f"abstract_nodes:{scope_id}", str(current))
        return False


# -- helpers ---------------------------------------------------------------

_VAGUE = (
    "things go wrong", "be careful", "something", "anything", "as needed",
    "when appropriate", "if necessary", "in general", "sometimes",
)

_IMPERATIVE_MARKERS = (
    "ignore previous", "ignore all previous", "disregard", "you must",
    "you should always", "from now on", "new instructions", "system:",
    "override", "instead of what", "do not tell", "reveal your",
)


def _is_vague(condition: str) -> bool:
    low = condition.lower()
    return any(v in low for v in _VAGUE)


def _is_imperative_to_agent(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _IMPERATIVE_MARKERS)


def _episodic_text(ep: Episode) -> tuple[str, str]:
    """The canonical (label, body) for an episodic node.

    Shared by `link` and `reinforce_only` so the text a node is embedded from
    and the text used to find it again cannot drift apart.
    """
    label = ep.task.description[:120] or f"task {ep.task.id}"
    status = ep.outcome.status if ep.outcome else "unreported"
    body = f"{label} — {status}"
    if ep.outcome and ep.outcome.detail:
        body += f": {ep.outcome.detail}"
    return label, body


def _candidate_text(kind: str, cand: dict[str, Any]) -> tuple[str, str]:
    if kind == "procedural":
        condition = cand.get("condition", "")
        action = cand.get("action", "")
        rationale = cand.get("rationale", "")
        label = f"When {condition}, {action}"[:120]
        body = label if not rationale else f"{label}. {rationale}"
        return label, body
    if kind == "failure_modes":
        label = (cand.get("label") or cand.get("symptom") or "")[:120]
        parts = [cand.get("symptom", ""), cand.get("root_cause", ""),
                 cand.get("resolution", "")]
        body = " — ".join(p for p in parts if p) or label
        return label, body
    label = (cand.get("label") or "")[:120]
    return label, cand.get("body", label)


def _evidence_rank(node: Node) -> tuple[int, int, float]:
    provenance_rank = {"asserted": 3, "observed": 2, "inferred": 1, "ambiguous": 0}
    return (
        provenance_rank.get(node.provenance, 0),
        len(node.source_ids),
        node.updated_at,
    )


_NEGATIONS = ("not ", "never ", "no ", "cannot ", "can't ", "doesn't ", "does not ",
              "won't ", "avoid ", "don't ")


def _looks_contradictory(a: str, b: str) -> bool:
    """Cheap polarity check.

    Two texts that are highly similar but differ in negation polarity are
    likely to conflict. Crude on purpose -- this only *flags* a conflict, and
    the resolution path (contest both, surface both) is safe when wrong.
    """
    la, lb = a.lower(), b.lower()
    neg_a = sum(1 for n in _NEGATIONS if n in la)
    neg_b = sum(1 for n in _NEGATIONS if n in lb)
    return (neg_a > 0) != (neg_b > 0)


def _approx_betweenness(edges: Sequence[tuple[str, str, float]]) -> dict[str, float]:
    """Degree-normalised centrality proxy.

    True betweenness is O(V*E) and runs over the whole scope; this stands in
    until a measurement shows the difference matters for ranking.
    """
    degree: dict[str, float] = defaultdict(float)
    for src, dst, w in edges:
        degree[src] += w
        degree[dst] += w
    if not degree:
        return {}
    peak = max(degree.values()) or 1.0
    return {n: d / peak for n, d in degree.items()}


def _episode_succeeded(node: Node) -> bool | None:
    """Outcome of an episodic node, or None when it cannot be determined.

    Reads the canonical body written by `_episodic_text` ("<label> - <status>").
    That couples to Reverie's own writer, not to any domain: the same module
    produces the text and reads it back.
    """
    body = (node.body or "").lower()
    if "\u2014 success" in body or "- success" in body:
        return True
    if "\u2014 failure" in body or "- failure" in body:
        return False
    return None


def _entropy(pos: int, total: int) -> float:
    if total == 0 or pos == 0 or pos == total:
        return 0.0
    p = pos / total
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _best_entity_split(
    in_scope: list[Node],
    ep_entities: dict[str, set[str]],
    outcomes: dict[str, "bool | None"],
    claimed: set[str],
) -> "tuple[str, float, float] | None":
    """Highest-information-gain binary split of `in_scope` on entity presence.

    Returns (entity_id, information_gain, success_rate_of_the_has_entity_side).

    A binary "links to e / does not" split is used rather than a multi-way split
    over feature values because it needs no notion of a feature: entities are
    opaque ids here, so the same code works whatever a domain calls things.
    """
    total = len(in_scope)
    base = _entropy(sum(1 for e in in_scope if outcomes[e.id]), total)

    candidates: set[str] = set()
    for e in in_scope:
        candidates |= ep_entities.get(e.id, set())
    candidates -= claimed

    best = None
    for ent in candidates:
        has = [e for e in in_scope if ent in ep_entities.get(e.id, set())]
        lacks = [e for e in in_scope if ent not in ep_entities.get(e.id, set())]
        if not has or not lacks:
            continue  # constant within scope: splits nothing
        h_pos = sum(1 for e in has if outcomes[e.id])
        l_pos = sum(1 for e in lacks if outcomes[e.id])
        weighted = (
            len(has) / total * _entropy(h_pos, len(has))
            + len(lacks) / total * _entropy(l_pos, len(lacks))
        )
        gain = base - weighted
        if best is None or gain > best[1]:
            best = (ent, gain, h_pos / len(has))
    return best
