"""The `Reverie` facade -- the 20-line integration promised by HLD G1."""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .attribution import AttributionEngine, AttributionStats
from .config import Config
from .consolidate import Consolidator, ConsolidationStats, Distiller
from .embed import get_embedder
from .ingest import IngestResult, Ingestor
from .models import (
    AgentRef, Episode, Node, OutcomeReport, RecallResult, Step, TaskRef, new_id,
)
from .recall import RecallEngine
from .store import SQLiteStore

__all__ = ["Reverie", "EpisodeRecorder"]

DEFAULT_DB = "~/.reverie/memory.db"


class EpisodeRecorder:
    """Captures an episode as the agent works.

    The context manager exists to make the correct integration the easiest
    one: it captures duration, exceptions, and the recall linkage without the
    agent author thinking about any of them.
    """

    def __init__(
        self, reverie: Reverie, task: str, *, task_type: str = "unknown",
        recall_id: str | None = None, task_id: str | None = None,
    ) -> None:
        self._reverie = reverie
        self._episode = Episode(
            scope_id=reverie.scope,
            task=TaskRef(
                id=task_id or new_id("task"), description=task,
                type=task_type, recall_id=recall_id,
            ),
            agent=reverie.agent,
        )
        self._started = 0.0
        self.result: IngestResult | None = None

    def step(
        self, kind: str, input: str = "", *, name: str = "",
        output: str = "", error: str | None = None, exit_code: int | None = None,
    ) -> EpisodeRecorder:
        self._episode.steps.append(
            Step(kind=kind, name=name or kind, input=input,
                 output_summary=output, error=error, exit_code=exit_code)
        )
        return self

    def entity(self, *labels: str) -> EpisodeRecorder:
        for label in labels:
            if label not in self._episode.entities:
                self._episode.entities.append(label)
        return self

    def used(self, *node_ids: str) -> EpisodeRecorder:
        """Declare which recalled memories actually influenced the work.

        The single highest-value field an integrator can populate: it turns
        attribution from correlational into near-exact (HLD 9.4).
        """
        for nid in node_ids:
            if nid not in self._episode.used_memories:
                self._episode.used_memories.append(nid)
        return self

    def outcome(
        self, status: str, *, tier: int = 1, evidence: str = "", detail: str = ""
    ) -> EpisodeRecorder:
        self._episode.outcome = OutcomeReport(
            status=status, signal_tier=tier, evidence=evidence, detail=detail
        )
        return self

    def __enter__(self) -> EpisodeRecorder:
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: Any) -> bool:
        self._episode.duration_ms = int((time.perf_counter() - self._started) * 1000)

        if exc_type is not None and self._episode.outcome is None:
            # An exception escaping the block is a tier-1 failure signal, and
            # losing the episode because the agent crashed is the worst
            # possible time to lose one.
            self._episode.outcome = OutcomeReport(
                status="failure", signal_tier=1,
                evidence=f"{exc_type.__name__}: {exc}",
                detail="".join(traceback.format_exception_only(exc_type, exc)).strip(),
            )

        self.result = self._reverie.remember(self._episode)
        return False  # never suppress


class Reverie:
    def __init__(
        self,
        scope: str = "agent:default",
        *,
        db: str | Path = DEFAULT_DB,
        config: Config | None = None,
        distiller: Distiller | None = None,
        agent: AgentRef | None = None,
        scope_chain: Sequence[str] | None = None,
    ) -> None:
        self.scope = scope
        self.config = config or Config()
        self.agent = agent or AgentRef()
        self.scope_chain = list(scope_chain) if scope_chain else [scope]

        self.store = SQLiteStore(
            db,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
        )
        self.embedder = get_embedder(self.config.embedding_model)
        self.ingestor = Ingestor(self.store, self.config.ingest)
        self.recaller = RecallEngine(self.store, self.config, self.embedder)
        self.attributor = AttributionEngine(
            self.store, self.config.attribution, self.config.decay
        )
        self.consolidator = Consolidator(
            self.store, self.config, distiller, self.embedder
        )

    # -- hot path ----------------------------------------------------------

    def recall(
        self, cue: str, *, budget_tokens: int | None = None,
        context: dict[str, str] | None = None,
    ) -> RecallResult:
        return self.recaller.recall(
            cue, self.scope, scope_chain=self.scope_chain,
            budget_tokens=budget_tokens, context=context,
        )

    def remember(self, episode: Episode) -> IngestResult:
        return self.ingestor.remember(episode)

    def episode(
        self, task: str, *, task_type: str = "unknown",
        recall_id: str | None = None, task_id: str | None = None,
    ) -> EpisodeRecorder:
        return EpisodeRecorder(
            self, task, task_type=task_type, recall_id=recall_id, task_id=task_id
        )

    def report_outcome(
        self, recall_id: str, outcome: str, *, tier: int = 1,
        evidence: str = "", idem_key: str | None = None,
    ) -> str:
        oid, _ = self.ingestor.report_outcome(
            self.scope, recall_id, outcome, tier,
            evidence=evidence, idem_key=idem_key,
        )
        return oid

    # -- background --------------------------------------------------------

    def consolidate(self, **kwargs: Any) -> ConsolidationStats:
        return self.consolidator.run(self.scope, **kwargs)

    def attribute(self, limit: int = 1000) -> AttributionStats:
        return self.attributor.run(limit)

    def dream(self, **kwargs: Any) -> tuple[ConsolidationStats, AttributionStats]:
        """One full offline cycle: attribute, then consolidate.

        Attribution runs first so that consolidation's prune and quarantine
        stages see the freshest utility posteriors.
        """
        attribution = self.attribute()
        consolidation = self.consolidate(**kwargs)
        return consolidation, attribution

    # -- inspection --------------------------------------------------------

    def inspect(self, node_id: str) -> dict[str, Any]:
        node = self.store.get_node(node_id)
        if node is None:
            return {}
        edges_out = self.store.edges_from([node_id], self.scope_chain, limit_per_node=50)
        return {
            "node": node,
            "utility": self.attributor.utility_report(node_id),
            "edges": edges_out.get(node_id, []),
            "community": (
                self.store.get_community(node.community_id) if node.community_id else None
            ),
        }

    def why(self, node_id: str) -> list[dict[str, Any]]:
        """Trace a memory back to the raw episodes that produced it."""
        node = self.store.get_node(node_id)
        if node is None:
            return []
        out: list[dict[str, Any]] = []
        for source in node.source_ids:
            row = self.store.conn.execute(
                "SELECT payload, created_at FROM episode_buffer WHERE id=?", (source,)
            ).fetchone()
            if row:
                out.append({"episode_id": source, "payload": row["payload"],
                            "created_at": row["created_at"]})
            else:
                out.append({"episode_id": source, "payload": None,
                            "note": "episode aged out of the buffer"})
        return out

    def path(self, src_label: str, dst_label: str, max_hops: int = 6) -> list[Node]:
        """Shortest association path between two memories.

        The `car -> my old Civic -> transmission failure -> avoid` demo, on
        the command line (HLD 10.3, 11.4).
        """
        src = self.store.find_by_label(src_label, self.scope_chain)
        dst = self.store.find_by_label(dst_label, self.scope_chain)
        if src is None or dst is None:
            return []
        if src.id == dst.id:
            return [src]

        # BFS over the edge table. Unweighted: "how are these connected" wants
        # the shortest chain, not the strongest one.
        frontier = [src.id]
        parents: dict[str, str | None] = {src.id: None}
        for _ in range(max_hops):
            if not frontier:
                break
            edge_map = self.store.edges_from(frontier, self.scope_chain, limit_per_node=64)
            nxt: list[str] = []
            for node_id in frontier:
                for edge in edge_map.get(node_id, ()):
                    if edge.dst in parents:
                        continue
                    parents[edge.dst] = node_id
                    if edge.dst == dst.id:
                        return self._reconstruct(parents, dst.id)
                    nxt.append(edge.dst)
            frontier = nxt
        return []

    def _reconstruct(self, parents: dict[str, str | None], end: str) -> list[Node]:
        chain: list[str] = []
        cur: str | None = end
        while cur is not None:
            chain.append(cur)
            cur = parents.get(cur)
        chain.reverse()
        nodes = self.store.get_nodes(chain)
        return [nodes[c] for c in chain if c in nodes]

    def forget(self, node_id: str, reason: str = "") -> None:
        self.store.delete_node(node_id, reason)

    def assert_fact(
        self, body: str, *, entities: Sequence[str] = (), label: str | None = None,
        memory_class: str = "semantic",
    ) -> str:
        """Human-authored memory. Wins conflicts (HLD 8.6).

        `memory_class` matters more than it looks. A conditioned action ("when
        X, do Y") filed as `semantic` competes in the wrong recall quota and is
        excluded from the machinery that treats procedures as reinforceable.
        Human review produces mostly *procedural* memory, so callers writing
        lessons should say so.
        """
        node = Node(
            id=new_id("mem"), scope_id=self.scope, type="lesson",
            memory_class=memory_class, label=(label or body)[:120], body=body,
            provenance="asserted",
        )
        self.store.add_node(node, self.embedder.embed(body))
        for label_ in entities:
            entity = self.consolidator._resolve_entity(label_, self.scope)
            self.consolidator._edge(
                self.scope, entity.id, node.id, "applies_to", 0.8, "asserted", 1.0
            )
        return node.id

    def stats(self) -> dict[str, Any]:
        conn = self.store.conn
        by_state = dict(
            conn.execute(
                "SELECT state, COUNT(*) FROM nodes WHERE scope_id=? GROUP BY state",
                (self.scope,),
            ).fetchall()
        )
        by_class = dict(
            conn.execute(
                "SELECT memory_class, COUNT(*) FROM nodes WHERE scope_id=? "
                "AND state='active' GROUP BY memory_class",
                (self.scope,),
            ).fetchall()
        )
        n_edges = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE scope_id=?", (self.scope,)
        ).fetchone()[0]
        n_recalls = conn.execute(
            "SELECT COUNT(*) FROM recalls WHERE scope_id=?", (self.scope,)
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM outcomes WHERE scope_id=? AND attributed=0", (self.scope,)
        ).fetchone()[0]

        return {
            "scope": self.scope,
            "nodes_by_state": by_state,
            "nodes_by_class": by_class,
            "edges": n_edges,
            "recalls": n_recalls,
            "buffer_depth": self.store.buffer_depth(self.scope),
            "pending_attributions": pending,
            "communities": self.store.list_communities(self.scope),
            "modularity": self.store.get_meta(f"modularity:{self.scope}"),
            "ablation": self.attributor.ablation_summary(self.scope),
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Reverie:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
