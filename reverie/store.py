"""SQLite storage adapter.

The graph lives in plain relational tables and traversal happens in application
code (HLD 6.2). Concurrency follows HLD 14.1: WAL, many readers, one writer,
short busy-timeout retries on the write paths.
"""

from __future__ import annotations

import array
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import schema
from .models import Edge, Node, new_id, now_ms

__all__ = ["SQLiteStore", "LockBusy"]

_BUSY_TIMEOUT_MS = 5_000
_MAX_RETRIES = 5


class LockBusy(RuntimeError):
    """Another process holds the consolidation lock for this scope."""


def _pack(vec: Sequence[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> array.array:
    a = array.array("f")
    a.frombytes(blob)
    return a


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _row_to_node(r: sqlite3.Row) -> Node:
    return Node(
        id=r["id"],
        scope_id=r["scope_id"],
        type=r["type"],
        memory_class=r["memory_class"],
        label=r["label"],
        body=r["body"],
        valence=r["valence"],
        provenance=r["provenance"],
        source_ids=json.loads(r["source_ids"]),
        state=r["state"],
        superseded_by=r["superseded_by"],
        merged_into=r["merged_into"],
        learned_under=json.loads(r["learned_under"]),
        community_id=r["community_id"],
        centrality=r["centrality"],
        alpha=r["alpha"],
        beta=r["beta"],
        attributions=r["attributions"],
        utility_at=r["utility_at"],
        strength=r["strength"],
        activations=r["activations"],
        last_activated=r["last_activated"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def _row_to_edge(r: sqlite3.Row) -> Edge:
    return Edge(
        id=r["id"],
        scope_id=r["scope_id"],
        src=r["src"],
        dst=r["dst"],
        relation=r["relation"],
        weight=r["weight"],
        provenance=r["provenance"],
        confidence=r["confidence"],
        traversals=r["traversals"],
        last_traversed=r["last_traversed"],
        created_at=r["created_at"],
    )


class SQLiteStore:
    """The default `MemoryStore`. One file, no daemon."""

    def __init__(
        self,
        path: str | Path = "~/.reverie/memory.db",
        *,
        embedding_model: str = "hash-64",
        embedding_dim: int = 64,
        check_same_thread: bool = True,
    ) -> None:
        if str(path) == ":memory:":
            self.path = ":memory:"
        else:
            self.path = str(Path(path).expanduser().resolve())
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            self.path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            check_same_thread=check_same_thread,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        schema.migrate(self.conn)
        schema.check_embedding_model(self.conn, embedding_model, embedding_dim)
        self.embedding_dim = embedding_dim

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A retrying write transaction (HLD 14.1)."""
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self.conn:
                    yield self.conn
                return
            except sqlite3.OperationalError as e:  # pragma: no cover - timing
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                last = e
                time.sleep(0.05 * (2**attempt))
        raise LockBusy(f"could not acquire write lock after {_MAX_RETRIES} attempts") from last

    @contextmanager
    def consolidation_lock(self, scope_id: str, ttl_ms: int = 3_600_000) -> Iterator[None]:
        """Advisory lock so two workers cannot consolidate one scope at once."""
        key = f"lock:consolidate:{scope_id}"
        now = now_ms()
        with self.write() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if row is not None and now - int(row[0]) < ttl_ms:
                raise LockBusy(f"consolidation already running for {scope_id}")
            c.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(now)),
            )
        try:
            yield
        finally:
            with self.write() as c:
                c.execute("DELETE FROM meta WHERE key=?", (key,))

    # -- nodes -------------------------------------------------------------

    def add_node(self, node: Node, embedding: Sequence[float] | None = None) -> str:
        if not node.id:
            node.id = new_id("mem")
        with self.write() as c:
            c.execute(
                """INSERT INTO nodes (
                     id, scope_id, type, memory_class, label, body, valence,
                     provenance, source_ids, state, superseded_by, merged_into,
                     learned_under, community_id, centrality, alpha, beta,
                     attributions, utility_at, strength, activations,
                     last_activated, created_at, updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    node.id, node.scope_id, node.type, node.memory_class,
                    node.label, node.body, node.valence, node.provenance,
                    json.dumps(node.source_ids), node.state, node.superseded_by,
                    node.merged_into, json.dumps(node.learned_under),
                    node.community_id, node.centrality, node.alpha, node.beta,
                    node.attributions, node.utility_at, node.strength,
                    node.activations, node.last_activated, node.created_at,
                    node.updated_at,
                ),
            )
            c.execute(
                "INSERT INTO node_fts (node_id, scope_id, label, body) VALUES (?,?,?,?)",
                (node.id, node.scope_id, node.label, node.body),
            )
            if embedding is not None:
                c.execute(
                    "INSERT OR REPLACE INTO node_vec (node_id, scope_id, embedding) VALUES (?,?,?)",
                    (node.id, node.scope_id, _pack(embedding)),
                )
        return node.id

    def get_node(self, node_id: str) -> Node | None:
        r = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return _row_to_node(r) if r else None

    def get_nodes(self, node_ids: Iterable[str]) -> dict[str, Node]:
        ids = list(node_ids)
        if not ids:
            return {}
        out: dict[str, Node] = {}
        for i in range(0, len(ids), 900):  # SQLite variable limit
            chunk = ids[i : i + 900]
            q = f"SELECT * FROM nodes WHERE id IN ({','.join('?' * len(chunk))})"
            for r in self.conn.execute(q, chunk):
                out[r["id"]] = _row_to_node(r)
        return out

    def update_node(self, node_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields.setdefault("updated_at", now_ms())
        encoded = {
            k: (json.dumps(v) if k in ("source_ids", "learned_under") else v)
            for k, v in fields.items()
        }
        sets = ", ".join(f"{k}=?" for k in encoded)
        with self.write() as c:
            c.execute(
                f"UPDATE nodes SET {sets} WHERE id=?", (*encoded.values(), node_id)
            )
            if "label" in fields or "body" in fields:
                n = self.get_node(node_id)
                if n:
                    c.execute("DELETE FROM node_fts WHERE node_id=?", (node_id,))
                    c.execute(
                        "INSERT INTO node_fts (node_id, scope_id, label, body) VALUES (?,?,?,?)",
                        (n.id, n.scope_id, n.label, n.body),
                    )

    def iter_nodes(
        self, scope_ids: Sequence[str], states: Sequence[str] = ("active",)
    ) -> Iterator[Node]:
        q = (
            f"SELECT * FROM nodes WHERE scope_id IN ({','.join('?' * len(scope_ids))}) "
            f"AND state IN ({','.join('?' * len(states))})"
        )
        for r in self.conn.execute(q, (*scope_ids, *states)):
            yield _row_to_node(r)

    def count_nodes(self, scope_id: str, state: str = "active") -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE scope_id=? AND state=?", (scope_id, state)
        ).fetchone()
        return int(r[0])

    def resolve_merges(self, node_id: str, _depth: int = 0) -> str | None:
        """Follow ``merged_into`` transitively (HLD 9.5).

        Returns the surviving node id, or None if the chain ends in a deleted
        node. The depth guard makes a cyclic chain -- which a buggy merge pass
        could create -- fail closed instead of hanging attribution.
        """
        seen: set[str] = set()
        cur = node_id
        while True:
            if cur in seen or len(seen) > 32:
                return None  # cycle or pathological chain
            seen.add(cur)
            r = self.conn.execute(
                "SELECT state, merged_into FROM nodes WHERE id=?", (cur,)
            ).fetchone()
            if r is None:
                return None
            if r["state"] == "deleted":
                return None
            if r["state"] == "merged" and r["merged_into"]:
                cur = r["merged_into"]
                continue
            return cur

    def merge_node(self, src_id: str, dst_id: str) -> None:
        """Tombstone ``src`` into ``dst``. Never deletes -- live recall logs
        still reference the absorbed node (HLD 8.5, 9.5)."""
        with self.write() as c:
            c.execute(
                "UPDATE nodes SET state='merged', merged_into=?, updated_at=? WHERE id=?",
                (dst_id, now_ms(), src_id),
            )
            c.execute("DELETE FROM node_fts WHERE node_id=?", (src_id,))
            c.execute("DELETE FROM node_vec WHERE node_id=?", (src_id,))

    def delete_node(self, node_id: str, reason: str = "") -> None:
        """Human `forget`. Cascades through every index (HLD 13.3)."""
        with self.write() as c:
            c.execute(
                "UPDATE nodes SET state='deleted', body='', label='[deleted]', "
                "source_ids='[]', updated_at=? WHERE id=?",
                (now_ms(), node_id),
            )
            c.execute("DELETE FROM node_fts WHERE node_id=?", (node_id,))
            c.execute("DELETE FROM node_vec WHERE node_id=?", (node_id,))
            c.execute("DELETE FROM edges WHERE src=? OR dst=?", (node_id, node_id))
            c.execute("DELETE FROM aliases WHERE node_id=?", (node_id,))

    # -- edges -------------------------------------------------------------

    def add_edge(self, edge: Edge) -> str:
        if not edge.id:
            edge.id = new_id("e")
        with self.write() as c:
            c.execute(
                """INSERT INTO edges (id, scope_id, src, dst, relation, weight,
                                      provenance, confidence, traversals,
                                      last_traversed, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(scope_id, src, dst, relation) DO UPDATE SET
                     weight = MAX(edges.weight, excluded.weight),
                     confidence = MAX(edges.confidence, excluded.confidence)""",
                (
                    edge.id, edge.scope_id, edge.src, edge.dst, edge.relation,
                    edge.weight, edge.provenance, edge.confidence,
                    edge.traversals, edge.last_traversed, edge.created_at,
                ),
            )
        return edge.id

    def edges_from(
        self, node_ids: Sequence[str], scope_ids: Sequence[str], limit_per_node: int = 12
    ) -> dict[str, list[Edge]]:
        """One query per *frontier*, not per node (HLD 10.2)."""
        if not node_ids:
            return {}
        out: dict[str, list[Edge]] = {}
        for i in range(0, len(node_ids), 900):
            chunk = list(node_ids)[i : i + 900]
            q = (
                f"SELECT * FROM edges WHERE scope_id IN ({','.join('?' * len(scope_ids))}) "
                f"AND src IN ({','.join('?' * len(chunk))}) "
                f"ORDER BY src, weight DESC"
            )
            for r in self.conn.execute(q, (*scope_ids, *chunk)):
                bucket = out.setdefault(r["src"], [])
                if len(bucket) < limit_per_node:
                    bucket.append(_row_to_edge(r))
        return out

    def neighbors_undirected(self, scope_ids: Sequence[str]) -> list[tuple[str, str, float]]:
        q = (
            f"SELECT src, dst, weight FROM edges "
            f"WHERE scope_id IN ({','.join('?' * len(scope_ids))})"
        )
        return [(r["src"], r["dst"], r["weight"]) for r in self.conn.execute(q, scope_ids)]

    # -- search ------------------------------------------------------------

    def fts_search(self, cue: str, scope_ids: Sequence[str], k: int = 20) -> list[tuple[str, float]]:
        tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in cue).split() if len(t) > 1]
        if not tokens:
            return []
        query = " OR ".join(tokens)
        try:
            rows = self.conn.execute(
                f"SELECT node_id, bm25(node_fts) AS score FROM node_fts "
                f"WHERE node_fts MATCH ? AND scope_id IN ({','.join('?' * len(scope_ids))}) "
                f"ORDER BY score LIMIT ?",
                (query, *scope_ids, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() returns negative numbers; more negative is better.
        return [(r["node_id"], -r["score"]) for r in rows]

    def vector_search(
        self, embedding: Sequence[float], scope_ids: Sequence[str], k: int = 20
    ) -> list[tuple[str, float]]:
        """Brute-force cosine.

        Exact and dependency-free. Linear in node count, which is fine to the
        low hundreds of thousands; the Postgres adapter swaps in pgvector.
        """
        q = (
            f"SELECT v.node_id, v.embedding FROM node_vec v "
            f"JOIN nodes n ON n.id = v.node_id "
            f"WHERE v.scope_id IN ({','.join('?' * len(scope_ids))}) "
            f"AND n.state IN ('active','contested','stale')"
        )
        scored = [
            (r["node_id"], _cosine(embedding, _unpack(r["embedding"])))
            for r in self.conn.execute(q, scope_ids)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def entity_neighbors(
        self, node_ids: Sequence[str], scope_ids: Sequence[str]
    ) -> dict[str, set[str]]:
        """node id -> the set of entity nodes it is linked to (either direction)."""
        if not node_ids or not scope_ids:
            return {}
        sp = ",".join("?" * len(scope_ids))
        np_ = ",".join("?" * len(node_ids))
        sql = (
            f"SELECT n, other FROM ("
            f"  SELECT e.src AS n, e.dst AS other FROM edges e "
            f"    JOIN nodes o ON o.id = e.dst "
            f"   WHERE e.scope_id IN ({sp}) AND e.src IN ({np_}) AND o.type = 'entity' "
            f"  UNION "
            f"  SELECT e.dst AS n, e.src AS other FROM edges e "
            f"    JOIN nodes o ON o.id = e.src "
            f"   WHERE e.scope_id IN ({sp}) AND e.dst IN ({np_}) AND o.type = 'entity'"
            f")"
        )
        args = [*scope_ids, *node_ids, *scope_ids, *node_ids]
        out: dict[str, set[str]] = {}
        for r in self.conn.execute(sql, args):
            out.setdefault(r["n"], set()).add(r["other"])
        return out

    def nodes_of(
        self, scope_id: str, *, node_type: str | None = None,
        memory_class: str | None = None, provenance: str | None = None,
        states: Sequence[str] = ("active", "contested"),
    ) -> list[Node]:
        """Filtered node fetch, for stages that sweep a whole class of memory."""
        from .store import _row_to_node
        q = ["SELECT * FROM nodes WHERE scope_id = ?"]
        args: list[Any] = [scope_id]
        if node_type:
            q.append("AND type = ?"); args.append(node_type)
        if memory_class:
            q.append("AND memory_class = ?"); args.append(memory_class)
        if provenance:
            q.append("AND provenance = ?"); args.append(provenance)
        if states:
            q.append(f"AND state IN ({','.join('?' * len(states))})"); args.extend(states)
        return [_row_to_node(r) for r in self.conn.execute(" ".join(q), args)]

    def conjunctive_neighbors(
        self,
        entity_ids: Sequence[str],
        scope_ids: Sequence[str],
        limit_rows: int = 8000,
    ) -> dict[str, set[str]]:
        """Map each node adjacent to a cue entity -> the set of cue entities it touches.

        This is the raw material for conjunctive scoring. Spreading activation
        alone is disjunctive: it floods outward from each cue entity
        independently, so a node touching one entity is reached just as
        readily as a node touching all of them. Precedent lookup needs the
        opposite -- the *intersection*. Measured on the on-call benchmark,
        activation-only retrieval surfaced an exact (service, symptom)
        precedent in 13% of briefs against naive recency's 42%.

        Returned as sets rather than counts so the caller can apply per-entity
        IDF weights; a cue entity attached to half the graph carries far less
        evidence than a rare one.
        """
        if not entity_ids or not scope_ids:
            return {}
        sp = ",".join("?" * len(scope_ids))
        ep = ",".join("?" * len(entity_ids))
        sql = (
            f"SELECT other, ent FROM ("
            f"  SELECT dst AS other, src AS ent FROM edges "
            f"   WHERE scope_id IN ({sp}) AND src IN ({ep}) "
            f"  UNION "
            f"  SELECT src AS other, dst AS ent FROM edges "
            f"   WHERE scope_id IN ({sp}) AND dst IN ({ep})"
            f") LIMIT ?"
        )
        args = [*scope_ids, *entity_ids, *scope_ids, *entity_ids, limit_rows]
        skip = set(entity_ids)
        out: dict[str, set[str]] = {}
        for row in self.conn.execute(sql, args):
            other = row["other"]
            if other in skip:
                continue
            out.setdefault(other, set()).add(row["ent"])
        return out

    def node_degrees(
        self, node_ids: Sequence[str], scope_ids: Sequence[str]
    ) -> dict[str, int]:
        """Total edge degree per node, for hub penalisation."""
        if not node_ids or not scope_ids:
            return {}
        sp = ",".join("?" * len(scope_ids))
        np_ = ",".join("?" * len(node_ids))
        # Count DISTINCT entity neighbours, not edges. Counting edges
        # double-counts any relation stored in both directions, which
        # inflated an episode's 4 entities to a degree of 8 and halved
        # every containment ratio computed from it.
        #
        # Count only edges to *entity* nodes. Total degree is the wrong
        # denominator for specificity: an episode is linked to its entities
        # and also to its temporal neighbours on the `preceded` spine, so a
        # perfectly specific episode with 3 entities showed degree 14 and
        # scored 0.21 instead of 1.0 -- penalised for having a past and a
        # future. Specificity is about how much of the candidate's *entity*
        # footprint the cue accounts for, and nothing else.
        sql = (
            f"SELECT n, COUNT(*) AS deg FROM ("
            f"  SELECT DISTINCT e.src AS n, e.dst AS other FROM edges e "
            f"    JOIN nodes o ON o.id = e.dst "
            f"   WHERE e.scope_id IN ({sp}) AND e.src IN ({np_}) AND o.type = 'entity' "
            f"  UNION "
            f"  SELECT DISTINCT e.dst AS n, e.src AS other FROM edges e "
            f"    JOIN nodes o ON o.id = e.src "
            f"   WHERE e.scope_id IN ({sp}) AND e.dst IN ({np_}) AND o.type = 'entity'"
            f") GROUP BY n"
        )
        args = [*scope_ids, *node_ids, *scope_ids, *node_ids]
        return {r["n"]: r["deg"] for r in self.conn.execute(sql, args)}

    def entity_degrees(
        self, entity_ids: Sequence[str], scope_ids: Sequence[str]
    ) -> dict[str, int]:
        """Neighbour count per entity, for IDF weighting."""
        if not entity_ids or not scope_ids:
            return {}
        sp = ",".join("?" * len(scope_ids))
        ep = ",".join("?" * len(entity_ids))
        sql = (
            f"SELECT ent, COUNT(*) AS df FROM ("
            f"  SELECT src AS ent, dst AS other FROM edges "
            f"   WHERE scope_id IN ({sp}) AND src IN ({ep}) "
            f"  UNION "
            f"  SELECT dst AS ent, src AS other FROM edges "
            f"   WHERE scope_id IN ({sp}) AND dst IN ({ep})"
            f") GROUP BY ent"
        )
        args = [*scope_ids, *entity_ids, *scope_ids, *entity_ids]
        return {r["ent"]: r["df"] for r in self.conn.execute(sql, args)}

    def find_by_label(self, label: str, scope_ids: Sequence[str]) -> Node | None:
        q = (
            f"SELECT * FROM nodes WHERE scope_id IN ({','.join('?' * len(scope_ids))}) "
            f"AND lower(label)=lower(?) AND state NOT IN ('deleted','merged') LIMIT 1"
        )
        r = self.conn.execute(q, (*scope_ids, label)).fetchone()
        return _row_to_node(r) if r else None

    def find_by_alias(self, alias: str, scope_ids: Sequence[str]) -> Node | None:
        q = (
            f"SELECT node_id FROM aliases "
            f"WHERE scope_id IN ({','.join('?' * len(scope_ids))}) AND lower(alias)=lower(?)"
        )
        r = self.conn.execute(q, (*scope_ids, alias)).fetchone()
        if not r:
            return None
        resolved = self.resolve_merges(r["node_id"])
        return self.get_node(resolved) if resolved else None

    def add_alias(self, scope_id: str, alias: str, node_id: str) -> None:
        with self.write() as c:
            c.execute(
                "INSERT OR IGNORE INTO aliases (scope_id, alias, node_id) VALUES (?,?,?)",
                (scope_id, alias, node_id),
            )

    def set_embedding(self, node_id: str, scope_id: str, embedding: Sequence[float]) -> None:
        with self.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO node_vec (node_id, scope_id, embedding) VALUES (?,?,?)",
                (node_id, scope_id, _pack(embedding)),
            )

    def get_embedding(self, node_id: str) -> list[float] | None:
        r = self.conn.execute(
            "SELECT embedding FROM node_vec WHERE node_id=?", (node_id,)
        ).fetchone()
        return list(_unpack(r["embedding"])) if r else None

    # -- recall log --------------------------------------------------------

    def log_recall(
        self,
        scope_id: str,
        cue: str,
        node_ids: Sequence[str],
        activations: dict[str, float],
        *,
        withheld_ids: Sequence[str] = (),
        is_ablation: bool = False,
        token_cost: int = 0,
        latency_ms: float = 0.0,
    ) -> str:
        rid = new_id("rc")
        with self.write() as c:
            c.execute(
                """INSERT INTO recalls (id, scope_id, cue, node_ids, activations,
                                        withheld_ids, is_ablation, token_cost,
                                        latency_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    rid, scope_id, cue, json.dumps(list(node_ids)),
                    json.dumps(activations), json.dumps(list(withheld_ids)),
                    int(is_ablation), token_cost, latency_ms, now_ms(),
                ),
            )
        return rid

    def get_recall(self, recall_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM recalls WHERE id=?", (recall_id,)).fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "scope_id": r["scope_id"],
            "cue": r["cue"],
            "node_ids": json.loads(r["node_ids"]),
            "activations": json.loads(r["activations"]),
            "withheld_ids": json.loads(r["withheld_ids"]),
            "is_ablation": bool(r["is_ablation"]),
            "token_cost": r["token_cost"],
            "latency_ms": r["latency_ms"],
            "created_at": r["created_at"],
        }

    # -- outcomes ----------------------------------------------------------

    def add_outcome(
        self,
        scope_id: str,
        outcome: str,
        signal_tier: int,
        *,
        recall_id: str | None = None,
        episode_id: str | None = None,
        idem_key: str | None = None,
        detail: str = "",
    ) -> tuple[str, bool]:
        """Returns (outcome_id, was_new). Duplicate idem_keys are dropped."""
        key = idem_key or f"{recall_id}:{outcome}:{signal_tier}"
        existing = self.conn.execute(
            "SELECT id FROM outcomes WHERE idem_key=?", (key,)
        ).fetchone()
        if existing:
            return existing["id"], False
        oid = new_id("oc")
        with self.write() as c:
            c.execute(
                """INSERT INTO outcomes (id, scope_id, recall_id, episode_id, idem_key,
                                         signal_tier, outcome, detail, attributed, created_at)
                   VALUES (?,?,?,?,?,?,?,?,0,?)""",
                (oid, scope_id, recall_id, episode_id, key, signal_tier, outcome,
                 detail, now_ms()),
            )
        return oid, True

    def pending_outcomes(self, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM outcomes WHERE attributed=0 AND recall_id IS NOT NULL "
            "ORDER BY signal_tier ASC, created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def outcomes_for_recall(self, recall_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM outcomes WHERE recall_id=? ORDER BY created_at ASC",
            (recall_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_attributed(self, outcome_id: str) -> None:
        with self.write() as c:
            c.execute("UPDATE outcomes SET attributed=1 WHERE id=?", (outcome_id,))

    # -- attribution ledger -----------------------------------------------

    def record_attribution(
        self, outcome_id: str, node_id: str, d_alpha: float, d_beta: float
    ) -> None:
        with self.write() as c:
            c.execute(
                """INSERT INTO attribution_log (id, outcome_id, node_id, d_alpha,
                                                d_beta, reversed, created_at)
                   VALUES (?,?,?,?,?,0,?)""",
                (new_id("al"), outcome_id, node_id, d_alpha, d_beta, now_ms()),
            )

    def attributions_for_outcome(self, outcome_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM attribution_log WHERE outcome_id=? AND reversed=0",
            (outcome_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_reversed(self, log_id: str) -> None:
        with self.write() as c:
            c.execute("UPDATE attribution_log SET reversed=1 WHERE id=?", (log_id,))

    def apply_utility_delta(self, node_id: str, d_alpha: float, d_beta: float, d_attr: int = 1) -> None:
        with self.write() as c:
            c.execute(
                "UPDATE nodes SET alpha = MAX(1e-6, alpha + ?), "
                "beta = MAX(1e-6, beta + ?), attributions = MAX(0, attributions + ?), "
                "updated_at = ? WHERE id = ?",
                (d_alpha, d_beta, d_attr, now_ms(), node_id),
            )

    # -- episode buffer ----------------------------------------------------

    def buffer_episode(
        self, scope_id: str, payload: str, *, client_key: str | None, task_type: str
    ) -> tuple[str, bool]:
        if client_key:
            existing = self.conn.execute(
                "SELECT id FROM episode_buffer WHERE scope_id=? AND client_key=?",
                (scope_id, client_key),
            ).fetchone()
            if existing:
                return existing["id"], False
        eid = new_id("ep")
        with self.write() as c:
            c.execute(
                """INSERT INTO episode_buffer (id, scope_id, client_key, payload,
                                               task_type, consolidated, created_at)
                   VALUES (?,?,?,?,?,0,?)""",
                (eid, scope_id, client_key, payload, task_type, now_ms()),
            )
        return eid, True

    def buffer_depth(self, scope_id: str | None = None) -> int:
        if scope_id:
            r = self.conn.execute(
                "SELECT COUNT(*) FROM episode_buffer WHERE scope_id=? AND consolidated=0",
                (scope_id,),
            ).fetchone()
        else:
            r = self.conn.execute(
                "SELECT COUNT(*) FROM episode_buffer WHERE consolidated=0"
            ).fetchone()
        return int(r[0])

    def pending_episodes(self, scope_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM episode_buffer WHERE scope_id=? AND consolidated=0 "
            "ORDER BY created_at ASC LIMIT ?",
            (scope_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_consolidated(self, episode_ids: Sequence[str]) -> None:
        if not episode_ids:
            return
        with self.write() as c:
            c.executemany(
                "UPDATE episode_buffer SET consolidated=1 WHERE id=?",
                [(e,) for e in episode_ids],
            )

    # -- counters (unsampled base rates, HLD 7) ---------------------------

    def bump_counter(
        self, scope_id: str, task_type: str, outcome: str, duration_ms: int, n_steps: int
    ) -> None:
        with self.write() as c:
            c.execute(
                """INSERT INTO counters (scope_id, task_type, outcome, n, sum_ms, sum_steps)
                   VALUES (?,?,?,1,?,?)
                   ON CONFLICT(scope_id, task_type, outcome) DO UPDATE SET
                     n = n + 1, sum_ms = sum_ms + excluded.sum_ms,
                     sum_steps = sum_steps + excluded.sum_steps""",
                (scope_id, task_type, outcome, duration_ms, n_steps),
            )
            c.execute(
                "INSERT INTO duration_samples (scope_id, task_type, duration_ms, n_steps, created_at) "
                "VALUES (?,?,?,?,?)",
                (scope_id, task_type, duration_ms, n_steps, now_ms()),
            )

    def base_rate(self, scope_id: str, task_type: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT outcome, n FROM counters WHERE scope_id=? AND task_type=?",
            (scope_id, task_type),
        ).fetchall()
        return {r["outcome"]: r["n"] for r in rows}

    def duration_median(self, scope_id: str, task_type: str) -> tuple[float, int] | None:
        rows = self.conn.execute(
            "SELECT duration_ms FROM duration_samples WHERE scope_id=? AND task_type=? "
            "ORDER BY duration_ms",
            (scope_id, task_type),
        ).fetchall()
        if not rows:
            return None
        vals = [r["duration_ms"] for r in rows]
        n = len(vals)
        mid = n // 2
        median = float(vals[mid]) if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
        return median, n

    # -- communities -------------------------------------------------------

    def upsert_community(
        self,
        cid: str,
        scope_id: str,
        *,
        label: str | None,
        summary: str | None,
        cohesion: float,
        node_count: int,
        prev_id: str | None,
    ) -> None:
        with self.write() as c:
            c.execute(
                """INSERT INTO communities (id, scope_id, label, summary, cohesion,
                                            node_count, prev_id, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     label=excluded.label, summary=excluded.summary,
                     cohesion=excluded.cohesion, node_count=excluded.node_count,
                     prev_id=excluded.prev_id, updated_at=excluded.updated_at""",
                (cid, scope_id, label, summary, cohesion, node_count, prev_id, now_ms()),
            )

    def get_community(self, cid: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM communities WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None

    def list_communities(self, scope_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM communities WHERE scope_id=? ORDER BY node_count DESC",
            (scope_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- meta --------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        with self.write() as c:
            c.execute(
                "INSERT INTO meta(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
