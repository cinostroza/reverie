"""Community detection for the abstract stage (HLD 8.7).

Distillation produces lessons about individual episodes. What it cannot
produce is the observation that thirty scattered memories are all really about
one thing. This module is the mechanical analogue of schema formation: cluster
the graph, and give each cluster an identity that survives re-clustering.

Louvain is implemented here rather than imported so the core stays
dependency-free. It is the standard modularity-maximising local-moving
algorithm with the usual aggregation phase.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence

__all__ = ["louvain", "modularity", "match_communities", "cohesion"]


def louvain(
    edges: Sequence[tuple[str, str, float]],
    *,
    resolution: float = 1.0,
    seed: int = 0,
    max_passes: int = 10,
) -> dict[str, int]:
    """Partition an undirected weighted graph. Returns node -> community index.

    Deterministic given ``seed``: node visit order is shuffled with a seeded
    RNG, which matters because Louvain is order-sensitive and an unstable
    partition would churn theme nodes on every consolidation cycle.
    """
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for src, dst, weight in edges:
        if src == dst or weight <= 0:
            continue
        adjacency[src][dst] = adjacency[src].get(dst, 0.0) + weight
        adjacency[dst][src] = adjacency[dst].get(src, 0.0) + weight

    if not adjacency:
        return {}

    original_nodes = sorted(adjacency)

    # Levels operate on *integer* super-node keys, never on caller-supplied
    # strings. An earlier version named super-nodes "c{index}" and parsed the
    # index back out, which broke the moment a real node was itself called
    # "c" -- and broke by crashing on some graphs and silently mislabelling
    # others.
    index_of = {n: i for i, n in enumerate(original_nodes)}
    graph: dict[int, dict[int, float]] = {
        index_of[n]: {index_of[m]: w for m, w in adj.items()}
        for n, adj in adjacency.items()
    }
    # super-node key -> the original nodes it contains
    contents: dict[int, set[str]] = {i: {n} for n, i in index_of.items()}

    membership: dict[str, int] = dict(index_of)
    rng = random.Random(seed)

    for _ in range(max_passes):
        comm, improved = _one_level(graph, resolution, rng)
        if not improved:
            break

        # Relabel this level's communities to dense 0..k-1.
        labels = {c: i for i, c in enumerate(sorted(set(comm.values())))}

        new_contents: dict[int, set[str]] = defaultdict(set)
        for super_node, cid in comm.items():
            new_contents[labels[cid]] |= contents[super_node]

        new_graph: dict[int, dict[int, float]] = defaultdict(dict)
        for src, adj in graph.items():
            csrc = labels[comm[src]]
            for dst, w in adj.items():
                cdst = labels[comm[dst]]
                new_graph[csrc][cdst] = new_graph[csrc].get(cdst, 0.0) + w

        for cid, originals in new_contents.items():
            for original in originals:
                membership[original] = cid

        graph = {k: dict(v) for k, v in new_graph.items()}
        contents = dict(new_contents)

        if len(graph) <= 1:
            break

    # Renumber densely and deterministically.
    ordered = sorted(set(membership.values()))
    renumber = {old: i for i, old in enumerate(ordered)}
    return {n: renumber[c] for n, c in membership.items()}


def _one_level(
    graph: dict[int, dict[int, float]], resolution: float, rng: random.Random
) -> tuple[dict[int, int], bool]:
    """Local moving phase: repeatedly move nodes to the neighbouring community
    that yields the largest modularity gain."""
    nodes = sorted(graph)
    comm = {n: i for i, n in enumerate(nodes)}

    degrees = {n: sum(adj.values()) for n, adj in graph.items()}
    self_loops = {n: graph[n].get(n, 0.0) for n in nodes}
    total_weight = sum(degrees.values()) / 2.0
    if total_weight <= 0:
        return comm, False

    comm_total = {comm[n]: degrees[n] for n in nodes}
    improved_any = False

    for _ in range(20):  # sweep until stable
        moved = False
        order = list(nodes)
        rng.shuffle(order)

        for node in order:
            node_comm = comm[node]
            deg = degrees[node]

            weights_to: dict[int, float] = defaultdict(float)
            for neighbor, w in graph[node].items():
                if neighbor != node:
                    weights_to[comm[neighbor]] += w

            # Remove node from its community before evaluating alternatives.
            comm_total[node_comm] -= deg

            best_comm = node_comm
            best_gain = weights_to.get(node_comm, 0.0) - resolution * comm_total[
                node_comm
            ] * deg / (2.0 * total_weight)

            for cand, w in weights_to.items():
                if cand == node_comm:
                    continue
                gain = w - resolution * comm_total[cand] * deg / (2.0 * total_weight)
                if gain > best_gain:
                    best_gain, best_comm = gain, cand

            comm_total[best_comm] += deg
            comm[node] = best_comm
            if best_comm != node_comm:
                moved = improved_any = True

        if not moved:
            break

    return comm, improved_any


def modularity(
    edges: Sequence[tuple[str, str, float]], partition: dict[str, int]
) -> float:
    if not edges:
        return 0.0
    degrees: dict[str, float] = defaultdict(float)
    total = 0.0
    internal: dict[int, float] = defaultdict(float)
    comm_degree: dict[int, float] = defaultdict(float)

    for src, dst, w in edges:
        if w <= 0:
            continue
        degrees[src] += w
        degrees[dst] += w
        total += w
        if partition.get(src) == partition.get(dst) and src in partition:
            internal[partition[src]] += w

    if total <= 0:
        return 0.0
    for node, deg in degrees.items():
        if node in partition:
            comm_degree[partition[node]] += deg

    return sum(
        internal[c] / total - (comm_degree[c] / (2.0 * total)) ** 2
        for c in comm_degree
    )


def cohesion(
    edges: Sequence[tuple[str, str, float]], members: Iterable[str]
) -> float:
    """Fraction of member edge weight that stays inside the community."""
    member_set = set(members)
    inside = outside = 0.0
    for src, dst, w in edges:
        s_in, d_in = src in member_set, dst in member_set
        if s_in and d_in:
            inside += w
        elif s_in or d_in:
            outside += w
    total = inside + outside
    return inside / total if total > 0 else 0.0


def match_communities(
    new_partition: dict[str, int],
    previous: dict[str, list[str]],
    *,
    jaccard_threshold: float = 0.5,
) -> dict[int, str | None]:
    """Map each new community index to a previous community id, or None.

    Naive re-clustering renumbers everything each run, which would make theme
    nodes churn and community ids useless for anything persistent (HLD 8.7
    step 2, and the "community instability" row in HLD 15). Matching by
    maximum node overlap keeps identity stable across runs.

    Greedy by descending overlap, one-to-one: two new communities cannot both
    inherit the same previous id.
    """
    groups: dict[int, set[str]] = defaultdict(set)
    for node, cid in new_partition.items():
        groups[cid].add(node)

    prev_sets = {pid: set(members) for pid, members in previous.items()}

    candidates: list[tuple[float, int, str]] = []
    for cid, members in groups.items():
        for pid, prev_members in prev_sets.items():
            union = members | prev_members
            if not union:
                continue
            score = len(members & prev_members) / len(union)
            if score >= jaccard_threshold:
                candidates.append((score, cid, pid))

    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    assigned: dict[int, str | None] = {cid: None for cid in groups}
    used: set[str] = set()
    for score, cid, pid in candidates:
        if assigned[cid] is None and pid not in used:
            assigned[cid] = pid
            used.add(pid)
    return assigned
