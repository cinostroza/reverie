"""Command line interface (HLD 11.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import DEFAULT_DB, Reverie
from .config import Config

_TICK, _CROSS, _WARN = "OK", "FAIL", "WARN"


def _mem(args: argparse.Namespace) -> Reverie:
    config = Config.load(args.config) if getattr(args, "config", None) else Config()
    return Reverie(scope=args.scope, db=args.db, config=config)


def _out(text: str) -> None:
    # Windows consoles frequently run a legacy codepage; degrade rather than
    # raise a UnicodeEncodeError in the middle of a report.
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc) + "\n")


def cmd_init(args: argparse.Namespace) -> int:
    mem = _mem(args)
    _out(f"Initialised {mem.store.path}")
    _out("")
    _out("Integration snippet:")
    _out(
        f"""
    from reverie import Reverie

    mem = Reverie(scope="{args.scope}")
    result = mem.recall("<what the agent is about to do>")

    with mem.episode("<task>", recall_id=result.recall_id) as ep:
        ep.step("bash", "<command>", exit_code=code)
        ep.outcome("success" if code == 0 else "failure", tier=1)

    mem.dream()   # attribute outcomes, then consolidate
"""
    )
    mem.close()
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    mem = _mem(args)
    result = mem.recall(args.cue, budget_tokens=args.budget)
    _out(result.brief)
    _out("")
    _out(
        f"[{len(result.nodes)} nodes · {result.token_cost} tokens · "
        f"{result.latency_ms:.1f}ms · visited {result.visited}"
        + (" · HIT VISIT CAP" if result.hit_visit_cap else "")
        + (f" · ABLATION withheld {len(result.withheld_ids)}" if result.is_ablation else "")
        + f" · recall_id {result.recall_id or '-'}]"
    )
    mem.close()
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    mem = _mem(args)
    consolidation, attribution = mem.dream(force_abstract=args.abstract)
    _out(f"Attribution:   {attribution}")
    _out(f"Consolidation: {consolidation}")
    mem.close()
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    mem = _mem(args)
    info = mem.inspect(args.node_id)
    if not info:
        _out(f"No node {args.node_id}")
        mem.close()
        return 1
    node = info["node"]
    _out(f"{node.id}  [{node.type}/{node.memory_class}]  state={node.state}")
    _out(f"  label      {node.label}")
    _out(f"  body       {node.body}")
    _out(f"  provenance {node.provenance}   valence {node.valence:+.2f}")
    u = info["utility"]
    _out(
        f"  utility    lcb={u['lcb']:.3f} mean={u['mean']:.3f} "
        f"(alpha={u['alpha']:.2f} beta={u['beta']:.2f} n={u['attributions']})"
    )
    if info["community"]:
        c = info["community"]
        _out(f"  community  {c['label']} (cohesion {c['cohesion']:.2f}, {c['node_count']} nodes)")
    _out(f"  edges      {len(info['edges'])}")
    for e in info["edges"][:12]:
        target = mem.store.get_node(e.dst)
        _out(f"    -{e.relation}-> {target.label[:50] if target else e.dst} (w={e.weight:.2f})")
    mem.close()
    return 0


def cmd_why(args: argparse.Namespace) -> int:
    mem = _mem(args)
    sources = mem.why(args.node_id)
    if not sources:
        _out("No source episodes recorded.")
        mem.close()
        return 1
    for src in sources:
        _out(f"--- episode {src['episode_id']}")
        if src.get("payload"):
            payload = json.loads(src["payload"])
            _out(f"    task    {payload['task']['description']}")
            _out(f"    outcome {payload.get('outcome')}")
            for step in payload["steps"][:6]:
                _out(f"    step    {step['kind']} {step['name']}: {step['output_summary'][:70]}")
        else:
            _out(f"    {src.get('note')}")
    mem.close()
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    mem = _mem(args)
    chain = mem.path(args.src, args.dst)
    if not chain:
        _out(f"No association path between {args.src!r} and {args.dst!r}.")
        mem.close()
        return 1
    for i, node in enumerate(chain):
        arrow = "    " if i == 0 else " -> "
        _out(f"{arrow}{node.label[:70]}  [{node.type}]")
    mem.close()
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    mem = _mem(args)
    mem.forget(args.node_id, args.reason)
    _out(f"Deleted {args.node_id}. Cascaded through edges, embeddings, and FTS.")
    mem.close()
    return 0


def cmd_assert(args: argparse.Namespace) -> int:
    mem = _mem(args)
    entities = args.entities.split(",") if args.entities else []
    node_id = mem.assert_fact(args.fact, entities=[e.strip() for e in entities if e.strip()])
    _out(f"Asserted {node_id} (provenance=asserted; wins conflicts, never auto-quarantined)")
    mem.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    mem = _mem(args)
    s = mem.stats()
    _out(f"scope           {s['scope']}")
    _out(f"nodes by state  {s['nodes_by_state']}")
    _out(f"nodes by class  {s['nodes_by_class']}")
    _out(f"edges           {s['edges']}")
    _out(f"recalls logged  {s['recalls']}")
    _out(f"buffer depth    {s['buffer_depth']}")
    _out(f"pending attrib  {s['pending_attributions']}")
    _out(f"modularity      {s['modularity'] or '-'}")
    if s["communities"]:
        _out("communities")
        for c in s["communities"][:10]:
            _out(f"  {c['label'][:44]:<44} {c['node_count']:>4} nodes  cohesion {c['cohesion']:.2f}")
    ab = s["ablation"]
    if ab["lift"] is not None:
        _out(
            f"ablation lift   {ab['lift']:+.3f} "
            f"(full {ab['full_recall_success_rate']:.2f} n={ab['n_full']} vs "
            f"ablated {ab['ablated_success_rate']:.2f} n={ab['n_ablated']})"
        )
    else:
        _out(f"ablation lift   not enough data (full n={ab['n_full']}, ablated n={ab['n_ablated']})")
    mem.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """The first question every user has: is this thing actually working?"""
    mem = _mem(args)
    conn = mem.store.conn
    scope = args.scope
    problems = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal problems
        if not ok:
            problems += 1
        _out(f"  [{_TICK if ok else _WARN}] {label}{('  — ' + detail) if detail else ''}")

    _out("Reverie health check")
    _out("")

    n_active = mem.store.count_nodes(scope)
    check("memory graph populated", n_active > 0, f"{n_active} active nodes")

    depth = mem.store.buffer_depth(scope)
    check("episode buffer draining", depth < 1000, f"{depth} unconsolidated episodes")

    pending = conn.execute(
        "SELECT COUNT(*) FROM outcomes WHERE scope_id=? AND attributed=0", (scope,)
    ).fetchone()[0]
    check("attribution keeping up", pending < 100, f"{pending} outcomes pending")

    orphan_recalls = conn.execute(
        "SELECT COUNT(*) FROM recalls r WHERE r.scope_id=? AND NOT EXISTS "
        "(SELECT 1 FROM outcomes o WHERE o.recall_id = r.id)", (scope,)
    ).fetchone()[0]
    n_recalls = conn.execute(
        "SELECT COUNT(*) FROM recalls WHERE scope_id=?", (scope,)
    ).fetchone()[0]
    # An unattributed recall teaches the system nothing. This is the single
    # most common integration mistake and the one that silently disables the
    # entire point of the project.
    check(
        "recalls are getting outcomes",
        n_recalls == 0 or orphan_recalls / max(n_recalls, 1) < 0.5,
        f"{orphan_recalls}/{n_recalls} recalls have no reported outcome",
    )

    dangling = conn.execute(
        "SELECT COUNT(*) FROM edges e WHERE e.scope_id=? AND NOT EXISTS "
        "(SELECT 1 FROM nodes n WHERE n.id = e.dst)", (scope,)
    ).fetchone()[0]
    check("no dangling edges", dangling == 0, f"{dangling} edges point at missing nodes")

    quarantined = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE scope_id=? AND state='quarantined'", (scope,)
    ).fetchone()[0]
    check("quarantine rate sane", quarantined <= max(3, n_active * 0.2),
          f"{quarantined} quarantined of {n_active + quarantined}")

    dupes = conn.execute(
        "SELECT COUNT(*) FROM (SELECT label FROM nodes WHERE scope_id=? AND state='active' "
        "GROUP BY lower(label) HAVING COUNT(*) > 1)", (scope,)
    ).fetchone()[0]
    check("no obvious duplicates", dupes == 0, f"{dupes} labels appear more than once")

    tier1 = conn.execute(
        "SELECT COUNT(*) FROM outcomes WHERE scope_id=? AND signal_tier=1", (scope,)
    ).fetchone()[0]
    total_outcomes = conn.execute(
        "SELECT COUNT(*) FROM outcomes WHERE scope_id=?", (scope,)
    ).fetchone()[0]
    check(
        "outcomes are ground truth",
        total_outcomes == 0 or tier1 / total_outcomes > 0.5,
        f"{tier1}/{total_outcomes} are tier 1; system quality is bounded here",
    )

    _out("")
    _out(f"{problems} issue(s) found." if problems else "All checks passed.")
    mem.close()
    return 1 if problems else 0


def cmd_reembed(args: argparse.Namespace) -> int:
    from .embed import get_embedder

    mem = Reverie(scope=args.scope, db=args.db, config=Config())
    embedder = get_embedder(args.model)
    n = 0
    for node in mem.store.iter_nodes(
        [args.scope], states=("active", "contested", "stale", "quarantined")
    ):
        mem.store.set_embedding(node.id, node.scope_id, embedder.embed(f"{node.label} {node.body}"))
        n += 1
    mem.store.set_meta("embedding_model", embedder.name)
    mem.store.set_meta("embedding_dim", str(embedder.dim))
    _out(f"Re-embedded {n} nodes with {embedder.name}.")
    mem.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reverie",
        description="Long-term memory for AI agents that learns which memories are worth having.",
    )
    p.add_argument("--db", default=DEFAULT_DB, help=f"database path (default {DEFAULT_DB})")
    p.add_argument("--scope", default="agent:default", help="memory scope")
    p.add_argument("--config", default=None, help="path to a JSON config file")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and print an integration snippet").set_defaults(func=cmd_init)

    sp = sub.add_parser("recall", help="show what the agent would see for a cue")
    sp.add_argument("cue")
    sp.add_argument("--budget", type=int, default=None)
    sp.set_defaults(func=cmd_recall)

    sp = sub.add_parser("consolidate", help="run the dream cycle now")
    sp.add_argument("--abstract", action="store_true", help="force community detection")
    sp.set_defaults(func=cmd_consolidate)

    sp = sub.add_parser("inspect", help="node, edges, utility history")
    sp.add_argument("node_id")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("why", help="trace a memory back to its source episodes")
    sp.add_argument("node_id")
    sp.set_defaults(func=cmd_why)

    sp = sub.add_parser("path", help="shortest association path between two memories")
    sp.add_argument("src")
    sp.add_argument("dst")
    sp.set_defaults(func=cmd_path)

    sp = sub.add_parser("forget", help="delete a memory and cascade")
    sp.add_argument("node_id")
    sp.add_argument("--reason", default="")
    sp.set_defaults(func=cmd_forget)

    sp = sub.add_parser("assert", help="human-authored memory; wins conflicts")
    sp.add_argument("fact")
    sp.add_argument("--entities", default="")
    sp.set_defaults(func=cmd_assert)

    sub.add_parser("stats", help="graph, communities, ablation lift").set_defaults(func=cmd_stats)
    sub.add_parser("doctor", help="health check").set_defaults(func=cmd_doctor)

    sp = sub.add_parser("reembed", help="rebuild embeddings after a model change")
    sp.add_argument("--model", default="hash-64")
    sp.set_defaults(func=cmd_reembed)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
