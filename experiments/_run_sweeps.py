"""Driver for the scripted sweeps behind E7-E12.

E1-E6 are notebooks. E7-E12 were run ad hoc and recorded only as prose in
``README.md``, which meant the numbers the paper quotes could not be
regenerated from the tree. This script closes that gap: every table in
``README.md`` §E7-E12 and every number in ``paper/main.tex`` comes from one of
the sweeps below.

    python experiments/_run_sweeps.py             # everything
    python experiments/_run_sweeps.py canonical   # one sweep
    python experiments/_run_sweeps.py --list

Results land in ``experiments/results/<sweep>.json`` as raw per-run rows, so a
re-analysis never has to re-run the compute. Aggregation is in ``_analyse.py``.

Two properties are load-bearing and both were learned the hard way:

* **Paired over identical seeds.** Every arm in a sweep sees byte-identical
  task sequences. Seed variance dominates the effects we care about, so an
  unpaired comparison cannot resolve them.
* **Cluster-robust error bars.** Standard errors are computed across *runs*,
  never across tasks. Outcomes within a run are strongly correlated -- a run
  either discovers the rule or it does not -- and the per-task formula
  understates the error by ~1.8x. That manufactured one phantom finding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reverie.config import Config
from reverie.bench.agents import EvidenceAgent
from reverie.bench.backends import NoMemoryBackend, ReplayBackend, ReverieBackend
from reverie.bench.core import run_arm
from reverie.bench.families import IncidentFamily
from reverie.bench.human import HumanOracle

RESULTS = Path(__file__).parent / "results"

N_TASKS = 600
# 16 seeds is the floor for believing a comparison; see README §Reproducibility.
SEEDS_16 = tuple(range(16))
SEEDS_8 = tuple(range(8))
SEEDS_6 = tuple(range(6))

# The budget-matched competitor. `window` is effectively unbounded so the token
# budget, not the episode count, is what binds -- at long horizons the context
# window is the real constraint and a count-based window flatters replay.
REPLAY_1200 = dict(budget_tokens=1200, window=10**9)


# ---------------------------------------------------------------------------
# One unit of work
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Cell:
    """One (condition, seed) run. Picklable: it must cross a process boundary."""

    sweep: str
    condition: str
    seed: int
    backend: str                       # "none" | "replay" | "reverie"
    backend_kw: dict = field(default_factory=dict)
    family_kw: dict = field(default_factory=dict)
    oracle_kw: dict | None = None
    config_kw: dict = field(default_factory=dict)
    n_tasks: int = N_TASKS


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    """Expand "section.field" keys into the nested shape Config expects."""
    out: dict[str, Any] = {}
    for dotted, value in flat.items():
        section, _, field_name = dotted.partition(".")
        out.setdefault(section, {})[field_name] = value
    return out


def _run(cell: Cell) -> dict[str, Any]:
    family = IncidentFamily(**cell.family_kw)

    if cell.backend == "none":
        backend = NoMemoryBackend()
    elif cell.backend == "replay":
        backend = ReplayBackend(**cell.backend_kw)
    else:
        kw = dict(cell.backend_kw)
        if cell.config_kw:
            # config_kw is "section.field" -> value, e.g. consolidation.scope_guard.
            kw["config"] = Config.from_dict(_nest(cell.config_kw))
        if cell.oracle_kw is not None:
            kw["oracle"] = HumanOracle(**cell.oracle_kw)
        backend = ReverieBackend(**kw)

    t0 = time.time()
    result = run_arm(family, EvidenceAgent(), backend,
                     n_tasks=cell.n_tasks, seed=cell.seed, arm=cell.condition)
    wall = time.time() - t0

    recs = result.records
    half = len(recs) // 2
    return {
        "sweep": cell.sweep,
        "condition": cell.condition,
        "seed": cell.seed,
        "n_tasks": cell.n_tasks,
        "success": sum(r["success"] for r in recs) / len(recs),
        # Second-half success isolates the *learned* rate from the cold start,
        # which is what the density table reports.
        "success_2h": sum(r["success"] for r in recs[half:]) / (len(recs) - half),
        "tokens": sum(r["tokens"] for r in recs) / len(recs),
        "lessons": getattr(backend, "lessons_written", 0),
        "quarantine_rate": getattr(backend, "final_quarantine_rate", 0.0),
        "wall_s": round(wall, 1),
    }


# ---------------------------------------------------------------------------
# Sweep definitions -- each returns the list of cells to run
# ---------------------------------------------------------------------------

def _reverie(cond, seed, sweep, **kw) -> Cell:
    return Cell(sweep=sweep, condition=cond, seed=seed, backend="reverie", **kw)


def sweep_canonical() -> list[Cell]:
    """The three numbers the project rests on, paired over 16 seeds.

    README §"The canonical numbers"; paper abstract and §5.
    """
    cells = []
    for s in SEEDS_16:
        cells.append(Cell("canonical", "none", s, "none"))
        cells.append(Cell("canonical", "replay", s, "replay", backend_kw=REPLAY_1200))
        cells.append(_reverie("reverie", s, "canonical"))
        cells.append(_reverie("reverie+human", s, "canonical",
                              oracle_kw=dict(frequency=0.5, accuracy=1.0)))
    return cells


def sweep_attribution() -> list[Cell]:
    """E7/E9: does outcome attribution lift task performance? Three landscapes.

    `fallible` is the condition E7 pre-registered as the one under which
    attribution *should* work: few nodes, recalled often, outcome not in the
    text, and some of them wrong.
    """
    arms = {
        "attr+utility":  dict(attribution=True, use_utility_weight=True),
        "attr_no_util":  dict(attribution=True, use_utility_weight=False),
        "no_attr":       dict(attribution=False),
    }
    landscapes = {
        "static": (dict(), None),
        "drift": (dict(drift_at=300), None),
        "fallible": (dict(), dict(frequency=0.5, accuracy=0.6)),
    }
    cells = []
    for lname, (fkw, okw) in landscapes.items():
        for aname, akw in arms.items():
            for s in SEEDS_8:
                cells.append(_reverie(f"{lname}/{aname}", s, "attribution",
                                      backend_kw=akw, family_kw=fkw, oracle_kw=okw))
    return cells


def sweep_review_frequency() -> list[Cell]:
    """E9: how often must you ask? Accuracy fixed at 1.0."""
    cells = []
    for s in SEEDS_8:
        cells.append(_reverie("none", s, "review_frequency"))
        for f in (0.05, 0.10, 0.25, 0.50, 1.00):
            cells.append(_reverie(f"f={f:.2f}", s, "review_frequency",
                                  oracle_kw=dict(frequency=f, accuracy=1.0)))
    return cells


def sweep_review_quality() -> list[Cell]:
    """E9: the falsifiable test -- does lesson quality change behaviour at all?"""
    cells = []
    for s in SEEDS_6:
        cells.append(_reverie("none", s, "review_quality"))
        for a in (0.4, 0.6, 0.8, 1.0):
            cells.append(_reverie(f"acc={a:.1f}", s, "review_quality",
                                  oracle_kw=dict(frequency=0.5, accuracy=a)))
    return cells


def sweep_error_modes() -> list[Cell]:
    """E10.2: robust to a reviewer being wrong? Three ways of being wrong."""
    cells = []
    for s in SEEDS_8:
        cells.append(_reverie("none", s, "error_modes"))
        for mode in ("random", "systematic", "adversarial"):
            for acc in (0.8, 0.5):
                cells.append(_reverie(f"{mode}/acc={acc}", s, "error_modes",
                                      oracle_kw=dict(frequency=0.5, accuracy=acc,
                                                     error_mode=mode)))
    return cells


def sweep_over_general() -> list[Cell]:
    """E10.3: the one reviewer failure that lands *below* doing nothing.

    The reviewer keeps only `service` when the true rule also depends on
    `symptom` -- a confident, correct-sounding lesson that is wrong 5 times in 6.
    """
    cells = []
    for s in SEEDS_16:
        cells.append(_reverie("none", s, "over_general"))
        cells.append(_reverie("correct_scope", s, "over_general",
                              oracle_kw=dict(frequency=0.5, accuracy=1.0,
                                             keep_features=("service", "symptom"))))
        cells.append(_reverie("over_general", s, "over_general",
                              oracle_kw=dict(frequency=0.5, accuracy=1.0,
                                             keep_features=("service",))))
    return cells


def sweep_density() -> list[Cell]:
    """E10.1: the governing variable is repetition density, not landscape size.

    Task budget is held fixed while the landscape grows, so tasks-per-context
    falls 8.3 -> 2.0 -> 1.0. This is the paper's Figure 1 and its most
    transferable claim.
    """
    shapes = {"72": (12, 6), "300": (30, 10), "1200": (60, 20)}
    cells = []
    for label, (svc, sym) in shapes.items():
        fkw = dict(n_services=svc, n_symptoms=sym)
        for s in SEEDS_6:
            cells.append(Cell("density", f"{label}/replay", s, "replay",
                              backend_kw=REPLAY_1200, family_kw=fkw))
            cells.append(_reverie(f"{label}/none", s, "density", family_kw=fkw))
            cells.append(_reverie(f"{label}/human", s, "density", family_kw=fkw,
                                  oracle_kw=dict(frequency=0.5, accuracy=1.0)))
    return cells


def sweep_staleness() -> list[Cell]:
    """E10.4: a reviewer who was right about a world that has since moved."""
    fkw = dict(drift_at=300)
    cells = []
    for s in SEEDS_8:
        cells.append(_reverie("drift/none", s, "staleness", family_kw=fkw))
        cells.append(_reverie("drift/human", s, "staleness", family_kw=fkw,
                              oracle_kw=dict(frequency=0.5, accuracy=1.0)))
        # 600 tasks / session_len 20 = 30 sessions; drift at task 300 = session 15.
        cells.append(_reverie("drift/human_stale", s, "staleness", family_kw=fkw,
                              oracle_kw=dict(frequency=0.5, accuracy=1.0,
                                             stop_after_session=15)))
    return cells


def sweep_guard() -> list[Cell]:
    """E12: the over-generalisation guard, against its four locked predictions.

    H1 (primary): over-general recovers toward the correct-scope arm.
    H2: the guard does *not* help action errors -- it is scope-only by design.
    H3: it costs approximately nothing when the reviewer was right.
    """
    conditions = {
        "over_general": dict(frequency=0.5, accuracy=1.0, keep_features=("service",)),
        "correct_scope": dict(frequency=0.5, accuracy=1.0),
        "random_err": dict(frequency=0.5, accuracy=0.5, error_mode="random"),
        "adversarial": dict(frequency=0.5, accuracy=0.5, error_mode="adversarial"),
    }
    cells = []
    for cname, okw in conditions.items():
        for guard in (True, False):
            for s in SEEDS_6:
                cells.append(_reverie(f"{cname}/guard={guard}", s, "guard",
                                      oracle_kw=okw, config_kw={"consolidation.scope_guard": guard}))
    return cells


def sweep_landscape() -> list[Cell]:
    """Is the headline retrieval effect a property of the system or of one landscape?

    Every number recorded before 2026-09-04 came from a single hidden landscape.
    `IncidentFamily` derives its feature->strategy rules from `landscape_seed`, and
    every sweep left that at its default, so 16 "seeds" varied the task *sequence*
    while the rules the agent had to learn stayed fixed. That makes the landscape an
    uncontrolled constant, not a controlled variable, and any effect that depends on
    which rules happen to be easy for a recency window is unidentifiable.

    This sweep crosses 8 landscapes with 6 task seeds. The arms stay paired inside
    each (landscape, seed) pair, so the paired difference is still exact; what changes
    is that the difference is now averaged over landscapes rather than conditioned on
    one.
    """
    cells = []
    for li, lseed in enumerate(20260812 + i for i in range(8)):
        fkw = dict(landscape_seed=lseed)
        for s in SEEDS_6:
            cells.append(Cell("landscape", f"L{li}/replay", s, "replay",
                              backend_kw=REPLAY_1200, family_kw=fkw))
            cells.append(_reverie(f"L{li}/reverie", s, "landscape", family_kw=fkw))
            cells.append(_reverie(f"L{li}/reverie+human", s, "landscape", family_kw=fkw,
                                  oracle_kw=dict(frequency=0.5, accuracy=1.0)))
    return cells


def sweep_guard_baseline() -> list[Cell]:
    """Does the scope guard cost anything when there is no reviewer at all?

    E12 measured the guard only under review, and concluded it was "nearly free"
    from the correct-scope arm. But consolidation *itself* promotes corroborated
    episodes into lessons, so the guard has candidates to act on even when no
    human ever speaks -- a case that was never tested. If it narrows or demotes
    engine-generated lessons, it degrades the plain retrieval arm, which is
    exactly the arm whose headline moved after the guard landed.
    """
    cells = []
    for s_ in SEEDS_16:
        for guard in (True, False):
            cells.append(_reverie(f"no_review/guard={guard}", s_, "guard_baseline",
                                  config_kw={"consolidation.scope_guard": guard}))
    return cells


SWEEPS: dict[str, Callable[[], list[Cell]]] = {
    "canonical": sweep_canonical,
    "attribution": sweep_attribution,
    "review_frequency": sweep_review_frequency,
    "review_quality": sweep_review_quality,
    "error_modes": sweep_error_modes,
    "over_general": sweep_over_general,
    "density": sweep_density,
    "staleness": sweep_staleness,
    "guard": sweep_guard,
    "landscape": sweep_landscape,
    "guard_baseline": sweep_guard_baseline,
}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweeps", nargs="*", help="sweeps to run (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 4)))
    args = ap.parse_args()

    if args.list:
        for name, fn in SWEEPS.items():
            print(f"{name:18s} {len(fn()):4d} runs")
        return

    names = args.sweeps or list(SWEEPS)
    unknown = [n for n in names if n not in SWEEPS]
    if unknown:
        sys.exit(f"unknown sweep(s): {', '.join(unknown)}")

    RESULTS.mkdir(exist_ok=True)
    for name in names:
        cells = SWEEPS[name]()
        print(f"[{name}] {len(cells)} runs on {args.workers} workers", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_run, cells, chunksize=1))
        out = RESULTS / f"{name}.json"
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"[{name}] wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
