"""Why attribution measures zero: the arithmetic, measured on a live graph.

    python experiments/_attribution_diagnostic.py

Backs the paper's §"an arithmetic account" and README §E9. The ablation sweep
(`_run_sweeps.py attribution`) shows attribution contributes nothing; this shows
*why*, and the reason is a budget rather than a bug:

    total credit mass in the system == number of task outcomes
    nodes sharing it                == episodes + lessons
    -> ~1 unit of evidence per node, before it is split across a 9-item brief

A Beta posterior cannot separate anything on one unit of evidence. So the check
is not "is the estimator good" but "did the posterior move at all" -- and the
diagnostic statistic is the share of nodes still sitting on the untouched
Beta(1,1) prior, whose LCB is exactly 0.250.

Runs the same 600-task on-call landscape the headline numbers use, with
attribution fully enabled and a reviewer present, which is the condition E7
pre-registered as the one most favourable to attribution working.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reverie.bench.agents import EvidenceAgent
from reverie.bench.backends import ReverieBackend
from reverie.bench.core import run_arm
from reverie.bench.families import IncidentFamily
from reverie.bench.human import HumanOracle

SEEDS = range(4)
N_TASKS = 600
PRIOR_LCB = 0.250   # utility_lcb of an untouched Beta(1, 1)


def probe(seed: int) -> dict:
    backend = ReverieBackend(oracle=HumanOracle(frequency=0.5, accuracy=0.6))

    # `finish()` closes the store, and the posterior is what we came for, so
    # the graph is inspected before the run is allowed to tear itself down.
    captured: dict = {}
    original_finish = ReverieBackend.finish

    def capture(self) -> None:
        mem = self._mem
        if mem is not None:
            rows = mem.store.conn.execute(
                "SELECT id, memory_class, alpha, beta, attributions "
                "FROM nodes WHERE scope_id=? AND state != 'deleted'",
                (self.scope,),
            ).fetchall()
            captured["rows"] = [tuple(r) for r in rows]
            captured["utility"] = {
                r[0]: mem.attributor.utility(r[0]) for r in rows
            }
        original_finish(self)

    ReverieBackend.finish = capture
    try:
        result = run_arm(IncidentFamily(), EvidenceAgent(), backend,
                         n_tasks=N_TASKS, seed=seed, arm="diagnostic")
    finally:
        ReverieBackend.finish = original_finish

    rows = captured["rows"]
    utils = captured["utility"]
    attrs = [r[4] for r in rows]
    lessons = [r for r in rows if r[1] == "procedural"]
    at_prior = sum(1 for u in utils.values() if abs(u - PRIOR_LCB) < 1e-9)

    return {
        "seed": seed,
        "success": sum(x["success"] for x in result.records) / len(result.records),
        "n_nodes": len(rows),
        "n_lessons": len(lessons),
        "outcomes": N_TASKS,
        "credit_per_node": N_TASKS / len(rows) if rows else float("nan"),
        "median_attributions": statistics.median(attrs) if attrs else 0,
        "mean_attributions": statistics.fmean(attrs) if attrs else 0,
        "median_lesson_attributions": (
            statistics.median([r[4] for r in lessons]) if lessons else 0),
        "frac_at_prior": at_prior / len(rows) if rows else float("nan"),
        "utility_min": min(utils.values()) if utils else float("nan"),
        "utility_max": max(utils.values()) if utils else float("nan"),
        # Quartiles of the utility LCB. If the posterior had moved, these would
        # spread; the check is whether they sit on top of each other at 0.250.
        "utility_q": statistics.quantiles(sorted(utils.values()), n=4) if len(utils) > 3
                     else [float("nan")] * 3,
    }


def main() -> None:
    runs = [probe(s) for s in SEEDS]

    def avg(key: str) -> float:
        return statistics.fmean(r[key] for r in runs)

    print(f"\nAttribution diagnostic -- {N_TASKS} tasks, {len(runs)} seeds, "
          "reviewer present at accuracy 0.6\n")
    print(f"  outcomes observed          {N_TASKS}")
    print(f"  nodes sharing the credit   {avg('n_nodes'):.0f} "
          f"({avg('n_lessons'):.0f} of them human lessons)")
    print(f"  -> credit units per node   {avg('credit_per_node'):.2f}")
    print()
    print(f"  median attributions/node   {avg('median_attributions'):.1f}")
    print(f"  mean attributions/node     {avg('mean_attributions'):.2f}")
    print(f"  median for lessons only    {avg('median_lesson_attributions'):.1f}")
    print()
    print(f"  nodes still at the prior   {avg('frac_at_prior'):.1%}  "
          f"(utility LCB exactly {PRIOR_LCB})")
    print(f"  utility LCB range          {avg('utility_min'):.3f} - "
          f"{avg('utility_max'):.3f}")
    q = [statistics.fmean(r["utility_q"][i] for r in runs) for i in range(3)]
    print(f"  utility LCB quartiles      p25 {q[0]:.3f} | p50 {q[1]:.3f} | "
          f"p75 {q[2]:.3f}")
    print()
    print("A posterior that has not left its prior cannot rank anything. The "
          "constraint is\noutcomes / nodes, which no amount of better credit "
          "assignment changes -- and it\ngets worse, not better, as the graph "
          "grows.\n")


if __name__ == "__main__":
    main()
