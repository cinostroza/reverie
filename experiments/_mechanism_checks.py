"""Mechanism checks quoted in the paper that no sweep produces.

    python experiments/_mechanism_checks.py

1. How often the scope guard acts under over-broad review (paper §4.5).
2. The scripted agent on the exact E11 briefs, so the language-model result
   can be compared with the policy that produced every other number (§4.4).
3. Run-to-run variation of the memory system at a fixed seed (§2, footnote).
"""

from __future__ import annotations

import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reverie.bench.agents import EvidenceAgent
from reverie.bench.backends import ReverieBackend
from reverie.bench.core import Evidence, Task, run_arm
from reverie.bench.families import IncidentFamily
from reverie.bench.human import HumanOracle
from reverie.consolidate import Consolidator


def guard_activity(seed: int) -> dict:
    """Count narrowing operations under a reviewer who names only the service."""
    counts = {"narrowed": 0}
    original = Consolidator.validate_scopes

    def counted(self, scope_id, stats):
        before = stats.scopes_narrowed
        original(self, scope_id, stats)
        counts["narrowed"] += stats.scopes_narrowed - before

    Consolidator.validate_scopes = counted
    try:
        backend = ReverieBackend(oracle=HumanOracle(
            frequency=0.5, accuracy=1.0, keep_features=("service",)))
        run_arm(IncidentFamily(), EvidenceAgent(), backend, n_tasks=600, seed=seed,
                arm="guard")
    finally:
        Consolidator.validate_scopes = original
    counts["lessons"] = backend.lessons_written
    return counts


def scripted_on_probe_briefs() -> dict[int, tuple[int, int]]:
    """Replay experiments/probes/mkbrief.py's briefs through the scripted agent.

    Exploration is switched off so the comparison is between the two policies'
    judgements, not the scripted agent's 12% random moves.
    """
    svc, sym, wrong, right = "svc-03", "sym-2", "restart", "failover"
    strategies = ("restart", "rollback", "scale_out", "clear_cache", "failover",
                  "page_owner")
    task = Task("oncall_incident", 0,
                {"service": svc, "symptom": sym, "severity": "sev2"}, strategies)
    rows = [line.split("\t") for line in
            (ROOT / "experiments/probes/results.tsv").read_text().splitlines()
            if line.strip()]
    agent = EvidenceAgent(epsilon=0.0)

    out = {}
    for n in sorted({int(r[0]) for r in rows}):
        mine = [r for r in rows if int(r[0]) == n]
        model_followed_evidence = sum(1 for r in mine if r[2].strip() == right)
        scripted_followed_evidence = 0
        for r in mine:
            rng = random.Random(int(r[1]))       # same seed mkbrief.py used
            evidence = [Evidence(f"service={svc},symptom={sym}", wrong, True, weight=3.0)]
            for _ in range(n):
                sev = rng.choice(["sev1", "sev2", "sev3"])
                ctx = f"service={svc},severity={sev},symptom={sym}"
                evidence += [Evidence(ctx, wrong, False), Evidence(ctx, right, True)]
            if agent.choose(task, evidence, random.Random(0)) == right:
                scripted_followed_evidence += 1
        out[n] = (model_followed_evidence, scripted_followed_evidence, len(mine))
    return out


def repeat_same_seed(_: int) -> float:
    r = run_arm(IncidentFamily(), EvidenceAgent(), ReverieBackend(), n_tasks=600,
                seed=0, arm="repeat")
    return sum(x["success"] for x in r.records[300:]) / 300


def main() -> None:
    with ProcessPoolExecutor(max_workers=8) as pool:
        guard = list(pool.map(guard_activity, range(4)))
        repeats = list(pool.map(repeat_same_seed, range(4)))

    print("1. Scope guard under over-broad review (4 runs, 600 tasks)")
    for s, g in enumerate(guard):
        print(f"   seed {s}: {g['narrowed']:3d} narrowing operations, "
              f"{g['lessons']} lessons written")

    print("\n2. E11 briefs: runs that followed the evidence over the wrong lesson")
    print("   contradicting pairs   claude-opus-5   scripted agent (same briefs)")
    for n, (model, scripted, total) in scripted_on_probe_briefs().items():
        print(f"   {n:>19d}   {model:>6d}/{total}      {scripted:>6d}/{total}")

    print("\n3. Reverie, seed 0, repeated 4 times: second-half success")
    print("   " + "  ".join(f"{x:.3f}" for x in repeats)
          + f"   (sd {statistics.stdev(repeats):.3f})")
    print("   The environment is deterministic given a seed; the memory system is")
    print("   not (wall-clock recency, random bits in node ids).")


if __name__ == "__main__":
    main()
