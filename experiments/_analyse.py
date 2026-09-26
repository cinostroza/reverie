"""Aggregate the sweep results into the tables README and the paper quote.

    python experiments/_analyse.py            # every sweep that has results
    python experiments/_analyse.py canonical

Reads ``experiments/results/*.json`` written by ``_run_sweeps.py``; runs no
simulation itself, so re-analysis is free.

## The two statistics, and why

**Cluster-robust standard error.** `sqrt(p(1-p)/n_tasks)` assumes independent
trials. Outcomes within a run are strongly correlated -- a run either discovers
the rule or it does not -- so that formula understates the error by ~1.8x and
has already manufactured one phantom finding in this project. Everything here
is `stdev(per-run means) / sqrt(n_runs)`: the run is the unit of analysis, not
the task.

**Paired differences.** Every arm sees byte-identical task sequences at a given
seed, so the per-seed difference cancels seed variance entirely. The +0.056
retrieval effect is only resolvable paired; unpaired it is marginal. Whenever
two conditions share seeds, the paired difference is the number to report.

Bands printed as `+/-` are **2 x SE** -- an approximate 95% interval. A
difference is called significant when it exceeds its own band.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

RESULTS = Path(__file__).parent / "results"


def load(sweep: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{sweep}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def by_condition(rows: Sequence[dict], metric: str = "success") -> dict[str, dict[int, float]]:
    """condition -> {seed: value}. Keyed by seed so pairing is possible."""
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        out.setdefault(r["condition"], {})[r["seed"]] = r[metric]
    return out


def summarise(values: dict[int, float]) -> tuple[float, float, int]:
    """mean, 2*cluster-robust SE, n."""
    vals = list(values.values())
    n = len(vals)
    if n < 2:
        return (vals[0] if vals else float("nan")), float("nan"), n
    se = statistics.stdev(vals) / (n ** 0.5)
    return statistics.fmean(vals), 2 * se, n


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, int]:
    """mean(b - a) over shared seeds, 2*SE, n. Positive means b is better."""
    shared = sorted(set(a) & set(b))
    diffs = [b[s] - a[s] for s in shared]
    if len(diffs) < 2:
        return float("nan"), float("nan"), len(diffs)
    se = statistics.stdev(diffs) / (len(diffs) ** 0.5)
    return statistics.fmean(diffs), 2 * se, len(diffs)


def _table(title: str, rows: list[tuple[str, ...]]) -> None:
    print(f"\n### {title}")
    ncols = len(rows[0])
    widths = [max(len(r[i]) for r in rows) for i in range(ncols)]
    for i, r in enumerate(rows):
        print("| " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(r)) + " |")
        if i == 0:
            print("|" + "|".join("-" * (w + 2) for w in widths) + "|")


def report_simple(sweep: str, baseline: str | None = None,
                  title: str | None = None) -> None:
    """One row per condition, scored on both metrics.

    Second-half is primary throughout, matching the canonical table: a memory
    system that has not yet accumulated a graph is not the system under test, and
    scoring the cold start penalises retrieval for the one thing it cannot avoid.
    Whole-run is kept beside it because it is what a fixed-length deployment
    actually experiences, and the two disagree by enough to flip a sign.
    """
    rows = load(sweep)
    if not rows:
        print(f"\n### {sweep}: no results (run `_run_sweeps.py {sweep}`)")
        return
    half = by_condition(rows, "success_2h")
    whole = by_condition(rows, "success")
    tok = by_condition(rows, "tokens")
    base = half.get(baseline) if baseline else None

    table = [("condition", "success (2nd half)", "whole run", "tokens",
              "paired vs " + (baseline or "-"))]
    for name in half:
        mean, band, n = summarise(half[name])
        wmean, _, _ = summarise(whole[name])
        tmean, _, _ = summarise(tok[name])
        if base is not None and name != baseline:
            d, dband, _ = paired(base, half[name])
            sig = "significant" if abs(d) > dband else "n.s."
            delta = f"{d:+.3f} +/-{dband:.3f} {sig}"
        else:
            delta = "-"
        table.append((name, f"{mean:.3f} +/-{band:.3f} (n={n})", f"{wmean:.3f}",
                      f"{tmean:6.0f}", delta))
    _table(title or sweep, table)


def report_canonical() -> None:
    """The headline arms, reported on *both* metrics -- because they disagree.

    Which half of the run you score is not a presentational detail here: it flips
    the sign of the retrieval effect. Reverie starts with an empty graph and has
    to fill it, while a recency window is useful from the second task onward, so
    whole-run scoring charges Reverie for a cold start that a deployed system
    pays once and a benchmark re-pays every run. Second-half scoring is the
    defensible choice for a steady-state claim, and it is what the numbers
    recorded before 2026-09-04 used -- but they did not say so, and reading them
    as whole-run makes the system look worse than replay.

    So both are printed, always, and any quoted figure has to name its metric.
    """
    rows = load("canonical")
    if not rows:
        print("\n### canonical: no results")
        return
    tok = by_condition(rows, "tokens")

    for metric, note in (("success_2h", "second half of run -- steady state, the headline"),
                         ("success", "whole run -- includes Reverie's cold start")):
        cond = by_condition(rows, metric)
        table = [("arm", metric, "tokens/prompt", "")]
        for name in ["none", "replay", "reverie", "reverie+human"]:
            if name not in cond:
                continue
            mean, band, n = summarise(cond[name])
            tmean, _, _ = summarise(tok[name])
            table.append((name, f"{mean:.3f} +/-{band:.3f} (n={n})", f"{tmean:,.0f}", ""))
        _table(f"Canonical numbers -- {note}", table)

        print("\n  paired over identical seeds:")
        for label, a, b in [
            ("retrieval  (reverie - replay)", "replay", "reverie"),
            ("human review (+human - reverie)", "reverie", "reverie+human"),
            ("memory overall (+human - replay)", "replay", "reverie+human"),
        ]:
            if a in cond and b in cond:
                d, band, n = paired(cond[a], cond[b])
                sig = "significant" if abs(d) > band else "NOT significant"
                print(f"    {label:34s} {d:+.3f} +/-{band:.3f}  (n={n})  {sig}")


def report_attribution() -> None:
    rows = load("attribution")
    if not rows:
        print("\n### attribution: no results")
        return
    cond = by_condition(rows, "success_2h")
    table = [("landscape", "attr + utility", "attr, no utility", "no attribution")]
    for land in ("static", "drift", "fallible"):
        cells = []
        for arm in ("attr+utility", "attr_no_util", "no_attr"):
            key = f"{land}/{arm}"
            if key in cond:
                m, b, _ = summarise(cond[key])
                cells.append(f"{m:.3f} +/-{b:.3f}")
            else:
                cells.append("-")
        table.append((land, *cells))
    _table("Attribution ablation (E7/E9)", table)

    print("\nAttribution lift, paired (attr+utility - no_attr):")
    for land in ("static", "drift", "fallible"):
        a, b = f"{land}/no_attr", f"{land}/attr+utility"
        if a in cond and b in cond:
            d, band, n = paired(cond[a], cond[b])
            sig = "significant" if abs(d) > band else "NOT significant"
            print(f"  {land:10s} {d:+.3f} +/-{band:.3f} (n={n})  {sig}")


def report_density() -> None:
    rows = load("density")
    if not rows:
        print("\n### density: no results")
        return
    # Second-half success: the density claim is about the *learned* rate, so the
    # cold start is excluded.
    cond = by_condition(rows, "success_2h")
    # tasks/context is n_tasks / n_contexts, computed rather than asserted: the
    # figure recorded for the 1200-context row before 2026-09-04 was 1.0, which is
    # arithmetically impossible at a fixed 600-task budget (600/1200 = 0.5).
    n_tasks = rows[0]["n_tasks"]
    table = [("contexts (tasks/context)", "no human", "human f=0.5", "replay (1200 tok)")]
    for label in ("72", "300", "1200"):
        per = n_tasks / int(label)
        cells = []
        for arm in ("none", "human", "replay"):
            key = f"{label}/{arm}"
            m, b, _ = summarise(cond[key]) if key in cond else (float("nan"),) * 3
            cells.append(f"{m:.3f} +/-{b:.3f}")
        table.append((f"{label} ({per:.1f})", *cells))
    _table("Repetition density (E10.1) -- second-half success, chance 0.167", table)

    print("\nReverie vs budget-matched replay, paired (positive = Reverie wins):")
    for label in ("72", "300", "1200"):
        a, b = f"{label}/replay", f"{label}/none"
        if a in cond and b in cond:
            d, band, n = paired(cond[a], cond[b])
            verdict = "Reverie wins" if d > band else ("replay wins" if -d > band else "tie")
            print(f"  {label:>4s} contexts  {d:+.3f} +/-{band:.3f} (n={n})  {verdict}")


def report_guard() -> None:
    rows = load("guard")
    if not rows:
        print("\n### guard: no results")
        return
    cond = by_condition(rows, "success_2h")
    table = [("review condition", "guard off", "guard on", "paired effect of guard")]
    for name in ("over_general", "correct_scope", "random_err", "adversarial"):
        off, on = f"{name}/guard=False", f"{name}/guard=True"
        if off not in cond or on not in cond:
            continue
        mo, bo, _ = summarise(cond[off])
        mn, bn, _ = summarise(cond[on])
        d, band, n = paired(cond[off], cond[on])
        sig = "significant" if abs(d) > band else "n.s."
        table.append((name, f"{mo:.3f} +/-{bo:.3f}", f"{mn:.3f} +/-{bn:.3f}",
                      f"{d:+.3f} +/-{band:.3f} {sig}"))
    _table("Over-generalisation guard (E12) -- H1 is the primary prediction", table)



def report_landscape() -> None:
    """Does the headline effect survive being averaged over landscapes?"""
    rows = load("landscape")
    if not rows:
        print("\n### landscape: no results")
        return
    # Steady state, matching the canonical table's metric.
    cond = by_condition(rows, "success_2h")
    lands = sorted({c.split("/")[0] for c in cond}, key=lambda x: int(x[1:]))

    table = [("landscape", "replay", "reverie", "reverie - replay (paired)")]
    diffs_ret, diffs_hum = [], []
    for L in lands:
        r, v = f"{L}/replay", f"{L}/reverie"
        mr, br, _ = summarise(cond[r])
        mv, bv, _ = summarise(cond[v])
        d, band, _ = paired(cond[r], cond[v])
        diffs_ret.append(d)
        h = f"{L}/reverie+human"
        if h in cond:
            dh, _, _ = paired(cond[v], cond[h])
            diffs_hum.append(dh)
        table.append((L, f"{mr:.3f} +/-{br:.3f}", f"{mv:.3f} +/-{bv:.3f}",
                      f"{d:+.3f} +/-{band:.3f}"))
    _table("Landscape robustness -- one row per hidden rule set", table)

    def across(name: str, vals: list[float]) -> None:
        m = statistics.fmean(vals)
        band = 2 * statistics.stdev(vals) / (len(vals) ** 0.5)
        wins = sum(1 for v in vals if v > 0)
        sig = "significant" if abs(m) > band else "NOT significant"
        print(f"  {name:32s} {m:+.3f} +/-{band:.3f} across {len(vals)} landscapes "
              f"({wins}/{len(vals)} positive)  {sig}")

    print("\nEffect averaged over landscapes (the landscape is now a random effect,")
    print("not an uncontrolled constant):")
    across("retrieval (reverie - replay)", diffs_ret)
    if diffs_hum:
        across("human review (+human - reverie)", diffs_hum)
    print(f"\n  spread of the retrieval effect: {min(diffs_ret):+.3f} to "
          f"{max(diffs_ret):+.3f} -- if this straddles zero, a single-landscape")
    print("  estimate could report either sign.")


REPORTS = {
    "canonical": report_canonical,
    "attribution": report_attribution,
    "density": report_density,
    "guard": report_guard,
    "landscape": report_landscape,
    "review_frequency": lambda: report_simple(
        "review_frequency", baseline="none", title="Review frequency (E9)"),
    "review_quality": lambda: report_simple(
        "review_quality", baseline="none", title="Review quality (E9)"),
    "error_modes": lambda: report_simple(
        "error_modes", baseline="none", title="Reviewer error modes (E10.2)"),
    "over_general": lambda: report_simple(
        "over_general", baseline="none", title="Over-generalisation (E10.3)"),
    "staleness": lambda: report_simple(
        "staleness", baseline="drift/none", title="Stale human knowledge (E10.4)"),
    "guard_baseline": lambda: report_simple(
        "guard_baseline", baseline="no_review/guard=False",
        title="Scope guard with no reviewer at all (E12 follow-up)"),
}


def main() -> None:
    names = sys.argv[1:] or list(REPORTS)
    for name in names:
        if name not in REPORTS:
            sys.exit(f"unknown sweep: {name}")
        REPORTS[name]()
    print()


if __name__ == "__main__":
    main()
