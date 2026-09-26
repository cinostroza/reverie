"""Figure 1 for the paper: what the benchmark controls, and what the levers are worth.

    python experiments/_figure1.py

Reads ``experiments/results/*.json``; writes ``paper/figures/fig1.{pdf,png}``.

Many reviewers read Figure 1 and the abstract and nothing else, so it has to
carry both of the paper's claims on its own:

**Left -- the methodological claim.** Whether selective retrieval beats a recency
window is decided by repetition density, not by how good the retrieval is. The
crossover is the point of the benchmark, so the panel is built around it rather
than around the arms.

**Right -- the empirical claim.** The three levers as *paired* effects with
their cluster-robust 95% bands. Paired, because seed variance swamps every
effect here; cluster-robust, because within-run outcomes are correlated. A bar
whose band crosses zero is a null, and attribution's does.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _analyse import by_condition, load, paired, summarise  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

# Okabe-Ito, matching the notebooks so every figure in the paper is one family.
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def panel_density(ax) -> None:
    rows = load("density")
    if not rows:
        ax.text(0.5, 0.5, "no density results", ha="center", va="center")
        return
    cond = by_condition(rows, "success_2h")
    n_tasks = rows[0]["n_tasks"]
    sizes = [72, 300, 1200]
    x = [n_tasks / s for s in sizes]

    series = [
        ("Retrieval (Reverie)", "none", BLUE, "o", "-"),
        ("+ human review", "human", GREEN, "s", "-"),
        ("Replay (1200 tok)", "replay", ORANGE, "^", "--"),
    ]
    curves = {}
    for label, arm, colour, marker, style in series:
        means, bands = [], []
        for s in sizes:
            m, b, _ = summarise(cond[f"{s}/{arm}"])
            means.append(m)
            bands.append(b)
        curves[arm] = means
        ax.errorbar(x, means, yerr=bands, label=label, color=colour, marker=marker,
                    ms=4.5, lw=1.8, ls=style, capsize=2.5, elinewidth=1)

    # Shade where the recency window is at least as good as selective retrieval:
    # that region is the finding, so it is drawn rather than described.
    lose = [xi for xi, r, v in zip(x, curves["replay"], curves["none"]) if r >= v]
    if lose:
        ax.axvspan(min(x) * 0.8, max(lose) * 1.45, color=GREY, alpha=0.10, lw=0)
        # Right-aligned at the top of the shaded band, clear of the legend.
        ax.text(max(lose) * 1.4, ax.get_ylim()[1], "replay wins ", va="top", ha="right",
                fontsize=8, color=GREY, style="italic")

    ax.axhline(1 / 6, color=GREY, lw=0.9, ls=":", zorder=0)
    ax.text(max(x) * 0.98, 1 / 6, "chance ", ha="right", va="bottom",
            fontsize=7.5, color=GREY)

    ax.set_xscale("log")
    # Log scale volunteers its own minor tick labels, which collide with ours.
    ax.set_xticks(x)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([f"{xi:.1f}\n({s} ctx)" for xi, s in zip(x, sizes)])
    ax.set_xlabel("repetition density (tasks per distinct context)")
    ax.set_ylabel("success rate, second half of run")
    ax.set_title("Density decides whether memory beats recency", loc="left")
    ax.legend(loc="upper left", fontsize=8)


def panel_levers(ax) -> None:
    """Every bar is a *paired* effect on second-half success, the same metric
    as the left panel. Retrieval and review come from the 8-landscape sweep
    rather than the single-landscape canonical run, because a one-landscape
    estimate of the retrieval effect is not identified -- it ranges from
    -0.007 to +0.061 depending on which hidden rule set you happen to draw.
    """
    bars = []

    attribution = load("attribution")
    if attribution:
        cond = by_condition(attribution, "success_2h")
        # The paired lift in each of the three ablations, pooled: the claim is
        # that attribution is worth nothing anywhere, not on average.
        diffs = []
        for land in ("static", "drift", "fallible"):
            a, b = f"{land}/no_attr", f"{land}/attr+utility"
            if a in cond and b in cond:
                diffs.append(paired(cond[a], cond[b]))
        if diffs:
            m = statistics.fmean(d[0] for d in diffs)
            band = max(d[1] for d in diffs)
            bars.append(("Outcome\nattribution", m, band, ORANGE))

    landscape = load("landscape")
    if landscape:
        cond = by_condition(landscape, "success_2h")
        lands = sorted({c.split("/")[0] for c in cond}, key=lambda x: int(x[1:]))

        def across(a: str, b: str) -> tuple[float, float]:
            vals = [paired(cond[f"{L}/{a}"], cond[f"{L}/{b}"])[0] for L in lands]
            return (statistics.fmean(vals),
                    2 * statistics.stdev(vals) / (len(vals) ** 0.5))

        m, band = across("replay", "reverie")
        bars.append(("Retrieval\nvs replay", m, band, BLUE))
        m, band = across("reverie", "reverie+human")
        bars.append(("Human\nreview", m, band, GREEN))

    if not bars:
        ax.text(0.5, 0.5, "no results", ha="center", va="center")
        return

    y = range(len(bars))
    ax.barh(list(y), [b[1] for b in bars], xerr=[b[2] for b in bars],
            color=[b[3] for b in bars], height=0.55, alpha=0.85,
            error_kw=dict(ecolor="#333333", capsize=3, elinewidth=1.1))
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels([b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlabel("paired change in success rate (95% CI)")
    ax.set_title("What each lever is actually worth", loc="left")

    for i, (_, m, band, _c) in enumerate(bars):
        null = abs(m) <= band
        ax.text(m + (band + 0.008) * (1 if m >= 0 else -1), i,
                f"{m:+.3f}" + ("  (null)" if null else ""),
                va="center", ha="left" if m >= 0 else "right", fontsize=8)
    lo = min(b[1] - b[2] for b in bars)
    hi = max(b[1] + b[2] for b in bars)
    ax.set_xlim(lo - 0.09, hi + 0.09)
    ax.grid(axis="y", visible=False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    panel_density(axes[0])
    panel_levers(axes[1])
    fig.tight_layout(w_pad=2.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1.{ext}")
    print(f"wrote {OUT / 'fig1.pdf'} and .png")


if __name__ == "__main__":
    main()
