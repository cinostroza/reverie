"""Figure 1 for the paper.

    python experiments/_figure1.py

Reads ``experiments/results/*.json``; writes ``paper/figures/fig1.{pdf,png}``.

(a) Second-half success against repetition density for Reverie, Reverie with
    review, and the token-capped recency baseline (``density`` sweep).
(b) Paired effects with 95% intervals: attribution in each of its three
    conditions (``attribution`` sweep, one environment instance, 8 seeds), and
    retrieval and review averaged over eight environment instances
    (``landscape`` sweep). Each attribution condition gets its own bar; pooling
    them would hide that they are separate experiments.
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

# Okabe-Ito, matching the notebooks.
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
CHANCE = (1 / 6) * 0.95   # uniform choice over 6 strategies, less the 5% flake rate

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def panel_density(ax) -> None:
    rows = load("density")
    cond = by_condition(rows, "success_2h")
    n_tasks = rows[0]["n_tasks"]
    sizes = [1200, 300, 72]
    x = [n_tasks / s for s in sizes]

    for label, arm, colour, marker, style in [
        ("Reverie with review", "human", GREEN, "s", "-"),
        ("Reverie", "none", BLUE, "o", "-"),
        ("Replay (1,200 tokens)", "replay", ORANGE, "^", "--"),
    ]:
        stats = [summarise(cond[f"{s}/{arm}"]) for s in sizes]
        ax.errorbar(x, [m for m, _, _ in stats], yerr=[b for _, b, _ in stats],
                    label=label, color=colour, marker=marker, ms=4, lw=1.6,
                    ls=style, capsize=2.5, elinewidth=0.9)

    ax.axhline(CHANCE, color=GREY, lw=0.8, ls=":", zorder=0)
    ax.text(x[0] * 0.92, CHANCE + 0.004, f"chance ({CHANCE:.3f})", ha="left",
            va="bottom", fontsize=7.5, color=GREY)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([f"{xi:.1f}" for xi in x])
    ax.set_xlim(x[0] * 0.8, x[-1] * 1.25)
    ax.set_xlabel("Tasks per distinct context (log scale)")
    ax.set_ylabel("Success rate, tasks 301–600")
    ax.legend(loc="upper left")
    ax.text(-0.14, 1.02, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


def panel_effects(ax) -> None:
    bars = []

    attr = by_condition(load("attribution"), "success_2h")
    for land, label in [("static", "Attribution, static"),
                        ("drift", "Attribution, drift"),
                        ("fallible", "Attribution, fallible lessons")]:
        d, band, _ = paired(attr[f"{land}/no_attr"], attr[f"{land}/attr+utility"])
        bars.append((label, d, band, ORANGE))

    land = by_condition(load("landscape"), "success_2h")
    instances = sorted({c.split("/")[0] for c in land}, key=lambda s: int(s[1:]))

    def across(a: str, b: str) -> tuple[float, float]:
        diffs = [paired(land[f"{i}/{a}"], land[f"{i}/{b}"])[0] for i in instances]
        return statistics.fmean(diffs), 2 * statistics.stdev(diffs) / len(diffs) ** 0.5

    bars.append(("Retrieval vs. replay", *across("replay", "reverie"), BLUE))
    bars.append(("Review ($f = 0.5$) vs. Reverie", *across("reverie", "reverie+human"),
                 GREEN))

    y = list(range(len(bars)))
    ax.barh(y, [b[1] for b in bars], xerr=[b[2] for b in bars],
            color=[b[3] for b in bars], height=0.55, alpha=0.85,
            error_kw=dict(ecolor="#333333", capsize=2.5, elinewidth=0.9))
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlabel("Paired change in success rate (95% interval)")
    ax.grid(axis="y", visible=False)
    for i, (_, m, band, _c) in enumerate(bars):
        ax.text(max(m + band, 0) + 0.008, i, f"{m:+.3f}".replace("-", "\N{MINUS SIGN}"),
                va="center", fontsize=8)
    ax.set_xlim(min(b[1] - b[2] for b in bars) - 0.02,
                max(b[1] + b[2] for b in bars) + 0.06)
    ax.text(-0.52, 1.02, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    panel_density(axes[0])
    panel_effects(axes[1])
    fig.tight_layout(w_pad=3.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1.{ext}")
    print(f"wrote {OUT / 'fig1.pdf'} and .png")


if __name__ == "__main__":
    main()
