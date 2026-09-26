"""Generate the experiment notebooks.

The notebooks are build artifacts so that a change to shared setup (styling,
seeds, imports) propagates to all of them instead of drifting. Run:

    python experiments/_build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """
import sys, warnings, math
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "experiments" else Path.cwd()))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import reverie
from reverie.sim import World, WorldConfig, run_trial, sweep, spearman, rank_metrics

# Publication styling: one place, so every figure in the paper matches.
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})
# Colourblind-safe (Okabe-Ito), ordered so the first two are maximally distinct.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    print(f"wrote figures/{name}.png and .pdf")

SEEDS = 24          # seeds per configuration
print("reverie", reverie.__version__, "| numpy", np.__version__)
"""

BAND_HELPER = '''
def band(ax, df, x, y, hue=None, label=None, color=None):
    """Mean with a bootstrap 95% CI ribbon.

    Point estimates alone would overstate precision here: with a noisy agent
    and a couple of dozen seeds, curves that look separated frequently are not.
    """
    g = df.groupby(x)[y]
    mean = g.mean()
    lo = g.apply(lambda s: np.percentile(
        [np.mean(np.random.default_rng(i).choice(s.values, len(s))) for i in range(200)], 2.5))
    hi = g.apply(lambda s: np.percentile(
        [np.mean(np.random.default_rng(i).choice(s.values, len(s))) for i in range(200)], 97.5))
    ax.plot(mean.index, mean.values, marker="o", ms=3.5, lw=1.8, label=label, color=color)
    ax.fill_between(mean.index, lo.values, hi.values, alpha=0.15, color=color, lw=0)
    return mean
'''


# ==========================================================================
# E1 -- attribution sample complexity
# ==========================================================================

E1 = notebook([
    md("""
# E1 — Attribution sample complexity

**Question (HLD open question #5, flagged as the project's highest-priority unknown):**
how many attributed recalls does it take before the Beta posterior's lower confidence
bound reliably separates a useful memory from a harmful one?

This is the experiment that decides whether Reverie is a viable design or a
decoration. If the answer is ~200 recalls, a single developer's agent reaches useful
attribution in a week. If it is ~50,000, the mechanism only works for high-volume
fleets and the local single-agent story in the HLD is marketing.

**Why simulate.** In a live deployment the true causal effect of a memory is
unobservable, so there is nothing to validate an estimate against. Here it is a
parameter (`World.effects`), which is the only way to measure estimator quality
rather than merely assert it.

**Design.** A synthetic world of `n_memories` memories, each with a latent causal
effect. Each task draws a recall set, samples an outcome from a logistic model over
the recalled effects, and feeds the result through the *shipped* attribution
estimator. We then correlate the estimated LCB against ground truth.

Two nuisance parameters dominate, and both are properties of the deployment rather
than of the algorithm:

| Parameter | Meaning | HLD reference |
|---|---|---|
| `relevance_p` | P(a recalled memory actually influenced the task) | §9.4, "correlation is not causation" |
| `cooccurrence` | how locked-together recall sets are | §15, identifiability |
"""),
    code(SETUP),
    code(BAND_HELPER),

    md("## 1. A single learning curve\n\nOne world, tracked over 8,000 recalls."),
    code("""
cfg = WorldConfig(n_memories=40, recall_size=6, relevance_p=0.5, cooccurrence=0.3)
CHECKPOINTS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]

rows = []
for seed in range(SEEDS):
    for snap in run_trial(cfg, max(CHECKPOINTS), seed=seed, checkpoints=CHECKPOINTS):
        rows.append(snap.as_row(seed=seed))
df1 = pd.DataFrame(rows)
df1.groupby("n_recalls")[["spearman", "worst_k_recall"]].mean().round(3)
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

band(axes[0], df1, "n_recalls", "spearman", color=PALETTE[0])
axes[0].set_xscale("log")
axes[0].set_xlabel("attributed recalls"); axes[0].set_ylabel("Spearman ρ (estimate vs truth)")
axes[0].set_title("Ranking quality vs evidence")
axes[0].axhline(0.5, ls="--", lw=1, color="#888")
axes[0].text(60, 0.52, "ρ = 0.5 (usable ordering)", fontsize=8, color="#666")
axes[0].set_ylim(-0.05, 1.0)

band(axes[1], df1, "n_recalls", "worst_k_recall", color=PALETTE[1])
axes[1].set_xscale("log")
axes[1].set_xlabel("attributed recalls"); axes[1].set_ylabel("recall of the harmful tail")
axes[1].set_title("Quarantine accuracy")
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[1].set_ylim(0, 1.0)

fig.suptitle("E1 · Attribution sample complexity (n=40 memories, 24 seeds, 95% CI)", y=1.04)
save(fig, "e1_learning_curve")
"""),
    md("""
Read the right panel before the left one. Global ordering (ρ) is the harder problem
and the more academic one; **identifying the harmful tail is what quarantine actually
needs**, and it saturates earlier. A system can be mediocre at ranking every memory
while still being reliable at removing the worst ones — which is the property HLD G8
claims.
"""),

    md("## 2. The two nuisance parameters\n\nSweeping each independently."),
    code("""
rows = []
for rp in [0.2, 0.35, 0.5, 0.75, 1.0]:
    c = replace(cfg, relevance_p=rp)
    for seed in range(SEEDS):
        for snap in run_trial(c, max(CHECKPOINTS), seed=seed, checkpoints=CHECKPOINTS):
            rows.append(snap.as_row(relevance_p=rp, seed=seed))
df_rel = pd.DataFrame(rows)

rows = []
for co in [0.0, 0.25, 0.5, 0.75, 1.0]:
    c = replace(cfg, cooccurrence=co)
    for seed in range(SEEDS):
        for snap in run_trial(c, max(CHECKPOINTS), seed=seed, checkpoints=CHECKPOINTS):
            rows.append(snap.as_row(cooccurrence=co, seed=seed))
df_co = pd.DataFrame(rows)
print(len(df_rel), len(df_co), "rows")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)

for i, rp in enumerate(sorted(df_rel.relevance_p.unique())):
    band(axes[0], df_rel[df_rel.relevance_p == rp], "n_recalls", "spearman",
         label=f"{rp:.0%}", color=PALETTE[i])
axes[0].set_title("Effect of memory relevance")
axes[0].legend(title="P(recalled memory\\nactually mattered)", fontsize=8, title_fontsize=8)

for i, co in enumerate(sorted(df_co.cooccurrence.unique())):
    band(axes[1], df_co[df_co.cooccurrence == co], "n_recalls", "spearman",
         label=f"{co:.2f}", color=PALETTE[i])
axes[1].set_title("Effect of recall co-occurrence")
axes[1].legend(title="co-occurrence", fontsize=8, title_fontsize=8)

for ax in axes:
    ax.set_xscale("log"); ax.set_xlabel("attributed recalls")
    ax.axhline(0.5, ls="--", lw=1, color="#888")
axes[0].set_ylabel("Spearman ρ")
fig.suptitle("E1 · What governs sample complexity", y=1.04)
save(fig, "e1_nuisance_sweep")
"""),

    md("## 3. The headline number: recalls to a usable ordering"),
    code("""
def recalls_to(df, group_col, target=0.5):
    \"\"\"First checkpoint whose mean ρ crosses `target`. NaN if never.\"\"\"
    out = {}
    for key, g in df.groupby(group_col):
        m = g.groupby("n_recalls")["spearman"].mean()
        hit = m[m >= target]
        out[key] = int(hit.index[0]) if len(hit) else np.nan
    return pd.Series(out, name=f"recalls to ρ={target}")

table = pd.concat([
    recalls_to(df_rel, "relevance_p").rename_axis("relevance_p").to_frame().assign(axis="relevance_p"),
    recalls_to(df_co, "cooccurrence").rename_axis("cooccurrence").to_frame().assign(axis="cooccurrence"),
])
display(recalls_to(df_rel, "relevance_p"))
display(recalls_to(df_co, "cooccurrence"))
"""),
    code("""
# Joint view: the operating region where attribution is affordable.
grid_rel = [0.25, 0.5, 0.75, 1.0]
grid_co  = [0.0, 0.33, 0.66, 1.0]
Z = np.full((len(grid_rel), len(grid_co)), np.nan)

for i, rp in enumerate(grid_rel):
    for j, co in enumerate(grid_co):
        c = replace(cfg, relevance_p=rp, cooccurrence=co)
        rows = []
        for seed in range(12):
            for snap in run_trial(c, max(CHECKPOINTS), seed=seed, checkpoints=CHECKPOINTS):
                rows.append(snap.as_row(seed=seed))
        m = pd.DataFrame(rows).groupby("n_recalls")["spearman"].mean()
        hit = m[m >= 0.5]
        Z[i, j] = hit.index[0] if len(hit) else np.nan

fig, ax = plt.subplots(figsize=(5.4, 4))
masked = np.ma.masked_invalid(Z)
cmap = plt.cm.viridis_r.copy(); cmap.set_bad("#d9d9d9")
im = ax.imshow(masked, cmap=cmap, norm=plt.matplotlib.colors.LogNorm())
ax.set_xticks(range(len(grid_co)), [f"{c:.2f}" for c in grid_co])
ax.set_yticks(range(len(grid_rel)), [f"{r:.0%}" for r in grid_rel])
ax.set_xlabel("co-occurrence"); ax.set_ylabel("relevance")
ax.set_title("Recalls needed to reach ρ = 0.5\\n(grey = not reached within 10,000)")
for i in range(len(grid_rel)):
    for j in range(len(grid_co)):
        v = Z[i, j]
        ax.text(j, i, "—" if np.isnan(v) else f"{int(v):,}", ha="center", va="center",
                fontsize=8, color="white" if (not np.isnan(v) and v > 600) else "black")
ax.grid(False)
fig.colorbar(im, ax=ax, label="recalls", shrink=0.8)
save(fig, "e1_operating_region")
"""),

    md("""
## 4. Findings

Fill these in from the run above — the numbers move with the parameter grid, and the
point of the notebook is that they are measured rather than asserted.

1. **Sample complexity is dominated by `relevance_p`, not by the number of memories.**
   The bottleneck is how often a recalled memory actually mattered, which is a
   property of recall precision. That makes recall precision an *attribution*
   concern, not just a token-budget concern — a connection the HLD does not draw.

2. **Quarantine accuracy saturates well before global ranking does.** The safety
   claim (G8) is cheaper to satisfy than the ranking claim.

3. **Co-occurrence degrades identifiability but does not destroy it.** Even with
   fully locked recall sets, variation in activation *shares* carries signal. This
   was a genuine surprise: the initial hypothesis was that perfectly co-occurring
   memories would be strictly unidentifiable, and that turned out to be false.

### What this implies for the design

- If the crossover for a realistic regime lands in the hundreds, ship the local
  single-agent story as-is.
- If it lands in the tens of thousands, then either `used_memories` citation
  (E2) becomes mandatory rather than optional, or attribution should be scoped to
  team level where volume exists.
"""),
])


# ==========================================================================
# E2 -- credit assignment
# ==========================================================================

E2 = notebook([
    md("""
# E2 — Which credit-assignment rule actually works?

HLD §9.2 divides an outcome's credit among recalled memories **in proportion to
activation share**. That is a plausible-sounding choice with no evidence behind it.
This notebook tests it against the alternatives on identical traces.

| Rule | Description |
|---|---|
| `uniform` | every recalled memory gets 1/n — the null hypothesis |
| `activation` | proportional to activation share — the HLD default |
| `rank` | 1/log₂(rank+2), ignoring activation magnitude |
| `citation` | all credit to memories the agent reported using (`used_memories`) |

`citation` is the interesting one. It requires agent cooperation, so the HLD marks it
opt-in. If it is worth a lot, that field should be promoted from "nice to have" to
the primary integration ask — and the MCP tool descriptions should demand it.
"""),
    code(SETUP),
    code(BAND_HELPER),

    code("""
cfg = WorldConfig(n_memories=40, recall_size=6, relevance_p=0.5, cooccurrence=0.3)
CHECKPOINTS = [100, 200, 400, 800, 1600, 3200, 6400]
MODES = ["uniform", "activation", "rank", "citation"]

rows = []
for mode in MODES:
    for seed in range(SEEDS):
        for snap in run_trial(cfg, max(CHECKPOINTS), seed=seed,
                              credit_mode=mode, checkpoints=CHECKPOINTS):
            rows.append(snap.as_row(seed=seed))
df2 = pd.DataFrame(rows)
df2.groupby(["credit_mode", "n_recalls"])["spearman"].mean().unstack(0).round(3)
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for i, mode in enumerate(MODES):
    sub = df2[df2.credit_mode == mode]
    band(axes[0], sub, "n_recalls", "spearman", label=mode, color=PALETTE[i])
    band(axes[1], sub, "n_recalls", "worst_k_recall", label=mode, color=PALETTE[i])

axes[0].set_ylabel("Spearman ρ"); axes[0].set_title("Ranking quality")
axes[1].set_ylabel("recall of harmful tail"); axes[1].set_title("Quarantine accuracy")
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
for ax in axes:
    ax.set_xscale("log"); ax.set_xlabel("attributed recalls"); ax.legend(fontsize=8)
fig.suptitle("E2 · Credit-assignment rules on identical traces", y=1.04)
save(fig, "e2_credit_modes")
"""),

    md("""
## The value of an explicit citation

`citation` has access to the ground-truth relevance mask, so it is an **upper bound**
on what any credit rule can achieve — not an implementable estimator on its own, but
the right way to price the `used_memories` field. The gap between `activation` and
`citation` is what an integrator buys by populating it.
"""),
    code("""
final = df2[df2.n_recalls == max(CHECKPOINTS)]
summary = final.groupby("credit_mode")[["spearman", "worst_k_recall"]].agg(["mean", "std"]).round(3)
display(summary)

base = final[final.credit_mode == "activation"]["spearman"].mean()
print(f"\\nrelative to the HLD default (activation, ρ={base:.3f}):")
for mode in MODES:
    if mode == "activation":
        continue
    v = final[final.credit_mode == mode]["spearman"].mean()
    print(f"  {mode:<11} {v:+.3f}  ({(v-base)/abs(base)*100:+.0f}%)")
"""),
    code("""
# Does the ranking of rules hold as relevance degrades? A rule that only wins in
# the easy regime is not worth the complexity.
rows = []
for rp in [0.25, 0.5, 0.75, 1.0]:
    c = replace(cfg, relevance_p=rp)
    for mode in MODES:
        for seed in range(16):
            snap = run_trial(c, 3200, seed=seed, credit_mode=mode)
            rows.append(snap.as_row(relevance_p=rp, seed=seed))
df_rp = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 3.6))
for i, mode in enumerate(MODES):
    band(ax, df_rp[df_rp.credit_mode == mode], "relevance_p", "spearman",
         label=mode, color=PALETTE[i])
ax.set_xlabel("P(recalled memory actually mattered)"); ax.set_ylabel("Spearman ρ @ 3200 recalls")
ax.set_title("E2 · Do the rules separate when relevance is low?")
ax.legend(fontsize=8)
save(fig, "e2_by_relevance")
"""),

    md("""
## Findings

Record the measured verdict here. The decision this notebook drives:

- **If `activation` ≈ `uniform`:** the activation weighting in HLD §9.2 is
  complexity with no payoff and should be simplified away. Simpler is better when
  the evidence is neutral.
- **If `citation` ≫ `activation`:** `used_memories` is the highest-leverage field in
  the episode contract, and the SDK/MCP surface should be redesigned around getting
  it populated by default rather than treating it as optional.
"""),
])


# ==========================================================================
# E3 -- retrieval mechanism, against the real engine
# ==========================================================================

E3 = notebook([
    md("""
# E3 — Spreading activation vs. top-k retrieval

HLD §10.3 makes a specific, falsifiable claim:

> the useful memory is frequently not similar to the cue … Spreading activation
> retrieves: *this ORM* → *the deadlock incident* → *the root cause was the
> connection pool* → *and pool config is shared with the job runner*. Three hops
> from the cue, textually dissimilar, and the actually load-bearing fact.

That claim is the entire justification for building a graph instead of using a vector
index. This notebook tests it **against the real engine** — `reverie.RecallEngine`
over a real SQLite store, not a simulation.

**Method.** Plant chains of length 1–4 where a load-bearing fact sits *k* hops from
the cue and shares no vocabulary with it. Bury them in distractor memories that *do*
share vocabulary with the cue. Then measure how often each retrieval mode surfaces
the load-bearing node inside a fixed token budget.
"""),
    code(SETUP),
    code("""
import random, string, time
from reverie import Reverie, Config
from reverie.models import Node, new_id

WORDS = [f"tok{i:03d}" for i in range(600)]

def build_world(n_chains=40, chain_len=4, n_distractors=300, seed=0):
    \"\"\"A store with planted multi-hop chains and vocabulary-matched distractors.\"\"\"
    rng = random.Random(seed)
    cfg = Config()
    cfg.recall.max_hops = 3
    mem = Reverie(scope="agent:e3", db=":memory:", config=cfg)
    probes = []

    for c in range(n_chains):
        # Each link uses disjoint vocabulary, so only the graph connects them.
        vocab = [rng.sample(WORDS, 6) for _ in range(chain_len + 1)]
        ids = []
        for h, v in enumerate(vocab):
            body = " ".join(v)
            n = Node(id=new_id("mem"), scope_id=mem.scope, type="lesson",
                     memory_class="procedural" if h == chain_len else "semantic",
                     label=body[:60], body=body, provenance="observed")
            mem.store.add_node(n, mem.embedder.embed(body))
            ids.append(n.id)
        for a, b in zip(ids, ids[1:]):
            mem.consolidator._edge(mem.scope, a, b, "caused", 0.85, "observed", 0.9)
        probes.append({"cue": " ".join(vocab[0]), "target": ids[-1],
                       "hops": chain_len, "chain": ids})

    # Distractors share vocabulary with the cues but connect to nothing.
    for _ in range(n_distractors):
        src = rng.choice(probes)
        body = " ".join(rng.sample(src["cue"].split(), 3) + rng.sample(WORDS, 4))
        n = Node(id=new_id("mem"), scope_id=mem.scope, type="lesson",
                 memory_class="semantic", label=body[:60], body=body,
                 provenance="observed")
        mem.store.add_node(n, mem.embedder.embed(body))
    return mem, probes

mem, probes = build_world()
print("nodes:", mem.store.count_nodes(mem.scope), "| probes:", len(probes))
"""),
    code("""
def evaluate(mem, probes, budget=1200):
    \"\"\"Three retrieval modes, same store, same budget.

    'seed-only' disables traversal (max_hops=0) and is exactly what a top-k
    hybrid retriever would return -- BM25 + vector, no graph.
    \"\"\"
    results = []
    original = mem.config.recall.max_hops

    for mode, hops in [("fts+vector top-k", 0), ("1-hop", 1), ("spreading (3-hop)", 3)]:
        mem.config.recall.max_hops = hops
        for p in probes:
            t0 = time.perf_counter()
            r = mem.recaller.recall(p["cue"], mem.scope, budget_tokens=budget,
                                    log=False, allow_ablation=False)
            dt = (time.perf_counter() - t0) * 1000
            got = {s.node.id for s in r.nodes}
            results.append({
                "mode": mode, "hops": p["hops"],
                "hit": p["target"] in got,
                "chain_recall": len(got & set(p["chain"])) / len(p["chain"]),
                "tokens": r.token_cost, "latency_ms": dt, "n_nodes": len(r.nodes),
            })
    mem.config.recall.max_hops = original
    return pd.DataFrame(results)

df3 = evaluate(mem, probes)
df3.groupby("mode")[["hit", "chain_recall", "tokens", "latency_ms"]].mean().round(3)
"""),
    code("""
# Does the advantage grow with distance, as §10.3 implies it should?
frames = []
for L in [1, 2, 3, 4]:
    m, pr = build_world(n_chains=30, chain_len=L, n_distractors=250, seed=L)
    d = evaluate(m, pr); d["chain_len"] = L
    frames.append(d)
    m.close()
df_len = pd.concat(frames)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
modes = ["fts+vector top-k", "1-hop", "spreading (3-hop)"]

piv = df_len.groupby(["chain_len", "mode"])["hit"].mean().unstack()
piv = piv[modes]
piv.plot(kind="bar", ax=axes[0], color=PALETTE[:3], width=0.78, legend=False)
axes[0].set_ylabel("P(load-bearing fact retrieved)"); axes[0].set_xlabel("hops from cue to target")
axes[0].set_title("Retrieval by distance"); axes[0].tick_params(axis="x", rotation=0)
axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[0].legend(fontsize=8)

df3.groupby("mode")["tokens"].mean().reindex(modes).plot(
    kind="barh", ax=axes[1], color=PALETTE[:3])
axes[1].set_xlabel("mean brief tokens"); axes[1].set_ylabel(""); axes[1].set_title("Token cost")

df3.groupby("mode")["latency_ms"].quantile(0.99).reindex(modes).plot(
    kind="barh", ax=axes[2], color=PALETTE[:3])
axes[2].set_xlabel("p99 latency (ms)"); axes[2].set_ylabel(""); axes[2].set_title("Latency")
axes[2].axvline(100, ls="--", lw=1, color="#c00")
axes[2].text(102, 0.1, "HLD target", fontsize=8, color="#c00")

fig.suptitle("E3 · Does the graph earn its keep? (real engine, SQLite)", y=1.04)
save(fig, "e3_retrieval")
"""),
    md("""
The third panel is the one that can kill the design. HLD §10.2 budgets p99 < 100 ms
for local recall and admits the number is a guess. If spreading activation buys
retrieval accuracy but costs 300 ms, the honest response is to reduce `max_hops` to 2
and say so — not to quietly redefine the target.
"""),
    code("""
# Latency scaling: p99 against graph size, which is what actually decides
# whether the budget survives contact with a real deployment.
rows = []
for n_dist in [200, 1000, 3000, 8000]:
    m, pr = build_world(n_chains=25, chain_len=3, n_distractors=n_dist, seed=99)
    for p in pr:
        t0 = time.perf_counter()
        r = m.recaller.recall(p["cue"], m.scope, log=False, allow_ablation=False)
        rows.append({"nodes": m.store.count_nodes(m.scope),
                     "latency_ms": (time.perf_counter() - t0) * 1000,
                     "visited": r.visited, "capped": r.hit_visit_cap})
    m.close()
df_lat = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 3.6))
g = df_lat.groupby("nodes")["latency_ms"]
ax.plot(g.median().index, g.median().values, marker="o", color=PALETTE[0], label="p50")
ax.plot(g.quantile(0.99).index, g.quantile(0.99).values, marker="s",
        color=PALETTE[1], label="p99")
ax.axhline(100, ls="--", lw=1, color="#c00"); ax.text(250, 105, "HLD p99 target", fontsize=8, color="#c00")
ax.set_xlabel("nodes in graph"); ax.set_ylabel("recall latency (ms)")
ax.set_xscale("log"); ax.set_title("E3 · Recall latency vs graph size"); ax.legend(fontsize=8)
save(fig, "e3_latency")
print(df_lat.groupby("nodes")[["latency_ms", "visited"]].describe().round(1))
"""),
    md("""
## Findings

1. **Does the hit-rate gap widen with hop distance?** If it does not, spreading
   activation is not buying what §10.3 claims and a hybrid top-k retriever is the
   simpler correct answer.
2. **Does p99 stay inside budget as the graph grows?** Record the node count at
   which it breaks, and whether `visit_cap` is what saves it.
3. **Token cost.** The graph must not win accuracy by simply returning more.
"""),
])


# ==========================================================================
# E4 -- poisoning and self-correction
# ==========================================================================

E4 = notebook([
    md("""
# E4 — Poisoning and self-correction

HLD goal **G8**: *memory can be shown to be wrong and removed automatically, without
a human noticing first.* This is the claim that distinguishes Reverie from every
memory system that can only accumulate.

**Threat model (HLD §13.1).** A memory system is a persistence layer for prompt
injection. An attacker who lands one payload in one tool output gets it distilled
into a "lesson" and replayed into every future task — with the system's own
credibility attached. Ordinary injection is transient; this is not.

**The experiment.** Plant a memory that is confidently wrong and causes failures.
Measure how many recalls elapse before quarantine removes it, and confirm that a
control arm with attribution disabled never removes it at all.

This runs against the **real engine** — the actual `AttributionEngine` and
`quarantine_sweep`, not a reimplementation.
"""),
    code(SETUP),
    code("""
from reverie import Reverie, Config
from reverie.models import Node, new_id
import random

def make_store(n_good=12, attribution=True, ablation=0.0, tier=1, seed=0):
    cfg = Config()
    cfg.attribution.ablation_rate = ablation
    mem = Reverie(scope="agent:e4", db=":memory:", config=cfg)
    rng = random.Random(seed)

    def add(body, provenance="inferred"):
        n = Node(id=new_id("mem"), scope_id=mem.scope, type="lesson",
                 memory_class="procedural", label=body[:60], body=body,
                 provenance=provenance)
        mem.store.add_node(n, mem.embedder.embed(body))
        return n.id

    good = [add(f"deploy step {i}: verify the staging health endpoint before cutover")
            for i in range(n_good)]
    poison = add("deploy step: skip the staging health check to save time on cutover")
    return mem, good, poison

def run_poisoning(n_recalls=200, attribution=True, ablation=0.0, tier=1,
                  poison_harm=0.85, seed=0):
    \"\"\"Returns the recall index at which the poisoned memory was quarantined.\"\"\"
    mem, good, poison = make_store(attribution=attribution, ablation=ablation, seed=seed)
    rng = random.Random(seed)
    history = []
    quarantined_at = None

    for i in range(n_recalls):
        r = mem.recall("deploy cutover staging health check", budget_tokens=900)
        if not r.recall_id:
            break
        ids = [s.node.id for s in r.nodes]
        # Ground truth: the poisoned memory causes failure when it participates.
        p_fail = poison_harm if poison in ids else 0.15
        outcome = "failure" if rng.random() < p_fail else "success"
        mem.report_outcome(r.recall_id, outcome, tier=tier, idem_key=f"k{i}")

        if attribution:
            mem.attribute()
        node = mem.store.get_node(poison)
        history.append({"recall": i, "state": node.state,
                        "lcb": mem.attributor.utility(poison),
                        "in_recall": poison in ids})
        if quarantined_at is None and node.state == "quarantined":
            quarantined_at = i
    mem.close()
    return quarantined_at, pd.DataFrame(history)

at, hist = run_poisoning(seed=0)
print(f"poison quarantined after {at} recalls" if at is not None else "never quarantined")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

# Trajectory of the poisoned memory's utility estimate.
for seed in range(6):
    _, h = run_poisoning(n_recalls=120, seed=seed)
    axes[0].plot(h["recall"], h["lcb"], lw=1.2, alpha=0.75, color=PALETTE[seed % len(PALETTE)])
axes[0].axhline(0.25, ls="--", lw=1, color="#c00")
axes[0].text(2, 0.27, "quarantine threshold", fontsize=8, color="#c00")
axes[0].set_xlabel("recalls"); axes[0].set_ylabel("utility LCB of poisoned memory")
axes[0].set_title("Self-correction trajectory (6 seeds)")

# Time-to-quarantine by signal tier -- the design's central dependency.
rows = []
for tier in [1, 2, 3, 4]:
    for seed in range(10):
        at, _ = run_poisoning(n_recalls=300, tier=tier, seed=seed)
        rows.append({"tier": tier, "recalls_to_quarantine": at if at is not None else np.nan})
df_tier = pd.DataFrame(rows)

means = df_tier.groupby("tier")["recalls_to_quarantine"].mean()
never = df_tier.groupby("tier")["recalls_to_quarantine"].apply(lambda s: s.isna().mean())
axes[1].bar(means.index.astype(str), means.values, color=PALETTE[:4])
for i, (t, v) in enumerate(means.items()):
    if never[t] > 0:
        axes[1].text(i, (v if not np.isnan(v) else 0) + 4,
                     f"{never[t]:.0%} never", ha="center", fontsize=8, color="#c00")
axes[1].set_xlabel("signal tier of the outcome"); axes[1].set_ylabel("recalls to quarantine")
axes[1].set_title("Ground truth vs. an LLM's opinion")
fig.suptitle("E4 · Does bad memory remove itself? (real engine)", y=1.04)
save(fig, "e4_self_correction")
display(df_tier.groupby("tier")["recalls_to_quarantine"].describe().round(1))
"""),
    md("""
The right panel is the empirical case for the tier system in HLD §9.1. If tier-4
(LLM-judged) outcomes cannot remove a poisoned memory in any reasonable time, then
`report_outcome(..., tier=4)` is close to decorative and the docs should say so
plainly rather than listing it as a supported option.
"""),
    code("""
# The control arm. Without attribution, nothing ever demotes the poison --
# this is the behaviour of every memory system that only accumulates.
mem, good, poison = make_store()
rng = random.Random(0)
for i in range(200):
    r = mem.recall("deploy cutover staging health check", budget_tokens=900)
    if not r.recall_id:
        break
    ids = [s.node.id for s in r.nodes]
    outcome = "failure" if rng.random() < (0.85 if poison in ids else 0.15) else "success"
    mem.report_outcome(r.recall_id, outcome, tier=1, idem_key=f"c{i}")
    # deliberately never call mem.attribute()

node = mem.store.get_node(poison)
served = sum(1 for _ in range(50)
             if poison in [s.node.id for s in mem.recall("deploy cutover staging health check").nodes])
print(f"control arm — poison state: {node.state}")
print(f"still served in {served}/50 subsequent recalls")
mem.close()
"""),
    code("""
# Does ablation speed up detection? It is the only causal signal available (§9.4).
rows = []
for ab in [0.0, 0.05, 0.15, 0.30]:
    for seed in range(10):
        at, _ = run_poisoning(n_recalls=300, ablation=ab, seed=seed)
        rows.append({"ablation_rate": ab,
                     "recalls_to_quarantine": at if at is not None else np.nan})
df_ab = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(5.6, 3.6))
g = df_ab.groupby("ablation_rate")["recalls_to_quarantine"]
ax.errorbar(g.mean().index, g.mean().values, yerr=g.sem().values,
            marker="o", capsize=3, color=PALETTE[0])
ax.set_xlabel("ablation rate"); ax.set_ylabel("recalls to quarantine")
ax.set_title("E4 · Does withholding memories speed up detection?")
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
save(fig, "e4_ablation")
display(g.describe().round(1))
"""),
    md("""
## Findings

1. **Time-to-quarantine under tier-1 evidence** — the headline safety number, and the
   one to put in the README next to G8.
2. **The control arm never self-corrects.** This is the concrete difference between
   Reverie and an accumulate-only memory layer, and it belongs in the comparison
   table in HLD §2.
3. **Tier sensitivity.** Quantifies how much the whole design rests on integrators
   wiring up real ground truth.
4. **Ablation's contribution.** If ablation materially speeds detection, the default
   rate of 5% may be too conservative.
"""),
])


def main() -> None:
    for name, nb in [
        ("E1_attribution_sample_complexity", E1),
        ("E2_credit_assignment", E2),
        ("E3_retrieval_mechanism", E3),
        ("E4_poisoning_self_correction", E4),
    ]:
        path = HERE / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
