"""Task C report: interpretability on the 20 batch-2 worms + positional
control (10 batch-2 shuffle models). Tables + figures (labelled and
text-free versions). Descriptive/exploratory throughout (per pre-reg note).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from connectome_control.graph import load, build_graph, MUSCLE_RE, SENSORY

Q = os.path.expanduser("~/cc-queue")
FIG = f"{Q}/figs_taskC"
os.makedirs(FIG, exist_ok=True)
SURF, INK, MUT, GRID = "#0A1210", "#C9D8D2", "#6E8A80", "#1E2F29"
GREEN, BLUE = "#199e70", "#3987e5"

MOTOR_PRE = ("DA", "DB", "DD", "VA", "VB", "VD", "AS", "VC")
LOCOMOTION = ("AVA", "AVB", "PVC") + MOTOR_PRE


def classify(nm):
    if MUSCLE_RE.match(nm):
        return "muscle"
    if not nm[0].isupper():
        return "effector"
    if nm in SENSORY:
        return "sensory(input)"
    if any(nm.startswith(p) and len(nm) <= len(p) + 2 for p in MOTOR_PRE):
        return "motor"
    return "inter"


def style(ax, blank):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if blank:
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")


def save(fig, name):
    for blank in (False, True):
        for ax in fig.axes:
            if blank:
                style(ax, True)
        fig.savefig(f"{FIG}/{name}{'_blank' if blank else ''}.png",
                    dpi=180, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def onehop_sep(drops, chem, in_idx):
    """rank-biserial separation of per-neuron ablation drops:
    1-hop chem targets of inputs vs all other non-input nodes."""
    hop = set(np.nonzero(chem[:, in_idx].sum(1) > 0)[0].tolist())
    ins = set(map(int, in_idx))
    a = [drops[k] for k in range(len(drops)) if k in hop and k not in ins]
    b = [drops[k] for k in range(len(drops))
         if k not in hop and k not in ins]
    U, p = mannwhitneyu(a, b, alternative="greater")
    return 2 * U / (len(a) * len(b)) - 1, p


def main():
    c = load()
    names = c.names
    worms = [np.load(f) for f in sorted(glob.glob(f"{Q}/taskC/*.npz"))]
    shufs = {os.path.basename(f)[:-4]: np.load(f)
             for f in sorted(glob.glob(f"{Q}/taskC_shuffle/*.npz"))}
    L = ["# Task C report — interpretability on trained worms",
         "", f"20 batch-2 worm models; ablations from {24} fixed hang "
         "starts x 600 steps; phases split at first upright-band entry. "
         "All analyses descriptive/exploratory (pre-registered as such).",
         ""]

    # ---- 1+4: neuron ablations, consistency, top table
    D = {ph: np.array([z["drops"][i] for z in worms])
         for i, ph in enumerate(("swing", "catch"))}
    L.append("## 1. Single-neuron ablation (held-rate drop, mean over "
             "20 models)\n")
    L.append("| rank | neuron | class | locomotion? | swing drop | "
             "catch drop | consistency (mean rank) |")
    L.append("|---|---|---|---|---|---|---|")
    score = (D["swing"] + D["catch"]).mean(0)
    ranks = np.array([np.argsort(np.argsort(-(dz)))
                      for dz in D["swing"] + D["catch"]])
    mean_rank = ranks.mean(0)
    top = np.argsort(-score)[:15]
    for rk, k in enumerate(top, 1):
        nm = names[k]
        loco = "YES" if (any(nm.startswith(p) for p in ("AVA", "AVB", "PVC"))
                         or any(nm.startswith(p) and len(nm) > len(p)
                                and nm[len(p)].isdigit()
                                for p in MOTOR_PRE)) else ""
        L.append(f"| {rk} | {nm} | {classify(nm)} | {loco} | "
                 f"{D['swing'][:,k].mean()*100:.0f} pts | "
                 f"{D['catch'][:,k].mean()*100:.0f} pts | "
                 f"{mean_rank[k]:.0f}/448 |")
    for ph in ("swing", "catch"):
        rhos = [spearmanr(D[ph][a], D[ph][b])[0]
                for a in range(20) for b in range(a + 1, 20)]
        L.append(f"\n- {ph} cross-model consistency: mean pairwise "
                 f"Spearman rho = {np.mean(rhos):+.2f} "
                 f"(range {np.min(rhos):+.2f} to {np.max(rhos):+.2f})")
    cls_top = {}
    for k in np.argsort(-score)[:30]:
        cls_top[classify(names[k])] = cls_top.get(classify(names[k]), 0) + 1
    L.append(f"- class composition of top 30: {cls_top}")
    L.append("- No classical locomotion neurons (AVA/AVB/PVC or A/B/D "
             "motor classes) in the top ranks: importance concentrates "
             "in the chemosensory periphery (the input-injection sites) "
             "and amphid interneurons (RIA/AIB/AIZ/AIA), i.e. the "
             "biological navigation hub, not the motor backbone.\n")

    # figure 1: top-20 drops
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    fig.patch.set_facecolor(SURF)
    idx = np.argsort(-score)[:20]
    x = np.arange(20)
    ax.bar(x - 0.2, D["swing"][:, idx].mean(0) * 100, 0.38,
           color=GREEN, label="swing phase")
    ax.bar(x + 0.2, D["catch"][:, idx].mean(0) * 100, 0.38,
           color=BLUE, label="catch phase")
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in idx],
                                         rotation=60, ha="right",
                                         color=MUT, fontsize=7)
    ax.set_ylabel("held-rate drop (pts)", color=MUT)
    ax.set_title("Top-20 single-neuron ablation effects (mean of 20 worms)",
                 color=INK, fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, labelcolor=INK, fontsize=8)
    style(ax, False)
    save(fig, "c1_neuron_ablation")

    # ---- 2: edges
    ed = np.concatenate([z["edge_drops"] for z in worms])
    L.append("## 2. Top-50 edge ablations (per model)\n"
             f"- single-edge ablation drops (edges pre-selected by "
             f"|weight x presynaptic activity|): median "
             f"{np.median(ed)*100:.1f} pts, 90th pct "
             f"{np.percentile(ed,90)*100:.0f} pts, max "
             f"{ed.max()*100:.0f} pts — at INFERENCE time, individual "
             f"high-traffic edges carry large effects. Contrast with "
             f"training-time rewiring (Task A dose-response; targeted "
             f"null), where no location is precious because training "
             f"re-routes: the acute/trained dissociation appears at the "
             f"edge level too. (Eval noise at 24 episodes is ~+/-10 pts; "
             f"individual values are coarse.)\n")

    # ---- 3: PCA
    ang = np.array([z["angles"] for z in worms])
    L.append("## 3. Swing/catch activity subspaces (top-10 PCs)\n"
             f"- principal angles (deg), median across models: " +
             ", ".join(f"{v:.0f}" for v in np.median(ang, 0)) +
             "\n- shared dominant mode (first angles ~10-16 deg) with "
             "increasingly phase-specific higher components (8th-10th "
             "angles 70-90 deg).\n")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.patch.set_facecolor(SURF)
    for a in ang:
        ax.plot(range(1, 11), a, color=GREEN, alpha=0.25, lw=1)
    ax.plot(range(1, 11), np.median(ang, 0), color=INK, lw=2,
            label="median")
    ax.set_xlabel("principal angle #", color=MUT)
    ax.set_ylabel("angle (deg)", color=MUT)
    ax.set_title("Swing vs catch subspace angles (20 worms)",
                 color=INK, fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, labelcolor=INK, fontsize=8)
    style(ax, False)
    save(fig, "c3_subspace_angles")

    # ---- positional control
    L.append("## 4. Positional control (10 shuffle models: 5 best + 5 "
             "median)\n")
    chem_w = c.chem
    sep_w = [onehop_sep(z["drops"].sum(0), chem_w, c.sensory)[0]
             for z in worms]
    sep_s = []
    for key, z in shufs.items():
        kind = key.split("_")[1]
        ch, _, in_idx, _ = build_graph(kind)
        sep_s.append(onehop_sep(z["drops"].sum(0), ch, in_idx)[0])
    U, p = mannwhitneyu(sep_w, sep_s, alternative="two-sided")
    L.append(f"- 1-hop-from-input dominance (rank-biserial separation of "
             f"ablation effects, 1-hop chem targets of inputs vs rest):")
    L.append(f"  - worms: median {np.median(sep_w):+.2f} "
             f"(IQR {np.percentile(sep_w,25):+.2f} to "
             f"{np.percentile(sep_w,75):+.2f})")
    L.append(f"  - shuffles (own graphs): median {np.median(sep_s):+.2f} "
             f"(IQR {np.percentile(sep_s,25):+.2f} to "
             f"{np.percentile(sep_s,75):+.2f})")
    L.append(f"  - worm vs shuffle two-sided MW p = {p:.3f}")
    concl = ("input-adjacency dominates ablation importance in BOTH "
             "populations -> positional (proximity-to-input) effect, "
             "not worm-specific biology"
             if p > 0.05 and np.median(sep_s) > 0.3 else
             "input-adjacency dominance is stronger in worms than in "
             "shuffles -> not purely positional")
    L.append(f"  - reading: {concl}\n")
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    fig.patch.set_facecolor(SURF)
    for i, (vals, col, lab) in enumerate(
            ((sep_w, GREEN, "worms"), (sep_s, BLUE, "shuffles"))):
        xs = np.full(len(vals), i) + np.random.default_rng(0).uniform(
            -0.08, 0.08, len(vals))
        ax.scatter(xs, vals, color=col, s=22, alpha=0.85, label=lab)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["worms", "shuffles"],
                                              color=MUT)
    ax.set_ylabel("1-hop separation (rank-biserial)", color=MUT)
    ax.set_title("Is ablation importance just input proximity?",
                 color=INK, fontsize=10)
    style(ax, False)
    save(fig, "c4_positional_control")

    out = f"{Q}/TASKC_REPORT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}  figs -> {FIG}/")


if __name__ == "__main__":
    main()
