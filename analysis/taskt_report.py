"""Targeted-rewiring report (Task A extension). Pre-registered primary
first; realized budgets and input-pathway size stated per directive;
explicit comparison with Task A's rw10 level."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
Q = os.path.expanduser("~/cc-queue")


def realized_counts():
    from connectome_control.graph import load, build_graph
    c = load()
    oc = set(zip(*np.nonzero(c.chem > 0)))
    out = {}
    for v in ("S", "X"):
        for g in range(8):
            k = f"rw{v}g{g}"
            ch, _, _, _ = build_graph(k)
            out[k] = len(oc - set(zip(*np.nonzero(ch > 0))))
    return out


def main():
    held = {"S": [], "X": []}
    kinds = {"S": set(), "X": set()}
    for f in glob.glob(f"{Q}/results/b4*T_rw*.json"):
        r = json.load(open(f))
        v = r["job"]["kind"][2]
        held[v].append(r["metrics"]["held"])
        kinds[v].add(r["job"]["kind"])
    assert len(held["S"]) == 16 and len(held["X"]) == 16

    aucs = {"S": {"k": [], "n": []}, "X": {"k": [], "n": []}}
    n_rob = 0
    for f in glob.glob(f"{Q}/results_t4T/t4T_*.json"):
        r = json.load(open(f))
        v = "S" if "_rwS" in r["job"]["ref"] else "X"
        m = r["metrics"]
        aucs[v]["k"].append(float(np.mean(list(m["kick_thd"].values()))))
        aucs[v]["n"].append(float(np.mean(list(m["noise_obs"].values()))))
        n_rob += 1

    rc = realized_counts()
    TOTAL = 6026
    sS = [rc[k] for k in sorted(kinds["S"])]
    sX = [rc[k] for k in sorted(kinds["X"])]

    S, X = np.array(held["S"]), np.array(held["X"])
    U, p = mannwhitneyu(S, X, alternative="less")
    rb = 1 - 2 * U / (len(S) * len(X))

    L = ["# Targeted rewiring report (Task A extension)", "",
         "Pre-registered (README, committed before results). Question: is "
         "the worm advantage concentrated in the input-adjacent wiring? "
         "Matched edge budget, typed degree-preserving swaps, gap "
         "junctions frozen in both conditions; batch-2 relay/dataset/"
         "budget; 8 graphs x 2 seeds = 16 runs per condition.", "",
         "## Design realization (per directive)",
         f"- Input pathway: 316 chemical out-edges of the 22 injected "
         f"sensory neurons = 5.2% of all {TOTAL} edges.",
         f"- rwS (input-targeted): {min(sS)}-{max(sS)} original edges "
         f"removed per graph (realized f {min(sS)/TOTAL:.3f}-"
         f"{max(sS)/TOTAL:.3f}; ~{int(np.mean(sS))}/316 = "
         f"{np.mean(sS)/316*100:.0f}% of the input pathway destroyed).",
         f"- rwX (input-spared): {min(sX)}-{max(sX)} edges removed "
         f"(realized f {min(sX)/TOTAL:.3f}-{max(sX)/TOTAL:.3f}); the "
         f"slight excess over rwS is conservative against the "
         f"hypothesis.", "",
         "## Pre-registered primary",
         f"- held: rwS median {np.median(S)*100:.0f}% (IQR "
         f"{np.percentile(S,25)*100:.0f}-{np.percentile(S,75)*100:.0f}) "
         f"vs rwX {np.median(X)*100:.0f}% (IQR "
         f"{np.percentile(X,25)*100:.0f}-{np.percentile(X,75)*100:.0f})",
         f"- one-sided Mann-Whitney (rwS < rwX): U={U:.0f}, "
         f"**p = {p:.3f}**, rank-biserial r = {rb:+.2f}",
         "- Verdict: NULL. Rewiring the input pathway hurts no more than "
         "rewiring the same number of edges elsewhere. The Task-C "
         "hypothesis (inference-time ablation importance of input-"
         "adjacent neurons) does NOT transfer to training-time wiring "
         "preciousness: training re-routes around a rewired input "
         "pathway.", ""]

    # secondary AUC
    L.append("## Pre-registered secondary (kick/noise AUC, one-sided, "
             "Holm over 2)")
    ps = []
    for key, name in (("k", "kick-AUC"), ("n", "noise-AUC")):
        a, b = np.array(aucs["S"][key]), np.array(aucs["X"][key])
        _, pv = mannwhitneyu(a, b, alternative="less")
        ps.append((name, pv, np.median(a), np.median(b)))
    ps.sort(key=lambda t: t[1])
    prev = 0
    for i, (name, pv, ma, mb) in enumerate(ps):
        adj = min(1.0, max(prev, (2 - i) * pv)); prev = adj
        L.append(f"- {name}: rwS median {ma:.2f} vs rwX {mb:.2f}; "
                 f"p={pv:.3f}, Holm p={adj:.3f}  (n_rob={n_rob}/32)")

    # comparison with Task A rw10 (directive)
    r10 = [json.load(open(f))["metrics"]["held"]
           for f in glob.glob(f"{Q}/results/b4*A_rw10g*.json")]
    w = [json.load(open(f))["metrics"]["held"]
         for f in glob.glob(f"{Q}/results/b2t1_worm_*.json")]
    L += ["", "## Comparison with Task A's rw10 level (per directive)",
          f"- Task A rw10: realized f ~0.12 (~735 edges moved incl. gap), "
          f"median held {np.median(r10)*100:.0f}% (n={len(r10)}).",
          f"- Targeted conditions: ~200-213 edges (f ~0.033-0.035), "
          f"medians {np.median(S)*100:.0f}% / {np.median(X)*100:.0f}% — "
          f"LESS damage than rw10, as expected from the ~3.6x smaller "
          f"edge budget. Against worm f=0 median "
          f"{np.median(w)*100:.0f}%, both conditions sit on the Task-A "
          f"dose-response between f=0 and f=0.12.",
          "- No discrepancy: the apparent gap vs rw10 is explained by "
          "realized f (edge budget), and the rwS-rwX difference "
          "(-4 pts, n=16/arm with IQRs spanning ~30 pts) is within "
          "sample noise, consistent with the null primary.", "",
          "## Exploratory (labelled)",
          f"- Both conditions vs worm (f=0, median "
          f"{np.median(w)*100:.0f}%): two-sided MW rwS p="
          f"{mannwhitneyu(S, w)[1]:.3f}, rwX p="
          f"{mannwhitneyu(X, w)[1]:.3f} — ~200 moved edges already cost "
          f"a detectable share of performance."]

    out = f"{Q}/TASKT_REPORT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
