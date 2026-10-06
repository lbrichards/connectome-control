"""Task F report: FULL input-pathway rewiring (pre-registered ext. 2)."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

Q = os.path.expanduser("~/cc-queue")


def main():
    S, X, refS, refX = [], [], [], []
    for f in glob.glob(f"{Q}/results/b4*F_rw*.json"):
        r = json.load(open(f))
        if r["job"]["kind"].startswith("rwSF"):
            S.append(r["metrics"]["held"]); refS.append(os.path.basename(f)[:-5])
        else:
            X.append(r["metrics"]["held"]); refX.append(os.path.basename(f)[:-5])
    S, X = np.array(S), np.array(X)
    U, p = mannwhitneyu(S, X, alternative="less")
    rb = 1 - 2 * U / (len(S) * len(X))
    U2, p2 = mannwhitneyu(S, X, alternative="two-sided")
    r10 = [json.load(open(f))["metrics"]["held"]
           for f in glob.glob(f"{Q}/results/b4*A_rw10g*.json")]

    L = ["# Task F report — full input-pathway rewiring", "",
         "Pre-registered (README, committed before results). rwSF: ALL "
         "316 input-pathway edges removed (verified per graph) plus "
         "268-296 measured non-pool collateral (totals 584-612, realized "
         "f ~0.10). rwXF: per-graph matched totals entirely outside the "
         "pathway. 12 graphs x 2 seeds per arm; batch-2 relay/dataset/"
         "budget.", "",
         "## Pre-registered primary",
         f"- held: rwSF median {np.median(S)*100:.0f}% (IQR "
         f"{np.percentile(S,25)*100:.0f}-{np.percentile(S,75)*100:.0f}, "
         f"n=24) vs rwXF {np.median(X)*100:.0f}% (IQR "
         f"{np.percentile(X,25)*100:.0f}-{np.percentile(X,75)*100:.0f}, "
         f"n=24)",
         f"- one-sided MW (rwSF < rwXF, the registered direction): "
         f"U={U:.0f}, **p = {p:.3f}**, rank-biserial r = {rb:+.2f}",
         f"- the registered hypothesis FAILS; the point estimate is "
         f"REVERSED (two-sided p = {p2:.3f}, not significant either "
         f"way).",
         "- Verdict: at full dose the input pathway is, if anything, the "
         "LEAST training-precious place to rewire — consistent with the "
         "trainable input projection (w_in) adapting to whatever targets "
         "the sensors land on, while same-size disruption of the "
         "recurrent core is harder to train around. Together with the "
         "K=200 null, the worm advantage is not localized in the input "
         "wiring.", ""]

    aucs = {}
    for f in glob.glob(f"{Q}/results_t4F/t4F_*.json"):
        r = json.load(open(f)); m = r["metrics"]
        aucs[r["job"]["ref"]] = (
            float(np.mean(list(m["kick_thd"].values()))),
            float(np.mean(list(m["noise_obs"].values()))))
    L.append("## Pre-registered secondary (kick/noise AUC, one-sided "
             "rwSF<rwXF, Holm over 2)")
    ps = []
    for mi, name in ((0, "kick-AUC"), (1, "noise-AUC")):
        a = [aucs[r][mi] for r in refS if r in aucs]
        b = [aucs[r][mi] for r in refX if r in aucs]
        _, pv = mannwhitneyu(a, b, alternative="less")
        ps.append((name, pv, np.median(a), np.median(b), len(a) + len(b)))
    ps.sort(key=lambda t: t[1])
    prev = 0
    for i, (name, pv, ma, mb, n) in enumerate(ps):
        adj = min(1.0, max(prev, (2 - i) * pv)); prev = adj
        L.append(f"- {name}: rwSF median {ma:.2f} vs rwXF {mb:.2f}; "
                 f"p={pv:.3f}, Holm p={adj:.3f} (n={n}/48)")

    L += ["", "## Exploratory (labelled)",
          f"- Dose context: both arms sit near Task A's rw10 level "
          f"(median {np.median(r10)*100:.0f}% at ~735 edges) — "
          f"consistent with damage scaling with edge count, not "
          f"location.",
          f"- rwSF vs rwXF reversed-direction reading: rwSF > rwXF "
          f"one-sided p = {1-p+1/(len(S)*len(X)):.3f} (post-hoc, "
          f"NOT pre-registered; report as hypothesis-generating only)."]

    out = f"{Q}/TASKF_REPORT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
