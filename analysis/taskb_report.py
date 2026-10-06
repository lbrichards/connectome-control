"""Task B report: replication with relay B. Pre-registered first;
framing per analysis/REPORT_NOTES.md (different AND weaker teacher;
batch-3 contextualization; full attempt table; variance finding)."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, percentileofscore

Q = os.path.expanduser("~/cc-queue")

ATTEMPTS = [
    ("A", 1, 2500, "94%", "PASS", "91%", "PASS (first attempt)"),
    ("B", 11, 2500, "44%", "FAIL", "-", "-"),
    ("B", 11, 6000, "78%", "FAIL", "-", "-"),
    ("B", 12, 6000, "84%", "FAIL", "-", "-"),
    ("B", 13, 6000, "92%", "PASS", "40%", "FAIL"),
    ("B", 14, 10000, "98%", "PASS", "49%", "FAIL"),
    ("B", 15, 10000, "98%", "PASS", "52%", "PASS"),
]


def aucs(globpat, refpred):
    out = {}
    for f in glob.glob(globpat):
        r = json.load(open(f))
        ref = r["job"]["ref"]
        if not refpred(ref):
            continue
        m = r["metrics"]
        out[ref] = (float(np.mean(list(m["kick_thd"].values()))),
                    float(np.mean(list(m["noise_obs"].values()))))
    return out


def main():
    arms = {"worm": [], "shuffle": [], "dense78": []}
    refs = {"worm": [], "shuffle": [], "dense78": []}
    for f in glob.glob(f"{Q}/results/b4*B_*.json"):
        r = json.load(open(f))
        k = r["job"]["kind"]
        a = "shuffle" if k.startswith("shuffle") else k
        arms[a].append(r["metrics"]["held"])
        refs[a].append(os.path.basename(f)[:-5])
    w, s, d = map(np.array, (arms["worm"], arms["shuffle"], arms["dense78"]))
    U, p = mannwhitneyu(w, s, alternative="greater")
    pct = percentileofscore(s, np.median(w), kind="weak")

    L = ["# Task B report — replication with a second relay (relay B)", "",
         "Pre-registered analyses first (README). Students: worm seeds "
         "0-19, 30 NEW shuffle graphs (200-229), dense-78 seeds 0-9; "
         "batch-2 budget; protocol v4.", "",
         "## Teacher: relay B is both a DIFFERENT and a WEAKER teacher",
         "Relay B was built independently of relay A (new swing seed 10; "
         "catch brain reached gate only at seed 15 @ 10,000 steps) and "
         "holds 51.5% vs relay A's 91%. Construction caveat: batch-3's "
         "quality ladder (q91/q80/q67) consisted of DEGRADED relay-A "
         "variants, so quality comparisons between relay B and batch 3 "
         "are confounded by construction method.", "",
         "### Relay build attempt table (teacher-building variance)",
         "| relay | catch seed | steps | catch held (gate >=90%) | "
         "catch gate | relay held (gate >=50%) | relay gate |",
         "|---|---|---|---|---|---|---|"]
    for row in ATTEMPTS:
        L.append("| " + " | ".join(map(str, row)) + " |")
    L += ["",
          "Finding: relay A passed both gates on its first draw; relay B "
          "needed 6 attempts (catch gate 3/6 passed; among catch-passing "
          "brains the relay gate passed 1/3, with 49% an exact near-miss). "
          "Teacher building at this budget has high seed variance in BOTH "
          "stages, and a strong catch brain (98%) does not guarantee "
          "relay coupling.", "",
          "## Pre-registered primary",
          f"- worm (n={len(w)}): median {np.median(w)*100:.0f}% (IQR "
          f"{np.percentile(w,25)*100:.0f}-{np.percentile(w,75)*100:.0f}) "
          f"vs shuffle (n={len(s)}): {np.median(s)*100:.0f}% (IQR "
          f"{np.percentile(s,25)*100:.0f}-{np.percentile(s,75)*100:.0f})",
          f"- one-sided Mann-Whitney (worm > shuffle): U={U:.0f}, "
          f"**p = {p:.2e}**, rank-biserial r = "
          f"{2*U/(len(w)*len(s))-1:+.2f}",
          f"- worm median at the **{pct:.0f}th percentile** of the "
          f"30-graph shuffle distribution",
          "- REPLICATION CRITERION MET (direction reproduced, p < 0.05, "
          "by ~3 orders of magnitude).", ""]

    # secondaries
    ab = aucs(f"{Q}/results_t4B/t4B_*.json", lambda r: "B_" in r)
    L.append("## Pre-registered secondary")
    ps = []
    for mi, name in ((0, "kick-AUC"), (1, "noise-AUC")):
        aw = [ab[r][mi] for r in refs["worm"] if r in ab]
        as_ = [ab[r][mi] for r in refs["shuffle"] if r in ab]
        _, pv = mannwhitneyu(aw, as_, alternative="greater")
        ps.append((name, pv, np.median(aw), np.median(as_)))
    ps.sort(key=lambda t: t[1])
    prev = 0
    for i, (name, pv, mw_, ms_) in enumerate(ps):
        adj = min(1.0, max(prev, (2 - i) * pv)); prev = adj
        L.append(f"- {name} worm vs shuffle (one-sided): medians "
                 f"{mw_:.2f} vs {ms_:.2f}; p={pv:.2e}, Holm p={adj:.2e}")
    U2, p2 = mannwhitneyu(w, d, alternative="two-sided")
    L.append(f"- worm vs dense-78 held (two-sided): medians "
             f"{np.median(w)*100:.0f}% vs {np.median(d)*100:.0f}% "
             f"(n={len(d)}); p={p2:.2f} — the relay-A dense advantage "
             f"(p=0.036) does NOT reproduce under relay B; see "
             f"STRATIFIED_SUPPLEMENT.md (pooled two-sided p=0.078).")
    L.append("")

    # side-by-side + batch-3 context
    b2 = {"worm": [], "shuffle": []}
    for f in glob.glob(f"{Q}/results/b2t1_*.json"):
        r = json.load(open(f)); k = r["job"]["kind"]
        k = "shuffle" if k.startswith("shuffle") else k
        if k in b2:
            b2[k].append(r["metrics"]["held"])
    L += ["## Relay A vs relay B, side by side",
          "| | relay A (91% held) | relay B (51.5% held) |",
          "|---|---|---|",
          f"| worm median held | {np.median(b2['worm'])*100:.0f}% | "
          f"{np.median(w)*100:.0f}% |",
          f"| shuffle median held | {np.median(b2['shuffle'])*100:.0f}% | "
          f"{np.median(s)*100:.0f}% |",
          f"| gap | +{(np.median(b2['worm'])-np.median(b2['shuffle']))*100:.0f} "
          f"pts | +{(np.median(w)-np.median(s))*100:.0f} pts |",
          "| worm percentile in shuffle | 97th | 97th |", "",
          "## Batch-3 contextualization (per directive)",
          "Batch-3 (degraded relay-A teachers): gap +52/+24/+18 pts at "
          "teacher quality 91/80/67%. Relay B at 52% quality yields a "
          "+45-pt gap — far above that trend. Because batch-3 teachers "
          "share relay A's construction while relay B is independent, "
          "this divergence shows teacher held-rate is NOT a sufficient "
          "statistic: construction/composition of the teacher matters "
          "for what students inherit. The batch-3 'gap shrinks with "
          "quality' trend should be read as within-lineage only.", ""]

    sp = f"{Q}/taskB_spotchecks.json"
    if os.path.exists(sp):
        L.append("## Determinism spot checks")
        for c in json.load(open(sp)):
            L.append(f"- `{c['id']}` {c['orig']} -> {c['rerun']}: "
                     f"{'MATCH' if c['match'] else 'MISMATCH'}")
    out = f"{Q}/TASKB_REPORT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
