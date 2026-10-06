"""Stratified analysis across relays A (batch 2) and B (Task B).

van Elteren (stratified Wilcoxon, locally-best weights 1/(N_s+1)) plus a
meta-analytic combination of per-relay rank-biserial effects (fixed effect,
inverse null-variance weights) and Cochran's Q heterogeneity between the
two relays. One-sided for worm>shuffle (pre-registered direction),
two-sided for worm vs dense-78.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import norm, rankdata, mannwhitneyu, chi2

Q_ = os.path.expanduser("~/cc-queue")


def load_strata():
    strata = {"A": {"worm": [], "shuffle": [], "dense78": []},
              "B": {"worm": [], "shuffle": [], "dense78": []}}
    for f in glob.glob(f"{Q_}/results/b2t1_*.json"):
        r = json.load(open(f)); k = r["job"]["kind"]
        k = "shuffle" if k.startswith("shuffle") else k
        strata["A"][k].append(r["metrics"]["held"])
    for f in glob.glob(f"{Q_}/results/b4*B_*.json"):
        r = json.load(open(f)); k = r["job"]["kind"]
        k = "shuffle" if k.startswith("shuffle") else k
        strata["B"][k].append(r["metrics"]["held"])
    return strata


def van_elteren(pairs, alternative):
    """pairs: list of (x, y) per stratum; tests x vs y."""
    T = ET = VT = 0.0
    for x, y in pairs:
        x, y = np.asarray(x, float), np.asarray(y, float)
        n1, n2 = len(x), len(y)
        N = n1 + n2
        r = rankdata(np.concatenate([x, y]))
        W = r[:n1].sum()                    # rank-sum of x
        w = 1.0 / (N + 1)
        T += w * W
        ET += w * n1 * (N + 1) / 2
        # tie-corrected variance of W
        _, counts = np.unique(np.concatenate([x, y]), return_counts=True)
        tie = ((counts**3 - counts).sum()) / (N * (N - 1))
        VT += w**2 * (n1 * n2 / 12.0) * (N + 1 - tie)
    z = (T - ET) / np.sqrt(VT)
    if alternative == "greater":
        p = norm.sf(z)
    else:
        p = 2 * norm.sf(abs(z))
    return z, p


def per_stratum(x, y, alternative):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    U, p = mannwhitneyu(x, y, alternative=alternative)
    rb = 2 * U / (n1 * n2) - 1
    var_rb = (n1 + n2 + 1) / (3 * n1 * n2)   # null-variance approximation
    return U, p, rb, var_rb


def meta(rbs, vars_):
    w = 1 / np.asarray(vars_)
    rb = float((w * np.asarray(rbs)).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    Qh = float((w * (np.asarray(rbs) - rb) ** 2).sum())
    p_het = float(chi2.sf(Qh, len(rbs) - 1))
    return rb, se, Qh, p_het


def main():
    s = load_strata()
    L = ["# Stratified analysis across relays A and B", "",
         "Strata: relay A (batch 2; teacher held 91%) and relay B (Task B; "
         "teacher held 51.5%). van Elteren stratified Wilcoxon (weights "
         "1/(N_s+1), tie-corrected) + fixed-effect combination of per-"
         "relay rank-biserial effects (inverse null-variance weights) and "
         "Cochran's Q heterogeneity.", ""]
    for comp, alt in (("shuffle", "greater"), ("dense78", "two-sided")):
        L.append(f"## worm vs {comp} "
                 f"({'one-sided, pre-registered' if alt=='greater' else 'two-sided'})")
        rbs, vs = [], []
        for st in ("A", "B"):
            x, y = s[st]["worm"], s[st][comp]
            U, p, rb, v = per_stratum(x, y, alt)
            rbs.append(rb); vs.append(v)
            L.append(f"- relay {st}: worm median "
                     f"{np.median(x)*100:.0f}% (n={len(x)}) vs {comp} "
                     f"{np.median(y)*100:.0f}% (n={len(y)}); MW p={p:.2e}, "
                     f"rank-biserial r={rb:+.2f}")
        z, p_ve = van_elteren([(s[st]["worm"], s[st][comp])
                               for st in ("A", "B")], alt)
        rb, se, Qh, p_het = meta(rbs, vs)
        L.append(f"- POOLED van Elteren: z={z:+.2f}, p={p_ve:.2e}")
        L.append(f"- POOLED effect (fixed): r={rb:+.2f} (SE {se:.2f}); "
                 f"heterogeneity Q={Qh:.2f}, p={p_het:.3f}")
        L.append("")
    L += ["## Construction confound (per directive)",
          "Batch-3 teachers (q91/q80/q67) were DEGRADED versions of relay "
          "A (same swing brain, catch budget reduced), while relay B was "
          "built independently (new seeds, different catch brain). "
          "Teacher-quality comparisons across batch 3 and Task B are "
          "therefore confounded by construction method: held-rate alone "
          "is not a sufficient statistic for a teacher. Relay B (52% "
          "held) produced a +45-pt worm-shuffle gap where the batch-3 "
          "trend at similar quality predicted ~+18-24 pts."]
    out = f"{Q_}/STRATIFIED_SUPPLEMENT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
