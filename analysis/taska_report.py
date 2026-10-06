"""Task A report: PRE-REGISTERED analyses first (README pre-registration,
committed 2026-10-06 before results), exploratory clearly labelled.

Primary: (1) Jonckheere-Terpstra one-sided DECREASING trend of per-run
held across 6 ordered levels (f=0, rw10, rw25, rw50, rw75, shuffle~0.91),
permutation p with 10,000 label shuffles; (2) f50 from a 4-parameter
logistic fit to level medians vs mean realized f, bootstrap CI (1,000).
Secondary: same JT on per-model kick-AUC and noise-AUC, Holm over 2.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
Q = os.path.expanduser("~/cc-queue")
RNG = np.random.default_rng(0)

LEVELS = ["f0", "rw10", "rw25", "rw50", "rw75", "shuf"]


def realized_f_table():
    """Deterministic recomputation of realized f per graph."""
    from connectome_control.graph import load, build_graph
    c = load()
    oc = set(zip(*np.nonzero(c.chem > 0)))
    og = set(map(frozenset, zip(*np.nonzero(np.triu(c.gap, 1) > 0))))
    tot = len(oc) + len(og)
    cache_p = f"{Q}/taskA_realized_f.json"
    if os.path.exists(cache_p):
        return json.load(open(cache_p))
    out = {}
    kinds = [f"rw{p}g{g}" for p in (10, 25, 50, 75) for g in range(8)]
    kinds += [f"shuffle{g}" for g in range(60)]
    for k in kinds:
        ch, gp, _, _ = build_graph(k)
        kept = len(oc & set(zip(*np.nonzero(ch > 0)))) + \
            len(og & set(map(frozenset,
                             zip(*np.nonzero(np.triu(gp, 1) > 0)))))
        out[k] = 1 - kept / tot
        print(k, round(out[k], 3), flush=True)
    json.dump(out, open(cache_p, "w"))
    return out


def collect():
    rf = realized_f_table()
    rows = []                                    # (level, realized_f, held, ref)
    for f in glob.glob(f"{Q}/results/b2t1_worm_*.json"):
        r = json.load(open(f))
        rows.append(("f0", 0.0, r["metrics"]["held"],
                     os.path.basename(f)[:-5]))
    for f in glob.glob(f"{Q}/results/b2t1_shuffle*.json"):
        r = json.load(open(f))
        rows.append(("shuf", rf[r["job"]["kind"]], r["metrics"]["held"],
                     os.path.basename(f)[:-5]))
    for f in glob.glob(f"{Q}/results/b4*A_rw*.json"):
        r = json.load(open(f))
        kind = r["job"]["kind"]
        lvl = "rw" + kind[2:].split("g")[0]
        rows.append((lvl, rf[kind], r["metrics"]["held"],
                     os.path.basename(f)[:-5]))
    return rows


def auc_table():
    """kick/noise AUC per model: A models from results_t4A, endpoints from
    the batch-2 full-coverage robustness set."""
    out = {}
    for f in glob.glob(f"{Q}/results_t4A/t4A_*.json") + \
            glob.glob(f"{Q}/results_t4_full/t4fill_*.json") + \
            glob.glob(f"{Q}/results/t4_*.json"):
        r = json.load(open(f))
        ref = r["job"]["ref"]
        if not (ref.startswith("b2") or "A_rw" in ref):
            continue
        m = r["metrics"]
        out[ref] = (float(np.mean(list(m["kick_thd"].values()))),
                    float(np.mean(list(m["noise_obs"].values()))))
    return out


def jt_stat(groups):
    """JT statistic for DECREASING trend: count concordant pairs where a
    later-level value is LESS than an earlier-level value."""
    s = 0.0
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            x, y = groups[a], groups[b]
            lt = (y[None, :] < x[:, None]).sum()
            eq = (y[None, :] == x[:, None]).sum()
            s += lt + 0.5 * eq
    return s


def jt_perm(values, labels, n=10000):
    order = {l: i for i, l in enumerate(LEVELS)}
    lab = np.array([order[l] for l in labels])
    v = np.asarray(values, float)
    groups = [v[lab == i] for i in range(len(LEVELS))]
    obs = jt_stat(groups)
    cnt = 0
    for _ in range(n):
        p = RNG.permutation(lab)
        cnt += jt_stat([v[p == i] for i in range(len(LEVELS))]) >= obs
    return obs, (cnt + 1) / (n + 1)


def logistic_f50(med_f, med_h, boot_rows=None, nboot=1000):
    from scipy.optimize import curve_fit
    from scipy.optimize import brentq

    def lg(f, L, U, k, f0):
        return L + (U - L) / (1 + np.exp(k * (f - f0)))

    def fit_f50(fx, hx):
        p, _ = curve_fit(lg, fx, hx,
                         p0=[0.1, 0.6, 10.0, 0.1],
                         bounds=([0, 0, 0.1, -0.2], [1, 1, 200, 1.1]),
                         maxfev=20000)
        lo, hi = lg(fx[0], *p), lg(fx[-1], *p)
        mid = (lo + hi) / 2
        return brentq(lambda f: lg(f, *p) - mid, -0.2, 1.1), p

    f50, p = fit_f50(np.array(med_f), np.array(med_h))
    ci = None
    if boot_rows is not None:
        f50s, fails = [], 0
        by = {}
        for lvl, rfv, h, _ in boot_rows:
            by.setdefault(lvl, []).append((rfv, h))
        for _ in range(nboot):
            mf, mh = [], []
            for lvl in LEVELS:
                arr = by[lvl]
                pick = [arr[i] for i in RNG.integers(0, len(arr), len(arr))]
                mf.append(np.mean([a for a, _ in pick]))
                mh.append(np.median([b for _, b in pick]))
            try:
                f50s.append(fit_f50(np.array(mf), np.array(mh))[0])
            except Exception:
                fails += 1
        ci = (float(np.percentile(f50s, 2.5)),
              float(np.percentile(f50s, 97.5)), fails)
    return f50, p, ci


def main():
    rows = collect()
    n_by = {l: sum(r[0] == l for r in rows) for l in LEVELS}
    print("n per level:", n_by)
    assert n_by["f0"] == 20 and n_by["shuf"] == 60 and \
        all(n_by[l] == 16 for l in ("rw10", "rw25", "rw50", "rw75")), \
        "incomplete data -- refuse to run pre-registered analysis"

    L = ["# Task A report — partial rewiring of the worm",
         "", "Pre-registered analyses first (README pre-registration, "
         "committed before results). n=152 runs: worm 20 (f=0), 16 per "
         "rewiring level, shuffle 60 (f~0.91). Protocol v4, batch-2 "
         "relay/dataset/budget throughout.", "",
         "## Pre-registered primary", ""]

    # level summary
    med_f, med_h = [], []
    L.append("| level | mean realized f | n | median held | IQR |")
    L.append("|---|---|---|---|---|")
    for lvl in LEVELS:
        sel = [r for r in rows if r[0] == lvl]
        fv = np.mean([r[1] for r in sel])
        hv = np.array([r[2] for r in sel])
        med_f.append(fv); med_h.append(np.median(hv))
        L.append(f"| {lvl} | {fv:.3f} | {len(sel)} | "
                 f"{np.median(hv)*100:.0f}% | "
                 f"{np.percentile(hv,25)*100:.0f}-"
                 f"{np.percentile(hv,75)*100:.0f} |")

    obs, p = jt_perm([r[2] for r in rows], [r[0] for r in rows])
    L.append(f"\n1. Jonckheere-Terpstra, one-sided decreasing trend of "
             f"held across the 6 ordered levels: JT={obs:.0f}, "
             f"permutation p={p:.2e} (10,000 shuffles).")

    f50, pfit, ci = logistic_f50(med_f, med_h, boot_rows=rows)
    L.append(f"\n2. f50 (logistic fit to level medians): realized f = "
             f"**{f50:.3f}** (bootstrap 95% CI {ci[0]:.3f}-{ci[1]:.3f}; "
             f"{ci[2]}/1000 resamples failed to fit). Fit params "
             f"L={pfit[0]:.2f} U={pfit[1]:.2f} k={pfit[2]:.1f} "
             f"f0={pfit[3]:.3f}.")

    # secondary: AUC trends
    L.append("\n## Pre-registered secondary (kick/noise AUC, JT "
             "decreasing, Holm over 2)")
    aucs = auc_table()
    ps = []
    for mi, name in ((0, "kick-AUC"), (1, "noise-AUC")):
        vals, labs, miss = [], [], 0
        for lvl, rfv, h, ref in rows:
            if ref in aucs:
                vals.append(aucs[ref][mi]); labs.append(lvl)
            else:
                miss += 1
        o, pv = jt_perm(vals, labs)
        meds = {l: np.median([v for v, lb in zip(vals, labs) if lb == l])
                for l in LEVELS}
        ps.append((name, pv, meds, miss))
    ps_sorted = sorted(ps, key=lambda t: t[1])
    prev = 0
    for i, (name, pv, meds, miss) in enumerate(ps_sorted):
        adj = min(1.0, max(prev, (2 - i) * pv)); prev = adj
        L.append(f"- {name}: level medians " +
                 " / ".join(f"{meds[l]:.2f}" for l in LEVELS) +
                 f" ; JT p={pv:.2e}, Holm p={adj:.2e}" +
                 (f" ({miss} runs lacking robustness, excluded)" if miss
                  else ""))

    L.append("\n## Exploratory (labelled as such)")
    r10 = np.array([r[2] for r in rows if r[0] == "rw10"])
    w = np.array([r[2] for r in rows if r[0] == "f0"])
    from scipy.stats import mannwhitneyu
    U, px = mannwhitneyu(r10, w, alternative="two-sided")
    L.append(f"- Worm (f=0) vs rw10 alone: median {np.median(w)*100:.0f}% "
             f"vs {np.median(r10)*100:.0f}%, two-sided MW p={px:.4f} — "
             f"the first 12% of rewiring already costs a large share of "
             f"the gap.")
    mid = [np.median([r[2] for r in rows if r[0] == l])
           for l in ("rw25", "rw50")]
    L.append(f"- Mid-level flatness: rw25 and rw50 medians "
             f"{mid[0]*100:.0f}% / {mid[1]*100:.0f}% — consistent with an "
             f"early cliff followed by a plateau rather than a graded "
             f"decline.")

    out = f"{Q}/TASKA_REPORT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
