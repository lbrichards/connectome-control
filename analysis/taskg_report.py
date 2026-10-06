"""Task G report: graph properties that predict performance.

Pre-registered (README): Spearman of each of the 10 registered properties
vs per-graph mean held (3 seeds) across the 90 shuffled graphs, Holm over
10; ridge out-of-sample (train relay-A 60, predict relay-B 30, alpha by
LOO-CV on relay A); placement of the worm and Task-A rewired graphs.
Refuses unless all 90 graphs have 3 seeds and properties are computed.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

Q = os.path.expanduser("~/cc-queue")
FAMILY = ["reciprocity", "ffl", "cycle3", "near_input_cycles",
          "path_in_eff", "reach2", "lscc", "spectral_radius",
          "clustering", "modularity"]
SIDE = ["rich_club_77", "reciprocity_nn"]


def outcomes():
    """graph kind -> {"held": [...], "kick": [...], "noise": [...]}"""
    held = {}
    for pat, cond in ((f"{Q}/results/b2t1_shuffle*.json", None),
                      (f"{Q}/results/b4*B_shuffle*.json", None),
                      (f"{Q}/results/b4*G_shuffle*.json", None),
                      (f"{Q}/results/b4*A_rw*.json", None),
                      (f"{Q}/results/b2t1_worm_*.json", None)):
        for f in glob.glob(pat):
            r = json.load(open(f))
            k = r["job"]["kind"]
            held.setdefault(k, []).append(
                (r["metrics"]["held"], os.path.basename(f)[:-5]))
    aucs = {}
    for d in ("results", "results_t4_full", "results_t4B", "results_t4G",
              "results_t4A"):
        for f in glob.glob(f"{Q}/{d}/t4*.json"):
            r = json.load(open(f)); m = r["metrics"]
            aucs[r["job"]["ref"]] = (
                float(np.mean(list(m["kick_thd"].values()))),
                float(np.mean(list(m["noise_obs"].values()))))
    return held, aucs


def main():
    props = json.load(open(f"{Q}/taskG_props.json"))
    held, aucs = outcomes()
    shuffles = [f"shuffle{g}" for g in range(60)] + \
               [f"shuffle{g}" for g in range(200, 230)]
    for k in shuffles:
        assert len(held.get(k, [])) == 3, \
            f"{k}: {len(held.get(k, []))}/3 seeds -- REFUSING"
        assert k in props, f"{k}: properties missing -- REFUSING"

    y = np.array([np.mean([h for h, _ in held[k]]) for k in shuffles])
    yk = np.array([np.mean([aucs[r][0] for _, r in held[k] if r in aucs])
                   for k in shuffles])
    yn = np.array([np.mean([aucs[r][1] for _, r in held[k] if r in aucs])
                   for k in shuffles])
    X = {p: np.array([props[k][p] for k in shuffles]) for p in
         FAMILY + SIDE}

    L = ["# Task G report — graph properties that predict performance", "",
         "Pre-registered analyses first. 90 shuffled graphs x 3 seeds; "
         "outcome = per-graph mean held. Worm NOT included in the "
         "correlation sample (placement only).", "",
         "## Pre-registered primary: Spearman vs mean held "
         "(Holm over the 10 registered properties)", "",
         "| property | rho | raw p | Holm p |", "|---|---|---|---|"]
    tests = []
    for p in FAMILY:
        rho, pv = spearmanr(X[p], y)
        tests.append([p, rho, pv])
    tests.sort(key=lambda t: t[2])
    prev = 0
    holm = {}
    for i, (p, rho, pv) in enumerate(tests):
        adj = min(1.0, max(prev, (10 - i) * pv)); prev = adj
        holm[p] = (rho, pv, adj)
    for p in FAMILY:
        rho, pv, adj = holm[p]
        star = " **" if adj < 0.05 else ""
        L.append(f"| {p} | {rho:+.2f} | {pv:.4f} | {adj:.4f}{star} |")
    surv = [p for p in FAMILY if holm[p][2] < 0.05]
    L.append("")
    L.append(f"Survivors after Holm: {surv if surv else 'NONE'}.")
    if not surv:
        L.append("Stated plainly per pre-registration: NO registered "
                 "property survives correction.")
    L.append("\nSide metrics (registered, outside the family): " +
             ", ".join(f"{p} rho={spearmanr(X[p], y)[0]:+.2f} "
                       f"(p={spearmanr(X[p], y)[1]:.3f})" for p in SIDE))

    # ridge out-of-sample
    from numpy.linalg import solve
    A_idx = list(range(60)); B_idx = list(range(60, 90))
    Xm = np.stack([X[p] for p in FAMILY], 1)
    mu, sd = Xm[A_idx].mean(0), Xm[A_idx].std(0) + 1e-12
    Z = (Xm - mu) / sd
    yA = y[A_idx]; ymu = yA.mean()

    def ridge_fit(Zt, yt, alpha):
        n, d = Zt.shape
        return solve(Zt.T @ Zt + alpha * np.eye(d), Zt.T @ (yt - ymu))

    best, best_err = None, np.inf
    for alpha in (0.01, 0.1, 1, 3, 10, 30, 100, 300):
        errs = []
        for i in A_idx:
            tr = [j for j in A_idx if j != i]
            w = ridge_fit(Z[tr], y[tr], alpha)
            errs.append((Z[i] @ w + ymu - y[i]) ** 2)
        if np.mean(errs) < best_err:
            best, best_err = alpha, np.mean(errs)
    w = ridge_fit(Z[A_idx], yA, best)
    predB = Z[B_idx] @ w + ymu
    rho_oos, p_oos = spearmanr(predB, y[B_idx])
    L += ["", "## Pre-registered out-of-sample check",
          f"- ridge (alpha={best} by LOO-CV on relay A), trained on the "
          f"60 relay-A graphs -> predict the 30 relay-B graphs: "
          f"Spearman(pred, obs) = **{rho_oos:+.2f}** (p={p_oos:.4f})",
          f"- coefficient sizes (standardized): " +
          ", ".join(f"{p}={w[i]:+.3f}" for i, p in enumerate(FAMILY))]

    # placement
    def place(kind, obs):
        z = (np.array([props[kind][p] for p in FAMILY]) - mu) / sd
        return float(z @ w + ymu), obs
    worm_obs = np.mean([h for h, _ in held["worm"]])
    pw = place("worm", worm_obs)
    L += ["", "## Placement (model fitted on relay-A shuffles)",
          f"- worm: predicted {pw[0]*100:.0f}% vs observed "
          f"{pw[1]*100:.0f}% (batch-2 20-seed mean)"]
    rows = []
    for p_ in (10, 25, 50, 75):
        preds, obss = [], []
        for g in range(8):
            k = f"rw{p_}g{g}"
            if k in props and k in held:
                pr, ob = place(k, np.mean([h for h, _ in held[k]]))
                preds.append(pr); obss.append(ob)
        rows.append((p_, np.mean(preds), np.mean(obss)))
    L.append("- Task-A rewired graphs (level, mean predicted, mean "
             "observed): " +
             "; ".join(f"rw{p_}: {pr*100:.0f}%/{ob*100:.0f}%"
                       for p_, pr, ob in rows))
    dec = all(rows[i][1] >= rows[i+1][1] for i in range(len(rows)-1))
    L.append(f"- predicted performance declines monotonically with "
             f"rewiring: {'YES' if dec else 'NO'}")

    out = f"{Q}/TASKG_REPORT.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
