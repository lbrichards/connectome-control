"""Phase-1 publication refresh (ANALYSIS ONLY — no new training).

1. Task A dose-response recomputed with the FULL 40-seed relay-A worm
   endpoint (f=0), shuffle endpoint = 60 relay-A graphs (first seed,
   like-for-like).
2. rliable-style summaries per relay and arm (worm/shuffle/dense-78):
   IQM, stratified bootstrap 95% CIs, performance profiles, probability
   of improvement.
3. Paper figures -> figs_paper/ (labelled + text-free), full-sample data.

Writes ~/cc-queue/PAPER_REFRESH.md.
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
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from taska_report import jt_perm, logistic_f50, realized_f_table, LEVELS

Q = os.path.expanduser("~/cc-queue")
REPO = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(REPO, "figs_paper")
os.makedirs(FIG, exist_ok=True)
SURF, INK, MUT, GRID = "#0A1210", "#C9D8D2", "#6E8A80", "#1E2F29"
COL = {"worm": "#199e70", "shuffle": "#3987e5", "dense78": "#c98500"}
LBL = {"worm": "C. elegans", "shuffle": "typed shuffle",
       "dense78": "dense-78"}
RNG = np.random.default_rng(7)


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
        lg = ax.get_legend()
        if lg: lg.remove()


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, facecolor=SURF,
                bbox_inches="tight")
    for ax in fig.axes:
        style(ax, True)
    fig.savefig(f"{FIG}/{name}_blank.png", dpi=200, facecolor=SURF,
                bbox_inches="tight")
    plt.close(fig)


def cells():
    """(relay, arm) -> np.array of held (40 worm / 40 dense / first-seed
    shuffles 60|30)."""
    out = {("A", "worm"): [], ("A", "dense78"): [],
           ("B", "worm"): [], ("B", "dense78"): []}
    for f in glob.glob(f"{Q}/results/b2t1_*.json"):
        r = json.load(open(f)); k = r["job"]["kind"]
        if k in ("worm", "dense78"):
            out[("A", k)].append(r["metrics"]["held"])
    for pat in (f"{Q}/results/b4*B_*.json", f"{Q}/results/b4*H_*.json"):
        for f in glob.glob(pat):
            r = json.load(open(f)); j = r["job"]; k = j["kind"]
            if k not in ("worm", "dense78"):
                continue
            relay = "B" if j.get("relay_dir") == "relay_B" else "A"
            out[(relay, k)].append(r["metrics"]["held"])
    sh = {("A", "shuffle"): {}, ("B", "shuffle"): {}}
    for pat in (f"{Q}/results/b2t1_shuffle*.json",
                f"{Q}/results/b4*B_shuffle*.json"):
        for f in glob.glob(pat):
            r = json.load(open(f)); j = r["job"]
            if j["tseed"] != 0:
                continue
            g = int(j["kind"][7:])
            relay = "B" if g >= 200 else "A"
            sh[(relay, "shuffle")][g] = r["metrics"]["held"]
    for k, d in sh.items():
        out[k] = list(d.values())
    return {k: np.array(v) for k, v in out.items()}


def iqm(x):
    x = np.sort(x); n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(x[lo:hi].mean())


def boot_ci(x, stat, n=10000):
    vals = [stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def poi(a, b, n=10000):
    """probability of improvement P(a > b) + bootstrap CI."""
    def p_est(aa, bb):
        return float((aa[:, None] > bb[None, :]).mean()
                     + 0.5 * (aa[:, None] == bb[None, :]).mean())
    est = p_est(a, b)
    vals = [p_est(a[RNG.integers(0, len(a), len(a))],
                  b[RNG.integers(0, len(b), len(b))]) for _ in range(n)]
    return est, float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))


def main():
    C = cells()
    L = ["# Phase-1 publication refresh (analysis only, full samples)", ""]

    # ---- rliable-style table
    L += ["## IQM (interquartile mean) of held-rate, stratified bootstrap "
          "95% CIs", "",
          "| relay | arm | n | IQM | 95% CI | median |",
          "|---|---|---|---|---|---|"]
    for relay in ("A", "B"):
        for arm in ("worm", "shuffle", "dense78"):
            x = C[(relay, arm)]
            m = iqm(x); lo, hi = boot_ci(x, iqm)
            L.append(f"| {relay} | {LBL[arm]} | {len(x)} | "
                     f"{m*100:.0f}% | {lo*100:.0f}-{hi*100:.0f}% | "
                     f"{np.median(x)*100:.0f}% |")
    L += ["", "## Probability of improvement P(X > Y), bootstrap 95% CI",
          "", "| relay | comparison | P | CI |", "|---|---|---|---|"]
    for relay in ("A", "B"):
        for a, b in (("worm", "shuffle"), ("dense78", "worm")):
            p, lo, hi = poi(C[(relay, a)], C[(relay, b)])
            L.append(f"| {relay} | {LBL[a]} > {LBL[b]} | {p:.2f} | "
                     f"{lo:.2f}-{hi:.2f} |")

    # ---- figure: performance profiles
    taus = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, relay in zip(axes, ("A", "B")):
        for arm in ("worm", "shuffle", "dense78"):
            x = C[(relay, arm)]
            prof = [(x >= t).mean() for t in taus]
            ax.plot(taus * 100, prof, color=COL[arm], lw=2, label=LBL[arm])
            bs = np.array([[(x[RNG.integers(0, len(x), len(x))] >= t).mean()
                            for t in taus] for _ in range(300)])
            ax.fill_between(taus * 100, np.percentile(bs, 2.5, 0),
                            np.percentile(bs, 97.5, 0), color=COL[arm],
                            alpha=0.18, lw=0)
        ax.set_xlabel("held-rate threshold τ (%)", color=MUT)
        ax.set_title(f"relay {relay}", color=INK, fontsize=10)
        style(ax, False)
    axes[0].set_ylabel("fraction of runs ≥ τ", color=MUT)
    axes[0].legend(facecolor=SURF, edgecolor=GRID, labelcolor=INK,
                   fontsize=8)
    save(fig, "p1_performance_profiles")

    # ---- Task A dose-response, 40-seed f0 endpoint
    rf = realized_f_table()
    rows = [("f0", 0.0, h) for h in C[("A", "worm")]]
    for f in glob.glob(f"{Q}/results/b4*A_rw*.json"):
        r = json.load(open(f)); kind = r["job"]["kind"]
        rows.append(("rw" + kind[2:].split("g")[0], rf[kind],
                     r["metrics"]["held"]))
    for f in glob.glob(f"{Q}/results/b2t1_shuffle*.json"):
        r = json.load(open(f))
        rows.append(("shuf", rf[r["job"]["kind"]], r["metrics"]["held"]))
    obs, p = jt_perm([r[2] for r in rows], [r[0] for r in rows])
    med_f = [np.mean([r[1] for r in rows if r[0] == l]) for l in LEVELS]
    med_h = [np.median([r[2] for r in rows if r[0] == l]) for l in LEVELS]
    rows4 = [(l, f_, h, "") for l, f_, h in rows]
    f50, pfit, ci = logistic_f50(med_f, med_h, boot_rows=rows4)
    L += ["", "## Task A dose-response, full 40-seed f=0 endpoint "
          "(analysis-only refresh)",
          f"- level medians (f=0 .. shuffle): " +
          " / ".join(f"{m*100:.0f}%" for m in med_h),
          f"- JT one-sided decreasing: p = {p:.2e} (n = "
          f"{len(rows)} runs)",
          f"- f50 = {f50:.3f} (bootstrap 95% CI {ci[0]:.3f}-{ci[1]:.3f}; "
          f"{ci[2]}/1000 fit failures) — the wide CI from the "
          f"pre-registered analysis persists; report f50 as "
          f"order-of-magnitude only."]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    fig.patch.set_facecolor(SURF)
    for lvl in LEVELS:
        xs = [r[1] for r in rows if r[0] == lvl]
        ys = [r[2] * 100 for r in rows if r[0] == lvl]
        ax.scatter(np.array(xs) + RNG.uniform(-0.004, 0.004, len(xs)), ys,
                   s=14, color=COL["worm"] if lvl == "f0" else
                   (COL["shuffle"] if lvl == "shuf" else "#7da88f"),
                   alpha=0.6)
    ax.plot(med_f, np.array(med_h) * 100, color=INK, lw=2, marker="o",
            ms=4, label="level median")
    ax.set_xlabel("realized rewiring fraction f", color=MUT)
    ax.set_ylabel("held (%)", color=MUT)
    ax.set_title("Dose-response: worm function vs rewiring "
                 "(40-seed endpoint)", color=INK, fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, labelcolor=INK, fontsize=8)
    style(ax, False)
    save(fig, "p2_dose_response")

    # ---- strip plot per relay/arm
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, relay in zip(axes, ("A", "B")):
        for i, arm in enumerate(("worm", "shuffle", "dense78")):
            x = C[(relay, arm)]
            ax.scatter(np.full(len(x), i) +
                       RNG.uniform(-0.14, 0.14, len(x)),
                       x * 100, s=12, color=COL[arm], alpha=0.65)
            ax.hlines(np.median(x) * 100, i - 0.25, i + 0.25,
                      color=INK, lw=2)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([LBL[a] for a in ("worm", "shuffle", "dense78")],
                           color=MUT, fontsize=8)
        ax.set_title(f"relay {relay}", color=INK, fontsize=10)
        style(ax, False)
    axes[0].set_ylabel("held (%)", color=MUT)
    save(fig, "p3_arm_distributions")

    out = f"{Q}/PAPER_REFRESH.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}; figures -> figs_paper/")


if __name__ == "__main__":
    main()
