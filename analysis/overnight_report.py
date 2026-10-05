#!/usr/bin/env python3
"""Full analysis + figures + morning report from coordinator results.

Reads ~/cc-queue/results/*.json (v2 schema) and the queue DB. Robust to
partial completion: reports whatever exists, labelled as such. Writes:
  ~/cc-queue/figs/*.png           (labelled + *_blank.png)
  ~/cc-queue/MORNING_REPORT.md
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, spearmanr

Q = os.path.expanduser("~/cc-queue")
FIG = f"{Q}/figs"
os.makedirs(FIG, exist_ok=True)

SURF, INK, MUT, GRID = "#0A1210", "#C9D8D2", "#6E8A80", "#1E2F29"
COL = {"worm": "#199e70", "shuffle": "#3987e5", "dense78": "#c98500",
       "dense448": "#9BB0A8"}
LBL = {"worm": "C. elegans", "shuffle": "typed shuffle",
       "dense78": "dense-78", "dense448": "dense-448"}
ARMS = ["worm", "shuffle", "dense78"]


def armof(kind):
    return "shuffle" if kind.startswith("shuffle") else kind


def load_results():
    out = []
    for f in sorted(glob.glob(f"{Q}/results/*.json")):
        try:
            r = json.load(open(f))
            if "job" in r:
                out.append(r)
        except Exception:
            pass
    return out


def style(ax, blank):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if blank:
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, facecolor=SURF,
                bbox_inches="tight")
    plt.close(fig)


def strip(groups, fname, ylab, title, ylim=None):
    rng = np.random.default_rng(1)
    for blank in (False, True):
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        fig.patch.set_facecolor(SURF)
        for i, (name, vals, col) in enumerate(groups):
            v = np.asarray(vals, float)
            if len(v) == 0:
                continue
            w = np.where(v == (ylim[0] if ylim else 0), 0.46, 0.30) \
                if ylim else np.full(len(v), 0.30)
            x = i + (rng.random(len(v)) - 0.5) * w
            ax.scatter(x, v, s=40, color=col, edgecolors="none", alpha=0.9,
                       zorder=3)
            ax.hlines(np.median(v), i - 0.24, i + 0.24, color=col, lw=2.4,
                      zorder=4)
            if not blank:
                ax.text(i, -0.085, f"{name}\nn={len(v)}",
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="top", color=INK, fontsize=9)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([])
        ax.set_xlim(-0.55, len(groups) - 0.45)
        if ylim:
            ax.set_ylim(*ylim)
        if not blank:
            ax.set_ylabel(ylab, color=INK, fontsize=10)
            ax.set_title(f"{title} — bars: median", color=INK, fontsize=10.5,
                         loc="left", pad=10)
        style(ax, blank)
        save(fig, fname + ("_blank" if blank else ""))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, c-h), min(1, c+h)


def main():
    R = load_results()
    t1 = [r for r in R if r["job"].get("tier") == 1]
    t2 = [r for r in R if r["job"].get("tier") == 2]
    t3 = [r for r in R if r["job"].get("tier") in (3, 9)]
    t4 = [r for r in R if r["job"].get("type") == "robust"]

    L = []
    L.append(f"# Overnight run — morning report\n\nGenerated {time.ctime()}\n")

    # --- queue / host summary from DB
    try:
        c = sqlite3.connect(f"{Q}/queue.db")
        L.append("## Completion\n")
        L.append("| tier | DONE | pending | running | FAILED |")
        L.append("|---|---|---|---|---|")
        agg = {}
        for t, s, n in c.execute(
                "SELECT tier,state,COUNT(*) FROM jobs GROUP BY tier,state"):
            agg.setdefault(t, {})[s] = n
        for t in sorted(agg):
            a = agg[t]
            L.append(f"| {t} | {a.get('DONE',0)} | {a.get('pending',0)} | "
                     f"{a.get('running',0)} | {a.get('FAILED',0)} |")
        L.append("\n## Per-host throughput (whole run)\n")
        L.append("| host | jobs done | avg min/job |")
        L.append("|---|---|---|")
        for h, n, mn in c.execute("""SELECT host, COUNT(*),
                AVG((done_at-claimed_at)/60.0) FROM jobs
                WHERE state='DONE' GROUP BY host"""):
            L.append(f"| {h} | {n} | {round(mn or 0,1)} |")
        fails = c.execute("""SELECT id, host, error FROM jobs
                             WHERE state='FAILED'""").fetchall()
        L.append(f"\n## Failures: {len(fails)}\n")
        for jid, h, e in fails[:20]:
            L.append(f"- `{jid}` on {h}: {e}")
        halted = os.path.exists(f"{Q}/HALT")
        if halted:
            L.append(f"\n**QUEUE HALTED**: {open(f'{Q}/HALT').read().strip()}")
    except Exception as e:
        L.append(f"(queue db unavailable: {e})")

    # --- Tier 1 analysis
    L.append(f"\n## Tier 1 — core comparison ({len(t1)} results)\n")
    if t1:
        by = {a: [r for r in t1 if armof(r["job"]["kind"]) == a] for a in ARMS}
        L.append("| arm | n | catch-at-all | held med | off-track med | "
                 "%catchable | conv(catchable) |")
        L.append("|---|---|---|---|---|---|---|")
        catch_tbl = {}
        for a in ARMS:
            runs = by[a]
            if not runs:
                continue
            held = np.array([r["metrics"]["held"] for r in runs])
            off = np.array([r["metrics"]["off_track"] for r in runs])
            k = int((held > 0).sum()); n = len(runs)
            catch_tbl[a] = (k, n - k)
            fc = [r["metrics"]["frac_catchable"] for r in runs
                  if r["metrics"]["frac_catchable"] is not None]
            cc = [r["metrics"]["conversion_catchable"] for r in runs
                  if r["metrics"]["conversion_catchable"] is not None]
            L.append(f"| {LBL[a]} | {n} | {k}/{n} ({k/n*100:.0f}%) | "
                     f"{np.median(held)*100:.0f}% | {np.median(off)*100:.0f}% | "
                     f"{np.median(fc)*100:.0f}% | "
                     f"{(np.median(cc)*100 if cc else float('nan')):.0f}% |")
            mse = [r.get("final_mse") for r in runs]
            if len(runs) >= 5:
                rho, p = spearmanr(mse, held)
                L.append(f"|  | | within-arm Spearman(MSE,held) "
                         f"rho={rho:+.2f} p={p:.3f} | | | | |")
        for pair in (("worm", "shuffle"), ("worm", "dense78")):
            if pair[0] in catch_tbl and pair[1] in catch_tbl:
                o, p = fisher_exact([catch_tbl[pair[0]], catch_tbl[pair[1]]])
                L.append(f"\nFisher catch-at-all {pair[0]} vs {pair[1]}: "
                         f"OR={o:.2f} p={p:.4f}")
        # off-track vs prototype (runaway fix check)
        PROTO_OFF = {"worm": 66, "shuffle": 99, "dense78": 100}
        L.append("\nOff-track rate vs prototype (median-model medians, "
                 "prototype eval at L=3):")
        for a in ARMS:
            if by[a]:
                nowv = np.median([r["metrics"]["off_track"] for r in by[a]])*100
                L.append(f"- {LBL[a]}: now {nowv:.0f}% vs prototype "
                         f"{PROTO_OFF[a]}%")
        gate_f = f"{Q}/tier4_gate.json"
        if os.path.exists(gate_f):
            g4 = json.load(open(gate_f))
            L.append(f"\nTier-4 demo-grade gate: {g4['passed']}/{g4['total']} "
                     "tier-1 models passed (quiet_hold >= 0.80)")
        # graph-level
        shuf = by["shuffle"]
        if shuf:
            gf = {}
            for r in shuf:
                gf.setdefault(r["job"]["kind"], []).append(
                    r["metrics"]["held"] > 0)
            fr = np.array([np.mean(v) for v in gf.values()])
            worm_runs = by["worm"]
            if worm_runs:
                wf = np.mean([r["metrics"]["held"] > 0 for r in worm_runs])
                nge = int((fr >= wf).sum())
                L.append(f"\n### Graph-level: worm catch fraction {wf:.2f} vs "
                         f"{len(fr)} shuffled graphs (median {np.median(fr):.2f}, "
                         f"range {fr.min():.2f}-{fr.max():.2f}); "
                         f"{nge}/{len(fr)} graphs >= worm -> permutation "
                         f"p={(nge+1)/(len(fr)+1):.3f}")
        # figures
        strip([(LBL[a], [r["metrics"]["held"]*100 for r in by[a]], COL[a])
               for a in ARMS], "t1_held", "held (%)",
              "Tier 1: held per run", ylim=(-4, 100))
        strip([(LBL[a], [r["metrics"]["off_track"]*100 for r in by[a]], COL[a])
               for a in ARMS], "t1_offtrack", "off-track (%)",
              "Tier 1: off-track rate per run", ylim=(-4, 104))
        for blank in (False, True):
            fig, ax = plt.subplots(figsize=(4.6, 3.4))
            fig.patch.set_facecolor(SURF)
            for i, a in enumerate(ARMS):
                runs = by[a]
                if not runs:
                    continue
                k = sum(r["metrics"]["held"] > 0 for r in runs); n = len(runs)
                lo, hi = wilson(k, n)
                ax.bar(i, k/n*100, width=0.5, color=COL[a], zorder=3)
                ax.errorbar(i, k/n*100, yerr=[[(k/n-lo)*100], [(hi-k/n)*100]],
                            color=INK, elinewidth=1.3, capsize=4, zorder=4)
                if not blank:
                    ax.text(i, hi*100+3, f"{k/n*100:.0f}%", ha="center",
                            color=INK, fontsize=10)
                    ax.text(i, -8, f"{LBL[a]}\nn={n}", ha="center", va="top",
                            color=INK, fontsize=9)
            ax.set_ylim(0, 112); ax.set_xticks(range(len(ARMS)))
            ax.set_xticklabels([])
            if not blank:
                ax.set_ylabel("runs that catch at all (%)", color=INK,
                              fontsize=10)
                ax.set_title("Tier 1: catch-at-all — 95% Wilson CI",
                             color=INK, fontsize=10.5, loc="left", pad=10)
            style(ax, blank)
            save(fig, "t1_catch" + ("_blank" if blank else ""))

    # --- Tier 2
    L.append(f"\n## Tier 2 — matched-MSE ({len(t2)} results)\n")
    if t2:
        by2 = {a: [r for r in t2 if armof(r["job"]["kind"]) == a] for a in ARMS}
        L.append("| arm | n | hit target | steps-to-target med | held med | "
                 "catch-at-all |")
        L.append("|---|---|---|---|---|---|")
        for a in ARMS:
            runs = by2[a]
            if not runs:
                continue
            hit = [r for r in runs if r.get("hit_target")]
            st = [r["steps_to_target"] for r in hit]
            held = np.array([r["metrics"]["held"] for r in hit]) if hit else np.array([])
            L.append(f"| {LBL[a]} | {len(runs)} | {len(hit)}/{len(runs)} | "
                     f"{(np.median(st) if st else float('nan')):.0f} | "
                     f"{(np.median(held)*100 if len(held) else float('nan')):.0f}% | "
                     f"{int((held>0).sum()) if len(held) else 0}/{len(hit)} |")
        strip([(LBL[a],
                [r["steps_to_target"] for r in by2[a] if r.get("hit_target")],
                COL[a]) for a in ARMS],
              "t2_steps", "steps to MSE 0.10",
              "Tier 2: trainability (capped runs excluded)")
        strip([(LBL[a],
                [r["metrics"]["held"]*100 for r in by2[a] if r.get("hit_target")],
                COL[a]) for a in ARMS],
              "t2_held", "held (%)",
              "Tier 2: held at matched clone MSE", ylim=(-4, 100))

    # --- Tier 3 / Tier 4
    try:
        c9 = sqlite3.connect(f"{Q}/queue.db")
        n9 = dict(c9.execute("SELECT state, COUNT(*) FROM jobs WHERE tier=9 "
                             "GROUP BY state").fetchall())
        pend9 = n9.get("pending", 0) + n9.get("running", 0)
    except Exception:
        pend9 = 0
    L.append(f"\n## Tier 3 — dense-448 to convergence ({len(t3)} results)"
             + (f" — **STILL RUNNING: {pend9} job(s) outstanding**"
                if pend9 else " — complete") + "\n")
    for r in t3:
        L.append(f"- seed {r['job']['tseed']}: "
                 f"{'hit' if r.get('hit_target') else 'CAP'} "
                 f"@{r.get('steps_to_target')} | held "
                 f"{r['metrics']['held']*100:.0f}% | off "
                 f"{r['metrics']['off_track']*100:.0f}%")
    L.append(f"\n## Tier 4 — robustness ({len(t4)} results)\n")
    for r in t4:
        L.append(f"- ref `{r['job']['ref']}`: {json.dumps(r['metrics'])}")

    # --- rerun checks (written by pilot)
    rr = f"{Q}/rerun_checks.json"
    if os.path.exists(rr):
        L.append("\n## Cross-machine re-run checks\n")
        for e in json.load(open(rr)):
            L.append(f"- `{e['id']}` {e['orig_host']} -> {e['rerun_host']}: "
                     f"{'MATCH' if e['match'] else 'MISMATCH'}")

    open(f"{Q}/MORNING_REPORT.md", "w").write("\n".join(L) + "\n")
    print(f"report -> {Q}/MORNING_REPORT.md | figs -> {FIG}/")


if __name__ == "__main__":
    main()
