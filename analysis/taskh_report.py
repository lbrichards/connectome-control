"""Definitive worm-vs-dense-78 report (pre-registered, FINAL sample).

REFUSES to run unless all 160 runs (40 per cell) and robustness results
for every model are present. Run ONCE on complete data.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from stratified_relays import van_elteren, per_stratum, meta

Q = os.path.expanduser("~/cc-queue")


def collect():
    """cells[(relay, arm)] -> list of (held, ref)."""
    cells = {("A", "worm"): [], ("A", "dense78"): [],
             ("B", "worm"): [], ("B", "dense78"): []}
    for f in glob.glob(f"{Q}/results/b2t1_*.json"):
        r = json.load(open(f)); k = r["job"]["kind"]
        if k in ("worm", "dense78"):
            cells[("A", k)].append((r["metrics"]["held"],
                                    os.path.basename(f)[:-5]))
    for pat in (f"{Q}/results/b4*B_*.json", f"{Q}/results/b4*H_*.json"):
        for f in glob.glob(pat):
            r = json.load(open(f)); j = r["job"]; k = j["kind"]
            if k not in ("worm", "dense78"):
                continue
            relay = "B" if j.get("relay_dir") == "relay_B" else "A"
            cells[(relay, k)].append((r["metrics"]["held"],
                                      os.path.basename(f)[:-5]))
    return cells


def load_aucs():
    out = {}
    for d in ("results", "results_t4_full", "results_t4B", "results_t4H"):
        for f in glob.glob(f"{Q}/{d}/t4*.json"):
            r = json.load(open(f))
            m = r["metrics"]
            out[r["job"]["ref"]] = (
                float(np.mean(list(m["kick_thd"].values()))),
                float(np.mean(list(m["noise_obs"].values()))))
    return out


def main():
    cells = collect()
    for key, v in cells.items():
        assert len(v) == 40, f"cell {key} has {len(v)}/40 runs -- REFUSING"
    aucs = load_aucs()
    missing = [ref for v in cells.values() for _, ref in v
               if ref not in aucs]
    assert not missing, f"{len(missing)} models lack robustness -- REFUSING"

    L = ["# Definitive worm vs dense-78 (pre-registered FINAL sample)", "",
         "40 seeds per cell x 2 relays x 2 arms = 160 runs. Stopping rule "
         "(README): no further seeds regardless of result.", ""]

    def strat_test(metric, alternative, name):
        pairs, rows = [], []
        for st in ("A", "B"):
            if metric == "held":
                x = [h for h, _ in cells[(st, "worm")]]
                y = [h for h, _ in cells[(st, "dense78")]]
            else:
                mi = 0 if metric == "kick" else 1
                x = [aucs[r][mi] for _, r in cells[(st, "worm")]]
                y = [aucs[r][mi] for _, r in cells[(st, "dense78")]]
            pairs.append((x, y))
            U, p, rb, v = per_stratum(x, y, alternative)
            rows.append(
                f"- relay {st}: worm median {np.median(x)*100:.0f}"
                f"{'%' if metric=='held' else ' (AUC x100)'} vs dense "
                f"{np.median(y)*100:.0f}"
                f"{'%' if metric=='held' else ''}; "
                f"MW p={p:.3f}, r={rb:+.2f}")
            if st == "A":
                rbs, vs = [rb], [v]
            else:
                rbs.append(rb); vs.append(v)
        z, p_ve = van_elteren(pairs, alternative)
        rb, se, Qh, p_het = meta(rbs, vs)
        rows.append(f"- POOLED van Elteren: z={z:+.2f}, p={p_ve:.4f}; "
                    f"pooled r={rb:+.2f} (SE {se:.2f}); heterogeneity "
                    f"Q={Qh:.2f} p={p_het:.3f}")
        return [f"## {name}"] + rows + [""], p_ve

    blk, _ = strat_test("held", "two-sided",
                        "Pre-registered primary: held-rate (two-sided, "
                        "stratified)")
    L += blk
    sec = []
    for met, nm in (("kick", "kick-AUC"), ("noise", "noise-AUC")):
        blk, pv = strat_test(met, "two-sided",
                             f"Pre-registered secondary: {nm}")
        sec.append((pv, blk))
    sec.sort(key=lambda t: t[0])
    prev = 0
    for i, (pv, blk) in enumerate(sec):
        adj = min(1.0, max(prev, (2 - i) * pv)); prev = adj
        blk[-2] += f"  [Holm p={adj:.4f}]"
        L += blk

    out = f"{Q}/WORM_VS_DENSE_FINAL.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
