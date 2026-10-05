#!/usr/bin/env python3
"""Corrected batch (batch 2): pilot gate -> full tiers.

Usage:
  batch2.py pilot            run the 8 pilot-gate jobs locally (parallel),
                             evaluate the gate, print verdict + tier-2 target
  batch2.py load <target>    generate + load the full tiers into the queue

Pilot gate: 4 worm seeds + 4 shuffle graphs on corrected Tier 1.
Proceed only if worm catch-at-all >= 3/4 (clearly off the floor).
Tier-2 target = median worm pilot final MSE; sanity: a matched pilot job
must need >=1000 steps and not cap.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = f"{REPO}/.venv/bin/python"

STD = dict(steps=3000, dagger=1, dagger_steps=2000,
           eval_ep=100, eval_steps=1000)


def jid(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def mk(tier, cost, **kw):
    cfg = dict(kw, tier=tier, cost=cost)
    cfg["id"] = f"b2t{tier}_{cfg['kind']}_s{cfg['tseed']}_{jid(cfg)}"
    return cfg


def pilot():
    jobs = [mk(1, 2, type="distill_bc", kind="worm", tseed=t, **STD)
            for t in range(4)]
    jobs += [mk(1, 3, type="distill_bc", kind=f"shuffle{g}", tseed=0, **STD)
             for g in range(4)]
    os.makedirs("/tmp/b2pilot", exist_ok=True)
    procs = []
    for j in jobs:
        p = f"/tmp/b2pilot/{j['id']}.json"
        json.dump(j, open(p, "w"))
        procs.append(subprocess.Popen(
            [PY, "-m", "connectome_control.jobs", p],
            cwd=REPO, env={**os.environ, "OMP_NUM_THREADS": "1"}))
    for p in procs:
        p.wait()
    res = []
    for j in jobs:
        r = json.load(open(f"{REPO}/results/{j['id']}.json"))
        res.append(r)
        m = r["metrics"]
        print(f"{j['kind']:<10} s{j['tseed']}  held {m['held']*100:3.0f}%  "
              f"off {m['off_track']*100:3.0f}%  quiet {m['quiet_hold']*100:3.0f}%  "
              f"mse {r['final_mse']:.3f}")
    worms = [r for r in res if r["job"]["kind"] == "worm"]
    catch = sum(r["metrics"]["held"] > 0 for r in worms)
    target = round(float(np.median([r["final_mse"] for r in worms])), 4)
    print(f"\nworm catch-at-all: {catch}/4 (gate: >=3)")
    print(f"tier-2 target (median worm pilot MSE): {target}")
    if catch < 3:
        print("PILOT GATE FAILED -- do not load the full batch")
        sys.exit(1)
    # sanity: matched job with this target must be non-trivial
    mj = mk(0, 1, type="distill_matched", kind="worm", tseed=50,
            target=target, cap=12000, eval_ep=30, eval_steps=500)
    p = f"/tmp/b2pilot/{mj['id']}.json"
    json.dump(mj, open(p, "w"))
    subprocess.run([PY, "-m", "connectome_control.jobs", p], cwd=REPO,
                   check=True)
    r = json.load(open(f"{REPO}/results/{mj['id']}.json"))
    st = r["steps_to_target"]
    print(f"matched sanity: steps_to_target={st} hit={r['hit_target']}")
    if not (1000 <= st < 12000 and r["hit_target"]):
        print("TIER-2 TARGET SANITY FAILED (trivial or capped) -- adjust")
        sys.exit(1)
    print(f"PILOT GATE PASSED; load with: batch2.py load {target}")


def load(target):
    target = float(target)
    out = "/tmp/b2jobs.jsonl"
    with open(out, "w") as f:
        def emit(cfg):
            f.write(json.dumps(cfg) + "\n")
        # Tier 1: 60 shuffle graphs x1, worm x20, dense78 x20
        for g in range(60):
            emit(mk(1, 3, type="distill_bc", kind=f"shuffle{g}", tseed=0,
                    **STD))
        for t in range(20):
            emit(mk(1, 2, type="distill_bc", kind="worm", tseed=t, **STD))
            emit(mk(1, 1, type="distill_bc", kind="dense78", tseed=t, **STD))
        # Tier 2: matched at the calibrated target
        M = dict(target=target, cap=12000, eval_ep=100, eval_steps=1000)
        for t in range(20):
            emit(mk(2, 2, type="distill_matched", kind="worm", tseed=t, **M))
            emit(mk(2, 1, type="distill_matched", kind="dense78", tseed=t,
                    **M))
        for g in range(20):
            emit(mk(2, 3, type="distill_matched", kind=f"shuffle{g}",
                    tseed=0, **M))
        # Tier 3: dense-448, same target, extended cap
        for t in range(5):
            emit(mk(3, 10, type="distill_conv448", kind="dense448", tseed=t,
                    target=target, cap=40000, eval_ep=100, eval_steps=1000))
    subprocess.run([PY, f"{REPO}/queue/qctl.py", "load", out], check=True)
    print("full batch loaded (tier 4 emitted by pilot on tier-1 drain, "
          "refs gated on quiet_hold >= 0.80)")


if __name__ == "__main__":
    {"pilot": pilot, "load": lambda: load(sys.argv[2])}[sys.argv[1]]()
