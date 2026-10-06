#!/usr/bin/env python3
"""Task A (partial rewiring) + Task B (relay-B replication) job loaders.

Pre-registered in README (committed 2026-10-06 before any results).

Usage:
  batch4.py loadA    64 jobs: rw{10,25,50,75}g{0..7} x tseed {0,1},
                     batch-2 relay/dataset/budget (endpoints reuse batch 2)
  batch4.py loadB    60 jobs: worm s0-19, shuffle g200-229 s0, dense78
                     s0-9, all distilled from relay_B (gates must have
                     passed; refuses to load if assets are missing)

Interleaving: both tasks share tier 1 and cost 2; ids start with 4 hash
hex chars so qctl's (tier, cost DESC, id ASC) claim order mixes A and B
pseudorandomly once both are loaded.
"""

import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = f"{REPO}/.venv/bin/python"

STD = dict(steps=3000, dagger=1, dagger_steps=2000,
           eval_ep=100, eval_steps=1000)


def jid(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def mk(task, **kw):
    cfg = dict(kw, tier=1, cost=2)
    h = jid(cfg)
    cfg["id"] = f"b4{h[:4]}{task}_{cfg['kind']}_s{cfg['tseed']}_{h[4:12]}"
    return cfg


def _load(jobs, tag):
    out = f"/tmp/b4jobs_{tag}.jsonl"
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    subprocess.run([PY, f"{REPO}/queue/qctl.py", "load", out], check=True)
    print(f"{tag}: {len(jobs)} jobs loaded")


def loadA():
    jobs = [mk("A", type="distill_bc", kind=f"rw{p}g{g}", tseed=t, **STD)
            for p in (10, 25, 50, 75) for g in range(8) for t in (0, 1)]
    _load(jobs, "A")


def loadB():
    rd = f"{REPO}/src/connectome_control/data/relay_B"
    assert os.path.exists(f"{rd}/SHA256SUMS"), "relay_B assets missing"
    assert os.path.exists(f"{rd}/HELD"), "relay_B held-rate not recorded"
    print(f"relay_B held-rate: {open(f'{rd}/HELD').read().strip()}")
    jobs = [mk("B", type="distill_bc", kind="worm", tseed=t,
               relay_dir="relay_B", **STD) for t in range(20)]
    jobs += [mk("B", type="distill_bc", kind=f"shuffle{g}", tseed=0,
                relay_dir="relay_B", **STD) for g in range(200, 230)]
    jobs += [mk("B", type="distill_bc", kind="dense78", tseed=t,
                relay_dir="relay_B", **STD) for t in range(10)]
    _load(jobs, "B")


def loadT():
    """Targeted-rewiring extension (pre-registered): tier 2, behind Task B."""
    jobs = []
    for v in ("S", "X"):
        for g in range(8):
            for t in (0, 1):
                cfg = dict(type="distill_bc", kind=f"rw{v}g{g}", tseed=t,
                           tier=2, cost=2, **STD)
                h = jid(cfg)
                cfg["id"] = f"b4{h[:4]}T_{cfg['kind']}_s{t}_{h[4:12]}"
                jobs.append(cfg)
    _load(jobs, "T")


def loadF():
    """Full input-pathway rewiring (pre-registered ext. 2): tier 3."""
    jobs = []
    for v in ("SF", "XF"):
        for g in range(12):
            for t in (0, 1):
                cfg = dict(type="distill_bc", kind=f"rw{v}g{g}", tseed=t,
                           tier=3, cost=2, **STD)
                h = jid(cfg)
                cfg["id"] = f"b4{h[:4]}F_{cfg['kind']}_s{t}_{h[4:12]}"
                jobs.append(cfg)
    _load(jobs, "F")


def loadG():
    """Task G seed extension (pre-registered): +2 seeds per shuffled
    graph, same relay as the graph's original run. Tier 4 (behind F)."""
    jobs = []
    for g in range(60):                       # relay-A graphs (batch 2)
        for t in (1, 2):
            cfg = dict(type="distill_bc", kind=f"shuffle{g}", tseed=t,
                       tier=4, cost=2, **STD)
            h = jid(cfg)
            cfg["id"] = f"b4{h[:4]}G_{cfg['kind']}_s{t}_{h[4:12]}"
            jobs.append(cfg)
    for g in range(200, 230):                 # relay-B graphs (Task B)
        for t in (1, 2):
            cfg = dict(type="distill_bc", kind=f"shuffle{g}", tseed=t,
                       relay_dir="relay_B", tier=4, cost=2, **STD)
            h = jid(cfg)
            cfg["id"] = f"b4{h[:4]}G_{cfg['kind']}_s{t}_{h[4:12]}"
            jobs.append(cfg)
    _load(jobs, "G")


if __name__ == "__main__":
    {"loadA": loadA, "loadB": loadB, "loadT": loadT, "loadF": loadF, "loadG": loadG}[sys.argv[1]]()
