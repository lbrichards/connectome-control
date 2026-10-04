#!/usr/bin/env python3
"""Generate the overnight job set as JSONL for qctl load.

Job ID = sha256 of the full config (idempotent; re-loading skips DONE).
Costs order longest-first within a tier (shuffles slowest empirically).
Tier 4 is emitted by a second pass once Tier 1 is DONE (needs refs) --
see make_tier4.py.
"""

import hashlib
import json
import sys

STD = dict(steps=3000, dagger=3, dagger_ep=96, dagger_steps=1500,
           eval_ep=100, eval_steps=1000)
MATCH = dict(target=0.10, cap=12000, eval_ep=100, eval_steps=1000)


def jid(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def emit(out, tier, cost, cfg):
    cfg = dict(cfg, tier=tier, cost=cost)
    cfg["id"] = f"t{tier}_{cfg['kind']}_s{cfg['tseed']}_{jid(cfg)}"
    out.write(json.dumps(cfg) + "\n")


with open(sys.argv[1] if len(sys.argv) > 1 else "jobs.jsonl", "w") as f:
    # ---- Tier 1: core comparison (BC + DAgger vs v4 teacher)
    for g in range(20):
        for t in (0, 1):
            emit(f, 1, 3, dict(type="bc", kind=f"shuffle{g}", tseed=t,
                               demo_seed=100 + 2 * g + t, **STD))
    for t in range(20):
        emit(f, 1, 2, dict(type="bc", kind="worm", tseed=t, demo_seed=t, **STD))
    for t in range(20):
        emit(f, 1, 1, dict(type="bc", kind="dense78", tseed=t,
                           demo_seed=t, **STD))
    # ---- Tier 2: matched-MSE (no DAgger; mediation test)
    for t in range(20):
        emit(f, 2, 2, dict(type="matched", kind="worm", tseed=t,
                           demo_seed=t, **MATCH))
        emit(f, 2, 1, dict(type="matched", kind="dense78", tseed=t,
                           demo_seed=t, **MATCH))
    for g in range(20):
        emit(f, 2, 3, dict(type="matched", kind=f"shuffle{g}", tseed=0,
                           demo_seed=100 + 2 * g, **MATCH))
    # ---- Tier 3: dense-448 to convergence
    for t in range(5):
        emit(f, 3, 10, dict(type="conv448", kind="dense448", tseed=t,
                            demo_seed=t, target=0.10, cap=40000,
                            eval_ep=100, eval_steps=1000))
print("jobs.jsonl written")
