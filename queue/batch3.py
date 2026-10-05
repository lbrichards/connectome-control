#!/usr/bin/env python3
"""Batch 3: teacher-quality sweep. Does the worm-vs-shuffle gap widen as
teacher quality drops? (error-tolerance hypothesis)

3 relay quality levels (q91/q80/q67 asset dirs) x arms:
  worm x10, shuffle graphs 0-19 x1, dense-78 x10  -> 40 jobs/level, 120 total.
All distill_bc with relay_dir pointing at the level's assets.
"""
import hashlib, json, sys

STD = dict(steps=3000, dagger=1, dagger_steps=2000,
           eval_ep=100, eval_steps=1000)

def jid(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b3jobs.jsonl", "w") as f:
    def emit(level, cost, **kw):
        cfg = dict(kw, tier=1, cost=cost, relay_dir=f"relay_{level}")
        cfg["id"] = f"b3{level}_{cfg['kind']}_s{cfg['tseed']}_{jid(cfg)}"
        f.write(json.dumps(cfg) + "\n")
    for level in ("q91", "q80", "q67"):
        for g in range(20):
            emit(level, 3, type="distill_bc", kind=f"shuffle{g}", tseed=0, **STD)
        for t in range(10):
            emit(level, 2, type="distill_bc", kind="worm", tseed=t, **STD)
            emit(level, 1, type="distill_bc", kind="dense78", tseed=t, **STD)
print("batch-3 jobs written")
