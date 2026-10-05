# connectome-control

Clean replication of the C. elegans connectome cart-pole study, protocol v4.
History and preliminary results: [connectome-control-prototype]
(https://github.com/lbrichards/connectome-control-prototype) (archived).

## Protocol v4 (encoded in `src/connectome_control/__init__.py` + `protocol.py`)
- 20 ms action-delay canon (ms-based; steps = round(ms/1000*Hz)).
- Positions-only observations; NO pending-action inputs for imitation arms.
- Teachers delay-aware (one-step RK4 prediction); `teacher_v4` is the only
  DAgger/demo label source — both gates pass at 100%/0% (hanging + off-centre).
- Typed shuffle control: per-class, degree-exact double-edge swaps.
- Metrics: held + off-track co-primary; pole-fell reported as censored;
  catch conversion split catchable (|thd|<=1.0 rad/s at entry) vs fly-through.

## Run
```
uv venv --python 3.12 .venv && uv pip sync requirements.lock
.venv/bin/python -m pip install -e . --no-deps   # or: PYTHONPATH=src
```
Single job: `.venv/bin/python -m connectome_control.jobs job.json`

## Multi-machine (pull queue; macmini coordinates, all hosts work)
1. coordinator: `python smoke/make_smoke_jobs.py` (or real job maker)
   and ensure `bin/claim.sh` is at `~/cc-queue/claim.sh`.
2. each host: `bin/preflight.sh mbp mba` must PASS (same commit, same
   lockfile versions, CPU backend).
3. each host: N copies of `bin/worker.sh macmini` (slots: mbp 6,
   macmini 4, mba 2 — mba is fanless, do not oversubscribe), under
   `caffeinate -is` so machines stay awake.
4. results land in coordinator `~/cc-queue/results/`; aggregate with
   `analysis/aggregate.py`.

Every result JSON carries a manifest: graph spec, seed, budget, protocol
version, git hash, machine, library versions, weights sha256. Jobs are
deterministic (CPU, single-thread, seeded): the same job on two machines
must produce identical weight hashes.

## Secrets policy

No credentials in the repo, ever. Deployment secrets (Vercel, R2, etc.) live in the provider's environment-variable settings; local development uses `.env` files which are gitignored (`web/.env.example` documents expected keys). Worker host aliases are supplied via `CC_HOSTS` or `~/cc-queue/hosts.txt`, not committed. Licensed under the MIT License (see LICENSE).

## Web demo model selection

Rule (fixed): after batch-2 Tier 1 completes, the demo ships the worm at the MEDIAN held-rate among the 20 seeds; if that median is below the 20% demo-grade bar, the BEST of the 20 seeds is shipped and labelled as such. No automatic pickup on thresholds at other times.

Current selection: **v4 C. elegans, seed 16, best of 20 seeds (median failed demo-grade) (94% held, 4.3s to catch, 93% conversion) — criteria: ttc<=5s & quiet>=80% & convC>=50%**

## Notes for protocol v5 (logged mid-study, v4 unchanged)

- Pump-direction chatter audit (2026-10-05), prompted by the web demo's
  classical controller buzzing at full force near rest. The demo bug was a
  delay-induced limit cycle: deciding `sign(theta_dot cos theta)` from the
  CURRENT state while the force lands one delay later reverses theta_dot
  every tick (fixed in the demo only: pump on the queue-predicted state +
  `tanh(20 d)` direction). The research stack was audited and is CLEAN —
  `teacher_v4`'s one-RK4-step prediction through the pending action already
  provides the delay compensation: 0 full-amplitude sign-flip transitions in
  100 teacher closed-loop hang episodes, 0/429 distillation episodes, and 0
  flips across all 20 batch-2 worm students (correlation with held therefore
  undefined/moot). For v5, consider replacing `_pump`'s hard `sign()` with
  `tanh(k d)` anyway for regularity near rest; no evidence it matters at
  20 ms delay.
