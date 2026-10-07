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

Rule (since 2026-10-06, both tabs): each arm ships its BEST demo-grade seed of the 20 batch-2 seeds (time-to-catch <= 5 s, quiet-hold >= 80%, conversion on catchable >= 50%), labelled 'best of 20 seeds' — showcase vs showcase. Research claims are unaffected; note the research result is NO ESTABLISHED DIFFERENCE between worm and dense-78 across two teachers (pooled two-sided p = 0.078).

Current selection: **v4 C. elegans, seed 16, best of 20 seeds (94% held, 4.3s to catch, 93% conversion) — criteria: ttc<=5s & quiet>=80% & convC>=50%**

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

## PRE-REGISTRATION (2026-10-06, committed before any Task A/B results)

Protocol v4 throughout; same fleet rules (manifests, leases <=180 s with
heartbeats, <=2 retries, halt conditions, determinism spot checks: 2 DONE
jobs re-run on a second host must match weights sha256 bitwise).

### Task A — partial rewiring of the worm

Graphs: rewire a fraction f of worm edges by typed, degree-preserving
double-edge swaps (identical move set and class rules as the batch-2 typed
shuffle), stopping when the fraction of ORIGINAL edges no longer present
reaches the target. f is always REALIZED f, measured per graph against the
original edge set (chem directed + gap undirected pairs, diagonal gap
excluded) and recorded in the manifest. Targets: 0.10, 0.25, 0.50, 0.75.
Measured constraint, stated in advance: a full typed shuffle realizes only
f ~= 0.91 (typed degree preservation forces ~9% edge overlap), so the
curve's right endpoint is the batch-2 shuffle arm at realized f ~= 0.91,
NOT f = 1. Endpoints REUSE batch 2 unchanged: f=0 = the 20 batch-2 worm
runs; f~=0.91 = the 60 batch-2 shuffle runs. All Task-A jobs use the
IDENTICAL batch-2 relay assets, distillation dataset, training budget
(steps=3000, dagger=1, dagger_steps=2000) and eval (100 ep x 1000 steps).
Per target level: 8 independent graphs (rewire seeds 0-7) x 2 training
seeds (0, 1) = 16 runs; 64 new jobs total.

Robustness: run_robust (seed 777, unchanged) on EVERY Task-A model plus
reuse of the batch-2 endpoint models' existing suites; no gate.

PRIMARY (pre-registered): (1) Jonckheere-Terpstra test for a DECREASING
trend of per-run held-rate across the 6 ordered levels (f = 0, 0.10, 0.25,
0.50, 0.75, ~0.91), permutation p (10,000 label permutations within the
pooled sample, JT statistic, one-sided). (2) f50: fit a 4-parameter
logistic held_med(f) = L + (U - L) / (1 + exp(k (f - f0))) to the six
level medians against mean realized f per level; f50 is the realized f
where the fitted curve crosses the midpoint of its fitted values at the
two endpoints. CI: bootstrap runs within each level (1,000 resamples,
percentile 2.5/97.5). SECONDARY: same JT trend test on per-model kick-AUC
and noise-AUC (mean survival across the 3 levels, as in the batch-2
supplement), Holm across these 2 tests. Anything else is exploratory and
will be labelled as such.

### Task B — replication with a second relay (relay B)

Relay B: built with relay.py exactly as the batch-2 relay (relay A) but
with NEW brain seeds (swing seed 10, catch seed 11); same stage gates
(swing reach >= 85%, catch held >= 90%, relay held >= 50%); relay B's
held-rate recorded before any student training. Students (all distilled
from relay B, batch-2 budget unchanged): worm x training seeds 0-19 (same
init seeds as batch 2, enabling a paired exploratory comparison), typed
shuffle x 30 NEW graphs (seeds 200-229; disjoint from batch-2 graphs 0-59
and batch-3 graphs) x 1 seed, dense-78 x training seeds 0-9. 60 jobs.
Robustness suite on every model, no gate.

PRIMARY (pre-registered): worm vs shuffle held-rate, one-sided
Mann-Whitney (worm > shuffle), plus the worm median's percentile within
the 30-graph shuffle distribution. SECONDARY: kick-AUC and noise-AUC worm
vs shuffle (one-sided MW, Holm across the 2); worm vs dense-78 held-rate
(two-sided MW). Report relay B results side by side with relay A
(batch 2) for every endpoint. Replication criterion, stated in advance:
the worm-vs-shuffle held-rate direction reproduces with one-sided
p < 0.05.

### Task A extension — targeted rewiring (pre-registered 2026-10-06,
### committed before any results; Task C ablation interim was already
### seen and is the HYPOTHESIS SOURCE, stated for transparency)

Question: is the worm advantage concentrated in the input-adjacent wiring?
Measured design constraints, stated in advance: the 22 injected amphid
sensory neurons have 316 chemical out-edges (5.2% of all 6,026 edges), so
the requested "10% of edges" cannot be confined to that set; and within-
pool swap recreation caps removals at ~220 of the 316. Matched budget used
instead: K = 200 original edges removed in BOTH variants (= 63% of the
input wiring, 3.3% of all edges). rwS<g>: typed degree-preserving swaps
confined to sensory out-edges. rwX<g>: swaps confined to all OTHER
chemical edges (realized counts land at 200-213; the spared variant's
slight excess is conservative against the hypothesis). Gap junctions
untouched in both. 8 graphs (seeds 0-7) x 2 training seeds (0,1) = 16
runs per variant, 32 jobs, batch-2 relay/dataset/budget, queued at tier 2
behind Task B.

PRIMARY (pre-registered): held-rate rwS vs rwX, one-sided Mann-Whitney
(rwS < rwX; direction from the Task C interim finding that ablation
importance concentrates on input-adjacent neurons). SECONDARY: two-sided
comparison of each variant against the Task-A rw10 level (nearest
whole-graph budget); kick/noise AUC rwS vs rwX (one-sided, Holm over 2).

### Task C extension — positional control (shuffle ablations)

Same single-neuron ablation suite on 10 batch-2 shuffle models (the 5
best and 5 median by held-rate). Question: does ablation importance
concentrate on the input neurons' DIRECT chemical targets (1-hop
neighbours in each model's own graph) there too? Comparison metric:
rank-biserial separation of ablation effects, 1-hop-from-input nodes vs
the rest, worms vs shuffles. Exploratory (labelled as such; no
pre-registered test — Task C is descriptive).

#### Task B deviation log (2026-10-06, before any Task B results)
Relay B's catch brain at the pre-registered seed 11 (2,500 steps, the
relay-A budget) FAILED the >=90% catch gate (44%). Per the gate's purpose,
the teacher is retried: catch seed 11 at 6,000 steps, then seeds 12, 13 if
needed; first to pass the unchanged gate is used. Swing brain (seed 10,
100% reach) is kept. The pre-registered student analyses are unchanged;
the realized catch seed/budget will be reported in the Task B report.

#### Relay B stopping rule (2026-10-06, logged while seed-13 attempt was
#### already training, before its result was known)
Ladder: catch seed 13 @ 6,000 steps; then seeds 14 and 15 @ 10,000. Max 3
more attempts after seeds 11/12 @ 6,000 (84% best so far) and the original
seed 11 @ 2,500. The >=90% gate is UNCHANGED. First pass -> use it. If
none passes: STOP; no sub-gate relay is used for any primary analysis;
Task B replication is deferred and the Task B report documents the full
attempt table instead. Catch-brain pass rates across all attempts (relay
A + B) will be reported as a finding on teacher-building variance.

Clarification (2026-10-06, logged during rung 14, before its result):
seed 13 @ 6,000 PASSED the catch gate (>=90%) but the combined relay
failed the relay gate (held 40% < 50%). Both stage gates were always
required to use a relay; a rung failing EITHER gate counts as a failed
attempt. Ladder unchanged: rungs 14 and 15 @ 10,000 remain the last two.

### Task A extension 2 — FULL input-pathway rewiring (pre-registered
### 2026-10-06, committed before any results; motivated by the low-power
### caveat on the K=200 targeted test)

rwSF<g>: every original input-pathway edge removed (316/316, verified per
seed) via typed degree-preserving swaps; within-pool swaps plateau, so
surviving pool originals are swapped against random same-class partners,
each such swap removing one non-pool original as measured COLLATERAL
(268-296 across seeds 0-11; totals 584-612 edges, realized f ~0.10 —
comparable to Task A's rw10 level). rwXF<g>: a per-graph MATCHED total of
original edges removed entirely OUTSIDE the pool (realized within ~3% of
the rwSF total; pool untouched). Gap junctions untouched in both. 12
graphs (seeds 0-11) x 2 training seeds (0,1) = 24 runs per arm, 48 jobs,
batch-2 relay/dataset/budget, queued at tier 3 behind Task B.

PRIMARY (pre-registered): held-rate, one-sided Mann-Whitney
(rwSF < rwXF). SECONDARY: kick-AUC and noise-AUC (one-sided, Holm over
2). Exploratory: both arms vs the Task A rw10 level (nearest whole-graph
dose).

## PRE-REGISTRATION: Task G — graph properties that predict performance
## (correlational; committed 2026-10-06 BEFORE adding seeds or computing
## any property value)

Data: the 90 shuffled graphs (60 relay-A/batch-2, 30 relay-B/Task-B), the
32 Task-A partially rewired graphs, and the worm. Two additional training
seeds (tseed 1, 2) per shuffled graph, same protocol and same relay as
each graph's original run (180 jobs, tier 4, behind Task F). Per-graph
outcome: mean held-rate across its 3 seeds; kick/noise AUC secondaries.

Fixed property list (computed on each graph; "union graph" = binarized
chem + chem^T + gap, diagonal zeroed; "directed graph" = binarized chem):
1. Reciprocity: fraction of directed chem edges whose reverse edge exists
   (overall, and restricted to neuron->neuron edges).
2. Directed 3-node motifs on chem: feedforward-loop count and 3-cycle
   (feedback) count.
3. 2-cycles and 3-cycles (chem) with all nodes within 2 directed hops
   downstream of the 22 input neurons.
4. Mean directed shortest-path length (chem, unweighted) from input
   neurons to effector nodes (muscles + lowercase nodes), averaged over
   reachable pairs; unreachable pairs reported as a count, not imputed.
5. Number of nodes reachable within 2 directed hops of the inputs.
6. Largest strongly connected component size (chem).
7. Spectral radius of the count-weighted adjacency (chem + gap).
8. Mean clustering coefficient of the union graph.
9. Modularity: networkx louvain_communities (union graph, weight=None,
   seed=0), modularity of the returned partition.
10. Rich-club coefficient of the union graph at k* = 77 (the minimum
    union-graph degree among the worm's command hubs AVAL/AVAR/AVBL/
    AVBR), unnormalized (stated as such; no null-model normalization).

PRIMARY (pre-registered): Spearman correlation of each property with
per-graph mean held across the 90 shuffled graphs; Holm over exactly
these 10 tests: (1) reciprocity-overall, (2) FFL count, (3) 3-cycle
count, (4) near-input cycles (2-cycles + 3-cycles within 2 hops of
inputs, one summed statistic), (5) input->effector mean path length,
(6) 2-hop reach, (7) LSCC size, (8) spectral radius, (9) clustering,
(10) modularity. Rich-club@77 and neuron->neuron reciprocity are
reported alongside but OUTSIDE the Holm family (register note: family
fixed at these 10 before any computation).
Out-of-sample: ridge regression (alpha chosen by LOO-CV on relay-A
graphs only, standardized properties) trained on the 60 relay-A graphs,
predicting the 30 relay-B graphs; report Spearman(predicted, observed).
Placement: predict the worm and the Task-A rewired graphs from the same
model; report predicted-vs-observed and whether prediction declines with
realized f. Anything further is exploratory and labelled.

## Web demo dense-78 tab selection

Rule (same as the worm's, since 2026-10-06): the BEST demo-grade seed of the 20, labelled 'best of 20 seeds'. Tab blurb: "A conventional dense network with the same number of connections (about 6,000)." Research context: no established worm/dense-78 difference across two teachers (pooled).

Current selection: **v4 dense-78, seed 2, best of 20 seeds (91% held, 3.0s to catch, 90% conversion) — criteria: ttc<=5s & quiet>=80% & convC>=50%**

## PRE-REGISTRATION: worm vs dense-78 definitive sample (committed
## 2026-10-06 ~21:00, BEFORE any of these runs exist)

Design (fixed in advance): bring every cell to 40 training seeds —
relay A: +20 worm (tseeds 20-39), +20 dense-78 (tseeds 20-39);
relay B: +20 worm (tseeds 20-39), +30 dense-78 (tseeds 10-39).
90 new jobs, identical protocol/budget/relays as the original cells;
robustness suite (run_robust, seed 777) on every new model. Queued at
tier 5, strictly behind Task G and its robustness fill; no code changes
and no deploys required.

PRE-REGISTERED ANALYSIS (run ONCE, only when all 160 runs and their
robustness results are present; the script refuses otherwise):
- Primary: worm vs dense-78 held-rate, TWO-SIDED, stratified by relay
  (van Elteren, weights 1/(N_s+1), tie-corrected); report pooled and
  per-relay rank-biserial effects and Cochran's Q heterogeneity.
- Secondary: kick-AUC and noise-AUC, same stratified test, Holm over 2.

STOPPING RULE: this is the FINAL sample for the worm-vs-dense-78
question. No further seeds will be added regardless of the result.

## Reproduction (one command per result)

All runs are CPU-only, single-threaded per job, bitwise-deterministic on
arm64 (seeds in each job config; weights sha256 in each manifest).
`uv venv && uv pip sync requirements.lock` prepares the environment.
Data DOI: 10.5281/zenodo.XXXXXXX (placeholder until minted).

| result | command | runtime (one M-class core) |
|---|---|---|
| relay-A teacher + distillation dataset | `python -m connectome_control.relay` | ~1 h |
| relay-B teacher (Task B) | `python -m connectome_control.relay relayB` | ~2-4 h (seed ladder) |
| one Tier-1 student (any arm) | `python -m connectome_control.jobs <job.json>` (configs emitted by `queue/batch2.py load` / `queue/batch4.py loadA..H`) | ~30 min |
| full batch-2 replication (260 jobs) | `queue/batch2.py pilot && queue/batch2.py load <target>` + workers (`bin/worker2.sh`) | ~1 night on 17 workers |
| Tasks A/T/B/F/G/H (466 jobs) | `queue/batch4.py loadA|loadT|loadB|loadF|loadG|loadH` | ~1 day on 17 workers |
| robustness suite for a model | `connectome_control.jobs.run_robust({'ref': <id>})` | ~2.5 min |
| pre-registered analyses + reports | `python analysis/taska_report.py` (likewise taskb/taskf/taskg/taskh, stratified_relays, taskt) | minutes |
| interpretability (Task C) | `python analysis/interp_worms.py all && python analysis/taskc_report.py` | ~2 h |
| paper figures (full-sample) | `python analysis/paper_refresh.py` | ~5 min |
| web demo models + parity | `python tools/select_demo_model.py [worm|dense78] && python tools/export_web_model.py && cd web && npx vitest run` | ~10 min |
