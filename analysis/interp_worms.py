"""Task C: interpretability on the 20 batch-2 worm students (inference only).

Per model:
  1. single-neuron ablation (zero that neuron's state h -> activity and gap
     drive silenced) applied only during the SWING phase (before first entry
     into the upright band) or only during the CATCH phase (after), measured
     as held-rate drop vs the unablated baseline on a fixed start set;
  2. edge contributions: mean |W_chem[i,j] * act_j(t)| per phase on baseline
     rollouts; the global top-50 edges (max over phases) ablated one at a
     time (whole episode), held-rate drop recorded;
  3. activity PCA per phase + principal angles between swing/catch
     subspaces (top 10 PCs).
Aggregation across the 20 models happens in interp_report.py.

Run: python analysis/interp_worms.py <model.pt> <out.npz>   (one model)
     python analysis/interp_worms.py all                    (pool, nice)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from connectome_control import DT, F_MAX
from connectome_control.physics import rk4_batch, wrap
from connectome_control.protocol import hang_starts, obs_swing, UP_TOL, HOLD_N
from connectome_control.jobs import _build_net

Q = os.path.expanduser("~/cc-queue")
EPS, STEPS = 24, 600
torch.set_num_threads(1)


@torch.no_grad()
def rollout_abl(net, norm, starts, neuron=None, phase=None, zero_edge=None,
                collect=False):
    """Protocol-v4 closed loop (one-step delay) with optional ablation.
    neuron+phase: zero h[neuron] while in that phase ('swing' = before first
    band entry, 'catch' = after). zero_edge=(i,j): chem weight zeroed whole
    episode. collect: return (h history, entry times) too."""
    o_mu, o_sd, a_mu, a_sd = norm
    s = starts.copy(); n = len(s)
    h = torch.zeros(n, net.n)
    pre = net.precompute()
    if zero_edge is not None:
        wc, g, leak, decay = pre
        wc = wc.clone(); wc[zero_edge[0], zero_edge[1]] = 0.0
        pre = (wc, g, leak, decay)
    a = np.zeros(n)
    entered = np.zeros(n, bool)
    t_enter = np.full(n, STEPS, int)
    alive = np.ones(n, bool)
    up_hist = np.zeros((STEPS, n), bool)
    H = np.zeros((STEPS, n, net.n), np.float32) if collect else None
    for t in range(STEPS):
        ob = obs_swing(s).astype(np.float32)
        x = torch.from_numpy((ob - o_mu) / o_sd)
        h = net.cell(h, net.input_drive(x), pre)
        if neuron is not None:
            if phase == "swing":
                m = ~entered
            else:
                m = entered
            if m.any():
                h[torch.from_numpy(m), neuron] = 0.0
        if collect:
            H[t] = h.numpy()
        cmd = np.clip(net.read(h).squeeze(-1).numpy() * a_sd + a_mu,
                      -F_MAX, F_MAX)
        s = np.where(alive[:, None], rk4_batch(s, a, DT), s)
        a = cmd
        th = wrap(s[:, 2])
        up = np.abs(th) < UP_TOL
        just = up & ~entered
        t_enter[just] = t
        entered |= up
        up_hist[t] = up & alive
        alive &= np.abs(s[:, 0]) <= 3.0
    held = float((alive & up_hist[-HOLD_N:].all(0)).mean())
    if collect:
        return held, H, t_enter
    return held


def run_model(pt_path, out_path):
    ck = torch.load(pt_path, weights_only=False)
    job = ck["job"]
    net = _build_net(job["kind"], job["tseed"])
    net.load_state_dict(ck["state"])
    norm = ck["norm"]
    starts = hang_starts(EPS, np.random.default_rng(2026))

    base, H, t_enter = rollout_abl(net, norm, starts, collect=True)
    print(f"{job['id']}: baseline held {base*100:.0f}%", flush=True)

    # ---- 1. neuron ablations per phase
    drops = np.zeros((2, net.n), np.float32)
    for pi, phase in enumerate(("swing", "catch")):
        for k in range(net.n):
            drops[pi, k] = base - rollout_abl(net, norm, starts,
                                              neuron=k, phase=phase)
        print(f"  {phase} ablations done", flush=True)

    # ---- 2. edge contributions + top-50 edge ablations
    with torch.no_grad():
        wc = net.precompute()[0].numpy()
    act = np.tanh(H)                                  # (T, E, n)
    sw_mask = np.arange(STEPS)[:, None] < t_enter[None, :]
    contrib = np.zeros((2,) + wc.shape, np.float32)
    for pi, m in enumerate((sw_mask, ~sw_mask)):
        mean_abs_act = np.abs(act[m]).mean(0) if m.any() else \
            np.zeros(net.n, np.float32)
        contrib[pi] = np.abs(wc) * mean_abs_act[None, :]
    score = contrib.max(0)
    ii, jj = np.nonzero(score)
    order = np.argsort(-score[ii, jj])[:50]
    top_edges = np.stack([ii[order], jj[order]], 1)
    edge_drops = np.array([base - rollout_abl(net, norm, starts,
                                              zero_edge=(int(i), int(j)))
                           for i, j in top_edges], np.float32)
    print("  edge ablations done", flush=True)

    # ---- 3. PCA per phase + principal angles (top 10 PCs)
    X_sw = act[sw_mask]; X_ca = act[~sw_mask]
    out_pca = {}
    basis = {}
    for tag, X in (("swing", X_sw), ("catch", X_ca)):
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        ev = S**2 / max(len(X) - 1, 1)
        out_pca[f"evr_{tag}"] = (ev / ev.sum())[:20]
        basis[tag] = Vt[:10].T                        # (n, 10)
    sv = np.linalg.svd(basis["swing"].T @ basis["catch"],
                       compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(sv, -1, 1)))

    np.savez_compressed(
        out_path, base=base, drops=drops, top_edges=top_edges,
        edge_drops=edge_drops, edge_scores=score[top_edges[:, 0],
                                                 top_edges[:, 1]],
        angles=angles, held_b2=json.load(open(
            pt_path.replace(".pt", ".json")))["metrics"]["held"],
        **out_pca)
    print(f"  -> {out_path}", flush=True)


def main_all():
    from multiprocessing import Pool
    os.makedirs(f"{Q}/taskC", exist_ok=True)
    work = []
    for f in sorted(glob.glob(f"{Q}/results/b2t1_worm_s*.pt")):
        out = f"{Q}/taskC/{os.path.basename(f)[:-3]}.npz"
        if not os.path.exists(out):
            work.append((f, out))
    print(f"{len(work)} models to run", flush=True)
    with Pool(3) as p:
        p.starmap(run_model, work)
    print("TASK C MODELS DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        main_all()
    else:
        run_model(sys.argv[1], sys.argv[2])
