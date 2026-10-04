"""Job runner with manifests. Deterministic: CPU, single thread, seeded.

Job types (dispatch on job["type"]):
  bc       Tier 1: BC + DAgger vs teacher_v4 on pre-generated demos.
  matched  Tier 2: same, but train to EMA-MSE target (cap), no DAgger.
  conv448  Tier 3: dense-448, matched-target with extended cap.
  robust   Tier 4: robustness suite on a referenced DONE model
           (pole kicks, cart-position shifts, sensor noise).
           host_req=coordinator since it reads synced weights.

Every result carries a full manifest (graph spec, seed, budget, protocol
version, git hash, machine, library versions, weights sha256).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from . import PROTOCOL_VERSION, DT, F_MAX
from .graph import build_graph
from .model import ConnectomeRNN
from .physics import rk4, wrap
from .protocol import (obs_swing, hang_starts, rollout, metrics, UP_TOL,
                       HOLD_N)
from .teachers import teacher_v4

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def collect_demos(n_ep, seq, seed, noise=1.0):
    rng = np.random.default_rng(seed)
    O, A = [], []
    tries = 0
    while len(O) < n_ep and tries < 40 * n_ep:
        tries += 1
        s = hang_starts(1, rng)[0]
        a = 0.0
        obs, act, traj = [], [], []
        ok = True
        for _ in range(seq):
            obs.append([s[0], np.sin(s[2]), np.cos(s[2])])
            cmd = teacher_v4(s, a)
            act.append(cmd)
            traj.append(s[2])
            s = rk4(s, a, DT)
            a = float(np.clip(cmd + noise * rng.normal(), -F_MAX, F_MAX))
            if abs(s[0]) > 3.0:
                ok = False
                break
        if not ok or not (np.abs(wrap(np.array(traj))) < UP_TOL).any():
            continue
        O.append(obs); A.append(act)
    return np.array(O, np.float32), np.array(A, np.float32)


def _build_net(kind, tseed):
    chem, gap, in_idx, n = build_graph(kind)
    return ConnectomeRNN(chem, gap, np.asarray(in_idx, np.int64),
                         np.arange(n, dtype=np.int64), n_in=3, n_out=1,
                         seed=tseed, spectral_radius=1.3, in_gain=2.0,
                         dt_over_tau_init=0.2)


def _norm(O, A):
    d = O.shape[-1]
    return (O.reshape(-1, d).mean(0), O.reshape(-1, d).std(0) + 1e-6,
            A.mean(), A.std() + 1e-6)


def train_bc(O, A, net, steps, bs=12, lr=5e-3, seed=0,
             ema_target=None, cap=None):
    norm = _norm(O, A)
    o_mu, o_sd, a_mu, a_sd = norm
    xs = torch.from_numpy((O - o_mu) / o_sd)
    ys = torch.from_numpy((A - a_mu) / a_sd)
    wt = torch.from_numpy((1.0 + 7.0 * (O[..., 2] > 0.955)).astype(np.float32))
    total = cap if ema_target is not None else steps
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    g = torch.Generator().manual_seed(seed + 11)
    curve, ema, used, hit = [], None, total, ema_target is None
    for st in range(1, total + 1):
        i = torch.randint(0, len(xs), (bs,), generator=g)
        err = (net(xs[i]).squeeze(-1) - ys[i]) ** 2 * wt[i]
        loss = err.sum() / wt[i].sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        v = loss.item()
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        if st % 100 == 0:
            curve.append(round(v, 5))
        if ema_target is not None and st >= 300 and ema <= ema_target:
            used, hit = st, True
            break
    return norm, curve, used, hit, (round(ema, 5) if ema else None)


def dagger_round(net, norm, n_ep, seq, seed):
    label = lambda s, a: np.array([teacher_v4(si, ai)
                                   for si, ai in zip(s, a)])
    _, _, _, O, A = rollout(net, norm,
                            hang_starts(n_ep, np.random.default_rng(seed)),
                            seq, label_fn=label)
    return O.astype(np.float32), A.astype(np.float32)


def weights_hash(net):
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def manifest(job, net, extra):
    return {
        "protocol": PROTOCOL_VERSION, "git": _git_hash(),
        "machine": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__,
        "weights_sha256": weights_hash(net) if net is not None else None,
        **extra,
    }


# ------------------------------------------------------------- job types

def run_bc(job):
    from .datasets import load as load_demos
    torch.manual_seed(job["tseed"])
    net = _build_net(job["kind"], job["tseed"])
    O, A = load_demos(job["demo_seed"])
    norm, curve, _, _, _ = train_bc(O, A, net, job["steps"], seed=job["tseed"])
    for rnd in range(job.get("dagger", 0)):
        dO, dA = dagger_round(net, norm, job.get("dagger_ep", 96),
                              O.shape[1], 100 + rnd)
        O = np.concatenate([O, dO]); A = np.concatenate([A, dA])
        norm, curve, _, _, _ = train_bc(O, A, net,
                                        job.get("dagger_steps", 1500),
                                        seed=job["tseed"])
    TH, alive, exit_t = rollout(net, norm,
        hang_starts(job["eval_ep"], np.random.default_rng(999)),
        job["eval_steps"])
    return net, norm, metrics(TH, alive, exit_t), {"final_mse": curve[-1],
                                                   "curve": curve}


def run_matched(job):
    from .datasets import load as load_demos
    torch.manual_seed(job["tseed"])
    net = _build_net(job["kind"], job["tseed"])
    O, A = load_demos(job["demo_seed"])
    norm, curve, used, hit, ema = train_bc(
        O, A, net, 0, seed=job["tseed"],
        ema_target=job["target"], cap=job["cap"])
    TH, alive, exit_t = rollout(net, norm,
        hang_starts(job["eval_ep"], np.random.default_rng(999)),
        job["eval_steps"])
    return net, norm, metrics(TH, alive, exit_t), {
        "steps_to_target": used, "hit_target": hit, "ema_mse": ema,
        "final_mse": curve[-1] if curve else None}


def run_robust(job):
    """Perturbation suite on a referenced DONE model (coordinator-local)."""
    ref = os.path.expanduser(f"~/cc-queue/results/{job['ref']}.pt")
    ck = torch.load(ref, weights_only=False)
    net = _build_net(ck["job"]["kind"], ck["job"]["tseed"])
    net.load_state_dict(ck["state"])
    norm = ck["norm"]
    o_mu, o_sd, a_mu, a_sd = norm
    rng = np.random.default_rng(777)
    out = {}
    for mode, mags in (("kick_thd", [0.3, 0.6, 1.0]),
                       ("shift_x", [0.3, 0.6, 1.0]),
                       ("noise_obs", [0.01, 0.03, 0.06])):
        rates = []
        for mag in mags:
            held = 0; N = 40
            for _ in range(N):
                s = np.array([0, 0, 0.05, 0.0])
                h = torch.zeros(1, net.n); pre = net.precompute(); a = 0.0
                ok = True; applied = False
                for t in range(500):
                    ob = np.array([[s[0], np.sin(s[2]), np.cos(s[2])]],
                                  np.float32)
                    if mode == "noise_obs":
                        ob = ob + rng.normal(0, mag, ob.shape).astype(np.float32)
                    x = torch.from_numpy((ob - o_mu) / o_sd)
                    h = net.cell(h, net.input_drive(x), pre)
                    cmd = float(np.clip(net.read(h).item() * a_sd + a_mu,
                                        -F_MAX, F_MAX))
                    s = rk4(s, a, DT); a = cmd
                    if t == 100 and not applied:
                        applied = True
                        if mode == "kick_thd":
                            s[3] += mag * (1 if rng.random() < .5 else -1)
                        elif mode == "shift_x":
                            s[0] += mag * (1 if rng.random() < .5 else -1)
                    if abs(s[0]) > 3.0 or (t > 150 and
                            abs(wrap(s[2])) > np.radians(45)):
                        ok = False; break
                held += ok
            rates.append(held / N)
        out[mode] = dict(zip(map(str, mags), rates))
    return None, None, out, {"ref": job["ref"]}


def run_job(job, out_dir="results"):
    t0 = time.time()
    runner = {"bc": run_bc, "matched": run_matched,
              "conv448": run_matched, "robust": run_robust}[job["type"]]
    net, norm, m, extra = runner(job)
    res = {"job": job, "metrics": m, **extra,
           "manifest": manifest(job, net,
                                {"minutes": round((time.time() - t0)/60, 2)})}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/{job['id']}.json", "w"), indent=1)
    if net is not None:
        torch.save({"state": net.state_dict(), "norm": norm, "job": job},
                   f"{out_dir}/{job['id']}.pt")
    else:
        torch.save({"job": job}, f"{out_dir}/{job['id']}.pt")
    return res


def main():
    job = json.load(open(sys.argv[1]))
    res = run_job(job)
    print(f"[{job['id']}] ok in {res['manifest']['minutes']} min "
          f"on {res['manifest']['machine']}")


if __name__ == "__main__":
    main()
