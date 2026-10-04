"""Job runner with manifests.

A job is a JSON dict: {"id", "kind", "tseed", "steps", "n_demos", "seq",
"eval_ep", "eval_steps"}. Running one produces results/<id>.json with the
metrics plus a full manifest (graph spec, seed, budget, protocol version,
git hash, machine, library versions, weights hash) and, alongside it,
results/<id>.pt (the trained weights).

Deterministic by construction: CPU, single thread, all RNGs seeded from
the job. The same job on two machines must produce identical weight
hashes — that is the acceptance check.
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
from .protocol import (obs_swing, hang_starts, rollout, metrics, UP_TOL)
from .teachers import teacher_v4

torch.set_num_threads(1)


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))).decode().strip()
    except Exception:
        return "unknown"


def collect_demos(n_ep, seq, seed, noise=1.0):
    """Delayed v4-teacher demonstrations (swing-up task, hanging starts)."""
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


def train_bc(O, A, net, steps, bs=12, lr=5e-3, seed=0):
    d = O.shape[-1]
    norm = (O.reshape(-1, d).mean(0), O.reshape(-1, d).std(0) + 1e-6,
            A.mean(), A.std() + 1e-6)
    o_mu, o_sd, a_mu, a_sd = norm
    xs = torch.from_numpy((O - o_mu) / o_sd)
    ys = torch.from_numpy((A - a_mu) / a_sd)
    wt = torch.from_numpy((1.0 + 7.0 * (O[..., 2] > 0.955)).astype(np.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed + 11)
    curve = []
    for st in range(1, steps + 1):
        i = torch.randint(0, len(xs), (bs,), generator=g)
        err = (net(xs[i]).squeeze(-1) - ys[i]) ** 2 * wt[i]
        loss = err.sum() / wt[i].sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        if st % 100 == 0:
            curve.append(round(loss.item(), 5))
    return norm, curve


def weights_hash(net):
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def run_job(job, out_dir="results"):
    t0 = time.time()
    torch.manual_seed(job["tseed"])
    chem, gap, in_idx, n = build_graph(job["kind"])
    net = ConnectomeRNN(chem, gap, np.asarray(in_idx, np.int64),
                        np.arange(n, dtype=np.int64), n_in=3, n_out=1,
                        seed=job["tseed"], spectral_radius=1.3, in_gain=2.0,
                        dt_over_tau_init=0.2)
    O, A = collect_demos(job["n_demos"], job["seq"], job["tseed"])
    norm, curve = train_bc(O, A, net, job["steps"], seed=job["tseed"])
    TH, alive, exit_t = rollout(
        net, norm, hang_starts(job["eval_ep"], np.random.default_rng(999)),
        job["eval_steps"])
    m = metrics(TH, alive, exit_t)
    res = {
        "job": job,
        "metrics": m,
        "final_mse": curve[-1] if curve else None,
        "curve": curve,
        "manifest": {
            "protocol": PROTOCOL_VERSION,
            "git": _git_hash(),
            "machine": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "weights_sha256": weights_hash(net),
            "demos": int(len(O)),
            "minutes": round((time.time() - t0) / 60, 2),
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{job['id']}.json", "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state": net.state_dict(), "norm": norm, "job": job},
               f"{out_dir}/{job['id']}.pt")
    return res


def main():
    job = json.load(open(sys.argv[1]))
    res = run_job(job)
    m = res["metrics"]
    print(f"[{job['id']}] held {m['held']*100:.0f}% off {m['off_track']*100:.0f}% "
          f"mse {res['final_mse']} sha {res['manifest']['weights_sha256'][:12]} "
          f"({res['manifest']['minutes']} min on "
          f"{res['manifest']['machine']})")


if __name__ == "__main__":
    main()
