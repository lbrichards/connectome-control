"""Relay-distillation assets (protocol v4): swing brain, catch brain,
cold-handoff relay, and the shared distillation dataset.

Built ONCE on the coordinator (stage-gated), checksummed, rsynced to
workers. Tier-1 jobs train students on the distillation dataset and may
run one DAgger round labelled by this relay (deterministic artifact).

Stage gates (abort if unmet — prototype v3 values):
  swing reach >= 85%   catch held >= 90%   relay held >= 50%
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

from . import DT, F_MAX
from .graph import build_graph
from .model import ConnectomeRNN
from .physics import rk4_batch, wrap
from .protocol import (obs_swing, hang_starts, arrival_starts, rollout,
                       metrics, UP_TOL, HOLD_N)
from .jobs import collect_demos, train_bc, dagger_round, _build_net
from .teachers import teacher_v4, lqr_catch

GATE = (0.25, 1.8, 1.2, 0.5)
RELAY_DIR = os.path.join(os.path.dirname(__file__), "data", "relay")

obs_catch = lambda s: np.stack([s[:, 0], wrap(s[:, 2])], 1)


def train_swing(seed=0):
    torch.manual_seed(seed)
    net = _build_net("worm", seed)
    O, A = collect_demos(320, 350, seed)
    norm, *_ = train_bc(O, A, net, 5000, seed=seed)
    for rnd in range(5):
        dO, dA = dagger_round(net, norm, 128, 350, 100 + rnd)
        O = np.concatenate([O, dO]); A = np.concatenate([A, dA])
        norm, *_ = train_bc(O, A, net, 2000, seed=seed)
    TH, alive, exit_t = rollout(net, norm,
        hang_starts(100, np.random.default_rng(999)), 1000)
    up = (np.abs(TH) < UP_TOL); up[~np.isfinite(TH)] = False
    reach = float(up.any(0).mean())
    print(f"swing: reach {reach*100:.0f}%", flush=True)
    assert reach >= 0.85, f"SWING GATE FAILED: {reach:.2f}"
    return net, norm


def _collect_catch(n_ep, seq, seed, noise=0.6):
    rng = np.random.default_rng(seed)
    O, A = [], []
    tries = 0
    while len(O) < n_ep and tries < 40 * n_ep:
        tries += 1
        s = arrival_starts(1, rng)[0]
        a = 0.0
        obs, act, traj = [], [], []
        ok = True
        for _ in range(seq):
            obs.append([s[0], wrap(np.array([s[2]]))[0]])
            cmd = lqr_catch(s, a)
            act.append(cmd)
            traj.append(s[2])
            from .physics import rk4
            s = rk4(s, a, DT)
            a = float(np.clip(cmd + noise * rng.normal(), -F_MAX, F_MAX))
            if abs(s[0]) > 3.0:
                ok = False; break
        th = wrap(np.array(traj[-75:]))
        if not ok or not (np.abs(th) < UP_TOL).all():
            continue
        O.append(obs); A.append(act)
    return np.array(O, np.float32), np.array(A, np.float32)


def train_catch(seed=1, steps=2500):
    torch.manual_seed(seed)
    chem, gap, in_idx, n = build_graph("worm")
    net = ConnectomeRNN(chem, gap, np.asarray(in_idx, np.int64),
                        np.arange(n, dtype=np.int64), n_in=2, n_out=1,
                        seed=seed, spectral_radius=1.3, in_gain=2.0,
                        dt_over_tau_init=0.2)
    O, A = _collect_catch(256, 250, seed)
    norm, *_ = train_bc(O, A, net, steps, seed=seed, wt=np.ones_like(A))
    # eval from arrivals (delayed, 2-input obs)
    held = _eval_catch(net, norm)
    print(f"catch({steps}): held from arrivals {held*100:.0f}%", flush=True)
    return net, norm, held


@torch.no_grad()
def _eval_catch(net, norm, n_ep=200):
    o_mu, o_sd, a_mu, a_sd = norm
    s = arrival_starts(n_ep, np.random.default_rng(99))
    h = torch.zeros(n_ep, net.n); pre = net.precompute()
    a = np.zeros(n_ep)
    up = np.zeros((500, n_ep), bool); alive = np.ones(n_ep, bool)
    for t in range(500):
        ob = obs_catch(s).astype(np.float32)
        x = torch.from_numpy((ob - o_mu) / o_sd)
        h = net.cell(h, net.input_drive(x), pre)
        cmd = np.clip(net.read(h).squeeze(-1).numpy() * a_sd + a_mu,
                      -F_MAX, F_MAX)
        s = np.where(alive[:, None], rk4_batch(s, a, DT), s)
        a = cmd
        up[t] = (np.abs(wrap(s[:, 2])) < UP_TOL) & alive
        alive &= np.abs(s[:, 0]) <= 3.0
    return float((alive & up[-HOLD_N:].all(0)).mean())


@torch.no_grad()
def relay_rollout(swi, swi_n, cat, cat_n, starts_arr, steps):
    s = starts_arr.copy(); n = len(s)
    hs = torch.zeros(n, swi.n); hc = torch.zeros(n, cat.n)
    ps, pc = swi.precompute(), cat.precompute()
    modeB = np.zeros(n, bool); a = np.zeros(n)
    up = np.zeros((steps, n), bool); alive = np.ones(n, bool)
    O = np.zeros((steps, n, 3), np.float32)
    A = np.zeros((steps, n), np.float32)
    for t in range(steps):
        w = wrap(s[:, 2])
        enter = (~modeB) & (np.abs(w) < GATE[0]) & (np.abs(s[:, 3]) < GATE[1]) \
                & (np.abs(s[:, 1]) < GATE[2])
        if enter.any():
            hc[torch.from_numpy(enter)] = 0.0
        modeB = (modeB | enter) & (np.abs(w) <= GATE[3])
        m, sd, am, asd = swi_n
        obS = obs_swing(s).astype(np.float32)
        hs.copy_(swi.cell(hs, swi.input_drive(
            torch.from_numpy((obS - m) / sd)), ps))
        uS = np.clip(swi.read(hs).squeeze(-1).numpy() * asd + am,
                     -F_MAX, F_MAX)
        m, sd, am, asd = cat_n
        obC = obs_catch(s).astype(np.float32)
        hc.copy_(cat.cell(hc, cat.input_drive(
            torch.from_numpy((obC - m) / sd)), pc))
        uC = np.clip(cat.read(hc).squeeze(-1).numpy() * asd + am,
                     -F_MAX, F_MAX)
        cmd = np.where(modeB, uC, uS)
        O[t] = obS; A[t] = cmd
        s = np.where(alive[:, None], rk4_batch(s, a, DT), s)
        a = cmd
        up[t] = (np.abs(wrap(s[:, 2])) < UP_TOL) & alive
        alive &= np.abs(s[:, 0]) <= 3.0
    held = alive & up[-HOLD_N:].all(0)
    return held, up, O.transpose(1, 0, 2), A.T


def load_relay(relay_dir=None):
    rd = relay_dir or RELAY_DIR
    swi = _build_net("worm", 0)
    ck = torch.load(f"{rd}/swing.pt", weights_only=False)
    swi.load_state_dict(ck["state"]); swi_n = ck["norm"]
    chem, gap, in_idx, n = build_graph("worm")
    cat = ConnectomeRNN(chem, gap, np.asarray(in_idx, np.int64),
                        np.arange(n, dtype=np.int64), n_in=2, n_out=1,
                        seed=1, spectral_radius=1.3, in_gain=2.0,
                        dt_over_tau_init=0.2)
    ck = torch.load(f"{rd}/catch.pt", weights_only=False)
    cat.load_state_dict(ck["state"]); cat_n = ck["norm"]
    return swi, swi_n, cat, cat_n


def build_assets():
    os.makedirs(RELAY_DIR, exist_ok=True)
    swi, swi_n = train_swing()
    torch.save({"state": swi.state_dict(), "norm": swi_n},
               f"{RELAY_DIR}/swing.pt")
    cat, cat_n, ch = train_catch()
    assert ch >= 0.90, f"CATCH GATE FAILED: {ch:.2f}"
    torch.save({"state": cat.state_dict(), "norm": cat_n},
               f"{RELAY_DIR}/catch.pt")
    held, up, _, _ = relay_rollout(swi, swi_n, cat, cat_n,
        hang_starts(200, np.random.default_rng(999)), 1000)
    print(f"relay: held {held.mean()*100:.0f}%", flush=True)
    assert held.mean() >= 0.50, f"RELAY GATE FAILED: {held.mean():.2f}"
    h1, _, O1, A1 = relay_rollout(swi, swi_n, cat, cat_n,
        hang_starts(300, np.random.default_rng(0)), 600)
    h2, _, O2, A2 = relay_rollout(swi, swi_n, cat, cat_n,
        arrival_starts(200, np.random.default_rng(1)), 600)
    O = np.concatenate([O1[h1], O2[h2]])
    A = np.concatenate([A1[h1], A2[h2]])
    np.savez_compressed(f"{RELAY_DIR}/distill.npz", O=O, A=A)
    print(f"distill dataset: {len(O)} episodes", flush=True)
    sums = []
    for f in ("swing.pt", "catch.pt", "distill.npz"):
        h = hashlib.sha256(open(f"{RELAY_DIR}/{f}", "rb").read()).hexdigest()
        sums.append(f"{h}  {f}")
    open(f"{RELAY_DIR}/SHA256SUMS", "w").write("\n".join(sums) + "\n")
    print("relay assets complete + checksummed", flush=True)


def build_quality_variant(tag, target_lo, target_hi, catch_steps0):
    """Build a relay asset set whose held-rate lands in [lo, hi] by
    adjusting the catch brain's training budget (bisection, <=4 tries).
    Swing brain is shared from the main assets."""
    import shutil
    out = os.path.join(os.path.dirname(RELAY_DIR), f"relay_{tag}")
    os.makedirs(out, exist_ok=True)
    swi = _build_net("worm", 0)
    ck = torch.load(f"{RELAY_DIR}/swing.pt", weights_only=False)
    swi.load_state_dict(ck["state"]); swi_n = ck["norm"]
    shutil.copy(f"{RELAY_DIR}/swing.pt", f"{out}/swing.pt")
    lo_s, hi_s = 150, 3000
    steps = catch_steps0
    best = None
    for attempt in range(4):
        cat, cat_n, _ = train_catch(steps=steps)
        held, up, _, _ = relay_rollout(swi, swi_n, cat, cat_n,
            hang_starts(200, np.random.default_rng(999)), 1000)
        hm = float(held.mean())
        print(f"[{tag}] catch_steps={steps} -> relay held {hm*100:.0f}%",
              flush=True)
        best = (cat, cat_n, hm, steps)
        if target_lo <= hm <= target_hi:
            break
        if hm > target_hi: hi_s = steps; steps = (lo_s + steps) // 2
        else: lo_s = steps; steps = (steps + hi_s) // 2
    cat, cat_n, hm, steps = best
    torch.save({"state": cat.state_dict(), "norm": cat_n,
                "relay_held": hm, "catch_steps": steps}, f"{out}/catch.pt")
    h1, _, O1, A1 = relay_rollout(swi, swi_n, cat, cat_n,
        hang_starts(300, np.random.default_rng(0)), 600)
    h2, _, O2, A2 = relay_rollout(swi, swi_n, cat, cat_n,
        arrival_starts(200, np.random.default_rng(1)), 600)
    O = np.concatenate([O1[h1], O2[h2]]); A = np.concatenate([A1[h1], A2[h2]])
    np.savez_compressed(f"{out}/distill.npz", O=O, A=A)
    sums = []
    for f in ("swing.pt", "catch.pt", "distill.npz"):
        h = hashlib.sha256(open(f"{out}/{f}", "rb").read()).hexdigest()
        sums.append(f"{h}  {f}")
    open(f"{out}/SHA256SUMS", "w").write("\n".join(sums) + "\n")
    print(f"[{tag}] assets done: relay held {hm*100:.0f}%, "
          f"{len(O)} episodes", flush=True)
    return hm


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quality":
        # q91 = the main assets, re-used; build q80 and q67 by calibration
        import shutil
        out91 = os.path.join(os.path.dirname(RELAY_DIR), "relay_q91")
        os.makedirs(out91, exist_ok=True)
        for f in ("swing.pt", "catch.pt", "distill.npz", "SHA256SUMS"):
            shutil.copy(f"{RELAY_DIR}/{f}", f"{out91}/{f}")
        print("[q91] reusing main relay assets", flush=True)
        build_quality_variant("q80", 0.74, 0.86, 1000)
        build_quality_variant("q67", 0.60, 0.73, 450)
    else:
        build_assets()
