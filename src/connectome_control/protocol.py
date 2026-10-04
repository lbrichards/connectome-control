"""Protocol v4 as code: observations, delayed rollouts, metrics, gates.

Observations are POSITIONS ONLY (x, sin th, cos th) — no velocities, no
pending-action channels (causal-confusion amendment).

Metrics (co-primary: held and off-track rate):
  held        : upright (|th|<12 deg) for the final HOLD_S seconds, on track
  off_track   : |x| > TRACK at any point (episode terminal)
  pole_fell   : alive at the end but not holding — reported as CENSORED by
                track exits (off-track episodes never get the chance)
  approaches  : entries into the band after >=0.5 s outside
  catchable   : approach with |theta_dot| at entry <= CATCHABLE_THR
  conversion  : approach followed by HOLD_S continuously in-band
"""

from __future__ import annotations

import numpy as np
import torch

from . import DT, F_MAX, TRACK, UP_TOL_DEG, HOLD_S, CATCHABLE_THR
from .physics import rk4_batch, wrap

UP_TOL = np.radians(UP_TOL_DEG)
HOLD_N = int(round(HOLD_S / DT))
OUT_N = int(round(0.5 / DT))


def obs_swing(s):
    return np.stack([s[:, 0], np.sin(s[:, 2]), np.cos(s[:, 2])], 1)


def hang_starts(n, rng):
    return np.stack([[rng.uniform(-.15, .15), 0,
                      np.pi + rng.uniform(-.15, .15),
                      rng.uniform(-.05, .05)] for _ in range(n)])


def arrival_starts(n, rng):
    return np.stack([[rng.uniform(-.8, .8), rng.uniform(-1.5, 1.5),
                      rng.uniform(-.35, .35), rng.uniform(-2.2, 2.2)]
                     for _ in range(n)])


@torch.no_grad()
def rollout(net, norm, starts, steps, label_fn=None):
    """Delayed closed loop (one-step buffer, first tick zero force)."""
    o_mu, o_sd, a_mu, a_sd = norm
    s = starts.copy(); n = len(s)
    h = torch.zeros(n, net.n); pre = net.precompute()
    a = np.zeros(n)
    TH = np.zeros((steps, n)); alive = np.ones(n, bool)
    exit_t = np.full(n, -1)
    O, A = [], []
    for t in range(steps):
        ob = obs_swing(s).astype(np.float32)
        if label_fn is not None:
            O.append(ob); A.append(label_fn(s, a))
        x = torch.from_numpy((ob - o_mu) / o_sd)
        h = net.cell(h, net.input_drive(x), pre)
        cmd = np.clip(net.read(h).squeeze(-1).numpy() * a_sd + a_mu,
                      -F_MAX, F_MAX)
        s = np.where(alive[:, None], rk4_batch(s, a, DT), s)
        a = cmd
        TH[t] = wrap(s[:, 2])
        just = alive & (np.abs(s[:, 0]) > TRACK)
        exit_t[just] = t
        alive &= ~just
    out = (TH, alive, exit_t)
    if label_fn is not None:
        out += (np.stack(O, 1), np.stack(A, 1))
    return out


def metrics(TH, alive, exit_t):
    steps, n = TH.shape
    up = np.abs(TH) < UP_TOL
    up[~np.isfinite(TH)] = False
    held = alive & up[-HOLD_N:].all(0)
    off = exit_t >= 0
    fell = ~held & ~off
    appr = catchable = conv = conv_c = 0
    for e in range(n):
        end = exit_t[e] if exit_t[e] >= 0 else steps
        ib = up[:end, e]
        outside = OUT_N; t = 1
        while t < end:
            if ib[t] and outside >= OUT_N:
                thd = abs(TH[t, e] - TH[t - 1, e]) / DT
                c = thd <= CATCHABLE_THR
                v = t + HOLD_N <= end and ib[t:t + HOLD_N].all()
                appr += 1; catchable += c; conv += v; conv_c += (c and v)
                while t < end and ib[t]:
                    t += 1
                outside = 0
            else:
                outside = outside + 1 if not ib[t] else 0
                t += 1
    return {
        "held": float(held.mean()),
        "off_track": float(off.mean()),
        "pole_fell_censored": float(fell.mean()),
        "approaches": int(appr),
        "frac_catchable": float(catchable / appr) if appr else None,
        "conversion_all": float(conv / appr) if appr else None,
        "conversion_catchable": float(conv_c / catchable) if catchable else None,
    }
