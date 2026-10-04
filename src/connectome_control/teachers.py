"""Analytic teachers. teacher_v4 is THE v4 label source.

All teachers are delay-aware (protocol v4): they predict one exact RK4 step
under the pending action, then apply the undelayed law — the optimal
structure for pure input delay.

v4 teacher gates (must both pass before DAgger use — verified):
  hanging starts   : held 100%, off-track 0%
  off-centre starts: held 100%, off-track 0%
    (|x|<=2.5 m, theta free, theta_dot +/-4, x_dot +/-1.5)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from . import DT, F_MAX
from .physics import (G, M_CART, M_POLE, LENGTH, deriv, rk4, wrap, energy)


def _linearize(eps=1e-6):
    s0 = np.zeros(4)
    a = np.zeros((4, 4))
    for j in range(4):
        d = np.zeros(4); d[j] = eps
        a[:, j] = (deriv(s0 + d, 0.0) - deriv(s0 - d, 0.0)) / (2 * eps)
    b = ((deriv(s0, eps) - deriv(s0, -eps)) / (2 * eps)).reshape(4, 1)
    return a, b


def lqr_gain(q=None, r=0.1):
    a, b = _linearize()
    if q is None:
        q = np.diag([1.0, 1.0, 10.0, 1.0])
    rr = np.array([[r]])
    p = solve_continuous_are(a, b, q, rr)
    return (np.linalg.inv(rr) @ b.T @ p).ravel()


_K = lqr_gain()

# v4 pumping parameters (gate-fixed): stiffer centering + wall-blend
_E_TARGET = 0.06
_K_ENERGY = 28.0
_K_X, _K_XD = 3.0, 4.0
_WALL, _WALL_GAIN, _WALL_DAMP = 1.5, 16.0, 2.0
_CATCH_ANGLE, _CATCH_RATE = 0.30, 2.2


def _pump(sp):
    e_err = _E_TARGET - energy(sp[2], sp[3])
    d = sp[3] * np.cos(sp[2])
    s_dir = 1.0 if d >= 0 else -1.0
    a_cmd = -_K_ENERGY * e_err * s_dir
    u = a_cmd * (M_CART + M_POLE) - _K_X * sp[0] - _K_XD * sp[1]
    if abs(sp[0]) > _WALL:
        u = u - _WALL_GAIN * (sp[0] - np.sign(sp[0]) * _WALL) - _WALL_DAMP * sp[1]
    return u


def teacher_v4(s, a):
    """Swing-up + catch, delay-aware, off-centre-safe. a = pending action."""
    sp = rk4(s, a, DT)
    w = wrap(sp[2])
    if abs(w) < _CATCH_ANGLE and abs(sp[3]) < _CATCH_RATE:
        u = -(_K[0] * sp[0] + _K[1] * sp[1] + _K[2] * w + _K[3] * sp[3])
    else:
        u = _pump(sp)
    return float(np.clip(u, -F_MAX, F_MAX))


def lqr_catch(s, a):
    """Pure predictive clipped-LQR (catch/balance demonstrations)."""
    sp = rk4(s, a, DT)
    w = wrap(sp[2])
    u = -(_K[0] * sp[0] + _K[1] * sp[1] + _K[2] * w + _K[3] * sp[3])
    return float(np.clip(u, -F_MAX, F_MAX))
