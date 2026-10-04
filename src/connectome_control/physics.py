"""Cart-pole physics (RK4) — identical constants to the prototype."""

from __future__ import annotations

import numpy as np

G, M_CART, M_POLE, LENGTH = 9.8, 1.0, 0.1, 0.5
J_POLE = (4.0 / 3.0) * M_POLE * LENGTH ** 2


def wrap(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


def energy(th, th_dot):
    """Pendulum energy relative to rest-upright (<= 0 below)."""
    return 0.5 * J_POLE * th_dot ** 2 + M_POLE * G * LENGTH * (np.cos(th) - 1.0)


def deriv(s, u):
    _, xd, th, td = s
    ct, st = np.cos(th), np.sin(th)
    tot = M_CART + M_POLE
    tmp = (u + M_POLE * LENGTH * td * td * st) / tot
    ta = (G * st - ct * tmp) / (LENGTH * (4.0 / 3.0 - M_POLE * ct * ct / tot))
    return np.array([xd, tmp - M_POLE * LENGTH * ta * ct / tot, td, ta])


def rk4(s, u, dt):
    k1 = deriv(s, u)
    k2 = deriv(s + 0.5 * dt * k1, u)
    k3 = deriv(s + 0.5 * dt * k2, u)
    k4 = deriv(s + dt * k3, u)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _deriv_batch(s, u):
    xd, th, td = s[:, 1], s[:, 2], s[:, 3]
    ct, st = np.cos(th), np.sin(th)
    tot = M_CART + M_POLE
    tmp = (u + M_POLE * LENGTH * td * td * st) / tot
    ta = (G * st - ct * tmp) / (LENGTH * (4.0 / 3.0 - M_POLE * ct * ct / tot))
    return np.stack([xd, tmp - M_POLE * LENGTH * ta * ct / tot, td, ta], 1)


def rk4_batch(s, u, dt):
    k1 = _deriv_batch(s, u)
    k2 = _deriv_batch(s + 0.5 * dt * k1, u)
    k3 = _deriv_batch(s + 0.5 * dt * k2, u)
    k4 = _deriv_batch(s + dt * k3, u)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
