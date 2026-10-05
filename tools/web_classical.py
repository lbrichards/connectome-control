"""Demo classical controller on the TRAINING plant, verified headlessly.

Semantics mirror the web demo exactly (and the TypeScript port must mirror
this file): 50 Hz control, RK4 at 5 ms substeps, actions enter a delay
queue and apply when due, end stops (cart stops dead; control continues).

Controller: energy pumping (target slightly above upright energy) with
x-centering and wall-blend, then LQR catch; the catch predicts forward
through every queued-but-unapplied action (so it tolerates large delays).
LQR gains are computed for THIS plant, discretised at the 50 Hz control
rate (dlqr via solve_discrete_are on the zoh-discretised linearisation).

Verification (run as a script):
  1. swing-up success from hanging at the default 20 ms delay
  2. 300-sequence random-push stress test -- zero stalls allowed
  3. delay breaking point (largest delay with >=90% success)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from connectome_control.physics import (G, M_CART, M_POLE, LENGTH, J_POLE,
                                        deriv, rk4, wrap, energy)

F_MAX, TRACK = 18.0, 3.0
CTRL_DT, SUBSTEPS = 0.02, 4
H = CTRL_DT / SUBSTEPS                      # 5 ms physics substep
E_UP = 0.0                                  # energy datum: rest upright


def _linearize(eps=1e-6):
    s0 = np.zeros(4)
    A = np.zeros((4, 4))
    for j in range(4):
        d = np.zeros(4); d[j] = eps
        A[:, j] = (deriv(s0 + d, 0.0) - deriv(s0 - d, 0.0)) / (2 * eps)
    B = ((deriv(s0, eps) - deriv(s0, -eps)) / (2 * eps)).reshape(4, 1)
    return A, B


def dlqr_gain(q=(1.0, 1.0, 30.0, 3.0), r=0.1, dt=CTRL_DT):
    A, B = _linearize()
    M = np.zeros((5, 5)); M[:4, :4] = A; M[:4, 4:] = B
    Md = expm(M * dt)
    Ad, Bd = Md[:4, :4], Md[:4, 4:]
    Q, R = np.diag(q), np.array([[r]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return K.ravel()


K_D = dlqr_gain()

# pumping constants (v4 teacher family, rescaled check below)
E_TARGET = 1.06 * (2 * M_POLE * G * LENGTH) - (2 * M_POLE * G * LENGTH)
# energy datum is rest-upright = 0; hanging = -2 m g l. "6% above upright"
# in mockup terms == E_TARGET = +0.06 * (2 m g l) above the upright datum.
K_ENERGY = 28.0
# demo-only chatter fix: smooth pump direction. The research teacher keeps
# hard sign() (protocol v4 is frozen); tanh(K_DIR*thd*cos th) removes the
# full-amplitude force alternation near rest without changing behaviour
# once the swing is moving (tanh saturates by |thd*cos th| ~ 0.15).
K_DIR = 20.0
K_X, K_XD = 3.0, 4.0
WALL, WALL_GAIN, WALL_DAMP = 1.5, 16.0, 2.0
CATCH_ANGLE, CATCH_RATE = 0.30, 2.2


class DemoSim:
    """Plant + delay queue + end stops, demo-exact."""

    def __init__(self, s0, delay_ms=20.0):
        self.s = np.array(s0, float)
        self.delay = delay_ms / 1000.0
        self.queue = []                      # (apply_at, u)
        self.applied = 0.0
        self.t = 0.0
        self.mode = "swing"

    # ---- controller -----------------------------------------------------
    def _predict_through_queue(self):
        p = self.s.copy()
        t = self.t
        cur = self.applied
        n = int(round(self.delay / H))
        q = list(self.queue)
        for _ in range(n):
            for (at, u) in q:
                if abs(at - t) < 1e-9:
                    cur = u
            p = rk4(p, cur, H)
            t = round(t + H, 6)
        return p

    def controller(self):
        s = self.s
        th = wrap(s[2])
        if self.mode == "swing" and abs(th) < 0.6 and abs(s[3]) < 5.0:
            self.mode = "catch"
        elif self.mode == "catch" and abs(th) > 1.0:
            self.mode = "swing"
        p = self._predict_through_queue()
        if self.mode == "catch":
            z = np.array([p[0], p[1], wrap(p[2]), p[3]])
            F = -float(K_D @ z)
        else:
            # pump on the PREDICTED post-delay state (same structure as the
            # catch mode and the v4 research teacher): deciding the pump
            # direction from the current theta_dot while the force lands
            # 20 ms later creates a full-amplitude tick-rate limit cycle
            # near rest (the delayed kick reverses theta_dot every tick).
            e_err = E_TARGET - energy(p[2], p[3])
            d = p[3] * np.cos(p[2])
            s_dir = np.tanh(K_DIR * d)
            a = -K_ENERGY * e_err * s_dir
            F = a * (M_CART + M_POLE) - K_X * p[0] - K_XD * p[1]
            if abs(p[0]) > WALL:
                F += -WALL_GAIN * (p[0] - np.sign(p[0]) * WALL) \
                     - WALL_DAMP * p[1]
        return float(np.clip(F, -F_MAX, F_MAX))

    # ---- one 50 Hz tick --------------------------------------------------
    def tick(self, push=0.0):
        if push:
            self.s[3] += push
        u = self.controller()
        self.queue.append((round(self.t + self.delay, 6), u))
        for _ in range(SUBSTEPS):
            for (at, uu) in self.queue:
                if at <= self.t + 1e-9:
                    self.applied = uu
            self.queue = [(at, uu) for (at, uu) in self.queue
                          if at > self.t + 1e-9]
            self.s = rk4(self.s, self.applied, H)
            self.t = round(self.t + H, 6)
            if abs(self.s[0]) >= TRACK:      # end stop
                side = np.sign(self.s[0])
                self.s[0] = side * TRACK
                if np.sign(self.s[1]) == side:
                    self.s[1] = 0.0
        return self.s


UP = np.radians(12)


def run_episode(delay_ms, pushes=(), seed=0, ticks=1000):
    rng = np.random.default_rng(seed)
    s0 = [rng.uniform(-.15, .15), 0, np.pi + rng.uniform(-.15, .15),
          rng.uniform(-.05, .05)]
    sim = DemoSim(s0, delay_ms)
    push_map = dict(pushes)
    upright_run = 0
    last_push_tick = max([t for t, _ in pushes], default=0)
    ok_after = None
    for k in range(ticks):
        sim.tick(push=push_map.get(k, 0.0))
        if abs(wrap(sim.s[2])) < UP:
            upright_run += 1
        else:
            upright_run = 0
        if k > last_push_tick and upright_run >= 150 and ok_after is None:
            ok_after = k
    return ok_after is not None, upright_run >= 150


def verify():
    print("plant: classic cart-pole (m_pole=0.1, half-length=0.5, g=9.8, "
          "frictionless, 4/3 dynamics); F_max=18, track +/-3")
    print(f"dLQR gains @50 Hz: {np.round(K_D, 3)}")
    print(f"energy target: +{E_TARGET:.3f} J above upright "
          f"(={1 + E_TARGET/(2*M_POLE*G*LENGTH):.2f}x upright rise)")

    ok = 0
    for i in range(200):
        a, b = run_episode(20, seed=i)
        ok += b
    print(f"\n1. swing-up+hold from hanging, 20 ms delay: {ok/2:.0f}% "
          f"({'PASS' if ok >= 196 else 'FAIL'})")

    stalls = 0
    rng = np.random.default_rng(42)
    for i in range(300):
        n_push = rng.integers(3, 8)
        pushes = [(int(rng.integers(100, 700)),
                   float(rng.uniform(0.8, 1.6) * rng.choice([-1, 1])))
                  for _ in range(n_push)]
        recovered, _ = run_episode(20, pushes=tuple(pushes), seed=1000 + i,
                                   ticks=1200)
        stalls += not recovered
    print(f"2. random-push stress, 300 sequences: {stalls} stalls "
          f"({'PASS' if stalls == 0 else 'FAIL'})")

    print("3. delay sweep (success = held at end, 50 eps each):")
    breaking = 0
    for dms in range(0, 205, 10):
        okd = sum(run_episode(dms, seed=5000 + i)[1] for i in range(50))
        marker = ""
        if okd >= 45:
            breaking = dms
        print(f"   {dms:>3} ms: {okd*2}%")
        if okd < 10:
            break
    print(f"   -> breaking point: last delay with >=90% success = "
          f"{breaking} ms")


if __name__ == "__main__":
    verify()
