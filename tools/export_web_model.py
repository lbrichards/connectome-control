"""Export the web demo's model JSON + parity fixtures.

Single source of truth: the JSON carries the PLANT constants alongside the
network; the web physics must read them from here, never hardcode.

Model selection: median-by-held v4 worm from the overnight Tier-1 results
(falls back to --pt for an explicit checkpoint).

Outputs:
  web/public/models/worm_v4.json      (plant + network + norm + meta)
  shared/fixtures/worm_fixture.json    (500-tick closed-loop trajectory)
  shared/fixtures/classical_fixture.json
Fixture tolerance (documented for the TS test): max abs state error 1e-5.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import connectome_control as cc
from connectome_control.graph import build_graph
from connectome_control.model import ConnectomeRNN
from connectome_control.physics import (G, M_CART, M_POLE, LENGTH, rk4, wrap)
from web_classical import DemoSim, H, SUBSTEPS, CTRL_DT, K_D, E_TARGET

Q = os.path.expanduser("~/cc-queue")
REPO = os.path.join(os.path.dirname(__file__), "..")

PLANT = {
    "dynamics": "barto43",
    "M_cart": M_CART, "m_pole": M_POLE, "half_length": LENGTH, "g": G,
    "friction": 0.0,
    "f_max": cc.F_MAX, "track": cc.TRACK,
    "ctrl_hz": cc.HZ, "substeps": SUBSTEPS, "delay_ms": cc.DELAY_MS,
}


def pick_median_worm():
    rows = []
    for f in glob.glob(f"{Q}/results/t1_worm_*.json"):
        r = json.load(open(f))
        if r["job"]["kind"] == "worm":
            rows.append((r["metrics"]["held"], r["job"]["tseed"],
                         f.replace(".json", ".pt")))
    if not rows:
        return None
    rows.sort()
    return rows[len(rows) // 2]


def export_model(pt_path, held, tseed, label):
    ck = torch.load(pt_path, weights_only=False)
    chem, gap, in_idx, n = build_graph("worm")
    net = ConnectomeRNN(chem, gap, np.asarray(in_idx, np.int64),
                        np.arange(n, dtype=np.int64), n_in=3, n_out=1,
                        seed=tseed, spectral_radius=1.3, in_gain=2.0,
                        dt_over_tau_init=0.2)
    net.load_state_dict(ck["state"])
    norm = ck["norm"]
    with torch.no_grad():
        wc, g, leak, decay = net.precompute()
        wc, g = wc.numpy(), g.numpy()
    r6 = lambda x: round(float(x), 6)
    ii, jj = np.nonzero(wc)
    o_mu, o_sd, a_mu, a_sd = norm
    model = {
        "format": "connectome-control-web-1",
        "protocol": cc.PROTOCOL_VERSION,
        "label": label,
        "plant": PLANT,
        "n": net.n,
        "chem": [[int(i), int(j), r6(wc[i, j])] for i, j in zip(ii, jj)],
        "gap": [[int(i), int(j), r6(g[i, j])]
                for i, j in zip(*np.nonzero(g))],
        "leak": [r6(v) for v in leak],
        "decay": [r6(v) for v in decay],
        "bias": [r6(v) for v in net.bias.detach().numpy()],
        "w_in": [[r6(v) for v in row] for row in net.w_in.detach().numpy()],
        "in_idx": [int(v) for v in net.in_idx.numpy()],
        "read_w": [r6(v) for v in net.readout.weight.detach().numpy().ravel()],
        "read_b": r6(net.readout.bias.item()),
        "o_mu": [float(v) for v in o_mu], "o_sd": [float(v) for v in o_sd],
        "a_mu": float(a_mu), "a_sd": float(a_sd),
    }
    out = f"{REPO}/web/public/models/worm_v4.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(model, open(out, "w"))
    kb = os.path.getsize(out) // 1024
    print(f"model -> web/public/models/worm_v4.json ({kb} KB)  [{label}]")
    return net, norm


def worm_fixture_from_json(model, ticks=500):
    """Closed loop replaying the EXPORTED JSON in float64 -- the fixture
    tests the shipped artifact, matching the TS runtime's arithmetic."""
    n = model["n"]
    h = np.zeros(n)
    bias = np.array(model["bias"]); leak = np.array(model["leak"])
    decay = np.array(model["decay"]); read_w = np.array(model["read_w"])
    o_mu = np.array(model["o_mu"]); o_sd = np.array(model["o_sd"])
    a_mu, a_sd, read_b = model["a_mu"], model["a_sd"], model["read_b"]
    chem, gap = model["chem"], model["gap"]
    w_in = np.array(model["w_in"]); in_idx = np.array(model["in_idx"])
    s = np.array([0.05, 0.0, np.pi + 0.08, 0.0])
    queue, applied, t = [], 0.0, 0.0
    delay = cc.DELAY_MS / 1000.0
    states, forces = [list(map(float, s))], []
    for k in range(ticks):
        act = np.tanh(h)
        acc = bias.copy()
        for (i, j, w) in chem: acc[i] += w * act[j]
        for (i, j, w) in gap: acc[i] += w * h[j]
        ob = np.array([s[0], np.sin(s[2]), np.cos(s[2])])
        drv = w_in @ ((ob - o_mu) / o_sd)
        acc[in_idx] += drv
        h = h * decay + (acc / leak) * (1 - decay)
        out = read_b + float(read_w @ np.tanh(h))
        u = float(np.clip(out * a_sd + a_mu, -cc.F_MAX, cc.F_MAX))
        queue.append((round(t + delay, 6), u))
        forces.append(u)
        for _ in range(SUBSTEPS):
            for (at, uu) in queue:
                if at <= t + 1e-9:
                    applied = uu
            queue = [(at, uu) for (at, uu) in queue if at > t + 1e-9]
            s = rk4(s, applied, H)
            t = round(t + H, 6)
            if abs(s[0]) >= cc.TRACK:
                side = np.sign(s[0])
                s[0] = side * cc.TRACK
                if np.sign(s[1]) == side:
                    s[1] = 0.0
        states.append(list(map(float, s)))
    return {"controller": "worm", "plant": PLANT, "ticks": ticks,
            "s0": states[0], "forces": forces, "states": states[1:],
            "tolerance_abs": 1e-5}


def classical_fixture(ticks=500):
    sim = DemoSim([0.05, 0.0, np.pi + 0.08, 0.0], delay_ms=cc.DELAY_MS)
    states, forces = [list(map(float, sim.s))], []
    for _ in range(ticks):
        u_before = len(sim.queue)
        sim.tick()
        forces.append(float(sim.queue[-1][1]) if len(sim.queue) > u_before - SUBSTEPS
                      else float(sim.applied))
        states.append(list(map(float, sim.s)))
    return {"controller": "classical", "plant": PLANT, "ticks": ticks,
            "s0": states[0], "forces": forces, "states": states[1:],
            "lqr_K": [float(v) for v in K_D],
            "energy_target": float(E_TARGET),
            "tolerance_abs": 1e-5}


DEMO_GRADE_HELD = 0.20     # a median model below this is not demo-grade

def main():
    pick = pick_median_worm()
    if pick is not None and pick[0] >= DEMO_GRADE_HELD:
        held, tseed, pt = pick
        label = f"v4 C. elegans, seed {tseed}, median by held-rate " \
                f"({held*100:.0f}% held)"
    else:
        if pick is not None:
            print(f"v4 median worm holds {pick[0]*100:.0f}% (< "
                  f"{DEMO_GRADE_HELD*100:.0f}% demo-grade bar); "
                  "falling back per SPEC")
        pt = os.path.expanduser(
            "~/projects/Connectome/student_distilled_v3.pt")
        held, tseed, label = 0.45, 3, \
            "prototype (distilled worm, protocol v3, 45% held)"
    net, norm = export_model(pt, held, tseed, label)

    model_json = json.load(open(f"{REPO}/web/public/models/worm_v4.json"))
    fx = worm_fixture_from_json(model_json)
    os.makedirs(f"{REPO}/shared/fixtures", exist_ok=True)
    json.dump(fx, open(f"{REPO}/shared/fixtures/worm_fixture.json", "w"))
    fc = classical_fixture()
    json.dump(fc, open(f"{REPO}/shared/fixtures/classical_fixture.json", "w"))
    up = sum(1 for s in fx["states"] if abs(wrap(s[2])) < np.radians(12))
    print(f"fixtures -> shared/fixtures/ (worm time-up in fixture: "
          f"{up/len(fx['states'])*100:.0f}%)")


if __name__ == "__main__":
    main()
