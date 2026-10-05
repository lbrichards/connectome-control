// Classical controller: energy pumping -> predictive LQR catch.
// Mirrors tools/web_classical.py exactly; gains and energy target are
// injected from the fixture/export, not computed here.
import { makePhysics, Plant, State, wrap } from "../physics";
import { DelaySim } from "../delay";

export interface ClassicalParams {
  K: [number, number, number, number];  // dLQR @ control rate
  energyTarget: number;                 // J above rest-upright datum
}

// constants identical to web_classical.py
const K_ENERGY = 28, K_X = 3, K_XD = 4, K_DIR = 20;
const WALL = 1.5, WALL_GAIN = 16, WALL_DAMP = 2;
const CATCH_ANGLE = 0.3, CATCH_RATE = 2.2;
const SWING_IN = 0.6, SWING_RATE = 5, CATCH_OUT = 1.0;

export class Classical {
  mode: "swing" | "catch" = "swing";
  private phys: ReturnType<typeof makePhysics>;
  constructor(private plant: Plant, private prm: ClassicalParams) {
    this.phys = makePhysics(plant);
  }
  reset(): void { this.mode = "swing"; }

  force(sim: DelaySim): number {
    const s = sim.s;
    const th = wrap(s[2]);
    if (this.mode === "swing" && Math.abs(th) < SWING_IN &&
        Math.abs(s[3]) < SWING_RATE) this.mode = "catch";
    else if (this.mode === "catch" && Math.abs(th) > CATCH_OUT)
      this.mode = "swing";
    let F: number;
    const p = sim.predictThroughQueue();
    if (this.mode === "catch") {
      const z: State = [p[0], p[1], wrap(p[2]), p[3]];
      const K = this.prm.K;
      F = -(K[0] * z[0] + K[1] * z[1] + K[2] * z[2] + K[3] * z[3]);
    } else {
      // Pump on the PREDICTED post-delay state with a smooth direction
      // (tanh): deciding from the current theta_dot while the force lands
      // one delay later produces a full-amplitude tick-rate limit cycle
      // near rest. Mirrors web_classical.py exactly.
      const { M_cart: M, m_pole: m } = this.plant;
      const eErr = this.prm.energyTarget - this.phys.energy(p[2], p[3]);
      const d = p[3] * Math.cos(p[2]);
      const sdir = Math.tanh(K_DIR * d);
      const a = -K_ENERGY * eErr * sdir;
      F = a * (M + m) - K_X * p[0] - K_XD * p[1];
      if (Math.abs(p[0]) > WALL)
        F += -WALL_GAIN * (p[0] - Math.sign(p[0]) * WALL) - WALL_DAMP * p[1];
    }
    const fm = this.plant.f_max;
    return Math.max(-fm, Math.min(fm, F));
  }
}
