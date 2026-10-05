// Parity: TypeScript plant + controllers must replay the Python fixtures
// within the stated tolerance (abs state error <= 1e-5 per component).
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { DelaySim } from "../src/delay";
import { Classical } from "../src/controllers/classical";
import { Worm, WormModel } from "../src/controllers/worm";
import { Plant, State } from "../src/physics";

const fx = (name: string) =>
  JSON.parse(readFileSync(new URL(`../../shared/fixtures/${name}`,
                                  import.meta.url), "utf8"));
const model = JSON.parse(readFileSync(
  new URL("../public/models/worm_v4.json", import.meta.url),
  "utf8")) as WormModel;

function maxErr(a: State, b: number[]): number {
  return Math.max(...a.map((v, i) => Math.abs(v - (b[i] as number))));
}

describe("plant + worm parity", () => {
  it("replays the worm fixture within tolerance", () => {
    const f = fx("worm_fixture.json");
    const plant = f.plant as Plant;
    const sim = new DelaySim(plant, f.s0 as State, plant.delay_ms);
    const worm = new Worm(model);
    let worst = 0;
    for (let k = 0; k < f.ticks; k++) {
      const s = sim.s;
      const u = worm.force([s[0], Math.sin(s[2]), Math.cos(s[2])]);
      expect(Math.abs(u - f.forces[k])).toBeLessThan(1e-4); // force, N
      sim.tick(u);
      worst = Math.max(worst, maxErr(sim.s, f.states[k]));
      expect(worst).toBeLessThan(f.tolerance_abs * 10 + 1e-5);
    }
    expect(worst).toBeLessThan(f.tolerance_abs);
  });
});

describe("plant + classical parity", () => {
  it("replays the classical fixture within tolerance", () => {
    const f = fx("classical_fixture.json");
    const plant = f.plant as Plant;
    const sim = new DelaySim(plant, f.s0 as State, plant.delay_ms);
    const ctl = new Classical(plant, {
      K: f.lqr_K as [number, number, number, number],
      energyTarget: f.energy_target as number,
    });
    let worst = 0;
    for (let k = 0; k < f.ticks; k++) {
      const u = ctl.force(sim);
      sim.tick(u);
      worst = Math.max(worst, maxErr(sim.s, f.states[k]));
    }
    expect(worst).toBeLessThan(f.tolerance_abs);
  });
});
