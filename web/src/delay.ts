// Delay queue + end-stop stepping, demo-exact (mirrors tools/web_classical.py).
import { makePhysics, Plant, State } from "./physics";

export interface QueuedAction { at: number; u: number }

export class DelaySim {
  s: State;
  t = 0;
  applied = 0;
  queue: QueuedAction[] = [];
  delayS: number;
  readonly h: number;           // physics substep
  readonly substeps: number;
  private phys: ReturnType<typeof makePhysics>;
  private plant: Plant;
  endStopFlash = -10;

  constructor(plant: Plant, s0: State, delayMs?: number) {
    this.plant = plant;
    this.phys = makePhysics(plant);
    this.s = [...s0] as State;
    this.delayS = (delayMs ?? plant.delay_ms) / 1000;
    this.substeps = plant.substeps;
    this.h = 1 / plant.ctrl_hz / plant.substeps;
  }

  /** advance one control period, with `u` entering the delay queue */
  tick(u: number): void {
    this.queue.push({ at: +(this.t + this.delayS).toFixed(6), u });
    for (let i = 0; i < this.substeps; i++) {
      for (const q of this.queue) if (q.at <= this.t + 1e-9) this.applied = q.u;
      this.queue = this.queue.filter((q) => q.at > this.t + 1e-9);
      this.s = this.phys.rk4(this.s, this.applied, this.h);
      this.t = +(this.t + this.h).toFixed(6);
      if (Math.abs(this.s[0]) >= this.plant.track) {   // end stop
        const side = Math.sign(this.s[0]);
        this.s[0] = side * this.plant.track;
        if (Math.sign(this.s[1]) === side) {
          this.s[1] = 0;
          this.endStopFlash = this.t;
        }
      }
    }
  }

  /** predict the state after the full delay, replaying queued actions */
  predictThroughQueue(): State {
    let p = [...this.s] as State;
    let t = this.t;
    let cur = this.applied;
    const n = Math.round(this.delayS / this.h);
    for (let i = 0; i < n; i++) {
      for (const q of this.queue) if (Math.abs(q.at - t) < 1e-9) cur = q.u;
      p = this.phys.rk4(p, cur, this.h);
      t = +(t + this.h).toFixed(6);
    }
    return p;
  }
}
