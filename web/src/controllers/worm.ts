// Worm connectome controller: leaky-integrator cell over the C. elegans
// mask. Protocol v4: inputs are positions only (x, sin th, cos th);
// no velocities, no action inputs. Recurrent state persists across ticks.
// Verified in Python/JS cross-checks to ~1e-6 per step.
import { Plant } from "../physics";

export interface WormModel {
  format: string;
  protocol: string;
  label: string;
  plant: Plant;
  n: number;
  chem: [number, number, number][];   // [post, pre, w]
  gap: [number, number, number][];
  leak: number[];
  decay: number[];
  bias: number[];
  w_in: number[][];                   // [n_sensory][3]
  in_idx: number[];
  read_w: number[];
  read_b: number;
  o_mu: number[]; o_sd: number[];
  a_mu: number; a_sd: number;
}

export class Worm {
  private h: Float64Array;
  private act: Float64Array;
  private acc: Float64Array;
  constructor(public m: WormModel) {
    this.h = new Float64Array(m.n);
    this.act = new Float64Array(m.n);
    this.acc = new Float64Array(m.n);
  }
  reset(): void { this.h.fill(0); }

  /** one 50 Hz step: obs = [x, sin th, cos th] -> force command (N) */
  force(obs: [number, number, number]): number {
    const m = this.m, h = this.h, act = this.act, acc = this.acc;
    for (let i = 0; i < m.n; i++) {
      act[i] = Math.tanh(h[i]!);
      acc[i] = m.bias[i]!;
    }
    for (const [i, j, w] of m.chem) acc[i]! += w * act[j]!;
    for (const [i, j, w] of m.gap) acc[i]! += w * h[j]!;
    for (let k = 0; k < m.in_idx.length; k++) {
      let dr = 0;
      const row = m.w_in[k]!;
      for (let c = 0; c < 3; c++)
        dr += row[c]! * ((obs[c]! - m.o_mu[c]!) / m.o_sd[c]!);
      acc[m.in_idx[k]!]! += dr;
    }
    let out = m.read_b;
    for (let i = 0; i < m.n; i++) {
      h[i] = h[i]! * m.decay[i]! + (acc[i]! / m.leak[i]!) * (1 - m.decay[i]!);
      out += m.read_w[i]! * Math.tanh(h[i]!);
    }
    const u = out * m.a_sd + m.a_mu;
    const fm = m.plant.f_max;
    return Math.max(-fm, Math.min(fm, u));
  }
}

export async function loadWorm(url: string): Promise<Worm> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model load failed: ${r.status}`);
  const m = (await r.json()) as WormModel;
  if (m.format !== "connectome-control-web-1")
    throw new Error(`unknown model format ${m.format}`);
  return new Worm(m);
}
