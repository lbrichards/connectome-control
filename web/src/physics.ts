// Cart-pole physics — classic Barto 4/3 dynamics, RK4.
// ALL constants come from the model JSON's plant block (single source of
// truth); nothing numeric is hardcoded here.

export interface Plant {
  dynamics: string;        // "barto43"
  M_cart: number;
  m_pole: number;
  half_length: number;
  g: number;
  friction: number;        // 0 for this plant
  f_max: number;
  track: number;
  ctrl_hz: number;
  substeps: number;
  delay_ms: number;
}

export type State = [number, number, number, number]; // x, xd, phi, phid
// phi convention: 0 = upright, pi = hanging (training convention).

export const wrap = (a: number): number =>
  Math.atan2(Math.sin(a), Math.cos(a));

export function makePhysics(p: Plant) {
  if (p.dynamics !== "barto43") throw new Error(`unknown dynamics ${p.dynamics}`);
  const { M_cart: M, m_pole: m, half_length: L, g } = p;
  const tot = M + m;

  function deriv(s: State, F: number): State {
    const [, xd, th, td] = s;
    const ct = Math.cos(th), st = Math.sin(th);
    const tmp = (F + m * L * td * td * st) / tot;
    const ta = (g * st - ct * tmp) / (L * (4 / 3 - (m * ct * ct) / tot));
    const xdd = tmp - (m * L * ta * ct) / tot;
    return [xd, xdd, td, ta];
  }

  function rk4(s: State, F: number, h: number): State {
    const add = (a: State, k: State, c: number): State =>
      [a[0] + c * k[0], a[1] + c * k[1], a[2] + c * k[2], a[3] + c * k[3]];
    const k1 = deriv(s, F);
    const k2 = deriv(add(s, k1, h / 2), F);
    const k3 = deriv(add(s, k2, h / 2), F);
    const k4 = deriv(add(s, k3, h), F);
    return [
      s[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
      s[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
      s[2] + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
      s[3] + (h / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
    ];
  }

  function energy(th: number, td: number): number {
    const J = (4 / 3) * m * L * L;
    return 0.5 * J * td * td + m * g * L * (Math.cos(th) - 1);
  }

  return { deriv, rk4, energy };
}
