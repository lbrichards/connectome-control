// App: faithful port of reference-mockup v1.1 onto the engine modules.
// Plant + controller parameters come from the model JSON (single source
// of truth) — nothing numeric about the plant is hardcoded here.
import { Plant, State, wrap } from "./physics";
import { DelaySim } from "./delay";
import { Classical } from "./controllers/classical";
import { Worm, WormModel } from "./controllers/worm";

const PUSH = 1.6, HIST = 500, OUT_GAP = 30, UP_TOL = 0.21;

type Mode = "classical" | "worm" | "dense";
interface HistPoint { x: number; xd: number; th: number; thd: number; u: number }

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const cv = $<HTMLCanvasElement>("c"), ctx = cv.getContext("2d")!;
const lv = $<HTMLCanvasElement>("lanes"), lctx = lv.getContext("2d")!;
const timerEl = $("timer"), msgEl = $("msg"), panelSub = $("panelSub");
const modelTag = $("modelTag"), pauseBtn = $("pause");
const delayInput = $<HTMLInputElement>("delay"), delayOut = $("delayOut");

interface Lane {
  name: string; unit: string; get: (h: HistPoint) => number;
  lo: number; hi: number; wraps?: boolean; out?: boolean;
}
const LANES: Record<Mode, Lane[]> = {
  classical: [
    { name: "x", unit: "m", get: (h) => h.x, lo: -3, hi: 3 },
    { name: "ẋ", unit: "m/s", get: (h) => h.xd, lo: -6, hi: 6 },
    { name: "θ", unit: "rad", get: (h) => h.th, lo: -Math.PI, hi: Math.PI, wraps: true },
    { name: "θ̇", unit: "rad/s", get: (h) => h.thd, lo: -12, hi: 12 },
    { name: "force", unit: "N", get: (h) => h.u, lo: -18, hi: 18, out: true },
  ],
  worm: [
    { name: "x", unit: "m", get: (h) => h.x, lo: -3, hi: 3 },
    { name: "sin θ", unit: "", get: (h) => Math.sin(h.th), lo: -1, hi: 1 },
    { name: "cos θ", unit: "", get: (h) => Math.cos(h.th), lo: -1, hi: 1 },
    { name: "force", unit: "N", get: (h) => h.u, lo: -18, hi: 18, out: true },
  ],
  dense: [
    { name: "x", unit: "m", get: (h) => h.x, lo: -3, hi: 3 },
    { name: "sin θ", unit: "", get: (h) => Math.sin(h.th), lo: -1, hi: 1 },
    { name: "cos θ", unit: "", get: (h) => Math.cos(h.th), lo: -1, hi: 1 },
    { name: "force", unit: "N", get: (h) => h.u, lo: -18, hi: 18, out: true },
  ],
};
const SUBS: Record<Mode, string> = {
  classical: "Full state, θ = 0 upright, plus the actions it has already committed during the delay.",
  worm: "Positions only, θ = 0 upright. No velocities and no record of its own actions: the network infers what it needs from its own memory.",
  dense: "Positions only, θ = 0 upright — the same inputs as the worm, at a fixed 20 ms delay.",
};

let plant: Plant;
let worm: Worm | null = null;
let dense: Worm | null = null;
let classical: Classical;
let mode: Mode = "classical";
let sim: DelaySim;
let upright = 0, paused = false, acc = 0, last: number | null = null;
let pushMark: { dir: number; t: number } | null = null;
let hist: HistPoint[] = [];
let classicalDelayMs = 20;

function hangStart(): State {
  // Sample the TRAINING start distribution. A fixed symmetric start is
  // out of distribution for the worm (measured: it never catches from
  // [0,0,pi+0.04,0], median 4.7 s from this distribution).
  const u = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
  return [u(-0.15, 0.15), 0, Math.PI + u(-0.15, 0.15), u(-0.05, 0.05)];
}
function currentDelay(): number {
  return mode === "classical" ? classicalDelayMs : plant.delay_ms;
}
function reset(): void {
  const keep = sim ? ([...sim.s] as State) : hangStart();
  sim = new DelaySim(plant, hangStart(), currentDelay());
  void keep;
  classical.reset();
  worm?.reset();
  upright = 0; acc = 0; pushMark = null; hist = [];
}
function setMode(m: Mode): void {
  mode = m;
  $("modeClassical").setAttribute("aria-pressed", String(m === "classical"));
  $("modeWorm").setAttribute("aria-pressed", String(m === "worm"));
  $("modeDense").setAttribute("aria-pressed", String(m === "dense"));
  $("delayCtl").hidden = m !== "classical";
  $("delayFixed").hidden = m === "classical";
  modelTag.hidden = m !== "worm";
  $("denseTag").hidden = m !== "dense";
  $("classicalNote").hidden = m !== "classical";
  // physical state carries over; controller memory does not
  const s = [...sim.s] as State;
  sim = new DelaySim(plant, s, currentDelay());
  classical.reset();
  worm?.reset();
  dense?.reset();
  panelSub.textContent = SUBS[m];
  layout();
}
function push(dir: number): void {
  if (paused) return;
  sim.s[3] += dir * PUSH * (Math.cos(sim.s[2]) >= 0 ? 1 : -1);
  pushMark = { dir, t: sim.t };
}

function tick(): void {
  let u = 0;
  if (mode === "classical") u = classical.force(sim);
  else {
    const net = mode === "worm" ? worm : dense;
    if (net) {
      const s = sim.s;
      u = net.force([s[0], Math.sin(s[2]), Math.cos(s[2])]);
    }
  }
  sim.tick(u);
  upright = Math.abs(wrap(sim.s[2])) < UP_TOL ? upright + 1 / plant.ctrl_hz : 0;
  const s = sim.s;
  hist.push({ x: s[0], xd: s[1], th: wrap(s[2]), thd: s[3], u: sim.applied });
  if (hist.length > HIST) hist.shift();
}

function layout(): void {
  const dpr = window.devicePixelRatio || 1;
  const rows = LANES[mode].length;
  lv.style.height = `${rows * 26 + 8 + OUT_GAP}px`;
  for (const c of [cv, lv]) {
    const r = c.getBoundingClientRect();
    c.width = Math.round(r.width * dpr);
    c.height = Math.round(r.height * dpr);
    c.getContext("2d")!.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}
function colors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n: string) => cs.getPropertyValue(n).trim();
  return { ink: g("--ink"), mid: g("--mid"), rail: g("--rail"),
           faint: g("--faint"), accent: g("--accent") };
}

function drawStage(C: ReturnType<typeof colors>): void {
  const W = cv.clientWidth, Hh = cv.clientHeight;
  ctx.clearRect(0, 0, W, Hh);
  const track = plant.track;
  const poleLen = 2 * plant.half_length;   // half_length is pivot-to-CoM;
  const span = 2 * track + 0.7;            // the visible rod is twice that
  const sc = Math.min((W * 0.92) / span, (Hh * 0.82) / (2 * poleLen + 0.5));
  const cx = W / 2, railY = Hh * 0.5 + 0.12 * sc;
  const X = (v: number) => cx + v * sc;
  const s = sim.s;
  ctx.strokeStyle = C.rail; ctx.lineWidth = 2; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(X(-track - 0.25), railY);
  ctx.lineTo(X(track + 0.25), railY); ctx.stroke();
  for (const e of [-1, 1]) {
    const ex = X(e * (track + 0.25));
    ctx.beginPath(); ctx.moveTo(ex, railY - 10); ctx.lineTo(ex, railY + 10);
    ctx.stroke();
  }
  const cw = 0.5 * sc, ch = 0.22 * sc, px = X(s[0]), py = railY - ch;
  ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
  ctx.strokeRect(px - cw / 2, py, cw, ch);
  // phi: 0 = upright -> tip above pivot at phi = 0
  const tx = px + poleLen * Math.sin(s[2]) * sc, ty = py - poleLen * Math.cos(s[2]) * sc;
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tx, ty); ctx.stroke();
  ctx.fillStyle = Math.abs(wrap(s[2])) < UP_TOL ? C.accent : C.ink;
  ctx.beginPath(); ctx.arc(tx, ty, Math.max(5, 0.045 * sc), 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = C.ink;
  ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI); ctx.fill();
  if (pushMark && sim.t - pushMark.t < 0.25) {
    const d = pushMark.dir, gx = tx - d * 26;
    ctx.strokeStyle = C.mid; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(gx - d * 8, ty - 8); ctx.lineTo(gx, ty);
    ctx.lineTo(gx - d * 8, ty + 8); ctx.stroke();
  }
}

function drawLanes(C: ReturnType<typeof colors>): void {
  const W = lv.clientWidth, lanes = LANES[mode], rowH = 26, labelW = 150;
  lctx.clearRect(0, 0, W, lv.clientHeight);
  lctx.font = '13px system-ui,-apple-system,"Segoe UI",sans-serif';
  lctx.textBaseline = "middle";
  const cur = hist[hist.length - 1];
  lanes.forEach((ln, i) => {
    const y0 = 4 + i * rowH + (ln.out ? OUT_GAP : 0);
    const mid = y0 + rowH / 2, plotX = labelW, plotW = W - labelW - 4;
    if (ln.out) {           // the output gets its own heading: never an input
      lctx.strokeStyle = C.faint; lctx.lineWidth = 1;
      lctx.beginPath(); lctx.moveTo(0, y0 - OUT_GAP + 4);
      lctx.lineTo(W, y0 - OUT_GAP + 4); lctx.stroke();
      lctx.fillStyle = C.ink;
      lctx.font = '600 13px system-ui,-apple-system,"Segoe UI",sans-serif';
      lctx.fillText("What it does", 0, y0 - OUT_GAP / 2 + 4);
      lctx.font = '13px system-ui,-apple-system,"Segoe UI",sans-serif';
    }
    lctx.fillStyle = C.ink; lctx.fillText(ln.name, 0, mid);
    const v = cur ? ln.get(cur) : 0;
    const txt = (v >= 0 ? " " : "−") + Math.abs(v).toFixed(2) +
      (ln.unit ? " " + ln.unit : "");
    lctx.fillStyle = C.mid;
    lctx.fillText(txt, 52, mid);
    const yOf = (val: number) =>
      y0 + rowH - 4 -
      ((Math.max(ln.lo, Math.min(ln.hi, val)) - ln.lo) / (ln.hi - ln.lo)) *
      (rowH - 8);
    lctx.strokeStyle = C.faint; lctx.lineWidth = 1;
    lctx.beginPath(); lctx.moveTo(plotX, yOf(0));
    lctx.lineTo(plotX + plotW, yOf(0)); lctx.stroke();
    if (hist.length < 2) return;
    lctx.strokeStyle = ln.out ? C.mid : C.ink; lctx.lineWidth = 1.5;
    lctx.beginPath();
    let prev: number | null = null;
    hist.forEach((hp, k) => {
      const xx = plotX + (plotW * (k + HIST - hist.length)) / (HIST - 1);
      const val = ln.get(hp);
      if (prev === null || (ln.wraps && Math.abs(val - prev) > Math.PI))
        lctx.moveTo(xx, yOf(val));
      else lctx.lineTo(xx, yOf(val));
      prev = val;
    });
    lctx.stroke();
  });
}

function draw(): void {
  const C = colors();
  drawStage(C); drawLanes(C);
  timerEl.textContent = `Upright ${upright.toFixed(1)} s`;
  msgEl.textContent = sim.t - sim.endStopFlash < 1.2 ? "End stop" : "";
}

function frame(ts: number): void {
  if (last === null) last = ts;
  const dt = Math.min(0.1, (ts - last) / 1000);
  last = ts;
  const ctrlDt = 1 / plant.ctrl_hz;
  if (!paused) {
    acc += dt;
    while (acc >= ctrlDt) { tick(); acc -= ctrlDt; }
  }
  draw();
  requestAnimationFrame(frame);
}

async function boot(): Promise<void> {
  const r = await fetch("/models/worm_v4.json");
  const model = (await r.json()) as WormModel &
    { classical: { K: [number, number, number, number]; energy_target: number } };
  plant = model.plant;
  worm = new Worm(model);
  classical = new Classical(plant, {
    K: model.classical.K, energyTarget: model.classical.energy_target,
  });
  modelTag.textContent = `Model: ${model.label}`;
  try {
    const rd = await fetch("/models/dense_v4.json");
    if (rd.ok) {
      const dm = (await rd.json()) as WormModel;
      dense = new Worm(dm);
      $("denseTag").textContent =
        `A conventional dense network with the same number of connections (about 6,000). Model: ${dm.label}`;
    } else {
      $("modeDense").hidden = true;
    }
  } catch {
    $("modeDense").hidden = true;
  }
  classicalDelayMs = plant.delay_ms;
  delayInput.value = String(plant.delay_ms);
  delayOut.textContent = `${plant.delay_ms} ms`;
  ($("delayFixed")).textContent = `Delay ${plant.delay_ms} ms, as trained`;
  sim = new DelaySim(plant, hangStart(), plant.delay_ms);
  panelSub.textContent = SUBS[mode];
  $("classicalNote").hidden = mode !== "classical";
  reset();
  layout();
  requestAnimationFrame(frame);
}

// ---- events
$("modeClassical").onclick = () => setMode("classical");
$("modeWorm").onclick = () => setMode("worm");
$("modeDense").onclick = () => setMode("dense");
$("reset").onclick = () => reset();
pauseBtn.onclick = () => {
  paused = !paused;
  pauseBtn.textContent = paused ? "Run" : "Pause";
};
delayInput.oninput = () => {
  classicalDelayMs = Number(delayInput.value);
  delayOut.textContent = `${classicalDelayMs} ms`;
  if (mode === "classical") {
    const s = [...sim.s] as State;
    sim = new DelaySim(plant, s, classicalDelayMs);
  }
};
addEventListener("keydown", (e) => {
  if (document.activeElement === delayInput) return;  // slider keeps arrows
  if (e.key === "ArrowLeft") { push(-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { push(1); e.preventDefault(); }
  else if (e.key === "r" || e.key === "R") reset();
  else if (e.key === " ") { pauseBtn.click(); e.preventDefault(); }
});
$("stage").addEventListener("pointerdown", (e) => {
  const r = ($("stage")).getBoundingClientRect();
  push(e.clientX - r.left > r.width / 2 ? 1 : -1);
});
addEventListener("resize", layout);

boot();
