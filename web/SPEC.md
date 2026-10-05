# Web demo v1: build spec

## Purpose
A public, fully client-side cart-pole demo. Spartan in every detail except the motion.
Two controllers on one page: Classical (analytical) and Worm connectome (trained network).

## Reference
`reference-mockup.html` is the approved mockup. Treat it as the behavioural and visual
spec: layout, copy, interactions, colours, light/dark handling, lanes panel, end stop.
Its physics, delay queue and classical controller are correct and tested; port them to
TypeScript faithfully. Do not restyle.

## Keep exactly as in the mockup
- Header: title, two-option controller switch (Classical / Worm connectome), upright timer.
- Stage: rail with end stops, outlined cart, pole line, tip dot (accent colour when upright),
  push chevron. End stop: cart stops dead, "End stop" flashes, control continues.
- Lanes panel: "What the controller sees" lists the controller's inputs, each with live
  value and a 10 s sparkline. Classical: x, ẋ, θ, θ̇ (θ = 0 upright). Worm: x, sin θ,
  cos θ. A separate "What it does" heading shows the output force. The force must never
  be presented as a worm input: the worm receives no action input (protocol v4,
  causal-confusion fix).
- Footer: Reset, Pause, hint text. Delay slider (0–200 ms, step 5, default 20) shown ONLY
  for Classical; Worm shows "Delay 20 ms, as trained".
- Inputs: arrow keys push the pole tip (fixed impulse); tap left/right half on touch;
  R resets; space pauses. Arrow keys adjust the slider only while it is focused.
- Switching controllers keeps the physical state and resets controller memory.
- 50 Hz control, RK4 physics at 5 ms substeps, actions applied after the delay.

## Worm controller (the new part)
- Load weights from a static JSON file shipped with the site (no backend).
- Implement exactly the network used in the repo: leaky-integrator cell, connectome mask,
  time constants, input map (22 sensory neurons), readout, input normalisation, recurrent
  state persisting across ticks, reset on Reset and on switching to the worm.
- Inputs: x, sin θ, cos θ only (protocol v4: positions only, no action input).
- Fixed 20 ms action delay.
- Show which model is loaded in small text (e.g. "Model: v4 C. elegans, seed N, median by
  held-rate"). Use the median model, not the best.
- Until validated v4 weights exist, build with the prototype distilled worm and label it
  "prototype".

## Parity (required before shipping)
- Python exporter writes the model JSON plus fixture trajectories: initial states, the
  action sequence and resulting states for several hundred ticks, for both controllers.
- A TypeScript test replays the fixtures and must match within a stated tolerance
  (report it). Run in CI.

## Repo layout (connectome-control)
- Python library and pipeline stay where they are.
- `web/`: the demo. Vite + TypeScript, no UI framework. Suggested modules:
  `src/physics.ts`, `src/delay.ts`, `src/controllers/classical.ts`,
  `src/controllers/worm.ts`, `src/ui/stage.ts`, `src/ui/lanes.ts`, `src/main.ts`.
  Weights in `web/public/models/`.
- `tools/export_web_model.py`: exports model JSON and parity fixtures.
- `shared/fixtures/`: parity fixtures, read by the TypeScript tests.

## Deployment
- Vercel project with root directory `web/`. Static output only.
- Preview deploys per branch; production from main.

## Acceptance
1. Visual and behavioural match with the reference mockup on desktop and phone width.
2. Parity test passes for both controllers.
3. Lighthouse accessibility check passes; keyboard focus visible; touch targets ≥ 44 px.
4. Page weight small (model JSON is tens of kB); runs at 60 fps on a mid-range phone.
5. Deployed preview URL reported back.

---
## Amendment (2026-10-05): plant decision supersedes mockup constants
One plant for the whole demo = the TRAINING plant (classic cart-pole:
pole mass 0.1, half-length 0.5, g 9.8, frictionless, 4/3 dynamics),
training force limit (18 N) and track (±3 m). Both tabs use it. The
exporter (`tools/export_web_model.py`) writes the plant constants into the
model JSON; the web physics reads them from there — never hardcoded.
Classical retuned for this plant (dLQR @50 Hz, rescaled pumping; forward
prediction through queued actions, x-centering, wall-blend, +6% energy
target retained) and verified headlessly: 100% swing-up (200 eps),
0/300 stalls under random pushes, delay breaking point 130 ms (vs the
mockup plant's ~110 ms). Parity fixtures cover plant + both controllers,
tolerance 1e-5 abs over 500 ticks, run in CI.
Demo model selection rule: see repo README ("Web demo model selection").
