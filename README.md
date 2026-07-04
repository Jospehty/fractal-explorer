# fractal-explorer

Fly through an infinite 3D fractal. There is no bottom: as you approach any
surface the world *rebases* — new detail grows beneath you while the
structures you left behind become colossal vistas overhead. Depth level and
total zoom (which can exceed 10^50) are shown in the HUD.

No build step, no dependencies: plain JavaScript + WebGL2.

## How to Run Locally

This is a static web project. Serve the files with any local web server to
avoid browser security restrictions (CORS) when loading local assets.

**Option 1: Python (Recommended)**
```bash
python3 -m http.server 8080
```
Then navigate to `http://localhost:8080`.

**Option 2: Node.js / npm**
```bash
npx serve .
```

**Option 3: VS Code**
Install the **Live Server** extension, right-click `index.html`, and select
"Open with Live Server".

## Controls

| Input | Action |
| --- | --- |
| mouse | look |
| W A S D | fly |
| Space / Shift | rise / sink |
| scroll | speed — the *only* speed control; rebasing carries it across zoom levels so ×1 always feels the same relative to nearby terrain |
| Q / E | roll |
| L | drop a point light just ahead of you, sized to your current scale (for dark interiors) |
| C | cycle the manual light color (auto / warm / cold / ember / emerald / violet / white) |
| O | toggle automatic light spawning (off = fully manual lighting) |
| X | remove the light nearest to you |
| F | autopilot dive |
| G | toggle light glow |
| 1 / 2 / 3 | render quality (soft shadows, light shadows) |
| N | new world (seed is shareable via the URL: `?seed=...`) |
| H | help overlay |
| P | pause |

**Screenshots:** hold perfectly still. The renderer detects a static camera,
ramps to full resolution and accumulates sub-pixel-jittered frames into a
supersampled, shimmer-free image (the HUD shows `still ×N` while converging).
Any input drops instantly back to the responsive path.

## How it works

Everything worth knowing about the math lives in a long header comment in
`js/world.js`; a short version:

- The fractal is a kaleidoscopic IFS: each absolute depth level `j` has its
  own fold recipe (reflection-group fold + uniform scale + translation),
  palette and lighting, derived deterministically from `hash(seed, j)` in
  `js/levelgen.js`. Levels are grouped into biomes that share stylistic biases.
- The camera lives in frame `K` (the *rebase level*). Approaching the surface
  pushes the camera through the level-`K` fold (float64, exact) and increments
  `K`, so camera-local numbers stay O(1) at unbounded depth.
- The GPU raymarches with a two-phase distance estimator (`js/shaders.js`):
  a float64-maintained chain of outer levels renders the giant vistas without
  float32 precision loss, then the plain fold pipeline handles inner detail.
  `js/world.js` and the shader implement the *same* fold, and the test suite
  validates them against each other bit-for-bit.
- Point lights are world objects in frame-`K` coordinates; rebasing rescales
  radius (×s) and intensity (×s²) so lighting is scale-invariant. Each level
  spawns its light at most once per session; lights fade IN over a few seconds
  (no pop at the rebase that spawns them), sit against walls a few units out
  (pools and cascading shadows, not camera floodlights), fade with zoom age,
  and some biomes spawn none at all. The mood aims dark: pools of light only
  mean something next to real darkness. `O/L/C/X` give full manual control.
- Fog is continuous across rebases: density and the march horizon live in
  frame units, which rescale by `s` at every rebase, so a `fogMul` state
  ramps within each level and is rescaled by exactly `1/s` at descend — the
  optical depth along any ray never jumps (`updateFog` in `world.js`).

## Development

Files:

- `js/vec.js` — float64 vec3/mat3 helpers
- `js/levelgen.js` — deterministic per-level fold/palette/light parameters
- `js/world.js` — world model: rebasing, outer-level chain, DE, lights
- `js/shaders.js` — GLSL raymarcher (+ accumulation blit) as JS template strings
- `js/renderer.js` — WebGL2 plumbing, dynamic resolution, still-frame accumulation
- `js/input.js` — pointer-lock mouse look, WASD, wheel speed, touch
- `js/main.js` — app loop: camera, movement, rebase/parity handling, HUD

### Tests

```bash
node test/world.test.js
```

Validates the infinite-zoom architecture numerically at many depths: chain
consistency, rebase invariance of the distance estimator, ascend∘descend
identity, and a float32 emulation of the exact shader pipeline against the
float64 reference (this is the test that precision survives unbounded depth).
Run it after touching `world.js`, `levelgen.js` or the fold code in
`shaders.js`.

### Visual tools

Headless Playwright harnesses (need `playwright` importable and a Chromium
binary; they render via SwiftShader):

```bash
node tools/screenshot.mjs [seed] [outdir]   # autopilot dive, shots at many depths
node tools/stylecheck.mjs                   # each fold style in isolation
node tools/seamtest.mjs                     # before/after rebase seam images
node tools/gpuprobe.mjs                     # GPU map() vs CPU DE, bit-exact
node tools/cpurender.mjs                    # tiny CPU reference render
```

Useful URL parameters: `?seed=N` world seed · `?det=1` deterministic test
mode (exposes `window.__explorer`) · `?sdbg=stats|hitmask|shade|...` shader
debug views · `?style=0|1|2` force a fold style · `?canon=1` canonical Menger.

### Tuning notes (hard-won, do not casually change)

- Fold styles must keep every fold boundary a pure reflection; translation
  boundaries break the seamless-rebase property (see `world.js` header).
- Scale/translation anchors in `levelgen.js` are held exactly at per-style
  connectivity bands; even ±2% per-level jitter erodes the attractor to dust.
- `HORIZON` (world.js) and `TMAX` + the horizon fade (shaders.js) are coupled:
  the effective march limit is `TMAX / fogMul` with `fogMul >= 0.85`, and full
  fog must be reached before `2*HORIZON` so dropping an outer level is never
  visible (`0.92 * TMAX / 0.85 <= 2*HORIZON`). Enlarging the outer window
  further degrades the rebase-transport outlier budget in the tests — the
  major-outlier tier there was calibrated against the current values.
