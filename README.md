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
| V | photo mode: freeze the camera, release the mouse, and change exposure, sun position/color, haze, surface colors and lens live |
| T | save the current frame as a PNG (combine with holding still for supersampled shots) |
| F | autopilot dive |
| G | toggle light glow |
| 1 / 2 / 3 | render quality (soft shadows, light shadows) |
| N | new world (seed is shareable via the URL: `?seed=...`) |
| H | help overlay |
| P | pause |

**Screenshots:** hold perfectly still. The renderer detects a static camera,
switches to full resolution and accumulates sub-pixel-jittered samples into a
supersampled, shimmer-free image (the HUD shows `still ×N` completed samples).
Samples are rendered PROGRESSIVELY — a band of rows per frame, sized to a
~20 ms budget — so stopping never stalls the page on any GPU.
The held image also *refines*: after ~8 frames it switches to half the level-
of-detail footprint, a bigger step budget and high-quality shadows, and after
~40 frames halves the footprint again — each switch fades in (the
accumulator restarts at weight 1/3), so blurry or warped far structure
resolves into real geometry. Any input drops instantly back to the
responsive path. **Photo mode (V)** keeps the camera frozen while you adjust
the look; every control is a live shader uniform, so the image re-converges
immediately.

## How it works

Everything worth knowing about the math lives in the header comments of
`js/world.js` (and next to each technique in `js/shaders.js`); a short
version:

- **The fractal.** A kaleidoscopic IFS: each absolute depth level `j` has its
  own fold recipe (reflection-group fold + uniform scale + translation),
  palette and lighting, derived deterministically from `hash(seed, j)` in
  `js/levelgen.js`. Levels are grouped into biomes that share a fold style and
  a curated color *mood* (abyssal teal & bone, ember cathedral, violet nebula,
  jade temple, glacier, copper & verdigris, obsidian & gold, coral, sandstone
  dusk). Six fold styles. The organic ones: BRANCH (a fold IFS that mirrors
  only one or two axes, with a large twist — the copies grow off each other
  like a plant; three presets: *cauliflower* romanesco/brain-coral clusters,
  *membrane* torn curved sheets, *lichen* crinkled coral crust. A BRANCH
  level can be a *composite* of m small sub-steps, so its shape is fractal
  at every scale instead of one coarse jump per level) and DIHEDRAL (an
  n-fold radial kaleidoscope, n = 4–7, twisted about its axis level by
  level — porous coral caverns, overhangs, tunnels and spirals). Organic
  biomes *drift*: their twist and pull glide along a sine through the zone,
  so the forms keep evolving as you zoom instead of one recipe repeating.
  The geometric ones: POLY (crystalline),
  MENGER (architectural), OCTA (canyons) and ICOSA (the icosahedral H3
  kaleidoscope — rounded, organic). MENGER/OCTA additionally use offset fold
  planes (the Mandelbox box-fold trick). Every fold op must be a pure reflection, and every fold must be
  continuous (curved maps such as sphere folds are ruled out).
- **Infinite zoom (rebasing).** The camera lives in frame `K`. Approaching the
  surface pushes the camera through the level-`K` fold (float64, exact) and
  increments `K`, so camera-local numbers stay O(1) at unbounded depth.
- **The outer window = perturbation theory.** The vistas around you are the
  previous levels, rendered through a float64-maintained chain of the
  camera's position at each outer level. A ray sample is carried through
  them as a *delta* from the camera's chain point: linear where it shares the
  camera's branch, and otherwise through `foldDelta`, which evaluates
  `Fold(c + d) - Fold(c)` from the delta and the chain point's own branch
  decision values — the same trick deep-zoom Mandelbrot renderers use. No
  sample ever loses relative precision, however many levels (up to 48) lie
  between it and the camera. That is what allows a horizon of ~10^11 local
  units: structures ~10^10 times your size stay visible.
- **Per-level bounding balls.** The distance estimate `(|x - C| - R)/dr`
  uses a bounding ball (centre and radius) of the attractor after the last
  applied level, derived per absolute level by backward recursion from a
  fixed point far below. Off-centre balls matter for BRANCH, whose
  attractors sit well away from the origin. A point's distance is identical
  before and after a rebase, even where biomes with very different shapes
  meet.
- **Autopilot.** Sphere-traces a fan of rays and heads for the nearest real
  hit, plus one ray aimed at the deeper attractor's bounding ball. Following
  the distance gradient got trapped in basins, and in sparse organic worlds
  the bound can be loose with nothing actually there. A 5 s stall widens the
  fan.
- **Performance control.** Dynamic resolution is driven by real GPU time
  (`EXT_disjoint_timer_query_webgl2`) or the frame interval, shrinking in
  proportion to the overload (a slow GPU settles within a few frames) and
  growing back slowly under a ceiling; if even the minimum resolution is too
  slow, quality steps down automatically. Weak/integrated GPUs start at
  quality 0 (no shadows/AO). Only the fast shader variant compiles at
  startup; the perturbative one compiles in the background
  (`KHR_parallel_shader_compile`). Outer-level skipping uses a binary search,
  and the per-pixel coloring pass skips levels too coarse to matter.
- **Over-relaxed marching** (Keinert et al. 2014): primary rays step 1.25x
  the safe distance while consecutive spheres overlap, falling back safely
  when they don't.
- **Per-level skipping.** A sample within radius `R_b` of the camera takes the
  linear branch at every level `0..b`, so one precomposed matrix replaces all
  of them (inside the smallest radius the whole outer window vanishes). The
  perturbative fold is compiled into a second program variant used only on
  frames where some outer chain point sits within `4e-3·|c|` of a fold crease
  (`world.needsPerturb`); otherwise the two variants are identical.
- **Branch continuity.** When the camera crosses a fold crease of an outer
  level, its chart is re-derived from the *continuation* of the old one: the
  physical camera point is `A_old⁻¹(c)`, and its true fold gives the new
  chart. Folds are continuous, so the correction is the mirror across the
  crossed crease — an exact symmetry, invisible, and never larger than twice
  the step. Large moves are sub-stepped (`World.moveCamera`) so creases are
  crossed one at a time. (Searching *any* valid preimage instead used to
  teleport the camera: POLY/OCTA fold images overlap themselves.)
- **Nothing pops at a zoom step.** Lights are carried by the camera-branch
  affine; movement speed, fog, shadow reach, AO and headlight use a
  per-level sawtooth (`fogMul`) rescaled by exactly `1/s` at each rebase;
  "up", the sun and the star field ride a world-anchored sky basis (frame
  axes are re-charted at every rebase); sky/haze/sun moods are interpolated
  continuously in depth.
- **Scale-aware color.** A level only colors a surface if its structures are
  near the scale it is viewed at (orbit traps get a smooth penalty in
  `log(viewing distance / feature size)`), and glowing veins are drawn per
  level with a scale window — they fade in as they become resolvable and out
  before they could wash a colossal surface. Both are continuous in camera
  motion and exactly rebase-invariant.
- **Atmosphere.** Haze extinguishes per *decade* of distance (log-distance
  fog), so giant structures dissolve gradually into a dark haze that glows
  toward the sun (forward scattering), instead of vanishing at a fixed
  multiple of your size.
- **Point lights** are world objects in frame-`K` coordinates; rebasing
  rescales radius (×s) and intensity (×s²) so lighting is scale-invariant.
  Each level spawns its light at most once per session, lights fade in, sit
  against walls, fade with zoom age, and some biomes spawn none.
  `O/L/C/X` give full manual control.

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
binary; they render via SwiftShader, so they are slow but deterministic):

```bash
node tools/tour.mjs [seed] [outdir] [depths]   # per depth: wall / along / out views + contact sheet
node tools/lookdev.mjs seed depths views '[{...look},...]' out.png  # same frames, lighting variants
node tools/screenshot.mjs [seed] [outdir]      # autopilot dive, shots at many depths
node tools/stylecheck.mjs                      # each fold style in isolation
node tools/seamtest.mjs [K]                    # before/after a forced rebase (shots-debug/)
node tools/gpuprobe.mjs [K]                    # GPU map() vs CPU DE, bit-exact
node tools/cpurender.mjs                       # tiny CPU reference render
```

Useful URL parameters: `?seed=N` world seed · `?det=1` deterministic test
mode (exposes `window.__explorer`) · `?sdbg=stats|hitmask|shade|noemis|...`
shader debug views · `?style=0|1|2|3` force a fold style · `?canon=1`
canonical Menger · `?style=4&n=5&h=0.6&fscale=3.3&twist=0.4` explore the
dihedral style (order, ring height, scale, axial twist) ·
`?look={"fill":0.8,"exposure":1.2}` override the lighting/grade defaults
(`LOOK_DEFAULTS` in `js/shaders.js`; the same values photo mode edits).

### Tuning notes (hard-won, do not casually change)

- Fold styles must keep every fold boundary a pure reflection; translation
  boundaries break the seamless-rebase property (see `world.js` header).
- Scale/translation anchors in `levelgen.js` are held exactly at per-style
  connectivity bands; even ±2% per-level jitter erodes the attractor to dust.
- `HORIZON` (world.js) and `TMAX` + the log-space horizon fade (shaders.js)
  are coupled: the effective march limit is `TMAX / fogMul` with
  `fogMul >= 0.85`, and full fog must be reached before `2*HORIZON` so
  dropping an outer level is never visible (`0.92 * TMAX / 0.85 <= 2*HORIZON`).
  `B_HARD` must cover the horizon at the smallest scale factor (1.78^48 ~ 1e12).
- The outer phase exists in FOUR places that must stay structurally
  identical: the `OUTER` template in `shaders.js`, `World.de` (float64), the
  float32 emulation in `test/world.test.js`, and `foldDelta` in both
  languages. The tests compare them bit-for-bit-ish at depths up to 10^48.
- BRANCH presets (`BRANCH_PRESETS`): a fold IFS keeps only the part of each
  copy inside the fold image, so most twist axes prune the tree to dust.
  The axis is fixed near (0.3, 0.9, 0.2), and `twMin`/`hr` clamp the
  *drifted* per-level twist and height, measured with ray-hit density
  sweeps. Cauliflower turns to dust below twist ≈1.15, and membrane above
  height ≈0.7. POLY is solid only at s = 1.66–1.78.
- Small-scale styles (BRANCH ≈1.6–2 per level) can saturate the 48-level
  window before the horizon. `World.windowReach` then raises the fog floor
  so geometry beyond the window is always fully fogged.
- DIHEDRAL connectivity band: solid drums below s≈2.8, porous tori and
  coral at 2.95–3.45 (`STYLE_ANCHOR`); n ≤ 7 so its fold fits in
  `MAX_STAGES` (1 + up to 7 mirror ops).
- Coloring and veins must stay scale-aware: any per-level effect taken as a
  plain min/max over the whole (48-level) window is dominated by root levels.
