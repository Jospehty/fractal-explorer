/* shaders.js — GLSL for the raymarched deep-zoom renderer.
   The fold pipeline here MUST stay structurally identical to evalFold() in
   world.js (validated bit-for-bit against it by test/world.test.js through a
   Math.fround emulation of this exact code). */
(function (global) {
  'use strict';

  const VERT = `#version 300 es
  void main() {
    vec2 p = vec2[3](vec2(-1.,-1.), vec2(3.,-1.), vec2(-1.,3.))[gl_VertexID];
    gl_Position = vec4(p, 0., 1.);
  }`;

  // Constants injected to match world.js (ICO_N is the single source of
  // truth for the icosahedral mirror normal — see world.js)
  const DEFS = (WM) => `
  #define B_HARD ${WM.B_HARD}
  #define N_INNER ${WM.N_INNER}
  #define SLOTS ${WM.SLOTS}
  #define ESCAPE2 ${WM.ESCAPE2.toFixed(1)}
  #define MAX_LIGHTS ${WM.MAX_LIGHTS_GPU}
  #define ICO_N vec3(${WM.ICO_N.map((v) => v.toFixed(10)).join(', ')})
  #define OV ${WM.OUTER_VEC4}
  #define LV ${WM.LVL_VEC4}
  #define ABS_REL2 ${WM.ABS_REL2.toExponential(4)}
  `;

  /* Look constants (lighting balance / grade) — uploaded as uniforms
     (uLookA..D), so photo mode can change them live. Overridable for
     look-dev via the URL: ?look={"fill":0.5,...} (see main.js). */
  const LOOK_DEFAULTS = {
    exposure: 1.4,   // pre-tonemap exposure
    sun: 1.0,        // key light (sun) strength
    sunSat: 0.6,     // how much of the sun's tint reaches surfaces (1 = full)
    fill: 1.0,       // hemispheric sky fill strength
    ground: 0.5,     // fill from below, relative to the haze color
    bounce: 0.06,    // warm-ish bounce from the side facing away from the sun
    head: 0.22,      // camera headlight
    rim: 1.6,        // haze rim on grazing surfaces
    sat: 1.3,        // albedo saturation
    huePreserve: 0.5,// 0 = plain ACES (bleaches bright colors), 1 = hue-preserving
    albHue: 0,       // surface hue rotation (turns) — photo mode
    glow: 1,         // emissive vein strength
    vignette: 0.22,
    hazeMul: 1,      // haze density multiplier — photo mode
    hazeHue: 0,      // sky/haze hue rotation (turns) — photo mode
    customSun: false,// photo mode: place the sun yourself
    sunAz: 40, sunEl: 35, sunHue: 0.09, sunTint: 0.4,
  };

  /* Outer phase, generated inline into map() and mapDE() (a shared GLSL
     function with inout state was measured ~2.4x slower: it put the escape
     test and mode flags on the hot path). Per outer level the delta d (camera
     relative, frame of that level) takes:
       linear branch   |d| < margin : d = J d             (the common case)
       perturbative    otherwise    : d = foldDelta(d)    (exact, relative)
       absolute        |d| > 4e-3|c| (ABS_REL2) : x = c + d, plain folds for
                       the rest of the window (tail loop) — absolute rounding
                       is now small relative to d (see world.js ABS_REL2).
     trapHook: GLSL run after every level with xs = level-local position. */
  const OUTER = (trapHook, perturb) => `
      int bAbs = B_HARD;
      for (int b = bStart; b < B_HARD; b++) {
        if (b >= uB) break;
        vec4 cm = outerData[b*OV+3];
        float dd = dot(d, d);
        dr *= lvl[b*LV].w;
        if (dd < cm.w * cm.w) {
          d = mat3(outerData[b*OV].xyz, outerData[b*OV+1].xyz, outerData[b*OV+2].xyz) * d;
          ${trapHook ? '{ vec3 xs = ((b + 1 < uB) ? outerData[(b+1)*OV+3].xyz : uCamK) + d; ' + trapHook + ' }' : ''}
          continue;
        }
        vec3 cn = (b + 1 < uB) ? outerData[(b+1)*OV+3].xyz : uCamK;
        if (dd > ABS_REL2 * max(dot(cm.xyz, cm.xyz), 1.0)) {
          x = foldLevel(cm.xyz + d, b);
          bAbs = b;
          sl = b;
          if (dot(x, x) > ESCAPE2) esc = true;
          ${trapHook ? '{ vec3 xs = x; ' + trapHook + ' }' : ''}
          break;
        } else {
          ${perturb ? 'd = foldDelta(d, b);' : 'd = foldLevel(cm.xyz + d, b) - cn; /* fast variant: unreachable unless world.needsPerturb */'}
          vec3 xe = cn + d;
          if (dot(xe, xe) > ESCAPE2) { x = xe; esc = true; sl = b; break; }
        }
        ${trapHook ? '{ vec3 xs = cn + d; ' + trapHook + ' }' : ''}
      }
      if (!esc && bAbs < B_HARD) {
        // absolute tail: remaining outer levels as plain folds
        for (int b = 0; b < B_HARD; b++) {
          if (b <= bAbs) continue;
          if (b >= uB) break;
          x = foldLevel(x, b);
          dr *= lvl[b*LV].w;
          sl = b;
          if (dot(x, x) > ESCAPE2) { esc = true; break; }
          ${trapHook ? '{ vec3 xs = x; ' + trapHook + ' }' : ''}
        }
      } else if (!esc) {
        x = uCamK + d;
      }`;

  /* opts.perturb: compile the perturbative outer fold (foldDelta). It is
     only ever NEEDED when some outer chain point sits within 4e-3·|c| of a
     fold crease (world.needsPerturb); otherwise the branch is unreachable
     and the two variants render identically. Its mere presence costs ~40%
     march speed (register pressure), so the renderer keeps both programs
     and picks per frame. */
  const FRAG = (WM, opts) => { const PERT = !(opts && opts.perturb === false); return `#version 300 es
  precision highp float;
  precision highp int;
  ${DEFS(WM)}

  out vec4 fragColor;

  uniform vec2  uRes;
  uniform float uTime;
  uniform mat3  uCamBasis;   // columns: right, up, forward
  uniform float uFovTan;     // tan(fov/2)
  uniform vec3  uCamK;       // camera position in frame K
  uniform mat3  uW;          // composed inverse linear map (frame K -> K-B)
  uniform float uWScale;
  uniform int   uB;          // active outer levels
  uniform vec3  uSunDir;
  uniform vec3  uSunCol;
  uniform vec3  uSkyCol;
  uniform vec3  uFogCol;
  uniform int   uLightN;
  uniform vec4  uLightPos[MAX_LIGHTS]; // xyz, w = radius
  uniform vec4  uLightCol[MAX_LIGHTS]; // rgb, w = intensity
  uniform int   uQuality;    // 0 low, 1 medium, 2 high
  uniform float uGlowAmt;
  uniform vec2  uJitter;     // sub-pixel jitter for still-frame accumulation
  uniform float uHaze;       // haze extinction per decade of distance (mood)
  // Look / photo-mode controls (renderer.look, see LOOK_DEFAULTS):
  uniform vec4  uLookA;      // exposure, sun strength, sun tint, sky fill
  uniform vec4  uLookB;      // ground fill, bounce, headlight, haze rim
  uniform vec4  uLookC;      // albedo saturation, hue-preserving tonemap, albedo hue shift (turns), vein glow
  uniform vec4  uLookD;      // detail (LOD/hit footprint, 1 = live), vignette, step budget scale, -
  uniform mat3  uSkyBasis;   // WORLD axes in frame-K coords (world.skyBasis):
                             // "up" and the star field ride it so they never
                             // jump when a rebase re-charts the frame
  uniform float uFogMul;     // fog continuity multiplier (world.fogMul): scales
                             // fog density up / march limit down as the camera
                             // progresses through a level, so optical depth is
                             // continuous across the rebase rescale

  layout(std140) uniform Params {
    vec4 lvl[SLOTS * LV];       // per slot (LV=7): rot0(w=scale) rot1(w=style) rot2(w=foldL|n) trans(w=emissive) pal(w=accent) foldOff|N2(w=glow hue) (bound after level)
    vec4 outerData[B_HARD * OV]; // per outer: jac0 jac1 jac2 (c.xyz, margin) stages0-3 stages4-7
  };

  // attractor bounding radius after slot s's fold (world.js boundAfter)
  float bnd(int s) { return lvl[s*LV+6].x; }

  /* ---- the fold (mirror of world.js evalFold) ---- */
  vec3 foldLevel(vec3 x, int s) {
    vec4 r0 = lvl[s*LV+0], r1 = lvl[s*LV+1], r2 = lvl[s*LV+2], tr = lvl[s*LV+3];
    x = mat3(r0.xyz, r1.xyz, r2.xyz) * x;
    float style = r1.w;
    if (style < 0.5) {            // POLY: diagonal plane folds
      if (x.x + x.y < 0.0) x.xy = -x.yx;
      if (x.x + x.z < 0.0) x.xz = -x.zx;
      if (x.y + x.z < 0.0) x.yz = -x.zy;
    } else if (style < 1.5) {     // MENGER: offset abs + sort desc
      vec3 fo = lvl[s*LV+5].xyz;
      x = fo + abs(x - fo);
      if (x.x < x.y) x.xy = x.yx;
      if (x.x < x.z) x.xz = x.zx;
      if (x.y < x.z) x.yz = x.zy;
    } else if (style < 2.5) {     // OCTA: offset abs + partial sort
      vec3 fo = lvl[s*LV+5].xyz;
      x = fo + abs(x - fo);
      if (x.x < x.y) x.xy = x.yx;
      if (x.y < x.z) x.yz = x.zy;
    } else if (style < 3.5) {     // ICOSA: H3 kaleidoscope, 2x (abs + golden plane)
      x = abs(x);
      x -= 2.0 * min(0.0, dot(x, ICO_N)) * ICO_N;
      x = abs(x);
      x -= 2.0 * min(0.0, dot(x, ICO_N)) * ICO_N;
    } else {                      // DIHEDRAL: z mirror + D_n wedge (alternating mirrors)
      x.z = abs(x.z);
      vec3 n2 = lvl[s*LV+5].xyz;
      int nOps = int(r2.w + 0.5);
      nOps = (nOps & 1) == 1 ? nOps : nOps + 1;
      for (int k = 0; k < 7; k++) {
        if (k >= nOps) break;
        if ((k & 1) == 0) x.y = abs(x.y);   // N1 = y=0 plane (exact)
        else x -= 2.0 * min(0.0, dot(x, n2)) * n2;
      }
    }
    return x * r0.w + tr.xyz;
  }

  /* ---- perturbative fold (mirror of world.js foldDelta) ----
     Fold(c + d) - Fold(c) for outer slot b from the delta d and the chain
     point's decision scalars alone: full RELATIVE precision in d at any
     magnification (deep-zoom perturbation theory). */
  float stg(int b, int k) { return outerData[b*OV + 4 + (k >> 2)][k & 3]; }
  void absOp(inout float di, float a) {
    float v = a + di;
    if (a >= 0.0) { if (v < 0.0) di = -2.0 * a - di; }
    else di = v < 0.0 ? -di : 2.0 * a + di;
  }
  vec2 swapOp(vec2 dpq, float D) {       // "if x_p < x_q swap", D = ref x_p - x_q
    bool ref = D < 0.0, pix = D + dpq.x - dpq.y < 0.0;
    if (ref && pix) return dpq.yx;
    if (pix) return vec2(-D + dpq.y, D + dpq.x);
    if (ref) return vec2(D + dpq.x, -D + dpq.y);
    return dpq;
  }
  vec2 flipOp(vec2 dab, float Sg) {      // POLY flip, Sg = ref x_a + x_b
    bool ref = Sg < 0.0, pix = Sg + dab.x + dab.y < 0.0;
    if (ref && pix) return -dab.yx;
    if (pix) return vec2(-Sg - dab.y, -Sg - dab.x);
    if (ref) return vec2(Sg + dab.x, Sg + dab.y);
    return dab;
  }
  vec3 planeOp(vec3 d, float Tn, vec3 N) { // "if x·N < 0 reflect", Tn = ref x·N
    float dn = dot(d, N);
    bool ref = Tn < 0.0, pix = Tn + dn < 0.0;
    float f = (ref && pix) ? -2.0 * dn : (pix ? -2.0 * (Tn + dn) : (ref ? 2.0 * Tn : 0.0));
    return d + f * N;
  }
  vec3 goldOp(vec3 d, float Tn) { return planeOp(d, Tn, ICO_N); } // golden plane
  vec3 foldDelta(vec3 d, int b) {
    vec4 r0 = lvl[b*LV+0], r1 = lvl[b*LV+1], r2 = lvl[b*LV+2];
    d = mat3(r0.xyz, r1.xyz, r2.xyz) * d;
    float style = r1.w;
    if (style < 0.5) {            // POLY
      d.xy = flipOp(d.xy, stg(b, 0));
      d.xz = flipOp(d.xz, stg(b, 1));
      d.yz = flipOp(d.yz, stg(b, 2));
    } else if (style < 2.5) {     // MENGER / OCTA: offset abs + (partial) sort
      absOp(d.x, stg(b, 0)); absOp(d.y, stg(b, 1)); absOp(d.z, stg(b, 2));
      d.xy = swapOp(d.xy, stg(b, 3));
      if (style < 1.5) {
        d.xz = swapOp(d.xz, stg(b, 4));
        d.yz = swapOp(d.yz, stg(b, 5));
      } else {
        d.yz = swapOp(d.yz, stg(b, 4));
      }
    } else if (style < 3.5) {     // ICOSA
      absOp(d.x, stg(b, 0)); absOp(d.y, stg(b, 1)); absOp(d.z, stg(b, 2));
      d = goldOp(d, stg(b, 3));
      absOp(d.x, stg(b, 4)); absOp(d.y, stg(b, 5)); absOp(d.z, stg(b, 6));
      d = goldOp(d, stg(b, 7));
    } else {                      // DIHEDRAL
      absOp(d.z, stg(b, 0));
      vec3 n2 = lvl[b*LV+5].xyz;
      int nOps = int(r2.w + 0.5);
      nOps = (nOps & 1) == 1 ? nOps : nOps + 1;
      for (int k = 0; k < 7; k++) {
        if (k >= nOps) break;
        if ((k & 1) == 0) absOp(d.y, stg(b, k + 1)); // y=0 mirror == abs on y
        else d = planeOp(d, stg(b, k + 1), n2);
      }
    }
    return d * r0.w;
  }

  /* ---- two-phase distance estimator ----
     p: camera-relative frame-K position. coneEps: feature size below which we
     band-limit (adaptive iteration count).
     trapOut: (slotA, trapA, slotB, trapB) — the two smallest orbit traps and
     their owning level slots. Shading crossfades the two slots' palettes by
     trap separation, so colors blend smoothly where trap ownership changes
     instead of snapping (both across surfaces and while zooming).
     glow: emissive vein light, accumulated per level with a SCALE WINDOW
     (see veinGlow) so veins are anchored to absolute levels yet only drawn
     at resolvable, non-overwhelming scales. */
  /* Scale-aware trap weight: a level may only color a surface if its
     structures are near the scale it is viewed at. dr*tView ~ (viewing
     distance / level feature size): levels whose features dwarf the viewing
     distance (ancient outer levels — you see a tiny piece of them) are
     smoothly penalized. Without this, the min over a 48-level window was
     almost always won by a root level, and the whole dive wore one color.
     Continuous in camera motion, and rebase-invariant (dr*t is). */
  float trapPenalty(float dr, float tView) {
    return 0.22 * max(0.0, -log2(max(dr * tView, 1e-30)) - 1.0);
  }

  /* Emissive veins of level slot s at level-local point xs. Each level's
     vein (a filament around a plane, core ~0.008 level units = 0.008/dr in
     frame K) fades IN as its core becomes wider than ~a pixel (sub-pixel
     veins were pure noise) and OUT before it exceeds a small fraction of the
     viewing distance (root-level veins used to paint whole colossal surfaces
     in glow). */
  vec3 veinGlow(int s, vec3 xs, float dr, float tView, float pixT) {
    float em = lvl[s*LV+3].w;
    if (em <= 0.0) return vec3(0.0);
    float w = 0.008 / dr;                        // vein core width, frame-K units
    float fade = smoothstep(0.8, 2.5, w / pixT)  // resolvable (pixels)
               * (1.0 - smoothstep(0.0015, 0.008, w / tView)); // filaments, not stripes
    if (fade <= 0.0) return vec3(0.0);
    // bright core + soft halo: reads as a glowing filament, not flat paint
    float dpl = abs(xs.z + 0.4 * xs.x - 0.35);
    float v = smoothstep(0.006, 0.0, dpl) + 0.35 * exp(-dpl * 90.0);
    // vivid glow color from the level's glow hue (sat 0.8)
    float h = lvl[s*LV+5].w;
    vec3 c = clamp(abs(fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    c = mix(vec3(1.0), c, 0.8);
    return c * (em * v * fade * 0.6);
  }

  float map(vec3 p, float coneEps, float tView, out vec4 trapOut, out vec3 glow) {
    vec3 x = vec3(0.0);
    float dr;
    int sl = uB; // slot of the last fold applied (selects the bounding radius)
    bool esc = false;
    float m1 = 1e9, m2 = 1e9;            // two smallest traps
    float s1 = float(uB), s2 = float(uB); // and their slots
    vec3 glowAcc = vec3(0.0);
    float pixT = coneEps / 0.22; // pixel footprint at the hit (t * pix)
    if (uB > 0) {
      vec3 d = uW * p;
      dr = uWScale;
      int bStart = 0; // traps need every level: no skipping here
      // Trap sampling through ALL outer levels keeps coloring anchored to
      // ABSOLUTE levels (with the scale-aware penalty — see trapPenalty)
      ${OUTER(`float ms = length(xs) * 0.9 + 0.12 * abs(xs.y) + trapPenalty(dr, tView);
          if (ms < m1) { m2 = m1; s2 = s1; m1 = ms; s1 = float(b); }
          else if (ms < m2) { m2 = ms; s2 = float(b); }
          glowAcc += veinGlow(b, xs, dr, tView, pixT);`, PERT)}
    } else {
      x = uCamK + p;
      dr = 1.0;
    }
    if (!esc) {
      for (int i = 0; i < N_INNER; i++) {
        int s = uB + i;
        x = foldLevel(x, s);
        dr *= lvl[s*LV].w;
        sl = s;
        float xx2 = dot(x, x);
        float m = sqrt(xx2) * 0.9 + 0.12 * abs(x.y) // slight anisotropy for banding
                + trapPenalty(dr, tView);
        if (m < m1) { m2 = m1; s2 = s1; m1 = m; s1 = float(s); }
        else if (m < m2) { m2 = m; s2 = float(s); }
        glowAcc += veinGlow(s, x, dr, tView, pixT);
        if (xx2 > ESCAPE2) break;
        ${global.__SHADER_DEBUG === 'nocone' ? '' : 'if (1.0 < dr * coneEps) break;'}
      }
    }
    trapOut = vec4(s1, m1, s2, m2);
    glow = glowAcc;
    // (|x| - R)/dr is a rigorous distance lower bound for any truncation depth
    // (R = bounding radius of the attractor after the last applied level,
    // per absolute level — see world.js boundAfter) — hole-free LOD.
    return (length(x) - bnd(sl)) / dr;
  }

  /* Distance-only variant used for ALL marching, normal, AO and shadow
     samples. Two savings over map(): no orbit-trap bookkeeping, and per-
     level SKIPPING — a sample within R_b of the camera takes the linear
     branch at every level 0..b, so one precomposed matrix Q_b replaces them
     all; within R_{B-1} (= RLin) every outer level is linear and they
     compose to the IDENTITY, so the outer phase vanishes. Traps are computed
     by one full map() call at the hit point. */
  float mapDE(vec3 p, float coneEps) {
    vec3 x = vec3(0.0);
    float dr;
    int sl = uB;
    bool esc = false;
    // skip the leading outer levels this sample is linear through: R_b is
    // nonincreasing, so count while |p| < R_b (see world.js, per-level skip)
    float pp = dot(p, p);
    int bStart = 0;
    for (int b = 0; b < B_HARD; b++) {
      if (b >= uB) break;
      float R = outerData[b*OV].w;
      if (pp >= R * R) break;
      bStart = b + 1;
    }
    if (bStart < uB) {
      vec3 d;
      if (bStart == 0) { d = uW * p; dr = uWScale; }
      else {
        int q = bStart - 1;
        d = mat3(outerData[q*OV+6].xyz, outerData[q*OV+7].xyz, outerData[q*OV+8].xyz) * p;
        dr = outerData[q*OV+1].w;
      }
      ${OUTER('', PERT)}
    } else {
      // linear through EVERY outer level: they compose to the identity
      x = uCamK + p;
      dr = 1.0;
    }
    if (!esc) {
      for (int i = 0; i < N_INNER; i++) {
        int s = uB + i;
        x = foldLevel(x, s);
        dr *= lvl[s*LV].w;
        sl = s;
        if (dot(x, x) > ESCAPE2) break;
        ${global.__SHADER_DEBUG === 'nocone' ? '' : `
        if (1.0 < dr * coneEps) {
          // SMOOTH LOD: instead of truncating at a hard iteration band (which
          // draws visible detail-rings that step as you move), blend this
          // band's bound with one fold deeper by the fractional band
          // position. Both are rigorous lower bounds for any truncation
          // depth, so the mix is hole-free.
          int s2 = min(s + 1, SLOTS - 1);
          float sc = lvl[s2*LV].w;
          float f = clamp((dr * coneEps - 1.0) / max(sc - 1.0, 0.1), 0.0, 1.0);
          float deA = (length(x) - bnd(s)) / dr;
          float deB = (length(foldLevel(x, s2)) - bnd(s2)) / (dr * sc);
          return mix(deB, deA, f);
        }`}
      }
    }
    return (length(x) - bnd(sl)) / dr;
  }

  float mapD(vec3 p, float coneEps) { return mapDE(p, coneEps); }

  vec3 calcNormal(vec3 p, float eps, float coneEps) {
    vec2 k = vec2(1.0, -1.0);
    return normalize(
      k.xyy * mapD(p + k.xyy * eps, coneEps) +
      k.yyx * mapD(p + k.yyx * eps, coneEps) +
      k.yxy * mapD(p + k.yxy * eps, coneEps) +
      k.xxx * mapD(p + k.xxx * eps, coneEps));
  }

  float softShadow(vec3 ro, vec3 rd, float t0, float tmax, float k, float coneEps, int steps) {
    float res = 1.0;
    float t = t0;
    for (int i = 0; i < 64; i++) {
      if (i >= steps || t >= tmax) break;
      // LOD cone widens along the ray: a soft penumbra cannot resolve fine
      // detail far from the surface, and coarser DE evaluations are cheaper
      float h = mapD(ro + rd * t, coneEps + t * 0.015);
      if (h < 1e-6 * t) return 0.0;
      res = min(res, k * h / t);
      if (res < 0.015) return 0.0; // effectively occluded: stop paying
      // minimum step grows with distance travelled (a cone): penumbrae are
      // soft far from the surface, so fine steps there were wasted work
      t += clamp(h, max(t0 * 0.5, t * 0.06), tmax * 0.12);
    }
    return clamp(res, 0.0, 1.0);
  }

  float calcAO(vec3 p, vec3 n, float scale, float coneEps) {
    float occ = 0.0, sca = 1.0;
    // 4 taps spanning the same radius as the old 5 (weights renormalized)
    for (int i = 0; i < 4; i++) {
      float h = (0.01 + 0.173 * float(i)) * scale;
      float d = mapD(p + n * h, coneEps + h * 0.2);
      occ += (h - d) * sca / scale;
      sca *= 0.66;
    }
    return clamp(1.0 - 2.5 * occ, 0.0, 1.0);
  }

  /* analytic inscatter of a point light along the ray (closed form) */
  float inscatter(vec3 ro, vec3 rd, vec3 lp, float tmax) {
    vec3 q = ro - lp;
    float b = dot(rd, q);
    float c = dot(q, q);
    float h = max(c - b * b, 1e-8);
    float hs = inversesqrt(h);
    return (atan((tmax + b) * hs) - atan(b * hs)) * hs;
  }

  float hash12(vec2 v) {
    vec3 p3 = fract(vec3(v.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  /* Haze color seen along direction rd: deep and dark, glowing toward the
     sun by forward scattering. Far structure dissolves into THIS, so distant
     colossi read as backlit silhouettes against a luminous sun-side haze and
     sink into darkness on the far side — instead of a flat pastel wash. */
  vec3 hazeColor(vec3 rd) {
    float sd = max(dot(rd, uSunDir), 0.0);
    float upd = dot(rd, uSkyBasis[1]);
    return uFogCol * (1.0 + 0.6 * clamp(upd, 0.0, 1.0))
         + uSunCol * (0.035 * sd * sd + 0.16 * pow(sd, 18.0))
         + uSkyCol * 0.12;
  }

  vec3 skyColor(vec3 rd) {
    float sd = max(dot(rd, uSunDir), 0.0);
    float upd = dot(rd, uSkyBasis[1]);
    vec3 sky = mix(hazeColor(rd), uSkyCol, clamp(upd * 0.6 + 0.2, 0.0, 1.0) * 0.6);
    sky += uSunCol * (pow(sd, 900.0) * 6.0 + pow(sd, 90.0) * 0.25);
    // starfield in WORLD directions (rides the sky basis through rebases)
    vec3 wd = transpose(uSkyBasis) * rd;
    vec2 sc = vec2(atan(wd.z, wd.x), asin(clamp(wd.y, -1.0, 1.0))) * 60.0;
    vec2 cf = fract(sc) - 0.5;
    float hs = hash12(floor(sc));
    float star = smoothstep(0.991, 1.0, hs) * smoothstep(0.14, 0.02, dot(cf, cf));
    float lum = dot(uSkyCol, vec3(0.333));
    sky += mix(vec3(1.0, 0.8, 0.6), vec3(0.65, 0.8, 1.0), fract(hs * 97.0))
         * star * clamp(0.25 - lum, 0.0, 0.25) * 6.0;
    return sky;
  }

  // hue rotation about the grey axis (h in turns): material accent colors
  vec3 hueRot(vec3 c, float h) {
    float a = h * 6.2831853;
    const vec3 k = vec3(0.57735027);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
  }

  vec3 palette(int slot) { return lvl[slot*LV+4].rgb; }
  float accentOf(int slot) { return lvl[slot*LV+4].w; }

  vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
  }

  uniform float uProbeOn;
  uniform vec3  uProbe;

  void main() {
    if (uProbeOn > 0.5) {
      // debug: output map(uProbe) bit-exactly, one byte per pixel in R
      // (alpha channel is unusable on an opaque canvas)
      vec4 tt; vec3 tp;
      float dv = map(uProbe, 0.0, 1.0, tt, tp);
      uint bb = floatBitsToUint(dv);
      uint sh = uint(gl_FragCoord.x) * 8u;
      fragColor = vec4(float((bb >> sh) & 255u) / 255.0, 0.0, 0.0, 1.0);
      return;
    }
    // uJitter: sub-pixel offset for still-frame accumulation (zero when moving)
    vec2 uv = ((gl_FragCoord.xy + uJitter) * 2.0 - uRes) / uRes.y;
    float pix = 2.0 * uFovTan / uRes.y;              // radians per pixel (approx)
    vec3 rd = normalize(uCamBasis * vec3(uv * uFovTan, 1.0));
    vec3 ro = vec3(0.0);                             // camera at origin of frame K

    // Horizon ~1e11 local units: structures up to ~10^10 times your size stay
    // visible (as hazy colossi). Coupled to world.js HORIZON — see there.
    const float TMAX = 1.0e11;
    float tmax = TMAX / uFogMul; // effective horizon, continuous across rebases
    int   MAXSTEP = int(float(uQuality == 0 ? 100 : (uQuality == 1 ? 150 : 210)) * uLookD.z);
    // detail < 1 (still-frame refinement): finer LOD and hit footprint, so a
    // held frame resolves smoothed/blobby far detail into real structure
    float detail = uLookD.x;
    float t = 0.0;
    float d = 0.0;
    vec4 trap = vec4(0.0);
    vec3 veins = vec3(0.0);
    bool hit = false;
    bool exhausted = false;
    int steps = 0;
    /* Over-relaxed sphere tracing (Keinert et al. 2014): step 1.25x the
       safe distance while consecutive unbounding spheres overlap; if they
       stop overlapping the long step may have skipped geometry, so fall back
       to the safe step from the previous point and stay conservative. Same
       surfaces, ~20-30% fewer steps in open space and along walls. */
    float omega = 1.25, tPrev = 0.0, dPrev = 0.0;
    for (int i = 0; i < 512; i++) {
      if (t > tmax) break;
      if (i >= MAXSTEP) { hit = true; exhausted = true; break; } // grazing geometry
      steps = i;
      vec3 p = ro + rd * t;
      d = mapDE(p, t * pix * 0.22 * detail); // trap-free marching; traps sampled at hit
      if (omega > 1.0 && d + dPrev < (t - tPrev) * 1.0001) {
        t = tPrev + min(dPrev, tmax * 0.5) * 0.92;
        omega = 1.0;
        continue;
      }
      if (d < t * pix * 0.45 * detail + 2e-6) { hit = true; break; }
      tPrev = t; dPrev = d;
      t += min(d, tmax * 0.5) * 0.92 * omega;
    }
    if (hit) map(ro + rd * t, t * pix * 0.22 * detail, t, trap, veins); // orbit traps for shading
    ${global.__SHADER_DEBUG === 'stats'
      ? `fragColor = vec4(clamp(log2(t + 1.0) / 15.0, 0.0, 1.0),
                          float(steps) / float(MAXSTEP), hit ? 1.0 : 0.0, 1.0); return;`
      : ''}

    ${global.__SHADER_DEBUG === 'hitmask'
      ? 'fragColor = vec4(hit ? 1.0 : 0.0, fract(log2(t + 1.0) * 0.2), trap.y * 0.2, 1.0); return;'
      : ''}
    vec3 col;
    vec3 dbgLights = vec3(0.0);
    float fogT = min(t, tmax);
    if (hit) {
      vec3 p = ro + rd * t;
      float coneEps = t * pix * 0.22 * detail;
      float eps = max(t * pix * 0.4 * detail, 2e-6);
      vec3 n = calcNormal(p, eps, coneEps);
      if (dot(n, rd) > 0.0) n = -n;

      int slot = int(trap.x + 0.5);
      int slotB = int(trap.z + 0.5);
      // crossfade the two nearest-trap palettes by trap separation: where the
      // argmin is about to switch (trapA ~ trapB) the blend is 50/50, so level
      // colors shift as smooth gradients instead of snapping at trap-ownership
      // boundaries — in space across a surface AND in time while zooming
      float sep = clamp((trap.w - trap.y) / (0.25 * (trap.y + trap.w) + 1e-5), 0.0, 1.0);
      float wA = 0.5 + 0.5 * sep * sep * (3.0 - 2.0 * sep); // smoothstep easing
      vec3 base = mix(palette(slotB), palette(slot), wA);
      // two-tone material: the level's accent hue takes over where the orbit
      // trap is large (outer shells / ridges) — same smooth crossfade weights,
      // so it never snaps either
      vec3 acc = max(hueRot(base, mix(accentOf(slotB), accentOf(slot), wA)), 0.0);
      float am = smoothstep(0.45, 1.25, trap.y);
      vec3 alb = mix(base, acc * 1.1, am * 0.75);
      // trap-based tone variation: crevices darken, ridges lighten
      float tv = clamp(trap.y * 0.7, 0.0, 1.4);
      alb *= 0.4 + 0.65 * tv;
      alb = max(mix(vec3(dot(alb, vec3(0.299, 0.587, 0.114))), alb, uLookC.x), 0.0);
      if (uLookC.z != 0.0) alb = max(hueRot(alb, uLookC.z), 0.0); // photo mode color shift
      vec3 up = uSkyBasis[1];

      // NOTE every absolute length here divides by uFogMul (or multiplies,
      // for extinctions): plain frame-unit constants rescale by s at every
      // rebase and used to make shadows/AO/headlight visibly pop per zoom step
      float ao = calcAO(p, n, max(t * 0.05, 0.02 / uFogMul), coneEps);
      float ao2 = ao * ao;
      float sunDiff = max(dot(n, uSunDir), 0.0);
      float sh = 1.0;
      if (uQuality >= 1 && sunDiff > 0.001) {
        // LOCAL reach (~12x your scale, or proportional to the hit distance
        // for far geometry): the sun sculpts cascading shadows from nearby
        // structure. Unlimited reach let the colossal parent levels above
        // you shadow EVERYTHING deep down, leaving interiors flat fill-lit;
        // the colossi get their depth from the haze instead.
        sh = softShadow(p + n * eps * 3.0, uSunDir, eps * 8.0, max(12.0 / uFogMul, t * 0.6), 9.0, coneEps, uQuality == 1 ? 28 : 64);
      }
      vec3 hv = normalize(uSunDir - rd);
      float spec = pow(max(dot(n, hv), 0.0), 42.0) * 0.45;

      ${global.__SHADER_DEBUG === 'shade'
        ? 'fragColor = vec4(dot(alb, vec3(0.33)), sunDiff * sh, ao, 1.0); return;'
        : ''}
      /* Lighting model: one key light (the biome sun, soft shadows), a
         hemispheric sky fill shaped by AO (dark corners are a feature — the
         fill is what keeps their FORM readable), a faint bounce opposite the
         sun, a small camera headlight, and a haze rim. The sun's tint is
         partly desaturated on surfaces: a strongly tinted sun times a
         complementary albedo just turns every lit face grey.
         ("up" is world-anchored via the sky basis — see uSkyBasis) */
      vec3 sunLit = mix(vec3(dot(uSunCol, vec3(0.333))), uSunCol, uLookA.z);
      vec3 hemi = mix(uFogCol * uLookB.x, uSkyCol, 0.5 + 0.5 * dot(n, up));
      vec3 amb = hemi * uLookA.w * 4.0 * (ao2 * 0.85 + 0.15);
      amb += sunLit * uLookB.y * max(0.0, -dot(n, uSunDir)) * ao;
      col = alb * (amb + sunLit * sunDiff * sh * uLookA.y);
      col += sunLit * spec * sh * sunDiff;
      // headlight: soft camera-attached fill so unlit caves stay readable
      float head = max(dot(n, -rd), 0.0);
      float tf = t * uFogMul; // scale-continuous distance for the falloff
      col += alb * (uLookB.z * head * head * ao / (1.0 + tf * tf * 0.20));
      // rim: grazing surfaces pick up the haze behind them, so dark shapes
      // keep readable silhouettes against the glow instead of merging
      col += hazeColor(rd) * pow(1.0 - head, 4.0) * uLookB.w * (0.3 + 0.7 * ao);

      vec3 dbgBase = col;
      // point lights: POOLS, not floodlights — the clamp is well below sun
      // strength, so a light can never fullbright its surroundings; it reads
      // as a hot pool near the source falling off into darkness
      int lshN = uQuality >= 2 ? 2 : (uQuality >= 1 ? 1 : 0); // lights that cast shadows
      for (int li = 0; li < MAX_LIGHTS; li++) {
        if (li >= uLightN) break;
        vec3 lp = uLightPos[li].xyz;
        float rad = uLightPos[li].w;
        vec3 lv = lp - p;
        float dist2 = dot(lv, lv);
        // fog extinction keeps receding giant suns from stacking overexposure
        float atten = min(uLightCol[li].w / (dist2 + rad * rad), 0.5)
                    * exp(-sqrt(dist2) * 1.2e-4 * uFogMul);
        if (atten < 0.002) continue;
        vec3 ld = lv * inversesqrt(dist2);
        float diff = max(dot(n, ld), 0.0);
        float lsh = 1.0;
        // light shadows are what make the pools cascade across the geometry
        if (li < lshN && diff * atten > 0.02) {
          lsh = softShadow(p + n * eps * 3.0, ld, eps * 8.0, sqrt(dist2), 10.0, coneEps,
                           uQuality >= 2 ? 32 : 20);
        }
        float lspec = pow(max(dot(n, normalize(ld - rd)), 0.0), 30.0) * 0.5;
        col += uLightCol[li].rgb * (diff + lspec) * atten * lsh;
      }

      // emissive veins: accumulated per level in map() with a scale window
      // (veinGlow), so they appear as you approach and never wash or speckle
      ${global.__SHADER_DEBUG === 'noemis' ? '' : `
      col += veins * (0.55 + 0.45 * ao) * uLookC.w;
      `}
      dbgLights = col - dbgBase;
    } else {
      col = skyColor(rd);
      t = tmax;
    }

    // distance fog toward mood color (keeps giant vistas readable). The air
    // is deliberately clear: most of the horizon work is done by the fade
    // below, so distant structure stays visible across most of the range.
    // All densities scale with uFogMul (continuity across rebases).
    /* LOG-DISTANCE haze: equal extinction per DECADE of scale-continuous
       distance u = t*fogMul. Linear fog put a hard ceiling on how big a
       visible structure could be relative to you (everything past a fixed
       multiple of your size was simply gone); per-decade haze lets a wall
       10^6x your size read as a dim colossal silhouette — the sense that the
       world above you is enormous and continues forever. */
    float u = fogT * uFogMul;
    float decades = log(1.0 + u * 0.3) * 0.4342945;
    float fog = 1.0 - exp(-decades * uHaze);
    // step-exhausted rays (grazing geometry) are softened only at range:
    // distant silhouettes lose speckle, near caves stay dark
    if (exhausted) fog = max(fog, clamp((decades - 1.5) * 0.15, 0.0, 0.5));
    // guarantee FULL fog before tmax (in log space), so the farthest vistas
    // dissolve into the haze instead of popping out at the far plane
    fog = max(fog, smoothstep(log(tmax) - 2.3, log(tmax) - 0.08, log(max(fogT, 1e-6))));
    col = mix(col, hazeColor(rd), hit ? fog : fog * 0.25);

    // light inscatter (glowing orbs + volume glow). NOTE: stored intensity
    // grows ~s^2 per rebase to keep surface lighting scale-invariant, but the
    // scatter integral only falls off ~1/distance, so it must be clamped or
    // ancient giant lights white out the frame.
    vec3 glowAcc = vec3(0.0);
    for (int li = 0; li < MAX_LIGHTS; li++) {
      if (li >= uLightN) break;
      float sc = inscatter(ro, rd, uLightPos[li].xyz, fogT);
      float ext = exp(-length(uLightPos[li].xyz) * 1.2e-4 * uFogMul); // fog extinction of the glow
      // tight orb profile: the raw 1/h scatter is too broad and washes the
      // whole frame; sc^2/(1+sc/2) keeps a hot core with negligible skirt
      float orb = uLightCol[li].w * sc * sc * 0.013 / (1.0 + 0.5 * sc);
      glowAcc += uLightCol[li].rgb * min(orb, 0.9) * ext * uGlowAmt;
    }
    // soft-compress the SUM of glows: several overlapping orbs used to stack
    // linearly and blow the frame out to white
    col += glowAcc / (1.0 + 0.55 * dot(glowAcc, vec3(0.333)));
    // sun glare kept subtle when occluded
    col += uSunCol * pow(max(dot(rd, uSunDir), 0.0), 8.0) * 0.03;

    ${global.__SHADER_DEBUG === 'terms2' ? 'fragColor = vec4(dot(dbgLights, vec3(0.33)), dot(col, vec3(0.33)) * 0.5, hit ? 0.2 : 0.8, 1.0); return;' : ''}
    // tonemap: ACES, partly hue-preserving (plain ACES bleaches saturated
    // brights toward white — the "pale mint" look)
    col *= uLookA.x;
    float lumIn = max(dot(col, vec3(0.2126, 0.7152, 0.0722)), 1e-6);
    vec3 tmHue = col * (aces(vec3(lumIn)).x / lumIn);
    col = clamp(mix(aces(col), tmHue, uLookC.y), 0.0, 1.0);
    col = pow(col, vec3(1.0 / 2.2));
    // vignette + dither
    float vig = 1.0 - uLookD.y * dot(uv * 0.62, uv * 0.62);
    col *= vig;
    col += (hash12(gl_FragCoord.xy + fract(uTime) * 61.7) - 0.5) / 255.0;

    fragColor = vec4(col, 1.0);
  }`; };

  /* Accumulation blit: out = mix(prev, src, blend). Used two ways by the
     renderer: blend = 1/n averages a new jittered frame into the running
     accumulation; blend = 1 is a plain copy (display pass). texelFetch —
     the targets are always the exact canvas size, no filtering wanted. */
  const BLIT_FRAG = `#version 300 es
  precision highp float;
  uniform sampler2D uSrc;
  uniform sampler2D uPrev;
  uniform float uBlend;
  out vec4 fragColor;
  void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    fragColor = mix(texelFetch(uPrev, px, 0), texelFetch(uSrc, px, 0), uBlend);
  }`;

  const Shaders = { VERT, FRAG, BLIT_FRAG, LOOK_DEFAULTS };
  global.Shaders = Shaders;
  if (typeof module !== 'undefined' && module.exports) module.exports = Shaders;
})(typeof window !== 'undefined' ? window : globalThis);
