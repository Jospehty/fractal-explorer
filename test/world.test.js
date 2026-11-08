/* world.test.js — numerical validation of the infinite-zoom architecture.
   Run: node test/world.test.js

   Verifies, at many depths along a simulated dive:
     1. chain consistency:   Fold_j(c_j) == c_{j+1} for the whole outer window
     2. rebase invariance:   DE is unchanged (up to frame scale) across descend
     3. ascend∘descend == id
     4. the float32 SHADER pipeline (emulated here with Math.fround, consuming
        the exact packed UBO bytes) matches the float64 reference DE — this is
        the test that the margin/linear-branch scheme really preserves precision
        at unbounded depth. */
'use strict';
const V = require('../js/vec.js');
const LG = require('../js/levelgen.js');
const WM = require('../js/world.js');

const fr = Math.fround;
let failures = 0;
function check(cond, msg) {
  if (!cond) { failures++; console.error('  FAIL:', msg); }
}

/* ---------- float32 emulation of the GLSL map() ---------- */
function f3(x, y, z) { return [fr(x), fr(y), fr(z)]; }
function fAdd(a, b) { return f3(a[0] + b[0], a[1] + b[1], a[2] + b[2]); }
function fSub(a, b) { return f3(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function fMat(cols, v) { // cols: 9 floats column-major, like GLSL mat3
  return f3(
    fr(fr(cols[0] * v[0]) + fr(fr(cols[3] * v[1]) + fr(cols[6] * v[2]))),
    fr(fr(cols[1] * v[0]) + fr(fr(cols[4] * v[1]) + fr(cols[7] * v[2]))),
    fr(fr(cols[2] * v[0]) + fr(fr(cols[5] * v[1]) + fr(cols[8] * v[2])))
  );
}
function fLen(a) { return fr(Math.hypot(a[0], a[1], a[2])); }

// GLSL float32 image of the icosahedral mirror normal (matches the literal
// injected into the shader by DEFS in shaders.js)
const ICO_NF = WM.ICO_N.map((v) => fr(parseFloat(v.toFixed(10))));

function foldF32(x, slot) {
  // slot: {rotCols[9], scale, style, foldL, trans[3]}
  x = fMat(slot.rotCols, x);
  const S = slot.style;
  const fo = slot.foldOff;
  if (S > 0.5 && S < 1.5) { // MENGER: offset abs + sort desc
    x = f3(fo[0] + Math.abs(fr(x[0] - fo[0])), fo[1] + Math.abs(fr(x[1] - fo[1])), fo[2] + Math.abs(fr(x[2] - fo[2])));
    if (x[0] < x[1]) { const t = x[0]; x[0] = x[1]; x[1] = t; }
    if (x[0] < x[2]) { const t = x[0]; x[0] = x[2]; x[2] = t; }
    if (x[1] < x[2]) { const t = x[1]; x[1] = x[2]; x[2] = t; }
  } else if (S < 0.5) { // POLY
    if (fr(x[0] + x[1]) < 0) { const t = fr(-x[0]); x[0] = fr(-x[1]); x[1] = t; }
    if (fr(x[0] + x[2]) < 0) { const t = fr(-x[0]); x[0] = fr(-x[2]); x[2] = t; }
    if (fr(x[1] + x[2]) < 0) { const t = fr(-x[1]); x[1] = fr(-x[2]); x[2] = t; }
  } else if (S > 2.5) { // ICOSA: 2x (abs + golden plane fold), unconditional form
    for (let round = 0; round < 2; round++) {
      x = f3(Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2]));
      const dt = fr(fr(x[0] * ICO_NF[0]) + fr(fr(x[1] * ICO_NF[1]) + fr(x[2] * ICO_NF[2])));
      const s2 = fr(2 * Math.min(0, dt));
      x = f3(x[0] - fr(s2 * ICO_NF[0]), x[1] - fr(s2 * ICO_NF[1]), x[2] - fr(s2 * ICO_NF[2]));
    }
  } else { // OCTA: offset abs + partial sort
    x = f3(fo[0] + Math.abs(fr(x[0] - fo[0])), fo[1] + Math.abs(fr(x[1] - fo[1])), fo[2] + Math.abs(fr(x[2] - fo[2])));
    if (x[0] < x[1]) { const t = x[0]; x[0] = x[1]; x[1] = t; }
    if (x[1] < x[2]) { const t = x[1]; x[1] = x[2]; x[2] = t; }
  }
  return f3(
    fr(x[0] * slot.scale) + slot.trans[0],
    fr(x[1] * slot.scale) + slot.trans[1],
    fr(x[2] * slot.scale) + slot.trans[2]
  );
}

function readSlots(world, ubo) {
  const slots = [];
  for (let i = 0; i < WM.SLOTS; i++) {
    const o = i * 24;
    slots.push({
      rotCols: [ubo[o], ubo[o + 1], ubo[o + 2], ubo[o + 4], ubo[o + 5], ubo[o + 6], ubo[o + 8], ubo[o + 9], ubo[o + 10]],
      scale: ubo[o + 3], style: ubo[o + 7], foldL: ubo[o + 11],
      trans: [ubo[o + 12], ubo[o + 13], ubo[o + 14]],
      foldOff: [ubo[o + 20], ubo[o + 21], ubo[o + 22]],
    });
  }
  const base = WM.SLOTS * 24;
  const outer = [];
  for (let b = 0; b < world.outer.length; b++) {
    const o = base + b * WM.OUTER_VEC4 * 4;
    outer.push({
      jacCols: [ubo[o], ubo[o + 1], ubo[o + 2], ubo[o + 4], ubo[o + 5], ubo[o + 6], ubo[o + 8], ubo[o + 9], ubo[o + 10]],
      c: [ubo[o + 12], ubo[o + 13], ubo[o + 14]], margin: ubo[o + 15],
      st: Array.from(ubo.slice(o + 16, o + 16 + WM.MAX_STAGES)),
      R: ubo[o + 3], drAfter: ubo[o + 7],
      qCols: [ubo[o + 24], ubo[o + 25], ubo[o + 26], ubo[o + 28], ubo[o + 29], ubo[o + 30], ubo[o + 32], ubo[o + 33], ubo[o + 34]],
    });
  }
  return { slots, outer };
}

/* float32 emulation of GLSL foldDelta (perturbative outer fold) */
function foldDeltaF32(d, slot, st) {
  d = fMat(slot.rotCols, d);
  d = d.slice();
  const absOp = (i, a) => {
    const v = fr(a + d[i]);
    if (a >= 0) { if (v < 0) d[i] = fr(fr(-2 * a) - d[i]); }
    else d[i] = v < 0 ? -d[i] : fr(fr(2 * a) + d[i]);
  };
  const swapOp = (p, q, D) => {
    const ref = D < 0, pix = fr(fr(D + d[p]) - d[q]) < 0;
    const dp = d[p], dq = d[q];
    if (ref && pix) { d[p] = dq; d[q] = dp; }
    else if (pix) { d[p] = fr(-D + dq); d[q] = fr(D + dp); }
    else if (ref) { d[p] = fr(D + dp); d[q] = fr(-D + dq); }
  };
  const flipOp = (a, b, Sg) => {
    const ref = Sg < 0, pix = fr(fr(Sg + d[a]) + d[b]) < 0;
    const da = d[a], db = d[b];
    if (ref && pix) { d[a] = -db; d[b] = -da; }
    else if (pix) { d[a] = fr(-Sg - db); d[b] = fr(-Sg - da); }
    else if (ref) { d[a] = fr(Sg + da); d[b] = fr(Sg + db); }
  };
  const goldOp = (Tn) => {
    const dn = fr(fr(d[0] * ICO_NF[0]) + fr(fr(d[1] * ICO_NF[1]) + fr(d[2] * ICO_NF[2])));
    const ref = Tn < 0, pix = fr(Tn + dn) < 0;
    const f = (ref && pix) ? fr(-2 * dn) : (pix ? fr(-2 * fr(Tn + dn)) : (ref ? fr(2 * Tn) : 0));
    d = f3(d[0] + fr(f * ICO_NF[0]), d[1] + fr(f * ICO_NF[1]), d[2] + fr(f * ICO_NF[2]));
  };
  const S = slot.style;
  if (S < 0.5) { flipOp(0, 1, st[0]); flipOp(0, 2, st[1]); flipOp(1, 2, st[2]); }
  else if (S < 2.5) {
    absOp(0, st[0]); absOp(1, st[1]); absOp(2, st[2]);
    swapOp(0, 1, st[3]);
    if (S < 1.5) { swapOp(0, 2, st[4]); swapOp(1, 2, st[5]); } else swapOp(1, 2, st[4]);
  } else {
    absOp(0, st[0]); absOp(1, st[1]); absOp(2, st[2]); goldOp(st[3]);
    absOp(0, st[4]); absOp(1, st[5]); absOp(2, st[6]); goldOp(st[7]);
  }
  return f3(d[0] * slot.scale, d[1] * slot.scale, d[2] * slot.scale);
}

/* Emulates the shader's two-phase map(). pRel is camera-relative (frame K). */
function mapF32(world, packed, pRel) {
  const { slots, outer } = packed;
  const B = outer.length;
  const camF = f3(world.camPos[0], world.camPos[1], world.camPos[2]);
  const WCols = V.mToCols(world.W).map(fr);
  const pf = f3(pRel[0], pRel[1], pRel[2]);
  // per-level skip, exactly as mapDE
  const pp = fr(fr(pf[0] * pf[0]) + fr(pf[1] * pf[1]) + fr(pf[2] * pf[2]));
  let b0 = 0;
  while (b0 < B && pp < fr(outer[b0].R * outer[b0].R)) b0++;
  if (B && b0 === B) {
    let x = fAdd(camF, pf), dr = 1;
    for (let i = 0; i < WM.N_INNER; i++) {
      x = foldF32(x, slots[B + i]);
      dr = fr(dr * slots[B + i].scale);
      if (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] > WM.ESCAPE2) break;
    }
    return fr((fLen(x) - fr(world.bound)) / dr);
  }
  let dr = b0 === 0 ? fr(world.WScale) : outer[b0 - 1].drAfter;
  let d = !B ? pf : (b0 === 0 ? fMat(WCols, pf) : fMat(outer[b0 - 1].qCols, pf));
  // mirrors the OUTER template in shaders.js exactly
  let x = null;
  let bAbs = -1, esc = false;
  for (let b = b0; b < B; b++) {
    const dd = fr(fr(d[0] * d[0]) + fr(d[1] * d[1]) + fr(d[2] * d[2]));
    dr = fr(dr * slots[b].scale);
    const cNext = b + 1 < B ? outer[b + 1].c : camF;
    if (dd < fr(outer[b].margin * outer[b].margin)) {
      d = fMat(outer[b].jacCols, d);
    } else if (dd > fr(WM.ABS_REL2 * Math.max(fr(fr(outer[b].c[0] * outer[b].c[0]) + fr(fr(outer[b].c[1] * outer[b].c[1]) + fr(outer[b].c[2] * outer[b].c[2]))), 1))) {
      x = foldF32(fAdd(outer[b].c, d), slots[b]);
      bAbs = b;
      if (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] > WM.ESCAPE2) esc = true;
      break;
    } else {
      d = foldDeltaF32(d, slots[b], outer[b].st);
      const xe = fAdd(cNext, d);
      if (xe[0] * xe[0] + xe[1] * xe[1] + xe[2] * xe[2] > WM.ESCAPE2) { x = xe; esc = true; break; }
    }
  }
  if (!esc && bAbs >= 0) {
    for (let b = bAbs + 1; b < B; b++) {
      x = foldF32(x, slots[b]);
      dr = fr(dr * slots[b].scale);
      if (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] > WM.ESCAPE2) { esc = true; break; }
    }
  } else if (!esc) {
    x = fAdd(camF, d);
  }
  if (!esc) {
    for (let i = 0; i < WM.N_INNER; i++) {
      x = foldF32(x, slots[B + i]);
      dr = fr(dr * slots[B + i].scale);
      if (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] > WM.ESCAPE2) break;
    }
  }
  return fr((fLen(x) - fr(world.bound)) / dr);
}

/* ---------- deterministic RNG for test points ---------- */
let rs = 12345;
function rnd() { rs = (Math.imul(rs, 1103515245) + 12345) >>> 0; return rs / 4294967296; }
function rndDir() { return V.norm([rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1]); }

/* ---------- checks at one depth ---------- */
function checkChain(world, tag) {
  const O = world.outer;
  for (let b = 0; b < O.length; b++) {
    const next = b + 1 < O.length ? O[b + 1].c : world.camPos;
    const r = WM.evalFold(O[b].c, world.params(O[b].level));
    const err = V.dist(r.x, next);
    check(err < 1e-8, `${tag} chain fold(c_${b}) -> c_${b + 1} err=${err}`);
    check(O[b].margin >= 0 && isFinite(O[b].margin), `${tag} margin finite`);
  }
}

function checkRebaseInvariance(world, tag) {
  // Sample near points, record DE, force-descend, compare, ascend back.
  const pts = [], des = [];
  for (let i = 0; i < 24; i++) {
    const r = 0.003 + rnd() * 0.4;
    pts.push(V.add(world.camPos, V.scale(rndDir(), r)));
    des.push(world.de(pts[i]));
  }
  const P = world.params(world.K);
  const camBefore = V.clone(world.camPos);
  let truncScale = world.bound;
  for (let i = 0; i < WM.N_INNER; i++) truncScale /= world.params(world.K + i).scale;
  // The rendered frame-(K+1) world is the frame-K world seen through the
  // camera-branch AFFINE of Fold_K (the same map descend() applies to lights).
  // Transporting with the true fold instead would mirror points lying across
  // a level-K plane — a different world location, not a rebase error.
  const rA = WM.evalFold(camBefore, P, true);
  const bA = V.sub(rA.x, V.mMulV(rA.M, camBefore));
  world.descend();
  for (let i = 0; i < 24; i++) {
    const p2 = V.add(V.mMulV(rA.M, pts[i]), bA);
    const de2 = world.de(p2) / P.scale;
    const err = Math.abs(de2 - des[i]);
    const rDist = V.dist(pts[i], camBefore);
    // EXACT up to (a) float64 noise of the composed window matrices and
    // (b) the CPU reference's fixed N_INNER truncation, which reaches one
    // level deeper after the rebase (detail at scale bound/Π s_inner). The
    // dropped outermost level lies beyond the horizon by construction.
    const tol = 1e-7 * (rDist + Math.abs(des[i])) + 2 * truncScale;
    check(err < tol,
      `${tag} DE invariance pt${i}: before=${des[i]} after=${de2} err=${err} r=${rDist}`);
  }
  world.ascend();
  const err = V.dist(world.camPos, camBefore);
  check(err < 1e-10, `${tag} ascend∘descend identity err=${err}`);
}

/* Per-level skipping (world.de / mapDE): a sample within R_b of the camera
   jumps past levels 0..b with the precomposed linear map Q_b (and inside
   RLin = R_{B-1} the whole outer phase vanishes: the maps compose to the
   identity). This must agree with the unskipped walk at EVERY radius, up to
   the walk's own rounding noise (W is a product of up to ~48 inversions).
   A genuine R_b/Q_b bug errs at the scale of r or the DE itself. */
function checkFastPath(world, tag) {
  for (let i = 0; i < 80; i++) {
    const r = Math.pow(10, -4 + rnd() * 10);
    const p = V.add(world.camPos, V.scale(rndDir(), r));
    const fast = world.de(p);
    const slow = world.de(p, true);
    const err = Math.abs(fast - slow);
    const tol = 3e-6 * (Math.abs(slow) + r) + 1e-10;
    check(err < tol, `${tag} skip-path DE r=${r.toExponential(2)} fast=${fast} slow=${slow} err=${err}`);
  }
}

function checkF32Pipeline(world, tag) {
  const ubo = new Float32Array((WM.SLOTS * 6 + WM.B_HARD * WM.OUTER_VEC4) * 4);
  world.packUBO(ubo);
  const packed = readSlots(world, ubo);
  let worst = { rel: 0 };
  for (let i = 0; i < 250; i++) {
    // log-uniform radius from very near (1e-4) to the deep horizon (1e10)
    const r = Math.pow(10, -4 + rnd() * 14);
    const rel3 = V.scale(rndDir(), r);
    const de64 = world.de(V.add(world.camPos, rel3));
    const de32 = mapF32(world, packed, rel3);
    // Tolerance: absolute error allowed grows with distance from camera
    // (far geometry only needs sub-angular accuracy; a pixel is ~1e-3 rad).
    // With the perturbative outer phase (foldDelta) every sample keeps full
    // RELATIVE precision, so this is 10x tighter than the old absolute-fold
    // pipeline could meet (which failed outright near outer fold creases).
    const tol = Math.max(3e-5 * Math.abs(de64), 5e-7 + 2.5e-5 * r, 1e-8);
    const err = Math.abs(de32 - de64);
    if (err / tol > worst.rel) worst = { rel: err / tol, r, de64, de32 };
    check(err < tol, `${tag} f32-vs-f64 r=${r.toExponential(2)} de64=${de64} de32=${de32} err=${err}`);
  }
  console.log(`  ${tag} f32 worst err/tol = ${worst.rel.toFixed(3)} (r=${worst.r && worst.r.toExponential(1)})`);
}

/* ---------- dive driver ---------- */
function dive(seed, depthChecks) {
  console.log(`\n=== dive seed=${seed} ===`);
  const world = new WM.World(seed);
  // walk toward the fractal, then keep pressing toward the surface
  const done = new Set();
  const maxDepth = Math.max(...depthChecks);
  let steps = 0;
  while (world.K <= maxDepth && steps++ < 20000) {
    world.updateChain();
    const d = world.de(world.camPos);
    if (d < 0) {
      // inside geometry (shouldn't happen with DE-clamped steps, but recover)
      world.camPos = V.add(world.camPos, V.scale(world.grad(world.camPos), -d * 2 + 1e-4));
      continue;
    }
    // dive by probing: move toward whichever direction minimizes sampled DE.
    // Mix of short and long probes so flat wall pockets can be escaped
    // (ICOSA terrain has larger flat pockets than the angular styles, hence
    // the extra long-range probes).
    let bestP = null, bestD = Infinity;
    for (let c = 0; c < 18; c++) {
      const radius = d * (c < 8 ? 0.6 : 1.5 + rnd() * 10);
      const cand = V.add(world.camPos, V.scale(rndDir(), radius));
      const cd = world.de(cand);
      if (cd >= 0 && cd < bestD) { bestD = cd; bestP = cand; }
    }
    if (bestP) world.moveCamera(V.sub(bestP, world.camPos));
    world.rebaseToCamera();
    const due = depthChecks.filter((k) => k <= world.K && !done.has(k));
    for (const k of due) {
      done.add(k);
      const tag = `K=${world.K} zoom=1e${world.log10Zoom.toFixed(1)}`;
      console.log(`-- checking at ${tag} (steps=${steps})`);
      world.updateChain();
      checkChain(world, tag);
      checkFastPath(world, tag);
      checkRebaseInvariance(world, tag);
      world.updateChain();
      checkF32Pipeline(world, tag);
      break; // one check per step; remaining depths handled as K grows
    }
  }
  check(world.chainFailures === 0, `chain wedge normalization failed ${world.chainFailures}x`);
  // a correction reflects across ONE crease the camera just crossed within
  // its sub-step, so it can never exceed 2x that step (maxCorrection records
  // the excess over the bound)
  check(world.maxCorrection < 1e-6, `wedge correction exceeded the reflection bound by ${world.maxCorrection}`);
  const missed = depthChecks.filter((k) => !done.has(k));
  check(missed.length === 0, `dive reached all depths (missed: ${missed}) steps=${steps}`);
}

/* Stress: many seeds (all styles, biome transitions), chain-continuity
   invariants only — this is where "sometimes broken" lived. */
function stress(seeds, depth) {
  console.log(`\n=== stress: ${seeds.length} seeds to K=${depth} ===`);
  for (const seed of seeds) {
    const world = new WM.World(seed);
    let steps = 0;
    while (world.K < depth && steps++ < 4000) {
      const d = world.de(world.camPos);
      let bestP = null, bestD = Infinity;
      for (let c = 0; c < 12; c++) {
        const cand = V.add(world.camPos, V.scale(rndDir(), d * (c < 6 ? 0.6 : 1.5 + rnd() * rnd() * 40)));
        const cd = world.de(cand);
        if (cd >= 0 && cd < bestD) { bestD = cd; bestP = cand; }
      }
      if (bestP) world.moveCamera(V.sub(bestP, world.camPos));
      world.rebaseToCamera();
      if (rnd() < 0.05 && world.K > 2) { world.ascend(); world.ascend(); } // zoom back out sometimes
    }
    check(world.K >= depth, `stress seed ${seed} stalled at K=${world.K}`);
    check(world.chainFailures === 0, `stress seed ${seed}: ${world.chainFailures} chain failures`);
    check(world.maxCorrection < 1e-6, `stress seed ${seed}: correction excess ${world.maxCorrection}`);
    check(Number.isFinite(world.de(world.camPos)), `stress seed ${seed}: non-finite camera DE`);
  }
}
stress([...Array(40)].map((_, i) => 1000 + i * 7919), 45);

dive(1337, [1, 3, 6, 10, 16, 25, 40, 60]);
dive(777, [2, 8, 20, 50]);
// A pathologically deep dive: zoom ~ 10^56. Nothing should degrade.
dive(42, [150]);

if (failures) { console.error(`\n${failures} FAILURES`); process.exit(1); }
console.log('\nALL TESTS PASSED');
