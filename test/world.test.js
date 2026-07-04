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
  if (S > 0.5 && S < 1.5) { // MENGER
    x = f3(Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2]));
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
  } else { // OCTA
    x = f3(Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2]));
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
    });
  }
  const base = WM.SLOTS * 24;
  const outer = [];
  for (let b = 0; b < world.outer.length; b++) {
    const o = base + b * 16;
    outer.push({
      jacCols: [ubo[o], ubo[o + 1], ubo[o + 2], ubo[o + 4], ubo[o + 5], ubo[o + 6], ubo[o + 8], ubo[o + 9], ubo[o + 10]],
      c: [ubo[o + 12], ubo[o + 13], ubo[o + 14]], margin: ubo[o + 15],
    });
  }
  return { slots, outer };
}

/* Emulates the shader's two-phase map(). pRel is camera-relative (frame K). */
function mapF32(world, packed, pRel) {
  const { slots, outer } = packed;
  const B = outer.length;
  const camF = f3(world.camPos[0], world.camPos[1], world.camPos[2]);
  const WCols = V.mToCols(world.W).map(fr);
  let dr = fr(world.WScale);
  let d = B ? fMat(WCols, f3(pRel[0], pRel[1], pRel[2])) : f3(pRel[0], pRel[1], pRel[2]);
  let x = null; // set when escaped during the outer phase
  for (let b = 0; b < B; b++) {
    if (fLen(d) < outer[b].margin) {
      d = fMat(outer[b].jacCols, d);
      dr = fr(dr * slots[b].scale);
    } else {
      const xn = foldF32(fAdd(outer[b].c, d), slots[b]);
      dr = fr(dr * slots[b].scale);
      if (xn[0] * xn[0] + xn[1] * xn[1] + xn[2] * xn[2] > WM.ESCAPE2) { x = xn; break; }
      const cNext = b + 1 < B ? outer[b + 1].c : camF;
      d = fSub(xn, cNext);
    }
  }
  if (!x) {
    x = fAdd(camF, d);
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
  world.descend();
  let outliers = 0;
  for (let i = 0; i < 24; i++) {
    const p2 = WM.evalFold(pts[i], P).x;
    const de2 = world.de(p2) / P.scale;
    const err = Math.abs(de2 - des[i]);
    const rDist = V.dist(pts[i], camBefore);
    // Exact invariance holds unless the point crosses fold boundaries at two
    // different outer levels at once (blind inverse transport bakes in a
    // mirror the forward folds can't collapse — see world.js header). That
    // known case displaces only structure near old fold planes. Tiering
    // (relative to r + |DE|, since points at the camera's surface-distance
    // scale can be displaced by up to their boundary distance):
    //   minor  < 10%  — unlimited; below the visibility of a rebase seam
    //   major  < 50%  — at most a handful per sample set; occurs when the
    //                   camera sits in a wedge corner with several fold
    //                   boundaries close (empirical: up to 6/24 at shallow K
    //                   where the window necessarily spans every level, and
    //                   after the HORIZON=1.35e5 window extension). More or
    //                   bigger than this is a regression.
    const rel = err / (rDist + Math.abs(des[i]));
    const exact = err < 2e-4 * Math.max(Math.abs(des[i]), 1e-9) || rel < 2e-5;
    if (!exact && rel >= 0.1) outliers++;
    check(exact || rel < 0.5,
      `${tag} DE invariance pt${i}: before=${des[i]} after=${de2} err=${err} r=${rDist}`);
  }
  check(outliers <= 8, `${tag} too many major double-crossing outliers: ${outliers}/24`);
  world.ascend();
  const err = V.dist(world.camPos, camBefore);
  check(err < 1e-10, `${tag} ascend∘descend identity err=${err}`);
}

/* The near-field fast path (world.RLin, mirrored by mapDE in the shader)
   must agree with the full outer walk everywhere inside RLin: within that
   radius every level takes its linear branch and the branches compose to
   the identity, so the fast path (skip the outer phase) is EXACT — any
   disagreement beyond the slow path's own rounding noise is a bug in the
   RLin margin computation. */
function checkFastPath(world, tag) {
  if (!isFinite(world.RLin) || world.RLin <= 0) return;
  for (let i = 0; i < 60; i++) {
    const r = world.RLin * 0.999 * Math.pow(10, -3 * rnd());
    const p = V.add(world.camPos, V.scale(rndDir(), r));
    const fast = world.de(p);
    const slow = world.de(p, true);
    const err = Math.abs(fast - slow);
    // the SLOW path carries the noise here: W is a product of ~15 matrix
    // inversions and the walk folds O(1) orbit numbers ~15 times (observed
    // ~3e-7 relative at deep K). A genuine RLin bug errs at the scale of r
    // or the DE itself, 5+ orders larger than this tolerance.
    const tol = 3e-6 * (Math.abs(slow) + r) + 1e-10;
    check(err < tol, `${tag} fast-path DE r=${r.toExponential(2)} fast=${fast} slow=${slow} err=${err}`);
  }
}

function checkF32Pipeline(world, tag) {
  const ubo = new Float32Array((WM.SLOTS * 6 + WM.B_HARD * 4) * 4);
  world.packUBO(ubo);
  const packed = readSlots(world, ubo);
  let worst = { rel: 0 };
  for (let i = 0; i < 250; i++) {
    // log-uniform radius from very near (1e-4) to vista distance (2e3)
    const r = Math.pow(10, -4 + rnd() * 7.3);
    const rel3 = V.scale(rndDir(), r);
    const de64 = world.de(V.add(world.camPos, rel3));
    const de32 = mapF32(world, packed, rel3);
    // Tolerance: absolute error allowed grows with distance from camera
    // (far geometry only needs sub-angular accuracy; a pixel is ~1e-3 rad).
    // The 2.5e-4*r coefficient is calibrated for ICOSA: unlike the other
    // styles, whose folds are EXACT in f32 (abs/swap/negate), its Householder
    // fold rounds, so f32/f64 escape iterations diverge more often at range.
    // Still ~5x sub-pixel at the default FOV.
    const tol = Math.max(3e-4 * Math.abs(de64), 5e-6 + 2.5e-4 * r, 1e-7);
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
    if (bestP) world.camPos = bestP;
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
  const missed = depthChecks.filter((k) => !done.has(k));
  check(missed.length === 0, `dive reached all depths (missed: ${missed}) steps=${steps}`);
}

dive(1337, [1, 3, 6, 10, 16, 25, 40, 60]);
dive(777, [2, 8, 20, 50]);
// A pathologically deep dive: zoom ~ 10^56. Nothing should degrade.
dive(42, [150]);

if (failures) { console.error(`\n${failures} FAILURES`); process.exit(1); }
console.log('\nALL TESTS PASSED');
