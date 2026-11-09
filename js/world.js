/* world.js — the deep-zoom world model.
   ============================================================================
   THE INFINITE-ZOOM ARCHITECTURE
   ----------------------------------------------------------------------------
   The fractal is a kaleidoscopic IFS where every absolute level j has its own
   parameters (levelgen.js). One "iteration" maps frame-j coordinates to
   frame-(j+1) coordinates:        x_{j+1} = Fold_j(x_j)
   where Fold_j is a piecewise reflection group fold (rotation, abs/plane/box
   folds) followed by a uniform scale s_j and a translation. Level-j features
   have size O(1) in frame j. Every fold boundary is a REFLECTION, and each
   fold plane is an exact mirror symmetry of all geometry generated at that
   level and deeper (mirrored points have identical forward orbits). This
   property is what makes everything below seamless; do not add fold ops with
   translation boundaries (e.g. the classic Menger z-wrap) — they break it.

   The camera lives in frame K ("the rebase level"). When it nears the surface
   we REBASE: push the camera through Fold_K (float64, exact) and K++. Camera
   numbers stay O(1) forever: no lower bound on depth. Ascending inverts this
   via fold preimages.

   Outer levels (the giant vistas): the shader first runs a window of B outer
   levels K-B..K-1. Per frame we maintain in float64 the camera "chain":
     c_j    = camera position expressed in frame j
     M_j    = exact linear action (Jacobian) of Fold_j at c_j
     m_j    = distance from c_j to the nearest fold-branch boundary
     st_j   = c_j's branch DECISION scalars (one per fold op, see foldDelta)
   A ray point p (camera-relative, frame K) starts as d = W·p in frame K-B
   (W = product of inverse linear maps — a pure similarity, so tiny d keeps
   full *relative* float32 precision). Per outer level:
     |d| < m_j           → d ← M_j·d               (exact linear branch)
     |d| < 4e-3·|c_j|    → d ← foldDelta(d)        (exact PERTURBATIVE fold:
                            Fold(c+d) - Fold(c) from d and st_j alone — the
                            deep-zoom Mandelbrot trick; never forms c + d)
     else                → x = c_j + d, plain folds from here on (d is large,
                            so f32 rounding of c_j + d is small relative to it)
   No sample loses relative precision however many levels separate it from
   the camera, which is what lets the window span 48 levels (~1e11 horizon).
   Samples within R_b of the camera skip levels 0..b with one precomposed
   matrix Q_b (see _tryChain); within RLin = R_{B-1} the outer phase vanishes.
   After the outer window, x = c_K + d is a true frame-K coordinate and inner
   levels K..K+N run the plain fold pipeline on O(1) numbers.

   Chain maintenance is BRANCH CONTINUATION: c_j = A_j^-1(c_{j+1}) through
   last frame's affine at level j. If Fold_j(c_j) no longer equals c_{j+1},
   the camera crossed a fold crease of level j; its true frame-(j+1) position
   is Fold_j(c_j), so we re-chart with T = A_new·A_old^-1 — by continuity of
   the fold this is the mirror across the crossed crease, an exact local
   symmetry of the level-j world — lift T through the chain to frame K, and
   apply it to the camera, lights and sky. T moves the camera by at most 2x
   its step, and World.moveCamera sub-steps large moves so creases are
   crossed one at a time. (Preimage ENUMERATION is only used when no chart
   exists yet: POLY/OCTA fold images overlap themselves, so "some valid
   preimage" can lie in a far region — that used to teleport the camera.)
   ============================================================================ */
(function (global) {
  'use strict';
  const V = global.V || require('./vec.js');
  const LG = global.LevelGen || require('./levelgen.js');

  /* Outer window sizing: a level j may only be dropped from the window when the
     nearest crossing of its fold boundaries is beyond the fog horizon (in
     camera units, margin_j × Π s). Otherwise dropping it would visibly change
     geometry near the camera. The window is therefore dynamic: at least B_MIN
     levels for vistas, growing up to B_HARD when the camera lingers near an
     old fold plane. */
  const B_MIN = 8;
  const B_HARD = 48;
  /* Coupled to the renderer: the shader marches to tmax = TMAX/fogMul with
     TMAX = 1e11 and fogMul >= 0.85, and guarantees FULL fog by ~0.92*tmax
     (log-space horizon fade in shaders.js) — so the farthest visible geometry
     is at 0.92 * 1e11 / 0.85 = 1.08e11 <= 2*HORIZON, and dropping an outer
     level is never visible. A horizon this deep (structures ~10^10x your size
     stay visible) is only possible because the outer phase is perturbative
     (foldDelta): no sample ever loses relative precision, however many
     levels of magnification separate it from the camera. B_HARD = 48 levels
     covers it for the smallest scale (1.78^48 ~ 1e12). */
  const HORIZON = 6e10;
  const N_INNER = 16;     // inner detail levels
  const SLOTS = B_HARD + N_INNER;
  const ESCAPE2 = 900.0;  // escape radius^2 for the orbit
  /* Outer-phase precision switch. Absolute evaluation of c + d in f32 has
     relative error ~6e-8·|c|/|d| in the delta; once |d| > 4e-3·|c| that is
     <1.5e-5 (far below a pixel), so the cheaper absolute fold takes over.
     The perturbative fold (foldDelta) is thus only spent where it matters:
     samples just past a fold crease close to the camera. Squared ratio: */
  const ABS_REL2 = 1.6e-5;
  const DESCEND_DE = 0.02;
  const MAX_LIGHTS_STORED = 14;
  const MAX_LIGHTS_GPU = 6;

  const SQ2 = Math.SQRT2;

  /* ICOSA style: the H3 (icosahedral) kaleidoscope. Mirrors: the three
     coordinate planes plus the golden plane with normal ICO_N. All ops are
     pure reflections from a FINITE Coxeter group (order 120), which is what
     the rebase architecture requires (see header).
     ORIENTATION MATTERS: for the icosahedron with vertices at the cyclic
     permutations of (0, ±1, ±φ) — the orientation whose mirrors include the
     coordinate planes — the golden mirror normals are the ODD permutations
     of (±1, ±φ, ±1/φ)/2. The cyclic ones belong to the reflected dual: they
     generate an INFINITE group together with the coordinate mirrors, which
     breaks preimages and wedge normalization (symptom: "wedge normalization
     did not settle"). Verified: reflecting vertex (0,1,φ) in ICO_N gives
     vertex (1,-φ,0). |((1-φ), φ, 1)| = 2 exactly, so ICO_N is exact in both
     f32 and f64. */
  const PHI = (1 + Math.sqrt(5)) / 2;
  const ICO_N = [(1 - PHI) / 2, PHI / 2, 0.5];
  const ICO_HOUSE = (() => { // I - 2 n nᵀ, row-major
    const n = ICO_N;
    return [
      1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2],
      -2 * n[0] * n[1], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2],
      -2 * n[0] * n[2], -2 * n[1] * n[2], 1 - 2 * n[2] * n[2],
    ];
  })();

  /* DIHEDRAL style helpers: mirror N1 = y=0 plane; N2 (per level, from the
     order n) is uploaded to the GPU in the foldOff slot (see packUBO). */
  const DIH_N1 = [0, 1, 0];
  function dihedralOps(n) { return n % 2 ? n : n + 1; } // <= 7 (MAX_STAGES - 1)
  // BRANCH composite levels: sub-steps per level and the per-sub-step scale
  // (level scale = subS^subM). Other styles: one step of the level scale.
  function subM(P) { return P.subM || 1; }
  function subS(P) { return P.subM > 1 ? P.subS : P.scale; }
  function house(n) { // I - 2 n nᵀ, row-major
    return [
      1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2],
      -2 * n[0] * n[1], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2],
      -2 * n[0] * n[2], -2 * n[1] * n[2], 1 - 2 * n[2] * n[2],
    ];
  }

  /* ---- the fold: float64 reference implementation ------------------------
     rec=true records {M (linear action, row-major), margin (distance from the
     input point to the nearest branch boundary, in level-input units)}.
     MUST stay structurally identical to foldLevel() in shaders.js.          */
  function evalFold(xin, P, rec) {
    let x = V.mMulV(P.rot, xin);
    let M = rec ? P.rot.slice() : null;
    let margin = Infinity;
    const S = P.style;
    // st: the reference DECISION SCALARS of every fold op, in order (see
    // foldDelta). Their signs are this point's branch choices; with them, any
    // nearby point's fold can be evaluated in exact delta form.
    const st = rec ? [] : null;

    // Offset fold planes (MENGER/OCTA): the abs fold reflects across x_i=o_i
    // instead of the coordinate planes. Still a pure reflection per op — the
    // rebase machinery is untouched — but the copies no longer meet corner-on
    // -corner, which breaks the monotonous perfect self-similarity (this is
    // the Mandelbox box-fold trick). POLY/ICOSA keep concurrent planes: their
    // preimages enumerate a group orbit, which offsets would make infinite.
    const FO = P.foldOff || [0, 0, 0];
    if (S === LG.STYLE_MENGER) {
      if (rec) {
        st.push(x[0] - FO[0], x[1] - FO[1], x[2] - FO[2]);
        margin = Math.min(margin,
          Math.abs(x[0] - FO[0]), Math.abs(x[1] - FO[1]), Math.abs(x[2] - FO[2]));
        M = V.mMul([Math.sign(x[0] - FO[0]) || 1, 0, 0, 0, Math.sign(x[1] - FO[1]) || 1, 0, 0, 0, Math.sign(x[2] - FO[2]) || 1], M);
      }
      x = [FO[0] + Math.abs(x[0] - FO[0]), FO[1] + Math.abs(x[1] - FO[1]), FO[2] + Math.abs(x[2] - FO[2])];
      if (rec) margin = Math.min(margin,
        Math.abs(x[0] - x[1]) / SQ2, Math.abs(x[0] - x[2]) / SQ2, Math.abs(x[1] - x[2]) / SQ2);
      if (rec) st.push(x[0] - x[1]);
      if (x[0] < x[1]) { const t = x[0]; x[0] = x[1]; x[1] = t; if (rec) M = V.mMul([0,1,0, 1,0,0, 0,0,1], M); }
      if (rec) st.push(x[0] - x[2]);
      if (x[0] < x[2]) { const t = x[0]; x[0] = x[2]; x[2] = t; if (rec) M = V.mMul([0,0,1, 0,1,0, 1,0,0], M); }
      if (rec) st.push(x[1] - x[2]);
      if (x[1] < x[2]) { const t = x[1]; x[1] = x[2]; x[2] = t; if (rec) M = V.mMul([1,0,0, 0,0,1, 0,1,0], M); }
    } else if (S === LG.STYLE_POLY) {
      if (rec) { margin = Math.min(margin, Math.abs(x[0] + x[1]) / SQ2); st.push(x[0] + x[1]); }
      if (x[0] + x[1] < 0) { const t = -x[0]; x[0] = -x[1]; x[1] = t; if (rec) M = V.mMul([0,-1,0, -1,0,0, 0,0,1], M); }
      if (rec) { margin = Math.min(margin, Math.abs(x[0] + x[2]) / SQ2); st.push(x[0] + x[2]); }
      if (x[0] + x[2] < 0) { const t = -x[0]; x[0] = -x[2]; x[2] = t; if (rec) M = V.mMul([0,0,-1, 0,1,0, -1,0,0], M); }
      if (rec) { margin = Math.min(margin, Math.abs(x[1] + x[2]) / SQ2); st.push(x[1] + x[2]); }
      if (x[1] + x[2] < 0) { const t = -x[1]; x[1] = -x[2]; x[2] = t; if (rec) M = V.mMul([1,0,0, 0,0,-1, 0,-1,0], M); }
    } else if (S === LG.STYLE_ICOSA) {
      // H3 kaleidoscope: two rounds of (abs, golden-plane fold)
      for (let round = 0; round < 2; round++) {
        if (rec) {
          st.push(x[0], x[1], x[2]);
          margin = Math.min(margin, Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2]));
          M = V.mMul([Math.sign(x[0]) || 1, 0, 0, 0, Math.sign(x[1]) || 1, 0, 0, 0, Math.sign(x[2]) || 1], M);
        }
        x = [Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2])];
        const t = x[0] * ICO_N[0] + x[1] * ICO_N[1] + x[2] * ICO_N[2];
        if (rec) { margin = Math.min(margin, Math.abs(t)); st.push(t); }
        if (t < 0) {
          x = [x[0] - 2 * t * ICO_N[0], x[1] - 2 * t * ICO_N[1], x[2] - 2 * t * ICO_N[2]];
          if (rec) M = V.mMul(ICO_HOUSE, M);
        }
      }
    } else if (S === LG.STYLE_DIHEDRAL) {
      // DIHEDRAL: z mirror, then the D_n kaleidoscope around the z axis as an
      // alternating sequence of conditional reflections across y=0 (N1) and
      // the plane at angle pi/n (N2). dihedralOps(n) reflections reach the
      // wedge from any angle (each pair narrows it by 2pi/n). Pure
      // reflections of a finite group (order 4n) — rebase-safe.
      if (rec) {
        st.push(x[2]);
        margin = Math.min(margin, Math.abs(x[2]));
        M = V.mMul([1, 0, 0, 0, 1, 0, 0, 0, Math.sign(x[2]) || 1], M);
      }
      x = [x[0], x[1], Math.abs(x[2])];
      const N2 = P.dihN;
      const nOps = dihedralOps(P.foldN);
      for (let k = 0; k < nOps; k++) {
        const N = k % 2 ? N2 : DIH_N1;
        const t = x[0] * N[0] + x[1] * N[1] + x[2] * N[2];
        if (rec) { margin = Math.min(margin, Math.abs(t)); st.push(t); }
        if (t < 0) {
          x = [x[0] - 2 * t * N[0], x[1] - 2 * t * N[1], x[2] - 2 * t * N[2]];
          if (rec) M = V.mMul(house(N), M);
        }
      }
    } else if (S === LG.STYLE_BRANCH) {
      // BRANCH: mirror the first foldN axes (x; or x and y) — 2 or 4 copies.
      // With a strong rotation this is the classic KIFS tree: trunks
      // splitting into ever-finer branches. COMPOSITE level: the step
      // (rot, mirrors, x subS + trans) repeats subM times, so one level
      // scales by subS^subM (>= ~1.6). Thick organic forms need subS ~1.25,
      // which as single levels would leave 16 inner levels with ~50x of
      // detail and a 48-level window with ~1e5 of vista. Still pure
      // reflections + similarities: the rebase machinery is unchanged.
      const m = subM(P), ss = subS(P);
      let mag = 1; // intermediate magnification relative to the level input
      for (let it = 0; it < m; it++) {
        if (it > 0) {
          x = [x[0] * ss + P.trans[0], x[1] * ss + P.trans[1], x[2] * ss + P.trans[2]];
          x = V.mMulV(P.rot, x);
          mag *= ss;
          if (rec) M = V.mMul(P.rot, V.mScale(M, ss));
        }
        for (let i = 0; i < P.foldN; i++) {
          if (rec) {
            st.push(x[i]);
            margin = Math.min(margin, Math.abs(x[i]) / mag);
            const D = [1, 0, 0, 0, 1, 0, 0, 0, 1];
            D[i * 4] = Math.sign(x[i]) || 1;
            M = V.mMul(D, M);
          }
          x = x.slice();
          x[i] = Math.abs(x[i]);
        }
      }
    } else {
      // OCTA: (offset) abs fold followed by the two diagonal plane folds
      if (rec) {
        st.push(x[0] - FO[0], x[1] - FO[1], x[2] - FO[2]);
        margin = Math.min(margin,
          Math.abs(x[0] - FO[0]), Math.abs(x[1] - FO[1]), Math.abs(x[2] - FO[2]));
        M = V.mMul([Math.sign(x[0] - FO[0]) || 1, 0, 0, 0, Math.sign(x[1] - FO[1]) || 1, 0, 0, 0, Math.sign(x[2] - FO[2]) || 1], M);
      }
      x = [FO[0] + Math.abs(x[0] - FO[0]), FO[1] + Math.abs(x[1] - FO[1]), FO[2] + Math.abs(x[2] - FO[2])];
      if (rec) { margin = Math.min(margin, Math.abs(x[0] - x[1]) / SQ2); st.push(x[0] - x[1]); }
      if (x[0] < x[1]) { const t = x[0]; x[0] = x[1]; x[1] = t; if (rec) M = V.mMul([0,1,0, 1,0,0, 0,0,1], M); }
      if (rec) { margin = Math.min(margin, Math.abs(x[1] - x[2]) / SQ2); st.push(x[1] - x[2]); }
      if (x[1] < x[2]) { const t = x[1]; x[1] = x[2]; x[2] = t; if (rec) M = V.mMul([1,0,0, 0,0,1, 0,1,0], M); }
    }

    const sc = S === LG.STYLE_BRANCH ? subS(P) : P.scale; // composite: last sub-step
    x = [x[0] * sc + P.trans[0], x[1] * sc + P.trans[1], x[2] * sc + P.trans[2]];
    if (rec) M = V.mScale(M, sc);
    return rec ? { x, M, margin, st } : { x };
  }

  /* ---- perturbative fold ----------------------------------------------------
     Returns Fold(c + d) - Fold(c) given ONLY the delta d and the reference
     point's decision scalars st (from evalFold(c, P, true)) — never the
     absolute c + d. Every quantity is a difference of nearby values, so the
     result keeps full RELATIVE precision in d no matter how many levels of
     magnification separate this frame from the camera. This is exactly the
     perturbation technique of deep-zoom Mandelbrot renderers, applied to the
     piecewise-reflection folds: each op either takes the reference's branch
     (delta maps linearly) or the other one (a correction built from the
     reference's own signed distance to that op's mirror).
     MUST mirror foldDelta() in shaders.js and the f32 emulation in the tests. */
  function foldDelta(dIn, P, st) {
    const d = V.mMulV(P.rot, dIn);
    const S = P.style;
    let k = 0;
    // abs about an (offset) plane: a = reference distance to that plane
    const absOp = (i) => {
      const a = st[k++];
      const v = a + d[i];
      if (a >= 0) { if (v < 0) d[i] = -2 * a - d[i]; }
      else d[i] = v < 0 ? -d[i] : 2 * a + d[i];
    };
    // "if x_p < x_q swap": D = reference x_p - x_q
    const swapOp = (p, q) => {
      const D = st[k++];
      const ref = D < 0, pix = D + d[p] - d[q] < 0;
      const dp = d[p], dq = d[q];
      if (ref && pix) { d[p] = dq; d[q] = dp; }
      else if (pix) { d[p] = -D + dq; d[q] = D + dp; }
      else if (ref) { d[p] = D + dp; d[q] = -D + dq; }
    };
    // POLY "if x_a + x_b < 0: (x_a, x_b) = (-x_b, -x_a)": Sg = reference x_a + x_b
    const flipOp = (a, b) => {
      const Sg = st[k++];
      const ref = Sg < 0, pix = Sg + d[a] + d[b] < 0;
      const da = d[a], db = d[b];
      if (ref && pix) { d[a] = -db; d[b] = -da; }
      else if (pix) { d[a] = -Sg - db; d[b] = -Sg - da; }
      else if (ref) { d[a] = Sg + da; d[b] = Sg + db; }
    };
    // golden-plane reflection "if x·N < 0: x -= 2(x·N)N": Tn = reference x·N
    const goldOp = () => {
      const Tn = st[k++];
      const dn = d[0] * ICO_N[0] + d[1] * ICO_N[1] + d[2] * ICO_N[2];
      const ref = Tn < 0, pix = Tn + dn < 0;
      let f = 0;
      if (ref && pix) f = -2 * dn;
      else if (pix) f = -2 * (Tn + dn);
      else if (ref) f = 2 * Tn;
      if (f) { d[0] += f * ICO_N[0]; d[1] += f * ICO_N[1]; d[2] += f * ICO_N[2]; }
    };
    // generic conditional plane reflection "if x·N < 0: x -= 2(x·N)N"
    const planeOp = (N) => {
      const Tn = st[k++];
      const dn = d[0] * N[0] + d[1] * N[1] + d[2] * N[2];
      const ref = Tn < 0, pix = Tn + dn < 0;
      let f = 0;
      if (ref && pix) f = -2 * dn;
      else if (pix) f = -2 * (Tn + dn);
      else if (ref) f = 2 * Tn;
      if (f) { d[0] += f * N[0]; d[1] += f * N[1]; d[2] += f * N[2]; }
    };
    if (S === LG.STYLE_BRANCH) {
      const m = subM(P), ss = subS(P);
      for (let it = 0; it < m; it++) {
        if (it > 0) {
          const e = V.mMulV(P.rot, [d[0] * ss, d[1] * ss, d[2] * ss]);
          d[0] = e[0]; d[1] = e[1]; d[2] = e[2];
        }
        for (let i = 0; i < P.foldN; i++) absOp(i);
      }
      return [d[0] * ss, d[1] * ss, d[2] * ss];
    } else if (S === LG.STYLE_DIHEDRAL) {
      absOp(2);
      const nOps = dihedralOps(P.foldN);
      for (let i = 0; i < nOps; i++) planeOp(i % 2 ? P.dihN : DIH_N1);
    } else if (S === LG.STYLE_MENGER) {
      absOp(0); absOp(1); absOp(2);
      swapOp(0, 1); swapOp(0, 2); swapOp(1, 2);
    } else if (S === LG.STYLE_POLY) {
      flipOp(0, 1); flipOp(0, 2); flipOp(1, 2);
    } else if (S === LG.STYLE_ICOSA) {
      for (let round = 0; round < 2; round++) { absOp(0); absOp(1); absOp(2); goldOp(); }
    } else {
      absOp(0); absOp(1); absOp(2);
      swapOp(0, 1); swapOp(1, 2);
    }
    return [d[0] * P.scale, d[1] * P.scale, d[2] * P.scale];
  }
  const MAX_STAGES = 8; // ICOSA: 2 x (3 abs + 1 golden)
  const OUTER_VEC4 = 9;  // UBO vec4s per outer level (see packUBO)
  const LVL_VEC4 = 8;    // UBO vec4s per level slot (see packUBO)

  // style ops only (post-rotation, pre-scale part of the fold), for preimages
  function styleOps(v, P) {
    return evalFold(V.mMulV(V.mTranspose(P.rot), v), { ...P, scale: 1, trans: [0, 0, 0] }).x;
    // note: undoes the rotation first so only the style part applies
  }

  /* ---- fold preimage ------------------------------------------------------
     Find xin with Fold_j(xin) = y, choosing the candidate nearest `hint`.
     Returns null when y is outside the fold image (wedge exit).             */
  function foldPreimage(y, P, hint) {
    const w = [(y[0] - P.trans[0]) / P.scale, (y[1] - P.trans[1]) / P.scale, (y[2] - P.trans[2]) / P.scale];
    const h1 = V.mMulV(P.rot, hint); // hint in style-op input space
    const S = P.style;
    let cands = [];
    if (S === LG.STYLE_BRANCH) {
      // invert sub-steps from the last: u = (x - t)/s, mirror candidates,
      // undo the rotation; every branch combination (2^(k m) <= 256), each
      // validated through the full forward fold
      const ss = subS(P), m = subM(P), RT = V.mTranspose(P.rot);
      let xs = [y];
      for (let it = 0; it < m; it++) {
        const next = [];
        for (const x of xs) {
          const u = [(x[0] - P.trans[0]) / ss, (x[1] - P.trans[1]) / ss, (x[2] - P.trans[2]) / ss];
          for (let sg = 0; sg < (1 << P.foldN); sg++) {
            const c = u.slice();
            for (let i = 0; i < P.foldN; i++) if (sg & (1 << i)) c[i] = -c[i];
            next.push(V.mMulV(RT, c));
          }
        }
        xs = next;
      }
      let best = null, bestD = Infinity;
      const tolB = 1e-9 * Math.max(1, V.len(y));
      for (const v of xs) {
        if (V.dist(evalFold(v, P).x, y) > tolB) continue;
        const dd = V.dist(v, hint);
        if (dd < bestD) { bestD = dd; best = v; }
      }
      return best;
    }
    if (S === LG.STYLE_MENGER || S === LG.STYLE_OCTA) {
      // Preimages: undo the sorts (all permutations of w), then undo the
      // offset abs per component (v_i = a_i or its reflection 2*o_i - a_i).
      // Branch enumeration, NOT a group orbit — which is why offset (non-
      // concurrent) planes are fine for these styles. Validated below.
      const FO = P.foldOff || [0, 0, 0];
      const perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
      for (const pm of perms)
        for (let sg = 0; sg < 8; sg++)
          cands.push([
            sg & 1 ? 2 * FO[0] - w[pm[0]] : w[pm[0]],
            sg & 2 ? 2 * FO[1] - w[pm[1]] : w[pm[1]],
            sg & 4 ? 2 * FO[2] - w[pm[2]] : w[pm[2]],
          ]);
    } else {
      // POLY / ICOSA / DIHEDRAL / BRANCH: BFS orbit of w under the style's
      // mirror generators (finite groups — order 24, 120, 4n, 2-4 — so the
      // orbit closes)
      const refl = (N) => (v) => {
        const t = v[0] * N[0] + v[1] * N[1] + v[2] * N[2];
        return [v[0] - 2 * t * N[0], v[1] - 2 * t * N[1], v[2] - 2 * t * N[2]];
      };
      const R = S === LG.STYLE_BRANCH
        ? [(v) => [-v[0], v[1], v[2]], (v) => (P.foldN > 1 ? [v[0], -v[1], v[2]] : v)]
        : S === LG.STYLE_DIHEDRAL
        ? [(v) => [v[0], v[1], -v[2]], refl(DIH_N1), refl(P.dihN)]
        : S === LG.STYLE_ICOSA
        ? [
            (v) => [-v[0], v[1], v[2]],
            (v) => [v[0], -v[1], v[2]],
            (v) => [v[0], v[1], -v[2]],
            (v) => {
              const t = v[0] * ICO_N[0] + v[1] * ICO_N[1] + v[2] * ICO_N[2];
              return [v[0] - 2 * t * ICO_N[0], v[1] - 2 * t * ICO_N[1], v[2] - 2 * t * ICO_N[2]];
            },
          ]
        : [
            (v) => [-v[1], -v[0], v[2]],
            (v) => [-v[2], v[1], -v[0]],
            (v) => [v[0], -v[2], -v[1]],
          ];
      const cap = S === LG.STYLE_ICOSA ? 200 : 60;
      const seen = new Map();
      const key = (v) => v.map((x) => x.toFixed(12)).join(',');
      const queue = [w];
      seen.set(key(w), w);
      while (queue.length && seen.size < cap) {
        const v = queue.pop();
        for (const r of R) {
          const u = r(v);
          const k = key(u);
          if (!seen.has(k)) { seen.set(k, u); queue.push(u); }
        }
      }
      cands = [...seen.values()];
    }
    // validate candidates through the actual style ops and pick nearest hint
    let best = null, bestD = Infinity;
    const tol = 1e-9 * Math.max(1, V.len(w));
    for (const v of cands) {
      const r = evalFold(v, { ...P, rot: V.mIdent(), scale: 1, trans: [0, 0, 0] }).x;
      if (Math.abs(r[0] - w[0]) + Math.abs(r[1] - w[1]) + Math.abs(r[2] - w[2]) > tol) continue;
      const dd = V.dist(v, h1);
      if (dd < bestD) { bestD = dd; best = v; }
    }
    if (!best) return null;
    return V.mMulV(V.mTranspose(P.rot), best); // undo rotation
  }

  /* Wedge normalization: y is outside Fold_j's image. Returns the affine
     symmetry T (of all levels ≥ j+1) that reflects y back into the image:
     run the style ops on w=(y-trans)/s until fixed, record linear action. */
  function wedgeNormalize(y, P) {
    // (composite BRANCH levels have no single-step mirror group to
    // normalize with; only reachable without a chart, so leave y as is)
    if (P.style === LG.STYLE_BRANCH && subM(P) > 1) return { M: V.mIdent(), b: [0, 0, 0] };
    let w = [(y[0] - P.trans[0]) / P.scale, (y[1] - P.trans[1]) / P.scale, (y[2] - P.trans[2]) / P.scale];
    let T = V.mIdent();
    const bare = { ...P, rot: V.mIdent(), scale: 1, trans: [0, 0, 0] };
    for (let pass = 0; pass < 16; pass++) {
      const r = evalFold(w, bare, true);
      const moved = V.dist(r.x, w);
      T = V.mMul(r.M, T);
      w = r.x;
      if (moved < 1e-14) break;
    }
    // y-space affine: T(y) = M y + (trans - M trans)
    const M = T;
    const b = V.sub(P.trans, V.mMulV(M, P.trans));
    return { M, b };
  }

  /* ---- world state ------------------------------------------------------ */
  class World {
    constructor(seed, startPos) {
      this.seed = seed >>> 0;
      this.K = 0;
      this.camPos = startPos ? startPos.slice() : [0.6, 1.1, 3.4]; // frame-K, float64
      this.hints = [];                      // hints[j] = last-known c_j (preimage selector)
      this.hintAff = [];                    // hintAff[j] = last affine of Fold_j at c_j (branch continuation)
      this.chainFailures = 0;               // updateChain wedge loops that did not settle (should stay 0)
      this.maxCorrection = 0;               // largest wedge-correction displacement BEYOND the 2x-step
                                            // reflection bound (diagnostic; should stay ~0)
      this.outer = [];                      // per-frame chain, outer[0] = level K-Beff
      this.W = V.mIdent();
      this.WScale = 1;
      this.log10Zoom = 0;
      /* World-anchored sky frame: columns are the WORLD axes expressed in
         frame-K coordinates. Every chart change (rebase, wedge correction)
         re-expresses frame coordinates by a rotation/reflection, so anything
         that means a fixed direction in the world — "up" for the sky
         gradient and ambient, the star field, the sun — must ride this basis.
         (Using the frame's own y axis made the sky and ambient tint jump at
         rebases.) */
      this.skyBasis = V.mIdent();
      this.lights = [];                     // {pos (frame K), col, radius, intensity, level}
      this._litLevels = new Set();          // levels that already spawned their light (never respawn)
      this.autoLights = true;               // O key: disable to place lights manually only
      this.timeSec = 0;                     // world clock (advanced by tick) for light fade-ins
      this.pendingRot = V.mIdent();         // accumulated frame rotation for the renderer's camera basis
      /* Fog continuity multiplier. Fog density and the march limit live in
         frame-K units, but every rebase rescales those units by s — with
         fixed constants the whole fog field visibly thickened at each zoom
         step. fogMul makes optical depth CONTINUOUS: it relaxes toward s^f
         (f = fractional progress through the level, 0 after a rebase, 1 at
         the next), and descend/ascend rescale it by exactly 1/s / s — the
         same jump the frame distances make, so k*t never jumps. */
      this.fogMul = 1;
      this._pcache = new Map();
      this._bcache = new Map();
      this.updateChain();
    }

    params(j) {
      let p = this._pcache.get(j);
      if (!p) { p = LG.levelParams(this.seed, j); this._pcache.set(j, p); }
      return p;
    }

    _applyAffine(aff) {
      // apply a frame-K affine symmetry to everything world-anchored
      // (camera, lights, sky basis — the sun direction derives from it)
      this.camPos = V.add(V.mMulV(aff.M, this.camPos), aff.b);
      for (const L of this.lights) L.pos = V.add(V.mMulV(aff.M, L.pos), aff.b);
      this._rotSky(aff.M);
      // a motion in flight (moveCamera) is re-charted with everything else,
      // or the camera would bounce back across the crease it just crossed
      if (this._moveDir) this._moveDir = V.norm(V.mMulV(aff.M, this._moveDir));
      this.pendingRot = V.mMul(this._rotPart(aff.M), this.pendingRot);
    }
    _rotSky(M) {
      this.skyBasis = V.mMul(this._rotPart(M), this.skyBasis);
      // re-orthonormalize (Gram-Schmidt on columns; handedness preserved) so
      // thousands of rebases can't drift it
      const c = V.mToCols(this.skyBasis);
      let x = V.norm([c[0], c[1], c[2]]);
      let y = [c[3], c[4], c[5]];
      y = V.norm(V.sub(y, V.scale(x, V.dot(x, y))));
      let z = [c[6], c[7], c[8]];
      z = V.norm(V.sub(V.sub(z, V.scale(x, V.dot(x, z))), V.scale(y, V.dot(y, z))));
      this.skyBasis = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]];
    }
    _rotPart(M) {
      const s = Math.cbrt(Math.abs(
        M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6])
      )) || 1;
      return V.mScale(M, 1 / s);
    }

    /* Rebuild the outer chain from the current camera position. Each failed
       attempt applies one wedge correction and restarts; large camera jumps
       (autopilot/tests) at deep K can legitimately need a few corrections
       per level of the window, hence the generous cap. */
    updateChain() {
      for (let attempt = 0; attempt < 4 * B_HARD + 8; attempt++) {
        if (this._tryChain()) return;
      }
      // Should be unreachable; keep whatever partial chain we have.
      this.chainFailures++;
      console.warn('updateChain: wedge normalization did not settle');
    }

    _tryChain() {
      const BH = Math.min(this.K, B_HARD);
      const full = []; // walked entries, full[b] = level K-1-b
      const affs = []; // affs[b] = affine of Fold_{K-1-b} at c_{K-1-b} (frame j -> j+1)
      let cNext = this.camPos;
      let mag = 1; // Π s over processed levels: frame-j -> frame-K magnification
      for (let b = 0; b < BH; b++) {
        const j = this.K - 1 - b;
        const P = this.params(j);
        const hint = this.hints[j] || cNext;
        /* Branch CONTINUATION. The physical camera at level j is ALWAYS the
           old chart's continuation cand = A_old^-1(c_{j+1}) — that is the
           point the renderer was showing. If Fold_j(cand) still equals
           c_{j+1}, nothing happened. Otherwise the camera crossed a fold
           crease of level j, and its true frame-(j+1) position is
           Fold_j(cand): re-chart with T = A_new·A_old^-1. Folds are
           continuous, so A_new = A_old·σ with σ the mirror of the crossed
           crease, which is an exact local symmetry of the level-j world:
           T is invisible and moves the camera by at most ~2x its step.
           (Never search other preimages here: POLY/OCTA fold images overlap
           themselves, so a "valid" preimage can lie in a far region —
           jumping the chain there teleported the camera.) */
        let pre = null;
        let T = null;
        const ha = this.hintAff[j];
        if (ha) {
          const cand = V.mMulV(ha.Minv, V.sub(cNext, ha.b));
          const rc = evalFold(cand, P, true);
          if (Math.abs(rc.x[0] - cNext[0]) + Math.abs(rc.x[1] - cNext[1]) + Math.abs(rc.x[2] - cNext[2]) <
              1e-10 * Math.max(1, V.len(cNext))) {
            pre = cand;
          } else {
            const bNew = V.sub(rc.x, V.mMulV(rc.M, cand));
            // T(y) = A_new(A_old^-1(y)) = M y + (bNew - M b_old), M = M_new·M_old^-1
            const M = V.mMul(rc.M, ha.Minv);
            T = { M, b: V.sub(bNew, V.mMulV(M, ha.b)) };
            this.hints[j] = cand;
            this.hintAff[j] = { Minv: V.mInv(rc.M), b: bNew };
          }
        } else {
          // no chart yet (world construction): nearest preimage to the hint
          pre = foldPreimage(cNext, P, hint);
          // Camera outside Fold_j's image: reflect back with the wedge
          // normalization (only reachable without a chart)
          if (!pre) T = wedgeNormalize(cNext, P);
        }
        if (T) {
          // lift T from frame j+1 to frame K through the already-known
          // affines, transforming each frame's chain hint on the way up.
          // (The per-level affines hintAff[m] are unchanged by a lifted
          // re-chart: T_{m+1}·A_m·T_m^-1 = A_m by construction.)
          for (let bb = b - 1; bb >= 0; bb--) {
            const m = this.K - 1 - bb; // T is currently expressed in frame m
            if (this.hints[m]) this.hints[m] = V.add(V.mMulV(T.M, this.hints[m]), T.b);
            const A = affs[bb]; // frame m -> m+1
            const invAM = V.mInv(A.M);
            const M = V.mMul(A.M, V.mMul(T.M, invAM));
            const tb = V.add(V.mMulV(A.M, V.add(V.mMulV(T.M, V.mMulV(invAM, V.scale(A.b, -1))), T.b)), A.b);
            T = { M, b: tb };
          }
          const c0 = this.camPos;
          this._applyAffine(T);
          // diagnostics: a correction is a reflection across a crease the
          // camera just crossed, so it moves the camera by at most ~2x the
          // step. Anything large means the chain lost branch continuity.
          this.maxCorrection = Math.max(this.maxCorrection,
            V.dist(c0, this.camPos) - 2.05 * (this._stepNow || 0));
          return false; // restart the walk with the corrected camera
        }
        const r = evalFold(pre, P, true);
        mag *= P.scale;
        full.push({
          level: j, c: pre, M: r.M, st: r.st, margin: Math.max(0, r.margin * 0.999),
          scale: P.scale, crossing: r.margin * mag,
        });
        const aff = { M: r.M, b: V.sub(r.x, V.mMulV(r.M, pre)) };
        affs.push(aff);
        this.hints[j] = pre;
        this.hintAff[j] = { Minv: V.mInv(r.M), b: aff.b };
        cNext = pre;
      }
      // The level just beyond the walk leaves our view: its chart goes stale
      // (the camera keeps moving), so drop it. If an ascend brings it back it
      // is re-derived by enumeration — a stale chart would "continue" into a
      // wildly wrong re-chart.
      if (this.K - 1 - BH >= 0) this.hintAff[this.K - 1 - BH] = null;
      // Window size: keep at least B_MIN levels, and keep any older level whose
      // fold boundaries pass within the horizon (dropping it would visibly
      // change near geometry).
      let Beff = Math.min(this.K, B_MIN);
      for (let b = full.length - 1; b >= Beff; b--) {
        if (full[b].crossing < HORIZON * 2) { Beff = b + 1; break; }
      }
      /* Window reach. When the window is saturated at B_HARD, the next older
         level is dropped regardless of its crossing distance — fine for
         large-scale styles (1.78^48 ~ 1e12 > horizon), but the organic
         styles step by ~1.6-2x per level, so 48 levels span only ~1e10.
         updateFog keeps the march/fog limit inside this reach, so the level
         beyond the window is never visibly cut. */
      this.windowReach = (BH === B_HARD && Beff === BH) ? mag : Infinity;
      this.outer.length = 0;
      // Does any window level need the perturbative outer fold? (see
      // ABS_REL2 and the program variants in renderer.js)
      this.needsPerturb = false;
      for (let b = 0; b < Beff; b++) {
        const E = full[b];
        if (E.margin * E.margin < ABS_REL2 * Math.max(V.dot(E.c, E.c), 1)) this.needsPerturb = true;
      }
      let W = V.mIdent();
      let WScale = 1;
      for (let b = 0; b < Beff; b++) {
        this.outer.unshift(full[b]);
        W = V.mMul(V.mInv(full[b].M), W);
        WScale /= full[b].scale;
      }
      this.W = W;
      this.WScale = WScale;
      /* Near-field fast path radius: for |p - camPos| < RLin, EVERY outer
         level takes its linear branch, and the composed action
         M_{B-1}···M_0·W is the IDENTITY by construction (W is exactly the
         inverse product), so the outer phase can be skipped entirely —
         geometry within RLin of the camera is fully described by the inner
         levels. |d_b| = |p|·cum_b with cum_b = WScale·Π_{i<b} s_i, so the
         all-linear condition |d_b| < m_b for all b reduces to one radius
         test. The renderer uses this for all marching, normal, AO and shadow
         samples near the camera — most of them — removing ~B fold/mat ops
         per map() call. (It is also more accurate than the walk: the walk
         carries the rounding noise of W's accumulated matrix inversions.) */
      /* Per-level SKIP data (the deep-zoom "series approximation" idea):
         R_b = radius within which levels 0..b ALL take their linear branch
         (prefix min, nonincreasing in b), Q_b = their composed linear map
         (frame K -> frame of level b+1), drAfter_b = magnification after
         level b. A sample at |p| < R_b jumps past levels 0..b with ONE
         matrix multiply. RLin (all levels linear) is just R_{B-1}; the old
         all-or-nothing fast path collapsed whenever a single old crease
         passed near the camera, sending nearly every sample through the
         whole window. */
      let RLin = Infinity;
      let cum = WScale;
      let Q = W;
      for (let b = 0; b < this.outer.length; b++) {
        const E = this.outer[b];
        RLin = Math.min(RLin, E.margin / cum);
        cum *= E.scale;
        Q = V.mMul(E.M, Q);
        E.R = RLin;
        E.Q = Q;
        E.drAfter = cum;
      }
      this.RLin = this.outer.length ? RLin : Infinity;
      this.bound = this.boundAfter(this.K - 1); // informational (tools)
      return true;
    }

    /* Bounding radius of the attractor generated by levels > j, i.e. the
       radius to subtract when the DE ends right after level j's fold:
       DE = (|x| - R_{j+1}) / dr. Exact backward recursion of the invariant
       ball: x in G_i  =>  |x| <= (R_{i+1} + |t_i|)/s_i + 2|o_i|  (offset
       folds displace by <= 2|o|), started 24 levels ahead from the fixed
       point of that level (error ~ s^-24). PER ABSOLUTE LEVEL — so the DE
       of a point is the same before and after a rebase. (A single max over
       the current window changed whenever the window slid across biomes
       with different radii: a small pop at zoom steps.) Rigorous for any
       truncation depth, which is what keeps coarse LOD hole-free. */
    boundAfter(j) { return this.ballAfter(j).R; }

    /* Bounding BALL (center C, radius R) of the attractor generated by
       levels > j, in frame j+1: the DE after level j's fold is
       (|x - C| - R) / dr. Exact backward recursion per (sub-)step with map
       x -> s*fold(Rot x) + t: if the output attractor lies in B(C', R'),
       then fold(Rot x) lies in B(q, rho) with q = (C'-t)/s, rho = R'/s, so
       Rot x lies in the union of that ball's mirror images. For the BRANCH
       mirrors (x, or x and y) that union sits in a ball CENTERED ON THE
       MIRROR-FIXED SUBSPACE: center = q with the mirrored components zeroed,
       radius = |mirrored part of q| + rho. The old origin-centered ball was
       far too loose for off-center trees: the distance field had basins
       everywhere (local minima with no geometry nearby) that trapped the
       autopilot. Other styles' groups fix only the origin: center 0, radius
       |q| + rho (+ 2|o| for offset planes). Started 24 levels ahead from
       that level's own fixed point (error ~ s^-24); per ABSOLUTE level, so
       rebase-invariant; rigorous for any truncation depth (hole-free LOD). */
    ballAfter(j) {
      const key = j + 1;
      let B = this._bcache.get(key);
      if (B !== undefined) return B;
      const back = (P, C, R) => {
        const ss = subS(P);
        const q = [(C[0] - P.trans[0]) / ss, (C[1] - P.trans[1]) / ss, (C[2] - P.trans[2]) / ss];
        const rho = R / ss;
        if (P.style === LG.STYLE_BRANCH) {
          const k = P.foldN;
          const mir = k > 1 ? Math.hypot(q[0], q[1]) : Math.abs(q[0]);
          const c = [0, k > 1 ? 0 : q[1], q[2]];
          return { C: V.mMulV(V.mTranspose(P.rot), c), R: mir + rho };
        }
        const fo = P.foldOff ? Math.hypot(P.foldOff[0], P.foldOff[1], P.foldOff[2]) : 0;
        return { C: [0, 0, 0], R: V.len(q) + rho + 2 * fo };
      };
      const J = Math.max(0, key) + 24;
      const PJ = this.params(J);
      const tJ = Math.hypot(PJ.trans[0], PJ.trans[1], PJ.trans[2]);
      const foJ = PJ.foldOff ? Math.hypot(PJ.foldOff[0], PJ.foldOff[1], PJ.foldOff[2]) : 0;
      // level J's own fixed point (iterate its sub-step map to convergence)
      let cur = { C: [0, 0, 0], R: (tJ + 2 * subS(PJ) * foJ) / (subS(PJ) - 1) };
      for (let it = 0; it < 60; it++) cur = back(PJ, cur.C, cur.R);
      for (let i = J - 1; i >= Math.max(0, key); i--) {
        const P = this.params(i);
        for (let k = subM(P); k > 0; k--) cur = back(P, cur.C, cur.R);
      }
      B = { C: cur.C, R: cur.R * 1.05 }; // float safety margin
      this._bcache.set(key, B);
      return B;
    }

    /* Float64 distance estimator at frame-K point p (absolute frame-K coords).
       forceSlow disables level skipping (used by tests to validate it). */
    de(p, forceSlow) {
      let x, dr;
      let escaped = false;
      let lastJ = this.K - 1; // absolute level of the last fold applied
      if (this.outer.length) {
        const rel = V.sub(p, this.camPos);
        // skip the leading levels this sample is linear through (see R_b)
        const rl = V.len(rel);
        let b0 = 0;
        if (!forceSlow) while (b0 < this.outer.length && rl < this.outer[b0].R) b0++;
        if (b0 === this.outer.length) {
          // all outer levels act linearly here and compose to the identity
          x = p.slice();
          dr = 1;
        } else {
          // Outer phase in PERTURBATIVE form: d = (point - c_j) in frame j,
          // carried exactly through every outer fold (see foldDelta). Only
          // once d is large (the point is far from the camera at this level's
          // scale) do absolute coordinates take over — then their rounding is
          // small relative to d. (Absolute evaluation of c + d at the
          // outermost level loses s^B of precision: unusable for big windows.)
          // (structure mirrors the OUTER template in shaders.js exactly)
          const O = this.outer;
          let d = b0 === 0 ? V.mMulV(this.W, rel) : V.mMulV(O[b0 - 1].Q, rel);
          dr = b0 === 0 ? this.WScale : O[b0 - 1].drAfter;
          let bAbs = -1;
          x = null;
          for (let b = b0; b < O.length; b++) {
            const E = O[b];
            const P = this.params(E.level);
            const dd = V.dot(d, d);
            dr *= E.scale;
            const cn = b + 1 < O.length ? O[b + 1].c : this.camPos;
            if (dd < E.margin * E.margin) {
              d = V.mMulV(E.M, d);
            } else if (dd > ABS_REL2 * Math.max(V.dot(E.c, E.c), 1)) {
              x = evalFold(V.add(E.c, d), P).x;
              bAbs = b;
              lastJ = E.level;
              if (V.dot(x, x) > ESCAPE2) escaped = true;
              break;
            } else {
              d = foldDelta(d, P, E.st);
              const xe = V.add(cn, d);
              if (V.dot(xe, xe) > ESCAPE2) { x = xe; escaped = true; lastJ = E.level; break; }
            }
          }
          if (!escaped && bAbs >= 0) {
            // absolute tail: d is large here, so absolute rounding is small
            // relative to it
            for (let b = bAbs + 1; b < O.length; b++) {
              x = evalFold(x, this.params(O[b].level)).x;
              dr *= O[b].scale;
              lastJ = O[b].level;
              if (V.dot(x, x) > ESCAPE2) { escaped = true; break; }
            }
          } else if (!escaped) {
            x = V.add(this.camPos, d);
          }
        }
      } else {
        x = p.slice();
        dr = 1;
      }
      if (!escaped) {
        for (let i = 0; i < N_INNER; i++) {
          x = evalFold(x, this.params(this.K + i)).x;
          dr *= this.params(this.K + i).scale;
          lastJ = this.K + i;
          if (V.dot(x, x) > ESCAPE2) break;
        }
      }
      const ball = this.ballAfter(lastJ);
      return (V.dist(x, ball.C) - ball.R) / dr;
    }

    grad(p, h) {
      h = h || Math.max(1e-9, Math.abs(this.de(p)) * 1e-3);
      return V.norm([
        this.de([p[0] + h, p[1], p[2]]) - this.de([p[0] - h, p[1], p[2]]),
        this.de([p[0], p[1] + h, p[2]]) - this.de([p[0], p[1] - h, p[2]]),
        this.de([p[0], p[1], p[2] + h]) - this.de([p[0], p[1], p[2] - h]),
      ]);
    }

    /* Move the camera by a frame-K displacement, CONTINUOUSLY. The outer
       chain tracks the camera's branch at every level; a single large jump
       can cross several fold creases at once, and the preimage choice after
       such a jump is ambiguous (the wrong choice puts the camera "outside"
       an outer fold image and the wedge correction then teleports it). Sub-
       stepping keeps every step inside RLin — the radius in which NO outer
       crease exists — so creases are crossed one at a time and each wedge
       correction is a tiny exact reflection. Bounded: at most ~48 substeps
       (the floor of |delta|/32 stops Zeno-shrinking toward a crease). */
    moveCamera(delta) {
      let rem = V.len(delta);
      if (!(rem > 0)) return null;
      this._moveDir = V.scale(delta, 1 / rem);
      const floor = rem / 32;
      for (let i = 0; i < 48 && rem > 0; i++) {
        const lim = Number.isFinite(this.RLin) ? Math.max(0.5 * this.RLin, floor) : rem;
        const step = Math.min(rem, lim);
        this.camPos = V.add(this.camPos, V.scale(this._moveDir, step));
        rem -= step;
        this._stepNow = step;
        if (rem > 0) this.updateChain();
      }
      if (rem > 0) this.camPos = V.add(this.camPos, V.scale(this._moveDir, rem));
      this._stepNow = Math.max(this._stepNow, rem);
      const dirOut = this._moveDir;
      this._moveDir = null;
      this.updateChain();
      this._stepNow = 0;
      return dirOut; // the motion direction in the (possibly re-charted) frame
    }

    /* Sphere-trace a ray from frame-K point o along unit dir (CPU, float64).
       Returns the hit distance, or Infinity if nothing is hit within maxT.
       Used to steer toward REAL surfaces: in sparse attractors (trees,
       membranes) the lower-bound distance field has local minima with no
       geometry nearby, so following its gradient can circle forever. */
    rayHit(o, dir, maxT, hitEps) {
      let t = 0;
      for (let i = 0; i < 160; i++) {
        const d = this.de(V.add(o, V.scale(dir, t)));
        if (d < hitEps) return t;
        t += d * 0.9;
        if (t > maxT) break;
      }
      return Infinity;
    }

    /* ---- rebasing -------------------------------------------------------- */
    descend() {
      const P = this.params(this.K);
      const r = evalFold(this.camPos, P, true);
      this.hints[this.K] = this.camPos;
      // the exact branch affine just used — a stale one from an earlier visit
      // to this level would "continue" into a wrong (mirror-copy) preimage
      const s = P.scale, s2 = s * s;
      // World-anchored points must be carried by the camera-branch AFFINE
      // (r.M, b), never by the true fold: the rendered frame-(K+1) world is
      // exactly the frame-K world seen through that affine. evalFold() would
      // reflect anything lying across a level-K mirror (relative to the
      // camera) to its mirror image — lights visibly teleported at rebases.
      const bAff = V.sub(r.x, V.mMulV(r.M, this.camPos));
      this.hintAff[this.K] = { Minv: V.mInv(r.M), b: bAff };
      // x s^2 keeps the rebase photometrically EXACT (no blink); the age fade
      // is applied continuously at upload time in gpuLights().
      for (const L of this.lights) {
        L.pos = V.add(V.mMulV(r.M, L.pos), bAff);
        L.radius *= s;
        L.intensity *= s2;
      }
      this._rotSky(r.M);
      this.pendingRot = V.mMul(this._rotPart(r.M), this.pendingRot);
      this.camPos = r.x;
      this.K++;
      this.log10Zoom += Math.log10(s);
      this.fogMul /= s; // distances just grew by s: keep optical depth continuous
      this._spawnLight();
      this._pruneLights();
      this.updateChain();
    }

    ascend() {
      if (this.K <= 0) return false;
      const P = this.params(this.K - 1);
      // The physical camera at level K-1 is the chain's continuation through
      // the current branch affine (exact, unambiguous). Enumerating preimages
      // instead could pick another mirror copy for POLY/OCTA, whose fold
      // images overlap — a teleport on zoom-out.
      let pre = null;
      const ha = this.hintAff[this.K - 1];
      if (ha) {
        const cand = V.mMulV(ha.Minv, V.sub(this.camPos, ha.b));
        const chk = evalFold(cand, P).x;
        if (V.dist(chk, this.camPos) < 1e-10 * Math.max(1, V.len(this.camPos))) pre = cand;
      }
      if (!pre) pre = foldPreimage(this.camPos, P, this.hints[this.K - 1] || this.camPos);
      if (!pre) {
        this._applyAffine(wedgeNormalize(this.camPos, P));
        pre = foldPreimage(this.camPos, P, this.hints[this.K - 1] || this.camPos);
        if (!pre) return false; // shouldn't happen
      }
      const r = evalFold(pre, P, true); // exact affine at the preimage
      const invM = V.mInv(r.M);
      this.hintAff[this.K - 1] = null; // level K-1 becomes inner; re-derive on next descend
      const s = P.scale, s2 = s * s;
      const bb = V.sub(r.x, V.mMulV(r.M, pre));
      for (const L of this.lights) {
        L.pos = V.mMulV(invM, V.sub(L.pos, bb));
        L.radius /= s;
        L.intensity /= s2;
      }
      this._rotSky(invM);
      this.pendingRot = V.mMul(this._rotPart(invM), this.pendingRot);
      this.camPos = pre;
      this.K--;
      this.log10Zoom -= Math.log10(s);
      this.fogMul *= s; // distances just shrank by s: keep optical depth continuous
      this._pruneLights();
      this.updateChain();
      return true;
    }

    /* Keep the camera's local scale in a fixed band by rebasing. Returns DE. */
    rebaseToCamera() {
      let d = this.de(this.camPos);
      let guard = 0;
      while (d < DESCEND_DE && guard++ < 6) {
        this.descend();
        d = this.de(this.camPos);
      }
      while (this.K > 0 && guard++ < 12) {
        const sPrev = this.params(this.K - 1).scale;
        if (d < DESCEND_DE * sPrev * 2.1) break;
        if (!this.ascend()) break;
        d = this.de(this.camPos);
      }
      return d;
    }

    // Renderer support: consume the accumulated frame rotation (reflections /
    // rotations from rebases and wedge corrections) to keep the view basis
    // pointing at the same physical scene.
    takeFrameRotation() {
      const r = this.pendingRot;
      this.pendingRot = V.mIdent();
      return r;
    }

    /* ---- lights ---------------------------------------------------------- */
    _spawnLight() {
      if (!this.autoLights) return; // O key: manual placement only
      const P = this.params(this.K);
      if (!P.light.spawn) return;
      // A level spawns its light at most ONCE per session. Checking the live
      // light list is not enough: zooming out prunes the light (radius decay),
      // so bouncing in and out of an area used to pile up fresh full-intensity
      // copies until the frame washed out.
      if (this._litLevels.has(this.K)) return;
      this._litLevels.add(this.K);
      // Wall-hugging placement: probe deterministic directions and prefer a
      // spot NEAR geometry a few units out. The light reads as a pool on a
      // structure you can travel toward — grazing walls, casting shadows,
      // glimpsed around corners — instead of a lamp floating beside the
      // camera that floodlights everything the moment the level rebases.
      const R = (salt) => LG.rnd(this.seed, this.K, salt);
      let best = null, bestScore = -Infinity;
      for (let i = 0; i < 10; i++) {
        const dir = V.norm([R(60 + i * 4) * 2 - 1, R(61 + i * 4) * 2 - 1, R(62 + i * 4) * 2 - 1]);
        const dist = 1.5 + R(63 + i * 4) * 4.0;
        const p = V.add(this.camPos, V.scale(dir, dist));
        const d = this.de(p);
        if (d < 0.03) continue; // inside or embedded in a wall
        // ~0.3 off a wall is the sweet spot; mild preference for nearer spots
        const score = -Math.abs(d - 0.3) - dist * 0.03;
        if (score > bestScore) { bestScore = score; best = p; }
      }
      const pos = best ||
        V.add(this.camPos, V.add(V.scale(this.grad(this.camPos), 0.9), V.scale(P.light.off, 0.9)));
      this.lights.push({
        pos, col: P.light.col, radius: 0.05, intensity: P.light.intensity,
        level: this.K, spawnZoom: this.log10Zoom,
        spawnT: this.timeSec, fadeDur: 3.0, // grow in gently — never pop at a rebase
      });
    }

    /* Manual light drop (L key). Placed just ahead of the camera in frame-K
       units, so it is automatically sized to the current zoom scale and is
       carried through rebases like any spawned light. col overrides the
       level's palette color (C key cycles the choice in main.js). */
    addUserLight(forward, col) {
      const de = Math.max(Math.abs(this.de(this.camPos)), 0.005);
      const pos = V.add(this.camPos, V.scale(forward, Math.min(de * 2.0, 0.12) + 0.02));
      this.lights.push({
        pos, col: col || this.params(this.K).light.col, radius: 0.02, intensity: 0.14,
        level: this.K, spawnZoom: this.log10Zoom, spawnT: this.timeSec, fadeDur: 0.5,
      });
      this._pruneLights();
    }

    /* Remove the light nearest to the camera (X key). Returns false if none. */
    removeNearestLight() {
      if (!this.lights.length) return false;
      let bi = 0, bd = Infinity;
      for (let i = 0; i < this.lights.length; i++) {
        const d = V.dist(this.lights[i].pos, this.camPos);
        if (d < bd) { bd = d; bi = i; }
      }
      this.lights.splice(bi, 1);
      return true;
    }

    _pruneLights() {
      // beyond ~4e4 units the fog extinction makes a light invisible
      this.lights = this.lights.filter(
        (L) => L.radius > 2e-4 && L.radius < 4e4 && V.dist(L.pos, this.camPos) < 6e4
      );
      if (this.lights.length > MAX_LIGHTS_STORED)
        this.lights.splice(0, this.lights.length - MAX_LIGHTS_STORED);
    }

    /* ---- GPU upload ------------------------------------------------------- */
    /* UBO layout (std140, all vec4):
         vec4 lvl[SLOTS*7]  — per level slot: rotCol0(w=scale), rotCol1(w=style),
                              rotCol2(w=foldL / dihedral n), trans(w=emissive),
                              pal(w=accent), foldOff or dihedral N2 (w=glow hue),
                              (-, sub-step scale, sub-steps, -), bounding ball
                              after this level (center xyz, radius)
         vec4 outer[B_HARD*9] — per outer slot: jacCol0 (w=R_b), jacCol1 (w=drAfter_b),
                                jacCol2, (c.xyz, margin), stages 0-3, stages 4-7
                                (foldDelta decision scalars), Q_b cols 0-2
       Slot i covers absolute level (K - Beff + i).                            */
    packUBO(buf) {
      const B = this.outer.length;
      const f = buf;
      for (let i = 0; i < SLOTS; i++) {
        const j = this.K - B + i;
        const P = this.params(Math.max(0, j));
        const o = i * LVL_VEC4 * 4;
        const rc = V.mToCols(P.rot);
        f[o + 0] = rc[0]; f[o + 1] = rc[1]; f[o + 2] = rc[2]; f[o + 3] = P.scale;
        f[o + 4] = rc[3]; f[o + 5] = rc[4]; f[o + 6] = rc[5]; f[o + 7] = P.style;
        f[o + 8] = rc[6]; f[o + 9] = rc[7]; f[o + 10] = rc[8];
        f[o + 11] = (P.style === LG.STYLE_DIHEDRAL || P.style === LG.STYLE_BRANCH) ? P.foldN : P.foldL;
        f[o + 12] = P.trans[0]; f[o + 13] = P.trans[1]; f[o + 14] = P.trans[2]; f[o + 15] = P.emissive;
        f[o + 16] = P.pal[0]; f[o + 17] = P.pal[1]; f[o + 18] = P.pal[2]; f[o + 19] = P.accent || 0;
        // slot 5: offset planes (MENGER/OCTA) or the DIHEDRAL mirror normal N2
        const FO = P.style === LG.STYLE_DIHEDRAL ? P.dihN : (P.foldOff || [0, 0, 0]);
        f[o + 20] = FO[0]; f[o + 21] = FO[1]; f[o + 22] = FO[2];
        f[o + 23] = P.glowHue === undefined ? 0.55 : ((P.glowHue % 1) + 1) % 1;
        // slot 6: (-, sub-step scale, sub-steps, -); slot 7: bounding ball
        // (center, radius) after this level's fold (see ballAfter)
        const ball = this.ballAfter(Math.max(0, j));
        f[o + 24] = 0; f[o + 25] = subS(P); f[o + 26] = subM(P); f[o + 27] = 0;
        f[o + 28] = ball.C[0]; f[o + 29] = ball.C[1]; f[o + 30] = ball.C[2]; f[o + 31] = ball.R;
      }
      const base = SLOTS * LVL_VEC4 * 4;
      for (let b = 0; b < B_HARD; b++) {
        const o = base + b * OUTER_VEC4 * 4;
        if (b < B) {
          const E = this.outer[b];
          const jc = V.mToCols(E.M);
          f[o + 0] = jc[0]; f[o + 1] = jc[1]; f[o + 2] = jc[2]; f[o + 3] = Math.min(E.R, 1e30);
          f[o + 4] = jc[3]; f[o + 5] = jc[4]; f[o + 6] = jc[5]; f[o + 7] = E.drAfter;
          f[o + 8] = jc[6]; f[o + 9] = jc[7]; f[o + 10] = jc[8]; f[o + 11] = 0;
          f[o + 12] = E.c[0]; f[o + 13] = E.c[1]; f[o + 14] = E.c[2]; f[o + 15] = E.margin;
          for (let k = 0; k < MAX_STAGES; k++) f[o + 16 + k] = k < E.st.length ? E.st[k] : 0;
          const qc = V.mToCols(E.Q);
          for (let k = 0; k < 3; k++) {
            f[o + 24 + k * 4] = qc[k * 3]; f[o + 25 + k * 4] = qc[k * 3 + 1];
            f[o + 26 + k * 4] = qc[k * 3 + 2]; f[o + 27 + k * 4] = 0;
          }
        } else {
          for (let k = 0; k < OUTER_VEC4 * 4; k++) f[o + k] = 0;
        }
      }
      return f;
    }

    /* Advance the world clock (light fade-ins). Called once per frame, and by
       the deterministic test driver, so fades work identically in DET mode. */
    tick(dt) {
      this.timeSec += dt;
    }

    /* Per-frame fog relaxation: chase s^f (see fogMul in the constructor).
       The rebase rescales (descend/ascend) preserve continuity EXACTLY; this
       EMA only absorbs the drift between levels of different scale and the
       ascend hysteresis, both of which resolve smoothly over ~a second. */
    updateFog(deCam, dt) {
      const sK = this.params(this.K).scale;
      const de = Math.max(deCam || DESCEND_DE * sK, 1e-12);
      const f = Math.min(1, Math.max(0, Math.log((DESCEND_DE * sK) / de) / Math.log(sK)));
      // reach floor: the farthest fogged geometry (0.92*TMAX/fogMul, see
      // HORIZON) must stay within ~2x the saturated window's reach. Chased
      // like the rest of the target so a window change never pops the fog.
      const reachFloor = Number.isFinite(this.windowReach) ? 0.46 * 1e11 / this.windowReach : 0;
      const target = Math.max(Math.pow(sK, f), reachFloor);
      this.fogMul += (target - this.fogMul) * (1 - Math.exp(-(dt || 0.016) * 2.5));
      // floor 0.85: the level-drop-invisibility bound (see HORIZON) assumes
      // the effective march limit never exceeds TMAX/0.85
      this.fogMul = Math.min(Math.max(this.fogMul, 0.85), Math.max(sK * 1.6, reachFloor * 1.6));
    }

    /* Sky/fog mood, CONTINUOUS in depth: K plus fractional progress through
       the level inferred from the camera's DE (descend fires at DESCEND_DE,
       right after which DE ≈ DESCEND_DE * s). Discrete-K moods pop at every
       rebase. f is deliberately unclamped: it stays continuous through the
       ascend hysteresis band. */
    _depthD(deCam) {
      const sK = this.params(this.K).scale;
      const de = Math.max(deCam || DESCEND_DE * sK, 1e-12);
      return this.K + Math.log((DESCEND_DE * sK) / de) / Math.log(sK);
    }
    _moodAt(D) {
      const j0 = Math.max(0, Math.floor(D));
      const fr = Math.min(1, Math.max(0, D - j0));
      const a = this.params(j0).biome;
      const b = this.params(j0 + 1).biome;
      return { sky: V.lerp(a.sky, b.sky, fr), fog: V.lerp(a.fog, b.fog, fr),
        haze: a.haze + (b.haze - a.haze) * fr,
        sun: V.lerp(a.sun, b.sun, fr), sunW: V.lerp(a.sunW, b.sunW, fr) };
    }
    mood(deCam) {
      // Average three samples along the depth track: biome (zone) changes
      // spread over ~2.5 levels of zoom instead of stepping within one.
      const D = this._depthD(deCam);
      const a = this._moodAt(D - 0.8), b = this._moodAt(D), c = this._moodAt(D + 0.8);
      const avg = (k) => [
        (a[k][0] + b[k][0] + c[k][0]) / 3,
        (a[k][1] + b[k][1] + c[k][1]) / 3,
        (a[k][2] + b[k][2] + c[k][2]) / 3,
      ];
      // the sun's direction is WORLD-anchored (biome trait), mapped into the
      // current frame through the sky basis
      return { sky: avg('sky'), fog: avg('fog'), haze: (a.haze + b.haze + c.haze) / 3,
        sun: avg('sun'), sunDir: V.norm(V.mMulV(this.skyBasis, V.norm(avg('sunW')))) };
    }

    /* Continuous depth in log10 units (K progress inferred from camera DE),
       shared by mood() and the light age fade. */
    depthZoom(deCam) {
      const sK = this.params(this.K).scale;
      const de = Math.max(deCam || DESCEND_DE * sK, 1e-12);
      // exactly continuous across rebase: de scales by s as log10Zoom gains log10(s)
      const dz = Math.min(1.2, Math.max(-1.2, Math.log10(de / DESCEND_DE)));
      return this.log10Zoom - dz;
    }

    gpuLights(deCam) {
      const Z = this.depthZoom(deCam);
      const scored = this.lights.map((L) => {
        const d = Math.max(V.dist(L.pos, this.camPos), L.radius);
        // age fade: apparent brightness ~ (total zoom growth since spawn)^-1.1,
        // continuous in depth so rebases never blink; steep enough that lights
        // from levels you've left become glimmers, not ambient fill
        const fade = Math.pow(10, -1.1 * Math.max(0, Z - L.spawnZoom));
        // time fade-in after spawn (smoothstep) — no insta-illumination
        const a = L.fadeDur
          ? Math.min(1, Math.max(0, (this.timeSec - L.spawnT) / L.fadeDur)) : 1;
        const eff = L.intensity * fade * (a * a * (3 - 2 * a));
        return { L, eff, s: (eff / (d * d)) * Math.exp(-d * 1.2e-4 * this.fogMul) };
      });
      scored.sort((a, b) => b.s - a.s);
      // STICKY selection + ordering: only the top MAX_LIGHTS_GPU fit on the
      // GPU, and slots 0-1 cast shadows. Re-picking by raw score every frame
      // made lights (and their shadows) pop whenever two scores crossed, so
      // incumbents keep their place unless a challenger beats them by 35%.
      const sticky = new Set(this._gpuSel || []);
      const stickyTop = new Set(this._gpuTop || []);
      const rank = (x, set) => x.s * (set.has(x.L) ? 1.35 : 1);
      scored.sort((a, b) => rank(b, sticky) - rank(a, sticky));
      const sel = scored.slice(0, MAX_LIGHTS_GPU);
      sel.sort((a, b) => rank(b, stickyTop) - rank(a, stickyTop));
      this._gpuSel = sel.map((x) => x.L);
      this._gpuTop = sel.slice(0, 2).map((x) => x.L);
      return sel.map((x) => ({ ...x.L, intensity: x.eff }));
    }
  }

  const WorldMod = {
    World, evalFold, foldDelta, foldPreimage, wedgeNormalize, MAX_STAGES, ABS_REL2, OUTER_VEC4, LVL_VEC4,
    B_HARD, B_MIN, HORIZON, N_INNER, SLOTS, DESCEND_DE, ESCAPE2, MAX_LIGHTS_GPU,
    PHI, ICO_N, DIH_N1, dihedralOps, subM, subS,
  };
  global.WorldMod = WorldMod;
  if (typeof module !== 'undefined' && module.exports) module.exports = WorldMod;
})(typeof window !== 'undefined' ? window : globalThis);
