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
   A ray point p (camera-relative, frame K) starts as d = W·p in frame K-B
   (W = product of inverse linear maps — a pure similarity, so tiny d keeps
   full *relative* float32 precision). Per outer level:
     |d| < m_j → d ← M_j·d                       (exact linear branch)
     else      → x = c_j + d; d ← Fold_j(x) − c_{j+1}   (true fold; the point is
                  far away, so f32 absolute error is sub-pixel at its distance)
   After the outer window, x = c_K + d is a true frame-K coordinate and inner
   levels K..K+N run the plain fold pipeline on O(1) numbers.

   Chain maintenance uses fold PREIMAGES: c_j is a preimage of c_{j+1} under
   Fold_j, chosen nearest to the previous frame's c_j (mirror choices are
   symmetry-equivalent, so an occasional forced switch is invisible). If
   c_{j+1} exits Fold_j's image entirely (camera flew "through a kaleidoscope
   mirror"), we reflect it back with the wedge normalization T — an exact
   symmetry of levels ≥ j+1 — lift T through the chain to frame K, and apply it
   to the camera, lights and sun. T → identity at the mirror itself, so the
   correction is seamless.
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
  const B_HARD = 24;
  /* Coupled to the renderer: the shader marches to tmax = TMAX/fogMul with
     TMAX = 2.4e5 and fogMul >= 0.85, and guarantees FULL fog by 0.92*tmax
     (horizon fade in shaders.js) — so the farthest visible geometry is at
     0.92 * 2.4e5 / 0.85 = 2.6e5 <= 2*HORIZON, and dropping an outer level is
     never visible. Raising HORIZON further trades test guarantees for range:
     a larger outer window increases fold-boundary double-crossings when
     transporting near-camera points across rebases (the outlier budget in
     test/world.test.js) and grows far-field f32 error (still sub-angular —
     see the tolerance model there — but re-validate before touching this). */
  const HORIZON = 1.35e5;
  const N_INNER = 16;     // inner detail levels
  const SLOTS = B_HARD + N_INNER;
  const ESCAPE2 = 900.0;  // escape radius^2 for the orbit
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

  /* ---- the fold: float64 reference implementation ------------------------
     rec=true records {M (linear action, row-major), margin (distance from the
     input point to the nearest branch boundary, in level-input units)}.
     MUST stay structurally identical to foldLevel() in shaders.js.          */
  function evalFold(xin, P, rec) {
    let x = V.mMulV(P.rot, xin);
    let M = rec ? P.rot.slice() : null;
    let margin = Infinity;
    const S = P.style;

    // Offset fold planes (MENGER/OCTA): the abs fold reflects across x_i=o_i
    // instead of the coordinate planes. Still a pure reflection per op — the
    // rebase machinery is untouched — but the copies no longer meet corner-on
    // -corner, which breaks the monotonous perfect self-similarity (this is
    // the Mandelbox box-fold trick). POLY/ICOSA keep concurrent planes: their
    // preimages enumerate a group orbit, which offsets would make infinite.
    const FO = P.foldOff || [0, 0, 0];
    if (S === LG.STYLE_MENGER) {
      if (rec) {
        margin = Math.min(margin,
          Math.abs(x[0] - FO[0]), Math.abs(x[1] - FO[1]), Math.abs(x[2] - FO[2]));
        M = V.mMul([Math.sign(x[0] - FO[0]) || 1, 0, 0, 0, Math.sign(x[1] - FO[1]) || 1, 0, 0, 0, Math.sign(x[2] - FO[2]) || 1], M);
      }
      x = [FO[0] + Math.abs(x[0] - FO[0]), FO[1] + Math.abs(x[1] - FO[1]), FO[2] + Math.abs(x[2] - FO[2])];
      if (rec) margin = Math.min(margin,
        Math.abs(x[0] - x[1]) / SQ2, Math.abs(x[0] - x[2]) / SQ2, Math.abs(x[1] - x[2]) / SQ2);
      if (x[0] < x[1]) { const t = x[0]; x[0] = x[1]; x[1] = t; if (rec) M = V.mMul([0,1,0, 1,0,0, 0,0,1], M); }
      if (x[0] < x[2]) { const t = x[0]; x[0] = x[2]; x[2] = t; if (rec) M = V.mMul([0,0,1, 0,1,0, 1,0,0], M); }
      if (x[1] < x[2]) { const t = x[1]; x[1] = x[2]; x[2] = t; if (rec) M = V.mMul([1,0,0, 0,0,1, 0,1,0], M); }
    } else if (S === LG.STYLE_POLY) {
      if (rec) margin = Math.min(margin, Math.abs(x[0] + x[1]) / SQ2);
      if (x[0] + x[1] < 0) { const t = -x[0]; x[0] = -x[1]; x[1] = t; if (rec) M = V.mMul([0,-1,0, -1,0,0, 0,0,1], M); }
      if (rec) margin = Math.min(margin, Math.abs(x[0] + x[2]) / SQ2);
      if (x[0] + x[2] < 0) { const t = -x[0]; x[0] = -x[2]; x[2] = t; if (rec) M = V.mMul([0,0,-1, 0,1,0, -1,0,0], M); }
      if (rec) margin = Math.min(margin, Math.abs(x[1] + x[2]) / SQ2);
      if (x[1] + x[2] < 0) { const t = -x[1]; x[1] = -x[2]; x[2] = t; if (rec) M = V.mMul([1,0,0, 0,0,-1, 0,-1,0], M); }
    } else if (S === LG.STYLE_ICOSA) {
      // H3 kaleidoscope: two rounds of (abs, golden-plane fold)
      for (let round = 0; round < 2; round++) {
        if (rec) {
          margin = Math.min(margin, Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2]));
          M = V.mMul([Math.sign(x[0]) || 1, 0, 0, 0, Math.sign(x[1]) || 1, 0, 0, 0, Math.sign(x[2]) || 1], M);
        }
        x = [Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2])];
        const t = x[0] * ICO_N[0] + x[1] * ICO_N[1] + x[2] * ICO_N[2];
        if (rec) margin = Math.min(margin, Math.abs(t));
        if (t < 0) {
          x = [x[0] - 2 * t * ICO_N[0], x[1] - 2 * t * ICO_N[1], x[2] - 2 * t * ICO_N[2]];
          if (rec) M = V.mMul(ICO_HOUSE, M);
        }
      }
    } else {
      // OCTA: (offset) abs fold followed by the two diagonal plane folds
      if (rec) {
        margin = Math.min(margin,
          Math.abs(x[0] - FO[0]), Math.abs(x[1] - FO[1]), Math.abs(x[2] - FO[2]));
        M = V.mMul([Math.sign(x[0] - FO[0]) || 1, 0, 0, 0, Math.sign(x[1] - FO[1]) || 1, 0, 0, 0, Math.sign(x[2] - FO[2]) || 1], M);
      }
      x = [FO[0] + Math.abs(x[0] - FO[0]), FO[1] + Math.abs(x[1] - FO[1]), FO[2] + Math.abs(x[2] - FO[2])];
      if (rec) margin = Math.min(margin, Math.abs(x[0] - x[1]) / SQ2);
      if (x[0] < x[1]) { const t = x[0]; x[0] = x[1]; x[1] = t; if (rec) M = V.mMul([0,1,0, 1,0,0, 0,0,1], M); }
      if (rec) margin = Math.min(margin, Math.abs(x[1] - x[2]) / SQ2);
      if (x[1] < x[2]) { const t = x[1]; x[1] = x[2]; x[2] = t; if (rec) M = V.mMul([1,0,0, 0,0,1, 0,1,0], M); }
    }

    x = [x[0] * P.scale + P.trans[0], x[1] * P.scale + P.trans[1], x[2] * P.scale + P.trans[2]];
    if (rec) M = V.mScale(M, P.scale);
    return rec ? { x, M, margin } : { x };
  }

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
      // POLY / ICOSA: BFS orbit of w under the style's mirror generators
      // (both groups are finite — order 24 and 120 — so the orbit closes)
      const R = S === LG.STYLE_ICOSA
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
      this.outer = [];                      // per-frame chain, outer[0] = level K-Beff
      this.W = V.mIdent();
      this.WScale = 1;
      this.log10Zoom = 0;
      this.sunDir = V.norm([0.55, 0.75, 0.35]);
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
      this.updateChain();
    }

    params(j) {
      let p = this._pcache.get(j);
      if (!p) { p = LG.levelParams(this.seed, j); this._pcache.set(j, p); }
      return p;
    }

    _applyAffine(aff) {
      // apply a frame-K affine symmetry to everything world-anchored
      this.camPos = V.add(V.mMulV(aff.M, this.camPos), aff.b);
      for (const L of this.lights) L.pos = V.add(V.mMulV(aff.M, L.pos), aff.b);
      this.sunDir = V.norm(V.mMulV(aff.M, this.sunDir));
      this.pendingRot = V.mMul(this._rotPart(aff.M), this.pendingRot);
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
        const pre = foldPreimage(cNext, P, hint);
        if (!pre) {
          // Camera exited Fold_j's image: reflect back (symmetry of levels ≥ j+1),
          // lifted from frame j+1 to frame K through the already-known affines,
          // transforming each frame's chain hint on the way up.
          let T = wedgeNormalize(cNext, P);
          for (let bb = b - 1; bb >= 0; bb--) {
            const m = this.K - 1 - bb; // T is currently expressed in frame m
            if (this.hints[m]) this.hints[m] = V.add(V.mMulV(T.M, this.hints[m]), T.b);
            const A = affs[bb]; // frame m -> m+1
            const invAM = V.mInv(A.M);
            const M = V.mMul(A.M, V.mMul(T.M, invAM));
            const tb = V.add(V.mMulV(A.M, V.add(V.mMulV(T.M, V.mMulV(invAM, V.scale(A.b, -1))), T.b)), A.b);
            T = { M, b: tb };
          }
          this._applyAffine(T);
          return false; // restart the walk with the corrected camera
        }
        const r = evalFold(pre, P, true);
        mag *= P.scale;
        full.push({
          level: j, c: pre, M: r.M, margin: Math.max(0, r.margin * 0.999),
          scale: P.scale, crossing: r.margin * mag,
        });
        affs.push({ M: r.M, b: V.sub(r.x, V.mMulV(r.M, pre)) });
        this.hints[j] = pre;
        cNext = pre;
      }
      // Window size: keep at least B_MIN levels, and keep any older level whose
      // fold boundaries pass within the horizon (dropping it would visibly
      // change near geometry).
      let Beff = Math.min(this.K, B_MIN);
      for (let b = full.length - 1; b >= Beff; b--) {
        if (full[b].crossing < HORIZON * 2) { Beff = b + 1; break; }
      }
      this.outer.length = 0;
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
      let RLin = Infinity;
      let cum = WScale;
      for (let b = 0; b < this.outer.length; b++) {
        RLin = Math.min(RLin, this.outer[b].margin / cum);
        cum *= this.outer[b].scale;
      }
      this.RLin = this.outer.length ? RLin : Infinity;
      // Attractor bounding radius over the whole active window: the invariant
      // ball satisfies R >= |t_j|/(s_j - 1) for every level. The DE formula
      // (|x| - R)/dr is a rigorous lower bound for ANY truncation depth
      // (cone-break LOD, escape), which is what keeps coarse LOD hole-free.
      let bound = 2.0;
      for (let j = Math.max(0, this.K - this.outer.length); j <= this.K + N_INNER; j++) {
        const P = this.params(j);
        const t = Math.hypot(P.trans[0], P.trans[1], P.trans[2]);
        // offset folds displace by up to 2|o| before scaling: |Fold(x)| <=
        // s*(|x| + 2|o|) + |t|, so the invariant ball grows accordingly
        const fo = P.foldOff ? Math.hypot(P.foldOff[0], P.foldOff[1], P.foldOff[2]) : 0;
        bound = Math.max(bound, (t + 2 * P.scale * fo) / (P.scale - 1));
      }
      this.bound = bound * 1.05;
      return true;
    }

    /* Float64 distance estimator at frame-K point p (absolute frame-K coords).
       forceSlow skips the near-field fast path (used by tests to validate it). */
    de(p, forceSlow) {
      let x, dr;
      let escaped = false;
      if (this.outer.length) {
        const rel = V.sub(p, this.camPos);
        if (!forceSlow && V.len(rel) < this.RLin) {
          // all outer levels act linearly here and compose to the identity
          x = p.slice();
          dr = 1;
        } else {
          const d = V.mMulV(this.W, rel);
          x = V.add(this.outer[0].c, d);
          dr = this.WScale;
          for (let b = 0; b < this.outer.length; b++) {
            x = evalFold(x, this.params(this.outer[b].level)).x;
            dr *= this.outer[b].scale;
            if (V.dot(x, x) > ESCAPE2) { escaped = true; break; }
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
          if (V.dot(x, x) > ESCAPE2) break;
        }
      }
      return (V.len(x) - this.bound) / dr;
    }

    grad(p, h) {
      h = h || Math.max(1e-9, Math.abs(this.de(p)) * 1e-3);
      return V.norm([
        this.de([p[0] + h, p[1], p[2]]) - this.de([p[0] - h, p[1], p[2]]),
        this.de([p[0], p[1] + h, p[2]]) - this.de([p[0], p[1] - h, p[2]]),
        this.de([p[0], p[1], p[2] + h]) - this.de([p[0], p[1], p[2] - h]),
      ]);
    }

    /* ---- rebasing -------------------------------------------------------- */
    descend() {
      const P = this.params(this.K);
      const r = evalFold(this.camPos, P, true);
      this.hints[this.K] = this.camPos;
      const s = P.scale, s2 = s * s;
      // x s^2 keeps the rebase photometrically EXACT (no blink); the age fade
      // is applied continuously at upload time in gpuLights().
      for (const L of this.lights) {
        L.pos = evalFold(L.pos, P).x;
        L.radius *= s;
        L.intensity *= s2;
      }
      this.sunDir = V.norm(V.mMulV(r.M, this.sunDir));
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
      let pre = foldPreimage(this.camPos, P, this.hints[this.K - 1] || this.camPos);
      if (!pre) {
        this._applyAffine(wedgeNormalize(this.camPos, P));
        pre = foldPreimage(this.camPos, P, this.hints[this.K - 1] || this.camPos);
        if (!pre) return false; // shouldn't happen
      }
      const r = evalFold(pre, P, true); // exact affine at the preimage
      const invM = V.mInv(r.M);
      const s = P.scale, s2 = s * s;
      const bb = V.sub(r.x, V.mMulV(r.M, pre));
      for (const L of this.lights) {
        L.pos = V.mMulV(invM, V.sub(L.pos, bb));
        L.radius /= s;
        L.intensity /= s2;
      }
      this.sunDir = V.norm(V.mMulV(invM, this.sunDir));
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
         vec4 lvl[SLOTS*6]  — per level slot: rotCol0(w=scale), rotCol1(w=style),
                              rotCol2(w=foldL), trans(w=emissive), pal,
                              foldOff — see packUBO for exact packing
         vec4 outer[B_HARD*4] — per outer slot: jacCol0, jacCol1, jacCol2, (c.xyz, margin)
       Slot i covers absolute level (K - Beff + i).                            */
    packUBO(buf) {
      const B = this.outer.length;
      const f = buf;
      for (let i = 0; i < SLOTS; i++) {
        const j = this.K - B + i;
        const P = this.params(Math.max(0, j));
        const o = i * 6 * 4;
        const rc = V.mToCols(P.rot);
        f[o + 0] = rc[0]; f[o + 1] = rc[1]; f[o + 2] = rc[2]; f[o + 3] = P.scale;
        f[o + 4] = rc[3]; f[o + 5] = rc[4]; f[o + 6] = rc[5]; f[o + 7] = P.style;
        f[o + 8] = rc[6]; f[o + 9] = rc[7]; f[o + 10] = rc[8]; f[o + 11] = P.foldL;
        f[o + 12] = P.trans[0]; f[o + 13] = P.trans[1]; f[o + 14] = P.trans[2]; f[o + 15] = P.emissive;
        f[o + 16] = P.pal[0]; f[o + 17] = P.pal[1]; f[o + 18] = P.pal[2]; f[o + 19] = 0;
        const FO = P.foldOff || [0, 0, 0];
        f[o + 20] = FO[0]; f[o + 21] = FO[1]; f[o + 22] = FO[2]; f[o + 23] = 0;
      }
      const base = SLOTS * 6 * 4;
      for (let b = 0; b < B_HARD; b++) {
        const o = base + b * 4 * 4;
        if (b < B) {
          const E = this.outer[b];
          const jc = V.mToCols(E.M);
          f[o + 0] = jc[0]; f[o + 1] = jc[1]; f[o + 2] = jc[2]; f[o + 3] = 0;
          f[o + 4] = jc[3]; f[o + 5] = jc[4]; f[o + 6] = jc[5]; f[o + 7] = 0;
          f[o + 8] = jc[6]; f[o + 9] = jc[7]; f[o + 10] = jc[8]; f[o + 11] = 0;
          f[o + 12] = E.c[0]; f[o + 13] = E.c[1]; f[o + 14] = E.c[2]; f[o + 15] = E.margin;
        } else {
          for (let k = 0; k < 16; k++) f[o + k] = 0;
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
      const target = Math.pow(sK, f);
      this.fogMul += (target - this.fogMul) * (1 - Math.exp(-(dt || 0.016) * 2.5));
      // floor 0.85: the level-drop-invisibility bound (see HORIZON) assumes
      // the effective march limit never exceeds TMAX/0.85
      this.fogMul = Math.min(Math.max(this.fogMul, 0.85), sK * 1.6);
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
      return { sky: V.lerp(a.sky, b.sky, fr), fog: V.lerp(a.fog, b.fog, fr) };
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
      return { sky: avg('sky'), fog: avg('fog') };
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
    World, evalFold, foldPreimage, wedgeNormalize,
    B_HARD, B_MIN, HORIZON, N_INNER, SLOTS, DESCEND_DE, ESCAPE2, MAX_LIGHTS_GPU,
    PHI, ICO_N,
  };
  global.WorldMod = WorldMod;
  if (typeof module !== 'undefined' && module.exports) module.exports = WorldMod;
})(typeof window !== 'undefined' ? window : globalThis);
