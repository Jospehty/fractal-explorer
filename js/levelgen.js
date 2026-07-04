/* levelgen.js — deterministic per-level fractal parameters.
   Every absolute depth level j gets its own fold recipe (style, rotation, scale,
   translation, fold offsets), palette, and lighting/mood data, all derived from
   hash(seed, j). Levels are grouped into "biomes" (zones of consecutive levels)
   that share stylistic biases, so the character of the terrain shifts as you
   descend. Parameters are consumed both by the CPU world model (float64) and,
   via uniform upload, by the GPU shader — there is one source of truth, so no
   CPU/GPU consistency issues can arise from parameter generation. */
(function (global) {
  'use strict';
  const V = global.V || require('./vec.js');

  // ---- integer hashing (deterministic across platforms: uint32 ops only) ----
  function hash32(x) {
    x = x >>> 0;
    x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
    x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    return x;
  }
  function hashCombine(a, b) {
    return hash32((a ^ (Math.imul(b, 0x9e3779b9) >>> 0)) >>> 0);
  }
  // rand in [0,1) from (seed, level, salt)
  function rnd(seed, level, salt) {
    const h = hash32(hashCombine(hashCombine(seed >>> 0, level >>> 0), salt >>> 0));
    return h / 4294967296;
  }
  // symmetric rand in [-1,1)
  function srnd(seed, level, salt) {
    return rnd(seed, level, salt) * 2 - 1;
  }

  // ---- palettes / moods ----
  function hsv(h, s, v) {
    h = ((h % 1) + 1) % 1;
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, v, p];
      default: return [v, p, q];
    }
  }

  const STYLE_POLY = 0;   // diagonal plane folds (Sierpinski)  -> crystalline temples
  const STYLE_MENGER = 1; // abs + full sort                    -> architectural voids
  const STYLE_OCTA = 2;   // abs + partial sort                 -> canyon lattices
  const STYLE_TGLAD = 2;  // legacy alias

  /* Connectivity anchors: each style's attractor stays connected (solid walls
     rather than dust) only in a scale band matched to its copy count, and only
     when the translation pulls on EVERY axis. Do not widen these bands without
     visual verification — dust is the failure mode. */
  /* The critical scale where copies stop touching is s=2 for POLY (Sierpinski)
     and s=3 for MENGER-family; bands stay strictly below critical or the
     attractor disconnects. */
  const STYLE_ANCHOR = [
    { lo: 1.78, hi: 1.97, pull: [1.0, 1.0, 1.0] },   // POLY (tetra, ~4 copies)
    { lo: 2.50, hi: 2.95, pull: [1.0, 1.0, 0.52] },  // MENGER (many copies)
    { lo: 2.15, hi: 2.52, pull: [1.0, 0.82, 0.45] }, // OCTA
  ];

  // Biome: a run of consecutive levels sharing stylistic biases.
  // Deterministic biome segmentation: walk zone boundaries from a hash chain.
  const ZONE_MIN = 5, ZONE_VAR = 6;
  function zoneOf(seed, j) {
    // Zone lengths are hash(zoneIndex)-dependent; find which zone j falls in by
    // walking from level 0. Cache per seed for O(1) amortized lookups.
    if (!zoneOf._cache || zoneOf._seed !== seed) {
      zoneOf._cache = [0];
      zoneOf._seed = seed;
    }
    const c = zoneOf._cache;
    while (c[c.length - 1] <= j) {
      const zi = c.length - 1;
      const len = ZONE_MIN + Math.floor(rnd(seed, zi, 901) * ZONE_VAR);
      c.push(c[c.length - 1] + len);
    }
    let lo = 0, hi = c.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (c[mid] <= j) lo = mid; else hi = mid;
    }
    return { index: lo, start: c[lo], end: c[hi] };
  }

  function biomeParams(seed, zoneIndex) {
    const zseed = hashCombine(seed, hashCombine(zoneIndex, 0xb10e));
    const r = (salt) => hash32(hashCombine(zseed, salt)) / 4294967296;
    const hue = r(1);
    const styleRoll = r(2);
    // Each biome commits to one style (mixing styles within a biome shreds
    // connectivity); variety across biomes comes from anchors + palettes.
    const styleBias = styleRoll < 0.4 ? STYLE_MENGER : styleRoll < 0.75 ? STYLE_POLY : STYLE_OCTA;
    const anchor = STYLE_ANCHOR[styleBias];
    return {
      hue,
      hueSpread: 0.06 + r(3) * 0.14,
      sat: 0.35 + r(4) * 0.5,
      styleBias,
      anchor,
      scaleBase: anchor.lo + r(7) * (anchor.hi - anchor.lo),
      twist: 0.015 + r(6) * 0.065,       // rotation magnitude (radians) — small!
      twistAxis: V.norm([r(18) * 2 - 1, r(19) * 2 - 1, r(20) * 2 - 1]),
      spread: 0.97 + r(9) * 0.05,        // fold offset spread (kept near canonical)
      emissive: r(10) < 0.42,            // biome has glowing veins
      fog: hsv(hue + 0.5 + (r(11) - 0.5) * 0.2, 0.45 + r(12) * 0.35, 0.09 + r(13) * 0.2),
      sky: hsv(hue + 0.45 + (r(14) - 0.5) * 0.3, 0.4 + r(15) * 0.4, 0.2 + r(16) * 0.35),
      lightHue: hue + 0.33 + r(17) * 0.34,
    };
  }

  // Full parameter record for one absolute level j (j >= 0).
  function levelParams(seed, j) {
    if (LevelGen.OVERRIDE) {
      // debug: force styles/params to tune connectivity anchors & transitions
      const o = LevelGen.OVERRIDE;
      const zone5 = Math.floor(j / 5) % 2;
      let style = o.style;
      let twist = o.twist || 0;
      let pullMod = [1, 1, 1];
      if (o.alt === 1) style = zone5 ? (o.style2 || 1) : o.style;      // alternate styles
      if (o.alt === 2) {                                               // same style, alt params
        twist = zone5 ? (o.twist || 0) + 0.05 : (o.twist || 0);
        pullMod = zone5 ? [1, 0.9, 1.12] : [1, 1, 1];
      }
      const anchor = STYLE_ANCHOR[style];
      const s = o.scale && !o.alt ? o.scale : (anchor.lo + anchor.hi) / 2;
      const k = (s - 1) * (o.spread || 1);
      const rot = twist ? V.mAxisAngle(V.norm([0.3, 0.9, 0.2]), twist) : V.mIdent();
      return {
        level: j, zone: 0, biome: biomeParams(seed, 0),
        style, scale: s, rot,
        trans: [-k * anchor.pull[0] * pullMod[0], -k * anchor.pull[1] * pullMod[1], -k * anchor.pull[2] * pullMod[2]],
        foldL: 1,
        pal: hsv(0.55 + 0.06 * (j % 4) + zone5 * 0.3, 0.5, 0.75), emissive: 0,
        light: { spawn: false, col: [1, 1, 1], off: [0, 0, 0], intensity: 0 },
      };
    }
    if (LevelGen.CANON) {
      // debug: canonical Menger sponge at every level (known connected)
      return {
        level: j, zone: 0, biome: biomeParams(seed, 0),
        style: STYLE_MENGER, scale: 3, rot: V.mIdent(),
        trans: [-2, -2, -1], foldL: 1,
        pal: hsv(0.08 + 0.02 * (j % 5), 0.5, 0.8), emissive: 0,
        light: { spawn: j % 3 === 0, col: hsv(0.6, 0.6, 1), off: [0.5, 0.5, 0], intensity: 3 },
      };
    }
    const zone = zoneOf(seed, j);
    const biome = biomeParams(seed, zone.index);
    const r = (salt) => rnd(seed, j, salt);
    const sr = (salt) => srnd(seed, j, salt);

    const style = biome.styleBias;

    // Scale and translation are held EXACTLY at the biome anchor: numerical
    // porosity tests showed even ±2% per-level translation jitter erodes the
    // attractor into a sieve (the copy-covering condition is tight). Per-level
    // geometric variety comes from the rotation amount; everything else varies
    // per biome.
    const scale = biome.scaleBase;

    // Rotation: a coherent twist about the biome's FIXED axis. Per-level axis
    // wobble and large angles tear the attractor into dust — verified visually;
    // vary the amount only, gently, around the biome magnitude.
    const ang = biome.twist * (0.75 + r(5) * 0.5);
    const rot = V.mAxisAngle(biome.twistAxis, ang);

    const k = (scale - 1) * biome.spread;
    const pull = biome.anchor.pull;
    const foldL = 1.0;
    const trans = [-k * pull[0], -k * pull[1], -k * pull[2]];

    // Palette: biome hue drifting per level; deeper values, saturated
    const hue = biome.hue + sr(14) * biome.hueSpread;
    const pal = hsv(hue, Math.min(1, biome.sat * (0.9 + r(15) * 0.5)), 0.34 + r(16) * 0.42);
    const emissive = biome.emissive && r(17) < 0.5 ? 0.6 + r(18) * 2.2 : 0.0;

    // Point light spec for this level (spawned when the camera descends into it).
    const lightSpawn = r(20) < 0.35;
    const lightCol = hsv(biome.lightHue + sr(21) * 0.08, 0.55 + r(22) * 0.4, 1.0);
    const lightOff = [sr(23), sr(24) * 0.6 + 0.25, sr(25)]; // camera-relative spawn offset
    const lightInt = 0.05 + r(26) * 0.2;

    return {
      level: j, zone: zone.index, biome, style, scale, rot, trans, foldL,
      pal, emissive,
      light: { spawn: lightSpawn, col: lightCol, off: lightOff, intensity: lightInt },
    };
  }

  const LevelGen = { hash32, hashCombine, rnd, srnd, hsv, levelParams, zoneOf, biomeParams,
    STYLE_POLY, STYLE_MENGER, STYLE_OCTA, STYLE_TGLAD, STYLE_ANCHOR };
  global.LevelGen = LevelGen;
  if (typeof module !== 'undefined' && module.exports) module.exports = LevelGen;
})(typeof window !== 'undefined' ? window : globalThis);
