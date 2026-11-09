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
  const STYLE_ICOSA = 3;  // icosahedral kaleidoscope (H3)      -> rounded organic shells
  const STYLE_DIHEDRAL = 4; // n-fold radial kaleidoscope (D_n x Z2) + axial twist
                            //                                   -> spirals, petals, coral, arches
  const STYLE_BRANCH = 5;   // 1-2 mirror planes + strong rotation (KIFS "tree")
                            //                                   -> trunks, branches, roots, fronds

  /* DIHEDRAL geometry: order n (3..7), the copy center c = (cos a, sin a, h)
     on the wedge bisector a = pi/(2n) (copies form 2 rings of 2n), and the
     second mirror normal N2 = (sin(pi/n), -cos(pi/n), 0). */
  function dihedral(n, h) {
    const a = Math.PI / (2 * n);
    return {
      foldN: n,
      dihN: [Math.sin(Math.PI / n), -Math.cos(Math.PI / n), 0],
      pull: [Math.cos(a), Math.sin(a), h],
    };
  }
  const STYLE_TGLAD = 2;  // legacy alias

  /* Connectivity anchors: each style's attractor stays connected (solid walls
     rather than dust) only in a scale band matched to its copy count, and only
     when the translation pulls on EVERY axis. Do not widen these bands without
     visual verification — dust is the failure mode. */
  /* The critical scale where copies stop touching is s=2 for POLY (Sierpinski)
     and s=3 for MENGER-family; bands stay strictly below critical or the
     attractor disconnects. */
  const STYLE_ANCHOR = [
    { lo: 1.66, hi: 1.78, pull: [1.0, 1.0, 1.0] },   // POLY (tetra, ~4 copies). Measured
                                                     // solidity (ray hits): 90% at 1.70,
                                                     // <50% above 1.88 — sandy dust walls
    { lo: 2.50, hi: 2.95, pull: [1.0, 1.0, 0.52] },  // MENGER (many copies)
    { lo: 2.15, hi: 2.52, pull: [1.0, 0.82, 0.45] }, // OCTA
    { lo: 1.84, hi: 1.99, pull: [1.0, 0.9, 0.65] },  // ICOSA (stylecheck: solid at 1.86,
                                                     // stringy by 2.05 — keep below)
    { lo: 2.95, hi: 3.45, pull: null },              // DIHEDRAL (pull from dihedral(n, h))
    { lo: 1.5, hi: 1.9, pull: [1.0, 0.45, 0.6] },    // BRANCH (provisional — tuned visually)
  ];

  /* BRANCH presets (visually validated): sub-step scale range s, mirrors k,
     sub-steps per level m (level scale s^m), twist range (radians), pull y/z.
       cauliflower — romanesco / brain-coral clusters, fractal at every scale
       membrane    — smooth curved sheets torn with slits, alien skin
       lichen      — dense crinkled coral crust
     (Single-mirror "horn" forms looked great from outside but are thread-
     thin dust inside: excluded.)
     twMin / hr are hard limits on the EFFECTIVE per-level values (after the
     depth drift), measured as ray-hit density sweeps: cauliflower collapses
     to dust below twist ~1.15 (hit rate 0.3 -> 0.02 at 0.8), membrane above
     h ~0.7. Keep every drifted level inside these. */
  const BRANCH_PRESETS = [
    { name: 'cauliflower', k: 1, m: 3, s: [1.26, 1.3], tw: [1.2, 1.45], twMin: 1.15, py: 0.45, h: 0.6, hr: [0.48, 0.72] },
    { name: 'membrane', k: 2, m: 2, s: [1.28, 1.33], tw: [0.65, 0.9], twMin: 0.5, py: 0.45, h: 0.55, hr: [0.45, 0.62] },
    { name: 'lichen', k: 2, m: 1, s: [1.58, 1.66], tw: [0.85, 1.1], twMin: 0.65, py: 0.45, h: 0.6, hr: [0.48, 0.72] },
  ];

  // Biome: a run of consecutive levels sharing stylistic biases.
  // Deterministic biome segmentation: walk zone boundaries from a hash chain.
  const ZONE_MIN = 4, ZONE_VAR = 5;
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

  /* Curated color MOODS. Each is a harmonized scheme — surface, haze, sky,
     accent, sun, light color — instead of independent random hues: the old
     scheme made atmosphere and lights COMPLEMENTARY to the surfaces, and
     complementary colors mixed through fill light cancel to grey-brown mud.
     Hues in turns (0 red, .08 orange, .15 yellow, .33 green, .5 cyan,
     .6 blue, .75 violet, .85 magenta). Per biome a mood is jittered slightly,
     and zone edges blend continuously (levelParams), so every dive drifts
     through a sequence of distinct, coherent atmospheres. */
  const MOODS = [
    // surf [h,s,v]        haze [h,s,v]         sky [h,s,v]        accent  sun [h,s,k]       light  emis
    { surf: [0.10, 0.18, 0.62], haze: [0.50, 0.75, 0.035], sky: [0.52, 0.60, 0.09], acc: 0.40, sun: [0.10, 0.45, 1.5], light: 0.50, emis: 0.5 },  // abyssal teal & bone
    { surf: [0.07, 0.45, 0.55], haze: [0.99, 0.75, 0.030], sky: [0.02, 0.70, 0.07], acc: -0.07, sun: [0.06, 0.65, 1.7], light: 0.05, emis: 0.6 }, // ember cathedral
    { surf: [0.74, 0.25, 0.60], haze: [0.70, 0.70, 0.035], sky: [0.72, 0.60, 0.10], acc: 0.13, sun: [0.52, 0.35, 1.4], light: 0.85, emis: 0.45 }, // violet nebula
    { surf: [0.40, 0.45, 0.50], haze: [0.42, 0.65, 0.030], sky: [0.45, 0.50, 0.08], acc: -0.28, sun: [0.12, 0.25, 1.5], light: 0.12, emis: 0.35 }, // jade temple, gold
    { surf: [0.56, 0.18, 0.72], haze: [0.62, 0.70, 0.040], sky: [0.60, 0.55, 0.12], acc: -0.06, sun: [0.60, 0.30, 1.3], light: 0.55, emis: 0.3 },  // glacier, moonlight
    { surf: [0.06, 0.55, 0.50], haze: [0.55, 0.35, 0.030], sky: [0.55, 0.30, 0.08], acc: 0.40, sun: [0.10, 0.30, 1.5], light: 0.45, emis: 0.35 },  // copper & verdigris
    { surf: [0.08, 0.15, 0.24], haze: [0.08, 0.40, 0.015], sky: [0.08, 0.35, 0.04], acc: 0.04, sun: [0.10, 0.50, 1.3], light: 0.10, emis: 0.9 },   // obsidian & gold
    { surf: [0.98, 0.45, 0.65], haze: [0.60, 0.75, 0.035], sky: [0.58, 0.60, 0.10], acc: -0.48, sun: [0.12, 0.25, 1.4], light: 0.50, emis: 0.35 }, // coral & deep blue
    { surf: [0.09, 0.35, 0.62], haze: [0.78, 0.50, 0.030], sky: [0.80, 0.45, 0.09], acc: -0.12, sun: [0.05, 0.55, 1.6], light: 0.90, emis: 0.3 }, // sandstone dusk
  ];

  function biomeParams(seed, zoneIndex) {
    const zseed = hashCombine(seed, hashCombine(zoneIndex, 0xb10e));
    const r = (salt) => hash32(hashCombine(zseed, salt)) / 4294967296;
    const mood = MOODS[Math.floor(r(40) * MOODS.length)];
    const hue = mood.surf[0] + (r(1) - 0.5) * 0.06;
    const styleRoll = r(2);
    // Each biome commits to one style (mixing styles within a biome shreds
    // connectivity); variety across biomes comes from anchors + palettes.
    // DIHEDRAL gets the largest share: its porous, spiralling, cavernous
    // terrain is the most "natural"; the rest keep architectural variety
    // The organic families (BRANCH, DIHEDRAL) get most of the dive; the
    // architectural ones stay for contrast.
    const styleBias = styleRoll < 0.3 ? STYLE_BRANCH
      : styleRoll < 0.55 ? STYLE_DIHEDRAL
      : styleRoll < 0.7 ? STYLE_MENGER
      : styleRoll < 0.8 ? STYLE_POLY
      : styleRoll < 0.92 ? STYLE_ICOSA : STYLE_OCTA;
    const anchor = STYLE_ANCHOR[styleBias];
    // BRANCH biome: one of the visually validated presets, jittered
    const brP = styleBias === STYLE_BRANCH ? BRANCH_PRESETS[Math.floor(r(43) * BRANCH_PRESETS.length)] : null;
    const br = brP ? {
      name: brP.name,
      k: brP.k, m: brP.m,
      s: brP.s[0] + r(44) * (brP.s[1] - brP.s[0]),
      twist: brP.tw[0] + r(45) * (brP.tw[1] - brP.tw[0]),
      twMin: brP.twMin, hr: brP.hr,
      pull: [1.0, brP.py + (r(46) - 0.5) * 0.2, brP.h + (r(47) - 0.5) * 0.3],
    } : null;
    // DIHEDRAL biome shape: order n, ring height h (validated visually:
    // solid drums below s~2.8, porous tori/coral 2.95-3.5, spiral detail
    // from the axial twist)
    const dih = styleBias === STYLE_DIHEDRAL
      ? dihedral(4 + Math.floor(r(41) * 4), 0.3 + r(42) * 0.9) : null;
    return {
      hue,
      hueSpread: 0.06 + r(3) * 0.14,
      sat: Math.max(0.05, mood.surf[1] + (r(4) - 0.5) * 0.16),
      styleBias,
      anchor,
      scaleBase: br ? Math.pow(br.s, br.m) : anchor.lo + r(7) * (anchor.hi - anchor.lo),
      // rotation magnitude (radians) — small for the cubic styles (larger
      // tears them into dust); DIHEDRAL twists about its own axis, where
      // large angles stay connected and draw spirals across levels
      twist: br ? br.twist : dih ? 0.2 + r(6) * 0.3 : 0.015 + r(6) * 0.065,
      // BRANCH: the preset's validated axis, barely jittered — with a fold
      // IFS only the part of each copy inside the fold image survives, and
      // most random axes prune the tree down to dust (nothing to hit)
      twistAxis: br ? V.norm([0.3 + (r(18) - 0.5) * 0.08, 0.9 + (r(19) - 0.5) * 0.08, 0.2 + (r(20) - 0.5) * 0.08])
        : dih ? V.norm([(r(18) - 0.5) * 0.1, (r(19) - 0.5) * 0.1, 1])
        : V.norm([r(18) * 2 - 1, r(19) * 2 - 1, r(20) * 2 - 1]),
      dih,
      br,
      // phase of the smooth per-level shape drift (see levelParams)
      driftPhase: r(48) * 6.2832,
      spread: 0.97 + r(9) * 0.05,        // fold offset spread (kept near canonical)
      emissive: r(10) < mood.emis,       // biome has glowing veins
      // DARK atmosphere: deep, saturated haze. Brightness in the distance
      // comes from sunlight scattered in the haze (shader), not from a pale
      // sky color — pale fog washed every frame out to pastel.
      fog: hsv(mood.haze[0] + (r(11) - 0.5) * 0.05, mood.haze[1], mood.haze[2] * (0.8 + r(13) * 0.45)),
      sky: hsv(mood.sky[0] + (r(14) - 0.5) * 0.05, mood.sky[1], mood.sky[2] * (0.8 + r(16) * 0.45)),
      // the biome's sun: color/intensity from the mood, and a WORLD-anchored
      // direction (elevation/azimuth) — all interpolated
      // continuously with depth by World.mood()
      sun: hsv(mood.sun[0], mood.sun[1], 1).map((v) => v * mood.sun[2] * (0.85 + r(30) * 0.3)),
      sunW: (() => {
        const el = 0.25 + r(31) * 0.95, az = r(32) * 6.2832;
        return [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
      })(),
      // material accent: hue offset of the secondary (crevice / trap-region)
      // color relative to the level palette
      accent: mood.acc + (r(34) - 0.5) * 0.08,
      lightHue: mood.light + (r(17) - 0.5) * 0.08,
      // 0 = dark biome (no auto lights — sun, veins and the headlight carry
      // it), 1 = sparse, 2 = lit. Dark stretches are the point: pools of
      // light only mean something next to real darkness.
      lightMode: r(21) < 0.3 ? 0 : r(21) < 0.72 ? 1 : 2,
      valBase: mood.surf[2] + (r(22) - 0.5) * 0.12, // palette value anchor (smoothed per level)
      // haze extinction per decade of distance: clear biomes show colossal
      // structure far above; thick ones feel close and mysterious
      haze: 0.12 + r(28) * 0.22,
      // Offset fold planes (MENGER/OCTA only — see world.js evalFold): the
      // direction is a biome trait, the magnitude drifts per level. Band
      // validated via stylecheck: solid through 0.4*(s-1), eroding beyond.
      foldOffDir: V.norm([r(24) * 2 - 1, r(25) * 2 - 1, r(26) * 2 - 1]),
      foldOffMax: r(27) * 0.38,
    };
  }

  // shortest-arc hue interpolation (hue is circular in [0,1))
  function mixHue(a, b, t) {
    const d = ((b - a + 0.5) % 1 + 1) % 1 - 0.5;
    return a + d * t;
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
      const dh = style === STYLE_DIHEDRAL ? dihedral(o.n || 5, o.h !== undefined ? o.h : 0.5) : null;
      const br = style === STYLE_BRANCH;
      const pull = dh ? dh.pull : br ? [1.0, o.py !== undefined ? o.py : 0.45, o.h !== undefined ? o.h : 0.6] : anchor.pull;
      // DIHEDRAL twists about its symmetry axis (spirals); others about a fixed tilt
      const rot = twist ? V.mAxisAngle(dh ? [0, 0, 1] : V.norm([0.3, 0.9, 0.2]), twist) : V.mIdent();
      const foDir = V.norm([0.55, 0.8, 0.25]);
      const foMag = dh ? 0 : (o.foldoff || 0) * (s - 1);
      return {
        level: j, zone: 0, biome: biomeParams(seed, 0),
        style, scale: br && o.sub > 1 ? Math.pow(s, o.sub) : s, rot,
        trans: [-k * pull[0] * pullMod[0], -k * pull[1] * pullMod[1], -k * pull[2] * pullMod[2]],
        foldL: 1, foldN: dh ? dh.foldN : br ? Math.min(2, Math.max(1, o.n || 1)) : 0, dihN: dh ? dh.dihN : null,
        // BRANCH composite: ?sub=m repeats the step m times per level
        subM: br && o.sub > 1 ? o.sub : 1, subS: br && o.sub > 1 ? s : undefined,
        foldOff: [foDir[0] * foMag, foDir[1] * foMag, foDir[2] * foMag],
        pal: hsv(0.55 + 0.06 * (j % 4) + zone5 * 0.3, 0.5, 0.75), emissive: 0,
        light: { spawn: false, col: [1, 1, 1], off: [0, 0, 0], intensity: 0 },
      };
    }
    if (LevelGen.CANON) {
      // debug: canonical Menger sponge at every level (known connected)
      return {
        level: j, zone: 0, biome: biomeParams(seed, 0),
        style: STYLE_MENGER, scale: 3, rot: V.mIdent(),
        trans: [-2, -2, -1], foldL: 1, foldOff: [0, 0, 0],
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
    // Organic styles DRIFT smoothly with depth inside their zone (twist and
    // pull glide along a sine), so the forms keep evolving as you zoom
    // instead of one recipe repeating for the whole zone.
    const zlenD = Math.max(1, zone.end - zone.start);
    const drift = Math.sin(2 * Math.PI * (j - zone.start) / Math.max(4, zlenD) + biome.driftPhase);
    const organic = !!(biome.br || biome.dih);
    let ang = organic
      ? biome.twist * (1 + 0.18 * drift) * (0.95 + r(5) * 0.1)
      : biome.twist * (0.75 + r(5) * 0.5);
    if (biome.br) ang = Math.max(ang, biome.br.twMin);
    const rot = V.mAxisAngle(biome.twistAxis, ang);

    // translation: fixed point of the (sub-)step scale
    const k = ((biome.br ? biome.br.s : scale) - 1) * biome.spread;
    let pull = biome.br ? biome.br.pull : biome.dih ? biome.dih.pull : biome.anchor.pull;
    if (biome.br) {
      const hr = biome.br.hr;
      pull = [pull[0], pull[1] + 0.08 * drift, Math.min(hr[1], Math.max(hr[0], pull[2] - 0.1 * drift))];
    }
    const foldL = 1.0;
    const trans = [-k * pull[0], -k * pull[1], -k * pull[2]];

    // Offset fold planes: magnitude glides with zone position (like the
    // palette tracks) so the structural character morphs smoothly with depth
    const zlenO = Math.max(1, zone.end - zone.start);
    const ztO = (j - zone.start) / zlenO;
    const foMag = (style === STYLE_MENGER || style === STYLE_OCTA)
      ? biome.foldOffMax * (0.35 + 0.65 * Math.sin(ztO * 3.14159)) * (scale - 1)
      : 0;
    const foldOff = foMag
      ? [biome.foldOffDir[0] * foMag, biome.foldOffDir[1] * foMag, biome.foldOffDir[2] * foMag]
      : [0, 0, 0];

    // Palette: hue/sat/value tracks are CONTINUOUS across the whole dive.
    // Within a zone they glide along the biome's anchor; at zone edges they
    // pass through the midpoint with the neighbouring zone's anchor, so
    // biome changes are gradients too, never steps. (Independent per-level
    // jitter — the original scheme — made adjacent levels' colors jump.)
    const bPrev = biomeParams(seed, Math.max(0, zone.index - 1));
    const bNext = biomeParams(seed, zone.index + 1);
    const zlen = Math.max(1, zone.end - zone.start);
    const zt = (j - zone.start) / zlen; // 0..1 position within the zone
    const ease = (t) => t * t * (3 - 2 * t);
    let hue, sat, val;
    if (zt < 0.5) {
      const t = ease(zt * 2);
      hue = mixHue(mixHue(bPrev.hue, biome.hue, 0.5), biome.hue, t);
      sat = (bPrev.sat + biome.sat) / 2 + (biome.sat - (bPrev.sat + biome.sat) / 2) * t;
      val = (bPrev.valBase + biome.valBase) / 2 +
        (biome.valBase - (bPrev.valBase + biome.valBase) / 2) * t;
    } else {
      const t = ease((zt - 0.5) * 2);
      hue = mixHue(biome.hue, mixHue(biome.hue, bNext.hue, 0.5), t);
      sat = biome.sat + ((biome.sat + bNext.sat) / 2 - biome.sat) * t;
      val = biome.valBase + ((biome.valBase + bNext.valBase) / 2 - biome.valBase) * t;
    }
    hue += sr(14) * biome.hueSpread * 0.1; // tiny per-level texture only
    const pal = hsv(hue, Math.min(1, sat), val + (r(16) - 0.5) * 0.04);
    // emissive kept below ~1.6: higher values push the tonemapper into its
    // white plateau and whole vein-covered surfaces blow out
    const emissive = biome.emissive && r(17) < 0.5 ? 0.5 + r(18) * 1.1 : 0.0;

    // Point light spec for this level (spawned when the camera descends into
    // it). Density is a biome property: dark biomes spawn nothing, and even
    // lit ones stay sparse — pools of light against real darkness.
    const lightSpawn = r(20) < [0, 0.14, 0.3][biome.lightMode];
    const lightCol = hsv(biome.lightHue + sr(21) * 0.08, 0.55 + r(22) * 0.4, 1.0);
    const lightOff = [sr(23), sr(24) * 0.6 + 0.25, sr(25)]; // fallback spawn offset
    const lightInt = 0.08 + r(26) * 0.18;

    // accent hue offset glides across zone edges like the palette tracks
    const accent = zt < 0.5
      ? (bPrev.accent + biome.accent) / 2 + (biome.accent - (bPrev.accent + biome.accent) / 2) * ease(zt * 2)
      : biome.accent + ((biome.accent + bNext.accent) / 2 - biome.accent) * ease((zt - 0.5) * 2);

    // vein glow hue: the mood's light hue (vivid bioluminescence, never the
    // possibly-pale surface color — pale palettes glowed as white blotches)
    const glowHue = biome.lightHue + sr(35) * 0.03;

    return {
      level: j, zone: zone.index, biome, style, scale, rot, trans, foldL, foldOff,
      foldN: biome.dih ? biome.dih.foldN : biome.br ? biome.br.k : 0,
      dihN: biome.dih ? biome.dih.dihN : null,
      subM: biome.br ? biome.br.m : 1, subS: biome.br && biome.br.m > 1 ? biome.br.s : undefined,
      pal, emissive, accent, glowHue,
      light: { spawn: lightSpawn, col: lightCol, off: lightOff, intensity: lightInt },
    };
  }

  const LevelGen = { MOODS, hash32, hashCombine, rnd, srnd, hsv, mixHue, levelParams, zoneOf, biomeParams,
    STYLE_POLY, STYLE_MENGER, STYLE_OCTA, STYLE_ICOSA, STYLE_DIHEDRAL, STYLE_BRANCH, STYLE_TGLAD, STYLE_ANCHOR,
    dihedral, BRANCH_PRESETS };
  global.LevelGen = LevelGen;
  if (typeof module !== 'undefined' && module.exports) module.exports = LevelGen;
})(typeof window !== 'undefined' ? window : globalThis);
