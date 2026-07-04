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

  // Constants injected to match world.js
  const DEFS = (WM) => `
  #define B_HARD ${WM.B_HARD}
  #define N_INNER ${WM.N_INNER}
  #define SLOTS ${WM.SLOTS}
  #define ESCAPE2 ${WM.ESCAPE2.toFixed(1)}
  #define MAX_LIGHTS ${WM.MAX_LIGHTS_GPU}
  `;

  const FRAG = (WM) => `#version 300 es
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
  uniform float uBound;
  uniform vec2  uJitter;     // sub-pixel jitter for still-frame accumulation
  uniform float uFogMul;     // fog continuity multiplier (world.fogMul): scales
                             // fog density up / march limit down as the camera
                             // progresses through a level, so optical depth is
                             // continuous across the rebase rescale

  layout(std140) uniform Params {
    vec4 lvl[SLOTS * 6];        // per slot: rot0(w=scale) rot1(w=style) rot2(w=foldL) trans(w=emissive) pal spare
    vec4 outerData[B_HARD * 4]; // per outer: jac0 jac1 jac2 (c.xyz, margin)
  };

  /* ---- the fold (mirror of world.js evalFold) ---- */
  vec3 foldLevel(vec3 x, int s) {
    vec4 r0 = lvl[s*6+0], r1 = lvl[s*6+1], r2 = lvl[s*6+2], tr = lvl[s*6+3];
    x = mat3(r0.xyz, r1.xyz, r2.xyz) * x;
    float style = r1.w;
    if (style < 0.5) {            // POLY: diagonal plane folds
      if (x.x + x.y < 0.0) x.xy = -x.yx;
      if (x.x + x.z < 0.0) x.xz = -x.zx;
      if (x.y + x.z < 0.0) x.yz = -x.zy;
    } else if (style < 1.5) {     // MENGER: abs + sort desc
      x = abs(x);
      if (x.x < x.y) x.xy = x.yx;
      if (x.x < x.z) x.xz = x.zx;
      if (x.y < x.z) x.yz = x.zy;
    } else {                      // OCTA: abs + partial sort
      x = abs(x);
      if (x.x < x.y) x.xy = x.yx;
      if (x.y < x.z) x.yz = x.zy;
    }
    return x * r0.w + tr.xyz;
  }

  /* ---- two-phase distance estimator ----
     p: camera-relative frame-K position. coneEps: feature size below which we
     band-limit (adaptive iteration count).
     trapOut: (slotA, trapA, slotB, trapB) — the two smallest orbit traps and
     their owning level slots. Shading crossfades the two slots' palettes by
     trap separation, so colors blend smoothly where trap ownership changes
     instead of snapping (both across surfaces and while zooming).
     planeTrap: independent plane trap that drives the emissive veins. */
  float map(vec3 p, float coneEps, out vec4 trapOut, out float planeTrap) {
    vec3 x;
    float dr;
    bool esc = false;
    float m1 = 1e9, m2 = 1e9;            // two smallest traps
    float s1 = float(uB), s2 = float(uB); // and their slots
    float trapP = 1e9; // plane trap: varies across surfaces (emissive veins)
    if (uB > 0) {
      vec3 d = uW * p;
      dr = uWScale;
      for (int b = 0; b < B_HARD; b++) {
        if (b >= uB) break;
        vec4 cm = outerData[b*4+3];
        vec3 cn = (b + 1 < uB) ? outerData[(b+1)*4+3].xyz : uCamK;
        if (dot(d, d) < cm.w * cm.w) {
          d = mat3(outerData[b*4].xyz, outerData[b*4+1].xyz, outerData[b*4+2].xyz) * d;
          dr *= lvl[b*6].w;
        } else {
          vec3 xx = foldLevel(cm.xyz + d, b);
          dr *= lvl[b*6].w;
          if (dot(xx, xx) > ESCAPE2) { x = xx; esc = true; break; }
          d = xx - cn;
        }
        // Trap sampling through ALL outer levels keeps coloring anchored to
        // ABSOLUTE levels: any fixed-size sampling window would slide at each
        // rebase and re-color the surfaces owned by the dropped level.
        {
          vec3 xs = cn + d;
          float ms = length(xs) * 0.9 + 0.12 * abs(xs.y);
          if (ms < m1) { m2 = m1; s2 = s1; m1 = ms; s1 = float(b); }
          else if (ms < m2) { m2 = ms; s2 = float(b); }
          trapP = min(trapP, abs(xs.z + 0.4 * xs.x - 0.35));
        }
      }
      if (!esc) x = uCamK + d;
    } else {
      x = uCamK + p;
      dr = 1.0;
    }
    if (!esc) {
      for (int i = 0; i < N_INNER; i++) {
        int s = uB + i;
        x = foldLevel(x, s);
        dr *= lvl[s*6].w;
        float xx2 = dot(x, x);
        float m = sqrt(xx2) * 0.9 + 0.12 * abs(x.y); // slight anisotropy for banding
        if (m < m1) { m2 = m1; s2 = s1; m1 = m; s1 = float(s); }
        else if (m < m2) { m2 = m; s2 = float(s); }
        trapP = min(trapP, abs(x.z + 0.4 * x.x - 0.35));
        if (xx2 > ESCAPE2) break;
        ${global.__SHADER_DEBUG === 'nocone' ? '' : 'if (1.0 < dr * coneEps) break;'}
      }
    }
    trapOut = vec4(s1, m1, s2, m2);
    planeTrap = trapP;
    // (|x| - R)/dr is a rigorous distance lower bound for any truncation depth
    // (R = attractor bounding radius, uploaded per frame) — hole-free LOD.
    return (length(x) - uBound) / dr;
  }

  float mapD(vec3 p, float coneEps) { vec4 t; float pl; return map(p, coneEps, t, pl); }

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
      float h = mapD(ro + rd * t, coneEps + t * 2e-3);
      if (h < 1e-6 * t) return 0.0;
      res = min(res, k * h / t);
      t += clamp(h, t0 * 0.5, tmax * 0.12);
    }
    return clamp(res, 0.0, 1.0);
  }

  float calcAO(vec3 p, vec3 n, float scale, float coneEps) {
    float occ = 0.0, sca = 1.0;
    for (int i = 0; i < 5; i++) {
      float h = (0.01 + 0.13 * float(i)) * scale;
      float d = mapD(p + n * h, coneEps + h * 0.2);
      occ += (h - d) * sca / scale;
      sca *= 0.72;
    }
    return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
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

  vec3 skyColor(vec3 rd) {
    float sd = max(dot(rd, uSunDir), 0.0);
    vec3 sky = mix(uFogCol * 0.9, uSkyCol, clamp(rd.y * 0.5 + 0.55, 0.0, 1.0));
    sky += uSunCol * (pow(sd, 320.0) * 3.2 + pow(sd, 16.0) * 0.18);
    return sky;
  }

  vec3 palette(int slot) { return lvl[slot*6+4].rgb; }
  float emissiveOf(int slot) { return lvl[slot*6+3].w; }

  vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
  }

  float hash12(vec2 v) {
    vec3 p3 = fract(vec3(v.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  uniform float uProbeOn;
  uniform vec3  uProbe;

  void main() {
    if (uProbeOn > 0.5) {
      // debug: output map(uProbe) bit-exactly, one byte per pixel in R
      // (alpha channel is unusable on an opaque canvas)
      vec4 tt; float tp;
      float dv = map(uProbe, 0.0, tt, tp);
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

    const float TMAX = 2.4e5;
    float tmax = TMAX / uFogMul; // effective horizon, continuous across rebases
    int   MAXSTEP = uQuality == 0 ? 100 : (uQuality == 1 ? 150 : 210);
    float t = 0.0;
    float d = 0.0;
    vec4 trap = vec4(0.0);
    float trapPl = 1e9;
    bool hit = false;
    bool exhausted = false;
    int steps = 0;
    for (int i = 0; i < 256; i++) {
      if (t > tmax) break;
      if (i >= MAXSTEP) { hit = true; exhausted = true; break; } // grazing geometry
      steps = i;
      vec3 p = ro + rd * t;
      d = map(p, t * pix * 0.22, trap, trapPl);
      if (d < t * pix * 0.45 + 2e-6) { hit = true; break; }
      t += min(d, tmax * 0.5) * 0.92;
    }
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
      float coneEps = t * pix * 0.22;
      float eps = max(t * pix * 0.4, 2e-6);
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
      vec3 alb = mix(palette(slotB), palette(slot), wA);
      // trap-based tone variation: crevices darken, ridges lighten
      float tv = clamp(trap.y * 0.7, 0.0, 1.4);
      alb *= 0.45 + 0.62 * tv;

      float ao = calcAO(p, n, max(t * 0.05, 0.02), coneEps);
      float ao2 = ao * ao;
      float sunDiff = max(dot(n, uSunDir), 0.0);
      float sh = 1.0;
      if (uQuality >= 1 && sunDiff > 0.001) {
        sh = softShadow(p + n * eps * 3.0, uSunDir, eps * 8.0, 40.0, 9.0, coneEps, uQuality == 1 ? 40 : 64);
      }
      vec3 hv = normalize(uSunDir - rd);
      float spec = pow(max(dot(n, hv), 0.0), 42.0) * 0.6;

      ${global.__SHADER_DEBUG === 'shade'
        ? 'fragColor = vec4(dot(alb, vec3(0.33)), sunDiff * sh, ao, 1.0); return;'
        : ''}
      // deep interiors get little sun: keep ambient LOW so point lights,
      // emissive veins and the headlight carry the mood — dark corners are a
      // feature, the sun + shadows sculpt the vistas
      vec3 amb = (uSkyCol * (0.30 + 0.25 * n.y) * 0.26 + uFogCol * 0.08) * (ao2 * 0.85 + 0.15);
      col = alb * (amb + uSunCol * sunDiff * sh * 1.15);
      col += uSunCol * spec * sh * sunDiff;
      // headlight: soft camera-attached fill so unlit caves stay readable
      float head = max(dot(n, -rd), 0.0);
      col += alb * (0.22 * head * head * ao / (1.0 + t * t * 0.20));

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
        float atten = min(uLightCol[li].w / (dist2 + rad * rad), 0.5) * exp(-sqrt(dist2) * 1.2e-4);
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

      // emissive veins by level (blended across the same two trap slots as
      // the albedo so vein color/strength never snaps either)
      ${global.__SHADER_DEBUG === 'noemis' ? '' : `
      float em = mix(emissiveOf(slotB), emissiveOf(slot), wA);
      if (em > 0.0) {
        float vein = smoothstep(0.07, 0.012, trapPl);
        col += mix(palette(slotB), palette(slot), wA) * em * vein * (0.6 + 0.4 * ao);
      }`}
      dbgLights = col - dbgBase;
    } else {
      col = skyColor(rd);
      t = tmax;
    }

    // distance fog toward mood color (keeps giant vistas readable). The air
    // is deliberately clear: most of the horizon work is done by the fade
    // below, so distant structure stays visible across most of the range.
    // All densities scale with uFogMul (continuity across rebases).
    float fog = 1.0 - exp(-fogT * 1.2e-5 * uFogMul);
    fog = mix(fog, 1.0 - exp(-fogT * 5.0e-6 * uFogMul), 0.5);
    // step-exhausted rays get softened toward fog ONLY at range: grazing
    // silhouettes at distance lose their speckle, while tight caves near the
    // camera stay DARK instead of being brightened toward the sky color
    if (exhausted) fog = max(fog, min(0.5, fogT / (tmax * 0.15)));
    // guarantee FULL fog before tmax, so the farthest vistas fade into the
    // haze instead of popping out of existence at the far plane
    fog = max(fog, smoothstep(0.55 * tmax, 0.92 * tmax, fogT));
    vec3 fogCol = mix(uFogCol, skyColor(rd), 0.35);
    col = mix(col, fogCol, hit ? fog : fog * 0.25);

    // light inscatter (glowing orbs + volume glow). NOTE: stored intensity
    // grows ~s^2 per rebase to keep surface lighting scale-invariant, but the
    // scatter integral only falls off ~1/distance, so it must be clamped or
    // ancient giant lights white out the frame.
    for (int li = 0; li < MAX_LIGHTS; li++) {
      if (li >= uLightN) break;
      float sc = inscatter(ro, rd, uLightPos[li].xyz, fogT);
      float ext = exp(-length(uLightPos[li].xyz) * 1.2e-4); // fog extinction of the glow
      // tight orb profile: the raw 1/h scatter is too broad and washes the
      // whole frame; sc^2/(1+sc/2) keeps a hot core with negligible skirt
      float orb = uLightCol[li].w * sc * sc * 0.013 / (1.0 + 0.5 * sc);
      col += uLightCol[li].rgb * min(orb, 0.9) * ext * uGlowAmt;
    }
    // sun glare kept subtle when occluded
    col += uSunCol * pow(max(dot(rd, uSunDir), 0.0), 8.0) * 0.03;

    ${global.__SHADER_DEBUG === 'terms2' ? 'fragColor = vec4(dot(dbgLights, vec3(0.33)), dot(col, vec3(0.33)) * 0.5, hit ? 0.2 : 0.8, 1.0); return;' : ''}
    col = aces(col * 1.15);
    col = pow(col, vec3(1.0 / 2.2));
    // vignette + dither
    float vig = 1.0 - 0.22 * dot(uv * 0.62, uv * 0.62);
    col *= vig;
    col += (hash12(gl_FragCoord.xy + fract(uTime) * 61.7) - 0.5) / 255.0;

    fragColor = vec4(col, 1.0);
  }`;

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

  const Shaders = { VERT, FRAG, BLIT_FRAG };
  global.Shaders = Shaders;
  if (typeof module !== 'undefined' && module.exports) module.exports = Shaders;
})(typeof window !== 'undefined' ? window : globalThis);
