/* vec.js — minimal float64 vec3 / mat3 helpers (row-major mat3 as 9-element arrays).
   Used by the CPU-side world model. Works in browser (classic script) and node. */
(function (global) {
  'use strict';

  const V = {
    v3: (x = 0, y = 0, z = 0) => [x, y, z],
    clone: (a) => [a[0], a[1], a[2]],
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ],
    len: (a) => Math.hypot(a[0], a[1], a[2]),
    dist: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
    norm: (a) => {
      const l = Math.hypot(a[0], a[1], a[2]) || 1;
      return [a[0] / l, a[1] / l, a[2] / l];
    },
    lerp: (a, b, t) => [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ],

    // ---- mat3 (row-major: m[r*3+c]) ----
    mIdent: () => [1, 0, 0, 0, 1, 0, 0, 0, 1],
    mMulV: (m, v) => [
      m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
      m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
      m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ],
    mMul: (a, b) => {
      const r = new Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          r[i * 3 + j] =
            a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
      return r;
    },
    mScale: (m, s) => m.map((x) => x * s),
    mTranspose: (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]],
    // General 3x3 inverse (adjugate). Our matrices are (scale x orthogonal) so this
    // is well conditioned, but a general inverse keeps it exact for any op we add.
    mInv: (m) => {
      const a = m[0], b = m[1], c = m[2],
            d = m[3], e = m[4], f = m[5],
            g = m[6], h = m[7], i = m[8];
      const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
      const det = a * A + b * B + c * C;
      const id = 1 / det;
      return [
        A * id, (c * h - b * i) * id, (b * f - c * e) * id,
        B * id, (a * i - c * g) * id, (c * d - a * f) * id,
        C * id, (b * g - a * h) * id, (a * e - b * d) * id,
      ];
    },
    // Rotation about unit axis by angle (Rodrigues), row-major.
    mAxisAngle: (axis, ang) => {
      const [x, y, z] = axis;
      const c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
      return [
        t * x * x + c, t * x * y - s * z, t * x * z + s * y,
        t * x * y + s * z, t * y * y + c, t * y * z - s * x,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c,
      ];
    },
    // Column-major Float32 upload order for GLSL mat3 (3 columns).
    mToCols: (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]],
  };

  global.V = V;
  if (typeof module !== 'undefined' && module.exports) module.exports = V;
})(typeof window !== 'undefined' ? window : globalThis);
