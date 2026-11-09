/* renderer.js — WebGL2 setup, uniforms/UBO plumbing, dynamic resolution. */
(function (global) {
  'use strict';
  const V = global.V;
  const WM = global.WorldMod;
  const Shaders = global.Shaders;

  class Renderer {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext('webgl2', {
        antialias: false, depth: false, stencil: false, alpha: false,
        powerPreference: 'high-performance',
      });
      if (!gl) throw new Error('WebGL2 is required (try Chrome, Edge or Firefox).');
      this.gl = gl;

      const compile = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
          throw new Error('Shader compile error:\n' + gl.getShaderInfoLog(s));
        }
        return s;
      };
      const link = (fragSrc) => {
        const p = gl.createProgram();
        gl.attachShader(p, compile(gl.VERTEX_SHADER, Shaders.VERT));
        gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragSrc));
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
          throw new Error('Program link error:\n' + gl.getProgramInfoLog(p));
        }
        return p;
      };
      this._link = link;
      // Non-blocking compile (KHR_parallel_shader_compile): start now, query
      // completion later, so a big program never freezes the page.
      this._parallel = gl.getExtension('KHR_parallel_shader_compile');
      this._startLink = (fragSrc) => {
        const p = gl.createProgram();
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, Shaders.VERT); gl.compileShader(vs);
        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fragSrc); gl.compileShader(fs);
        gl.attachShader(p, vs); gl.attachShader(p, fs);
        gl.linkProgram(p);
        return { p, vs, fs };
      };
      // GPU timer (where the browser exposes it): REAL render cost for the
      // resolution controller. WebGL calls return immediately, so CPU-side
      // timing measured nothing.
      this._timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
      this._queries = [];
      this.gpuMs = null;
      this._buildMain();
      const prog = this.prog;

      // still-frame accumulation: blit/average program + ping-pong targets
      this.blitProg = link(Shaders.BLIT_FRAG);
      this.blitU = {
        uSrc: gl.getUniformLocation(this.blitProg, 'uSrc'),
        uPrev: gl.getUniformLocation(this.blitProg, 'uPrev'),
        uBlend: gl.getUniformLocation(this.blitProg, 'uBlend'),
      };
      gl.useProgram(this.blitProg);
      gl.uniform1i(this.blitU.uSrc, 0);
      gl.uniform1i(this.blitU.uPrev, 1);
      this.upProg = link(Shaders.UPSCALE_FRAG);
      this.upU = { uSrc: gl.getUniformLocation(this.upProg, 'uSrc'), uDst: gl.getUniformLocation(this.upProg, 'uDst') };
      gl.useProgram(this.upProg);
      gl.uniform1i(this.upU.uSrc, 0);
      this._live = null;        // offscreen target of the live (moving) frame
      this._pass = { y: 0, rows: 64, lastT: 0 }; // progressive still pass state
      gl.useProgram(prog);
      // float accumulation keeps averaging effective for hundreds of frames;
      // without the extension we fall back to RGBA8 and cap the frame count
      this.accumFloat = !!gl.getExtension('EXT_color_buffer_float');
      this.accumN = 0;          // frames accumulated so far (0 = direct path)
      this.stillFrames = 0;     // frames since the still image stopped resizing
      // live look / photo-mode settings (uniforms; see LOOK_DEFAULTS)
      this.look = Object.assign({}, Shaders.LOOK_DEFAULTS, global.__LOOK || {});
      this.lookVersion = 0;
      this._lastLookVersion = 0;
      this._targets = null;     // { w, h, scene, a, b }
      this._preStillScale = null;
      this._lastQuality = null;
      this._lastGlow = null;

      // uniform block
      this.uboData = new Float32Array((WM.SLOTS * WM.LVL_VEC4 + WM.B_HARD * WM.OUTER_VEC4) * 4);
      this.ubo = gl.createBuffer();
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo);
      gl.bufferData(gl.UNIFORM_BUFFER, this.uboData.byteLength, gl.DYNAMIC_DRAW);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.ubo);

      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      // dynamic resolution
      this.resScale = 0.5;   // start modest; strong GPUs grow quickly
      this.minScale = 0.15;
      this.maxScale = 1.0;
      this.targetMs = 15.5; // aim for a stable 60fps
      this.emaMs = 16;
      this.fixedScale = null; // set for deterministic screenshots
      this._capture = null;   // one-shot canvas capture callback (T key)

      this.lightPosBuf = new Float32Array(WM.MAX_LIGHTS_GPU * 4);
      this.lightColBuf = new Float32Array(WM.MAX_LIGHTS_GPU * 4);
    }

    /* (Re)build the main raymarching program — also used by look-dev tools
       after changing window.__LOOK, without reloading the page. */
    /* Program variants: 'fast' (compiled at startup) and 'precise' (the
       perturbative outer fold — see FRAG in shaders.js; only NEEDED on
       frames where world.needsPerturb). The precise one compiles in the
       background; until it is ready, those frames use 'fast' (identical
       except for tiny precision loss next to an outer fold crease). Halving
       the startup compile mattered on integrated GPUs, which appeared to
       hang. Also used by look-dev tools to rebuild after changes. */
    _buildMain() {
      const gl = this.gl;
      const prev = this.variants || {};
      for (const key in prev) if (prev[key] && prev[key].prog) gl.deleteProgram(prev[key].prog);
      this.variants = {};
      this.variants.fast = this._finishVariant(this._link(Shaders.FRAG(WM, { perturb: false })));
      this._pendingPrecise = null;
      this.prog = this.variants.fast.prog;
      this.U = this.variants.fast.U;
      this.accumN = 0;
    }
    _finishVariant(prog) {
      const gl = this.gl;
      gl.useProgram(prog);
      gl.uniformBlockBinding(prog, gl.getUniformBlockIndex(prog, 'Params'), 0);
      const U = {};
      for (const name of [
        'uRes', 'uTime', 'uCamBasis', 'uFovTan', 'uCamK', 'uW', 'uWScale', 'uB',
        'uSunDir', 'uSunCol', 'uSkyCol', 'uFogCol', 'uLightN', 'uLightPos',
        'uLightCol', 'uQuality', 'uGlowAmt', 'uProbeOn', 'uProbe',
        'uJitter', 'uFogMul', 'uHaze', 'uSkyBasis', 'uLookA', 'uLookB', 'uLookC', 'uLookD',
      ]) U[name] = gl.getUniformLocation(prog, name);
      return { prog, U };
    }
    // the precise variant if ready (blocking only when `force`), else null
    _precise(force) {
      const gl = this.gl;
      if (this.variants.precise) return this.variants.precise;
      if (!this._pendingPrecise) this._pendingPrecise = this._startLink(Shaders.FRAG(WM, { perturb: true }));
      const pend = this._pendingPrecise;
      const done = !this._parallel || force ||
        gl.getProgramParameter(pend.p, this._parallel.COMPLETION_STATUS_KHR);
      if (!done) return null;
      if (!gl.getProgramParameter(pend.p, gl.LINK_STATUS)) {
        console.warn('precise shader variant failed to link; staying on the fast one',
          gl.getShaderInfoLog(pend.fs), gl.getProgramInfoLog(pend.p));
        this.variants.precise = this.variants.fast;
      } else {
        this.variants.precise = this._finishVariant(pend.p);
      }
      gl.deleteShader(pend.vs); gl.deleteShader(pend.fs);
      this._pendingPrecise = null;
      return this.variants.precise;
    }

    /* Change look settings (photo mode). Restarts the still accumulation. */
    setLook(partial) {
      Object.assign(this.look, partial);
      this.lookVersion++;
    }
    resetLook() {
      this.look = Object.assign({}, Shaders.LOOK_DEFAULTS, global.__LOOK || {});
      this.lookVersion++;
    }

    /* ---- still-frame accumulation helpers ---- */
    _halton(i, base) {
      let f = 1, r = 0;
      while (i > 0) { f /= base; r += f * (i % base); i = Math.floor(i / base); }
      return r;
    }

    _makeTarget(w, h, internalFormat, linear) {
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const filt = linear ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, internalFormat, w, h);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { tex, fbo };
    }

    _dropTargets() {
      if (!this._targets) return;
      const gl = this.gl;
      for (const k of ['scene', 'a', 'b']) {
        gl.deleteTexture(this._targets[k].tex);
        gl.deleteFramebuffer(this._targets[k].fbo);
      }
      this._targets = null;
    }

    // (re)allocate accumulation targets at canvas size; returns true if reset
    _ensureTargets(w, h) {
      if (this._targets && this._targets.w === w && this._targets.h === h) return false;
      this._dropTargets();
      const gl = this.gl;
      const accFmt = this.accumFloat ? gl.RGBA16F : gl.RGBA8;
      this._targets = {
        w, h,
        scene: this._makeTarget(w, h, gl.RGBA8),
        a: this._makeTarget(w, h, accFmt),
        b: this._makeTarget(w, h, accFmt),
      };
      return true;
    }

    _ensureLive(w, h) {
      if (this._live && this._live.w === w && this._live.h === h) return;
      const gl = this.gl;
      if (this._live) { gl.deleteTexture(this._live.tex); gl.deleteFramebuffer(this._live.fbo); }
      this._live = Object.assign(this._makeTarget(w, h, gl.RGBA8, true), { w, h });
    }

    // fullscreen pass of the blit program: out = mix(prev, src, blend)
    _blit(src, prev, blend, fbo) {
      const gl = this.gl;
      gl.useProgram(this.blitProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, prev || src);
      gl.uniform1f(this.blitU.uBlend, blend);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    /* One-shot capture of the next rendered frame as a PNG blob (T key).
       Must read the canvas in the same task as the draw — the drawing buffer
       is not preserved across tasks. */
    captureNext(cb) {
      this._capture = cb;
    }
    _maybeCapture() {
      if (!this._capture) return;
      const cb = this._capture;
      this._capture = null;
      this.canvas.toBlob(cb, 'image/png');
    }

    /* Dynamic resolution. Measures the real cost: GPU time from timer
       queries where available, otherwise the frame interval (dtMs). Over
       budget, the scale shrinks in proportion (pixel cost ~ scale^2, so
       sqrt(target/measured)) — a slow GPU reaches a workable size within a
       few frames instead of crawling down 1.5% per frame (which looked like
       a hang). Under budget it grows slowly, and never straight back to a
       scale that just overloaded (ceiling with slow decay), so it doesn't
       oscillate. `overloadSec` tells main.js when even the minimum scale is
       too slow, so it can lower quality. */
    adaptResolution(dtMs) {
      if (this.fixedScale) { this.resScale = this.fixedScale; return; }
      const gpu = this.gpuMs;
      const useGpu = gpu !== null;
      const measured = useGpu ? gpu : dtMs;
      const target = useGpu ? 12.5 : 17.5; // GPU budget leaves room for the rest of the frame
      this.emaMs = this.emaMs * 0.8 + measured * 0.2;
      this.ceiling = Math.min(this.maxScale, (this.ceiling || this.maxScale) * 1.0015);
      if (this.emaMs > target * 1.12) {
        const k = Math.min(0.97, Math.max(0.7, Math.sqrt(target / this.emaMs)));
        this.ceiling = Math.max(this.minScale, this.resScale * 0.97);
        this.resScale = Math.max(this.minScale, this.resScale * k);
        this.emaMs = target; // let the new size be measured before reacting again
      } else if (this.emaMs < target * (useGpu ? 0.8 : 1.0)) {
        const up = useGpu ? Math.min(1.02, Math.sqrt(target / Math.max(this.emaMs, 1))) : 1.004;
        this.resScale = Math.min(this.ceiling, this.resScale * up);
      }
      if (this.resScale <= this.minScale * 1.001 && measured > target * 1.6) {
        this.overloadSec = (this.overloadSec || 0) + dtMs / 1000;
      } else {
        this.overloadSec = 0;
      }
    }

    // GPU timer bookkeeping around one frame's draws
    _timerBegin() {
      const gl = this.gl, ext = this._timer;
      if (!ext || this._queries.length >= 3) return null;
      const q = gl.createQuery();
      gl.beginQuery(ext.TIME_ELAPSED_EXT, q);
      return q;
    }
    _timerEnd(q) {
      const gl = this.gl, ext = this._timer;
      if (!q) return;
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      this._queries.push(q);
    }
    _timerPoll() {
      const gl = this.gl, ext = this._timer;
      if (!ext) return;
      const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
      while (this._queries.length) {
        const q = this._queries[0];
        if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
        const ns = gl.getQueryParameter(q, gl.QUERY_RESULT);
        if (!disjoint) this.gpuMs = ns / 1e6;
        gl.deleteQuery(q);
        this._queries.shift();
      }
    }

    resize() {
      const dpr = this.fixedScale ? 1 : Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(64, Math.round(this.canvas.clientWidth * dpr * this.resScale));
      const h = Math.max(64, Math.round(this.canvas.clientHeight * dpr * this.resScale));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
    }

    render(world, cam, opts) {
      const gl = this.gl;
      // pick the program variant (and use it: the accumulation blit may have
      // switched programs)
      this._timerPoll();
      const v = (world.needsPerturb || opts.probe) ? (this._precise(!!opts.probe) || this.variants.fast)
        : this.variants.fast;
      // warm the precise variant up in the background once we're running
      if (!this.variants.precise && this._parallel && !this._pendingPrecise) this._precise(false);
      this.prog = v.prog;
      this.U = v.U;
      gl.useProgram(this.prog);

      /* Still-frame accumulation, PROGRESSIVE: while the camera is still,
         the canvas jumps to full resolution and each frame renders only a
         BAND of rows of the next jittered sample (band height adapts to a
         ~20 ms budget), blending it into the running average. Samples
         complete band by band; rows not yet reached show the previous
         result (initially the last live frame, upscaled). Every frame stays
         cheap on any GPU, so stopping never stalls input — rendering a whole
         full-resolution, high-detail frame at once used to take seconds on
         integrated GPUs (the page looked hung) and caused lag spikes on fast
         ones. Shimmer/aliasing converge to a clean supersampled image, and
         the refinement stages below sharpen it further. */
      const accumulating = !!opts.still && !opts.probe;
      const wasStill = this._wasStill;
      this._wasStill = accumulating;
      if (wasStill !== accumulating) this.gpuMs = null; // band timings != live frame timings
      if (!accumulating) {
        if (this._preStillScale !== null) {
          this.resScale = this._preStillScale;
          this._preStillScale = null;
        }
        this.accumN = 0;
      } else {
        if (this._preStillScale === null) this._preStillScale = this.resScale;
        if (!this.fixedScale) this.resScale = this.maxScale;
        if (!wasStill) this._startPass = true;
      }
      if (opts.quality !== this._lastQuality || opts.glow !== this._lastGlow ||
          this.lookVersion !== this._lastLookVersion) {
        this._lastQuality = opts.quality;
        this._lastGlow = opts.glow;
        this._lastLookVersion = this.lookVersion;
        this.accumN = 0;
        this._pass.y = 0;
      }
      /* Progressive refinement while still: after the first complete samples
         the held image switches to finer LOD / hit footprint (detail), a
         bigger step budget and high-quality shadows, twice. Each switch keeps
         the accumulated image but restarts its weight at 1/3, so new detail
         fades in: blurry/warped far structure resolves into real detail.
         (Stages advance at sample boundaries, so a sample is never mixed.) */
      let detail = 1, stepMul = 1, quality = opts.quality;
      if (accumulating) {
        if (this._pass.y === 0) {
          const stage = this.forceStage !== undefined ? this.forceStage
            : this.accumN < 2 ? 0 : this.accumN < 6 ? 1 : 2;
          if (stage !== this._stage && stage > 0) this.accumN = Math.min(this.accumN, 2);
          this._stage = stage;
        }
        const stage = this._stage || 0;
        detail = [1, 0.5, 0.25][stage];
        stepMul = [1, 1.6, 2.4][stage];
        if (stage > 0) quality = Math.max(quality, 2);
        this.stillFrames++;
      } else {
        this._stage = 0;
        this.stillFrames = 0;
      }

      this.resize();
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);

      world.packUBO(this.uboData);
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo);
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this.uboData);

      const U = this.U;
      gl.uniform2f(U.uRes, this.canvas.width, this.canvas.height);
      gl.uniform1f(U.uTime, opts.time);
      // columns: right, up, forward
      gl.uniformMatrix3fv(U.uCamBasis, false, [
        cam.right[0], cam.right[1], cam.right[2],
        cam.up[0], cam.up[1], cam.up[2],
        cam.forward[0], cam.forward[1], cam.forward[2],
      ]);
      gl.uniform1f(U.uFovTan, Math.tan(opts.fov * 0.5));
      gl.uniform3f(U.uCamK, world.camPos[0], world.camPos[1], world.camPos[2]);
      gl.uniformMatrix3fv(U.uW, false, V.mToCols(world.W));
      gl.uniform1f(U.uWScale, world.WScale);
      gl.uniform1i(U.uB, (typeof window !== 'undefined' && window.__SHADER_DEBUG === 'noouter') ? 0 : world.outer.length);
      // mood: sky/haze/sun, all continuous in depth; directions world-anchored
      const mood = world.mood(opts.deCam);
      const Lk = this.look;
      let sunDir = mood.sunDir, sunCol = mood.sun;
      if (Lk.customSun) {
        // photo mode: sun placed in WORLD directions (az/el), own color
        const el = Lk.sunEl * Math.PI / 180, az = Lk.sunAz * Math.PI / 180;
        sunDir = V.norm(V.mMulV(world.skyBasis,
          [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)]));
        sunCol = global.LevelGen.hsv(Lk.sunHue, Lk.sunTint, 1).map((v) => v * 1.5);
      }
      const hueRot = (c, h) => { // rotation about the grey axis, h in turns
        if (!h) return c;
        const a = h * 2 * Math.PI, ca = Math.cos(a), sa = Math.sin(a), k = 0.57735027;
        const kd = k * (c[0] + c[1] + c[2]) * (1 - ca);
        const cr = [k * (c[2] - c[1]), k * (c[0] - c[2]), k * (c[1] - c[0])]; // k x c
        return c.map((v, i) => Math.max(0, v * ca + cr[i] * sa + k * kd));
      };
      gl.uniform3fv(U.uSunDir, sunDir);
      gl.uniform3fv(U.uSunCol, sunCol);
      gl.uniformMatrix3fv(U.uSkyBasis, false, V.mToCols(world.skyBasis));
      gl.uniform3fv(U.uSkyCol, hueRot(mood.sky, Lk.hazeHue));
      gl.uniform3fv(U.uFogCol, hueRot(mood.fog, Lk.hazeHue));
      gl.uniform1f(U.uHaze, mood.haze * Lk.hazeMul);
      gl.uniform4f(U.uLookA, Lk.exposure, Lk.sun, Lk.sunSat, Lk.fill);
      gl.uniform4f(U.uLookB, Lk.ground, Lk.bounce, Lk.head, Lk.rim);
      gl.uniform4f(U.uLookC, Lk.sat, Lk.huePreserve, Lk.albHue, Lk.glow);
      gl.uniform4f(U.uLookD, detail, Lk.vignette, stepMul, 0);

      const lights = world.gpuLights(opts.deCam);
      for (let i = 0; i < WM.MAX_LIGHTS_GPU; i++) {
        const L = lights[i];
        const o = i * 4;
        if (L) {
          // camera-relative positions (shader camera sits at the origin)
          this.lightPosBuf[o] = L.pos[0] - world.camPos[0];
          this.lightPosBuf[o + 1] = L.pos[1] - world.camPos[1];
          this.lightPosBuf[o + 2] = L.pos[2] - world.camPos[2];
          this.lightPosBuf[o + 3] = L.radius;
          this.lightColBuf[o] = L.col[0];
          this.lightColBuf[o + 1] = L.col[1];
          this.lightColBuf[o + 2] = L.col[2];
          this.lightColBuf[o + 3] = L.intensity;
        } else {
          this.lightPosBuf.fill(0, o, o + 4);
          this.lightColBuf.fill(0, o, o + 4);
        }
      }
      gl.uniform4fv(U.uLightPos, this.lightPosBuf);
      gl.uniform4fv(U.uLightCol, this.lightColBuf);
      gl.uniform1i(U.uLightN, (typeof window !== 'undefined' && window.__SHADER_DEBUG === 'nolights') ? 0 : lights.length);
      gl.uniform1i(U.uQuality, quality);
      gl.uniform1f(U.uGlowAmt, opts.glow);
      gl.uniform1f(U.uFogMul, world.fogMul || 1);
      gl.uniform1f(U.uProbeOn, opts.probe ? 1 : 0);
      if (opts.probe) gl.uniform3fv(U.uProbe, opts.probe);

      const W = this.canvas.width, H = this.canvas.height;
      if (!accumulating) {
        // live frame: render offscreen (kept for the still transition), copy
        gl.uniform2f(U.uJitter, 0, 0);
        if (opts.probe) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
          return;
        }
        this._ensureLive(W, H);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this._live.fbo);
        const q = this._timerBegin();
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        this._timerEnd(q);
        this._blit(this._live.tex, null, 1, null);
        this._maybeCapture();
        return;
      }

      /* progressive accumulation path */
      const P = this._pass;
      if (this._ensureTargets(W, H)) this._startPass = true;
      const T = this._targets;
      if (this._startPass) {
        // seed the accumulator with the last live frame, upscaled
        this._startPass = false;
        this.accumN = 0;
        P.y = 0;
        if (this._live) {
          gl.useProgram(this.upProg);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, this._live.tex);
          gl.uniform2f(this.upU.uDst, W, H);
          gl.bindFramebuffer(gl.FRAMEBUFFER, T.a.fbo);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
        }
        gl.useProgram(this.prog);
      }
      // band height from the measured cost of the previous band (GPU timer,
      // else the frame interval), aiming at ~20 ms per frame
      const now = performance.now();
      const measured = this.gpuMs !== null ? this.gpuMs : (P.lastT ? now - P.lastT : 0);
      if (measured > 0 && P.lastRows) {
        P.rows = Math.round(Math.min(H, Math.max(8, P.lastRows * Math.min(2, Math.max(0.5, 20 / measured)))));
      }
      P.lastT = now;
      const cap = this.accumFloat ? 1024 : 64; // 8-bit blending stalls past ~1/64
      const n = Math.min(this.accumN, cap - 1);
      // sample 0 unjittered so entering still mode never visibly shifts the image
      const jx = n > 0 ? this._halton((n % 256) + 1, 2) - 0.5 : 0;
      const jy = n > 0 ? this._halton((n % 256) + 1, 3) - 0.5 : 0;
      gl.uniform2f(U.uJitter, jx, jy);
      const y0 = P.y, y1 = Math.min(H, y0 + P.rows);
      P.lastRows = y1 - y0;
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(0, y0, W, y1 - y0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, T.scene.fbo);
      const q = this._timerBegin();
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this._timerEnd(q);
      // average the band into the running accumulation (a -> b)
      this._blit(T.scene.tex, T.a.tex, 1 / (n + 1), T.b.fbo);
      // display: finished rows of this sample from b, the rest from a
      gl.scissor(0, 0, W, y1);
      this._blit(T.b.tex, null, 1, null);
      if (y1 < H) {
        gl.scissor(0, y1, W, H - y1);
        this._blit(T.a.tex, null, 1, null);
      }
      gl.disable(gl.SCISSOR_TEST);
      P.y = y1;
      if (y1 >= H) { // sample complete
        const swap = T.a; T.a = T.b; T.b = swap;
        this.accumN++;
        P.y = 0;
      }
      this._maybeCapture();
    }
  }

  global.Renderer = Renderer;
})(typeof window !== 'undefined' ? window : globalThis);
