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
      const prog = link(Shaders.FRAG(WM));
      gl.useProgram(prog);
      this.prog = prog;

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
      gl.useProgram(prog);
      // float accumulation keeps averaging effective for hundreds of frames;
      // without the extension we fall back to RGBA8 and cap the frame count
      this.accumFloat = !!gl.getExtension('EXT_color_buffer_float');
      this.accumN = 0;          // frames accumulated so far (0 = direct path)
      this._targets = null;     // { w, h, scene, a, b }
      this._preStillScale = null;
      this._lastQuality = null;
      this._lastGlow = null;

      // uniform block
      this.uboData = new Float32Array((WM.SLOTS * 6 + WM.B_HARD * 4) * 4);
      this.ubo = gl.createBuffer();
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo);
      gl.bufferData(gl.UNIFORM_BUFFER, this.uboData.byteLength, gl.DYNAMIC_DRAW);
      const blockIdx = gl.getUniformBlockIndex(prog, 'Params');
      gl.uniformBlockBinding(prog, blockIdx, 0);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.ubo);

      this.U = {};
      for (const name of [
        'uRes', 'uTime', 'uCamBasis', 'uFovTan', 'uCamK', 'uW', 'uWScale', 'uB',
        'uSunDir', 'uSunCol', 'uSkyCol', 'uFogCol', 'uLightN', 'uLightPos',
        'uLightCol', 'uQuality', 'uGlowAmt', 'uBound', 'uProbeOn', 'uProbe',
        'uJitter', 'uFogMul',
      ]) this.U[name] = gl.getUniformLocation(prog, name);

      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      // dynamic resolution
      this.resScale = 0.75;
      this.minScale = 0.35;
      this.maxScale = 1.0;
      this.targetMs = 18;
      this.emaMs = 16;
      this.fixedScale = null; // set for deterministic screenshots

      this.lightPosBuf = new Float32Array(WM.MAX_LIGHTS_GPU * 4);
      this.lightColBuf = new Float32Array(WM.MAX_LIGHTS_GPU * 4);
    }

    /* ---- still-frame accumulation helpers ---- */
    _halton(i, base) {
      let f = 1, r = 0;
      while (i > 0) { f /= base; r += f * (i % base); i = Math.floor(i / base); }
      return r;
    }

    _makeTarget(w, h, internalFormat) {
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
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

    adaptResolution(frameMs) {
      if (this.fixedScale) { this.resScale = this.fixedScale; return; }
      this.emaMs = this.emaMs * 0.92 + frameMs * 0.08;
      if (this.emaMs > this.targetMs * 1.25) {
        this.resScale = Math.max(this.minScale, this.resScale * 0.96);
      } else if (this.emaMs < this.targetMs * 0.7) {
        this.resScale = Math.min(this.maxScale, this.resScale * 1.02);
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
      gl.useProgram(this.prog); // the accumulation blit may have switched programs

      /* Still-frame accumulation: while the camera is perfectly still we ramp
         the resolution to 100%, render with sub-pixel Halton jitter into an
         offscreen target and average the frames — shimmer/aliasing converge to
         a clean supersampled image (ideal for screenshots). Any movement drops
         straight back to the direct path and restores the adaptive scale. */
      const accumulating = !!opts.still && !opts.probe;
      if (!accumulating) {
        if (this._preStillScale !== null) {
          this.resScale = this._preStillScale;
          this._preStillScale = null;
        }
        this.accumN = 0;
      } else {
        if (this._preStillScale === null) this._preStillScale = this.resScale;
        if (!this.fixedScale && this.resScale < this.maxScale) {
          this.resScale = Math.min(this.maxScale, this.resScale * 1.15);
          this.accumN = 0; // still ramping resolution; sizes change per frame
        }
      }
      if (opts.quality !== this._lastQuality || opts.glow !== this._lastGlow) {
        this._lastQuality = opts.quality;
        this._lastGlow = opts.glow;
        this.accumN = 0;
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
      gl.uniform3fv(U.uSunDir, world.sunDir);
      gl.uniform3fv(U.uSunCol, opts.sunCol);
      const mood = world.mood(opts.deCam);
      gl.uniform3fv(U.uSkyCol, mood.sky);
      gl.uniform3fv(U.uFogCol, mood.fog);

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
      gl.uniform1i(U.uQuality, opts.quality);
      gl.uniform1f(U.uGlowAmt, opts.glow);
      gl.uniform1f(U.uBound, world.bound || 2.5);
      gl.uniform1f(U.uFogMul, world.fogMul || 1);
      gl.uniform1f(U.uProbeOn, opts.probe ? 1 : 0);
      if (opts.probe) gl.uniform3fv(U.uProbe, opts.probe);

      if (!accumulating) {
        gl.uniform2f(U.uJitter, 0, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        return;
      }

      /* accumulation path */
      if (this._ensureTargets(this.canvas.width, this.canvas.height)) this.accumN = 0;
      const T = this._targets;
      const cap = this.accumFloat ? 1024 : 64; // 8-bit blending stalls past ~1/64
      const n = Math.min(this.accumN, cap - 1);
      // frame 0 unjittered so entering still mode never visibly shifts the image
      const jx = n > 0 ? this._halton((n % 256) + 1, 2) - 0.5 : 0;
      const jy = n > 0 ? this._halton((n % 256) + 1, 3) - 0.5 : 0;
      gl.uniform2f(U.uJitter, jx, jy);
      gl.bindFramebuffer(gl.FRAMEBUFFER, T.scene.fbo);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      // average the new frame into the running accumulation (ping-pong a→b)
      gl.useProgram(this.blitProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, T.scene.tex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, T.a.tex);
      gl.uniform1f(this.blitU.uBlend, 1 / (n + 1));
      gl.bindFramebuffer(gl.FRAMEBUFFER, T.b.fbo);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      // display the accumulation (uBlend=1 → plain copy of unit 0)
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, T.b.tex);
      gl.uniform1f(this.blitU.uBlend, 1);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      const swap = T.a; T.a = T.b; T.b = swap;
      this.accumN++;
    }
  }

  global.Renderer = Renderer;
})(typeof window !== 'undefined' ? window : globalThis);
