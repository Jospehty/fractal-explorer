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
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, Shaders.VERT));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, Shaders.FRAG(WM)));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error('Program link error:\n' + gl.getProgramInfoLog(prog));
      }
      gl.useProgram(prog);
      this.prog = prog;

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
      gl.uniform1f(U.uProbeOn, opts.probe ? 1 : 0);
      if (opts.probe) gl.uniform3fv(U.uProbe, opts.probe);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  global.Renderer = Renderer;
})(typeof window !== 'undefined' ? window : globalThis);
