/* main.js — application loop: camera, movement, rebasing, autopilot, HUD. */
(function () {
  'use strict';
  const V = window.V;
  const WM = window.WorldMod;

  /* ---- seed from URL ---- */
  const params = new URLSearchParams(location.search);
  const hashSeed = (location.hash.match(/seed=(\d+)/) || [])[1];
  const seed = (parseInt(params.get('seed') || hashSeed, 10) >>> 0) ||
    ((Math.random() * 0xffffffff) >>> 0);
  const DET = params.get('det') === '1'; // deterministic mode for tests

  const canvas = document.getElementById('view');
  const hud = {
    depth: document.getElementById('hud-depth'),
    zoom: document.getElementById('hud-zoom'),
    speed: document.getElementById('hud-speed'),
    fps: document.getElementById('hud-fps'),
    res: document.getElementById('hud-res'),
    seed: document.getElementById('hud-seed'),
    auto: document.getElementById('hud-auto'),
    msg: document.getElementById('hud-msg'),
  };
  const overlay = document.getElementById('overlay');
  const helpBox = document.getElementById('help');

  window.__SHADER_DEBUG = params.get('sdbg') || '';
  let renderer;
  try {
    renderer = new window.Renderer(canvas);
  } catch (err) {
    overlay.innerHTML = `<div class="title">Unable to start</div><div class="sub">${err.message}</div>`;
    return;
  }

  if (params.get('canon')) window.LevelGen.CANON = true;
  if (params.get('style') !== null && params.get('style') !== undefined && params.get('style') !== '') {
    window.LevelGen.OVERRIDE = {
      style: parseInt(params.get('style'), 10) || 0,
      style2: parseInt(params.get('style2') || '1', 10),
      alt: parseInt(params.get('alt') || '0', 10),
      scale: parseFloat(params.get('fscale')) || 0,
      spread: parseFloat(params.get('spread')) || 0,
      twist: parseFloat(params.get('twist')) || 0,
    };
  }
  const world = new WM.World(seed);
  // Start at a comfortable distance from the root structure: pull the camera
  // in (or push out) along its radius until the surface is ~0.4 units away.
  (function settleStart() {
    for (let i = 0; i < 60; i++) {
      const d = world.de(world.camPos);
      if (d > 0.25 && d < 0.7) break;
      const dir = V.norm(world.camPos);
      world.camPos = V.add(world.camPos, V.scale(dir, d > 0.7 ? -(d - 0.4) * 0.6 : 0.12));
      if (V.len(world.camPos) < 0.2) break;
    }
    world.rebaseToCamera();
    world.takeFrameRotation();
    world.updateChain();
  })();
  hud.seed.textContent = 'seed ' + seed;

  /* ---- camera state ---- */
  const cam = {
    right: [1, 0, 0],
    up: [0, 1, 0],
    forward: [0, 0, -1],
  };
  // start looking at the fractal root
  (function aimAtOrigin() {
    const f = V.norm(V.scale(world.camPos, -1));
    cam.forward = f;
    cam.right = V.norm(V.cross(f, [0, 1, 0]));
    cam.up = V.norm(V.cross(cam.right, f));
  })();

  function rotateBasis(axis, ang) {
    const R = V.mAxisAngle(axis, ang);
    cam.right = V.mMulV(R, cam.right);
    cam.up = V.mMulV(R, cam.up);
    cam.forward = V.mMulV(R, cam.forward);
  }
  function orthonormalize() {
    cam.forward = V.norm(cam.forward);
    cam.up = V.norm(V.sub(cam.up, V.scale(cam.forward, V.dot(cam.up, cam.forward))));
    cam.right = V.norm(V.sub(
      V.sub(cam.right, V.scale(cam.forward, V.dot(cam.right, cam.forward))),
      V.scale(cam.up, V.dot(cam.right, cam.up))));
  }

  const input = new window.Input(canvas);
  const state = {
    quality: 1,
    fov: 1.15,
    glow: 1.0,
    auto: false,
    paused: false,
    sunCol: [1.0, 0.93, 0.82],
    deCam: world.de(world.camPos),
  };

  let msgTimer = null;
  function flashMsg(text) {
    hud.msg.textContent = text;
    clearTimeout(msgTimer);
    msgTimer = setTimeout(() => { hud.msg.textContent = ''; }, 1600);
  }

  input.onToggle = (what, val) => {
    if (what === 'lock') {
      overlay.classList.toggle('hidden', val);
      if (!val) overlay.querySelector('.title').textContent = 'Paused — click to fly';
    }
    if (what === 'auto') { state.auto = !state.auto; hud.auto.classList.toggle('hidden', !state.auto); }
    if (what === 'help') helpBox.classList.toggle('hidden');
    if (what === 'pause') state.paused = !state.paused;
    if (what === 'glow') state.glow = state.glow > 0 ? 0 : 1;
    if (what === 'q0') state.quality = 0;
    if (what === 'q1') state.quality = 1;
    if (what === 'q2') state.quality = 2;
    if (what === 'light') { world.addUserLight(cam.forward); flashMsg('light dropped'); }
    if (what === 'newseed') {
      location.href = location.pathname + '?seed=' + ((Math.random() * 0xffffffff) >>> 0);
    }
  };

  /* ---- movement with collision (DE is a safe lower bound) ---- */
  const MIN_DE = 0.0035;
  // Manual flight speed in frame-K units/s at multiplier ×1. Constant on
  // purpose: the user's scroll wheel is the ONLY thing that changes speed,
  // and rebasing rescales the frame, which carries the same APPARENT speed
  // across zoom levels. (The old DE-proportional speed auto-slowed you into
  // surfaces and sped you up as you left — removed per playtest feedback.)
  const FLY_SPEED = 0.06;

  // Rebases and wedge corrections can apply REFLECTIONS (det = -1) to the
  // camera basis. The mirrored world renders identically (the reflection is an
  // exact symmetry — that's why rebasing is seamless), but the apparent sense
  // of every screen-space rotation inverts with the basis handedness. Track
  // the accumulated parity and flip the user's look/roll input by it, so
  // controls never reverse after a zoom-level snap.
  let camParity = 1;
  function tryMove(delta) {
    let remaining = V.len(delta);
    if (remaining <= 0) return;
    let dir = V.scale(delta, 1 / remaining);
    for (let iter = 0; iter < 4 && remaining > 1e-9; iter++) {
      const de = world.de(world.camPos);
      const step = Math.min(remaining, Math.max(de - MIN_DE * 0.5, 0) * 0.75);
      if (step <= 1e-9) break;
      world.camPos = V.add(world.camPos, V.scale(dir, step));
      remaining -= step;
    }
    // gentle push-out if we ended up too close
    const de = world.de(world.camPos);
    if (de < MIN_DE) {
      const g = world.grad(world.camPos);
      world.camPos = V.add(world.camPos, V.scale(g, (MIN_DE - de) * 0.9));
    }
  }

  /* ---- autopilot: probe candidates, steer toward interesting descent ---- */
  const auto = { dir: null, timer: 0 };
  let autoRs = seed ^ 0x5bd1e995;
  function autoRnd() { autoRs = (Math.imul(autoRs, 1103515245) + 12345) >>> 0; return autoRs / 4294967296; }
  function autopilot(dt) {
    auto.timer -= dt;
    const de = state.deCam;
    if (!auto.dir) auto.dir = cam.forward.slice();
    if (auto.timer <= 0) {
      auto.timer = 0.3;
      // candidates in a cone around current heading + a few wild ones
      let best = null, bestScore = -Infinity;
      for (let i = 0; i < 12; i++) {
        const wild = i >= 9;
        const spread = wild ? 1.0 : 0.35;
        let d = V.norm(V.add(auto.dir, [
          (autoRnd() * 2 - 1) * spread,
          (autoRnd() * 2 - 1) * spread,
          (autoRnd() * 2 - 1) * spread,
        ]));
        const probe = V.add(world.camPos, V.scale(d, de * 2.2));
        const pd = world.de(probe);
        if (pd < 0) continue;
        // prefer descending DE but not straight into a wall; keep momentum
        const score = -pd / de + V.dot(d, auto.dir) * 0.35 + (wild ? 0.05 : 0);
        if (score > bestScore) { bestScore = score; best = d; }
      }
      if (best) auto.dir = best;
    }
    // steer camera smoothly toward auto.dir
    const cur = cam.forward;
    const target = auto.dir;
    const axis = V.cross(cur, target);
    const axLen = V.len(axis);
    if (axLen > 1e-6) {
      const ang = Math.min(Math.asin(Math.min(axLen, 1)), 1.6 * dt);
      rotateBasis(V.scale(axis, 1 / axLen), ang);
    }
    tryMove(V.scale(cam.forward, de * 0.55 * dt * 60 * 0.016 * input.speedMult * 12));
  }

  /* ---- HUD ---- */
  let hudTimer = 0;
  function updateHUD(dt, fps) {
    hudTimer -= dt;
    if (hudTimer > 0) return;
    hudTimer = 0.2;
    hud.depth.textContent = 'level ' + world.K;
    const z = world.log10Zoom;
    hud.zoom.textContent = 'zoom 10^' + z.toFixed(1);
    hud.speed.textContent = 'speed ×' + input.speedMult.toFixed(2);
    hud.fps.textContent = fps.toFixed(0) + ' fps';
    hud.res.textContent = Math.round(renderer.resScale * 100) + '%' +
      (renderer.accumN > 1 ? ' • still ×' + renderer.accumN : '');
  }

  /* ---- main loop ---- */
  let last = performance.now();
  let fpsEma = 60;
  function frame(now) {
    requestAnimationFrame(frame);
    let dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (state.paused) return;
    if (DET) dt = 1 / 60;

    // look (parity-corrected so reflections never invert the controls)
    const [yd, pd] = input.consumeLook();
    const roll = input.rollInput();
    if (yd) rotateBasis(cam.up, yd * camParity);
    if (pd) rotateBasis(cam.right, pd * camParity);
    if (roll) rotateBasis(cam.forward, roll * 1.4 * dt * camParity);
    // only re-orthonormalize when the basis actually changed: a perfectly
    // still camera must produce bit-identical uniforms for frame accumulation
    if (yd || pd || roll) orthonormalize();

    // move
    state.deCam = world.de(world.camPos);
    let moving = false;
    if (state.auto) {
      autopilot(dt);
      moving = true;
    } else {
      const mv = input.moveVector();
      if (mv[0] || mv[1] || mv[2]) {
        moving = true;
        const speed = FLY_SPEED * input.speedMult;
        const delta = V.add(
          V.add(V.scale(cam.right, mv[0] * speed * dt), V.scale(cam.up, mv[1] * speed * dt)),
          V.scale(cam.forward, mv[2] * speed * dt));
        tryMove(delta);
      }
    }

    // rebase to keep camera-local numbers O(1); rotate the view basis along
    state.deCam = world.rebaseToCamera();
    const R = world.takeFrameRotation();
    const rotApplied =
      R[0] !== 1 || R[4] !== 1 || R[8] !== 1 ||
      R[1] !== 0 || R[2] !== 0 || R[3] !== 0 || R[5] !== 0 || R[6] !== 0 || R[7] !== 0;
    if (rotApplied) {
      const detR =
        R[0] * (R[4] * R[8] - R[5] * R[7]) -
        R[1] * (R[3] * R[8] - R[5] * R[6]) +
        R[2] * (R[3] * R[7] - R[4] * R[6]);
      if (detR < 0) camParity = -camParity;
      cam.right = V.norm(V.mMulV(R, cam.right));
      cam.up = V.norm(V.mMulV(R, cam.up));
      cam.forward = V.norm(V.mMulV(R, cam.forward));
      if (auto.dir) auto.dir = V.norm(V.mMulV(R, auto.dir));
      orthonormalize();
    }
    world.updateChain();

    // a still camera enables the renderer's accumulation path (anti-shimmer +
    // supersampled screenshots); any input, motion or rebase resets it
    const still = !moving && !yd && !pd && !roll && !rotApplied;

    const t0 = performance.now();
    renderer.render(world, cam, {
      time: now / 1000, fov: state.fov, quality: state.quality,
      sunCol: state.sunCol, glow: state.glow, deCam: state.deCam, still,
    });
    const frameMs = performance.now() - t0 + dt * 0; // render cost only
    if (!still) renderer.adaptResolution(Math.max(frameMs, dt * 1000 * 0.5));

    fpsEma = fpsEma * 0.95 + (1 / Math.max(dt, 1e-3)) * 0.05;
    updateHUD(dt, fpsEma);
  }
  if (!DET) requestAnimationFrame(frame);

  /* ---- deterministic hooks for automated tests/screenshots ---- */
  if (DET) {
    renderer.fixedScale = parseFloat(params.get('scale') || '0.5');
    renderer.resScale = renderer.fixedScale;
    window.__explorer = {
      world, cam, state, input, renderer,
      get parity() { return camParity; },
      step(n) { // advance autopilot n fixed frames without rAF timing
        for (let i = 0; i < n; i++) {
          state.deCam = world.de(world.camPos);
          autopilot(1 / 60);
          world.rebaseToCamera();
          const R = world.takeFrameRotation();
          const detR =
            R[0] * (R[4] * R[8] - R[5] * R[7]) -
            R[1] * (R[3] * R[8] - R[5] * R[6]) +
            R[2] * (R[3] * R[7] - R[4] * R[6]);
          if (detR < 0) camParity = -camParity;
          cam.right = V.norm(V.mMulV(R, cam.right));
          cam.up = V.norm(V.mMulV(R, cam.up));
          cam.forward = V.norm(V.mMulV(R, cam.forward));
          if (auto.dir) auto.dir = V.norm(V.mMulV(R, auto.dir));
          orthonormalize();
          world.updateChain();
        }
      },
      renderOnce() {
        renderer.render(world, cam, {
          time: 1.0, fov: state.fov, quality: state.quality,
          sunCol: state.sunCol, glow: state.glow, deCam: world.de(world.camPos),
        });
        renderer.gl.finish();
      },
      // GPU map() at a camera-relative point, bit-exact via RGBA8 probe pixel
      gpuMap(p) {
        renderer.render(world, cam, {
          time: 1.0, fov: state.fov, quality: state.quality,
          sunCol: state.sunCol, glow: state.glow, probe: p, deCam: world.de(world.camPos),
        });
        const gl = renderer.gl;
        const px = new Uint8Array(16);
        gl.readPixels(0, 0, 4, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        const dv = new DataView(new ArrayBuffer(4));
        for (let i = 0; i < 4; i++) dv.setUint8(i, px[i * 4]);
        return dv.getFloat32(0, true);
      },
    };
  }
})();
