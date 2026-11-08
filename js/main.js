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
  // look-dev overrides for the lighting/grade constants (see LOOK_DEFAULTS
  // in shaders.js), e.g. ?look={"fill":0.5,"exposure":1.1}
  try { if (params.get('look')) window.__LOOK = JSON.parse(params.get('look')); } catch (e) { /* ignore */ }
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
      foldoff: parseFloat(params.get('foldoff')) || 0,
      n: parseInt(params.get('n') || '5', 10),
      h: params.get('h') !== null ? parseFloat(params.get('h')) : 0.5,
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
    deCam: world.de(world.camPos),
    lightColIdx: 0,
    photo: false,
  };

  // manual light palette (C cycles, L drops); 'auto' = current level's color
  const LIGHT_COLORS = [
    ['auto', null],
    ['warm', [1.0, 0.78, 0.5]],
    ['cold', [0.55, 0.75, 1.0]],
    ['ember', [1.0, 0.42, 0.25]],
    ['emerald', [0.4, 1.0, 0.55]],
    ['violet', [0.75, 0.5, 1.0]],
    ['white', [1, 1, 1]],
  ];

  let msgTimer = null;
  function flashMsg(text) {
    hud.msg.textContent = text;
    clearTimeout(msgTimer);
    msgTimer = setTimeout(() => { hud.msg.textContent = ''; }, 1600);
  }

  input.onToggle = (what, val) => {
    if (what === 'lock') {
      if (state.photo) return; // photo mode keeps the mouse free, no overlay
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
    if (what === 'light') {
      const c = LIGHT_COLORS[state.lightColIdx];
      world.addUserLight(cam.forward, c[1]);
      flashMsg('light dropped (' + c[0] + ')');
    }
    if (what === 'lightcolor') {
      state.lightColIdx = (state.lightColIdx + 1) % LIGHT_COLORS.length;
      flashMsg('light color: ' + LIGHT_COLORS[state.lightColIdx][0]);
    }
    if (what === 'autolights') {
      world.autoLights = !world.autoLights;
      flashMsg('auto lights ' + (world.autoLights ? 'on' : 'off'));
    }
    if (what === 'removelight') {
      flashMsg(world.removeNearestLight() ? 'light removed' : 'no lights to remove');
    }
    if (what === 'shot') {
      renderer.captureNext((blob) => {
        if (!blob) { flashMsg('screenshot failed'); return; }
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'fractal_' + seed + '_K' + world.K +
          '_z' + world.log10Zoom.toFixed(1) + '.png';
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      });
      flashMsg('screenshot saved');
    }
    if (what === 'photo') togglePhoto();
    if (what === 'newseed') {
      location.href = location.pathname + '?seed=' + ((Math.random() * 0xffffffff) >>> 0);
    }
  };

  /* ---- photo mode (V) ----
     Freezes the camera (so the still-frame accumulator refines the image)
     and releases the mouse for a panel of LIVE look controls: they are
     shader uniforms (renderer.setLook), so changes apply instantly and the
     accumulation restarts from the new settings. */
  const photoBox = document.getElementById('photo');
  const PHOTO_SPEC = [
    ['Light'],
    ['exposure', 'Exposure', 0.3, 3, 0.01],
    ['sun', 'Sun', 0, 3, 0.01],
    ['fill', 'Sky fill', 0, 3, 0.01],
    ['head', 'Headlight', 0, 1, 0.01],
    ['rim', 'Haze rim', 0, 4, 0.01],
    ['Sun'],
    ['customSun', 'Place the sun myself'],
    ['sunAz', 'Azimuth', 0, 360, 1],
    ['sunEl', 'Elevation', -15, 89, 1],
    ['sunHue', 'Sun hue', 0, 1, 0.005],
    ['sunTint', 'Sun tint', 0, 1, 0.01],
    ['Color'],
    ['albHue', 'Surface hue', -0.5, 0.5, 0.005],
    ['sat', 'Saturation', 0, 2, 0.01],
    ['hazeHue', 'Haze hue', -0.5, 0.5, 0.005],
    ['hazeMul', 'Haze density', 0, 3, 0.01],
    ['glow', 'Vein glow', 0, 4, 0.01],
    ['Lens'],
    ['fov', 'Field of view', 30, 120, 1],
    ['vignette', 'Vignette', 0, 0.7, 0.01],
  ];
  const photoInputs = {};
  function photoValue(key) {
    return key === 'fov' ? state.fov * 180 / Math.PI : renderer.look[key];
  }
  function buildPhoto() {
    const parts = ['<div class="ph-title">Photo mode</div>',
      '<div class="ph-note">Camera frozen. Changes apply live and the frame keeps refining. V to fly again.</div>'];
    for (const row of PHOTO_SPEC) {
      if (row.length === 1) { parts.push(`<div class="ph-group">${row[0]}</div>`); continue; }
      const [key, label, min, max, step] = row;
      if (min === undefined) {
        parts.push(`<label class="ph-check"><input type="checkbox" id="ph-${key}"> ${label}</label>`);
      } else {
        parts.push(`<div class="ph-row"><label for="ph-${key}">${label}</label>` +
          `<input type="range" id="ph-${key}" min="${min}" max="${max}" step="${step}">` +
          `<span id="ph-${key}-v"></span></div>`);
      }
    }
    parts.push('<div class="ph-buttons"><button id="ph-reset">Reset look</button>' +
      '<button id="ph-shot">Save PNG</button><button id="ph-exit">Fly again</button></div>');
    photoBox.innerHTML = parts.join('');
    for (const row of PHOTO_SPEC) {
      if (row.length === 1) continue;
      const key = row[0];
      const el = document.getElementById('ph-' + key);
      photoInputs[key] = el;
      el.addEventListener('input', () => {
        if (el.type === 'checkbox') {
          const patch = { customSun: el.checked };
          if (el.checked) Object.assign(patch, currentSunAngles());
          renderer.setLook(patch);
        } else if (key === 'fov') {
          state.fov = parseFloat(el.value) * Math.PI / 180;
          renderer.setLook({}); // restart accumulation
        } else {
          const v = parseFloat(el.value);
          // moving a sun slider implies placing the sun yourself
          const patch = { [key]: v };
          if (/^sun(Az|El|Hue|Tint)$/.test(key) && !renderer.look.customSun) {
            Object.assign(patch, currentSunAngles(), { [key]: v, customSun: true });
          }
          renderer.setLook(patch);
        }
        syncPhoto();
      });
    }
    document.getElementById('ph-reset').addEventListener('click', () => {
      renderer.resetLook(); state.fov = 1.15; syncPhoto();
    });
    document.getElementById('ph-shot').addEventListener('click', () => input.onToggle('shot'));
    document.getElementById('ph-exit').addEventListener('click', () => togglePhoto());
  }
  // the world's current sun, as world-frame azimuth/elevation (degrees)
  function currentSunAngles() {
    const m = world.mood(state.deCam);
    const w = V.mMulV(V.mTranspose(world.skyBasis), m.sunDir);
    return {
      sunAz: ((Math.atan2(w[2], w[0]) * 180 / Math.PI) + 360) % 360,
      sunEl: Math.asin(Math.max(-1, Math.min(1, w[1]))) * 180 / Math.PI,
    };
  }
  function syncPhoto() {
    for (const key in photoInputs) {
      const el = photoInputs[key];
      const v = photoValue(key);
      if (el.type === 'checkbox') { el.checked = !!v; continue; }
      el.value = v;
      const out = document.getElementById('ph-' + key + '-v');
      if (out) out.textContent = Math.abs(v) >= 10 ? Math.round(v) : Number(v).toFixed(2);
    }
  }
  function togglePhoto() {
    state.photo = !state.photo;
    if (state.photo) {
      if (!photoBox.firstChild) buildPhoto();
      syncPhoto();
      if (document.pointerLockElement) document.exitPointerLock();
    }
    input.suspended = state.photo;
    photoBox.classList.toggle('hidden', !state.photo);
    overlay.classList.add('hidden');
    flashMsg(state.photo ? 'photo mode' : 'flying');
  }

  /* ---- movement with collision (DE is a safe lower bound) ---- */
  const MIN_DE = 0.0035;
  // Manual flight speed in frame-K units/s at multiplier ×1 (divided by
  // world.fogMul at use). The scroll wheel is the ONLY thing that changes
  // speed. Dividing by fogMul makes the APPARENT speed continuous across
  // rebases: frame units rescale by s at every rebase, and fogMul carries
  // the exactly-compensating sawtooth (see world.js), so flying in or out
  // never steps your speed — the third incarnation of this code; the first
  // auto-scaled with surface distance, the second stepped at each rebase.
  const FLY_SPEED = 0.09;

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
      // moveCamera sub-steps across fold creases and returns the direction
      // re-charted by any mirror correction on the way (keep flying straight)
      dir = world.moveCamera(V.scale(dir, step)) || dir;
      remaining -= step;
    }
    // gentle push-out if we ended up too close
    const de = world.de(world.camPos);
    if (de < MIN_DE) {
      const g = world.grad(world.camPos);
      world.moveCamera(V.scale(g, (MIN_DE - de) * 0.9));
    }
  }

  /* ---- autopilot: probe candidates, steer toward interesting descent ---- */
  /* Stall detection: greedy DE descent can circle forever in a basin of the
     distance estimator (a local minimum of the lower bound, common in dusty
     offset-fold regions). If continuous depth stops increasing, probe wide
     and far for a while to find a way down. */
  const auto = { dir: null, timer: 0, bestDepth: -Infinity, sinceProgress: 0, escape: 0 };
  let autoRs = seed ^ 0x5bd1e995;
  function autoRnd() { autoRs = (Math.imul(autoRs, 1103515245) + 12345) >>> 0; return autoRs / 4294967296; }
  function autopilot(dt) {
    auto.timer -= dt;
    const de = state.deCam;
    if (!auto.dir) auto.dir = cam.forward.slice();
    const depth = world.depthZoom(de);
    if (depth > auto.bestDepth + 0.05) { auto.bestDepth = depth; auto.sinceProgress = 0; }
    else auto.sinceProgress += dt;
    if (auto.sinceProgress > 5 && auto.escape <= 0) { auto.escape = 2.5; auto.sinceProgress = 0; }
    auto.escape -= dt;
    const escaping = auto.escape > 0;
    if (auto.timer <= 0) {
      auto.timer = 0.3;
      // candidates in a cone around current heading + a few wild ones
      // (escaping a stall: all wild, probing much farther out)
      let best = null, bestScore = -Infinity;
      for (let i = 0; i < 12; i++) {
        const wild = escaping || i >= 9;
        const spread = wild ? 1.0 : 0.35;
        let d = V.norm(V.add(auto.dir, [
          (autoRnd() * 2 - 1) * spread,
          (autoRnd() * 2 - 1) * spread,
          (autoRnd() * 2 - 1) * spread,
        ]));
        const probe = V.add(world.camPos, V.scale(d, de * (escaping ? 6 + autoRnd() * 14 : 2.2)));
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
    let [yd, pd] = input.consumeLook();
    let roll = input.rollInput();
    if (state.photo) { yd = 0; pd = 0; roll = 0; } // frozen: the frame refines
    if (yd) rotateBasis(cam.up, yd * camParity);
    if (pd) rotateBasis(cam.right, pd * camParity);
    if (roll) rotateBasis(cam.forward, roll * 1.4 * dt * camParity);
    // only re-orthonormalize when the basis actually changed: a perfectly
    // still camera must produce bit-identical uniforms for frame accumulation
    if (yd || pd || roll) orthonormalize();

    // move
    state.deCam = world.de(world.camPos);
    let moving = false;
    if (state.photo) {
      // camera frozen
    } else if (state.auto) {
      autopilot(dt);
      moving = true;
    } else {
      const mv = input.moveVector();
      if (mv[0] || mv[1] || mv[2]) {
        moving = true;
        const speed = (FLY_SPEED / world.fogMul) * input.speedMult;
        const delta = V.add(
          V.add(V.scale(cam.right, mv[0] * speed * dt), V.scale(cam.up, mv[1] * speed * dt)),
          V.scale(cam.forward, mv[2] * speed * dt));
        tryMove(delta);
      }
    }

    // rebase to keep camera-local numbers O(1); rotate the view basis along
    state.deCam = world.rebaseToCamera();
    world.tick(dt);
    world.updateFog(state.deCam, dt);
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
      glow: state.glow, deCam: state.deCam, still,
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
          state.deCam = world.rebaseToCamera();
          world.tick(1 / 60);
          world.updateFog(state.deCam, 1 / 60);
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
          glow: state.glow, deCam: world.de(world.camPos),
        });
        renderer.gl.finish();
      },
      // GPU map() at a camera-relative point, bit-exact via RGBA8 probe pixel
      gpuMap(p) {
        renderer.render(world, cam, {
          time: 1.0, fov: state.fov, quality: state.quality,
          glow: state.glow, probe: p, deCam: world.de(world.camPos),
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
