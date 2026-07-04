/* input.js — pointer-lock mouse look, WASD flight, scroll-wheel speed,
   basic touch support. */
(function (global) {
  'use strict';

  class Input {
    constructor(canvas) {
      this.canvas = canvas;
      this.keys = new Set();
      this.yawDelta = 0;
      this.pitchDelta = 0;
      this.rollDelta = 0;
      this.speedMult = 1.0;
      this.pointerLocked = false;
      this.isDragging = false;
      this.onToggle = () => {};   // key callbacks (F, H, P, 1..3, N, R)
      this.touchActive = false;
      this._touchForward = 0;
      // Removed the click event since mousedown handles it better and avoids target-change cancellation
      document.addEventListener('pointerlockchange', () => {
        this.pointerLocked = document.pointerLockElement === canvas;
        this.onToggle('lock', this.pointerLocked);
      });
      document.addEventListener('mousedown', (e) => {
        if (e.target.closest('#help')) return;
        if (!this.pointerLocked && !this.touchActive) canvas.requestPointerLock();
        if (e.button === 0) {
          this.isDragging = true;
          this.onToggle('lock', true);
        }
      });
      document.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
          this.isDragging = false;
          // Only bring back the overlay if pointer lock actually failed
          if (!this.pointerLocked) {
            this.onToggle('lock', false);
          }
        }
      });
      document.addEventListener('mousemove', (e) => {
        if (!this.pointerLocked && !this.isDragging) return;
        // negative: positive rotation about cam.up/right turns the view
        // left/up, so FPS convention (mouse right => look right) needs the flip
        this.yawDelta -= e.movementX * 0.0021;
        this.pitchDelta -= e.movementY * 0.0021;
      });
      document.addEventListener('keydown', (e) => {
        if (e.repeat) return;
        this.keys.add(e.code);
        const map = {
          KeyF: 'auto', KeyH: 'help', KeyP: 'pause', KeyN: 'newseed',
          Digit1: 'q0', Digit2: 'q1', Digit3: 'q2', KeyG: 'glow',
          KeyL: 'light', KeyO: 'autolights', KeyC: 'lightcolor', KeyX: 'removelight',
        };
        if (map[e.code]) this.onToggle(map[e.code]);
      });
      document.addEventListener('keyup', (e) => this.keys.delete(e.code));
      window.addEventListener('blur', () => this.keys.clear());

      canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        // scroll is the ONLY speed control (movement speed is otherwise
        // constant per zoom level), so give it a wide range
        this.speedMult *= Math.exp(-e.deltaY * 0.0011);
        this.speedMult = Math.min(Math.max(this.speedMult, 0.02), 100);
      }, { passive: false });

      // ---- touch: 1 finger = look, 2 fingers = fly forward / pinch = speed
      let touches = new Map();
      let pinchDist = 0;
      canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        this.touchActive = true;
        for (const t of e.changedTouches) touches.set(t.identifier, { x: t.clientX, y: t.clientY });
        if (touches.size === 2) {
          const [a, b] = [...touches.values()];
          pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
        }
      }, { passive: false });
      canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (touches.size === 1) {
          const t = e.changedTouches[0];
          const prev = touches.get(t.identifier);
          if (prev) {
            this.yawDelta -= (t.clientX - prev.x) * 0.004;
            this.pitchDelta -= (t.clientY - prev.y) * 0.004;
            touches.set(t.identifier, { x: t.clientX, y: t.clientY });
          }
        } else if (touches.size >= 2) {
          for (const t of e.changedTouches)
            if (touches.has(t.identifier)) touches.set(t.identifier, { x: t.clientX, y: t.clientY });
          const [a, b] = [...touches.values()];
          const nd = Math.hypot(a.x - b.x, a.y - b.y);
          if (pinchDist > 0) this.speedMult = Math.min(Math.max(this.speedMult * (nd / pinchDist), 0.02), 100);
          pinchDist = nd;
          this._touchForward = 1; // two fingers down = fly forward
        }
      }, { passive: false });
      const touchEnd = (e) => {
        for (const t of e.changedTouches) touches.delete(t.identifier);
        if (touches.size < 2) { this._touchForward = 0; pinchDist = 0; }
      };
      canvas.addEventListener('touchend', touchEnd);
      canvas.addEventListener('touchcancel', touchEnd);
    }

    // movement intent in camera space: [right, up, forward]
    moveVector() {
      const k = this.keys;
      let f = (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0) + this._touchForward;
      let r = (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0);
      let u = (k.has('Space') ? 1 : 0) - (k.has('ShiftLeft') || k.has('ShiftRight') ? 1 : 0);
      return [r, u, f];
    }

    rollInput() {
      return (this.keys.has('KeyE') ? 1 : 0) - (this.keys.has('KeyQ') ? 1 : 0);
    }

    consumeLook() {
      const out = [this.yawDelta, this.pitchDelta];
      this.yawDelta = 0;
      this.pitchDelta = 0;
      return out;
    }
  }

  global.Input = Input;
})(typeof window !== 'undefined' ? window : globalThis);
