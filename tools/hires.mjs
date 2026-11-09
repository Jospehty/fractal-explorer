/* hires.mjs — final-quality still renders of a scouted viewpoint.
   Reproduces a tools/tour.mjs view (same deterministic autopilot dive),
   optionally nudges the framing and applies a photo-mode look, then renders
   at full refinement (finest LOD, HQ shadows) and accumulates jittered frames.
   Usage:
     node tools/hires.mjs <seed> <K> <wall|along|out> <out.png> [W=1920] [H=1080]
                          [frames=24] ['{"look":{...},"yaw":deg,"pitch":deg,"fov":deg}'] */
import { chromium } from 'playwright';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [seed, K, view, out, W = '1920', H = '1080', frames = '24', optArg = '{}'] = process.argv.slice(2);
const opt = JSON.parse(optArg);

const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'] });
const page = await browser.newPage({ viewport: { width: +W, height: +H } });
page.setDefaultTimeout(0);
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('file://' + resolve(root, 'index.html') + `?det=1&seed=${seed}&scale=1`);
await page.waitForFunction(() => window.__explorer !== undefined);
await page.evaluate(() => {
  document.getElementById('overlay').classList.add('hidden');
  document.getElementById('hud').classList.add('hidden');
});
await page.evaluate(async ({ K, view, opt }) => {
  const ex = window.__explorer, V = window.V, w = ex.world, cam = ex.cam;
  let g = 0;
  while (w.K < K && g++ < 600) { ex.step(20); await new Promise((r) => setTimeout(r, 0)); }
  // same framing as tools/tour.mjs
  const gr = w.grad(w.camPos);
  const helper = Math.abs(gr[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t = V.norm(V.cross(gr, helper));
  let f;
  if (view === 'wall') f = V.norm(V.add(V.scale(gr, -1), V.scale(t, 0.35)));
  else if (view === 'along') f = V.norm(V.add(t, V.scale(gr, -0.12)));
  else f = V.norm(V.add(gr, V.scale(t, 0.25)));
  cam.forward = f;
  cam.right = V.norm(V.cross(f, helper));
  cam.up = V.norm(V.cross(cam.right, f));
  // optional framing nudge (degrees)
  const rot = (axis, deg) => {
    const R = V.mAxisAngle(axis, deg * Math.PI / 180);
    cam.forward = V.mMulV(R, cam.forward); cam.right = V.mMulV(R, cam.right); cam.up = V.mMulV(R, cam.up);
  };
  if (opt.yaw) rot(cam.up, opt.yaw);
  if (opt.pitch) rot(cam.right, opt.pitch);
  if (opt.roll) rot(cam.forward, opt.roll);
  if (opt.fov) ex.state.fov = opt.fov * Math.PI / 180;
  if (opt.look) ex.renderer.setLook(opt.look);
}, { K: +K, view, opt });
const t0 = Date.now();
// still frames render progressively (a band of rows per call): call until
// `frames` complete samples are accumulated
await page.evaluate(() => { window.__explorer.renderer.forceStage = 2; }); // finest refinement from the start
let done = 0, calls = 0;
while (done < +frames) {
  done = await page.evaluate(() => {
    const ex = window.__explorer, R = ex.renderer;
    R.render(ex.world, ex.cam, { time: 1, fov: ex.state.fov, quality: 2, glow: 1,
      deCam: ex.world.de(ex.world.camPos), still: true });
    const gl = R.gl, px = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); // sync
    return R.accumN;
  });
  if (++calls === 1) console.log(`first band took ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
await page.screenshot({ path: out });
console.log('wrote', out, `${((Date.now() - t0) / 1000).toFixed(0)}s`);
await browser.close();
