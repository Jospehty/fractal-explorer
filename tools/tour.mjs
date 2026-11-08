/* tour.mjs — headless visual review harness.
   Dives with the deterministic autopilot and, at each target depth, renders
   three framings that together show what the world looks like from there:
     wall  — facing the nearest surface (detail, lighting, materials)
     along — grazing along it (relief, shadows)
     out   — facing away from it (vistas: how big the world above you is)
   Usage: node tools/tour.mjs [seed] [outdir] [depths=2,8,16,30,60] [extra-url-params]
   Writes <outdir>/<seed>_K<k>_<view>.png and a contact sheet if ImageMagick
   `montage` exists. Holding still accumulates frames (accum=N param). */
import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { execSync } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const seed = process.argv[2] || '1337';
const outdir = process.argv[3] || resolve(root, 'shots');
const depths = (process.argv[4] || '2,8,16,30,60').split(',').map(Number);
const extra = process.argv[5] || '';
const W = parseInt(process.env.TOUR_W || '480', 10), H = parseInt(process.env.TOUR_H || '270', 10);
mkdirSync(outdir, { recursive: true });

const browser = await chromium.launch({
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'],
});
const page = await browser.newPage({ viewport: { width: W, height: H } });
page.setDefaultTimeout(600000);
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.text()); });
await page.goto('file://' + resolve(root, 'index.html') + `?det=1&seed=${seed}&scale=1${extra}`);
await page.waitForFunction(() => window.__explorer !== undefined);
await page.evaluate(() => {
  document.getElementById('overlay').classList.add('hidden');
  document.getElementById('hud').classList.add('hidden');
});

const files = [];
for (const K of depths) {
  const info = await page.evaluate(async (K) => {
    const ex = window.__explorer;
    let guard = 0;
    while (ex.world.K < K && guard++ < 600) { ex.step(20); await new Promise((r) => setTimeout(r, 0)); }
    return { K: ex.world.K, z: ex.world.log10Zoom };
  }, K);
  if (info.K < K) { console.log('did not reach K=' + K); break; }
  await page.evaluate(() => {
    const c = window.__explorer.cam;
    window.__saved = { f: c.forward.slice(), r: c.right.slice(), u: c.up.slice() };
  });
  for (const view of ['wall', 'along', 'out']) {
    await page.evaluate((view) => {
      const ex = window.__explorer, V = window.V, w = ex.world, cam = ex.cam;
      const g = w.grad(w.camPos); // points away from the nearest surface
      const helper = Math.abs(g[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const t = V.norm(V.cross(g, helper));
      let f;
      if (view === 'wall') f = V.norm(V.add(V.scale(g, -1), V.scale(t, 0.35)));
      else if (view === 'along') f = V.norm(V.add(t, V.scale(g, -0.12)));
      else f = V.norm(V.add(g, V.scale(t, 0.25)));
      cam.forward = f;
      cam.right = V.norm(V.cross(f, helper));
      cam.up = V.norm(V.cross(cam.right, f));
      ex.renderOnce();
    }, view);
    const file = `${outdir}/${seed}_K${String(info.K).padStart(3, '0')}_${view}.png`;
    await page.screenshot({ path: file });
    files.push(file);
  }
  await page.evaluate(() => { // hand the autopilot its own heading back
    const c = window.__explorer.cam, s = window.__saved;
    c.forward = s.f; c.right = s.r; c.up = s.u;
  });
  console.log(`K=${info.K} zoom=1e${info.z.toFixed(1)}`);
}
await browser.close();
try {
  execSync(`montage ${files.join(' ')} -tile 3x -geometry ${W}x${H}+2+2 ${outdir}/${seed}_sheet.png`);
  console.log('sheet', `${outdir}/${seed}_sheet.png`);
} catch (e) { /* montage optional */ }
