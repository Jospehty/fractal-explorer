/* lookdev.mjs — render the SAME frames under several lighting/grade variants.
   Usage:
     node tools/lookdev.mjs <seed> <depths e.g. 12,30> <views e.g. wall,out> '<json array of look overrides>' <out.png>
   Each variant is a partial LOOK object (see LOOK_DEFAULTS in js/shaders.js);
   rows = variants, columns = (depth, view) pairs. */
import { chromium } from 'playwright';
import { execSync } from 'child_process';
import { mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [seed = '1337', depthArg = '12,30', viewArg = 'wall,along,out', varArg = '[{}]', out = 'lookdev.png'] = process.argv.slice(2);
const depths = depthArg.split(',').map(Number);
const views = viewArg.split(',');
const variants = JSON.parse(varArg);
const W = 400, H = 225;
const tmp = out.replace(/\.png$/, '') + '_parts';
mkdirSync(tmp, { recursive: true });

const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'] });
const page = await browser.newPage({ viewport: { width: W, height: H } });
page.setDefaultTimeout(900000);
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('file://' + resolve(root, 'index.html') + `?det=1&seed=${seed}&scale=1${process.env.LD_EXTRA || ''}`);
await page.waitForFunction(() => window.__explorer !== undefined);
await page.evaluate(() => {
  document.getElementById('overlay').classList.add('hidden');
  document.getElementById('hud').classList.add('hidden');
});

const grid = variants.map(() => []);
for (const K of depths) {
  await page.evaluate(async (K) => {
    const ex = window.__explorer;
    let g = 0;
    while (ex.world.K < K && g++ < 600) { ex.step(20); await new Promise((r) => setTimeout(r, 0)); }
    const c = ex.cam;
    window.__saved = { f: c.forward.slice(), r: c.right.slice(), u: c.up.slice() };
  }, K);
  for (const view of views) {
    for (let vi = 0; vi < variants.length; vi++) {
      await page.evaluate(({ view, look }) => {
        const ex = window.__explorer, V = window.V, w = ex.world, cam = ex.cam;
        window.__LOOK = look;
        ex.renderer.resetLook(); // look settings are live uniforms
        const g = w.grad(w.camPos);
        const helper = Math.abs(g[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
        const t = V.norm(V.cross(g, helper));
        let f;
        if (view === 'wall') f = V.norm(V.add(V.scale(g, -1), V.scale(t, 0.35)));
        else if (view === 'along') f = V.norm(V.add(t, V.scale(g, -0.12)));
        else if (view === 'out') f = V.norm(V.add(g, V.scale(t, 0.25)));
        else f = window.__saved.f; // 'auto' = autopilot heading
        cam.forward = f;
        cam.right = V.norm(V.cross(f, helper));
        cam.up = V.norm(V.cross(cam.right, f));
        ex.renderOnce();
      }, { view, look: variants[vi] });
      const file = `${tmp}/v${vi}_K${K}_${view}.png`;
      await page.screenshot({ path: file });
      grid[vi].push(file);
    }
  }
  await page.evaluate(() => {
    const c = window.__explorer.cam, s = window.__saved;
    c.forward = s.f; c.right = s.r; c.up = s.u;
  });
  console.log('K', K, 'done');
}
await browser.close();
const cols = depths.length * views.length;
execSync(`montage ${grid.flat().join(' ')} -tile ${cols}x -geometry ${W}x${H}+2+2 ${out}`);
console.log('wrote', out);
