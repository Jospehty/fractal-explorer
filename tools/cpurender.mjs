/* cpurender.mjs — dive to a depth, then render a tiny CPU image via world.de
   and save the GPU screenshot for the same state. Divergence localizer. */
import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outdir = resolve(root, 'shots-debug');
mkdirSync(outdir, { recursive: true });
const targetK = parseInt(process.argv[2] || '12', 10);

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'],
});
const page = await browser.newPage({ viewport: { width: 480, height: 270 } });
page.setDefaultTimeout(240000);
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('file://' + resolve(root, 'index.html') + '?det=1&seed=1337&scale=1' + (process.argv[3] ? '&sdbg=' + process.argv[3] : ''));
await page.waitForFunction(() => window.__explorer !== undefined);
await page.evaluate(() => {
  document.getElementById('overlay').classList.add('hidden');
  document.getElementById('hud').classList.add('hidden');
});

const res = await page.evaluate(async (targetK) => {
  const ex = window.__explorer;
  const V = window.V;
  let guard = 0;
  while (ex.world.K < targetK && guard++ < 400) {
    ex.step(30);
    await new Promise((r) => setTimeout(r, 0));
  }
  const w = ex.world, cam = ex.cam;
  // CPU render: 72x40 rays, march world.de
  const W = 72, H = 40;
  const fovTan = Math.tan(ex.state.fov / 2);
  let out = '';
  const chars = ' .:-=+*#%@';
  let hits = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = ((x + 0.5) / W * 2 - 1) * fovTan * (480 / 270);
      const py = -((y + 0.5) / H * 2 - 1) * fovTan;
      const rd = V.norm(V.add(V.add(V.scale(cam.right, px), V.scale(cam.up, py)), cam.forward));
      let t = 0, hit = false;
      const pixA = 2 * fovTan / H;
      for (let i = 0; i < 200; i++) {
        const d = w.de(V.scale(rd, t).map((v, k) => v + w.camPos[k]));
        if (d < t * pixA * 0.45 + 2e-6) { hit = true; break; }
        t += d * 0.92;
        if (t > 3e4) break;
      }
      if (hit) hits++;
      out += hit ? chars[Math.max(0, Math.min(9, 9 - Math.floor(Math.log10(t + 1e-6) + 3)))] : ' ';
    }
    out += '\n';
  }
  return { K: w.K, de: w.de(w.camPos), ascii: out, hitFrac: hits / (W * H), B: w.outer.length };
}, targetK);

console.log(`K=${res.K} B=${res.B} de=${res.de} cpuHitFrac=${(res.hitFrac * 100).toFixed(1)}%`);
console.log(res.ascii);
await page.evaluate(() => window.__explorer.renderOnce());
await page.screenshot({ path: `${outdir}/gpu_K${res.K}.png` });
console.log('gpu screenshot:', `${outdir}/gpu_K${res.K}.png`);
await browser.close();
