/* seamtest.mjs — rebase continuity: render, force descend (no movement), render. */
import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
mkdirSync(resolve(root, 'shots-debug'), { recursive: true });
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
const K = parseInt(process.argv[2] || '12', 10);
await page.evaluate(async (K) => {
  const ex = window.__explorer;
  let g = 0;
  while (ex.world.K < K && g++ < 400) { ex.step(30); await new Promise(r => setTimeout(r, 0)); }
}, K);
await page.evaluate(() => window.__explorer.renderOnce());
await page.screenshot({ path: 'shots-debug/seam_before.png' });
const info = await page.evaluate(() => {
  const ex = window.__explorer, V = window.V;
  const Kb = ex.world.K;
  const boundB = ex.world.bound, BeffB = ex.world.outer.length, WSb = ex.world.WScale;
  ex.world.descend();                       // force rebase with zero camera movement
  const R = ex.world.takeFrameRotation();
  ex.cam.right = V.norm(V.mMulV(R, ex.cam.right));
  ex.cam.up = V.norm(V.mMulV(R, ex.cam.up));
  ex.cam.forward = V.norm(V.mMulV(R, ex.cam.forward));
  ex.world.updateChain();
  ex.renderOnce();
  return { Kb, Ka: ex.world.K, boundB, boundA: ex.world.bound, BeffB, BeffA: ex.world.outer.length, WSb, WSa: ex.world.WScale };
});
await page.screenshot({ path: 'shots-debug/seam_after.png' });
console.log('K', info.Kb, '->', info.Ka, 'bound', info.boundB, '->', info.boundA, 'Beff', info.BeffB, '->', info.BeffA);
await browser.close();
