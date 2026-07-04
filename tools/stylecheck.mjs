/* stylecheck.mjs — render each fold style in isolation at two depths. */
import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outdir = resolve(root, 'shots-style');
mkdirSync(outdir, { recursive: true });

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'],
});

const variants = process.argv[2]
  ? [process.argv[2]]
  : ['style=0', 'style=1', 'style=2', 'style=0&twist=0.08', 'style=1&twist=0.08', 'style=2&twist=0.08'];

for (const v of variants) {
  const page = await browser.newPage({ viewport: { width: 480, height: 270 } });
  page.setDefaultTimeout(180000);
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto('file://' + resolve(root, 'index.html') + `?det=1&seed=1337&scale=1&${v}`);
  await page.waitForFunction(() => window.__explorer !== undefined, { timeout: 30000 });
  await page.evaluate(() => {
    document.getElementById('overlay').classList.add('hidden');
    document.getElementById('hud').classList.add('hidden');
  });
  const name = v.replace(/[^a-z0-9]+/gi, '_');
  const shoot = async (tag) => {
    const info = await page.evaluate(() => {
      window.__explorer.renderOnce();
      return { K: window.__explorer.world.K, de: window.__explorer.state.deCam };
    });
    await page.screenshot({ path: `${outdir}/${name}_${tag}_K${info.K}.png` });
    console.log(`${name} ${tag} K=${info.K} de=${info.de}`);
  };
  await shoot('a');
  await page.evaluate(async () => {
    const ex = window.__explorer;
    let guard = 0;
    while (ex.world.K < 8 && guard++ < 300) { ex.step(30); await new Promise((r) => setTimeout(r, 0)); }
  });
  await shoot('b');
  await page.close();
}
await browser.close();
