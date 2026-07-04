/* screenshot.mjs — headless visual validation.
   Renders the explorer at increasing depths via the deterministic autopilot
   hooks and saves screenshots. Usage:
     node tools/screenshot.mjs [seed] [outdir]
*/
import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const seed = process.argv[2] || '1337';
const outdir = process.argv[3] || resolve(root, 'shots');
mkdirSync(outdir, { recursive: true });

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: [
    '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader',
    '--disable-gpu-sandbox',
  ],
});
const page = await browser.newPage({ viewport: { width: 480, height: 270 } });
page.setDefaultTimeout(180000);
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.type(), m.text());
});
page.on('pageerror', (e) => console.log('[pageerror]', e.message));

const extra = process.argv[4] || '';
const url = 'file://' + resolve(root, 'index.html') + `?det=1&seed=${seed}&scale=1${extra}`;
console.log('loading', url);
await page.goto(url);
await page.waitForFunction(() => window.__explorer !== undefined, { timeout: 30000 });

// hide UI for clean shots
await page.evaluate(() => {
  document.getElementById('overlay').classList.add('hidden');
  document.getElementById('hud').classList.add('hidden');
});

async function shoot(name) {
  const info = await page.evaluate(() => {
    window.__explorer.renderOnce();
    const w = window.__explorer.world;
    return { K: w.K, zoom: w.log10Zoom.toFixed(1), de: window.__explorer.state.deCam };
  });
  const file = `${outdir}/${name}_K${info.K}_z${info.zoom}.png`;
  await page.screenshot({ path: file });
  console.log('saved', file, 'de=', info.de);
  return info;
}

await shoot(`${seed}_00_start`);

// dive to increasing depths with the autopilot
const targets = [2, 5, 9, 14, 20, 30];
for (let ti = 0; ti < targets.length; ti++) {
  const target = targets[ti];
  const ok = await page.evaluate(async (target) => {
    const ex = window.__explorer;
    let guard = 0;
    while (ex.world.K < target && guard++ < 400) {
      ex.step(30);
      await new Promise((r) => setTimeout(r, 0));
    }
    return ex.world.K >= target;
  }, target);
  if (!ok) { console.log(`did not reach K=${target}`); break; }
  await shoot(`${seed}_${String(ti + 1).padStart(2, '0')}`);
}

// rebase seam check: capture immediately before/after one descend
await page.evaluate(() => {
  const ex = window.__explorer;
  // walk forward until just above the descend threshold
  let guard = 0;
  while (ex.world.de(ex.world.camPos) > 0.0215 && guard++ < 2000) ex.step(1);
});
await shoot(`${seed}_seamA`);
await page.evaluate(() => window.__explorer.step(8));
await shoot(`${seed}_seamB`);

await browser.close();
console.log('done');
