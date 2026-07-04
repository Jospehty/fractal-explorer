/* gpuprobe.mjs — compare GPU map() vs CPU world.de at identical points. */
import { chromium } from 'playwright';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const targetK = parseInt(process.argv[2] || '25', 10);

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--disable-gpu-sandbox'],
});
const page = await browser.newPage({ viewport: { width: 64, height: 64 } });
page.setDefaultTimeout(240000);
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => console.log('[page]', m.text()));
await page.goto('file://' + resolve(root, 'index.html') + '?det=1&seed=1337&scale=1');
await page.waitForFunction(() => window.__explorer !== undefined);

const out = await page.evaluate(async (targetK) => {
  const ex = window.__explorer;
  const V = window.V;
  let guard = 0;
  while (ex.world.K < targetK && guard++ < 400) {
    ex.step(30);
    await new Promise((r) => setTimeout(r, 0));
  }
  let rs = 424242;
  const rnd = () => { rs = (Math.imul(rs, 1103515245) + 12345) >>> 0; return rs / 4294967296; };
  const rows = [];
  for (let i = 0; i < 60; i++) {
    const r = Math.pow(10, -3 + rnd() * 6);
    const dir = V.norm([rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1]);
    const p = V.scale(dir, r);
    const cpu = ex.world.de(V.add(ex.world.camPos, p));
    const gpu = ex.gpuMap(p);
    const err = Math.abs(gpu - cpu);
    const bad = err > Math.max(3e-4 * Math.abs(cpu), 5e-6 + 3e-5 * r);
    rows.push({ r: r.toExponential(2), cpu, gpu, err, bad });
  }
  return { K: ex.world.K, B: ex.world.outer.length, rows };
}, targetK);

console.log(`K=${out.K} B=${out.B}`);
let bads = 0;
for (const row of out.rows) {
  if (row.bad) { bads++; console.log('BAD', JSON.stringify(row)); }
}
console.log(`${bads}/${out.rows.length} bad probes`);
if (bads === 0) console.log('GPU matches CPU at sampled points');
await browser.close();
