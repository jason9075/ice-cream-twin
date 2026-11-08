// Headless smoke test: load the page, run a scripted scoop through window.sim, save screenshots.
//   npm run build && npx vite preview --port 4173 &   then   node tools/shot.mjs [url] [outdir]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const url = process.argv[2] || 'http://localhost:4173/';
const out = process.argv[3] || 'shots';
fs.mkdirSync(out, { recursive: true });
const chrome = process.env.CHROME || '/usr/bin/google-chrome';

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,PdfOopif', '--use-angle=vulkan', '--window-size=1400,860', '--no-sandbox'],
  // puppeteer's own --enable-features would override ours (last switch wins)
  ignoreDefaultArgs: ['--enable-features=PdfOopif'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 860 });
const errors = [];
// WebGPU canvases are not captured by headless screenshots, so grab an offscreen render instead
async function shot(name) {
  const url = await page.evaluate(() => window.__view.capturePNG(1400, 860));
  fs.writeFileSync(`${out}/${name}`, Buffer.from(url.split(',')[1], 'base64'));
  await page.screenshot({ path: `${out}/ui-${name}` });
}
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${m.text()}`);
});
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.sim && window.__view, { timeout: 30000 });
await new Promise((r) => setTimeout(r, 1500));
await shot('01-start.png');

// scripted scoop in the strawberry bowl (middle), run headless-fast with stepFrames
const state = await page.evaluate(() => {
  const s = window.sim;
  s.pause(true);
  const q = (x, y, z) => {
    const e = { x, y, z };
    const c = Math.cos, sn = Math.sin;
    // XYZ euler -> quaternion
    const c1 = c(e.x / 2), c2 = c(e.y / 2), c3 = c(e.z / 2), s1 = sn(e.x / 2), s2 = sn(e.y / 2), s3 = sn(e.z / 2);
    return [s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3];
  };
  const go = (p, quat, n) => {
    s.setTarget({ position: p, quaternion: quat });
    return s.stepFrames(n);
  };
  go([-0.02, 0.14, 0], q(0, 0, 0.5), 80);
  go([-0.015, 0.086, 0], q(0, 0, 0.5), 150);
  go([0.025, 0.088, 0], q(0, 0, 0), 150);
  return go([0.03, 0.15, 0.0], q(0, 0, 0), 100);
});
await page.evaluate(() => window.sim.pause(false));
await new Promise((r) => setTimeout(r, 800));
await shot('02-scooped.png');

// close-up of the spoon
await page.evaluate(() => {
  const { camera } = window.__view;
  camera.lookFrom({ x: 0.11, y: 0.24, z: 0.2 }, { x: 0.03, y: 0.13, z: 0 });
});
await new Promise((r) => setTimeout(r, 500));
await shot('03-closeup.png');

console.log(JSON.stringify({ scoopedGrams: state.scoopedGrams, particles: state.particles, bowls: state.bowlGrams, blocked: state.blocked }, null, 1));
console.log(errors.length ? errors.join('\n') : 'no console errors');
await browser.close();
