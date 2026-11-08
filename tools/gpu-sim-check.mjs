// Checks the GPU-resident simulation (src/gpu/sim) against the CPU reference in headless Chrome with a
// real GPU. Starts its own Vite dev server so the page can import the modules directly.
//   node tools/gpu-sim-check.mjs [test ...]   (default: every test in ALL; DIAG ones by name, see tools/gpu-sim-tests.js)
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const only = process.argv.slice(2);
const server = await createServer({ server: { port: 5183, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || '/usr/bin/google-chrome',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--no-sandbox'],
});
let failed = 0;
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log(`pageerror: ${e.message}`));
  if (process.env.DEBUG) page.on('console', (m) => console.log(`[page] ${m.text()}`));
  await page.goto('http://localhost:5183/tools/gpu-test.html');
  await page.waitForFunction(() => window.ready, { timeout: 30000 });
  if (process.env.DEBUG) await page.evaluate(() => (window.debugDiff = true));
  console.log(`adapter: ${await page.evaluate(() => window.adapterInfo)}`);
  const { all, diag } = await page.evaluate(async () => {
    const m = await import('/tools/gpu-sim-tests.js');
    return { all: Object.keys(m.ALL), diag: Object.keys(m.DIAG) };
  });
  for (const name of only.length ? only : all) {
    if (!all.includes(name) && !diag.includes(name)) throw new Error(`unknown test ${name} (${[...all, ...diag].join(', ')})`);
    const r = await page.evaluate(async (n) => {
      try {
        const m = await import('/tools/gpu-sim-tests.js');
        return await (m.ALL[n] ?? m.DIAG[n])();
      } catch (e) {
        return { ok: false, msg: `threw: ${e.message}\n${e.stack}` };
      }
    }, name);
    console.log(`${r.ok ? 'ok  ' : 'FAIL'} ${name}: ${r.msg}`);
    if (!r.ok) failed++;
  }
  const errs = await page.evaluate(() => window.gpuErrors);
  if (errs.length) {
    console.log(`FAIL WebGPU errors:\n  ${errs.slice(0, 5).join('\n  ')}`);
    failed++;
  }
} finally {
  await browser.close();
  await server.close();
}
process.exit(failed ? 1 : 0);
