// Checks the WebGPU particle solver (src/gpu/fluid-gpu.js) against the CPU reference (src/sim/fluid.js)
// in headless Chrome with a real GPU:
//   npm run build && npx vite preview --port 4173 &   then   node tools/fluid-gpu-check.mjs [url]
// 1. one sub-step from the same state: positions/velocities/contacts must agree to f32 precision
// 2. the same arm-driven scoop + tip-out on both solvers: scooped grams, mass, no escapes, frame time
// 3. throughput with many loose particles
import puppeteer from 'puppeteer-core';

const url = process.argv[2] || 'http://localhost:4173/';
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || '/usr/bin/google-chrome',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--no-sandbox'],
});
const errors = [];
let failed = 0;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed++;
};

async function open(solver) {
  const page = await browser.newPage();
  page.on('console', (m) => (m.type() === 'error' || m.type() === 'warning') && errors.push(`${m.type()}: ${m.text()}`));
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  await page.goto(`${url}?fluid=${solver}`, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => window.sim && window.__view?.sim, { timeout: 30000 });
  await page.evaluate(() => window.sim.pause(true));
  return page;
}

// shared in-page helpers: the arm scoop from tools/shot.mjs, then tip the scoop out over the bowl
const SCRIPT = `
  window.__q = (pitch) => {
    const a = Math.PI / 4, b = pitch / 2;
    return [Math.sin(a) * Math.sin(b), Math.sin(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.cos(a) * Math.cos(b)];
  };
  window.__go = async (p, pitch, n, log) => {
    window.sim.setTarget({ position: p, quaternion: window.__q(pitch) });
    for (let i = 0; i < n; i++) {
      const t0 = performance.now();
      await window.sim.stepFrames(1);
      log?.push(performance.now() - t0);
    }
  };
  window.__scoopAndTip = async (log) => {
    await __go([0, 0.14, 0.02], 0.5, 80, log);
    await __go([0, 0.086, 0.015], 0.5, 150, log);
    await __go([0, 0.088, -0.025], 0, 150, log);
    await __go([0, 0.15, -0.03], 0, 100, log);
    const scooped = window.sim.getState().scoopedGrams;
    window.sim.release();
    await __go([0, 0.15, -0.03], 0, 240, log);
    return scooped;
  };
`;

// ------------------------------------------------------------------ 1. one sub-step, same state
{
  const page = await open('cpu');
  await page.evaluate(SCRIPT);
  const r = await page.evaluate(async () => {
    const { sim, gpuFluid } = window.__view;
    const f = sim.fluid;
    // tip a scoop out so there are many particles in motion and in contact
    await __go([0, 0.14, 0.02], 0.5, 80);
    await __go([0, 0.086, 0.015], 0.5, 150);
    await __go([0, 0.088, -0.025], 0, 150);
    await __go([0, 0.15, -0.03], 0, 60);
    window.sim.release();
    await __go([0, 0.15, -0.03], 0, 25);
    const n = f.n;
    const keys = ['pos', 'vel', 'age', 'contact', 'cn', 'sv'];
    const snap = Object.fromEntries(keys.map((k) => [k, f[k].slice()]));
    const sq = sim._squeezed.slice();
    const restore = () => {
      for (const k of keys) f[k].set(snap[k]);
      sim._squeezed.set(sq);
      f.n = n;
    };
    const dt = 1 / 60 / Math.max(1, sim.params.fluid.substeps | 0);
    f.step(dt, sim.params.fluid, sim);
    const cpu = { pos: f.pos.slice(0, 3 * n), vel: f.vel.slice(0, 3 * n), contact: f.contact.slice(0, n) };
    restore();
    await gpuFluid.run(dt, 1, sim);
    const dp = [];
    const dv = [];
    let contactDiff = 0;
    for (let i = 0; i < n; i++) {
      dp.push(Math.hypot(f.pos[i * 3] - cpu.pos[i * 3], f.pos[i * 3 + 1] - cpu.pos[i * 3 + 1], f.pos[i * 3 + 2] - cpu.pos[i * 3 + 2]));
      dv.push(Math.hypot(f.vel[i * 3] - cpu.vel[i * 3], f.vel[i * 3 + 1] - cpu.vel[i * 3 + 1], f.vel[i * 3 + 2] - cpu.vel[i * 3 + 2]));
      if (f.contact[i] !== cpu.contact[i]) contactDiff++;
    }
    const pct = (a, q) => a.slice().sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(a.length * q))];
    restore();
    return { n, p50: pct(dp, 0.5), p99: pct(dp, 0.99), pmax: pct(dp, 1), v99: pct(dv, 0.99), vmax: pct(dv, 1), contactDiff, r: f.r };
  });
  console.log(`one sub-step, ${r.n} particles: |dpos| p50 ${r.p50.toExponential(2)} p99 ${r.p99.toExponential(2)} max ${r.pmax.toExponential(2)} m;` +
    ` |dvel| p99 ${r.v99.toExponential(2)} max ${r.vmax.toExponential(2)} m/s; contact differs on ${r.contactDiff}`);
  check(r.n > 200, `enough particles in motion (${r.n})`);
  check(r.p99 < 1e-6, 'GPU sub-step matches the CPU (99% of particles within 1 µm)');
  check(r.pmax < 0.05 * r.r, 'no particle more than 5% of a radius off');
  check(r.contactDiff <= Math.max(2, r.n * 0.005), 'contact flags agree');
  await page.close();
}

// ------------------------------------------------------------------ 2. the same scenario on both
const runs = {};
for (const solver of ['cpu', 'gpu']) {
  const page = await open(solver);
  await page.evaluate(SCRIPT);
  runs[solver] = await page.evaluate(async () => {
    const { sim } = window.__view;
    const log = [];
    let peak = 0;
    const scooped = await __scoopAndTip(log);
    peak = Math.max(peak, sim.fluid.n);
    const f = sim.fluid;
    let escaped = 0;
    for (let i = 0; i < f.n; i++) if (f.pos[i * 3 + 1] < f.r * 0.5 || !Number.isFinite(f.pos[i * 3])) escaped++;
    const st = window.sim.getState();
    const ms = log.slice().sort((a, b) => a - b);
    return {
      solver: st.fluidSolver,
      scooped,
      spilled: st.spilledGrams,
      particles: f.n,
      escaped,
      total: Object.values(st.bowlGrams).reduce((a, b) => a + b, 0) + st.scoopedGrams + st.eatenGrams + st.spilledGrams,
      medianMs: ms[Math.floor(ms.length / 2)],
    };
  });
  const r = runs[solver];
  console.log(`${solver}: solver=${r.solver} scooped ${r.scooped.toFixed(1)} g, spilled ${r.spilled.toFixed(2)} g, ${r.particles} particles left, median frame ${r.medianMs.toFixed(2)} ms`);
  check(r.solver === solver, `${solver} page runs the ${solver} solver`);
  check(r.escaped === 0, `${solver}: no particle under the table or NaN`);
  await page.close();
}
// carving already sheds particles during the scoop, so the two runs part ways from there on
check(Math.abs(runs.gpu.scooped - runs.cpu.scooped) < 0.15 * runs.cpu.scooped, `scooped grams agree within 15% (${runs.cpu.scooped.toFixed(1)} vs ${runs.gpu.scooped.toFixed(1)} g)`);
check(Math.abs(runs.gpu.total - runs.cpu.total) < 0.5, `ice-cream mass the same on both (${runs.cpu.total.toFixed(2)} vs ${runs.gpu.total.toFixed(2)} g)`);

// ------------------------------------------------------------------ 3. throughput
for (const N of [4000, 15000]) {
  const res = {};
  for (const solver of ['cpu', 'gpu']) {
    const page = await open(solver);
    res[solver] = await page.evaluate(async (N) => {
      const { sim } = window.__view;
      sim.params.fluid.maxParticles = N;
      const f = sim.fluid;
      f.configure(sim.params.fluid.particleRadius, sim.params.fluid.kernelScale);
      // a loose block of ice cream dropped onto the left bowl's ice, at rest spacing
      const s = f.d * 1.05;
      const side = Math.ceil(Math.sqrt(N / 12));
      for (let k = 0; f.n < N; k++) {
        const i = k % side, j = Math.floor(k / side) % side, l = Math.floor(k / side / side);
        f.spawn(-0.18 + (i - side / 2) * s, 0.13 + l * s, (j - side / 2) * s, 0, 0, 0, [240, 230, 200, 0]);
      }
      sim.params.fluid.stopSpeed = 0; // keep them all loose for the measurement
      sim.params.fluid.stampOnContact = false;
      const t = [];
      for (let k = 0; k < 60; k++) {
        const t0 = performance.now();
        await window.sim.stepFrames(1);
        t.push(performance.now() - t0);
      }
      t.sort((a, b) => a - b);
      return { n: f.n, ms: t[30] };
    }, N);
    await page.close();
  }
  console.log(`${N} loose particles: CPU ${res.cpu.ms.toFixed(1)} ms/frame, GPU ${res.gpu.ms.toFixed(1)} ms/frame (${(res.cpu.ms / res.gpu.ms).toFixed(1)}x)`);
  check(res.gpu.n === N && res.cpu.n === N, `${N}: all particles still loose`);
}

check(errors.length === 0, `no console errors${errors.length ? `:\n  ${errors.join('\n  ')}` : ''}`);
await browser.close();
process.exit(failed ? 1 : 0);
