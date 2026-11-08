// Page-side checks of the GPU-resident simulation (src/gpu/sim) against the CPU reference
// (src/sim). Loaded by tools/gpu-sim-check.mjs into tools/gpu-test.html through the Vite dev server.
import * as M from '/src/math.js';
import { Sim } from '/src/sim/sim.js';
import { BOWLS } from '/src/sim/sdf.js';
import { GpuSim } from '/src/gpu/sim/gpu-sim.js';
import { ST, STU, CNT } from '/src/gpu/sim/layout.js';

const device = () => window.device;
const maxAbs = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

function cpuTotalGrams(sim) {
  const s = sim.getState();
  const bowls = Object.values(s.bowlGrams).reduce((a, b) => a + b, 0);
  return bowls + s.scoopedGrams + s.eatenGrams + s.spilledGrams + sim.grams(sim.fluid.n * sim.particleVolume()) + sim.grams(sim.debtVolume());
}

export async function scans() {
  const gs = new GpuSim(device());
  const n = 100003;
  const u = new Uint32Array(n).map((_, i) => (i * 2654435761) % 7);
  const f = new Float32Array(n).map((_, i) => ((i * 40503) % 1000) / 997);
  gs.debugWrite('X', gs.L.X.SCANU, u);
  gs.debugWrite('W', gs.L.W.DVD, f);
  await gs.debugRun((fr, s) => {
    s._scan(fr, 'u', s.L.X.SCANU, s.L.X.LIST, n);
    s._scan(fr, 'f', s.L.W.DVD, s.L.W.PD, n);
  });
  const gu = await gs.debugRead('X', gs.L.X.LIST, n);
  const gf = await gs.debugRead('W', gs.L.W.PD, n);
  let au = 0;
  let af = 0;
  let badU = 0;
  let relF = 0;
  for (let i = 0; i < n; i++) {
    if (gu[i] !== au) badU++;
    relF = Math.max(relF, Math.abs(gf[i] - af) / Math.max(1, af));
    au += u[i];
    af += f[i];
  }
  return { ok: badU === 0 && relF < 1e-5, msg: `prefix scans of ${n}: u32 mismatches ${badU}, f32 max rel error ${relF.toExponential(1)}` };
}

export async function volumes() {
  const gs = new GpuSim(device());
  await gs.debugRun((fr, s) => [0, 1, 2].forEach((k) => s._gridOcc(fr, k, ST.volume + k)));
  const st = await gs.debugRead('W', gs.L.W.ST, 64);
  const cpu = gs.grids.map((g) => g.volume());
  const rel = Math.max(...cpu.map((v, k) => Math.abs(st[ST.volume + k] - v) / v));
  return { ok: rel < 1e-5, msg: `bowl volumes: max rel error ${rel.toExponential(1)} (${cpu.map((v) => (v * 1e6).toFixed(2)).join(', ')} cm³)` };
}

/** Carve a path on a CPU Sim, then smooth the same edited voxels on both sides. */
export async function smoothing() {
  const ref = new Sim();
  const q = new M.Quaternion().setFromAxisAngle(new M.Vector3(0, 0, 1), 0.5);
  for (let k = 0; k <= 20; k++) ref.interact(new M.Vector3(0.01 - 0.03 * (k / 20), 0.09, 0), q, 1 / 60, true);
  const g = ref.grids[1];
  const flags = Uint32Array.from(g.touched);
  const gs = new GpuSim(device());
  const off = gs.L.grids[1].off;
  gs.debugWrite('W', gs.L.W.D + off, g.data);
  gs.debugWrite('X', gs.L.X.FLAG + off, flags);
  const nTouched = g.nTouched;
  const sm = ref.params.ice.smoothing;
  g.smoothTouched(sm);
  g.settle(sm);
  const resid = g.takeResidual();
  await gs.debugRun((fr, s) => {
    s._smooth(fr, 1, 0, sm);
    s._smooth(fr, 1, 1, sm);
  });
  const d = await gs.debugRead('W', gs.L.W.D + off, g.n);
  const st = await gs.debugRead('W', gs.L.W.ST, 64);
  const err = maxAbs(d, g.data) / g.cell;
  const dres = Math.abs(st[ST.residual + 1] - resid);
  return {
    ok: err < 1e-3 && dres < 1e-9,
    msg: `smoothTouched + settle on ${nTouched} voxels: max diff ${err.toExponential(1)} cells, residual ${(st[ST.residual + 1] * 1e9).toFixed(3)} vs ${(resid * 1e9).toFixed(3)} mm³`,
  };
}

/** The same spoon path carved by Sim.interact and by the GPU passes. */
export async function interact() {
  const ref = new Sim();
  const gs = new GpuSim(device());
  const path = [];
  const dig = (p) => new M.Quaternion().setFromAxisAngle(new M.Vector3(0, 1, 0), Math.PI / 2).multiply(new M.Quaternion().setFromAxisAngle(new M.Vector3(0, 0, 1), p));
  for (let k = 0; k <= 30; k++) path.push([new M.Vector3(0, 0.14 - 0.055 * (k / 30), 0.02 - 0.005 * (k / 30)), dig(0.5)]);
  for (let k = 1; k <= 30; k++) path.push([new M.Vector3(0, 0.086 + 0.002 * (k / 30), 0.015 - 0.04 * (k / 30)), dig(0.5 * (1 - k / 30))]);
  ref._spawnBudget = 40; // one frame's crumb budget, as on the GPU
  ref._clampContent(); // the content floor, as GpuSim.reset applies it
  for (const [p, qq] of path) ref.interact(p, qq, 1 / 60, true);
  gs.debugGlobals();
  await gs.debugRun((fr, s) => {
    fr.run('frameStart', 1);
    for (const [p, qq] of path) s._interact(fr, p, qq, true);
    s._compact(fr);
  });
  const L = gs.L;
  const bowl = await gs.debugRead('W', L.W.D + L.grids[1].off, L.grids[1].count);
  const content = await gs.debugRead('W', L.W.D + L.grids[3].off, L.grids[3].count);
  const st = await gs.debugRead('W', L.W.ST, 64);
  const cnt = await gs.debugRead('A', L.A.CNT, 16);
  const bins = await gs.debugRead('A', L.A.BINS, L.NBINS);
  let owed = 0;
  for (const b of bins) owed += b * 1e-14;
  const negs = st[ST.negSettle] + st[ST.negSettle + 1] + st[ST.negSettle + 2];
  const cpuOwed = ref.debtVolume();
  const eb = maxAbs(bowl, ref.grids[1].data) / ref.grids[1].cell;
  // empty voxels may sit above the band (the GPU applies the content floor at reset); compare within it
  const band = ref.content.band;
  // a few voxels sit exactly on a test boundary (lx < SPOON.a at lx = 0.028): f32 and f64 disagree there
  const bad = [];
  for (let i = 0; i < content.length; i++) {
    if (Math.abs(Math.min(content[i], band) - Math.min(ref.content.data[i], band)) > 1e-3 * ref.content.cell) bad.push([i, content[i], ref.content.data[i]]);
  }
  if (window.debugDiff) console.log('content diffs', bad.length, JSON.stringify(bad.slice(0, 12)));
  const cv = ref.content.volume();
  let gcv = 0;
  for (const v of content) gcv += Math.min(1, Math.max(0, 0.5 - v / ref.content.cell));
  gcv *= ref.content.cell ** 3;
  const msg =
    `${path.length} carve steps: bowl max diff ${eb.toExponential(1)} cells, scoop voxels differing ${bad.length} of ${content.length}, ` +
    `scoop ${(gcv * 1e6).toFixed(3)} vs ${(cv * 1e6).toFixed(3)} cm³, crumbs ${cnt[CNT.n]} vs ${ref.fluid.n}, ` +
    `owed ${((owed - negs) * 1e6).toFixed(4)} vs ${(cpuOwed * 1e6).toFixed(4)} cm³`;
  const ok = eb < 1e-3 && bad.length <= 20 && Math.abs(gcv - cv) < 1e-3 * cv && Math.abs(cnt[CNT.n] - ref.fluid.n) <= 2 && Math.abs(owed - negs - cpuOwed) < 0.02e-6 + 0.02 * Math.abs(cpuOwed);
  return { ok, msg };
}

/** Whole frames: the CPU test scenario 'scoop vanilla, carry it, eat it' on both. */
export async function frames() {
  const ref = new Sim();
  const gs = new GpuSim(device());
  ref.updateStats(true);
  const B = BOWLS[0];
  const level = new M.Quaternion();
  const dig = new M.Quaternion().setFromEuler(new M.Euler(0, 0, 0.5));
  const legs = [
    [new M.Vector3(B.cx - 0.03, 0.14, 0), dig, 90],
    [new M.Vector3(B.cx - 0.02, 0.085, 0), dig, 150],
    [new M.Vector3(B.cx + 0.025, 0.088, 0), level, 150],
    [new M.Vector3(B.cx + 0.025, 0.16, 0), level, 120],
  ];
  const before = cpuTotalGrams(ref);
  let gpuBefore = null;
  let worstDrift = 0;
  let maxParticles = 0;
  let tGpu = 0;
  let tCpu = 0;
  for (const [p, q, n] of legs) {
    ref.spoon.setTarget(p, q);
    gs.spoon.setTarget(p, q);
    for (let f = 0; f < n; f++) {
      let t = performance.now();
      ref.step(1 / 60);
      tCpu += performance.now() - t;
      t = performance.now();
      await gs.step(1 / 60);
      tGpu += performance.now() - t;
      if (f % 10 === 0) {
        await gs.sync();
        if (gpuBefore === null) gpuBefore = gs.totalGrams();
        worstDrift = Math.max(worstDrift, Math.abs(gs.totalGrams() - gpuBefore));
        maxParticles = Math.max(maxParticles, gs.stats.n);
      }
    }
  }
  await gs.sync();
  const s = gs.getState();
  const c = ref.getState();
  const errs = window.gpuErrors.slice();
  const eatenGpu = await gs.eat();
  const eatenCpu = ref.eat();
  const msg =
    `scoop vanilla (510 frames): scooped GPU ${s.scoopedGrams.toFixed(1)} g vs CPU ${c.scoopedGrams.toFixed(1)} g, ` +
    `vanilla left ${s.bowlGrams.vanilla.toFixed(1)} vs ${c.bowlGrams.vanilla.toFixed(1)} g, peak particles ${maxParticles}, ` +
    `GPU mass drift ${worstDrift.toFixed(3)} g (CPU ${Math.abs(cpuTotalGrams(ref) - before).toFixed(3)} g), ` +
    `eaten ${eatenGpu.toFixed(1)} vs ${eatenCpu.toFixed(1)} g, ${(tGpu / 510).toFixed(2)} vs ${(tCpu / 510).toFixed(2)} ms/frame` +
    (errs.length ? `\n  GPU errors: ${errs.slice(0, 3).join(' | ')}` : '');
  const ok = errs.length === 0 && s.scoopedGrams > 3 && Math.abs(s.scoopedGrams - c.scoopedGrams) < 0.25 * c.scoopedGrams && worstDrift < 1.0 && eatenGpu > 3;
  return { ok, msg };
}

/** Where a frame's GPU time goes, mid-scoop (needs 'timestamp-query'). */
export async function profile() {
  const gs = new GpuSim(device());
  const B = BOWLS[0];
  const dig = new M.Quaternion().setFromEuler(new M.Euler(0, 0, 0.5));
  gs.spoon.setTarget(new M.Vector3(B.cx - 0.03, 0.14, 0), dig);
  for (let f = 0; f < 90; f++) await gs.step(1 / 60);
  gs.spoon.setTarget(new M.Vector3(B.cx - 0.02, 0.085, 0), dig);
  for (let f = 0; f < 60; f++) await gs.step(1 / 60);
  await gs.sync();
  gs.profileNextFrame();
  const t0 = performance.now();
  await gs.step(1 / 60);
  await gs.sync();
  const wall = performance.now() - t0;
  const by = await gs.profileResult;
  const rows = Object.entries(by).sort((a, b) => b[1].ms - a[1].ms);
  const total = rows.reduce((s, r) => s + r[1].ms, 0);
  const top = rows.slice(0, 14).map(([k, v]) => `    ${k.padEnd(18)} ${v.ms.toFixed(3)} ms  x${v.n}`).join('\n');
  return { ok: true, msg: `GPU ${total.toFixed(2)} ms in passes, wall ${wall.toFixed(2)} ms, ${gs.stats.n} particles\n${top}` };
}

export async function timing() {
  const gs = new GpuSim(device());
  const B = BOWLS[0];
  const dig = new M.Quaternion().setFromEuler(new M.Euler(0, 0, 0.5));
  gs.spoon.setTarget(new M.Vector3(B.cx - 0.03, 0.14, 0), dig);
  for (let f = 0; f < 90; f++) await gs.step(1 / 60);
  gs.spoon.setTarget(new M.Vector3(B.cx - 0.02, 0.085, 0), dig);
  const rows = [];
  for (let f = 0; f < 20; f++) {
    const t0 = performance.now();
    await gs.step(1 / 60);
    rows.push([performance.now() - t0, gs.stats.waitMs, gs.stats.cpuMs, gs.stats.ops]);
  }
  const t1 = performance.now();
  await gs.device.queue.onSubmittedWorkDone();
  const done = performance.now() - t1;
  // the same frame's commands again, timed by onSubmittedWorkDone instead of the stats read-back
  const fr = gs._lastFrame;
  let t2 = performance.now();
  gs.device.queue.submit([fr.encode().finish()]);
  await gs.device.queue.onSubmittedWorkDone();
  const again = performance.now() - t2;
  t2 = performance.now();
  gs.device.queue.submit([fr.encode().finish()]);
  await gs.device.queue.onSubmittedWorkDone();
  const again2 = performance.now() - t2;

  const med = (k) => rows.map((r) => r[k]).sort((a, b) => a - b)[10];
  return { ok: true, msg: `step ${med(0).toFixed(2)} ms: waiting for the GPU ${med(1).toFixed(2)} ms, CPU ${med(2).toFixed(2)} ms, ${med(3)} dispatches/copies; last frame done after ${done.toFixed(2)} ms; same commands resubmitted: ${again.toFixed(2)}, ${again2.toFixed(2)} ms` };
}

/** Cost of a dispatch by itself: N trivial dispatches in one submission. */
export async function overhead() {
  const gs = new GpuSim(device());
  const out = [];
  for (const n of [1, 50, 150, 300]) {
    await gs.debugRun((fr) => { for (let k = 0; k < n; k++) fr.run('frameStart', 1); });
    const t0 = performance.now();
    await gs.debugRun((fr) => { for (let k = 0; k < n; k++) fr.run('frameStart', 1); });
    out.push(`${n}: ${(performance.now() - t0).toFixed(2)} ms`);
  }
  const big = [];
  for (const name of ['smWrite', 'freezeStamp', 'ledgerPay', 'ledgerBins', 'scanCells']) {
    const t0 = performance.now();
    await gs.debugRun((fr) => fr.run(name, 1, { i0: [0, 0, 0, 0], f0: [0.5, 0, 0, 0] }));
    big.push(`${name} ${(performance.now() - t0).toFixed(2)}`);
  }
  return { ok: true, msg: `trivial dispatches per submission -> wall time: ${out.join(', ')}; single kernels: ${big.join(', ')}` };
}

/** Which kernel makes a submission slow: resubmit the last frame without one kernel at a time. */
export async function bisect() {
  const gs = new GpuSim(device());
  const B = BOWLS[0];
  const dig = new M.Quaternion().setFromEuler(new M.Euler(0, 0, 0.5));
  gs.spoon.setTarget(new M.Vector3(B.cx - 0.03, 0.14, 0), dig);
  for (let f = 0; f < 90; f++) await gs.step(1 / 60);
  gs.spoon.setTarget(new M.Vector3(B.cx - 0.02, 0.085, 0), dig);
  for (let f = 0; f < 5; f++) await gs.step(1 / 60);
  await gs.sync();
  const fr = gs._lastFrame;
  const all = fr.ops.slice();
  const time = async (ops) => {
    fr.ops = ops;
    let best = 1e9;
    for (let r = 0; r < 3; r++) {
      const t = performance.now();
      gs.device.queue.submit([fr.encode().finish()]);
      await gs.device.queue.onSubmittedWorkDone();
      best = Math.min(best, performance.now() - t);
    }
    return best;
  };
  const base = await time(all);
  const names = [...new Set(all.filter((o) => o.kind !== 'c').map((o) => o.name))];
  const rows = [];
  for (const n of names) rows.push([n, base - (await time(all.filter((o) => o.name !== n)))]);
  rows.push(['(copies)', base - (await time(all.filter((o) => o.kind !== 'c')))]);
  rows.sort((a, b) => b[1] - a[1]);
  return { ok: true, msg: `frame ${base.toFixed(2)} ms; saved by dropping:\n` + rows.slice(0, 10).map(([n, d]) => `    ${n.padEnd(18)} ${d.toFixed(2)} ms`).join('\n') };
}

async function drive(gs, pos, quat, frames, every) {
  gs.spoon.setTarget(pos, quat);
  for (let f = 0; f < frames; f++) {
    await gs.step(1 / 60);
    if (every && f % 10 === 0) {
      await gs.sync();
      every();
    }
  }
}

/** Tilting past the release angle drops the scoop back as particles (CPU test of the same name). */
export async function release() {
  const gs = new GpuSim(device());
  const B = BOWLS[2];
  const id = new M.Quaternion();
  await drive(gs, new M.Vector3(B.cx, 0.14, 0), id, 60);
  await drive(gs, new M.Vector3(B.cx, 0.085, 0), id, 150);
  await drive(gs, new M.Vector3(B.cx, 0.15, 0), id, 120);
  await gs.sync();
  const scooped = gs.getState().scoopedGrams;
  const m0 = gs.totalGrams();
  let drift = 0;
  let peak = 0;
  const flip = new M.Quaternion().setFromEuler(new M.Euler(Math.PI * 0.9, 0, 0));
  const watch = () => {
    drift = Math.max(drift, Math.abs(gs.totalGrams() - m0));
    peak = Math.max(peak, gs.stats.n);
  };
  await drive(gs, new M.Vector3(B.cx, 0.15, 0), flip, 120, watch);
  await gs.sync();
  const after = gs.getState().scoopedGrams;
  for (let f = 0; f < 240; f++) await gs.step(1 / 60);
  await gs.sync();
  watch();
  const left = gs.stats.n;
  const ok = scooped > 2 && after < 0.5 && peak > 100 && drift < 1 && window.gpuErrors.length === 0;
  return { ok, msg: `scooped ${scooped.toFixed(1)} g, after the flip ${after.toFixed(2)} g on the spoon, ${peak} particles at the peak, ${left} still loose 4 s later, mass drift ${drift.toFixed(3)} g` };
}

/** Plunging straight down, again and again, eating each time, keeps the mass (CPU test of the same name). */
export async function plunge() {
  const gs = new GpuSim(device());
  const B = BOWLS[0];
  const level = new M.Quaternion();
  await gs.step(1 / 60);
  await gs.sync();
  const m0 = gs.totalGrams();
  let drift = 0;
  const watch = () => (drift = Math.max(drift, Math.abs(gs.totalGrams() - m0)));
  let eaten = 0;
  const t0 = performance.now();
  for (let k = 0; k < 3; k++) {
    await drive(gs, new M.Vector3(B.cx, 0.14, 0), level, 60, watch);
    await drive(gs, new M.Vector3(B.cx, 0.0, 0), level, 240, watch);
    await drive(gs, new M.Vector3(B.cx, 0.15, 0), level, 150, watch);
    eaten += await gs.eat();
    watch();
  }
  const ms = (performance.now() - t0) / 1350;
  const ok = drift < 1.5 && eaten > 20 && window.gpuErrors.length === 0;
  return { ok, msg: `3 plunges: eaten ${eaten.toFixed(1)} g, mass ${m0.toFixed(1)} g, worst drift ${drift.toFixed(3)} g, ${ms.toFixed(2)} ms/frame` };
}

/** The same run twice gives the same numbers (atomics only add fixed-point, lists come from scans). */
export async function determinism() {
  const run = async () => {
    const gs = new GpuSim(device());
    const B = BOWLS[1];
    const dig = new M.Quaternion().setFromEuler(new M.Euler(0, 0, 0.5));
    await drive(gs, new M.Vector3(B.cx - 0.03, 0.14, 0), dig, 60);
    await drive(gs, new M.Vector3(B.cx - 0.02, 0.085, 0), dig, 120);
    await drive(gs, new M.Vector3(B.cx + 0.025, 0.088, 0), new M.Quaternion(), 120);
    await gs.sync();
    const L = gs.L;
    const d = await gs.debugRead('W', L.W.D + L.grids[1].off, L.grids[1].count);
    const c = await gs.debugRead('W', L.W.D + L.grids[3].off, L.grids[3].count);
    const pos = await gs.debugRead('W', L.W.POS, 3 * gs.stats.n);
    return { d, c, pos, n: gs.stats.n, st: gs.getState() };
  };
  const a = await run();
  const b = await run();
  const same = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
  const ok = a.n === b.n && same(a.d, b.d) && same(a.c, b.c) && same(a.pos, b.pos) && a.st.scoopedGrams === b.st.scoopedGrams;
  return { ok, msg: `two runs of 300 frames: bowl grid ${same(a.d, b.d) ? 'identical' : 'DIFFERENT'}, scoop ${same(a.c, b.c) ? 'identical' : 'DIFFERENT'}, ${a.n} vs ${b.n} particles ${same(a.pos, b.pos) ? 'at identical positions' : 'at DIFFERENT positions'}` };
}

/** Many loose particles dropped onto the ice: frame time. */
export async function stress() {
  const out = [];
  for (const N of [4000, 15000]) {
    const gs = new GpuSim(device());
    gs.params.fluid.maxParticles = N;
    gs.params.fluid.stopSpeed = 0; // keep them loose for the measurement
    const f = gs.cpu.fluid;
    f.configure(gs.params.fluid.particleRadius, gs.params.fluid.kernelScale);
    const s = f.d * 1.05;
    const side = Math.ceil(Math.sqrt(N / 12));
    const pos = new Float32Array(3 * N);
    for (let k = 0; k < N; k++) {
      const i = k % side, j = Math.floor(k / side) % side, l = Math.floor(k / side / side);
      pos.set([-0.18 + (i - side / 2) * s, 0.13 + l * s, (j - side / 2) * s], 3 * k);
    }
    const L = gs.L;
    gs.debugWrite('W', L.W.POS, pos);
    gs.debugWrite('W', L.W.FPOS, pos);
    gs.debugWrite('W', L.W.VEL, new Float32Array(3 * N));
    gs.debugWrite('W', L.W.SLOW, new Float32Array(N).fill(1));
    gs.debugWrite('X', L.X.AGE, new Uint32Array(N).fill(10));
    gs.debugWrite('X', L.X.RGBA, new Uint32Array(N).fill(0x00c8e6f0));
    gs.debugWrite('X', L.X.ALIVE, new Uint32Array(N).fill(1));
    gs.debugWrite('A', L.A.CNT + CNT.n, new Uint32Array([N]));
    const t = [];
    for (let k = 0; k < 60; k++) {
      const t0 = performance.now();
      await gs.step(1 / 60);
      t.push(performance.now() - t0);
    }
    await gs.sync();
    t.sort((a, b) => a - b);
    out.push(`${N}: ${t[30].toFixed(2)} ms/frame (${gs.stats.n} left)`);
  }
  return { ok: window.gpuErrors.length === 0, msg: `loose particles on the GPU simulation: ${out.join(', ')}` };
}

// run by default: correctness against the CPU, plus the stress timing
export const ALL = { scans, volumes, smoothing, interact, frames, release, plunge, determinism, stress };
// diagnostics, run by name: node tools/gpu-sim-check.mjs profile timing
export const DIAG = { profile, timing, overhead, bisect };
