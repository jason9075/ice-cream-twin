// The whole ice-cream simulation resident on the GPU, the way Claybook runs: SDF volumes, particles
// and the volume ledger live in GPU buffers, and every frame is one submission of compute passes:
//
//   carve + copy into the scoop (per spoon sub-step) -> carry -> tip-out -> PBF sub-steps -> freeze
//   (stamp into the SDFs) -> ledger (debts -> stamps) -> smoothing / redistancing -> stats -> render
//
// The CPU keeps what is small and sequential: the spoon controller (no-penetration against the analytic
// bowls), the arm, and deciding which passes to run. It reads back one small block of statistics per
// frame, a frame late, which is also when the CPU version reads them (contentVolume is updated at the
// end of a frame). Mirrors src/sim/sim.js, the reference implementation; tools/gpu-sim-check.mjs
// compares the two.
import * as M from '../../math.js';
import { Sim } from '../../sim/sim.js';
import { SPOON, BOWLS } from '../../sim/sdf.js';
import { CAP, MAX_SPILLED, PASS_BYTES, ST, STU, CNT, makeLayout, header } from './layout.js';
import { gridWGSL } from './k-grid.js';
import { fluidWGSL } from './k-fluid.js';
import { worldWGSL } from './k-world.js';
import { freezeWGSL } from './k-freeze.js';
import { ledgerWGSL } from './k-ledger.js';

const DISPLACED_PARTICLES_PER_FRAME = 40;
const MAX_SLOTS = 2048;
const RD_ITERS = 12;
const PARTICLE_WG = 64;

const renderWGSL = /* wgsl */ `
@group(0) @binding(7) var<storage, read_write> INST: array<f32>;
@group(0) @binding(8) var<storage, read_write> SINST: array<f32>;
// renderer instances (x, y, z, radius, r, g, b, _) + indirect draw arguments.
// P.i0 = (sphere index count, 0, 0, 0), P.f0.x = radius
@compute @workgroup_size(64)
fn particleInstances(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let n = nParticles();
  if (i == 0u) {
    IND[8] = u32(P.i0.x); IND[9] = n; IND[10] = 0u; IND[11] = 0u; IND[12] = 0u;
    let s = atomicLoad(&A[A_CNT + C_spilled]);
    IND[16] = u32(P.i0.x); IND[17] = s; IND[18] = 0u; IND[19] = 0u; IND[20] = 0u;
  }
  if (i < n) {
    let p = ld3(W_POS, i);
    let c = unpackRGBA(X[X_RGBA + i]) / 255.0;
    INST[i * 8u] = p.x; INST[i * 8u + 1u] = p.y; INST[i * 8u + 2u] = p.z; INST[i * 8u + 3u] = P.f0.x;
    INST[i * 8u + 4u] = c.x; INST[i * 8u + 5u] = c.y; INST[i * 8u + 6u] = c.z;
  }
  if (i < atomicLoad(&A[A_CNT + C_spilled])) {
    let p = ld3(W_SPOS, i);
    let c = unpackRGBA(X[X_SRGBA + i]) / 255.0;
    SINST[i * 8u] = p.x; SINST[i * 8u + 1u] = p.y; SINST[i * 8u + 2u] = p.z; SINST[i * 8u + 3u] = P.f0.x;
    SINST[i * 8u + 4u] = c.x; SINST[i * 8u + 5u] = c.y; SINST[i * 8u + 6u] = c.z;
  }
}
`;

/** Rows (local -> world) of a quaternion's rotation matrix, as Sim._setPose builds R. */
function poseRows(q, p) {
  const { x, y, z, w } = q;
  return {
    R0: [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
    R1: [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
    R2: [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
    T: [p.x, p.y, p.z, 0],
  };
}

/** One frame's command list: dispatches with their per-dispatch parameters, and copies. */
class Frame {
  constructor(gs) {
    this.gs = gs;
    this.slots = new ArrayBuffer(MAX_SLOTS * PASS_BYTES);
    this.f = new Float32Array(this.slots);
    this.i = new Int32Array(this.slots);
    this.n = 0;
    this.ops = [];
  }
  _slot(p = {}) {
    if (this.n >= MAX_SLOTS) throw new Error('GpuSim: too many dispatches in one frame');
    const s = this.n++;
    const o = (s * PASS_BYTES) / 4;
    const put = (arr, at, v) => v && arr.set(v, o + at);
    ['i0', 'i1', 'i2', 'i3'].forEach((k, j) => put(this.i, j * 4, p[k]));
    ['f0', 'f1', 'f2', 'f3', 'R0', 'R1', 'R2', 'T', 'q0', 'q1', 'x0', 'x1'].forEach((k, j) => put(this.f, 16 + j * 4, p[k]));
    return s;
  }
  run(name, groups, p) {
    if (groups > 0) this.ops.push({ kind: 'd', name, groups, slot: this._slot(p) });
  }
  /**
   * A per-particle kernel. Dispatched for the whole capacity rather than indirectly from the GPU's
   * particle count: Chrome validates every dispatchWorkgroupsIndirect (~0.5 ms each), while the
   * threads past the count return at once.
   */
  particles(name, p) {
    this.run(name, Math.ceil(CAP / PARTICLE_WG), p);
  }
  copy(fn) {
    this.ops.push({ kind: 'c', fn });
  }

  /** With `profile` = {querySet, names: []}, every dispatch gets its own timed pass. */
  encode(profile = null) {
    const gs = this.gs;
    const d = gs.device;
    if (this.n) d.queue.writeBuffer(gs.passBuf, 0, this.slots, 0, this.n * PASS_BYTES);
    const enc = d.createCommandEncoder({ label: 'gpu-sim frame' });
    let pass = null;
    for (const op of this.ops) {
      if (op.kind === 'c') {
        if (pass) pass.end();
        pass = null;
        op.fn(enc);
        continue;
      }
      if (profile && profile.whole) {
        // whole-frame timing: one timestamp at the start of the first pass, one at the end of the last
        if (!pass) {
          const first = !profile.started;
          profile.started = true;
          pass = enc.beginComputePass({ timestampWrites: first ? { querySet: profile.querySet, beginningOfPassWriteIndex: 0 } : { querySet: profile.querySet, endOfPassWriteIndex: 1 } });
        }
      } else if (profile) {
        if (pass) pass.end();
        const q = profile.names.length * 2;
        profile.names.push(op.name);
        pass = enc.beginComputePass({ timestampWrites: { querySet: profile.querySet, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
      }
      if (!pass) pass = enc.beginComputePass({ label: 'gpu-sim' });
      pass.setPipeline(gs.pipe[op.name]);
      pass.setBindGroup(0, gs.bind, [op.slot * PASS_BYTES]);
      pass.dispatchWorkgroups(op.groups);
    }
    if (pass) pass.end();
    return enc;
  }
}

export { Frame, poseRows };

export class GpuSim {
  constructor(device, params) {
    this.device = device;
    // the CPU reference simulation supplies the spoon controller, the initial ice and the grid layout
    this.cpu = new Sim(params);
    this.params = this.cpu.params;
    this.spoon = this.cpu.spoon;
    this.content = this.cpu.content;
    this.grids = this.cpu.grids;
    this.flavorRGBA = this.cpu.flavorRGBA;
    this.L = makeLayout(this.cpu);
    this.code = [header(this.L), gridWGSL, fluidWGSL, worldWGSL, freezeWGSL, ledgerWGSL, renderWGSL].join('\n');
    this._createGPU();
    this.stats = this._emptyStats();
    this.views = null;
    this.reset();
  }

  // ------------------------------------------------------------------ GPU objects
  _createGPU() {
    const d = this.device;
    const L = this.L;
    const S = GPUBufferUsage.STORAGE;
    const buf = (size, usage, label) => d.createBuffer({ size: Math.ceil(size / 4) * 4, usage, label });
    this.globals = buf(16 * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'gpu-sim globals');
    this.passBuf = buf(MAX_SLOTS * PASS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'gpu-sim passes');
    this.W = buf(L.W.size * 4, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'gpu-sim W');
    this.X = buf(L.X.size * 4, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'gpu-sim X');
    this.A = buf(L.A.size * 4, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'gpu-sim A');
    // indirect draw arguments for the particle instances (written by particleInstances)
    this.ind = buf(64 * 4, S | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, 'gpu-sim indirect');
    // render textures: per grid, distance rows (256 B) then colour rows (256 B)
    this.packOff = [];
    let pk = 0;
    for (const g of L.grids) {
      this.packOff.push(pk);
      pk += 2 * g.n[1] * g.n[2] * 64;
    }
    this.pack = buf(pk * 4, S | GPUBufferUsage.COPY_SRC, 'gpu-sim pack');
    this.dummy = [buf(CAP * 32, S, 'gpu-sim dummy instances'), buf(MAX_SPILLED * 32, S, 'gpu-sim dummy spilled')];
    this.readback = [0, 1].map((k) => buf(1536, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, `gpu-sim readback ${k}`));
    this.rbIndex = 0;
    const module = d.createShaderModule({ code: this.code, label: 'gpu-sim' });
    this.module = module;
    const e = (binding, type, dyn = false) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, hasDynamicOffset: dyn } });
    this.layout = d.createBindGroupLayout({
      entries: [e(0, 'uniform'), { ...e(1, 'uniform', true), buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PASS_BYTES } },
        ...[2, 3, 4, 5, 6, 7, 8].map((b) => e(b, 'storage'))],
    });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const names = [...this.code.matchAll(/@compute[^\n]*\n\s*fn (\w+)/g)].map((m) => m[1]);
    this.pipe = {};
    for (const name of names) this.pipe[name] = d.createComputePipeline({ layout: pl, compute: { module, entryPoint: name }, label: name });
    this._bind();
  }

  _bind(inst = this.dummy[0], sinst = this.dummy[1]) {
    this.bind = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.globals } },
        { binding: 1, resource: { buffer: this.passBuf, size: PASS_BYTES } },
        ...[this.W, this.X, this.A, this.ind, this.pack, inst, sinst].map((buffer, k) => ({ binding: k + 2, resource: { buffer } })),
      ],
    });
  }

  /**
   * Render straight from the simulation's buffers: particle and spilled-crumb instances (from
   * renderer.createInstances) and the four SDF volumes (renderer.createVolume(..., {external: true})).
   */
  attachViews({ particles, spilled, volumes }) {
    this.views = { particles, spilled, volumes };
    particles.indirect = { buffer: this.ind, offset: 8 * 4 };
    spilled.indirect = { buffer: this.ind, offset: 16 * 4 };
    this._bind(particles.sbuf, spilled.sbuf);
  }

  // ------------------------------------------------------------------ state
  _emptyStats() {
    return { contentVol: 0, volume: [0, 0, 0, 0], removed: 0, n: 0, spilled: 0, spilledVol: 0, eatenVol: 0, debtVol: 0, settleFrames: [0, 0, 0, 0] };
  }

  /** Upload a grid's distances and colours from the CPU Sim (initial ice, refill). */
  _uploadGrid(k) {
    const q = this.device.queue;
    const g = k < 3 ? this.grids[k] : this.content;
    const off = this.L.grids[k].off;
    q.writeBuffer(this.W, (this.L.W.D + off) * 4, g.data);
    q.writeBuffer(this.X, (this.L.X.COL + off) * 4, new Uint32Array(g.color.buffer, g.color.byteOffset, g.n));
    q.writeBuffer(this.X, (this.L.X.FLAG + off) * 4, new Uint32Array(g.n));
  }

  reset() {
    const cpu = this.cpu;
    cpu.reset();
    cpu._clampContent(); // builds the content floor (the spoon is empty, nothing is clamped)
    const q = this.device.queue;
    const L = this.L;
    for (let k = 0; k < 4; k++) this._uploadGrid(k);
    q.writeBuffer(this.W, (L.W.FLOOR) * 4, cpu._contentFloor);
    q.writeBuffer(this.W, L.W.ST * 4, new Float32Array(256));
    const stu = new Uint32Array(256);
    stu[STU.rng] = 12345;
    q.writeBuffer(this.X, L.X.STU * 4, stu);
    const a = new Uint32Array(L.A.size);
    a[L.A.CNT + CNT.firstIdx] = 0xffffffff;
    q.writeBuffer(this.A, 0, a);
    this.time = 0;
    this.frame = 0;
    this.speedScale = 1;
    this.cutForce = 0;
    this.eatenVolume = 0;
    this.releasing = false;
    this._releaseQueued = false;
    this._redistPending = false;
    this._lastTravel = 0;
    this._lastCarvePos = this.spoon.pos.clone();
    this._lastCarveQuat = this.spoon.quat.clone();
    this.stats = this._emptyStats();
    this.stats.volume = this.grids.map((g) => g.volume()).concat(0);
    this._pending = null;
  }

  refill() {
    this.cpu.refill();
    for (let k = 0; k < 3; k++) this._uploadGrid(k);
  }

  /** Tip everything on the spoon off as particles (next frame). */
  releaseContent() {
    this._releaseQueued = true;
  }

  tiltDeg(quat = this.spoon.quat) {
    return this.cpu.tiltDeg(quat);
  }

  // ------------------------------------------------------------------ helpers
  _globals(dtFrame, sub, pose, tilt) {
    const P = this.params;
    const fp = P.fluid;
    const ip = P.ice;
    const f = this.cpu.fluid;
    f.configure(fp.particleRadius, fp.kernelScale);
    const dt = dtFrame / sub;
    const r = f.r;
    const sp = this.spoon;
    const g = new Float32Array(64);
    const u = new Uint32Array(g.buffer);
    g.set([dt, fp.gravity, Math.max(0, 1 - fp.damping * dt), 1.2], 0);
    g.set([r, f.h, f.h2, f.rho0], 4);
    g.set([f.eps, f.d, 0.5 * r, fp.cohesion], 8);
    g.set([fp.friction, 1.5 * fp.friction * fp.gravity * dt, fp.viscosity, 1 / f.h], 12);
    g.set([fp.stopSpeed, fp.restFrames, fp.stampScale, fp.stampSmooth], 16);
    g.set([ip.smoothing, ip.accretionDist, ip.capacity * 1e-6, dtFrame], 20);
    g.set([8 * r * r * r, fp.stampOnContact ? 1 : 0, Math.min(fp.maxParticles, CAP), ip.displacedToParticles ? 1 : 0], 24);
    g.set([tilt < ip.releaseAngle - 10 ? 1 : 0, ip.accretion ? 1 : 0, ip.releaseAngle, tilt], 28);
    g.set(pose.R0, 32);
    g.set(pose.R1, 36);
    g.set(pose.R2, 40);
    g.set(pose.T, 44);
    g.set([sp.linVel.x, sp.linVel.y, sp.linVel.z, 0, sp.angVel.x, sp.angVel.y, sp.angVel.z, 0], 48);
    u.set([this.frame, DISPLACED_PARTICLES_PER_FRAME, 48, 256], 56);
    this.device.queue.writeBuffer(this.globals, 0, g);
  }

  _localBoxToWorld(pose, min, max) {
    const out = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let m = 0; m < 8; m++) {
      const l = [m & 1 ? max[0] : min[0], m & 2 ? max[1] : min[1], m & 4 ? max[2] : min[2]];
      for (let a = 0; a < 3; a++) {
        const R = [pose.R0, pose.R1, pose.R2][a];
        const v = R[0] * l[0] + R[1] * l[1] + R[2] * l[2] + pose.T[a];
        out[a] = Math.min(out[a], v);
        out[a + 3] = Math.max(out[a + 3], v);
      }
    }
    return out;
  }

  _contentBox(pose) {
    const c = this.content;
    return this._localBoxToWorld(pose, [c.ox, c.oy, c.oz], [c.ox + c.sx, c.oy + c.sy, c.oz + c.sz]);
  }

  _scan(fr, kind, src, dst, n) {
    const blocks = Math.ceil(n / 512);
    if (!blocks) return;
    const p = { i0: [src, dst, n, kind === 'u' ? 1 : 0] };
    fr.run(kind === 'u' ? 'scanBlockU' : 'scanBlockF', blocks, p);
    fr.run('scanSums', 1, p);
    fr.run('scanAdd', blocks, p);
  }

  _gridOcc(fr, k, stSlot, mode = 0) {
    const g = this.L.grids[k];
    const wgs = Math.ceil(g.count / 1024);
    fr.run('gridOcc', wgs, { i0: [k, 0, 0, 0] });
    fr.run('finishSum', 1, { i0: [0, wgs, stSlot, mode], f0: [g.c ** 3, 0, 0, 0] });
  }

  /** Volume-preserving smoothing of one grid (mode 0 touched, 1 settle, 2 settle without consuming). */
  _smooth(fr, k, mode, strength) {
    const total = this.L.grids[k].count;
    const p = { i0: [mode, k, 0, 0], f0: [strength, 0, 0, 0] };
    fr.run('smFlag', Math.ceil(total / 256), p);
    if (mode === 0) fr.run('smBox', Math.ceil(total / 256), p);
    this._scan(fr, 'u', this.L.X.SCANU, this.L.X.PRE, total);
    fr.run('smList', Math.ceil(total / 256), p);
    fr.run('smWrite', 1, p);
  }

  _compact(fr) {
    const L = this.L;
    const groups = Math.ceil(CAP / PARTICLE_WG);
    fr.run('compactMark', groups);
    this._scan(fr, 'u', L.X.SCANU, L.X.IDX, CAP);
    fr.run('compactScatter', groups);
    fr.run('compactCount', 1);
    fr.particles('compactCopy');
  }

  /** Sim.interact on the CPU side: decide, then record the GPU passes for this spoon pose. */
  _interact(fr, pos, quat, force = false) {
    const moved = this._lastCarvePos.distanceTo(pos) + this._lastCarveQuat.angleTo(quat) * 0.06;
    if (!force && moved < 0.0009) return;
    if (moved < 1e-7) return;
    this._lastCarvePos.copy(pos);
    this._lastCarveQuat.copy(quat);
    const pose = poseRows(quat, pos);
    const tilted = this.tiltDeg(quat) > this.params.ice.releaseAngle;
    const hb = this._contentBox(pose);
    const H = SPOON.handle;
    const hbHandle = this._localBoxToWorld(
      pose,
      [H[0][0] - 0.004, Math.min(...H.map((p) => p[1])) - 0.004, -0.004],
      [H[H.length - 1][0] + 0.004, Math.max(...H.map((p) => p[1])) + 0.004, 0.004],
    );
    const hits = (b, g) => !(b[3] < g.ox || b[0] > g.ox + g.sx || b[4] < g.oy || b[1] > g.oy + g.sy || b[5] < g.oz || b[2] > g.oz + g.sz);
    const touching = this.grids.map((g, k) => k).filter((k) => hits(hb, this.grids[k]) || hits(hbHandle, this.grids[k]));
    if (!touching.length) return;
    const L = this.L;
    const base = { R0: pose.R0, R1: pose.R1, R2: pose.R2, T: pose.T };
    // A. copy ice into the scoop
    if (!tilted) {
      const wgs = Math.ceil(L.NC / 256);
      fr.run('copyMark', wgs, { ...base, i0: [0, 0, 0, 0] });
      fr.run('finishSum', 1, { i0: [0, wgs, ST.gained, 0], f0: [this.content.cell ** 3, 0, 0, 0] });
      fr.run('copyFill', Math.ceil(L.NC / 64));
      fr.run('copyApply', Math.ceil(L.NC / 64));
      this._contentEdited = true;
    } else {
      fr.run('finishSum', 1, { i0: [0, 0, ST.gained, 0], f0: [0, 0, 0, 0] });
    }
    fr.run('copyDone', 1);
    // B. carve the bowls, head box then handle box, grid by grid
    for (const k of touching) {
      const g = this.grids[k];
      for (const [box, handleOnly] of [[hb, false], [hbHandle, true]]) {
        if (!hits(box, g)) continue;
        const c = g.cell;
        const i0 = Math.max(0, Math.floor((box[0] - g.ox) / c));
        const i1 = Math.min(g.nx - 1, Math.floor((box[3] - g.ox) / c));
        const j0 = Math.max(0, Math.floor((box[1] - g.oy) / c));
        const j1 = Math.min(g.ny - 1, Math.floor((box[4] - g.oy) / c));
        const k0 = Math.max(0, Math.floor((box[2] - g.oz) / c));
        const k1 = Math.min(g.nz - 1, Math.floor((box[5] - g.oz) / c));
        const s = [i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1];
        if (s.some((v) => v <= 0)) continue;
        const rn = s[0] * s[1] * s[2];
        const p = { ...base, i0: [k, i0, j0, k0], i1: [...s, (handleOnly ? 1 : 0) | (tilted ? 2 : 0)], i2: [0, 0, 0, 0] };
        fr.run('carveMark', Math.ceil(rn / 256), p);
        this._scan(fr, 'f', L.W.DVD, L.W.PD, rn);
        this._scan(fr, 'f', L.W.DVC, L.W.PC, rn);
        fr.run('carveSpawn', Math.ceil(rn / 64), p);
        fr.run('carveDone', 1, p);
      }
    }
    fr.run('interactDone', 1, { i0: [touching[0], 0, 0, 0] });
  }

  _release(fr) {
    const f = this.cpu.fluid;
    const c = this.content;
    const step = f.d;
    const n = (s) => (step / 2 < s ? Math.floor((s - step / 2) / step) + 1 : 0);
    const dims = [n(c.sx), n(c.sy), n(c.sz)];
    const rn = dims[0] * dims[1] * dims[2];
    const p = { i0: [...dims, 0], f0: [step, 0, 0, 0] };
    fr.run('releaseMark', Math.ceil(rn / 64), p);
    this._scan(fr, 'u', this.L.X.SCANU, this.L.X.LIST, rn);
    fr.run('releaseSpawn', Math.ceil(rn / 64), p);
    fr.run('releaseDone', 1, p);
    fr.run('clearContent', Math.ceil(this.L.NC / 64), { i0: [1, 0, 0, 0] });
  }

  // ------------------------------------------------------------------ main step
  /** One frame (async: waits for the previous frame's statistics, then submits this one). */
  stepAsync(dt) {
    return this.step(dt);
  }

  async step(dt) {
    const tw = performance.now();
    if (this._pending) await this._pending;
    this.stats.waitMs = performance.now() - tw;
    const tc = performance.now();
    dt = Math.min(dt, 1 / 30);
    const P = this.params;
    const sp = this.spoon;
    const startPos = sp.pos.clone();
    const startQuat = sp.quat.clone();
    const fr = new Frame(this);
    const L = this.L;
    this._contentEdited = false;

    fr.run('frameStart', 1);
    // 1. spoon motion, carving the ice along the swept path
    sp.p = P.spoon;
    sp.update(dt, this.speedScale, (pp, pq, np, nq) => this._interact(fr, np, nq));
    this._interact(fr, sp.pos, sp.quat, true);
    this._compact(fr);

    // cutting force from the volume the last frame's carving removed (read back a frame late)
    const travel = sp.pos.distanceTo(startPos) + startQuat.angleTo(sp.quat) * 0.03;
    const area = Math.min(this.stats.removed / Math.max(this._lastTravel, 0.0005), Math.PI * SPOON.a * SPOON.c);
    this._lastTravel = travel;
    const force = P.ice.hardness * 1000 * area;
    this.cutForce += (force - this.cutForce) * 0.3;
    const target = this.cutForce > P.spoon.maxCutForce ? P.spoon.maxCutForce / this.cutForce : 1;
    this.speedScale += (Math.max(0.05, target) - this.speedScale) * 0.3;

    // 2. carry loose ice in / on the spoon
    const pose = poseRows(sp.quat, sp.pos);
    const tilt = this.tiltDeg();
    const tilted = tilt > P.ice.releaseAngle;
    const sub = Math.max(1, P.fluid.substeps | 0);
    this._globals(dt, sub, pose, tilt);
    if (!tilted && !(startPos.distanceTo(sp.pos) < 1e-9 && startQuat.angleTo(sp.quat) < 1e-9)) {
      const qi = startQuat.clone().invert();
      const qd = sp.quat.clone().multiply(qi);
      fr.particles('carry', { q0: qi.toArray(), q1: qd.toArray(), x0: [...startPos.toArray(), 0], x1: [...sp.pos.toArray(), 0] });
    }
    // 3. tilted past the release angle: the scoop slides off as particles
    let release = this._releaseQueued;
    this._releaseQueued = false;
    if (tilted && !this.releasing) {
      release = true;
      this.releasing = true;
    } else if (tilt < P.ice.releaseAngle - 10) {
      this.releasing = false;
    }
    if (release) this._release(fr);

    // 4. particles
    fr.particles('markFrame');
    const tableGroups = Math.ceil((1 << 16) / PARTICLE_WG);
    for (let s = 0; s < sub; s++) {
      fr.particles('predict');
      fr.run('clearCells', tableGroups + 1);
      fr.particles('countCells');
      fr.run('scanCells', 1);
      fr.run('clearCells', tableGroups + 1);
      fr.particles('scatter');
      fr.run('sortCells', tableGroups);
      fr.particles('neighbors');
      for (let it = 0; it < P.fluid.iterations; it++) {
        fr.particles('lambdas');
        fr.particles('deltas');
        fr.particles('applyCollide');
      }
      fr.particles('velocities');
      fr.particles('xsph');
    }
    // 5. freeze resting particles into the SDFs
    const capGroups = Math.ceil(CAP / PARTICLE_WG);
    fr.run('freezeMark', capGroups);
    this._scan(fr, 'u', L.X.SCANU, L.X.IDX, CAP);
    fr.run('freezeList', capGroups);
    fr.run('freezeStamp', 1);
    this._compact(fr);
    // 6. ledger: owed ice back onto the heap
    const cup = this._contentBox(pose);
    const box = { x0: [cup[0], cup[1], cup[2], 0], x1: [cup[3], cup[4], cup[5], 0] };
    fr.run('ledgerBins', 1);
    fr.run('ledgerPay', 1, box);
    // 7. surface tension: volume-preserving smoothing of what was edited, then settling
    const sm = P.ice.smoothing;
    for (let k = 0; k < 3; k++) {
      this._smooth(fr, k, 0, sm);
      this._smooth(fr, k, 1, sm);
    }
    this._smooth(fr, 3, 0, sm);
    if (this._contentEdited || this.stats.settleFrames[3] > 0) this._redistPending = true;
    if (this._redistPending && this.frame % 3 === 0 && this.stats.contentVol > 0) {
      const c3 = this.content.cell ** 3;
      const wgs = Math.ceil(L.NC / 1024);
      fr.run('gridOcc', wgs, { i0: [3, 0, 0, 0] });
      fr.run('finishSum', 1, { i0: [0, wgs, ST.residual + 3, 1], f0: [c3, 0, 0, 0] });
      const g64 = Math.ceil(L.NC / 64);
      fr.run('rdInit', g64);
      for (let it = 0; it < RD_ITERS; it++) fr.run('rdIter', g64, { i0: [it & 1, 0, 0, 0] });
      // RD_ITERS is even: the result is back in RU0
      fr.run('rdFinish', g64);
      fr.run('gridOcc', wgs, { i0: [3, 0, 0, 0] });
      fr.run('finishSum', 1, { i0: [0, wgs, ST.residual + 3, 2], f0: [c3, 0, 0, 0] });
      this._redistPending = this.stats.settleFrames[3] > 0;
    }
    for (let k = 0; k < 3; k++) this._smooth(fr, 3, k === 0 ? 1 : 2, sm);
    const cw = Math.ceil(L.NC / 256);
    fr.run('clampContent', cw, { i0: [0, 0, 0, 0] });
    fr.run('finishSum', 1, { i0: [0, cw, ST.residual + 3, 1], f0: [this.content.cell ** 3, 0, 0, 0] });
    fr.run('ledgerResiduals', 1);
    // 8. statistics
    for (let k = 0; k < 4; k++) this._gridOcc(fr, k, ST.volume + k);
    this._gridOcc(fr, 3, ST.contentVol);
    fr.run('ledgerStats', 1);
    // 9. render: SDF textures and particle instances straight from the buffers
    this._encodeRender(fr);
    // statistics -> read back
    const rb = this.readback[this.rbIndex];
    this.rbIndex ^= 1;
    fr.copy((enc) => {
      enc.copyBufferToBuffer(this.W, L.W.ST * 4, rb, 0, 1024);
      enc.copyBufferToBuffer(this.X, L.X.STU * 4, rb, 1024, 256);
      enc.copyBufferToBuffer(this.A, (L.A.CNT) * 4, rb, 1280, 256);
    });
    this.stats.ops = fr.ops.length;
    this._lastFrame = fr;
    const t0 = performance.now();
    const enc = fr.encode(this.profile);
    const afterSubmit = this.profile ? this._resolveProfile(enc) : null;
    this.device.queue.submit([enc.finish()]);
    afterSubmit?.();
    this.stats.cpuMs = performance.now() - tc;
    this._pending = rb.mapAsync(GPUMapMode.READ).then(() => {
      this._readStats(rb.getMappedRange());
      rb.unmap();
      this.stats.gpuMs = performance.now() - t0;
    });
    this.time += dt;
    this.frame++;
  }

  _encodeRender(fr) {
    const v = this.views;
    if (!v) return;
    const f = this.cpu.fluid;
    const rad = (f.r > 0 ? f.r : this.params.fluid.particleRadius) * 1.15;
    fr.run('particleInstances', Math.ceil(Math.max(CAP, MAX_SPILLED) / 64), { i0: [v.particles.mesh.count, 0, 0, 0], f0: [rad, 0, 0, 0] });
    this.L.grids.forEach((g, k) => {
      fr.run('packGrid', Math.ceil((g.n[1] * g.n[2] * 64) / 64), { i0: [k, this.packOff[k], 0, 0] });
    });
    fr.copy((enc) => {
      this.L.grids.forEach((g, k) => {
        const vol = v.volumes[k];
        const rows = g.n[1] * g.n[2];
        const size = g.n;
        enc.copyBufferToTexture({ buffer: this.pack, offset: this.packOff[k] * 4, bytesPerRow: 256, rowsPerImage: g.n[1] }, { texture: vol.sdfTex }, size);
        enc.copyBufferToTexture({ buffer: this.pack, offset: (this.packOff[k] + rows * 64) * 4, bytesPerRow: 256, rowsPerImage: g.n[1] }, { texture: vol.colTex }, size);
      });
    });
  }

  _readStats(buf) {
    const f = new Float32Array(buf, 0, 256);
    const u = new Uint32Array(buf, 1024, 64);
    const c = new Uint32Array(buf, 1280, 64);
    const s = this.stats;
    s.contentVol = f[ST.contentVol];
    s.volume = [0, 1, 2, 3].map((k) => f[ST.volume + k]);
    s.removed = f[ST.removed];
    s.spilledVol = f[ST.spilledVol];
    s.eatenVolGpu = f[ST.eatenVol];
    s.debtVol = f[ST.debtVol];
    s.looseOwed = f[ST.looseOwed];
    s.n = c[CNT.n];
    s.spilled = c[CNT.spilled];
    s.nDebts = u[STU.nDebts];
    s.settleFrames = [0, 1, 2, 3].map((k) => u[STU.settleFrames + k]);
  }

  // ------------------------------------------------------------------ debugging / tests
  /** Time every dispatch of the next frame (needs a device with 'timestamp-query'). */
  profileNextFrame(whole = false) {
    const d = this.device;
    const querySet = d.createQuerySet({ type: 'timestamp', count: 4096 });
    this.profile = whole ? { querySet, names: ['frame'], whole: true } : { querySet, names: [] };
  }
  _resolveProfile(enc) {
    const d = this.device;
    const p = this.profile;
    const n = p.names.length * 2;
    const res = d.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = d.createBuffer({ size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.resolveQuerySet(p.querySet, 0, n, res, 0);
    enc.copyBufferToBuffer(res, 0, rd, 0, n * 8);
    this.profile = null;
    return () => (this.profileResult = rd.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(rd.getMappedRange());
      const by = {};
      p.names.forEach((name, k) => {
        const ms = Number(t[2 * k + 1] - t[2 * k]) / 1e6;
        by[name] = by[name] || { ms: 0, n: 0 };
        by[name].ms += ms;
        by[name].n++;
      });
      return by;
    }));
  }
  /** Record passes with build(frame, this), submit, and wait for the GPU. */
  async debugRun(build) {
    await this.sync();
    const fr = new Frame(this);
    build(fr, this);
    this.device.queue.submit([fr.encode().finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }
  /** Read `count` elements at `offset` of buffer 'W' (Float32Array), 'X' or 'A' (Uint32Array). */
  async debugRead(name, offset, count) {
    const d = this.device;
    const staging = d.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this[name], offset * 4, staging, 0, count * 4);
    d.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = name === 'W' ? new Float32Array(staging.getMappedRange().slice(0)) : new Uint32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }
  debugWrite(name, offset, data) {
    this.device.queue.writeBuffer(this[name], offset * 4, data);
  }
  /** Set the per-frame uniforms for the spoon's current pose (debug passes that need them). */
  debugGlobals(dt = 1 / 60) {
    const sp = this.spoon;
    this._globals(dt, Math.max(1, this.params.fluid.substeps | 0), poseRows(sp.quat, sp.pos), this.tiltDeg());
  }

  /** Wait until the last submitted frame's statistics are in. */
  async sync() {
    if (this._pending) await this._pending;
  }

  /** "Eat": empty the spoon, including loose bits riding on it. Resolves to grams eaten. */
  async eat() {
    await this.sync();
    const fr = new Frame(this);
    const pose = poseRows(this.spoon.quat, this.spoon.pos);
    this._globals(1 / 60, 1, pose, this.tiltDeg());
    const before = this.stats.eatenVolGpu || 0;
    const c3 = this.content.cell ** 3;
    const wgs = Math.ceil(this.L.NC / 1024);
    fr.run('gridOcc', wgs, { i0: [3, 0, 0, 0] });
    fr.run('finishSum', 1, { i0: [0, wgs, ST.eatenVol, 1], f0: [c3, 0, 0, 0] });
    fr.run('clearContent', Math.ceil(this.L.NC / 64), { i0: [0, 0, 0, 0] });
    fr.particles('eatMark');
    fr.run('eatDone', 1);
    this._compact(fr);
    this._gridOcc(fr, 3, ST.contentVol);
    this._gridOcc(fr, 3, ST.volume + 3);
    fr.run('ledgerStats', 1);
    this._encodeRender(fr);
    const rb = this.readback[this.rbIndex];
    this.rbIndex ^= 1;
    const L = this.L;
    fr.copy((enc) => {
      enc.copyBufferToBuffer(this.W, L.W.ST * 4, rb, 0, 1024);
      enc.copyBufferToBuffer(this.X, L.X.STU * 4, rb, 1024, 256);
      enc.copyBufferToBuffer(this.A, L.A.CNT * 4, rb, 1280, 256);
    });
    this.device.queue.submit([fr.encode().finish()]);
    this._pending = rb.mapAsync(GPUMapMode.READ).then(() => {
      this._readStats(rb.getMappedRange());
      rb.unmap();
    });
    await this._pending;
    const vol = this.stats.eatenVolGpu - before;
    return vol * 1e6 * this.params.ice.density;
  }

  // ------------------------------------------------------------------ readouts (Sim's interface)
  get contentVolume() {
    return this.stats.contentVol;
  }
  get bowlVolumes() {
    return this.stats.volume.slice(0, 3);
  }
  get eatenVolume() {
    return this.stats.eatenVolGpu || 0;
  }
  set eatenVolume(v) {}
  get spilledVolume() {
    return this.stats.spilledVol;
  }
  particleVolume() {
    return this.cpu.particleVolume();
  }
  debtVolume() {
    return this.stats.debtVol;
  }
  grams(vol) {
    return vol * 1e6 * this.params.ice.density;
  }
  /** Everything accounted for (g): ice in bowls and on the spoon, loose, owed, eaten, spilled. */
  totalGrams() {
    const s = this.stats;
    const vol = s.volume[0] + s.volume[1] + s.volume[2] + s.contentVol + s.n * this.particleVolume() + s.debtVol + (s.eatenVolGpu || 0) + s.spilledVol;
    return this.grams(vol);
  }

  getState() {
    const sp = this.spoon;
    const e = new M.Euler().setFromQuaternion(sp.quat, 'XYZ');
    const te = new M.Euler().setFromQuaternion(sp.targetQuat, 'XYZ');
    const s = this.stats;
    return {
      time: this.time,
      spoon: {
        position: sp.pos.toArray(),
        quaternion: sp.quat.toArray(),
        eulerXYZdeg: [e.x, e.y, e.z].map(M.MathUtils.radToDeg),
        linearVelocity: sp.linVel.toArray(),
        angularVelocity: sp.angVel.toArray(),
        tiltDeg: this.tiltDeg(),
      },
      target: {
        position: sp.targetPos.toArray(),
        quaternion: sp.targetQuat.toArray(),
        eulerXYZdeg: [te.x, te.y, te.z].map(M.MathUtils.radToDeg),
      },
      contact: sp.contacts > 0,
      contactSamples: sp.contacts,
      blocked: sp.blocked,
      speedScale: this.speedScale,
      cutForceN: this.cutForce,
      scoopedGrams: this.grams(s.contentVol),
      eatenGrams: this.grams(s.eatenVolGpu || 0),
      spilledGrams: this.grams(s.spilledVol),
      bowlGrams: Object.fromEntries(BOWLS.map((b, i) => [b.flavor.id, this.grams(s.volume[i])])),
      particles: s.n,
      simulation: 'gpu',
      gpuMs: s.gpuMs || 0,
    };
  }
}

