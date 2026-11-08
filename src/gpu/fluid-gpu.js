// The loose ice cream's Position Based Fluids solver (src/sim/fluid.js) as WebGPU compute shaders,
// the way Claybook runs its particles: predict -> hash-grid neighbours -> [density + cohesion ->
// collide with the SDFs] x iterations -> velocity + contact friction -> XSPH viscosity.
//
// The Sim keeps the particle list. Each frame run() uploads it, runs every sub-step on the GPU in
// one submission and reads it back, so spawning, carrying, freezing into the SDFs and the volume
// ledger stay on the CPU, unchanged. The collision SDFs (table, bowls, solid ice, spoon metal,
// the scoop on the spoon) are evaluated on the GPU: the analytic ones are ported to WGSL, the ice
// grids are uploaded (f32) whenever they change.
//
// Same algorithm as the CPU, in f32. Neighbour lists come out in the CPU's order (cells visited in
// the same order, each cell sorted by particle index), so a run is deterministic on a given GPU
// (except in pathologically crowded cells, see sortCells).
import { BOWLS, BOWL_SPACING, SPOON } from '../sim/sdf.js';
import { CONTACT_NONE, MAXN, TABLE } from '../sim/fluid.js';

const WG = 64;

/** Float sections of the F buffer and u32 sections of the U buffer, in units of `cap`. */
const F = { pos: 0, prev: 3, vel: 6, dp: 9, cn: 12, sv: 15, lambda: 18, size: 19 };
const U = { age: 0, contact: 1, squeezed: 2, cellOf: 3, sorted: 4, nc: 5, size: 6 };

const f32 = (v) => {
  const s = String(v);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

function shader(cap) {
  const B = BOWLS[0];
  const H = SPOON.handle;
  const o = (sec, k) => `${sec[k] * cap}u`;
  return /* wgsl */ `
const CAP: u32 = ${cap}u;
const TABLE: u32 = ${TABLE}u;
const MASK: u32 = ${TABLE - 1}u;
const MAXN: u32 = ${MAXN}u;
const POS: u32 = ${o(F, 'pos')};
const PREV: u32 = ${o(F, 'prev')};
const VEL: u32 = ${o(F, 'vel')};
const DP: u32 = ${o(F, 'dp')};
const CN: u32 = ${o(F, 'cn')};
const SV: u32 = ${o(F, 'sv')};
const LAM: u32 = ${o(F, 'lambda')};
const AGE: u32 = ${o(U, 'age')};
const CONTACT: u32 = ${o(U, 'contact')};
const SQUEEZED: u32 = ${o(U, 'squeezed')};
const CELLOF: u32 = ${o(U, 'cellOf')};
const SORTED: u32 = ${o(U, 'sorted')};
const NC: u32 = ${o(U, 'nc')};
const MAX_SEPARATION: f32 = 0.05;
const BOWL_SPACING: f32 = ${f32(BOWL_SPACING)};
const BOWL_CY: f32 = ${f32(B.cy)};
const BOWL_R: f32 = ${f32(B.r)};
const BOWL_T: f32 = ${f32(B.t)};
const BOWL_H: f32 = ${f32(B.h)};
const BOWL_W: f32 = ${f32(B.w)};
const SP_ABC: vec3f = vec3f(${f32(SPOON.a)}, ${f32(SPOON.b)}, ${f32(SPOON.c)});
const SP_T: f32 = ${f32(SPOON.t)};
const SP_HR: f32 = ${f32(SPOON.handleR)};
var<private> HANDLE: array<vec3f, ${H.length}> = array(${H.map((p) => `vec3f(${p.map(f32).join(', ')})`).join(', ')});
// central differences of the analytic SDFs: wider than the CPU's 1e-5 m, which is below f32 noise
const GEPS: f32 = 4e-5;

struct Params {
  a: vec4f,          // dt, gravity, damp, vmax
  b: vec4f,          // r, h, h2, rho0
  c: vec4f,          // eps, d0, maxStep, cohesion
  e: vec4f,          // friction, staticV, viscosity, 1/h
  n: vec4u,          // particle count
  R0: vec4f,         // spoon rotation, rows (local -> world)
  R1: vec4f,
  R2: vec4f,
  T: vec4f,          // spoon position (origin of its local frame)
  lin: vec4f,        // spoon linear velocity
  ang: vec4f,        // spoon angular velocity
  go: array<vec4f, 4>, // grid origin xyz, cell (bowl 0..2, then the spoon's content)
  gn: array<vec4u, 4>, // grid dims xyz, offset into G
  gb: vec4f,           // grid band
}

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read_write> Fb: array<f32>;
@group(0) @binding(2) var<storage, read_write> Ub: array<u32>;
@group(0) @binding(3) var<storage, read_write> NB: array<u32>;
@group(0) @binding(4) var<storage, read_write> CC: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> CS: array<u32>;
@group(0) @binding(6) var<storage, read> G: array<f32>;

fn ld(off: u32, i: u32) -> vec3f {
  let k = off + i * 3u;
  return vec3f(Fb[k], Fb[k + 1u], Fb[k + 2u]);
}
fn st(off: u32, i: u32, v: vec3f) {
  let k = off + i * 3u;
  Fb[k] = v.x;
  Fb[k + 1u] = v.y;
  Fb[k + 2u] = v.z;
}
fn W(r2: f32) -> f32 {
  let x = 1.0 - r2 / u.b.z;
  return x * x * x;
}
fn cellOf(p: vec3f) -> vec3i {
  return vec3i(floor(p * u.e.w));
}
fn hashc(c: vec3i) -> u32 {
  return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & MASK;
}

// ---------------------------------------------------------------- SDFs (src/sim/sdf.js)
fn nearestBowl(x: f32) -> u32 {
  return u32(clamp(i32(floor(x / BOWL_SPACING + 0.5)) + 1, 0, 2));
}
fn sdBowl(bi: u32, x: f32, y: f32, z: f32) -> f32 {
  let px = x - (f32(bi) - 1.0) * BOWL_SPACING;
  let py = y - BOWL_CY;
  let qx = sqrt(px * px + z * z);
  var d: f32;
  if (BOWL_H * qx < BOWL_W * py) {
    d = length(vec2f(qx - BOWL_W, py - BOWL_H));
  } else {
    d = abs(sqrt(qx * qx + py * py) - BOWL_R);
  }
  return d - BOWL_T;
}
fn sdEnv(p: vec3f) -> f32 {
  return min(p.y, sdBowl(nearestBowl(p.x), p.x, p.y, p.z));
}
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  if (k1 < 1e-12) {
    return -min(r.x, min(r.y, r.z));
  }
  return k0 * (k0 - 1.0) / k1;
}
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
fn sdSpoonSolid(p: vec3f) -> f32 {
  let shell = max(abs(sdEllipsoid(p, SP_ABC)) - SP_T / 2.0, p.y);
  var hd = 1e9;
  for (var k = 0; k < ${H.length - 1}; k++) {
    hd = min(hd, sdCapsule(p, HANDLE[k], HANDLE[k + 1], SP_HR));
  }
  return min(shell, hd);
}
fn gradSpoon(p: vec3f) -> vec3f {
  let g = vec3f(
    sdSpoonSolid(p + vec3f(GEPS, 0, 0)) - sdSpoonSolid(p - vec3f(GEPS, 0, 0)),
    sdSpoonSolid(p + vec3f(0, GEPS, 0)) - sdSpoonSolid(p - vec3f(0, GEPS, 0)),
    sdSpoonSolid(p + vec3f(0, 0, GEPS)) - sdSpoonSolid(p - vec3f(0, 0, GEPS)));
  let l = length(g);
  return select(g, g / l, l > 0.0);
}
fn gradBowl(bi: u32, p: vec3f) -> vec3f {
  let g = vec3f(
    sdBowl(bi, p.x + GEPS, p.y, p.z) - sdBowl(bi, p.x - GEPS, p.y, p.z),
    sdBowl(bi, p.x, p.y + GEPS, p.z) - sdBowl(bi, p.x, p.y - GEPS, p.z),
    sdBowl(bi, p.x, p.y, p.z + GEPS) - sdBowl(bi, p.x, p.y, p.z - GEPS));
  let l = length(g);
  return select(g, g / l, l > 0.0);
}

// ---------------------------------------------------------------- SDF grids (SdfGrid.sample/grad)
fn gsample(g: u32, p: vec3f) -> f32 {
  let o = u.go[g];
  let n = u.gn[g];
  let nf = vec3f(n.xyz);
  var f = (p - o.xyz) / o.w - 0.5;
  if (any(f < vec3f(-0.5)) || any(f > nf - 0.5)) {
    return u.gb[g];
  }
  f = max(f, vec3f(0.0));
  let ii = min(vec3u(f), n.xyz - 2u);
  let t = min(f - vec3f(ii), vec3f(1.0));
  let sxy = n.x * n.y;
  let k = n.w + ii.x + n.x * ii.y + sxy * ii.z;
  let c00 = mix(G[k], G[k + 1u], t.x);
  let c10 = mix(G[k + n.x], G[k + n.x + 1u], t.x);
  let c01 = mix(G[k + sxy], G[k + sxy + 1u], t.x);
  let c11 = mix(G[k + sxy + n.x], G[k + sxy + n.x + 1u], t.x);
  return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}
fn ggrad(g: u32, p: vec3f) -> vec3f {
  let e = u.go[g].w * 0.5;
  let v = vec3f(
    gsample(g, p + vec3f(e, 0, 0)) - gsample(g, p - vec3f(e, 0, 0)),
    gsample(g, p + vec3f(0, e, 0)) - gsample(g, p - vec3f(0, e, 0)),
    gsample(g, p + vec3f(0, 0, e)) - gsample(g, p - vec3f(0, 0, e)));
  let l = length(v);
  return select(vec3f(0, 1, 0), v / l, l >= 1e-9);
}

// ---------------------------------------------------------------- passes
@compute @workgroup_size(${WG})
fn predict(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let dt = u.a.x;
  var v = ld(VEL, i);
  v.y -= u.a.y * dt;
  v *= u.a.z;
  let sp = length(v);
  if (sp > u.a.w) { v *= u.a.w / sp; }
  st(VEL, i, v);
  let p = ld(POS, i);
  st(PREV, i, p);
  st(POS, i, p + v * dt);
  Ub[CONTACT + i] = ${CONTACT_NONE}u;
  Ub[AGE + i] = min(Ub[AGE + i] + 1u, 65535u);
}

@compute @workgroup_size(${WG})
fn clearCells(@builtin(global_invocation_id) id: vec3u) {
  if (id.x <= TABLE) { atomicStore(&CC[id.x], 0u); }
}

@compute @workgroup_size(${WG})
fn countCells(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let c = hashc(cellOf(ld(POS, i)));
  Ub[CELLOF + i] = c;
  atomicAdd(&CC[c], 1u);
}

var<workgroup> part: array<u32, 256>;

// exclusive prefix sum of the 65536 cell counts in one workgroup: 256 threads x 256 cells
@compute @workgroup_size(256)
fn scanCells(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  var s = 0u;
  for (var k = 0u; k < 256u; k++) { s += atomicLoad(&CC[t * 256u + k]); }
  part[t] = s;
  workgroupBarrier();
  if (t == 0u) {
    var acc = 0u;
    for (var k = 0u; k < 256u; k++) {
      let v = part[k];
      part[k] = acc;
      acc += v;
    }
    CS[TABLE] = acc;
  }
  workgroupBarrier();
  var acc = part[t];
  for (var k = 0u; k < 256u; k++) {
    let c = t * 256u + k;
    CS[c] = acc;
    acc += atomicLoad(&CC[c]);
  }
}

@compute @workgroup_size(${WG})
fn scatter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let c = Ub[CELLOF + i];
  Ub[SORTED + CS[c] + atomicAdd(&CC[c], 1u)] = i;
}

// atomics fill a cell in any order; sort it by particle index like the CPU's counting sort.
// A healthy fluid has ~6 particles per cell. A cell far denser than that (particles piled onto one
// spot) is left unsorted rather than paying O(len²): only its neighbour order is then arbitrary.
const SORT_MAX: u32 = 64u;
@compute @workgroup_size(${WG})
fn sortCells(@builtin(global_invocation_id) id: vec3u) {
  let c = id.x;
  if (c >= TABLE) { return; }
  let a = SORTED + CS[c];
  let b = SORTED + CS[c + 1u];
  if (b - a > SORT_MAX) { return; }
  for (var s = a + 1u; s < b; s++) {
    let v = Ub[s];
    var k = s;
    while (k > a && Ub[k - 1u] > v) {
      Ub[k] = Ub[k - 1u];
      k--;
    }
    Ub[k] = v;
  }
}

@compute @workgroup_size(${WG})
fn neighbors(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let p = ld(POS, i);
  let ic = cellOf(p);
  let base = i * MAXN;
  var cnt = 0u;
  for (var dz = -1; dz <= 1 && cnt < MAXN; dz++) {
    for (var dy = -1; dy <= 1 && cnt < MAXN; dy++) {
      for (var dx = -1; dx <= 1 && cnt < MAXN; dx++) {
        let c = hashc(ic + vec3i(dx, dy, dz));
        let e = CS[c + 1u];
        for (var s = CS[c]; s < e; s++) {
          let j = Ub[SORTED + s];
          if (j == i) { continue; }
          let d = ld(POS, j) - p;
          if (dot(d, d) < u.b.z) {
            // a hash collision can list a cell twice; skip duplicates
            var dup = false;
            for (var q = 0u; q < cnt; q++) {
              if (NB[base + q] == j) { dup = true; break; }
            }
            if (dup) { continue; }
            NB[base + cnt] = j;
            cnt++;
            if (cnt >= MAXN) { break; }
          }
        }
      }
    }
  }
  Ub[NC + i] = cnt;
}

@compute @workgroup_size(${WG})
fn lambdas(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let h = u.b.y;
  let rho0 = u.b.w;
  let p = ld(POS, i);
  var rho = 1.0;
  var gs = vec3f(0.0);
  var g2 = 0.0;
  let base = i * MAXN;
  let cnt = Ub[NC + i];
  for (var q = 0u; q < cnt; q++) {
    let d = p - ld(POS, NB[base + q]);
    let r2 = dot(d, d);
    if (r2 >= u.b.z) { continue; }
    rho += W(r2);
    let r = sqrt(r2);
    if (r < 1e-9) { continue; }
    let t = 1.0 - r / h;
    let gm = 3.0 * t * t / h / rho0;
    gs += gm * d / r;
    g2 += gm * gm;
  }
  let C = rho / rho0 - 1.0;
  Fb[LAM + i] = select(0.0, -C / (g2 + dot(gs, gs) + u.c.x), C > 0.0);
}

@compute @workgroup_size(${WG})
fn deltas(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let h = u.b.y;
  let rho0 = u.b.w;
  let d0 = u.c.y;
  let p = ld(POS, i);
  let li = Fb[LAM + i];
  var a = vec3f(0.0);
  let base = i * MAXN;
  let cnt = Ub[NC + i];
  let coh = select(0.0, u.c.w / f32(cnt), cnt > 0u);
  for (var q = 0u; q < cnt; q++) {
    let j = NB[base + q];
    let d = p - ld(POS, j);
    let r2 = dot(d, d);
    if (r2 >= u.b.z) { continue; }
    let r = sqrt(r2);
    if (r < 1e-9) { continue; }
    let t = 1.0 - r / h;
    let gm = 3.0 * t * t / h;
    let s = (-(li + Fb[LAM + j]) / rho0) * gm;
    a += s * d / r;
    if (r > d0 && coh > 0.0) {
      a -= (coh * 0.5 * (r - d0) / r) * d;
    }
  }
  let m = length(a);
  if (m > u.c.z) { a *= u.c.z / m; }
  st(DP, i, a);
}

/** Apply the correction, then project out of every solid (Sim.collideParticle). */
@compute @workgroup_size(${WG})
fn applyCollide(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let r = u.b.x;
  var p = ld(POS, i) + ld(DP, i);
  var best = 0.0;
  var ty = 0u;
  var nrm = vec3f(0, 1, 0);

  var d = p.y - r;
  if (d < 0.0) {
    p.y -= d;
    best = -d;
    ty = 1u;
  }
  let bi = nearestBowl(p.x);
  d = sdBowl(bi, p.x, p.y, p.z) - r;
  if (d < 0.0) {
    let g = gradBowl(bi, p);
    p -= d * g;
    if (-d >= best) {
      best = -d;
      ty = 1u;
      nrm = g;
    }
  }
  d = gsample(bi, p) - r;
  if (d < 0.0) {
    let g = ggrad(bi, p);
    p -= d * g;
    if (-d >= best) {
      best = -d;
      ty = 1u;
      nrm = g;
    }
  }

  let p0 = p;
  var rel = p - u.T.xyz;
  if (dot(rel, rel) < 0.04) {
    for (var k = 0; k < 2; k++) {
      let l = rel.x * u.R0.xyz + rel.y * u.R1.xyz + rel.z * u.R2.xyz; // R^T rel
      var dl: f32;
      var n = vec3f(0.0);
      if (k == 0) {
        dl = sdSpoonSolid(l) - r;
        if (dl < 0.0) { n = gradSpoon(l); }
      } else {
        dl = gsample(3u, l) - r;
        if (dl < 0.0) { n = ggrad(3u, l); }
      }
      if (dl >= 0.0) { continue; }
      let wn = vec3f(dot(u.R0.xyz, n), dot(u.R1.xyz, n), dot(u.R2.xyz, n));
      p -= dl * wn;
      rel = p - u.T.xyz;
      if (-dl >= best) {
        best = -dl;
        ty = 2u;
        nrm = wn;
      }
    }
  }
  // squeezed between the spoon and the bowl/table: keep it on the bowl's side, the CPU merges it
  if (ty == 2u && sdEnv(p) < 0.5 * r) {
    p = p0;
    Ub[SQUEEZED + i] = 1u;
  }
  st(POS, i, p);
  if (ty != 0u) {
    Ub[CONTACT + i] = ty;
    st(CN, i, nrm);
    var sv = vec3f(0.0);
    if (ty == 2u) { sv = u.lin.xyz + cross(u.ang.xyz, p - u.T.xyz); }
    st(SV, i, sv);
  }
}

@compute @workgroup_size(${WG})
fn velocities(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  var v = (ld(POS, i) - ld(PREV, i)) / u.a.x;
  let ct = Ub[CONTACT + i];
  let s = ld(SV, i);
  if (ct != 0u) {
    let n = ld(CN, i);
    var rv = v - s;
    var vn = dot(rv, n);
    if (vn < 0.0) {
      rv -= vn * n;
      vn = 0.0;
    } else if (vn > MAX_SEPARATION) {
      rv -= (vn - MAX_SEPARATION) * n;
      vn = MAX_SEPARATION;
    }
    let t = rv - vn * n;
    let kt = select(1.0 - u.e.x, 0.0, length(t) < u.e.y);
    v = s + vn * n + t * kt;
  }
  if (Ub[AGE + i] < 6u) {
    v = select(clamp(v, vec3f(-0.2), vec3f(0.2)), s, ct != 0u);
  }
  st(DP, i, v);
}

@compute @workgroup_size(${WG})
fn xsph(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.n.x) { return; }
  let p = ld(POS, i);
  let vi = ld(DP, i);
  var a = vec3f(0.0);
  var ws = 0.0;
  let base = i * MAXN;
  let cnt = Ub[NC + i];
  for (var q = 0u; q < cnt; q++) {
    let j = NB[base + q];
    let d = p - ld(POS, j);
    let r2 = dot(d, d);
    if (r2 >= u.b.z) { continue; }
    let w = W(r2);
    a += (ld(DP, j) - vi) * w;
    ws += w;
  }
  let k = select(0.0, u.e.z / max(ws, 1.0), ws > 0.0);
  st(VEL, i, vi + a * k);
}
`;
}

export class GpuFluid {
  constructor(device, sim) {
    const cap = sim.fluid.cap;
    this.device = device;
    this.cap = cap;
    const d = device;
    this.code = shader(cap);
    const module = d.createShaderModule({ code: this.code, label: 'fluid' });
    const S = GPUBufferUsage.STORAGE;
    const buf = (size, usage, label) => d.createBuffer({ size: Math.ceil(size / 4) * 4, usage, label });
    this.uniform = buf(21 * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'fluid params');
    this.F = buf(F.size * cap * 4, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'fluid F');
    this.U = buf(U.size * cap * 4, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'fluid U');
    this.NB = buf(MAXN * cap * 4, S, 'fluid neighbours');
    this.CC = buf((TABLE + 1) * 4, S, 'fluid cell counts');
    this.CS = buf((TABLE + 1) * 4, S, 'fluid cell starts');
    // collision grids: the three bowls' ice, then the spoon's content (spoon-local)
    this.grids = [...sim.grids, sim.content];
    this.gridOffset = [];
    let total = 0;
    for (const g of this.grids) {
      this.gridOffset.push(total);
      total += g.n;
    }
    this.G = buf(total * 4, S | GPUBufferUsage.COPY_DST, 'fluid grids');
    this.gridVersion = this.grids.map(() => -1);
    // read back: pos, vel, cn, sv (3 floats each) + age, contact, squeezed (u32)
    this.readback = buf(15 * cap * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'fluid readback');
    this.ages = new Uint32Array(cap);

    const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = d.createBindGroupLayout({
      entries: [entry(0, 'uniform'), ...[1, 2, 3, 4, 5].map((b) => entry(b, 'storage')), entry(6, 'read-only-storage')],
    });
    this.bind = d.createBindGroup({
      layout,
      entries: [this.uniform, this.F, this.U, this.NB, this.CC, this.CS, this.G].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.pipe = {};
    for (const name of ['predict', 'clearCells', 'countCells', 'scanCells', 'scatter', 'sortCells', 'neighbors', 'lambdas', 'deltas', 'applyCollide', 'velocities', 'xsph'])
      this.pipe[name] = d.createComputePipeline({ layout: pl, compute: { module, entryPoint: name }, label: name });
    this.params = new ArrayBuffer(21 * 16);
    this.pf = new Float32Array(this.params);
    this.pu = new Uint32Array(this.params);
    this.lastMs = 0;
  }

  _writeParams(dt, sim) {
    const f = sim.fluid;
    const p = sim.params.fluid;
    const sp = sim.spoon;
    const pf = this.pf;
    const pu = this.pu;
    pf.set([dt, p.gravity, Math.max(0, 1 - p.damping * dt), 1.2], 0);
    pf.set([f.r, f.h, f.h2, f.rho0], 4);
    pf.set([f.eps, f.d, 0.5 * f.r, p.cohesion], 8);
    pf.set([p.friction, 1.5 * p.friction * p.gravity * dt, p.viscosity, 1 / f.h], 12);
    pu.set([f.n, 0, 0, 0], 16);
    const R = sim.R;
    pf.set([R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0], 20);
    pf.set([sim.tx, sim.ty, sim.tz, 0], 32);
    pf.set([sp.linVel.x, sp.linVel.y, sp.linVel.z, 0, sp.angVel.x, sp.angVel.y, sp.angVel.z, 0], 36);
    this.grids.forEach((g, k) => {
      pf.set([g.ox, g.oy, g.oz, g.cell], 44 + k * 4);
      pu.set([g.nx, g.ny, g.nz, this.gridOffset[k]], 60 + k * 4);
      pf[76 + k] = g.band;
    });
    this.device.queue.writeBuffer(this.uniform, 0, this.params);
  }

  /**
   * Run `sub` PBF sub-steps of `dt` on the Sim's particles (the Sim's spoon pose must be set, as
   * Sim._stepBegin does). Same contract as calling sim.fluid.step(dt, params, sim) `sub` times.
   */
  async run(dt, sub, sim) {
    const t0 = performance.now();
    const f = sim.fluid;
    const n = f.n;
    if (!n) return;
    f.configure(sim.params.fluid.particleRadius, sim.params.fluid.kernelScale);
    const d = this.device;
    const q = d.queue;
    const cap = this.cap;
    this._writeParams(dt, sim);
    q.writeBuffer(this.F, F.pos * cap * 4, f.pos, 0, 3 * n);
    q.writeBuffer(this.F, F.vel * cap * 4, f.vel, 0, 3 * n);
    for (let i = 0; i < n; i++) this.ages[i] = f.age[i];
    q.writeBuffer(this.U, U.age * cap * 4, this.ages, 0, n);
    this.grids.forEach((g, k) => {
      if (g.version === this.gridVersion[k]) return;
      q.writeBuffer(this.G, this.gridOffset[k] * 4, g.data);
      this.gridVersion[k] = g.version;
    });

    const enc = d.createCommandEncoder({ label: 'fluid' });
    enc.clearBuffer(this.U, U.squeezed * cap * 4, n * 4);
    const pass = enc.beginComputePass({ label: 'fluid' });
    pass.setBindGroup(0, this.bind);
    const P = this.pipe;
    const groups = Math.ceil(n / WG);
    const run = (pipe, x) => {
      pass.setPipeline(pipe);
      pass.dispatchWorkgroups(x);
    };
    const iters = sim.params.fluid.iterations;
    for (let s = 0; s < sub; s++) {
      run(P.predict, groups);
      run(P.clearCells, Math.ceil((TABLE + 1) / WG));
      run(P.countCells, groups);
      run(P.scanCells, 1);
      run(P.clearCells, Math.ceil((TABLE + 1) / WG));
      run(P.scatter, groups);
      run(P.sortCells, Math.ceil(TABLE / WG));
      run(P.neighbors, groups);
      for (let it = 0; it < iters; it++) {
        run(P.lambdas, groups);
        run(P.deltas, groups);
        run(P.applyCollide, groups);
      }
      run(P.velocities, groups);
      run(P.xsph, groups);
    }
    pass.end();
    const rb = this.readback;
    const n3 = 3 * n * 4;
    enc.copyBufferToBuffer(this.F, F.pos * cap * 4, rb, 0, n3);
    enc.copyBufferToBuffer(this.F, F.vel * cap * 4, rb, n3, n3);
    enc.copyBufferToBuffer(this.F, F.cn * cap * 4, rb, 2 * n3, n3);
    enc.copyBufferToBuffer(this.F, F.sv * cap * 4, rb, 3 * n3, n3);
    enc.copyBufferToBuffer(this.U, U.age * cap * 4, rb, 4 * n3, n * 4);
    enc.copyBufferToBuffer(this.U, U.contact * cap * 4, rb, 4 * n3 + n * 4, n * 4);
    enc.copyBufferToBuffer(this.U, U.squeezed * cap * 4, rb, 4 * n3 + 2 * n * 4, n * 4);
    q.submit([enc.finish()]);

    await rb.mapAsync(GPUMapMode.READ, 0, 4 * n3 + 3 * n * 4);
    const m = rb.getMappedRange(0, 4 * n3 + 3 * n * 4);
    f.pos.set(new Float32Array(m, 0, 3 * n));
    f.vel.set(new Float32Array(m, n3, 3 * n));
    f.cn.set(new Float32Array(m, 2 * n3, 3 * n));
    f.sv.set(new Float32Array(m, 3 * n3, 3 * n));
    const age = new Uint32Array(m, 4 * n3, n);
    const contact = new Uint32Array(m, 4 * n3 + n * 4, n);
    const squeezed = new Uint32Array(m, 4 * n3 + 2 * n * 4, n);
    for (let i = 0; i < n; i++) {
      f.age[i] = age[i];
      f.contact[i] = contact[i];
      if (squeezed[i]) sim._squeezed[i] = 1;
    }
    rb.unmap();
    this.lastMs = performance.now() - t0;
  }
}
