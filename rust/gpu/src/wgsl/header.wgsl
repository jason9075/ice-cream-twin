// Shared by every kernel: per-frame globals (G), per-dispatch parameters (P, dynamic offset), the
// storage buffers, and helpers for grids, the analytic SDFs, reductions and the volume ledger.
// The layout constants (W_*, X_*, ...), grid descriptors and spoon / bowl numbers are generated in
// layout.rs and prepended.
struct Globals {
  g0: vec4f, // substep dt, gravity, damp, vmax
  g1: vec4f, // r, h, h2, rho0
  g2: vec4f, // eps, d0, maxStep, cohesion
  g3: vec4f, // friction, staticV, viscosity, 1/h
  g4: vec4f, // stopSpeed, restFrames, stampScale, stampSmooth
  g5: vec4f, // smoothing, accretionDist, capacity (m³), frame dt
  g6: vec4f, // particle volume, stampOnContact, maxParticles, displacedToParticles
  g7: vec4f, // allowSpoon, accretion, releaseAngle, tilt (deg)
  R0: vec4f, R1: vec4f, R2: vec4f, T: vec4f, // spoon pose after the spoon step (rows of R, position)
  lin: vec4f, ang: vec4f,                    // spoon velocity
  u0: vec4u,                                 // frame, spawn budget, debt stamps per frame, max debts
}
struct Pass {
  i0: vec4i, i1: vec4i, i2: vec4i, i3: vec4i,
  f0: vec4f, f1: vec4f, f2: vec4f, f3: vec4f,
  R0: vec4f, R1: vec4f, R2: vec4f, T: vec4f,
  q0: vec4f, q1: vec4f, x0: vec4f, x1: vec4f,
}

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> P: Pass;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<u32>;
@group(0) @binding(4) var<storage, read_write> A: array<atomic<u32>>;
// binding 5: PP (server.wgsl), binding 6: PK (grid.wgsl)

// ---------------------------------------------------------------- small helpers
fn ld3(off: u32, i: u32) -> vec3f {
  let k = off + i * 3u;
  return vec3f(W[k], W[k + 1u], W[k + 2u]);
}
fn st3(off: u32, i: u32, v: vec3f) {
  let k = off + i * 3u;
  W[k] = v.x;
  W[k + 1u] = v.y;
  W[k + 2u] = v.z;
}
fn nParticles() -> u32 { return atomicLoad(&A[A_CNT + C_n]); }
fn unpackRGBA(c: u32) -> vec4f {
  return vec4f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(c >> 24u));
}
fn packRGBA(c: vec4f) -> u32 {
  let v = vec4u(clamp(trunc(c), vec4f(0.0), vec4f(255.0)));
  return v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
}
/** Uint8 blend like SdfGrid.blendColor: c += (rgba - c) * w, truncated. */
fn blendRGBA(old: u32, src: u32, w: f32) -> u32 {
  let a = unpackRGBA(old);
  return packRGBA(a + (unpackRGBA(src) - a) * w);
}
/** The CPU's LCG (Sim._rand). */
fn rand() -> f32 {
  let s = X[X_STU + SU_rng] * 1664525u + 1013904223u;
  X[X_STU + SU_rng] = s;
  return f32(s) / 4294967296.0;
}
fn smin(a: f32, b: f32, k: f32) -> f32 {
  if (k <= 0.0) { return min(a, b); }
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return b + (a - b) * h - k * h * (1.0 - h);
}
fn toLocal(R0: vec4f, R1: vec4f, R2: vec4f, T: vec4f, p: vec3f) -> vec3f {
  let r = p - T.xyz;
  return r.x * R0.xyz + r.y * R1.xyz + r.z * R2.xyz;
}
fn toWorld(R0: vec4f, R1: vec4f, R2: vec4f, T: vec4f, l: vec3f) -> vec3f {
  return vec3f(dot(R0.xyz, l), dot(R1.xyz, l), dot(R2.xyz, l)) + T.xyz;
}
fn rotWorld(R0: vec4f, R1: vec4f, R2: vec4f, n: vec3f) -> vec3f {
  return vec3f(dot(R0.xyz, n), dot(R1.xyz, n), dot(R2.xyz, n));
}

// ---------------------------------------------------------------- grids (SdfGrid)
fn gcell(g: u32) -> f32 { return GO[g].w; }
fn occ(g: u32, d: f32) -> f32 { return clamp(0.5 - d / GO[g].w, 0.0, 1.0); }
fn vidx(g: u32, i: u32, j: u32, k: u32) -> u32 { let n = GN[g]; return i + n.x * (j + n.y * k); }
fn vijk(g: u32, idx: u32) -> vec3u {
  let n = GN[g];
  let sxy = n.x * n.y;
  let k = idx / sxy;
  let r = idx - k * sxy;
  let j = r / n.x;
  return vec3u(r - j * n.x, j, k);
}
fn vcenter(g: u32, c: vec3u) -> vec3f { return GO[g].xyz + (vec3f(c) + 0.5) * GO[g].w; }
fn gsize(g: u32) -> vec3f { return vec3f(GN[g].xyz) * GO[g].w; }
fn gcontains(g: u32, p: vec3f, m: f32) -> bool {
  let o = GO[g].xyz;
  return all(p >= o + m) && all(p <= o + gsize(g) - m);
}
/** Voxel index containing p, or -1. */
fn voxelAt(g: u32, p: vec3f) -> i32 {
  let c = vec3i(floor((p - GO[g].xyz) / GO[g].w));
  let n = vec3i(GN[g].xyz);
  if (any(c < vec3i(0)) || any(c >= n)) { return -1; }
  return c.x + n.x * (c.y + n.y * c.z);
}
fn gsample(g: u32, p: vec3f) -> f32 {
  let o = GO[g];
  let n = GN[g];
  let nf = vec3f(n.xyz);
  var fv = (p - o.xyz) / o.w - 0.5;
  if (any(fv < vec3f(-0.5)) || any(fv > nf - 0.5)) { return GBAND[g]; }
  fv = max(fv, vec3f(0.0));
  let ii = min(vec3u(fv), n.xyz - 2u);
  let t = min(fv - vec3f(ii), vec3f(1.0));
  let sxy = n.x * n.y;
  let k = W_D + n.w + ii.x + n.x * ii.y + sxy * ii.z;
  let c00 = mix(W[k], W[k + 1u], t.x);
  let c10 = mix(W[k + n.x], W[k + n.x + 1u], t.x);
  let c01 = mix(W[k + sxy], W[k + sxy + 1u], t.x);
  let c11 = mix(W[k + sxy + n.x], W[k + sxy + n.x + 1u], t.x);
  return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}
fn ggrad(g: u32, p: vec3f) -> vec3f {
  let e = GO[g].w * 0.5;
  let v = vec3f(
    gsample(g, p + vec3f(e, 0, 0)) - gsample(g, p - vec3f(e, 0, 0)),
    gsample(g, p + vec3f(0, e, 0)) - gsample(g, p - vec3f(0, e, 0)),
    gsample(g, p + vec3f(0, 0, e)) - gsample(g, p - vec3f(0, 0, e)));
  let l = length(v);
  return select(vec3f(0, 1, 0), v / l, l >= 1e-9);
}
fn touch(g: u32, idx: u32) {
  X[X_FLAG + GN[g].w + idx] = 1u;
  X[X_STU + SU_dirty + g] = 1u;
}
fn colorChanged(g: u32) { X[X_STU + SU_dirty + 4u + g] = 1u; }
fn distChanged(g: u32) { X[X_STU + SU_dirty + g] = 1u; }

// ---------------------------------------------------------------- SDFs (src/sim/sdf.js)
fn nearestBowl(x: f32) -> u32 { return u32(clamp(i32(floor(x / BOWL_SPACING + 0.5)) + 1, 0, 2)); }
fn bowlCx(b: u32) -> f32 { return (f32(b) - 1.0) * BOWL_SPACING; }
fn sdBowl(b: u32, p: vec3f) -> f32 {
  let px = p.x - bowlCx(b);
  let py = p.y - BOWL_CY;
  let qx = sqrt(px * px + p.z * p.z);
  var d: f32;
  if (BOWL_H * qx < BOWL_W * py) {
    d = length(vec2f(qx - BOWL_W, py - BOWL_H));
  } else {
    d = abs(sqrt(qx * qx + py * py) - BOWL_R);
  }
  return d - BOWL_T;
}
fn sdEnv(p: vec3f) -> f32 { return min(p.y, sdBowl(nearestBowl(p.x), p)); }
/** Bowl grid holding p, or 99u (Sim.gridAt). */
fn gridAt(p: vec3f) -> u32 {
  let b = nearestBowl(p.x);
  return select(99u, b, gcontains(b, p, 0.0));
}
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  if (k1 < 1e-12) { return -min(r.x, min(r.y, r.z)); }
  return k0 * (k0 - 1.0) / k1;
}
fn sdShell(p: vec3f) -> f32 {
  return max(abs(sdEllipsoid(p, vec3f(SP_A, SP_B, SP_C))) - SP_T / 2.0, p.y);
}
fn sdHull(p: vec3f) -> f32 {
  let h = SP_T / 2.0;
  return max(sdEllipsoid(p, vec3f(SP_A - h, SP_B - h, SP_C - h)), p.y);
}
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
fn sdHandle(p: vec3f) -> f32 {
  var d = BIG;
  for (var k = 0u; k < NHANDLE - 1u; k++) { d = min(d, sdCapsule(p, HANDLE[k], HANDLE[k + 1u], SP_HR)); }
  return d;
}
fn sdSolid(p: vec3f) -> f32 { return min(sdShell(p), sdHandle(p)); }
fn gradSolid(p: vec3f, handleOnly: bool) -> vec3f {
  var g: vec3f;
  if (handleOnly) {
    g = vec3f(sdHandle(p + vec3f(GEPS, 0, 0)) - sdHandle(p - vec3f(GEPS, 0, 0)),
              sdHandle(p + vec3f(0, GEPS, 0)) - sdHandle(p - vec3f(0, GEPS, 0)),
              sdHandle(p + vec3f(0, 0, GEPS)) - sdHandle(p - vec3f(0, 0, GEPS)));
  } else {
    g = vec3f(sdSolid(p + vec3f(GEPS, 0, 0)) - sdSolid(p - vec3f(GEPS, 0, 0)),
              sdSolid(p + vec3f(0, GEPS, 0)) - sdSolid(p - vec3f(0, GEPS, 0)),
              sdSolid(p + vec3f(0, 0, GEPS)) - sdSolid(p - vec3f(0, 0, GEPS)));
  }
  let l = length(g);
  return select(g, g / l, l > 0.0);
}
fn gradBowl(b: u32, p: vec3f) -> vec3f {
  let g = vec3f(sdBowl(b, p + vec3f(GEPS, 0, 0)) - sdBowl(b, p - vec3f(GEPS, 0, 0)),
                sdBowl(b, p + vec3f(0, GEPS, 0)) - sdBowl(b, p - vec3f(0, GEPS, 0)),
                sdBowl(b, p + vec3f(0, 0, GEPS)) - sdBowl(b, p - vec3f(0, 0, GEPS)));
  let l = length(g);
  return select(g, g / l, l > 0.0);
}
fn spoonPointVel(p: vec3f) -> vec3f { return G.lin.xyz + cross(G.ang.xyz, p - G.T.xyz); }

// ---------------------------------------------------------------- workgroup reductions (256 threads)
var<workgroup> redF: array<f32, 256>;
var<workgroup> redU: array<u32, 256>;
var<workgroup> redI: array<i32, 256>;
/** Deterministic tree sum over the workgroup; every thread gets the total. Call in uniform flow. */
fn wgSum(t: u32, v: f32) -> f32 {
  redF[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { redF[t] += redF[t + s]; }
    workgroupBarrier();
  }
  let r = redF[0];
  workgroupBarrier();
  return r;
}
/** Exclusive prefix sum of v over the workgroup (Hillis-Steele); returns (prefix, total). */
fn wgScanU(t: u32, v: u32) -> vec2u {
  redU[t] = v;
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var a = 0u;
    if (t >= s) { a = redU[t - s]; }
    workgroupBarrier();
    redU[t] += a;
    workgroupBarrier();
  }
  let incl = redU[t];
  let total = redU[255];
  workgroupBarrier();
  return vec2u(incl - v, total);
}
fn wgMinI(t: u32, v: i32) -> i32 {
  redI[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { redI[t] = min(redI[t], redI[t + s]); }
    workgroupBarrier();
  }
  let r = redI[0];
  workgroupBarrier();
  return r;
}
fn wgMaxI(t: u32, v: i32) -> i32 {
  redI[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { redI[t] = max(redI[t], redI[t + s]); }
    workgroupBarrier();
  }
  let r = redI[0];
  workgroupBarrier();
  return r;
}

// ---------------------------------------------------------------- ledger primitives
/** Owe vol m³ at world point p of bowl grid g: into the fixed-point bin there (atomic). */
fn oweBin(g: u32, p: vec3f, vol: f32) {
  if (vol <= 0.0) { return; }
  let bn = GBINS[g];
  let c = clamp(vec3i(floor((p - GO[g].xyz) / (GO[g].w * f32(BIN)))), vec3i(0), vec3i(bn.xyz) - 1);
  let b = bn.w + u32(c.x) + bn.x * (u32(c.y) + bn.y * u32(c.z));
  atomicAdd(&A[A_BINS + b], u32(round(vol / VOL_UNIT)));
}
/**
 * Sim._owe / _settle for the content grid (spoon-local p): hand positive ice to the bowl under the
 * spoon, or to the loose balance off the bowls. Single-thread contexts only (touches ST).
 */
fn oweContent(p: vec3f, vol: f32) {
  let w = toWorld(G.R0, G.R1, G.R2, G.T, p);
  let b = nearestBowl(w.x);
  let mid = vec3f(w.x, GO[b].y + gsize(b).y * 0.5, w.z);
  if (gcontains(b, mid, 2.0 * GO[b].w)) {
    if (vol > 0.0) {
      oweBin(b, vec3f(w.x, min(w.y, GO[b].y + gsize(b).y - 2.0 * GO[b].w), w.z), vol);
    } else {
      W[W_ST + S_negSettle + b] += -vol;
    }
  } else {
    W[W_ST + S_looseOwed] += vol;
  }
}
/** Sim._settle(g, p, vol): positive is owed, negative cancels debts / becomes credit (later). */
fn settle(g: u32, p: vec3f, vol: f32) {
  if (g == 3u) { oweContent(p, vol); return; }
  if (vol > 0.0) { oweBin(g, p, vol); } else { W[W_ST + S_negSettle + g] += -vol; }
}
