// Loose particles drawn as one continuous surface instead of separate balls, entirely on the GPU.
//
// bounds  (per particle)   their bounding box, as order-preserving u32 atomics
// setup   (one thread)     the grid around it: origin, cell, dims (a corner of the MAX_DIM³ textures,
//                          at most MAX_VOXELS cells), the indirect dispatch sizes, and the volume
//                          parameters the surface's render pass reads (U)
// splat   (per particle)   accumulate the smoothed-particle sums of Zhu & Bridson (2005):
//                          phi(x) = |x - x̄(x)| - r,  x̄ = Σ w_i x_i / Σ w_i,  w = (1 - |x - x_i|² / R²)³
// resolve (per voxel)      read and clear the sums, write distance (+ colour) textures
// blurX/Y/Z (per voxel)    separable Gaussian on the distance (smooths particle-sized bumps)
//
// Particles come in the simulation's packed layout: per particle (f32 x, y, z, u32 rgba).

struct Params {
  n : u32, maxDim : u32, maxVoxels : u32, tint : u32,
  voxel : f32, R : f32, r : f32, smoothing : f32,
};
struct Grid {
  origin : vec3f, cell : f32,
  dims : vec3u, blur : u32,
};
struct Vol { model : mat4x4f, inv : mat4x4f, size : vec4f, prm : vec4f };

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> I : array<vec4u>;
@group(0) @binding(2) var<storage, read_write> acc : array<atomic<u32>>;
@group(0) @binding(3) var phiTex : texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var colTex : texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> phiA : array<f32>;
@group(0) @binding(6) var<storage, read_write> phiB : array<f32>;
// box: lo xyz, hi xyz as ordered u32 (atomics)
@group(0) @binding(7) var<storage, read_write> box : array<atomic<u32>, 6>;
@group(0) @binding(8) var<storage, read_write> grid : Grid;
// dispatch sizes (copied to the indirect buffer): resolve x y z, blur x y z
@group(0) @binding(9) var<storage, read_write> indirect : array<u32, 6>;
@group(0) @binding(10) var<storage, read_write> vol : Vol;

const S = 1048576.0; // fixed-point scale of the accumulators (atomics are integer only)
const CH = 8u;       // Σw, Σw·u (3), Σw·rgba (4)

fn ordered(f : f32) -> u32 {
  let b = bitcast<u32>(f);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}
fn unordered(u : u32) -> f32 {
  return bitcast<f32>(select(~u, u & 0x7fffffffu, (u & 0x80000000u) != 0u));
}
fn pos(i : u32) -> vec3f { return bitcast<vec3f>(I[i].xyz); }

@compute @workgroup_size(64) fn bounds(@builtin(global_invocation_id) id : vec3u) {
  if (id.x >= P.n) { return; }
  let p = pos(id.x);
  for (var k = 0u; k < 3u; k++) {
    atomicMin(&box[k], ordered(p[k]));
    atomicMax(&box[3u + k], ordered(p[k]));
  }
}

@compute @workgroup_size(1) fn setup() {
  let lo = vec3f(unordered(atomicLoad(&box[0])), unordered(atomicLoad(&box[1])), unordered(atomicLoad(&box[2])));
  let hi = vec3f(unordered(atomicLoad(&box[3])), unordered(atomicLoad(&box[4])), unordered(atomicLoad(&box[5])));
  let pad = P.R + 2.0 * P.voxel;
  let o = lo - pad;
  let ext = hi - lo + 2.0 * pad;
  let md = f32(P.maxDim);
  var cell = max(P.voxel, max(ext.x, max(ext.y, ext.z)) / md);
  let c = ceil(ext / cell);
  let vox = c.x * c.y * c.z;
  if (vox > f32(P.maxVoxels)) { cell *= pow(vox / f32(P.maxVoxels), 1.0 / 3.0) * 1.01; }
  let dims = vec3u(min(vec3f(md), ceil(ext / cell)));
  var blur = 0u;
  if (P.smoothing > 0.0) { blur = u32(clamp(round(P.smoothing / cell), 1.0, 4.0)); }
  grid.origin = o;
  grid.cell = cell;
  grid.dims = dims;
  grid.blur = blur;
  let groups = (dims + 3u) / 4u;
  indirect[0] = groups.x;
  indirect[1] = groups.y;
  indirect[2] = groups.z;
  let bg = select(vec3u(0u), groups, blur > 0u);
  indirect[3] = bg.x;
  indirect[4] = bg.y;
  indirect[5] = bg.z;
  let size = vec3f(dims) * cell;
  vol.model = mat4x4f(vec4f(1, 0, 0, 0), vec4f(0, 1, 0, 0), vec4f(0, 0, 1, 0), vec4f(o, 1.0));
  vol.inv = mat4x4f(vec4f(1, 0, 0, 0), vec4f(0, 1, 0, 0), vec4f(0, 0, 1, 0), vec4f(-o, 1.0));
  vol.size = vec4f(size, f32(P.tint));
  vol.prm = vec4f(cell, md, md, md);
}

fn add(i : u32, v : f32) { atomicAdd(&acc[i], u32(v * S + 0.5)); }

@compute @workgroup_size(64) fn splat(@builtin(global_invocation_id) id : vec3u) {
  if (id.x >= P.n) { return; }
  let it = I[id.x];
  let c = unpack4x8unorm(it.w);
  let p = bitcast<vec3f>(it.xyz) - grid.origin;
  let g = p / grid.cell - 0.5;            // voxel centres sit at (k + 0.5) * cell
  let rc = P.R / grid.cell;
  let lo = vec3i(max(ceil(g - rc), vec3f(0.0)));
  let hi = vec3i(min(floor(g + rc), vec3f(grid.dims) - 1.0));
  let invR2 = 1.0 / (P.R * P.R);
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let d = p - (vec3f(f32(x), f32(y), f32(z)) + 0.5) * grid.cell;
        let q = dot(d, d) * invR2;
        if (q >= 1.0) { continue; }
        let w = (1.0 - q) * (1.0 - q) * (1.0 - q);
        let u = d / P.R * 0.5 + 0.5;    // particle offset from the voxel, mapped to 0..1
        let b = (u32(x) + grid.dims.x * (u32(y) + grid.dims.y * u32(z))) * CH;
        add(b, w);
        add(b + 1u, w * u.x);
        add(b + 2u, w * u.y);
        add(b + 3u, w * u.z);
        add(b + 4u, w * c.r);
        add(b + 5u, w * c.g);
        add(b + 6u, w * c.b);
        add(b + 7u, w * c.a);
      }
    }
  }
}

fn take(i : u32) -> f32 { return f32(atomicExchange(&acc[i], 0u)) / S; }
fn idx(c : vec3u) -> u32 { return c.x + grid.dims.x * (c.y + grid.dims.y * c.z); }

// Reads and clears the accumulators (the next frame starts from zero) and writes the textures.
@compute @workgroup_size(4, 4, 4) fn resolve(@builtin(global_invocation_id) id : vec3u) {
  if (any(id >= grid.dims)) { return; }
  let b = idx(id) * CH;
  let w = take(b);
  let u = vec3f(take(b + 1u), take(b + 2u), take(b + 3u));
  let c = vec4f(take(b + 4u), take(b + 5u), take(b + 6u), take(b + 7u));
  var phi = P.R;
  var col = vec4f(0.0);
  if (w > 1e-5) {
    let xbar = (u / w * 2.0 - 1.0) * P.R;   // weighted mean particle position, relative to the voxel
    phi = clamp(length(xbar) - P.r, -P.R, P.R);
    col = c / w;
  }
  textureStore(colTex, id, col);
  if (grid.blur == 0u) { textureStore(phiTex, id, vec4f(phi, 0.0, 0.0, 0.0)); } else { phiA[idx(id)] = phi; }
}

fn gauss(k : i32) -> f32 {
  let s = f32(grid.blur) * 0.5;
  return exp(-f32(k * k) / (2.0 * s * s));
}

@compute @workgroup_size(4, 4, 4) fn blurX(@builtin(global_invocation_id) id : vec3u) {
  if (any(id >= grid.dims)) { return; }
  let rad = i32(grid.blur);
  var sum = 0.0;
  var ws = 0.0;
  for (var k = -rad; k <= rad; k++) {
    let j = i32(id.x) + k;
    if (j < 0 || j >= i32(grid.dims.x)) { continue; }
    let w = gauss(k);
    sum += w * phiA[idx(vec3u(u32(j), id.y, id.z))];
    ws += w;
  }
  phiB[idx(id)] = sum / ws;
}
@compute @workgroup_size(4, 4, 4) fn blurY(@builtin(global_invocation_id) id : vec3u) {
  if (any(id >= grid.dims)) { return; }
  let rad = i32(grid.blur);
  var sum = 0.0;
  var ws = 0.0;
  for (var k = -rad; k <= rad; k++) {
    let j = i32(id.y) + k;
    if (j < 0 || j >= i32(grid.dims.y)) { continue; }
    let w = gauss(k);
    sum += w * phiB[idx(vec3u(id.x, u32(j), id.z))];
    ws += w;
  }
  phiA[idx(id)] = sum / ws;
}
@compute @workgroup_size(4, 4, 4) fn blurZ(@builtin(global_invocation_id) id : vec3u) {
  if (any(id >= grid.dims)) { return; }
  let rad = i32(grid.blur);
  var sum = 0.0;
  var ws = 0.0;
  for (var k = -rad; k <= rad; k++) {
    let j = i32(id.z) + k;
    if (j < 0 || j >= i32(grid.dims.z)) { continue; }
    let w = gauss(k);
    sum += w * phiA[idx(vec3u(id.x, id.y, u32(j)))];
    ws += w;
  }
  textureStore(phiTex, id, vec4f(sum / ws, 0.0, 0.0, 0.0));
}
