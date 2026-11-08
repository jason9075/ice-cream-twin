// Loose particles drawn as one continuous surface instead of separate balls.
//
// Every frame the particles the server streamed (already in the instance buffer) are splatted into a
// grid around them, and each voxel gets the "smoothed particle" distance of Zhu & Bridson (2005):
//   phi(x) = |x - x̄(x)| - r,   x̄ = Σ w_i x_i / Σ w_i,   w = (1 - |x - x_i|² / R²)³
// so a lone particle is a ball of radius r and neighbours melt into one smooth blob. The result is a
// 3D distance texture (plus colour) that the same sphere tracer as the ice cream draws (volume shader,
// `fluid` variant), so loose ice cream and solid ice cream share one material.

const WG = 64;
const S = 1 << 20; // fixed-point scale of the accumulators (atomics are integer only)
const CHANNELS = 8; // Σw, Σw·u (3), Σw·rgba (4)

const splatWGSL = /* wgsl */ `
struct U {
  origin : vec3f, cell : f32,
  dims : vec3u, n : u32,
  R : f32, r : f32, _a : f32, _b : f32,
};
struct Inst { p : vec4f, c : vec4f };   // same layout as the sphere instances: xyz + radius, rgba
@group(0) @binding(0) var<uniform> P : U;
@group(0) @binding(1) var<storage, read> I : array<Inst>;
@group(0) @binding(2) var<storage, read_write> acc : array<atomic<u32>>;
@group(0) @binding(3) var phiTex : texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var colTex : texture_storage_3d<rgba8unorm, write>;

const S = ${S}.0;

fn add(i : u32, v : f32) { atomicAdd(&acc[i], u32(v * S + 0.5)); }

@compute @workgroup_size(${WG}) fn splat(@builtin(global_invocation_id) id : vec3u) {
  if (id.x >= P.n) { return; }
  let it = I[id.x];
  let p = it.p.xyz - P.origin;
  let g = p / P.cell - 0.5;            // voxel centres sit at (k + 0.5) * cell
  let rc = P.R / P.cell;
  let lo = vec3i(max(ceil(g - rc), vec3f(0.0)));
  let hi = vec3i(min(floor(g + rc), vec3f(P.dims) - 1.0));
  let invR2 = 1.0 / (P.R * P.R);
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let d = p - (vec3f(f32(x), f32(y), f32(z)) + 0.5) * P.cell;
        let q = dot(d, d) * invR2;
        if (q >= 1.0) { continue; }
        let w = (1.0 - q) * (1.0 - q) * (1.0 - q);
        let u = d / P.R * 0.5 + 0.5;    // particle offset from the voxel, mapped to 0..1
        let b = (u32(x) + P.dims.x * (u32(y) + P.dims.y * u32(z))) * ${CHANNELS}u;
        add(b, w);
        add(b + 1u, w * u.x);
        add(b + 2u, w * u.y);
        add(b + 3u, w * u.z);
        add(b + 4u, w * it.c.r);
        add(b + 5u, w * it.c.g);
        add(b + 6u, w * it.c.b);
        add(b + 7u, w * it.c.a);
      }
    }
  }
}

fn take(i : u32) -> f32 { return f32(atomicExchange(&acc[i], 0u)) / S; }

// Reads and clears the accumulators (so the next frame starts from zero) and writes the textures.
@compute @workgroup_size(4, 4, 4) fn resolve(@builtin(global_invocation_id) id : vec3u) {
  if (any(id >= P.dims)) { return; }
  let b = (id.x + P.dims.x * (id.y + P.dims.y * id.z)) * ${CHANNELS}u;
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
  textureStore(phiTex, id, vec4f(phi, 0.0, 0.0, 0.0));
  textureStore(colTex, id, col);
}
`;

/**
 * The surface of `instances` (a createInstances() object). Its texture is a fixed `maxDim`³ allocation;
 * each frame uses the corner that covers the particles' bounding box, at `voxel` metres per cell or
 * coarser when the box would need more than `maxVoxels` cells.
 */
export function createFluidSurface(r, instances, { maxDim = 128, maxVoxels = 1_500_000 } = {}) {
  const d = r.device;
  const mod = d.createShaderModule({ code: splatWGSL, label: 'fluid splat' });
  mod.getCompilationInfo().then((info) => {
    for (const m of info.messages) console[m.type === 'error' ? 'error' : 'warn'](`WGSL fluid: ${m.lineNum}:${m.linePos} ${m.message}`);
  });
  const C = GPUShaderStage.COMPUTE;
  const bgl = d.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: C, buffer: {} },
      { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: C, buffer: { type: 'storage' } },
      { binding: 3, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '3d' } },
    ],
  });
  const layout = d.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const splat = d.createComputePipeline({ layout, compute: { module: mod, entryPoint: 'splat' } });
  const resolve = d.createComputePipeline({ layout, compute: { module: mod, entryPoint: 'resolve' } });

  const size = [maxDim, maxDim, maxDim];
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const phiTex = d.createTexture({ size, dimension: '3d', format: 'rgba16float', usage });
  const colTex = d.createTexture({ size, dimension: '3d', format: 'rgba8unorm', usage });
  const acc = d.createBuffer({ size: maxVoxels * CHANNELS * 4, usage: GPUBufferUsage.STORAGE });
  const ubuf = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = d.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: ubuf } },
      { binding: 1, resource: { buffer: instances.sbuf } },
      { binding: 2, resource: { buffer: acc } },
      { binding: 3, resource: phiTex.createView({ dimension: '3d' }) },
      { binding: 4, resource: colTex.createView({ dimension: '3d' }) },
    ],
  });
  const u = new ArrayBuffer(48);
  const uf = new Float32Array(u);
  const uu = new Uint32Array(u);

  // drawn by the volume shader's fluid variant over a unit box scaled to the active region
  const vol = r.createVolume(
    { nx: maxDim, ny: maxDim, nz: maxDim, ox: 0, oy: 0, oz: 0, sx: 1, sy: 1, sz: 1, band: 0, cell: 1 },
    r.createMesh(unitBox()),
    { external: true, textures: [phiTex, colTex] },
  );
  vol.fluid = true;

  const surf = {
    vol,
    maxDim,
    /** Particles, grid dims and cell size of the last update (for the debug readout). */
    info: { n: 0, dims: [0, 0, 0], cell: 0 },
    /** Box (world) the surface covers this frame, or null when there is nothing to draw. */
    box: null,
    /**
     * Fit the grid around the n particles of `instances` (bounds lo/hi, world metres).
     * r: surface radius, R: kernel radius, voxel: preferred cell size, tint: debug colour.
     */
    update(n, lo, hi, { r: rad, R, voxel, tint = false }) {
      surf.info.n = n;
      if (!n) {
        surf.box = null;
        return;
      }
      const pad = R + 2 * voxel;
      const o = [lo[0] - pad, lo[1] - pad, lo[2] - pad];
      const ext = [hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, hi[2] - lo[2] + 2 * pad];
      let cell = Math.max(voxel, Math.max(...ext) / maxDim);
      const vox = (c) => ext.reduce((a, e) => a * Math.ceil(e / c), 1);
      if (vox(cell) > maxVoxels) cell *= Math.cbrt(vox(cell) / maxVoxels) * 1.01;
      const dims = ext.map((e) => Math.min(maxDim, Math.ceil(e / cell)));
      uf.set([o[0], o[1], o[2], cell], 0);
      uu.set([dims[0], dims[1], dims[2], n], 4);
      uf.set([R, rad, 0, 0], 8);
      d.queue.writeBuffer(ubuf, 0, u);
      surf.info.dims = dims;
      surf.info.cell = cell;
      surf.box = { origin: o, size: dims.map((k) => k * cell) };
      vol.data.set([surf.box.size[0], surf.box.size[1], surf.box.size[2], tint ? 1 : 0], 32);
      vol.data.set([cell, maxDim, maxDim, maxDim], 36);
      vol.setModel(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, o[0], o[1], o[2], 1]));
    },
    /** Record the splat + resolve passes (before the frame's render passes). */
    encode(enc) {
      if (!surf.box) return;
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, bg);
      pass.setPipeline(splat);
      pass.dispatchWorkgroups(Math.ceil(surf.info.n / WG));
      pass.setPipeline(resolve);
      const [x, y, z] = surf.info.dims;
      pass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
      pass.end();
    },
  };
  return surf;
}

function unitBox() {
  // 8 corners, position only (the volume pipeline reads location 0 of a 24-byte vertex)
  const v = [];
  for (let k = 0; k < 8; k++) v.push(k & 1, (k >> 1) & 1, (k >> 2) & 1, 0, 0, 0);
  const f = [
    [0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5],
  ];
  const i = [];
  for (const [a, b, c, e] of f) i.push(a, b, c, a, c, e);
  return { vertices: new Float32Array(v), indices: new Uint32Array(i) };
}
