// What the server streams: particles and crumbs packed as (f32 x, y, z, u32 rgba), read back each frame.
// The grids go through packGrid (grid.wgsl); STU dirty marks say which ones changed.
@group(0) @binding(5) var<storage, read_write> PP: array<u32>;

// P.i0.x = where the crumbs start in PP (u32 index)
@compute @workgroup_size(64)
fn packParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i < nParticles()) {
    let p = ld3(W_POS, i);
    PP[i * 4u] = bitcast<u32>(p.x);
    PP[i * 4u + 1u] = bitcast<u32>(p.y);
    PP[i * 4u + 2u] = bitcast<u32>(p.z);
    PP[i * 4u + 3u] = X[X_RGBA + i];
  }
  if (i < atomicLoad(&A[A_CNT + C_spilled])) {
    let o = u32(P.i0.x) + i * 4u;
    let p = ld3(W_SPOS, i);
    PP[o] = bitcast<u32>(p.x);
    PP[o + 1u] = bitcast<u32>(p.y);
    PP[o + 2u] = bitcast<u32>(p.z);
    PP[o + 3u] = X[X_SRGBA + i];
  }
}
