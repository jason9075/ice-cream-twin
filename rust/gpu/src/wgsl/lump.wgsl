// Lumps: a released scoop keeps its shape (Fluid::match_lumps in icecream-core). Each solver iteration
// applyDeltas -> lumpMatch (one workgroup per lump: centroids, A = Σ (x - c)(q - c0)^T, rotation) ->
// lumpPull (each particle toward R (q - c0) + c) -> applyCollide; after the iterations lumpMatch +
// lumpPull with P.i0.x = 1 moves the rest place of particles pushed past G.g8.y (plastic flow).
// W_LUMPS, per lump id k (stride 32): R row-major (9), c (3), c0 (3), count, rotation quaternion (4).
const MIN_LUMP: f32 = 4.0;

fn lumpBase(k: u32) -> u32 { return W_LUMPS + k * 32u; }

@compute @workgroup_size(64)
fn applyDeltas(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  st3(W_POS, i, ld3(W_POS, i) + ld3(W_DP, i));
  st3(W_DP, i, vec3f(0.0));
}

var<workgroup> lumpA: array<vec4f, 256>;
var<workgroup> lumpB: array<vec4f, 256>;
var<workgroup> lumpC: array<vec4f, 256>;
var<workgroup> lumpCen: array<vec4f, 2>;

fn lumpReduce(t: u32) {
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      lumpA[t] += lumpA[t + s];
      lumpB[t] += lumpB[t + s];
      lumpC[t] += lumpC[t + s];
    }
    workgroupBarrier();
  }
}

fn quatMat(q: vec4f) -> mat3x3f {
  let x = q.x;
  let y = q.y;
  let z = q.z;
  let w = q.w;
  // columns
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

// rotational part of A (columns), iterating from q (Müller et al. 2016)
fn extractRotation(a: mat3x3f, q0: vec4f) -> vec4f {
  var q = q0;
  for (var it = 0; it < 20; it++) {
    let r = quatMat(q);
    let om0 = cross(r[0], a[0]) + cross(r[1], a[1]) + cross(r[2], a[2]);
    let dn = dot(r[0], a[0]) + dot(r[1], a[1]) + dot(r[2], a[2]);
    let om = om0 / (abs(dn) + 1e-9);
    let w = length(om);
    if (w < 1e-9) { break; }
    let ax = om / w;
    let d = vec4f(ax * sin(0.5 * w), cos(0.5 * w));
    let nq = vec4f(d.w * q.xyz + q.w * d.xyz + cross(d.xyz, q.xyz), d.w * q.w - dot(d.xyz, q.xyz));
    q = nq / length(nq);
  }
  return q;
}

@compute @workgroup_size(256)
fn lumpMatch(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let k = wg.x + 1u;
  let t = lid.x;
  let n = nParticles();
  var sx = vec4f(0.0);
  var sq = vec4f(0.0);
  for (var i = t; i < n; i += 256u) {
    if (X[X_LUMP + i] == k) {
      sx += vec4f(ld3(W_POS, i), 1.0);
      sq += vec4f(ld3(W_SHAPE, i), 0.0);
    }
  }
  lumpA[t] = sx;
  lumpB[t] = sq;
  lumpC[t] = vec4f(0.0);
  workgroupBarrier();
  lumpReduce(t);
  if (t == 0u) {
    let cnt = max(lumpA[0].w, 1.0);
    lumpCen[0] = vec4f(lumpA[0].xyz / cnt, lumpA[0].w);
    lumpCen[1] = vec4f(lumpB[0].xyz / cnt, 0.0);
  }
  workgroupBarrier();
  let c = lumpCen[0].xyz;
  let c0 = lumpCen[1].xyz;
  let cnt = lumpCen[0].w;
  // rows of A
  var r0 = vec3f(0.0);
  var r1 = vec3f(0.0);
  var r2 = vec3f(0.0);
  for (var i = t; i < n; i += 256u) {
    if (X[X_LUMP + i] == k) {
      let x = ld3(W_POS, i) - c;
      let q = ld3(W_SHAPE, i) - c0;
      r0 += x.x * q;
      r1 += x.y * q;
      r2 += x.z * q;
    }
  }
  lumpA[t] = vec4f(r0, 0.0);
  lumpB[t] = vec4f(r1, 0.0);
  lumpC[t] = vec4f(r2, 0.0);
  workgroupBarrier();
  lumpReduce(t);
  if (t == 0u) {
    let b = lumpBase(k);
    W[b + 15u] = cnt;
    if (cnt >= MIN_LUMP) {
      let a = transpose(mat3x3f(lumpA[0].xyz, lumpB[0].xyz, lumpC[0].xyz)); // columns from rows
      let q = extractRotation(a, vec4f(W[b + 16u], W[b + 17u], W[b + 18u], W[b + 19u]));
      var r = quatMat(q);
      for (var row = 0u; row < 3u; row++) {
        for (var col = 0u; col < 3u; col++) { W[b + row * 3u + col] = r[col][row]; }
      }
      for (var e = 0u; e < 3u; e++) {
        W[b + 9u + e] = c[e];
        W[b + 12u + e] = c0[e];
      }
      W[b + 16u] = q.x;
      W[b + 17u] = q.y;
      W[b + 18u] = q.z;
      W[b + 19u] = q.w;
    }
  }
}

// P.i0.x = 0: pull toward the goal by G.g8.x; 1: plastic update of the rest place past G.g8.y
@compute @workgroup_size(64)
fn lumpPull(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let k = X[X_LUMP + i];
  if (k == 0u) { return; }
  let b = lumpBase(k);
  if (W[b + 15u] < MIN_LUMP) {
    X[X_LUMP + i] = 0u;
    return;
  }
  let r0 = vec3f(W[b], W[b + 1u], W[b + 2u]);
  let r1 = vec3f(W[b + 3u], W[b + 4u], W[b + 5u]);
  let r2 = vec3f(W[b + 6u], W[b + 7u], W[b + 8u]);
  let c = vec3f(W[b + 9u], W[b + 10u], W[b + 11u]);
  let c0 = vec3f(W[b + 12u], W[b + 13u], W[b + 14u]);
  let q = ld3(W_SHAPE, i) - c0;
  let g = vec3f(dot(r0, q), dot(r1, q), dot(r2, q)) + c;
  let p = ld3(W_POS, i);
  if (P.i0.x == 0) {
    st3(W_POS, i, p + G.g8.x * (g - p));
  } else if (length(p - g) > G.g8.y) {
    let x = p - c;
    st3(W_SHAPE, i, r0 * x.x + r1 * x.y + r2 * x.z + c0); // R^T x + c0
  }
}
