// The PBF solver (src/sim/fluid.js) on GPU-resident particles: predict -> hash-grid neighbours
// (count, prefix sum, scatter, per-cell sort so the order matches the CPU) -> [density + cohesion ->
// collide with the SDFs] x iterations -> velocity + contact friction -> XSPH viscosity. Same algorithm
// as the CPU in f32; the particle count is read from the GPU (A[CNT.n]).
const MAX_SEPARATION: f32 = 0.05;
fn Wk(r2: f32) -> f32 {
  let x = 1.0 - r2 / G.g1.z;
  return x * x * x;
}
fn cellOfP(p: vec3f) -> vec3i { return vec3i(floor(p * G.g3.w)); }
fn hashc(c: vec3i) -> u32 {
  return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & MASK;
}
// ---------------------------------------------------------------- passes
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let dt = G.g0.x;
  var v = ld3(W_VEL, i);
  v.y -= G.g0.y * dt;
  v *= G.g0.z;
  let sp = length(v);
  if (sp > G.g0.w) { v *= G.g0.w / sp; }
  st3(W_VEL, i, v);
  let p = ld3(W_POS, i);
  st3(W_PREV, i, p);
  st3(W_POS, i, p + v * dt);
  X[X_CONTACT + i] = 0u;
  X[X_AGE + i] = min(X[X_AGE + i] + 1u, 65535u);
}

@compute @workgroup_size(64)
fn clearCells(@builtin(global_invocation_id) id: vec3u) {
  if (id.x <= TABLE) { atomicStore(&A[A_CC + id.x], 0u); }
}

@compute @workgroup_size(64)
fn countCells(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let c = hashc(cellOfP(ld3(W_POS, i)));
  X[X_CELL + i] = c;
  atomicAdd(&A[A_CC + c], 1u);
}

var<workgroup> partC: array<u32, 256>;

// exclusive prefix sum of the 65536 cell counts in one workgroup: 256 threads x 256 cells
@compute @workgroup_size(256)
fn scanCells(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  var s = 0u;
  for (var k = 0u; k < 256u; k++) { s += atomicLoad(&A[A_CC + t * 256u + k]); }
  partC[t] = s;
  workgroupBarrier();
  if (t == 0u) {
    var acc = 0u;
    for (var k = 0u; k < 256u; k++) {
      let v = partC[k];
      partC[k] = acc;
      acc += v;
    }
    X[X_CS + TABLE] = acc;
  }
  workgroupBarrier();
  var acc = partC[t];
  for (var k = 0u; k < 256u; k++) {
    let c = t * 256u + k;
    X[X_CS + c] = acc;
    acc += atomicLoad(&A[A_CC + c]);
  }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let c = X[X_CELL + i];
  X[X_SORTED + X[X_CS + c] + atomicAdd(&A[A_CC + c], 1u)] = i;
}

// atomics fill a cell in any order; sort it by particle index like the CPU's counting sort.
// A healthy fluid has ~6 particles per cell. A cell far denser than that (particles piled onto one
// spot) is left unsorted rather than paying O(len²): only its neighbour order is then arbitrary.
const SORT_MAX: u32 = 64u;
@compute @workgroup_size(64)
fn sortCells(@builtin(global_invocation_id) id: vec3u) {
  let c = id.x;
  if (c >= TABLE) { return; }
  let a = X_SORTED + X[X_CS + c];
  let b = X_SORTED + X[X_CS + c + 1u];
  if (b - a > SORT_MAX) { return; }
  for (var s = a + 1u; s < b; s++) {
    let v = X[s];
    var k = s;
    while (k > a && X[k - 1u] > v) {
      X[k] = X[k - 1u];
      k--;
    }
    X[k] = v;
  }
}

@compute @workgroup_size(64)
fn neighbors(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let p = ld3(W_POS, i);
  let ic = cellOfP(p);
  let base = i * MAXN;
  var cnt = 0u;
  for (var dz = -1; dz <= 1 && cnt < MAXN; dz++) {
    for (var dy = -1; dy <= 1 && cnt < MAXN; dy++) {
      for (var dx = -1; dx <= 1 && cnt < MAXN; dx++) {
        let c = hashc(ic + vec3i(dx, dy, dz));
        let e = X[X_CS + c + 1u];
        for (var s = X[X_CS + c]; s < e; s++) {
          let j = X[X_SORTED + s];
          if (j == i) { continue; }
          let d = ld3(W_POS, j) - p;
          if (dot(d, d) < G.g1.z) {
            // a hash collision can list a cell twice; skip duplicates
            var dup = false;
            for (var q = 0u; q < cnt; q++) {
              if (X[X_NB + base + q] == j) { dup = true; break; }
            }
            if (dup) { continue; }
            X[X_NB + base + cnt] = j;
            cnt++;
            if (cnt >= MAXN) { break; }
          }
        }
      }
    }
  }
  X[X_NC + i] = cnt;
}

@compute @workgroup_size(64)
fn lambdas(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let h = G.g1.y;
  let rho0 = G.g1.w;
  let p = ld3(W_POS, i);
  var rho = 1.0;
  var gs = vec3f(0.0);
  var g2 = 0.0;
  let base = i * MAXN;
  let cnt = X[X_NC + i];
  for (var q = 0u; q < cnt; q++) {
    let d = p - ld3(W_POS, X[X_NB + base + q]);
    let r2 = dot(d, d);
    if (r2 >= G.g1.z) { continue; }
    rho += Wk(r2);
    let r = sqrt(r2);
    if (r < 1e-9) { continue; }
    let t = 1.0 - r / h;
    let gm = 3.0 * t * t / h / rho0;
    gs += gm * d / r;
    g2 += gm * gm;
  }
  let C = rho / rho0 - 1.0;
  W[W_LAM + i] = select(0.0, -C / (g2 + dot(gs, gs) + G.g2.x), C > 0.0);
}

@compute @workgroup_size(64)
fn deltas(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let h = G.g1.y;
  let rho0 = G.g1.w;
  let d0 = G.g2.y;
  let p = ld3(W_POS, i);
  let li = W[W_LAM + i];
  var a = vec3f(0.0);
  let base = i * MAXN;
  let cnt = X[X_NC + i];
  let coh = select(0.0, G.g2.w / f32(cnt), cnt > 0u);
  for (var q = 0u; q < cnt; q++) {
    let j = X[X_NB + base + q];
    let d = p - ld3(W_POS, j);
    let r2 = dot(d, d);
    if (r2 >= G.g1.z) { continue; }
    let r = sqrt(r2);
    if (r < 1e-9) { continue; }
    let t = 1.0 - r / h;
    let gm = 3.0 * t * t / h;
    let s = (-(li + W[W_LAM + j]) / rho0) * gm;
    a += s * d / r;
    if (r > d0 && coh > 0.0) {
      a -= (coh * 0.5 * (r - d0) / r) * d;
    }
  }
  let m = length(a);
  if (m > G.g2.z) { a *= G.g2.z / m; }
  st3(W_DP, i, a);
}

/** Apply the correction, then project out of every solid (Sim.collideParticle). */
@compute @workgroup_size(64)
fn applyCollide(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let r = G.g1.x;
  var p = ld3(W_POS, i) + ld3(W_DP, i);
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
  d = sdBowl(bi, p) - r;
  if (d < 0.0) {
    let g = gradBowl(bi, p);
    p -= d * g;
    if (-d >= best) {
      best = -d;
      ty = 1u;
      nrm = g;
    }
  }
  // the bowl's ice, then ice frozen on the table
  for (var k = 0u; k < 2u; k++) {
    let gi = select(TABLE_G, bi, k == 0u);
    d = gsample(gi, p) - r;
    if (d < 0.0) {
      let g = ggrad(gi, p);
      p -= d * g;
      if (-d >= best) {
        best = -d;
        ty = 1u;
        nrm = g;
      }
    }
  }

  let p0 = p;
  var rel = p - G.T.xyz;
  if (dot(rel, rel) < 0.04) {
    for (var k = 0; k < 2; k++) {
      let l = rel.x * G.R0.xyz + rel.y * G.R1.xyz + rel.z * G.R2.xyz; // R^T rel
      var dl: f32;
      var n = vec3f(0.0);
      if (k == 0) {
        dl = sdSolid(l) - r;
        if (dl < 0.0) { n = gradSolid(l, false); }
      } else {
        dl = gsample(3u, l) - r;
        if (dl < 0.0) { n = ggrad(3u, l); }
      }
      if (dl >= 0.0) { continue; }
      let wn = vec3f(dot(G.R0.xyz, n), dot(G.R1.xyz, n), dot(G.R2.xyz, n));
      p -= dl * wn;
      rel = p - G.T.xyz;
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
    X[X_SQ + i] = 1u;
  }
  st3(W_POS, i, p);
  if (ty != 0u) {
    X[X_CONTACT + i] = ty;
    st3(W_CN, i, nrm);
    var sv = vec3f(0.0);
    if (ty == 2u) { sv = G.lin.xyz + cross(G.ang.xyz, p - G.T.xyz); }
    st3(W_SV, i, sv);
  }
}

@compute @workgroup_size(64)
fn velocities(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  var v = (ld3(W_POS, i) - ld3(W_PREV, i)) / G.g0.x;
  let ct = X[X_CONTACT + i];
  let s = ld3(W_SV, i);
  if (ct != 0u) {
    let n = ld3(W_CN, i);
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
    let kt = select(1.0 - G.g3.x, 0.0, length(t) < G.g3.y);
    v = s + vn * n + t * kt;
  }
  if (X[X_AGE + i] < 6u) {
    v = select(clamp(v, vec3f(-0.2), vec3f(0.2)), s, ct != 0u);
  }
  st3(W_DP, i, v);
}

@compute @workgroup_size(64)
fn xsph(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let p = ld3(W_POS, i);
  let vi = ld3(W_DP, i);
  var a = vec3f(0.0);
  var ws = 0.0;
  let base = i * MAXN;
  let cnt = X[X_NC + i];
  for (var q = 0u; q < cnt; q++) {
    let j = X[X_NB + base + q];
    let d = p - ld3(W_POS, j);
    let r2 = dot(d, d);
    if (r2 >= G.g1.z) { continue; }
    let w = Wk(r2);
    a += (ld3(W_DP, j) - vi) * w;
    ws += w;
  }
  let k = select(0.0, G.g3.z / max(ws, 1.0), ws > 0.0);
  st3(W_VEL, i, vi + a * k);
}
