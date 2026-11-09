// Grid kernels: prefix scans, volume reductions, volume-preserving smoothing (SdfGrid.smoothTouched /
// settle), redistancing (SdfGrid.redistance, Jacobi instead of fast sweeping), the content floor
// clamp, and packing the grids into the renderer's 3D textures.
// ================================================================ scans
// Exclusive prefix sum of n values at src into dst, in blocks of 512 (3 dispatches: block scan,
// scan of the block sums, add). P.i0 = (src, dst, n, 0). f32 values live in W, u32 in X.

@compute @workgroup_size(256)
fn scanBlockF(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let n = u32(P.i0.z);
  let base = wg.x * 512u;
  let a = select(0.0, W[u32(P.i0.x) + base + 2u * t], base + 2u * t < n);
  let b = select(0.0, W[u32(P.i0.x) + base + 2u * t + 1u], base + 2u * t + 1u < n);
  redF[t] = a + b;
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var v = 0.0;
    if (t >= s) { v = redF[t - s]; }
    workgroupBarrier();
    redF[t] += v;
    workgroupBarrier();
  }
  let excl = redF[t] - (a + b);
  if (base + 2u * t < n) { W[u32(P.i0.y) + base + 2u * t] = excl; }
  if (base + 2u * t + 1u < n) { W[u32(P.i0.y) + base + 2u * t + 1u] = excl + a; }
  if (t == 255u) { W[W_SCANB + wg.x] = redF[255]; }
}
@compute @workgroup_size(256)
fn scanBlockU(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let n = u32(P.i0.z);
  let base = wg.x * 512u;
  let a = select(0u, X[u32(P.i0.x) + base + 2u * t], base + 2u * t < n);
  let b = select(0u, X[u32(P.i0.x) + base + 2u * t + 1u], base + 2u * t + 1u < n);
  redU[t] = a + b;
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var v = 0u;
    if (t >= s) { v = redU[t - s]; }
    workgroupBarrier();
    redU[t] += v;
    workgroupBarrier();
  }
  let excl = redU[t] - (a + b);
  if (base + 2u * t < n) { X[u32(P.i0.y) + base + 2u * t] = excl; }
  if (base + 2u * t + 1u < n) { X[u32(P.i0.y) + base + 2u * t + 1u] = excl + a; }
  if (t == 255u) { W[W_SCANB + wg.x] = bitcast<f32>(redU[255]); }
}
// Scan block sums using a contiguous chunk per thread. The table grid needs
// more than 512 block sums; scanning only two per thread leaves stale offsets.
// SCANB holds nb input sums followed by nb exclusive offsets.
@compute @workgroup_size(256)
fn scanSums(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let nb = (u32(P.i0.z) + 511u) / 512u;
  let chunk = (nb + 255u) / 256u;
  let first = min(t * chunk, nb);
  let last = min(first + chunk, nb);
  if (P.i0.w == 1) {
    var sum = 0u;
    for (var i = first; i < last; i++) { sum += bitcast<u32>(W[W_SCANB + i]); }
    let scan = wgScanU(t, sum);
    var offset = scan.x;
    for (var i = first; i < last; i++) {
      let value = bitcast<u32>(W[W_SCANB + i]);
      W[W_SCANB + nb + i] = bitcast<f32>(offset);
      offset += value;
    }
  } else {
    var sum = 0.0;
    for (var i = first; i < last; i++) { sum += W[W_SCANB + i]; }
    redF[t] = sum;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride <<= 1u) {
      var value = 0.0;
      if (t >= stride) { value = redF[t - stride]; }
      workgroupBarrier();
      redF[t] += value;
      workgroupBarrier();
    }
    var offset = redF[t] - sum;
    for (var i = first; i < last; i++) {
      let value = W[W_SCANB + i];
      W[W_SCANB + nb + i] = offset;
      offset += value;
    }
  }
}
@compute @workgroup_size(256)
fn scanAdd(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = u32(P.i0.z);
  for (var e = 0u; e < 2u; e++) {
    let i = wg.x * 512u + 2u * lid.x + e;
    if (i >= n) { continue; }
    if (P.i0.w == 1) {
      X[u32(P.i0.y) + i] += bitcast<u32>(W[W_SCANB + (n + 511u) / 512u + wg.x]);
    } else {
      W[u32(P.i0.y) + i] += W[W_SCANB + (n + 511u) / 512u + wg.x];
    }
  }
}

// ================================================================ reductions
// Occupancy of grid P.i0.x, 1024 voxels per workgroup, partial sums to PART[P.i0.y + wg].
@compute @workgroup_size(256)
fn gridOcc(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let g = u32(P.i0.x);
  let n = GN[g].x * GN[g].y * GN[g].z;
  var s = 0.0;
  for (var e = 0u; e < 4u; e++) {
    let i = wg.x * 1024u + e * 256u + lid.x;
    if (i < n) { s += occ(g, W[W_D + GN[g].w + i]); }
  }
  let tot = wgSum(lid.x, s);
  if (lid.x == 0u) { W[W_PART + u32(P.i0.y) + wg.x] = tot; }
}
// Sum PART[P.i0.x .. + P.i0.y] * P.f0.x into ST[P.i0.z]: P.i0.w = 0 assign, 1 add, 2 subtract.
@compute @workgroup_size(256)
fn finishSum(@builtin(local_invocation_id) lid: vec3u) {
  var s = 0.0;
  for (var i = lid.x; i < u32(P.i0.y); i += 256u) { s += W[W_PART + u32(P.i0.x) + i]; }
  let tot = wgSum(lid.x, s) * P.f0.x;
  if (lid.x == 0u) {
    let k = W_ST + u32(P.i0.z);
    if (P.i0.w == 0) { W[k] = tot; } else if (P.i0.w == 1) { W[k] += tot; } else { W[k] -= tot; }
  }
}

// ================================================================ smoothing (one workgroup per grid)
var<workgroup> wgU: u32;
var<workgroup> wgF: f32;
fn uniformU(t: u32, v: u32) -> u32 {
  if (t == 0u) { wgU = v; }
  return workgroupUniformLoad(&wgU);
}
fn lap(g: u32, idx: u32, c: vec3u) -> f32 {
  let n = GN[g];
  let b = W_D + n.w;
  let sxy = n.x * n.y;
  return (W[b + select(idx, idx - 1u, c.x > 0u)] + W[b + select(idx, idx + 1u, c.x < n.x - 1u)] +
          W[b + select(idx, idx - n.x, c.y > 0u)] + W[b + select(idx, idx + n.x, c.y < n.y - 1u)] +
          W[b + select(idx, idx - sxy, c.z > 0u)] + W[b + select(idx, idx + sxy, c.z < n.z - 1u)]) / 6.0;
}
fn boxGet(g: u32, k: u32) -> i32 { return bitcast<i32>(X[X_STU + SU_boxes + g * 8u + k]); }
fn boxSet(g: u32, k: u32, v: i32) { X[X_STU + SU_boxes + g * 8u + k] = bitcast<u32>(v); }

/**
 * _writeVolumePreserving over LIST/TS[base .. base+n): values T - shift (at least TF) with the uniform
 * shift that keeps the enclosed volume (Newton in a shrinking bracket, 8 passes), residual to ST.
 */
fn volumePreservingWrite(t: u32, g: u32, n: u32, vb: f32) {
  let c = GO[g].w;
  let band = GBAND[g];
  let base = GN[g].w;
  var lo = -c;
  var hi = c;
  var shift = 0.0;
  var done = false;
  for (var it = 0; it < 8; it++) {
    var va = 0.0;
    var area = 0.0;
    for (var q = t; q < n; q += 256u) {
      let v = clamp(max(W[W_TS + base + q] - shift, W[W_TF + base + q]), -band, band);
      let o = 0.5 - v / c;
      if (o >= 1.0) { va += 1.0; } else if (o > 0.0) { va += o; area += 1.0; }
    }
    va = wgSum(t, va);
    area = wgSum(t, area);
    let err = vb - va;
    if (!done) {
      if (abs(err) < 0.02) {
        done = true;
      } else {
        if (err > 0.0) { lo = shift; } else { hi = shift; }
        var next = select(0.5 * (lo + hi), shift + err * c / area, area > 0.0);
        if (!(next > lo && next < hi)) { next = 0.5 * (lo + hi); }
        shift = next;
      }
    }
  }
  var va = 0.0;
  for (var q = t; q < n; q += 256u) {
    let v = clamp(max(W[W_TS + base + q] - shift, W[W_TF + base + q]), -band, band);
    W[W_D + base + X[X_LIST + base + q]] = v;
    va += occ(g, v);
  }
  va = wgSum(t, va);
  if (t == 0u && n > 0u) {
    distChanged(g);
    W[W_ST + S_residual + g] += (vb - va) * c * c * c;
    X[X_STU + SU_residualIdx + g] = X[X_LIST + base];
  }
}

// sim.rs spoon_keep_out: the lowest distance smoothing may leave in a bowl voxel, minus the distance to
// what carving cuts (spoon metal, the cup interior unless tilted, the ice on the spoon), so smoothing
// never refills it (a spoon held still would otherwise keep pumping ice into the cup). None for the scoop.
const NO_FLOOR: f32 = -1e30;
fn spoonKeepOut(g: u32, c: vec3u) -> f32 {
  if (g == 3u) { return NO_FLOOR; }
  let rel = GO[g].xyz + (vec3f(c) + 0.5) * GO[g].w - G.T.xyz;
  let l = rel.x * G.R0.xyz + rel.y * G.R1.xyz + rel.z * G.R2.xyz; // R^T rel
  if (any(l < KEEP_LO) || any(l > KEEP_HI)) { return NO_FLOOR; } // a band clear of all of it
  let pad = 0.4 * GO[g].w;
  var d = min(sdShell(l) - pad, sdHandle(l) - pad);
  let cs = gsample(3u, l);
  if (cs < GBAND[3] * 0.999) { d = min(d, cs); } // at the band: outside the scoop's grid or nothing near
  if (G.g7.w <= G.g7.z) { d = min(d, sdHullInterior(l)); }
  return -d;
}

// Smoothing one grid in four steps (P.i0 = (mode, grid, 0, 0), P.f0.x = strength):
//   smFlag  (whole grid, parallel)  which voxels take part -> SCANU, their bounding box (atomics)
//   scanU   SCANU -> PRE            their place in the list
//   smList  (whole grid, parallel)  LIST / TS (smoothed value from the values before any write), volume
//   smWrite (one workgroup)         the volume-preserving shift, writes, residual, settle bookkeeping
// mode 0: smoothTouched (this frame's edited voxels), 1: settle, 2: settle without consuming a frame
fn smTakes(g: u32, mode: i32, idx: u32, c: vec3u, sf: u32) -> bool {
  if (mode == 0) { return X[X_FLAG + GN[g].w + idx] != 0u; }
  if (sf == 0u || P.f0.x <= 0.0) { return false; }
  let ci = vec3i(c);
  let n = vec3i(GN[g].xyz);
  let r0 = max(vec3i(boxGet(g, 0u), boxGet(g, 1u), boxGet(g, 2u)) - 2, vec3i(0));
  let r1 = min(vec3i(boxGet(g, 3u), boxGet(g, 4u), boxGet(g, 5u)) + 2, n - 1);
  if (any(ci < r0) || any(ci > r1)) { return false; }
  let v = W[W_D + GN[g].w + idx];
  let lim = 2.0 * GO[g].w;
  return v <= lim && v >= -lim;
}
@compute @workgroup_size(256)
fn smFlag(@builtin(global_invocation_id) id: vec3u) {
  let g = u32(P.i0.y);
  let n = GN[g];
  let idx = id.x;
  if (idx == 0u) {
    // empty box for the atomics below
    atomicStore(&A[A_SMB + g * 8u], 0x7fffffffu); atomicStore(&A[A_SMB + g * 8u + 1u], 0x7fffffffu);
    atomicStore(&A[A_SMB + g * 8u + 2u], 0x7fffffffu);
  }
  if (idx >= n.x * n.y * n.z) { return; }
  let c = vijk(g, idx);
  let take = smTakes(g, P.i0.x, idx, c, X[X_STU + SU_settleFrames + g]);
  X[X_SCANU + idx] = select(0u, 1u, take);
}
// bounding box of this frame's edits (mode 0), after smFlag reset it
@compute @workgroup_size(256)
fn smBox(@builtin(global_invocation_id) id: vec3u) {
  let g = u32(P.i0.y);
  let n = GN[g];
  let idx = id.x;
  if (idx >= n.x * n.y * n.z || X[X_SCANU + idx] == 0u) { return; }
  let c = vijk(g, idx);
  atomicMin(&A[A_SMB + g * 8u], c.x); atomicMin(&A[A_SMB + g * 8u + 1u], c.y); atomicMin(&A[A_SMB + g * 8u + 2u], c.z);
  atomicMax(&A[A_SMB + g * 8u + 3u], c.x); atomicMax(&A[A_SMB + g * 8u + 4u], c.y); atomicMax(&A[A_SMB + g * 8u + 5u], c.z);
}
@compute @workgroup_size(256)
fn smList(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let g = u32(P.i0.y);
  let n = GN[g];
  let base = n.w;
  let idx = wg.x * 256u + lid.x;
  var vb = 0.0;
  if (idx < n.x * n.y * n.z && X[X_SCANU + idx] != 0u) {
    let c = vijk(g, idx);
    let v = W[W_D + base + idx];
    let o = X[X_PRE + idx];
    X[X_LIST + base + o] = idx;
    let k = select(P.f0.x * 0.5, P.f0.x, P.i0.x == 0);
    W[W_TS + base + o] = v + (lap(g, idx, c) - v) * k;
    W[W_TF + base + o] = spoonKeepOut(g, c);
    vb = occ(g, v);
  }
  let tot = wgSum(lid.x, vb);
  if (lid.x == 0u) { W[W_PART + wg.x] = tot; }
}
@compute @workgroup_size(256)
fn smWrite(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let g = u32(P.i0.y);
  let mode = P.i0.x;
  let n3 = GN[g];
  let total = n3.x * n3.y * n3.z;
  let n = uniformU(t, X[X_PRE + total - 1u] + X[X_SCANU + total - 1u]);
  let sf = uniformU(t, X[X_STU + SU_settleFrames + g]);
  if (n == 0u) { return; }
  var vb = 0.0;
  for (var w = t; w < (total + 255u) / 256u; w += 256u) { vb += W[W_PART + w]; }
  vb = wgSum(t, vb);
  if (g == TABLE_G && mode == 0 && P.f0.x > 0.0) {
    // what the table owes (stamps hold more than a particle) comes off its own ice here (Sim.table_owed)
    vb += W[W_ST + S_tableOwed] / (GO[g].w * GO[g].w * GO[g].w);
    storageBarrier();
    workgroupBarrier();
    if (t == 0u) { W[W_ST + S_tableOwed] = 0.0; }
  }
  if (mode == 0) {
    // this frame's edits: clear the flags, grow the settle box, keep settling for 30 frames
    for (var q = t; q < n; q += 256u) { X[X_FLAG + n3.w + X[X_LIST + n3.w + q]] = 0u; }
    if (t == 0u) {
      var b0 = vec3i(i32(atomicLoad(&A[A_SMB + g * 8u])), i32(atomicLoad(&A[A_SMB + g * 8u + 1u])), i32(atomicLoad(&A[A_SMB + g * 8u + 2u])));
      var b1 = vec3i(i32(atomicLoad(&A[A_SMB + g * 8u + 3u])), i32(atomicLoad(&A[A_SMB + g * 8u + 4u])), i32(atomicLoad(&A[A_SMB + g * 8u + 5u])));
      if (sf != 0u) {
        b0 = min(b0, vec3i(boxGet(g, 0u), boxGet(g, 1u), boxGet(g, 2u)));
        b1 = max(b1, vec3i(boxGet(g, 3u), boxGet(g, 4u), boxGet(g, 5u)));
      }
      boxSet(g, 0u, b0.x); boxSet(g, 1u, b0.y); boxSet(g, 2u, b0.z);
      boxSet(g, 3u, b1.x); boxSet(g, 4u, b1.y); boxSet(g, 5u, b1.z);
      X[X_STU + SU_settleFrames + g] = 30u;
      atomicStore(&A[A_SMB + g * 8u + 3u], 0u); atomicStore(&A[A_SMB + g * 8u + 4u], 0u); atomicStore(&A[A_SMB + g * 8u + 5u], 0u);
    }
    if (P.f0.x <= 0.0) { return; }
  } else if (t == 0u && mode == 1) {
    X[X_STU + SU_settleFrames + g] = sf - 1u;
  }
  volumePreservingWrite(t, g, n, vb);
}

// ================================================================ redistance (content grid)
// Jacobi Eikonal iterations from the voxels next to the zero level set: within the 4-cell band it
// reaches the same distances as the CPU's fast sweeping.
@compute @workgroup_size(64)
fn rdInit(@builtin(global_invocation_id) id: vec3u) {
  let q = id.x;
  if (q >= NC) { return; }
  let g = 3u;
  let n = GN[g];
  let b = W_D + n.w;
  let c = vijk(g, q);
  let sxy = n.x * n.y;
  let v = W[b + q];
  let c0 = GO[g].w;
  var best = BIG;
  let neg = v < 0.0;
  for (var m = 0u; m < 6u; m++) {
    var ok = true;
    var w = 0.0;
    switch m {
      case 0u: { ok = c.x > 0u; if (ok) { w = W[b + q - 1u]; } }
      case 1u: { ok = c.x < n.x - 1u; if (ok) { w = W[b + q + 1u]; } }
      case 2u: { ok = c.y > 0u; if (ok) { w = W[b + q - n.x]; } }
      case 3u: { ok = c.y < n.y - 1u; if (ok) { w = W[b + q + n.x]; } }
      case 4u: { ok = c.z > 0u; if (ok) { w = W[b + q - sxy]; } }
      default: { ok = c.z < n.z - 1u; if (ok) { w = W[b + q + sxy]; } }
    }
    if (!ok) { continue; }
    if (neg != (w < 0.0)) {
      var dd = abs(v - w);
      if (dd == 0.0) { dd = 1e-12; }
      best = min(best, abs(v) / dd * c0);
    }
  }
  W[W_RU0 + q] = best;
  W[W_RU1 + q] = best;
  X[X_CFLAG + q] = select(0u, 1u, best < BIG);
}
// P.i0.x = 0: RU0 -> RU1, 1: RU1 -> RU0
@compute @workgroup_size(64)
fn rdIter(@builtin(global_invocation_id) id: vec3u) {
  let q = id.x;
  if (q >= NC) { return; }
  let src = select(W_RU0, W_RU1, P.i0.x == 1);
  let dst = select(W_RU1, W_RU0, P.i0.x == 1);
  if (X[X_CFLAG + q] != 0u) { W[dst + q] = W[src + q]; return; }
  let g = 3u;
  let n = GN[g];
  let c = vijk(g, q);
  let sxy = n.x * n.y;
  let c0 = GO[g].w;
  var a = min(select(BIG, W[src + q - 1u], c.x > 0u), select(BIG, W[src + q + 1u], c.x < n.x - 1u));
  var bb = min(select(BIG, W[src + q - n.x], c.y > 0u), select(BIG, W[src + q + n.x], c.y < n.y - 1u));
  var e = min(select(BIG, W[src + q - sxy], c.z > 0u), select(BIG, W[src + q + sxy], c.z < n.z - 1u));
  if (a > bb) { let s = a; a = bb; bb = s; }
  if (bb > e) { let s = bb; bb = e; e = s; }
  if (a > bb) { let s = a; a = bb; bb = s; }
  var u = W[src + q];
  if (a < BIG) {
    var x = a + c0;
    if (x > bb) {
      let s2 = a + bb;
      x = 0.5 * (s2 + sqrt(max(0.0, s2 * s2 - 2.0 * (a * a + bb * bb - c0 * c0))));
      if (x > e) {
        let s3 = a + bb + e;
        x = (s3 + sqrt(max(0.0, s3 * s3 - 3.0 * (a * a + bb * bb + e * e - c0 * c0)))) / 3.0;
      }
    }
    u = min(u, x);
  }
  W[dst + q] = u;
}
@compute @workgroup_size(64)
fn rdFinish(@builtin(global_invocation_id) id: vec3u) {
  let q = id.x;
  if (q >= NC) { return; }
  let b = W_D + GN[3].w;
  let v = min(W[W_RU0 + q], GBAND[3]);
  W[b + q] = select(v, -v, W[b + q] < 0.0);
  if (q == 0u) { distChanged(3u); }
}

// ================================================================ content floor (Sim._clampContent)
@compute @workgroup_size(256)
fn clampContent(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let q = wg.x * 256u + lid.x;
  var removed = 0.0;
  if (q < NC) {
    let k = W_D + GN[3].w + q;
    let fl = W[W_FLOOR + q];
    if (W[k] < fl) {
      removed = occ(3u, W[k]) - occ(3u, fl);
      W[k] = fl;
      distChanged(3u);
    }
  }
  let tot = wgSum(lid.x, removed);
  if (lid.x == 0u) { W[W_PART + u32(P.i0.y) + wg.x] = tot; }
}

// ================================================================ render textures
// One u32 per 4 voxels of a row for the r8 distance texture, one per voxel for rgba8; rows padded to a
// multiple of 256 bytes (what copy_buffer_to_texture wants). P.i0 = (grid, pack offset in u32, distance
// words per row, colour words per row); 64 threads per row.
@group(0) @binding(6) var<storage, read_write> PK: array<u32>;
@compute @workgroup_size(64)
fn packGrid(@builtin(global_invocation_id) id: vec3u) {
  let g = u32(P.i0.x);
  let n = GN[g];
  let rows = n.y * n.z;
  let row = id.x / 64u;
  let w = id.x % 64u;
  if (row >= rows) { return; }
  let base = n.w + row * n.x;
  let s = 127.5 / GBAND[g];
  let dw = u32(P.i0.z);
  let cw = u32(P.i0.w);
  // distance: 4 voxels per u32
  for (var q = w; q < dw; q += 64u) {
    var word = 0u;
    for (var e = 0u; e < 4u; e++) {
      let x = q * 4u + e;
      if (x < n.x) {
        let qv = clamp(W[W_D + base + x] * s + 127.5, 0.0, 255.0);
        word |= u32(qv) << (8u * e);
      }
    }
    PK[u32(P.i0.y) + row * dw + q] = word;
  }
  // colour: one u32 per voxel, after the distance rows
  for (var q = w; q < cw; q += 64u) {
    var c = 0u;
    if (q < n.x) { c = X[X_COL + base + q]; }
    PK[u32(P.i0.y) + rows * dw + row * cw + q] = c;
  }
}
