// Spoon <-> ice kernels (Sim.interact, _carveGrid, _spawnDisplaced), carrying particles with the spoon,
// tipping the scoop out (releaseContent), eating, and particle bookkeeping (spawn, compaction).
//
// The CPU walks voxels and particles in order and keeps running sums (displaced volume -> one crumb
// per particle volume, the scoop's capture budget). Here every voxel runs at once and an exclusive
// prefix scan over the same order tells each voxel what the running sum was when it was reached.
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
/** Fluid.spawn into slot i. */
fn spawnAt(i: u32, p: vec3f, v: vec3f, rgba: u32) {
  st3(W_POS, i, p);
  st3(W_PREV, i, p);
  st3(W_FPOS, i, p);
  st3(W_VEL, i, v);
  X[X_REST + i] = 0u;
  X[X_AGE + i] = 0u;
  W[W_SLOW + i] = 1.0;
  X[X_CONTACT + i] = 0u;
  X[X_RGBA + i] = rgba;
  X[X_SQ + i] = 0u;
  X[X_ALIVE + i] = 1u;
}

// ================================================================ frame start
@compute @workgroup_size(1)
fn frameStart() {
  X[X_STU + SU_spawnLeft] = G.u0.y;
  W[W_ST + S_removed] = 0.0;
  X[X_STU + SU_frame] = G.u0.x;
}

// ================================================================ interact A: copy ice into the scoop
// P: R0..T = spoon pose, i0.x = tilted, i0.y = partial-sum slot.
@compute @workgroup_size(256)
fn copyMark(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let q = wg.x * 256u + lid.x;
  var gained = 0.0;
  if (q < NC) {
    X[X_CFLAG + q] = 0u;
    let n = GN[3];
    let c = vijk(3u, q);
    let cc = GO[3].w;
    let l = vcenter(3u, c);
    let cv = W[W_D + n.w + q];
    let accrete = G.g7.y > 0.5 && P.i0.x == 0 && W[W_ST + S_contentVol] < G.g5.z;
    let inHullBox = l.x > -SP_A && l.x < SP_A && l.y > -SP_B && l.y < 0.0 && l.z > -SP_C && l.z < SP_C;
    let below = l.y < 0.0 && sdEllipsoid(l, vec3f(SP_A + SP_T, SP_B + SP_T, SP_C + SP_T)) > 0.0;
    let nearWall = c.x < 3u || c.y < 2u || c.z < 3u || c.x >= n.x - 3u || c.y >= n.y - 3u || c.z >= n.z - 3u;
    var dR = select(BIG, cv - G.g5.y, accrete && !below && !nearWall);
    if (inHullBox) { dR = min(dR, sdHull(l)); }
    if (dR < cc) {
      let w = toWorld(P.R0, P.R1, P.R2, P.T, l);
      let g = gridAt(w);
      if (g != 99u) {
        let wv = gsample(g, w);
        if (wv < GBAND[g] * 0.999) {
          let nv = max(max(wv, dR), -sdShell(l));
          if (nv < cv) {
            gained = occ(3u, nv) - occ(3u, cv);
            W[W_CNV + q] = nv;
            var flag = 1u;
            let vi = voxelAt(g, w);
            if (vi >= 0 && nv < 2.0 * cc) {
              X[X_CSRC + q] = X[X_COL + GN[g].w + u32(vi)];
              flag = 3u;
              atomicMin(&A[A_CNT + C_firstIdx], q);
            }
            X[X_CFLAG + q] = flag;
          }
        }
      }
    }
  }
  let tot = wgSum(lid.x, gained);
  if (lid.x == 0u) { W[W_PART + u32(P.i0.y) + wg.x] = tot; }
}
// first ice on an empty spoon: its flavour becomes the base colour everywhere (SdfGrid.fillColor)
@compute @workgroup_size(64)
fn copyFill(@builtin(global_invocation_id) id: vec3u) {
  let q = id.x;
  if (q >= NC) { return; }
  let first = atomicLoad(&A[A_CNT + C_firstIdx]);
  if (W[W_ST + S_contentVol] < 1e-9 && first != 0xffffffffu) {
    X[X_COL + GN[3].w + q] = X[X_CSRC + first];
    if (q == 0u) { colorChanged(3u); }
  }
}
@compute @workgroup_size(64)
fn copyApply(@builtin(global_invocation_id) id: vec3u) {
  let q = id.x;
  if (q >= NC) { return; }
  let fl = X[X_CFLAG + q];
  if (fl == 0u) { return; }
  let k = W_D + GN[3].w + q;
  let cv = W[k];
  W[k] = W[W_CNV + q];
  touch(3u, q);
  if ((fl & 2u) != 0u) {
    let ci = X_COL + GN[3].w + q;
    X[ci] = blendRGBA(X[ci], X[X_CSRC + q], select(0.5, 1.0, cv > 0.0));
    colorChanged(3u);
  }
}
// gained -> capture budget (ST.gained was summed by finishSum), reset the first-colour marker
@compute @workgroup_size(1)
fn copyDone() {
  W[W_ST + S_captureBudget] = W[W_ST + S_gained];
  atomicStore(&A[A_CNT + C_firstIdx], 0xffffffffu);
}

// ================================================================ interact B: carve one grid region
// P: R0..T = spoon pose, i0 = (grid, region i0, j0, k0), i1 = (region size x, y, z, flags),
//    flags bit0 handleOnly, bit1 tilted; i2.x = partial slot.
fn regionVoxel(r: u32) -> vec3u {
  let s = vec3u(P.i1.xyz);
  return vec3u(P.i0.yzw) + vec3u(r % s.x, (r / s.x) % s.y, r / (s.x * s.y));
}
fn regionSize() -> u32 { return u32(P.i1.x * P.i1.y * P.i1.z); }

@compute @workgroup_size(256)
fn carveMark(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.x * 256u + lid.x;
  var removed = 0.0;
  if (r < regionSize()) {
    W[W_DVD + r] = 0.0;
    W[W_DVC + r] = 0.0;
    let g = u32(P.i0.x);
    let c = regionVoxel(r);
    let idx = vidx(g, c.x, c.y, c.z);
    let k = W_D + GN[g].w + idx;
    let w = W[k];
    let band = GBAND[g];
    if (w < band * 0.999) {
      let cc = GO[g].w;
      let pad = 0.4 * cc;
      let l = toLocal(P.R0, P.R1, P.R2, P.T, vcenter(g, c));
      let handleOnly = (P.i1.w & 1) != 0;
      let tilted = (P.i1.w & 2) != 0;
      var dS: f32;
      var dI = BIG;
      var dC = BIG;
      if (handleOnly) {
        dS = sdHandle(l) - pad;
      } else {
        dS = sdShell(l) - pad;
        if (!tilted) { dI = sdHull(l); }
        dC = gsample(3u, l);
      }
      var nw = -min(dS, min(dI, dC));
      if (nw > w) {
        nw = min(nw, band);
        let dv = occ(g, w) - occ(g, nw);
        W[k] = nw;
        touch(g, idx);
        removed = dv;
        let accrete = G.g7.y > 0.5 && !tilted && W[W_ST + S_contentVol] < G.g5.z;
        let isMetal = dS <= dI && dS <= dC;
        let isContentNoAccrete = !isMetal && dC < dI && !accrete;
        if (dv > 0.0) {
          if (isMetal || isContentNoAccrete) { W[W_DVD + r] = dv; } else { W[W_DVC + r] = dv; }
        }
      }
    }
  }
  let tot = wgSum(lid.x, removed);
  if (lid.x == 0u) { W[W_PART + u32(P.i2.x) + wg.x] = tot; }
}

// displaced voxels -> crumbs (one per particle volume, in voxel order) or owed; cup-swept voxels beyond
// what the scoop captured -> owed. PD / PC hold the exclusive prefix sums of DVD / DVC.
@compute @workgroup_size(64)
fn carveSpawn(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x;
  if (r >= regionSize()) { return; }
  let g = u32(P.i0.x);
  let cc = GO[g].w;
  let c3 = cc * cc * cc;
  let c = regionVoxel(r);
  let idx = vidx(g, c.x, c.y, c.z);
  let center = vcenter(g, c);
  let dvd = W[W_DVD + r];
  let dvc = W[W_DVC + r];
  let vp = G.g6.x;
  if (dvd > 0.0) {
    if (G.g6.w > 0.5) {
      let vpu = vp / c3;
      let acc0 = W[W_ST + S_displacedAcc] / c3;
      let kBase = floor(acc0 / vpu);
      let k0 = floor((acc0 + W[W_PD + r]) / vpu);
      let k1 = floor((acc0 + W[W_PD + r] + dvd) / vpu);
      let n0 = nParticles();
      let left = X[X_STU + SU_spawnLeft];
      let room = u32(max(0.0, min(G.g6.z, f32(CAP)) - f32(n0)));
      let handleOnly = (P.i1.w & 1) != 0;
      for (var m = k0; m < k1; m += 1.0) {
        let ord = u32(m - kBase);
        if (ord < left && ord < room) {
          // Sim._spawnDisplaced: push the crumb out of the metal, keep it inside the bowl
          let l = toLocal(P.R0, P.R1, P.R2, P.T, center);
          var d: f32;
          if (handleOnly) { d = sdHandle(l); } else { d = sdSolid(l); }
          let nrm = gradSolid(l, handleOnly);
          let pl = l + nrm * (-d + G.g1.x * 1.3);
          let w = toWorld(P.R0, P.R1, P.R2, P.T, pl);
          let slot = n0 + ord;
          if (sdEnv(w) < G.g1.x || gridAt(w) == 99u) {
            X[X_ALIVE + slot] = 0u;
            oweBin(g, center, vp);
          } else {
            spawnAt(slot, w, G.lin.xyz + cross(G.ang.xyz, w - P.T.xyz), X[X_COL + GN[g].w + idx]);
          }
        } else {
          oweBin(g, center, vp);
        }
      }
    } else {
      oweBin(g, center, dvd * c3);
    }
  }
  if (dvc > 0.0) {
    let budget = W[W_ST + S_captureBudget] / c3;
    let took = clamp(budget - W[W_PC + r], 0.0, dvc);
    if (dvc > took) { oweBin(g, center, (dvc - took) * c3); }
  }
}
// one thread: running sums after the region (displaced remainder, budget, spawned crumbs, removed)
@compute @workgroup_size(1)
fn carveDone() {
  let g = u32(P.i0.x);
  let cc = GO[g].w;
  let c3 = cc * cc * cc;
  let rn = regionSize();
  var removed = 0.0;
  for (var w = 0u; w < (rn + 255u) / 256u; w++) { removed += W[W_PART + u32(P.i2.x) + w]; }
  W[W_ST + S_removed] += removed * c3;
  if (rn == 0u) { return; }
  let totD = (W[W_PD + rn - 1u] + W[W_DVD + rn - 1u]) * c3;
  let totC = (W[W_PC + rn - 1u] + W[W_DVC + rn - 1u]) * c3;
  let vp = G.g6.x;
  if (G.g6.w > 0.5) {
    let acc0 = W[W_ST + S_displacedAcc];
    let k = u32(floor((acc0 + totD) / vp) - floor(acc0 / vp));
    let n0 = nParticles();
    let left = X[X_STU + SU_spawnLeft];
    let room = u32(max(0.0, min(G.g6.z, f32(CAP)) - f32(n0)));
    let used = min(k, min(left, room));
    let a = acc0 + totD;
    W[W_ST + S_displacedAcc] = a - floor(a / vp) * vp;
    X[X_STU + SU_spawnLeft] = left - used;
    atomicStore(&A[A_CNT + C_n], n0 + used);
  }
  let budget = W[W_ST + S_captureBudget];
  W[W_ST + S_captureBudget] = budget - min(budget, totC);
}
// end of one interact: the scoop picked up more than the bowls gave (resampling): the bowl owes it
// P.i0.x = first touching grid
@compute @workgroup_size(1)
fn interactDone() {
  let b = W[W_ST + S_captureBudget];
  if (b > 0.0) { W[W_ST + S_negSettle + u32(P.i0.x)] += b; }
  W[W_ST + S_captureBudget] = 0.0;
}

// ================================================================ carry particles with the spoon
// P: x0 = old position, q0 = inverse of the old rotation, x1 = new position, q1 = old -> new rotation
@compute @workgroup_size(64)
fn carry(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let p = ld3(W_POS, i);
  var take = X[X_CONTACT + i] == 2u;
  if (!take) {
    let l = qrot(P.q0, p - P.x0.xyz);
    take = sdHull(l) < 0.0 || gsample(3u, l) < 2.0 * G.g1.x;
  }
  if (!take) { return; }
  st3(W_POS, i, qrot(P.q1, p - P.x0.xyz) + P.x1.xyz);
  st3(W_PREV, i, qrot(P.q1, ld3(W_PREV, i) - P.x0.xyz) + P.x1.xyz);
}
@compute @workgroup_size(64)
fn markFrame(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  st3(W_FPOS, i, ld3(W_POS, i));
}

// ================================================================ tip the scoop out (releaseContent)
// lattice of the content box at the particle spacing; P.i0 = (lx, ly, lz count, 0), P.f0.x = spacing
fn latticePoint(r: u32) -> vec3f {
  let s = vec3u(P.i0.xyz);
  let c = vec3f(f32(r % s.x), f32((r / s.x) % s.y), f32(r / (s.x * s.y)));
  return GO[3].xyz + (c + 0.5) * P.f0.x;
}
@compute @workgroup_size(64)
fn releaseMark(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x;
  if (r >= u32(P.i0.x * P.i0.y * P.i0.z)) { return; }
  X[X_SCANU + r] = select(0u, 1u, W[W_ST + S_contentVol] > 0.0 && gsample(3u, latticePoint(r)) < 0.0);
}
@compute @workgroup_size(64)
fn releaseSpawn(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x;
  if (r >= u32(P.i0.x * P.i0.y * P.i0.z) || X[X_SCANU + r] == 0u) { return; }
  let n0 = nParticles();
  let ord = X[X_LIST + r];
  if (f32(n0 + ord) >= min(G.g6.z, f32(CAP))) { return; }
  let l = latticePoint(r);
  let w = toWorld(G.R0, G.R1, G.R2, G.T, l);
  let vi = voxelAt(3u, l);
  spawnAt(n0 + ord, w, spoonPointVel(w), X[X_COL + GN[3].w + u32(max(vi, 0))]);
}
@compute @workgroup_size(1)
fn releaseDone() {
  let rn = u32(P.i0.x * P.i0.y * P.i0.z);
  if (W[W_ST + S_contentVol] <= 0.0 || rn == 0u) { X[X_STU + SU_released] = 0u; return; }
  let tot = X[X_LIST + rn - 1u] + X[X_SCANU + rn - 1u];
  let n0 = nParticles();
  let room = u32(max(0.0, min(G.g6.z, f32(CAP)) - f32(n0)));
  atomicStore(&A[A_CNT + C_n], n0 + min(tot, room));
  X[X_STU + SU_released] = 1u;
}
// clear the content grid (after a release, or when eating); P.i0.x = 1: only if releaseDone fired
@compute @workgroup_size(64)
fn clearContent(@builtin(global_invocation_id) id: vec3u) {
  let q = id.x;
  if (q >= NC) { return; }
  if (P.i0.x == 1 && X[X_STU + SU_released] == 0u) { return; }
  W[W_D + GN[3].w + q] = GBAND[3];
  if (q == 0u) { distChanged(3u); }
}

// ================================================================ eat (Sim.eat)
@compute @workgroup_size(64)
fn eatMark(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let p = ld3(W_POS, i);
  let l = toLocal(G.R0, G.R1, G.R2, G.T, p);
  let onSpoon = X[X_CONTACT + i] == 2u || sdHull(l) < G.g1.x || (gcontains(3u, l, 0.0) && sdSolid(l) < 0.02);
  if (onSpoon) {
    X[X_ALIVE + i] = 0u;
    atomicAdd(&A[A_CNT + C_eatenN], 1u);
  }
}
@compute @workgroup_size(1)
fn eatDone() {
  W[W_ST + S_eatenVol] += f32(atomicLoad(&A[A_CNT + C_eatenN])) * G.g6.x;
  atomicStore(&A[A_CNT + C_eatenN], 0u);
}

// ================================================================ compaction (remove dead particles)
@compute @workgroup_size(64)
fn compactMark(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= CAP) { return; }
  X[X_SCANU + i] = select(0u, 1u, i < nParticles() && X[X_ALIVE + i] != 0u);
}
@compute @workgroup_size(64)
fn compactScatter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= CAP || X[X_SCANU + i] == 0u) { return; }
  let o = X[X_IDX + i];
  st3(W_QPOS, o, ld3(W_POS, i));
  st3(W_QVEL, o, ld3(W_VEL, i));
  st3(W_QCN, o, ld3(W_CN, i));
  st3(W_QSV, o, ld3(W_SV, i));
  W[W_QSLOW + o] = W[W_SLOW + i];
  X[X_QAGE + o] = X[X_AGE + i];
  X[X_QCONTACT + o] = X[X_CONTACT + i];
  X[X_QREST + o] = X[X_REST + i];
  X[X_QRGBA + o] = X[X_RGBA + i];
}
@compute @workgroup_size(1)
fn compactCount() {
  atomicStore(&A[A_CNT + C_n], X[X_IDX + CAP - 1u] + X[X_SCANU + CAP - 1u]);
}
@compute @workgroup_size(64)
fn compactCopy(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  st3(W_POS, i, ld3(W_QPOS, i));
  st3(W_VEL, i, ld3(W_QVEL, i));
  st3(W_CN, i, ld3(W_QCN, i));
  st3(W_SV, i, ld3(W_QSV, i));
  W[W_SLOW + i] = W[W_QSLOW + i];
  X[X_AGE + i] = X[X_QAGE + i];
  X[X_CONTACT + i] = X[X_QCONTACT + i];
  X[X_REST + i] = X[X_QREST + i];
  X[X_RGBA + i] = X[X_QRGBA + i];
  X[X_ALIVE + i] = 1u;
  X[X_SQ + i] = 0u;
}
