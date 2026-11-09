// Freezing (Sim.freezeResting): resting particles are stamped back into an SDF, Claybook's
// DieWhenStops + StampSDFWhenDies. Deciding is parallel; stamping is done one particle after another
// (in the CPU's order, highest index first) by one workgroup, all 256 threads sharing each stamp's
// voxels, so overlapping stamps and their measured volumes come out the same as on the CPU.
// codes in FRZ: 0 keep, 1..3 stamp into bowl grid code-1, 4 stamp into the scoop, 5 a static crumb
// (outside every grid), 6 out of the world (removed), 7 squeezed between spoon and bowl (owed to the
// bowl), 8 stamp into the table. A lump that has landed (LUMP_LANDING_FRAMES touching the world)
// freezes whole, in the shape it landed in.
fn embedded(contact: u32, p: vec3f) -> bool {
  let r = G.g1.x;
  if (contact == 1u) {
    let g = gridAt(p);
    return g != 99u && gsample(g, p) < 0.5 * r;
  }
  return gsample(3u, toLocal(G.R0, G.R1, G.R2, G.T, p)) < 0.5 * r;
}

const LUMP_LANDING_FRAMES: u32 = 6u;
// per lump: touching the world this frame (any of its particles), then count the frames in a row
@compute @workgroup_size(64)
fn lumpTouch(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= nParticles()) { return; }
  let l = X[X_LUMP + i];
  if (l != 0u && X[X_CONTACT + i] == 1u) { X[X_STU + SU_lumpTouch + l] = 1u; }
}
@compute @workgroup_size(1)
fn lumpLand() {
  for (var l = 1u; l <= MAX_LUMPS; l++) {
    let k = X_STU + SU_lumpLanded + l;
    X[k] = select(0u, X[k] + 1u, X[X_STU + SU_lumpTouch + l] != 0u);
    X[X_STU + SU_lumpTouch + l] = 0u;
  }
}
fn lumpLanded(i: u32) -> bool {
  let l = X[X_LUMP + i];
  return l != 0u && X[X_STU + SU_lumpLanded + l] >= LUMP_LANDING_FRAMES;
}

@compute @workgroup_size(64)
fn freezeMark(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= CAP) { return; }
  var code = 0u;
  if (i < nParticles()) {
    let p = ld3(W_POS, i);
    let rs = G.g4.z * G.g1.x;
    if (p.y < -0.2 || abs(p.x) > 1.0 || abs(p.z) > 1.0 || !(abs(p.x + p.y + p.z) < 1e30)) {
      code = 6u;
    } else if (X[X_SQ + i] != 0u) {
      X[X_SQ + i] = 0u;
      code = 7u;
    } else {
      let contact = X[X_CONTACT + i];
      let landed = lumpLanded(i);
      if (contact == 0u && !landed) {
        X[X_REST + i] = 0u;
      } else {
        // net displacement over the frame relative to the surface it rests on, smoothed
        let v = (p - ld3(W_FPOS, i)) / G.g5.w - ld3(W_SV, i);
        let slow = W[W_SLOW + i] * 0.6 + length(v) * 0.4;
        W[W_SLOW + i] = slow;
        var rest = 0u;
        if (slow < G.g4.x) { rest = X[X_REST + i] + 1u; }
        X[X_REST + i] = rest;
        let instant = G.g6.y > 0.5 && contact == 1u && X[X_AGE + i] > 3u;
        let emb = slow < 3.0 * G.g4.x && embedded(contact, p);
        if (f32(rest) >= G.g4.y || instant || emb || (landed && contact != 2u)) {
          if (contact == 2u) {
            if (G.g7.x > 0.5) {
              let l = toLocal(G.R0, G.R1, G.R2, G.T, p);
              let underCup = l.y < 0.0 && sdEllipsoid(l, vec3f(SP_A + SP_T, SP_B + SP_T, SP_C + SP_T)) > 0.0;
              if (!underCup && gcontains(3u, l, rs + 3.0 * GO[3].w)) { code = 4u; }
            }
          } else {
            let g = gridAt(p);
            if (g != 99u && gcontains(g, p, rs + GO[g].w)) { code = select(1u + g, 8u, g == TABLE_G); } else { code = 5u; }
          }
        }
      }
    }
  }
  X[X_FRZ + i] = code;
  X[X_SCANU + i] = select(0u, 1u, code != 0u);
}
@compute @workgroup_size(64)
fn freezeList(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= CAP || X[X_SCANU + i] == 0u) { return; }
  X[X_LIST + X[X_IDX + i]] = i;
}

/** SdfGrid.stampSphere, shared by the workgroup. Returns the added volume (m³). Uniform flow only. */
fn stampCoop(t: u32, g: u32, ctr: vec3f, R: f32, k: f32, rgba: u32) -> f32 {
  let c = GO[g].w;
  let reach = R + k + c;
  let n = vec3i(GN[g].xyz);
  let i0 = max(vec3i(floor((ctr - reach - GO[g].xyz) / c)), vec3i(0));
  let i1 = min(vec3i(floor((ctr + reach - GO[g].xyz) / c)), n - 1);
  let bs = vec3u(max(i1 - i0 + 1, vec3i(0)));
  let cnt = bs.x * bs.y * bs.z;
  let band = GBAND[g];
  var added = 0.0;
  for (var q = t; q < cnt; q += 256u) {
    let v3 = vec3u(i0) + vec3u(q % bs.x, (q / bs.x) % bs.y, q / (bs.x * bs.y));
    let x = vcenter(g, v3) - ctr;
    let ds = length(x) - R;
    if (ds > reach) { continue; }
    let idx = vidx(g, v3.x, v3.y, v3.z);
    let kk = W_D + GN[g].w + idx;
    let v = W[kk];
    let nv = max(smin(v, ds, k), -band);
    if (nv < v) {
      added += occ(g, nv) - occ(g, v);
      W[kk] = nv;
      touch(g, idx);
      if (ds < c) {
        let ci = X_COL + GN[g].w + idx;
        X[ci] = blendRGBA(X[ci], rgba, select(0.5, 1.0, v > 0.0));
        colorChanged(g);
      }
    }
  }
  let tot = wgSum(t, added) * c * c * c;
  storageBarrier();
  return tot;
}

var<workgroup> fzCode: u32;
@compute @workgroup_size(256)
fn freezeStamp(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let cnt = uniformU(t, X[X_IDX + CAP - 1u] + X[X_SCANU + CAP - 1u]);
  let vp = G.g6.x;
  let rs = G.g4.z * G.g1.x;
  for (var kq = 0u; kq < cnt; kq++) {
    let i = X[X_LIST + cnt - 1u - kq];
    if (t == 0u) { fzCode = X[X_FRZ + i]; }
    let code = workgroupUniformLoad(&fzCode);
    let p = ld3(W_POS, i);
    let rgba = X[X_RGBA + i];
    if (code >= 1u && code <= 3u) {
      let added = stampCoop(t, code - 1u, p, rs, G.g4.w, rgba);
      if (t == 0u) { settle(code - 1u, p, vp - added); }
    } else if (code == 4u) {
      let l = toLocal(G.R0, G.R1, G.R2, G.T, p);
      let added = stampCoop(t, 3u, l, rs, G.g4.w, rgba);
      if (t == 0u) { settle(3u, l, vp - added); }
    } else if (code == 8u) {
      let added = stampCoop(t, TABLE_G, p, rs, G.g4.w, rgba);
      if (t == 0u) { settle(TABLE_G, p, vp - added); }
    } else if (t == 0u) {
      if (code == 5u) {
        let s = atomicLoad(&A[A_CNT + C_spilled]);
        if (s < MAX_SPILLED) {
          st3(W_SPOS, s, p);
          X[X_SRGBA + s] = rgba;
          atomicStore(&A[A_CNT + C_spilled], s + 1u);
        }
        W[W_ST + S_spilledVol] += vp;
      } else if (code == 7u) {
        var g = gridAt(p);
        if (g == 99u) { g = nearestBowl(p.x); }
        oweBin(g, vec3f(p.x, max(p.y, GO[g].y + GO[g].w), p.z), vp);
      }
    }
    if (t == 0u) { X[X_ALIVE + i] = 0u; }
    storageBarrier();
    workgroupBarrier();
  }
}
