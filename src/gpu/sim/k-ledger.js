// The volume ledger on the GPU (Sim._owe / _settle / payDebts / _payCredit / _settleResidual).
//
// Ice removed but not yet put back is owed. Kernels that run in parallel add what they owe into
// fixed-point bins (3x3x3 voxels of a bowl grid, atomic adds, so the totals don't depend on thread
// order); once a frame one workgroup turns the bins into debts (the CPU's debt list, merged the same
// way), applies the pending "extra ice" (negative settles: cancel debts, else credit), and pays: it
// stamps owed ice onto the lowest free ice top near where it went missing, clear of the spoon, and
// shaves credit off the highest. Thread 0 makes the decisions; the workgroup evaluates candidate
// columns and stamps together.
export const ledgerWGSL = /* wgsl */ `
const DEBT_STAMPS: u32 = 48u;
const BOWL_T_HALF: f32 = 0.0015;
const PI: f32 = 3.14159265358979;

fn dPos(e: u32) -> vec3f { return vec3f(W[W_DEBTF + e * 4u], W[W_DEBTF + e * 4u + 1u], W[W_DEBTF + e * 4u + 2u]); }
fn dVol(e: u32) -> f32 { return W[W_DEBTF + e * 4u + 3u]; }
fn dSetVol(e: u32, v: f32) { W[W_DEBTF + e * 4u + 3u] = v; }
fn dGrid(e: u32) -> u32 { return X[X_DEBTU + e * 3u + 2u]; }
fn dRGBA(e: u32) -> u32 { return X[X_DEBTU + e * 3u]; }
fn dTries(e: u32) -> u32 { return X[X_DEBTU + e * 3u + 1u]; }
fn dCopy(dst: u32, src: u32) {
  for (var k = 0u; k < 4u; k++) { W[W_DEBTF + dst * 4u + k] = W[W_DEBTF + src * 4u + k]; }
  for (var k = 0u; k < 3u; k++) { X[X_DEBTU + dst * 3u + k] = X[X_DEBTU + src * 3u + k]; }
}
fn nDebts() -> u32 { return X[X_STU + SU_nDebts]; }
/** Remove debt e (the last one moves into its slot). */
fn dRemove(e: u32) {
  let n = nDebts() - 1u;
  if (e != n) { dCopy(e, n); }
  X[X_STU + SU_nDebts] = n;
}

/** Sim._owe for a bowl grid (thread 0 only): credit first, merge with a recent nearby debt, push. */
fn owe(g: u32, p: vec3f, vol0: f32, rgba: u32) {
  var vol = vol0;
  let cr = W[W_ST + S_credit + g];
  if (cr > 0.0) {
    let used = min(cr, vol);
    W[W_ST + S_credit + g] = cr - used;
    vol -= used;
    if (vol <= 0.0) { return; }
  }
  let n = nDebts();
  let near = 3.0 * GO[g].w;
  var e = i32(n) - 1;
  loop {
    if (e < max(0, i32(n) - 8)) { break; }
    let q = dPos(u32(e));
    if (dGrid(u32(e)) == g && all(abs(q - p) < vec3f(near))) {
      dSetVol(u32(e), dVol(u32(e)) + vol);
      return;
    }
    e--;
  }
  if (n >= MAX_DEBTS) {
    dSetVol(n - 1u, dVol(n - 1u) + vol);
    return;
  }
  W[W_DEBTF + n * 4u] = p.x;
  W[W_DEBTF + n * 4u + 1u] = p.y;
  W[W_DEBTF + n * 4u + 2u] = p.z;
  W[W_DEBTF + n * 4u + 3u] = vol;
  X[X_DEBTU + n * 3u] = rgba;
  X[X_DEBTU + n * 3u + 1u] = 0u;
  X[X_DEBTU + n * 3u + 2u] = g;
  X[X_STU + SU_nDebts] = n + 1u;
}
/** Extra ice in grid g: cancel its debts, newest first, the rest becomes credit (Sim._settle < 0). */
fn extraIce(g: u32, vol: f32) {
  var extra = vol;
  var e = i32(nDebts()) - 1;
  loop {
    if (e < 0 || extra <= 0.0) { break; }
    if (dGrid(u32(e)) == g) {
      let used = min(extra, dVol(u32(e)));
      dSetVol(u32(e), dVol(u32(e)) - used);
      extra -= used;
      if (dVol(u32(e)) <= 0.0) { dRemove(u32(e)); }
    }
    e--;
  }
  if (extra > 0.0) { W[W_ST + S_credit + g] += extra; }
}
fn binGrid(b: u32) -> u32 {
  if (b >= GBINS[2].w) { return 2u; }
  if (b >= GBINS[1].w) { return 1u; }
  return 0u;
}
fn binCenter(g: u32, b: u32) -> vec3f {
  let bn = GBINS[g];
  let r = b - bn.w;
  let c = vec3f(f32(r % bn.x), f32((r / bn.x) % bn.y), f32(r / (bn.x * bn.y)));
  let s = GO[g].w * f32(BIN);
  return min(GO[g].xyz + (c + 0.5) * s, GO[g].xyz + gsize(g) - 0.5 * GO[g].w);
}
fn colorAt(g: u32, p: vec3f) -> u32 {
  let vi = voxelAt(g, p);
  return X[X_COL + GN[g].w + u32(max(vi, 0))];
}

// ---------------------------------------------------------------- bins -> debts (one workgroup)
@compute @workgroup_size(256)
fn ledgerBins(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let chunk = (NBINS + 255u) / 256u;
  let b0 = min(t * chunk, NBINS);
  let b1 = min(b0 + chunk, NBINS);
  var cnt = 0u;
  for (var b = b0; b < b1; b++) { cnt += select(0u, 1u, atomicLoad(&A[A_BINS + b]) != 0u); }
  let sc = wgScanU(t, cnt);
  var o = sc.x;
  for (var b = b0; b < b1; b++) {
    if (atomicLoad(&A[A_BINS + b]) != 0u) {
      X[X_BINL + o] = b;
      o++;
    }
  }
  storageBarrier();
  workgroupBarrier();
  if (t != 0u) { return; }
  for (var k = 0u; k < sc.y; k++) {
    let b = X[X_BINL + k];
    let vol = f32(atomicExchange(&A[A_BINS + b], 0u)) * VOL_UNIT;
    let g = binGrid(b);
    let p = binCenter(g, b);
    owe(g, p, vol, colorAt(g, p));
  }
  for (var g = 0u; g < 3u; g++) {
    let ex = W[W_ST + S_negSettle + g];
    W[W_ST + S_negSettle + g] = 0.0;
    if (ex > 0.0) { extraIce(g, ex); }
  }
  // owed ice with nowhere to go on the spoon falls as loose particles (Sim._owe, content branch)
  let vp = G.g6.x;
  var n = nParticles();
  let maxN = u32(min(G.g6.z, f32(CAP)));
  loop {
    if (W[W_ST + S_looseOwed] < vp || n >= maxN) { break; }
    W[W_ST + S_looseOwed] -= vp;
    let w = toWorld(G.R0, G.R1, G.R2, G.T, vec3f(0.0, 0.0, 0.0));
    spawnAt(n, w + vec3f(0.0, 2.0 * G.g1.x, 0.0), vec3f(0.0), colorAt(3u, vec3f(0.0)));
    n++;
  }
  atomicStore(&A[A_CNT + C_n], n);
}

// ---------------------------------------------------------------- paying (one workgroup)
// shared: the debt being paid, candidate columns, the chosen spot
var<workgroup> lgAct: u32;     // 0 done, 1 search debt, 2 skip, 3 credit carve
var<workgroup> lgP: vec4f;     // debt x, y, z, rs
var<workgroup> lgDir: vec4f;   // search direction x, z, rimIn, grid
var<workgroup> lgBest: vec4f;  // chosen x, y, z, found
var<workgroup> lgTop: array<f32, 32>;
var<workgroup> lgRGBA: u32;
var<workgroup> lgCand: array<vec2f, 32>;

/** Top ice surface of column (x, z) in grid g, or -BIG (Sim._columnTop). */
fn columnTop(g: u32, x: f32, z: f32) -> f32 {
  let c = GO[g].w;
  var y = GO[g].y + gsize(g).y - c;
  loop {
    if (y <= GO[g].y + c) { break; }
    if (gsample(g, vec3f(x, y, z)) < 0.0) { return y; }
    y -= 0.5 * c;
  }
  return -BIG;
}
fn clearOfSpoon(p: vec3f, m: f32) -> bool {
  let l = toLocal(G.R0, G.R1, G.R2, G.T, p);
  return sdSolid(l) > m && sdHull(l) > m;
}
// cup + scoop world box: P.x0 = min, P.x1 = max
fn underCup(x: f32, y: f32, z: f32, m: f32) -> bool {
  return x > P.x0.x - m && x < P.x1.x + m && z > P.x0.z - m && z < P.x1.z + m && P.x1.y > y;
}
/** SdfGrid.carveSphere, shared by the workgroup; removed volume (m³). */
fn carveCoop(t: u32, g: u32, ctr: vec3f, R: f32) -> f32 {
  let c = GO[g].w;
  let reach = R + 2.0 * c;
  let n = vec3i(GN[g].xyz);
  let i0 = max(vec3i(floor((ctr - reach - GO[g].xyz) / c)), vec3i(0));
  let i1 = min(vec3i(floor((ctr + reach - GO[g].xyz) / c)), n - 1);
  let bs = vec3u(max(i1 - i0 + 1, vec3i(0)));
  let cnt = bs.x * bs.y * bs.z;
  var removed = 0.0;
  for (var q = t; q < cnt; q += 256u) {
    let v3 = vec3u(i0) + vec3u(q % bs.x, (q / bs.x) % bs.y, q / (bs.x * bs.y));
    let idx = vidx(g, v3.x, v3.y, v3.z);
    let kk = W_D + GN[g].w + idx;
    let v = W[kk];
    let nv = min(max(v, R - length(vcenter(g, v3) - ctr)), GBAND[g]);
    if (nv > v) {
      removed += occ(g, v) - occ(g, nv);
      W[kk] = nv;
      touch(g, idx);
    }
  }
  let tot = wgSum(t, removed) * c * c * c;
  storageBarrier();
  return tot;
}

@compute @workgroup_size(256)
fn ledgerPay(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let k = G.g4.w;
  var stamps = 0u;
  // -------- credit: shave extra ice off the highest free top (Sim._payCredit)
  for (var g = 0u; g < 3u; g++) {
    let c = GO[g].w;
    let rs = 1.5 * c;
    let minVol = 0.5 * (2.0 / 3.0) * PI * rs * rs * rs;
    var pending = false;
    if (t == 0u) {
      for (var e = 0u; e < nDebts(); e++) { if (dGrid(e) == g) { pending = true; } }
      lgAct = select(0u, 1u, W[W_ST + S_credit + g] >= minVol && !pending);
    }
    if (workgroupUniformLoad(&lgAct) == 0u) { continue; }
    for (var tries = 0u; tries < 6u; tries++) {
      if (t == 0u) {
        lgAct = select(0u, 1u, W[W_ST + S_credit + g] >= minVol);
        let rimR = BOWL_W - 0.006;
        for (var m = 0u; m < 12u; m++) {
          let a2 = rand() * PI * 2.0;
          let r2 = sqrt(rand()) * rimR;
          lgCand[m] = vec2f(bowlCx(g) + cos(a2) * r2, sin(a2) * r2);
        }
      }
      if (workgroupUniformLoad(&lgAct) == 0u) { break; }
      if (t < 12u) {
        let xz = lgCand[t];
        let top = columnTop(g, xz.x, xz.y);
        var ok = top > -BIG && !underCup(xz.x, top, xz.y, rs) && clearOfSpoon(vec3f(xz.x, top, xz.y), rs + 0.002);
        lgTop[t] = select(-BIG, top, ok);
      }
      workgroupBarrier();
      if (t == 0u) {
        var best = -BIG;
        var bi = 0u;
        for (var m = 0u; m < 12u; m++) { if (lgTop[m] > best) { best = lgTop[m]; bi = m; } }
        lgBest = vec4f(lgCand[bi].x, best, lgCand[bi].y, select(0.0, 1.0, best > -BIG));
      }
      stamps++;
      let bst = workgroupUniformLoad(&lgBest);
      if (bst.w > 0.5) {
        let removed = carveCoop(t, g, vec3f(bst.x, bst.y + 0.5 * rs, bst.z), rs);
        if (t == 0u) { W[W_ST + S_credit + g] -= removed; }
      }
      workgroupBarrier();
    }
    if (t == 0u) {
      let cr = W[W_ST + S_credit + g];
      if (cr < 0.0) {
        W[W_ST + S_credit + g] = 0.0;
        let mid = GO[g].xyz + 0.5 * gsize(g);
        owe(g, mid, -cr, colorAt(g, mid));
      }
    }
    workgroupBarrier();
  }

  // -------- debts: stamp owed ice onto the lowest free top near where it went missing (Sim.payDebts)
  var n = 0u;
  loop {
    if (t == 0u) {
      lgAct = 0u;
      loop {
        if (n >= nDebts() || stamps >= DEBT_STAMPS) { break; }
        let g = dGrid(n);
        let c = GO[g].w;
        let vol = dVol(n);
        let rs = min(2.5 * c, max(1.2 * c, pow(vol * 3.0 / (2.0 * PI), 1.0 / 3.0)));
        // wait until there is enough owed for a stamp of the minimum size (less churn)
        if (vol < 0.25 * (2.0 / 3.0) * PI * rs * rs * rs && dTries(n) == 0u && stamps > 0u) { n++; continue; }
        let a = rand() * PI * 2.0;
        lgP = vec4f(dPos(n), rs);
        lgDir = vec4f(cos(a), sin(a), BOWL_W - BOWL_T_HALF - rs - c, f32(g));
        lgRGBA = dRGBA(n);
        lgAct = 1u;
        break;
      }
    }
    if (workgroupUniformLoad(&lgAct) == 0u) { break; }
    let dp = workgroupUniformLoad(&lgP);
    let dd = workgroupUniformLoad(&lgDir);
    let g = u32(dd.w);
    let c = GO[g].w;
    let rs = dp.w;
    // 16 candidate columns along the direction; the search stops at the first one outside the grid
    if (t < 16u) {
      let xx = dp.x + dd.x * f32(t) * c;
      let zz = dp.z + dd.y * f32(t) * c;
      let inside = gcontains(g, vec3f(xx, GO[g].y + gsize(g).y * 0.5, zz), rs + c);
      var top = -BIG;
      if (inside && length(vec2f(xx - bowlCx(g), zz)) <= dd.z) {
        let tp = columnTop(g, xx, zz);
        if (tp > -BIG && tp <= GO[g].y + gsize(g).y - 2.0 * rs - c && !underCup(xx, tp, zz, rs) &&
            clearOfSpoon(vec3f(xx, tp, zz), rs + 0.002)) { top = tp; }
      }
      lgTop[t] = select(-2.0 * BIG, top, inside);
      lgCand[t] = vec2f(xx, zz);
    }
    workgroupBarrier();
    if (t == 0u) {
      var best = BIG;
      var bi = 99u;
      for (var m = 0u; m < 16u; m++) {
        if (lgTop[m] == -2.0 * BIG) { break; }
        if (lgTop[m] > -BIG && lgTop[m] < best) { best = lgTop[m]; bi = m; }
      }
      stamps++;
      X[X_DEBTU + n * 3u + 1u] = dTries(n) + 1u;
      if (bi == 99u && dTries(n) > 8u) {
        // stuck: jump to a random spot inside the bowl that isn't under the spoon
        for (var m = 0u; m < 8u; m++) {
          let a2 = rand() * PI * 2.0;
          let r2 = sqrt(rand()) * (BOWL_W - 0.01);
          let xx = bowlCx(g) + cos(a2) * r2;
          let zz = sin(a2) * r2;
          if (!underCup(xx, -1.0, zz, rs)) {
            W[W_DEBTF + n * 4u] = xx;
            W[W_DEBTF + n * 4u + 2u] = zz;
            break;
          }
        }
      }
      if (bi != 99u) { lgBest = vec4f(lgCand[bi].x, best, lgCand[bi].y, 1.0); } else { lgBest = vec4f(0.0); }
    }
    let bst = workgroupUniformLoad(&lgBest);
    let rgba = workgroupUniformLoad(&lgRGBA);
    if (bst.w > 0.5) {
      let added = stampCoop(t, g, bst.xyz, rs, k, rgba);
      if (t == 0u) { dSetVol(n, dVol(n) - added); }
    }
    if (t == 0u) {
      let v = dVol(n);
      if (v <= 0.0) {
        dRemove(n);
        if (v < 0.0) { W[W_ST + S_negSettle + g] += -v; }
      } else if (dTries(n) > 600u) {
        // nowhere to put it (the bowl is full to the brim): count it as spilled
        W[W_ST + S_spilledVol] += v;
        W[W_ST + S_debtSpilled] += v;
        dRemove(n);
      } else {
        n++;
      }
    }
    storageBarrier();
    workgroupBarrier();
  }
}

// ---------------------------------------------------------------- smoothing residuals (Sim._settleResidual)
@compute @workgroup_size(1)
fn ledgerResiduals() {
  for (var g = 0u; g < 4u; g++) {
    let r = W[W_ST + S_residual + g];
    W[W_ST + S_residual + g] = 0.0;
    if (abs(r) < 1e-12) { continue; }
    settle(g, vcenter(g, vijk(g, X[X_STU + SU_residualIdx + g])), r);
  }
}

// ---------------------------------------------------------------- bookkeeping
// ice owed but not yet re-deposited (Sim.debtVolume, plus what is still in the bins)
@compute @workgroup_size(256)
fn ledgerStats(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  var s = 0.0;
  for (var b = t; b < NBINS; b += 256u) { s += f32(atomicLoad(&A[A_BINS + b])) * VOL_UNIT; }
  for (var e = t; e < nDebts(); e += 256u) { s += dVol(e); }
  let tot = wgSum(t, s);
  if (t == 0u) {
    var v = tot + W[W_ST + S_looseOwed];
    for (var g = 0u; g < 4u; g++) { v -= W[W_ST + S_credit + g] + W[W_ST + S_negSettle + g]; }
    W[W_ST + S_debtVol] = v;
  }
}
`;
