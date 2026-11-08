import { smin } from './sdf.js';

/**
 * A narrow-band signed distance volume plus an RGBA colour volume
 * (rgb = linear albedo, a = speck/chip amount). Voxel centres sit at
 * origin + (i + 0.5) * cell, which matches GL 3D-texture sampling with uvw = local / size.
 */
export class SdfGrid {
  constructor(origin, cell, dims, bandCells = 4) {
    this.ox = origin[0];
    this.oy = origin[1];
    this.oz = origin[2];
    this.cell = cell;
    this.residual = 0;
    this.residualIdx = 0;
    this.nx = dims[0];
    this.ny = dims[1];
    this.nz = dims[2];
    this.n = this.nx * this.ny * this.nz;
    this.band = bandCells * cell;
    this.data = new Float32Array(this.n).fill(this.band);
    this.color = new Uint8Array(this.n * 4);
    this.bytes = new Uint8Array(this.n);
    // voxels edited this frame, smoothed afterwards (a cheap stand-in for surface tension)
    this.touched = new Uint8Array(this.n);
    this.touchedList = new Int32Array(this.n);
    this.touchedVal = new Float32Array(this.n);
    this.nTouched = 0;
    // bumped on every edit (setting `dirty`), so other consumers (the GPU particle solver) can tell
    // when to re-upload; `dirty` itself is cleared by the renderer
    this.version = 0;
    this.dirty = true;
    this.colorDirty = true;
    this.sx = this.nx * cell;
    this.sy = this.ny * cell;
    this.sz = this.nz * cell;
  }

  get dirty() {
    return this._dirty;
  }

  set dirty(v) {
    this._dirty = v;
    if (v) this.version++;
  }

  clear() {
    this.data.fill(this.band);
    this.dirty = true;
  }

  contains(x, y, z, margin = 0) {
    return (
      x >= this.ox + margin &&
      y >= this.oy + margin &&
      z >= this.oz + margin &&
      x <= this.ox + this.sx - margin &&
      y <= this.oy + this.sy - margin &&
      z <= this.oz + this.sz - margin
    );
  }

  sample(x, y, z) {
    const c = this.cell;
    let fx = (x - this.ox) / c - 0.5;
    let fy = (y - this.oy) / c - 0.5;
    let fz = (z - this.oz) / c - 0.5;
    const nx = this.nx;
    const ny = this.ny;
    const nz = this.nz;
    if (fx < -0.5 || fy < -0.5 || fz < -0.5 || fx > nx - 0.5 || fy > ny - 0.5 || fz > nz - 0.5) {
      return this.band;
    }
    if (fx < 0) fx = 0;
    else if (fx > nx - 1.000001) fx = nx - 1.000001;
    if (fy < 0) fy = 0;
    else if (fy > ny - 1.000001) fy = ny - 1.000001;
    if (fz < 0) fz = 0;
    else if (fz > nz - 1.000001) fz = nz - 1.000001;
    const i = fx | 0;
    const j = fy | 0;
    const k = fz | 0;
    const tx = fx - i;
    const ty = fy - j;
    const tz = fz - k;
    const d = this.data;
    const sxy = nx * ny;
    const o = i + nx * j + sxy * k;
    const c00 = d[o] + (d[o + 1] - d[o]) * tx;
    const c10 = d[o + nx] + (d[o + nx + 1] - d[o + nx]) * tx;
    const c01 = d[o + sxy] + (d[o + sxy + 1] - d[o + sxy]) * tx;
    const c11 = d[o + sxy + nx] + (d[o + sxy + nx + 1] - d[o + sxy + nx]) * tx;
    const c0 = c00 + (c10 - c00) * ty;
    const c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
  }

  grad(x, y, z, out) {
    const e = this.cell * 0.5;
    const gx = this.sample(x + e, y, z) - this.sample(x - e, y, z);
    const gy = this.sample(x, y + e, z) - this.sample(x, y - e, z);
    const gz = this.sample(x, y, z + e) - this.sample(x, y, z - e);
    const l = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (l < 1e-9) {
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
    } else {
      out[0] = gx / l;
      out[1] = gy / l;
      out[2] = gz / l;
    }
    return out;
  }

  /** Index of the voxel containing a point, or -1. */
  voxelAt(x, y, z) {
    const i = Math.floor((x - this.ox) / this.cell);
    const j = Math.floor((y - this.oy) / this.cell);
    const k = Math.floor((z - this.oz) / this.cell);
    if (i < 0 || j < 0 || k < 0 || i >= this.nx || j >= this.ny || k >= this.nz) return -1;
    return i + this.nx * (j + this.ny * k);
  }

  /**
   * Write smoothed values T[0..n) into voxels L[0..n), shifting them uniformly so the enclosed
   * volume is unchanged (volume-preserving curvature flow ~ surface tension at constant mass).
   */
  _writeVolumePreserving(L, T, n) {
    const d = this.data;
    const c = this.cell;
    const band = this.band;
    let vb = 0;
    for (let q = 0; q < n; q++) vb += this.occ(d[L[q]]);
    // find the uniform shift that restores the volume (values are T - shift and inside is
    // negative, so a larger shift means more volume). Volume is monotone and piecewise
    // linear in the shift: Newton steps kept inside a shrinking bracket converge in a few
    // passes. Limited to one cell; anything left is reported as residual for the ledger.
    let lo = -c;
    let hi = c;
    let shift = 0;
    const tol = 0.02;
    for (let it = 0; it < 8; it++) {
      let va = 0;
      let area = 0;
      for (let q = 0; q < n; q++) {
        let v = T[q] - shift;
        if (v > band) v = band;
        else if (v < -band) v = -band;
        const o = 0.5 - v / c;
        if (o >= 1) va += 1;
        else if (o > 0) {
          va += o;
          area++;
        }
      }
      const err = vb - va;
      if (Math.abs(err) < tol) break;
      if (err > 0) lo = shift;
      else hi = shift;
      let next = area > 0 ? shift + (err * c) / area : NaN;
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      shift = next;
    }
    let va = 0;
    for (let q = 0; q < n; q++) {
      let v = T[q] - shift;
      if (v > band) v = band;
      else if (v < -band) v = -band;
      d[L[q]] = v;
      va += this.occ(v);
    }
    // volume this pass lost (+) or gained (-), m³, for the caller's ledger
    this.residual += (vb - va) * c * c * c;
    if (n) this.residualIdx = L[0];
  }

  /** Volume the smoothing passes could not restore since the last call, and where. */
  takeResidual() {
    const r = this.residual;
    this.residual = 0;
    return r;
  }

  touch(idx) {
    if (!this.touched[idx]) {
      this.touched[idx] = 1;
      this.touchedList[this.nTouched++] = idx;
    }
  }

  /** One Jacobi Laplacian pass over this frame's edited voxels (and clear the list). */
  smoothTouched(strength) {
    const n = this.nTouched;
    if (!n) return;
    const L = this.touchedList;
    const T = this.touchedVal;
    const d = this.data;
    const nx = this.nx;
    const ny = this.ny;
    const nz = this.nz;
    const sxy = nx * ny;
    if (strength > 0) {
      for (let q = 0; q < n; q++) {
        const idx = L[q];
        const k = (idx / sxy) | 0;
        const r = idx - k * sxy;
        const j = (r / nx) | 0;
        const i = r - j * nx;
        const avg =
          (d[i > 0 ? idx - 1 : idx] +
            d[i < nx - 1 ? idx + 1 : idx] +
            d[j > 0 ? idx - nx : idx] +
            d[j < ny - 1 ? idx + nx : idx] +
            d[k > 0 ? idx - sxy : idx] +
            d[k < nz - 1 ? idx + sxy : idx]) /
          6;
        T[q] = d[idx] + (avg - d[idx]) * strength;
      }
      this._writeVolumePreserving(L, T, n);
      this.dirty = true;
    }
    // remember the edited region so it can keep settling for a while
    const b = this.settleBox || (this.settleBox = [0, 0, 0, 0, 0, 0]);
    if (!this.settleFrames) b.splice(0, 6, Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity);
    for (let q = 0; q < n; q++) {
      const idx = L[q];
      this.touched[idx] = 0;
      const k = (idx / sxy) | 0;
      const r = idx - k * sxy;
      const j = (r / nx) | 0;
      const i = r - j * nx;
      if (i < b[0]) b[0] = i;
      if (j < b[1]) b[1] = j;
      if (k < b[2]) b[2] = k;
      if (i > b[3]) b[3] = i;
      if (j > b[4]) b[4] = j;
      if (k > b[5]) b[5] = k;
    }
    this.nTouched = 0;
    this.settleFrames = 30;
  }

  /**
   * Keep smoothing the near-surface band of the recently edited region for a few frames
   * after the last edit (the scoop "settles" into a rounded shape instead of staying pitted).
   */
  settle(strength, consume = true) {
    if (!this.settleFrames || strength <= 0) return;
    if (consume) this.settleFrames--;
    const b = this.settleBox;
    const nx = this.nx;
    const ny = this.ny;
    const nz = this.nz;
    const sxy = nx * ny;
    const i0 = Math.max(0, b[0] - 2);
    const i1 = Math.min(nx - 1, b[3] + 2);
    const j0 = Math.max(0, b[1] - 2);
    const j1 = Math.min(ny - 1, b[4] + 2);
    const k0 = Math.max(0, b[2] - 2);
    const k1 = Math.min(nz - 1, b[5] + 2);
    const d = this.data;
    const L = this.touchedList;
    const T = this.touchedVal;
    const lim = 2 * this.cell;
    let n = 0;
    for (let k = k0; k <= k1; k++)
      for (let j = j0; j <= j1; j++) {
        let idx = i0 + nx * (j + ny * k);
        for (let i = i0; i <= i1; i++, idx++) {
          const v = d[idx];
          if (v > lim || v < -lim) continue;
          const avg =
            (d[i > 0 ? idx - 1 : idx] +
              d[i < nx - 1 ? idx + 1 : idx] +
              d[j > 0 ? idx - nx : idx] +
              d[j < ny - 1 ? idx + nx : idx] +
              d[k > 0 ? idx - sxy : idx] +
              d[k < nz - 1 ? idx + sxy : idx]) /
            6;
          L[n] = idx;
          T[n++] = v + (avg - v) * strength * 0.5;
        }
      }
    // the touched list is free outside smoothTouched, so it doubles as scratch space here
    this._writeVolumePreserving(L, T, n);
    if (n) this.dirty = true;
  }

  occ(d) {
    const o = 0.5 - d / this.cell;
    return o < 0 ? 0 : o > 1 ? 1 : o;
  }

  volume() {
    let s = 0;
    const d = this.data;
    const inv = 1 / this.cell;
    for (let i = 0; i < this.n; i++) {
      const o = 0.5 - d[i] * inv;
      if (o > 0) s += o > 1 ? 1 : o;
    }
    return s * this.cell * this.cell * this.cell;
  }

  setColor(idx, rgba) {
    const o = idx * 4;
    this.color[o] = rgba[0];
    this.color[o + 1] = rgba[1];
    this.color[o + 2] = rgba[2];
    this.color[o + 3] = rgba[3];
  }

  /** Blend a colour into a voxel (weight 0..1). */
  blendColor(idx, rgba, w) {
    const o = idx * 4;
    const c = this.color;
    c[o] += (rgba[0] - c[o]) * w;
    c[o + 1] += (rgba[1] - c[o + 1]) * w;
    c[o + 2] += (rgba[2] - c[o + 2]) * w;
    c[o + 3] += (rgba[3] - c[o + 3]) * w;
  }

  fillColor(rgba) {
    for (let i = 0; i < this.n; i++) this.setColor(i, rgba);
    this.colorDirty = true;
  }

  /** Union a sphere into the field with a smooth-min blend. Returns true if anything changed. */
  stampSphere(cx, cy, cz, R, k, rgba) {
    const c = this.cell;
    const reach = R + k + c;
    const i0 = Math.max(0, Math.floor((cx - reach - this.ox) / c));
    const i1 = Math.min(this.nx - 1, Math.floor((cx + reach - this.ox) / c));
    const j0 = Math.max(0, Math.floor((cy - reach - this.oy) / c));
    const j1 = Math.min(this.ny - 1, Math.floor((cy + reach - this.oy) / c));
    const k0 = Math.max(0, Math.floor((cz - reach - this.oz) / c));
    const k1 = Math.min(this.nz - 1, Math.floor((cz + reach - this.oz) / c));
    if (i0 > i1 || j0 > j1 || k0 > k1) return 0;
    const band = this.band;
    let changed = false;
    let added = 0;
    for (let kk = k0; kk <= k1; kk++) {
      const z = this.oz + (kk + 0.5) * c - cz;
      for (let jj = j0; jj <= j1; jj++) {
        const y = this.oy + (jj + 0.5) * c - cy;
        let idx = i0 + this.nx * (jj + this.ny * kk);
        for (let ii = i0; ii <= i1; ii++, idx++) {
          const x = this.ox + (ii + 0.5) * c - cx;
          const ds = Math.sqrt(x * x + y * y + z * z) - R;
          if (ds > reach) continue;
          const v = this.data[idx];
          let nv = smin(v, ds, k);
          if (nv < -band) nv = -band;
          if (nv < v) {
            added += this.occ(nv) - this.occ(v);
            this.data[idx] = nv;
            this.touch(idx);
            changed = true;
            if (rgba && ds < c) this.blendColor(idx, rgba, v > 0 ? 1 : 0.5);
          }
        }
      }
    }
    if (changed) {
      this.dirty = true;
      if (rgba) this.colorDirty = true;
    }
    return added * c * c * c;
  }

  /**
   * Rebuild a proper signed distance field from the current zero level set (fast sweeping
   * Eikonal solver). Copying, carving and stamping leave steep, non-distance values that
   * make the ray marcher overshoot and the smoothing stall. The volume change it causes is
   * added to `residual` for the caller's ledger.
   */
  redistance() {
    const { nx, ny, nz, cell: c, band } = this;
    const n = this.n;
    const d = this.data;
    const sxy = nx * ny;
    const before = this.volume();
    const u = this._ud || (this._ud = new Float32Array(n));
    const fixed = this._uf || (this._uf = new Uint8Array(n));
    const BIG = 1e9;
    // 1. voxels next to the surface keep their interpolated distance to it
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++) {
        let idx = nx * (j + ny * k);
        for (let i = 0; i < nx; i++, idx++) {
          const v = d[idx];
          let best = BIG;
          const neg = v < 0;
          for (let m = 0; m < 6; m++) {
            let w;
            if (m === 0) { if (i === 0) continue; w = d[idx - 1]; }
            else if (m === 1) { if (i === nx - 1) continue; w = d[idx + 1]; }
            else if (m === 2) { if (j === 0) continue; w = d[idx - nx]; }
            else if (m === 3) { if (j === ny - 1) continue; w = d[idx + nx]; }
            else if (m === 4) { if (k === 0) continue; w = d[idx - sxy]; }
            else { if (k === nz - 1) continue; w = d[idx + sxy]; }
            if (neg !== w < 0) {
              const t = (Math.abs(v) / (Math.abs(v - w) || 1e-12)) * c;
              if (t < best) best = t;
            }
          }
          if (best < BIG) {
            u[idx] = best;
            fixed[idx] = 1;
          } else {
            u[idx] = BIG;
            fixed[idx] = 0;
          }
        }
      }
    // 2. fast sweeping in the 8 diagonal orders
    for (let sweep = 0; sweep < 8; sweep++) {
      const di = sweep & 1 ? -1 : 1;
      const dj = sweep & 2 ? -1 : 1;
      const dk = sweep & 4 ? -1 : 1;
      for (let kk = 0; kk < nz; kk++) {
        const k = dk > 0 ? kk : nz - 1 - kk;
        for (let jj = 0; jj < ny; jj++) {
          const j = dj > 0 ? jj : ny - 1 - jj;
          for (let ii = 0; ii < nx; ii++) {
            const i = di > 0 ? ii : nx - 1 - ii;
            const idx = i + nx * (j + ny * k);
            if (fixed[idx]) continue;
            let a = Math.min(i > 0 ? u[idx - 1] : BIG, i < nx - 1 ? u[idx + 1] : BIG);
            let b = Math.min(j > 0 ? u[idx - nx] : BIG, j < ny - 1 ? u[idx + nx] : BIG);
            let e = Math.min(k > 0 ? u[idx - sxy] : BIG, k < nz - 1 ? u[idx + sxy] : BIG);
            // sort a <= b <= e
            if (a > b) [a, b] = [b, a];
            if (b > e) [b, e] = [e, b];
            if (a > b) [a, b] = [b, a];
            if (a >= BIG) continue;
            let x = a + c;
            if (x > b) {
              const s2 = a + b;
              x = 0.5 * (s2 + Math.sqrt(Math.max(0, s2 * s2 - 2 * (a * a + b * b - c * c))));
              if (x > e) {
                const s3 = a + b + e;
                x = (s3 + Math.sqrt(Math.max(0, s3 * s3 - 3 * (a * a + b * b + e * e - c * c)))) / 3;
              }
            }
            if (x < u[idx]) u[idx] = x;
          }
        }
      }
    }
    // 3. back to signed, band-limited values
    for (let q = 0; q < n; q++) {
      let v = u[q] > band ? band : u[q];
      d[q] = d[q] < 0 ? -v : v;
    }
    this.residual += before - this.volume();
    this.dirty = true;
  }

  /** Sharp CSG Cut of a sphere. Returns the removed volume (m³). */
  carveSphere(cx, cy, cz, R) {
    const c = this.cell;
    const reach = R + 2 * c;
    const i0 = Math.max(0, Math.floor((cx - reach - this.ox) / c));
    const i1 = Math.min(this.nx - 1, Math.floor((cx + reach - this.ox) / c));
    const j0 = Math.max(0, Math.floor((cy - reach - this.oy) / c));
    const j1 = Math.min(this.ny - 1, Math.floor((cy + reach - this.oy) / c));
    const k0 = Math.max(0, Math.floor((cz - reach - this.oz) / c));
    const k1 = Math.min(this.nz - 1, Math.floor((cz + reach - this.oz) / c));
    if (i0 > i1 || j0 > j1 || k0 > k1) return 0;
    const band = this.band;
    let removed = 0;
    for (let kk = k0; kk <= k1; kk++) {
      const z = this.oz + (kk + 0.5) * c - cz;
      for (let jj = j0; jj <= j1; jj++) {
        const y = this.oy + (jj + 0.5) * c - cy;
        let idx = i0 + this.nx * (jj + this.ny * kk);
        for (let ii = i0; ii <= i1; ii++, idx++) {
          const x = this.ox + (ii + 0.5) * c - cx;
          const v = this.data[idx];
          let nv = Math.max(v, R - Math.sqrt(x * x + y * y + z * z));
          if (nv > band) nv = band;
          if (nv > v) {
            removed += this.occ(v) - this.occ(nv);
            this.data[idx] = nv;
            this.touch(idx);
          }
        }
      }
    }
    if (removed > 0) this.dirty = true;
    return removed * c * c * c;
  }

  /** Quantise into the 8-bit texture: [-band, band] -> [0, 255]. */
  pack() {
    const s = 127.5 / this.band;
    const d = this.data;
    const b = this.bytes;
    for (let i = 0; i < this.n; i++) {
      const q = d[i] * s + 127.5;
      b[i] = q < 0 ? 0 : q > 255 ? 255 : q;
    }
  }
}
