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
    this.dirty = true;
    this.colorDirty = true;
    this.sx = this.nx * cell;
    this.sy = this.ny * cell;
    this.sz = this.nz * cell;
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
    let vb = 0;
    let va = 0;
    let area = 0;
    for (let q = 0; q < n; q++) {
      vb += this.occ(d[L[q]]);
      const o = this.occ(T[q]);
      va += o;
      if (o > 0 && o < 1) area++;
    }
    const shift = area > 0 ? ((vb - va) * c) / area : 0;
    for (let q = 0; q < n; q++) d[L[q]] = T[q] - shift;
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
  settle(strength) {
    if (!this.settleFrames || strength <= 0) return;
    this.settleFrames--;
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
    if (i0 > i1 || j0 > j1 || k0 > k1) return false;
    const band = this.band;
    let changed = false;
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
    return changed;
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
