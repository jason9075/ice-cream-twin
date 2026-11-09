//! A narrow-band signed distance volume plus an RGBA colour volume (rgb = linear albedo,
//! a = speck amount). Voxel centres sit at origin + (i + 0.5) * cell. Ported from the former JS `grid.js`
//! (distances stored as f32 like the JS `Float32Array`, arithmetic in f64).

use crate::sdf::smin;

/// JS `Uint8Array` store: ToUint8 (truncate, wrap mod 256).
#[inline]
fn to_u8(v: f64) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    (v.trunc() as i64).rem_euclid(256) as u8
}

/// `floor` for `smooth_touched` / `settle` without a keep-out.
pub const NO_FLOOR: &dyn Fn(usize) -> f64 = &|_| f64::NEG_INFINITY;

#[derive(Clone, Debug)]
pub struct SdfGrid {
    pub ox: f64,
    pub oy: f64,
    pub oz: f64,
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub n: usize,
    pub band: f64,
    pub data: Vec<f32>,
    pub color: Vec<u8>,
    pub bytes: Vec<u8>,
    touched: Vec<u8>,
    touched_list: Vec<usize>,
    touched_val: Vec<f32>,
    /// per entry of the list being written: the lowest value it may take (see `smooth_touched`)
    touched_floor: Vec<f32>,
    pub n_touched: usize,
    pub dirty: bool,
    pub color_dirty: bool,
    pub sx: f64,
    pub sy: f64,
    pub sz: f64,
    pub residual: f64,
    pub residual_idx: usize,
    /// m³ the next volume-preserving write should add (negative: take off) on top of keeping the volume;
    /// what it can't goes to `residual`
    pub volume_adjust: f64,
    pub settle_frames: u32,
    settle_box: [i64; 6],
    ud: Vec<f32>,
    uf: Vec<u8>,
}

impl SdfGrid {
    pub fn new(origin: [f64; 3], cell: f64, dims: [usize; 3], band_cells: f64) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        let band = band_cells * cell;
        SdfGrid {
            ox: origin[0],
            oy: origin[1],
            oz: origin[2],
            cell,
            nx: dims[0],
            ny: dims[1],
            nz: dims[2],
            n,
            band,
            data: vec![band as f32; n],
            color: vec![0; n * 4],
            bytes: vec![0; n],
            touched: vec![0; n],
            touched_list: vec![0; n],
            touched_val: vec![0.0; n],
            touched_floor: vec![f32::NEG_INFINITY; n],
            n_touched: 0,
            dirty: true,
            color_dirty: true,
            sx: dims[0] as f64 * cell,
            sy: dims[1] as f64 * cell,
            sz: dims[2] as f64 * cell,
            residual: 0.0,
            volume_adjust: 0.0,
            residual_idx: 0,
            settle_frames: 0,
            settle_box: [0; 6],
            ud: Vec::new(),
            uf: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        let b = self.band as f32;
        self.data.iter_mut().for_each(|v| *v = b);
        self.dirty = true;
    }

    pub fn contains(&self, x: f64, y: f64, z: f64, margin: f64) -> bool {
        x >= self.ox + margin
            && y >= self.oy + margin
            && z >= self.oz + margin
            && x <= self.ox + self.sx - margin
            && y <= self.oy + self.sy - margin
            && z <= self.oz + self.sz - margin
    }

    /// Trilinear sample; `band` outside the grid.
    pub fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let c = self.cell;
        let mut fx = (x - self.ox) / c - 0.5;
        let mut fy = (y - self.oy) / c - 0.5;
        let mut fz = (z - self.oz) / c - 0.5;
        let (nx, ny, nz) = (self.nx as f64, self.ny as f64, self.nz as f64);
        if fx < -0.5 || fy < -0.5 || fz < -0.5 || fx > nx - 0.5 || fy > ny - 0.5 || fz > nz - 0.5 {
            return self.band;
        }
        if fx < 0.0 {
            fx = 0.0;
        } else if fx > nx - 1.000001 {
            fx = nx - 1.000001;
        }
        if fy < 0.0 {
            fy = 0.0;
        } else if fy > ny - 1.000001 {
            fy = ny - 1.000001;
        }
        if fz < 0.0 {
            fz = 0.0;
        } else if fz > nz - 1.000001 {
            fz = nz - 1.000001;
        }
        let i = fx as usize;
        let j = fy as usize;
        let k = fz as usize;
        let tx = fx - i as f64;
        let ty = fy - j as f64;
        let tz = fz - k as f64;
        let d = &self.data;
        let nxu = self.nx;
        let sxy = self.nx * self.ny;
        let o = i + nxu * j + sxy * k;
        let g = |q: usize| d[q] as f64;
        let c00 = g(o) + (g(o + 1) - g(o)) * tx;
        let c10 = g(o + nxu) + (g(o + nxu + 1) - g(o + nxu)) * tx;
        let c01 = g(o + sxy) + (g(o + sxy + 1) - g(o + sxy)) * tx;
        let c11 = g(o + sxy + nxu) + (g(o + sxy + nxu + 1) - g(o + sxy + nxu)) * tx;
        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;
        c0 + (c1 - c0) * tz
    }

    pub fn grad(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let e = self.cell * 0.5;
        let gx = self.sample(x + e, y, z) - self.sample(x - e, y, z);
        let gy = self.sample(x, y + e, z) - self.sample(x, y - e, z);
        let gz = self.sample(x, y, z + e) - self.sample(x, y, z - e);
        let l = (gx * gx + gy * gy + gz * gz).sqrt();
        if l < 1e-9 {
            [0.0, 1.0, 0.0]
        } else {
            [gx / l, gy / l, gz / l]
        }
    }

    /// Index of the voxel containing a point.
    pub fn voxel_at(&self, x: f64, y: f64, z: f64) -> Option<usize> {
        let i = ((x - self.ox) / self.cell).floor();
        let j = ((y - self.oy) / self.cell).floor();
        let k = ((z - self.oz) / self.cell).floor();
        if i < 0.0 || j < 0.0 || k < 0.0 || i >= self.nx as f64 || j >= self.ny as f64 || k >= self.nz as f64 {
            return None;
        }
        Some(i as usize + self.nx * (j as usize + self.ny * k as usize))
    }

    /// Voxel centre of a linear index.
    pub fn center(&self, idx: usize) -> [f64; 3] {
        let i = idx % self.nx;
        let j = (idx / self.nx) % self.ny;
        let k = idx / (self.nx * self.ny);
        [
            self.ox + (i as f64 + 0.5) * self.cell,
            self.oy + (j as f64 + 0.5) * self.cell,
            self.oz + (k as f64 + 0.5) * self.cell,
        ]
    }

    #[inline]
    pub fn occ(&self, d: f64) -> f64 {
        let o = 0.5 - d / self.cell;
        if o < 0.0 {
            0.0
        } else if o > 1.0 {
            1.0
        } else {
            o
        }
    }

    pub fn volume(&self) -> f64 {
        let inv = 1.0 / self.cell;
        let mut s = 0.0;
        for &v in &self.data {
            let o = 0.5 - v as f64 * inv;
            if o > 0.0 {
                s += if o > 1.0 { 1.0 } else { o };
            }
        }
        s * self.cell * self.cell * self.cell
    }

    /// Write smoothed values T[0..n) into voxels L[0..n), shifting them uniformly so the
    /// enclosed volume is unchanged; what can't be restored is added to `residual`.
    /// Writes the first n entries of the touched list (`touched_val`, at least `touched_floor`), shifted
    /// together so the occupied volume stays the same; what can't be kept goes to `residual`.
    fn write_volume_preserving(&mut self, n: usize) {
        let c = self.cell;
        let band = self.band;
        let mut vb = 0.0;
        for q in 0..n {
            vb += self.occ(self.data[self.touched_list[q]] as f64);
        }
        vb += std::mem::take(&mut self.volume_adjust) / (c * c * c);
        let mut lo = -c;
        let mut hi = c;
        let mut shift = 0.0f64;
        let tol = 0.02;
        for _ in 0..8 {
            let mut va = 0.0;
            let mut area = 0u32;
            for q in 0..n {
                let mut v = (self.touched_val[q] as f64 - shift).max(self.touched_floor[q] as f64);
                if v > band {
                    v = band;
                } else if v < -band {
                    v = -band;
                }
                let o = 0.5 - v / c;
                if o >= 1.0 {
                    va += 1.0;
                } else if o > 0.0 {
                    va += o;
                    area += 1;
                }
            }
            let err = vb - va;
            if err.abs() < tol {
                break;
            }
            if err > 0.0 {
                lo = shift;
            } else {
                hi = shift;
            }
            let mut next = if area > 0 { shift + (err * c) / area as f64 } else { f64::NAN };
            if !(next > lo && next < hi) {
                next = 0.5 * (lo + hi);
            }
            shift = next;
        }
        let mut va = 0.0;
        for q in 0..n {
            let mut v = (self.touched_val[q] as f64 - shift).max(self.touched_floor[q] as f64);
            if v > band {
                v = band;
            } else if v < -band {
                v = -band;
            }
            let idx = self.touched_list[q];
            self.data[idx] = v as f32;
            va += self.occ(v);
        }
        self.residual += (vb - va) * c * c * c;
        if n > 0 {
            self.residual_idx = self.touched_list[0];
        }
    }

    /// Volume the smoothing passes could not restore since the last call.
    pub fn take_residual(&mut self) -> f64 {
        let r = self.residual;
        self.residual = 0.0;
        r
    }

    #[inline]
    /// Which voxels were edited since the last smoothing pass (1 = edited).
    pub fn touched_flags(&self) -> &[u8] {
        &self.touched
    }

    pub fn touch(&mut self, idx: usize) {
        if self.touched[idx] == 0 {
            self.touched[idx] = 1;
            self.touched_list[self.n_touched] = idx;
            self.n_touched += 1;
        }
    }

    #[inline]
    fn neighbour_avg(&self, idx: usize) -> f64 {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let sxy = nx * ny;
        let k = idx / sxy;
        let r = idx - k * sxy;
        let j = r / nx;
        let i = r - j * nx;
        let d = &self.data;
        (d[if i > 0 { idx - 1 } else { idx }] as f64
            + d[if i < nx - 1 { idx + 1 } else { idx }] as f64
            + d[if j > 0 { idx - nx } else { idx }] as f64
            + d[if j < ny - 1 { idx + nx } else { idx }] as f64
            + d[if k > 0 { idx - sxy } else { idx }] as f64
            + d[if k < nz - 1 { idx + sxy } else { idx }] as f64)
            / 6.0
    }

    /// One Jacobi Laplacian pass over this frame's edited voxels (and clear the list). `floor(idx)` is the
    /// lowest distance a voxel may end up with: the bowls pass the spoon's keep-out (what carving cuts), so
    /// smoothing never pushes ice back into the spoon (which would cut it again: a stationary spoon would
    /// keep pumping ice into the cup); `NO_FLOOR` for none.
    pub fn smooth_touched(&mut self, strength: f64, floor: &dyn Fn(usize) -> f64) {
        let n = self.n_touched;
        if n == 0 {
            return;
        }
        let sxy = self.nx * self.ny;
        if strength > 0.0 {
            for q in 0..n {
                let idx = self.touched_list[q];
                let avg = self.neighbour_avg(idx);
                let v = self.data[idx] as f64;
                self.touched_val[q] = (v + (avg - v) * strength) as f32;
                self.touched_floor[q] = floor(idx) as f32;
            }
            self.write_volume_preserving(n);
            self.dirty = true;
        }
        if self.settle_frames == 0 {
            self.settle_box = [i64::MAX, i64::MAX, i64::MAX, i64::MIN, i64::MIN, i64::MIN];
        }
        for q in 0..n {
            let idx = self.touched_list[q];
            self.touched[idx] = 0;
            let k = (idx / sxy) as i64;
            let r = idx - (k as usize) * sxy;
            let j = (r / self.nx) as i64;
            let i = (r % self.nx) as i64;
            let b = &mut self.settle_box;
            b[0] = b[0].min(i);
            b[1] = b[1].min(j);
            b[2] = b[2].min(k);
            b[3] = b[3].max(i);
            b[4] = b[4].max(j);
            b[5] = b[5].max(k);
        }
        self.n_touched = 0;
        self.settle_frames = 30;
    }

    /// Keep smoothing the near-surface band of the recently edited region for a few frames (`floor` as in
    /// `smooth_touched`).
    pub fn settle(&mut self, strength: f64, consume: bool, floor: &dyn Fn(usize) -> f64) {
        if self.settle_frames == 0 || strength <= 0.0 {
            return;
        }
        if consume {
            self.settle_frames -= 1;
        }
        let b = self.settle_box;
        let (nx, ny, nz) = (self.nx as i64, self.ny as i64, self.nz as i64);
        let i0 = 0.max(b[0] - 2);
        let i1 = (nx - 1).min(b[3] + 2);
        let j0 = 0.max(b[1] - 2);
        let j1 = (ny - 1).min(b[4] + 2);
        let k0 = 0.max(b[2] - 2);
        let k1 = (nz - 1).min(b[5] + 2);
        let lim = 2.0 * self.cell;
        let mut n = 0usize;
        let mut k = k0;
        while k <= k1 {
            let mut j = j0;
            while j <= j1 {
                let mut idx = (i0 + nx * (j + ny * k)) as usize;
                let mut i = i0;
                while i <= i1 {
                    let v = self.data[idx] as f64;
                    if !(v > lim || v < -lim) {
                        let avg = self.neighbour_avg(idx);
                        self.touched_list[n] = idx;
                        self.touched_val[n] = (v + (avg - v) * strength * 0.5) as f32;
                        self.touched_floor[n] = floor(idx) as f32;
                        n += 1;
                    }
                    i += 1;
                    idx += 1;
                }
                j += 1;
            }
            k += 1;
        }
        self.write_volume_preserving(n);
        if n > 0 {
            self.dirty = true;
        }
    }

    pub fn set_color(&mut self, idx: usize, rgba: [u8; 4]) {
        self.color[idx * 4..idx * 4 + 4].copy_from_slice(&rgba);
    }

    /// Blend a colour into a voxel (weight 0..1), with Uint8Array store semantics.
    pub fn blend_color(&mut self, idx: usize, rgba: [u8; 4], w: f64) {
        let o = idx * 4;
        for q in 0..4 {
            let c = self.color[o + q] as f64;
            self.color[o + q] = to_u8(c + (rgba[q] as f64 - c) * w);
        }
    }

    pub fn fill_color(&mut self, rgba: [u8; 4]) {
        for i in 0..self.n {
            self.set_color(i, rgba);
        }
        self.color_dirty = true;
    }

    fn range(&self, cx: f64, cy: f64, cz: f64, reach: f64) -> Option<[usize; 6]> {
        let c = self.cell;
        let i0 = 0f64.max(((cx - reach - self.ox) / c).floor());
        let i1 = (self.nx as f64 - 1.0).min(((cx + reach - self.ox) / c).floor());
        let j0 = 0f64.max(((cy - reach - self.oy) / c).floor());
        let j1 = (self.ny as f64 - 1.0).min(((cy + reach - self.oy) / c).floor());
        let k0 = 0f64.max(((cz - reach - self.oz) / c).floor());
        let k1 = (self.nz as f64 - 1.0).min(((cz + reach - self.oz) / c).floor());
        if i0 > i1 || j0 > j1 || k0 > k1 {
            return None;
        }
        Some([i0 as usize, i1 as usize, j0 as usize, j1 as usize, k0 as usize, k1 as usize])
    }

    /// Union a sphere into the field with a smooth-min blend. Returns the added volume (m³).
    pub fn stamp_sphere(&mut self, cx: f64, cy: f64, cz: f64, r: f64, k: f64, rgba: Option<[u8; 4]>) -> f64 {
        let c = self.cell;
        let reach = r + k + c;
        let Some([i0, i1, j0, j1, k0, k1]) = self.range(cx, cy, cz, reach) else { return 0.0 };
        let band = self.band;
        let mut changed = false;
        let mut added = 0.0;
        for kk in k0..=k1 {
            let z = self.oz + (kk as f64 + 0.5) * c - cz;
            for jj in j0..=j1 {
                let y = self.oy + (jj as f64 + 0.5) * c - cy;
                let mut idx = i0 + self.nx * (jj + self.ny * kk);
                for ii in i0..=i1 {
                    let x = self.ox + (ii as f64 + 0.5) * c - cx;
                    let ds = (x * x + y * y + z * z).sqrt() - r;
                    if ds <= reach {
                        let v = self.data[idx] as f64;
                        let mut nv = smin(v, ds, k);
                        if nv < -band {
                            nv = -band;
                        }
                        if nv < v {
                            let nvf = nv as f32;
                            added += self.occ(nv) - self.occ(v);
                            self.data[idx] = nvf;
                            self.touch(idx);
                            changed = true;
                            if let Some(col) = rgba {
                                if ds < c {
                                    self.blend_color(idx, col, if v > 0.0 { 1.0 } else { 0.5 });
                                }
                            }
                        }
                    }
                    idx += 1;
                }
            }
        }
        if changed {
            self.dirty = true;
            if rgba.is_some() {
                self.color_dirty = true;
            }
        }
        added * c * c * c
    }

    /// Sharp CSG Cut of a sphere. Returns the removed volume (m³).
    pub fn carve_sphere(&mut self, cx: f64, cy: f64, cz: f64, r: f64) -> f64 {
        let c = self.cell;
        let reach = r + 2.0 * c;
        let Some([i0, i1, j0, j1, k0, k1]) = self.range(cx, cy, cz, reach) else { return 0.0 };
        let band = self.band;
        let mut removed = 0.0;
        for kk in k0..=k1 {
            let z = self.oz + (kk as f64 + 0.5) * c - cz;
            for jj in j0..=j1 {
                let y = self.oy + (jj as f64 + 0.5) * c - cy;
                let mut idx = i0 + self.nx * (jj + self.ny * kk);
                for ii in i0..=i1 {
                    let x = self.ox + (ii as f64 + 0.5) * c - cx;
                    let v = self.data[idx] as f64;
                    let mut nv = v.max(r - (x * x + y * y + z * z).sqrt());
                    if nv > band {
                        nv = band;
                    }
                    if nv > v {
                        removed += self.occ(v) - self.occ(nv);
                        self.data[idx] = nv as f32;
                        self.touch(idx);
                    }
                    idx += 1;
                }
            }
        }
        if removed > 0.0 {
            self.dirty = true;
        }
        removed * c * c * c
    }

    /// Rebuild a proper signed distance field from the current zero level set (fast
    /// sweeping Eikonal solver). The volume change is added to `residual`.
    pub fn redistance(&mut self) {
        let (nx, ny, nz, c, band) = (self.nx, self.ny, self.nz, self.cell, self.band);
        let n = self.n;
        let sxy = nx * ny;
        let before = self.volume();
        if self.ud.len() != n {
            self.ud = vec![0.0; n];
            self.uf = vec![0; n];
        }
        const BIG: f32 = 1e9;
        {
            let d = &self.data;
            let u = &mut self.ud;
            let fixed = &mut self.uf;
            for k in 0..nz {
                for j in 0..ny {
                    let mut idx = nx * (j + ny * k);
                    for i in 0..nx {
                        let v = d[idx] as f64;
                        let mut best = BIG as f64;
                        let neg = v < 0.0;
                        let mut test = |w: f64| {
                            if neg != (w < 0.0) {
                                let mut den = (v - w).abs();
                                if den == 0.0 {
                                    den = 1e-12;
                                }
                                let t = (v.abs() / den) * c;
                                if t < best {
                                    best = t;
                                }
                            }
                        };
                        if i > 0 {
                            test(d[idx - 1] as f64);
                        }
                        if i < nx - 1 {
                            test(d[idx + 1] as f64);
                        }
                        if j > 0 {
                            test(d[idx - nx] as f64);
                        }
                        if j < ny - 1 {
                            test(d[idx + nx] as f64);
                        }
                        if k > 0 {
                            test(d[idx - sxy] as f64);
                        }
                        if k < nz - 1 {
                            test(d[idx + sxy] as f64);
                        }
                        if best < BIG as f64 {
                            u[idx] = best as f32;
                            fixed[idx] = 1;
                        } else {
                            u[idx] = BIG;
                            fixed[idx] = 0;
                        }
                        idx += 1;
                    }
                }
            }
            let big = BIG as f64;
            for sweep in 0..8 {
                let rev_i = sweep & 1 != 0;
                let rev_j = sweep & 2 != 0;
                let rev_k = sweep & 4 != 0;
                for kk in 0..nz {
                    let k = if rev_k { nz - 1 - kk } else { kk };
                    for jj in 0..ny {
                        let j = if rev_j { ny - 1 - jj } else { jj };
                        for ii in 0..nx {
                            let i = if rev_i { nx - 1 - ii } else { ii };
                            let idx = i + nx * (j + ny * k);
                            if fixed[idx] != 0 {
                                continue;
                            }
                            let mut a = (if i > 0 { u[idx - 1] as f64 } else { big }).min(if i < nx - 1 { u[idx + 1] as f64 } else { big });
                            let mut b = (if j > 0 { u[idx - nx] as f64 } else { big }).min(if j < ny - 1 { u[idx + nx] as f64 } else { big });
                            let mut e = (if k > 0 { u[idx - sxy] as f64 } else { big }).min(if k < nz - 1 { u[idx + sxy] as f64 } else { big });
                            if a > b {
                                std::mem::swap(&mut a, &mut b);
                            }
                            if b > e {
                                std::mem::swap(&mut b, &mut e);
                            }
                            if a > b {
                                std::mem::swap(&mut a, &mut b);
                            }
                            if a >= big {
                                continue;
                            }
                            let mut x = a + c;
                            if x > b {
                                let s2 = a + b;
                                x = 0.5 * (s2 + 0f64.max(s2 * s2 - 2.0 * (a * a + b * b - c * c)).sqrt());
                                if x > e {
                                    let s3 = a + b + e;
                                    x = (s3 + 0f64.max(s3 * s3 - 3.0 * (a * a + b * b + e * e - c * c)).sqrt()) / 3.0;
                                }
                            }
                            if x < u[idx] as f64 {
                                u[idx] = x as f32;
                            }
                        }
                    }
                }
            }
        }
        for q in 0..n {
            let uq = self.ud[q] as f64;
            let v = if uq > band { band } else { uq };
            self.data[q] = if (self.data[q] as f64) < 0.0 { -v as f32 } else { v as f32 };
        }
        self.residual += before - self.volume();
        self.dirty = true;
    }

    /// Quantise into the 8-bit texture: [-band, band] -> [0, 255].
    pub fn pack(&mut self) {
        let s = 127.5 / self.band;
        for i in 0..self.n {
            let q = self.data[i] as f64 * s + 127.5;
            self.bytes[i] = to_u8(if q < 0.0 { 0.0 } else if q > 255.0 { 255.0 } else { q });
        }
    }
}
