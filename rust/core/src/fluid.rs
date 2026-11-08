//! Position Based Fluids (Macklin & Müller 2013) on the CPU for the "loose" ice cream.
//! Ported from the former JS `fluid.js` (f32 particle arrays like its typed arrays, f64 arithmetic).
//!
//! predict -> hash-grid neighbours -> [density constraint (repulsive only) + cohesion
//! -> collide with SDFs] x iterations -> velocity from positions -> friction -> XSPH viscosity

use crate::math::js_hypot;
use crate::params::FluidParams;

const MAXN: usize = 48;
/// m/s of outward velocity a contact may produce
const MAX_SEPARATION: f64 = 0.05;
const TABLE_BITS: u32 = 16;
const TABLE: usize = 1 << TABLE_BITS;
const MASK: i32 = (TABLE - 1) as i32;

pub const CONTACT_NONE: u8 = 0;
/// table, bowl, solid ice cream
pub const CONTACT_WORLD: u8 = 1;
/// spoon metal or the ice cream already on the spoon
pub const CONTACT_SPOON: u8 = 2;

/// Projects particle `i` out of every solid and records its contact (normal, surface velocity).
pub trait Collide {
    fn collide(&mut self, f: &mut Fluid, i: usize);
}

#[derive(Clone, Debug)]
pub struct Fluid {
    pub cap: usize,
    pub n: usize,
    pub pos: Vec<f32>,
    pub prev: Vec<f32>,
    pub vel: Vec<f32>,
    pub dp: Vec<f32>,
    pub lambda: Vec<f32>,
    pub rest: Vec<u16>,
    pub age: Vec<u16>,
    pub contact: Vec<u8>,
    /// contact normal
    pub cn: Vec<f32>,
    /// velocity of the surface in contact
    pub sv: Vec<f32>,
    pub rgba: Vec<u8>,
    /// position at the start of the frame
    pub frame_pos: Vec<f32>,
    /// smoothed net speed over frames (rest detection)
    pub slow: Vec<f32>,
    nbr: Vec<i32>,
    nc: Vec<u8>,
    cell_count: Vec<i32>,
    cell_start: Vec<i32>,
    sorted: Vec<i32>,
    cell_of: Vec<i32>,
    /// particle radius (-1 until configured)
    pub r: f64,
    ks: f64,
    /// spacing (diameter)
    pub d: f64,
    pub h: f64,
    h2: f64,
    rho0: f64,
    eps: f64,
}

#[inline]
fn hash(ix: i32, iy: i32, iz: i32) -> usize {
    ((ix.wrapping_mul(73856093) ^ iy.wrapping_mul(19349663) ^ iz.wrapping_mul(83492791)) & MASK) as usize
}

/// JS `Math.floor(x)` used as an int32 cell index.
#[inline]
fn cell(v: f64) -> i32 {
    v.floor() as i32
}

impl Fluid {
    pub fn new(cap: usize) -> Self {
        Fluid {
            cap,
            n: 0,
            pos: vec![0.0; cap * 3],
            prev: vec![0.0; cap * 3],
            vel: vec![0.0; cap * 3],
            dp: vec![0.0; cap * 3],
            lambda: vec![0.0; cap],
            rest: vec![0; cap],
            age: vec![0; cap],
            contact: vec![0; cap],
            cn: vec![0.0; cap * 3],
            sv: vec![0.0; cap * 3],
            rgba: vec![0; cap * 4],
            frame_pos: vec![0.0; cap * 3],
            slow: vec![0.0; cap],
            nbr: vec![0; cap * MAXN],
            nc: vec![0; cap],
            cell_count: vec![0; TABLE + 1],
            cell_start: vec![0; TABLE + 1],
            sorted: vec![0; cap],
            cell_of: vec![0; cap],
            r: -1.0,
            ks: f64::NAN,
            d: 0.0,
            h: 0.0,
            h2: 0.0,
            rho0: 0.0,
            eps: 0.0,
        }
    }

    pub fn configure(&mut self, radius: f64, kernel_scale: f64) {
        if radius == self.r && kernel_scale == self.ks {
            return;
        }
        self.r = radius;
        self.ks = kernel_scale;
        self.d = 2.0 * radius;
        self.h = kernel_scale * self.d;
        self.h2 = self.h * self.h;
        let mut rho = 0.0;
        for i in -4i32..=4 {
            for j in -4i32..=4 {
                for k in -4i32..=4 {
                    let r2 = ((i * i + j * j + k * k) as f64) * self.d * self.d;
                    if r2 < self.h2 {
                        rho += self.w(r2);
                    }
                }
            }
        }
        self.rho0 = rho;
        let g = 3.0 / self.h / self.rho0;
        self.eps = 2.0 * g * g;
    }

    #[inline]
    fn w(&self, r2: f64) -> f64 {
        let x = 1.0 - r2 / self.h2;
        x * x * x
    }

    pub fn clear(&mut self) {
        self.n = 0;
    }

    pub fn spawn(&mut self, x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, rgba: [u8; 4]) -> Option<usize> {
        if self.n >= self.cap {
            return None;
        }
        let i = self.n;
        self.n += 1;
        let o = i * 3;
        for (k, v) in [x, y, z].into_iter().enumerate() {
            self.pos[o + k] = v as f32;
            self.prev[o + k] = v as f32;
            self.frame_pos[o + k] = v as f32;
        }
        self.vel[o] = vx as f32;
        self.vel[o + 1] = vy as f32;
        self.vel[o + 2] = vz as f32;
        self.rest[i] = 0;
        self.age[i] = 0;
        self.slow[i] = 1.0;
        self.contact[i] = CONTACT_NONE;
        self.rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
        Some(i)
    }

    pub fn remove_at(&mut self, i: usize) {
        self.n -= 1;
        let last = self.n;
        if i == last {
            return;
        }
        let (a, b) = (i * 3, last * 3);
        for k in 0..3 {
            self.pos[a + k] = self.pos[b + k];
            self.prev[a + k] = self.prev[b + k];
            self.vel[a + k] = self.vel[b + k];
            self.cn[a + k] = self.cn[b + k];
            self.sv[a + k] = self.sv[b + k];
            self.frame_pos[a + k] = self.frame_pos[b + k];
        }
        self.slow[i] = self.slow[last];
        for k in 0..4 {
            self.rgba[i * 4 + k] = self.rgba[last * 4 + k];
        }
        self.rest[i] = self.rest[last];
        self.age[i] = self.age[last];
        self.contact[i] = self.contact[last];
    }

    pub fn mark_frame(&mut self) {
        let m = self.n * 3;
        self.frame_pos[..m].copy_from_slice(&self.pos[..m]);
    }

    #[inline]
    pub fn p(&self, i: usize) -> [f64; 3] {
        [self.pos[i * 3] as f64, self.pos[i * 3 + 1] as f64, self.pos[i * 3 + 2] as f64]
    }

    fn build_neighbors(&mut self) {
        let n = self.n;
        let inv = 1.0 / self.h;
        self.cell_count.iter_mut().for_each(|v| *v = 0);
        for i in 0..n {
            let [x, y, z] = self.p(i);
            let c = hash(cell(x * inv), cell(y * inv), cell(z * inv));
            self.cell_of[i] = c as i32;
            self.cell_count[c] += 1;
        }
        let mut acc = 0;
        for c in 0..TABLE {
            self.cell_start[c] = acc;
            acc += self.cell_count[c];
        }
        self.cell_start[TABLE] = acc;
        self.cell_count.iter_mut().for_each(|v| *v = 0);
        for i in 0..n {
            let c = self.cell_of[i] as usize;
            self.sorted[(self.cell_start[c] + self.cell_count[c]) as usize] = i as i32;
            self.cell_count[c] += 1;
        }
        let h2 = self.h2;
        for i in 0..n {
            let [x, y, z] = self.p(i);
            let (ix, iy, iz) = (cell(x * inv), cell(y * inv), cell(z * inv));
            let mut cnt = 0usize;
            let base = i * MAXN;
            'outer: for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if cnt >= MAXN {
                            break 'outer;
                        }
                        let c = hash(ix.wrapping_add(dx), iy.wrapping_add(dy), iz.wrapping_add(dz));
                        let s0 = self.cell_start[c] as usize;
                        let e = s0 + self.cell_count[c] as usize;
                        for s in s0..e {
                            let j = self.sorted[s] as usize;
                            if j == i {
                                continue;
                            }
                            let ex = self.pos[j * 3] as f64 - x;
                            let ey = self.pos[j * 3 + 1] as f64 - y;
                            let ez = self.pos[j * 3 + 2] as f64 - z;
                            if ex * ex + ey * ey + ez * ez < h2 {
                                // a hash collision can list a cell twice; skip duplicates
                                if self.nbr[base..base + cnt].contains(&(j as i32)) {
                                    continue;
                                }
                                self.nbr[base + cnt] = j as i32;
                                cnt += 1;
                                if cnt >= MAXN {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            self.nc[i] = cnt as u8;
        }
    }

    /// One sub-step.
    pub fn step<C: Collide>(&mut self, dt: f64, p: &FluidParams, env: &mut C) {
        let n = self.n;
        if n == 0 {
            return;
        }
        self.configure(p.particle_radius, p.kernel_scale);
        let damp = 0f64.max(1.0 - p.damping * dt);
        let vmax = 1.2;

        for i in 0..n {
            let o = i * 3;
            self.vel[o + 1] = (self.vel[o + 1] as f64 - p.gravity * dt) as f32;
            self.vel[o] = (self.vel[o] as f64 * damp) as f32;
            self.vel[o + 1] = (self.vel[o + 1] as f64 * damp) as f32;
            self.vel[o + 2] = (self.vel[o + 2] as f64 * damp) as f32;
            let sp = js_hypot(&[self.vel[o] as f64, self.vel[o + 1] as f64, self.vel[o + 2] as f64]);
            if sp > vmax {
                for k in 0..3 {
                    self.vel[o + k] = (self.vel[o + k] as f64 * (vmax / sp)) as f32;
                }
            }
            for k in 0..3 {
                self.prev[o + k] = self.pos[o + k];
                self.pos[o + k] = (self.pos[o + k] as f64 + self.vel[o + k] as f64 * dt) as f32;
            }
            self.contact[i] = CONTACT_NONE;
            if self.age[i] < 65535 {
                self.age[i] += 1;
            }
        }

        self.build_neighbors();
        let h = self.h;
        let rho0 = self.rho0;
        let d0 = self.d;
        let max_step = 0.5 * self.r;

        for _ in 0..p.iterations {
            // density constraint multipliers
            for i in 0..n {
                let pi = self.p(i);
                let mut rho = 1.0;
                let (mut gx, mut gy, mut gz, mut g2) = (0.0, 0.0, 0.0, 0.0);
                let base = i * MAXN;
                for q in 0..self.nc[i] as usize {
                    let j = self.nbr[base + q] as usize;
                    let pj = self.p(j);
                    let (dx, dy, dz) = (pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]);
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 >= self.h2 {
                        continue;
                    }
                    rho += self.w(r2);
                    let r = r2.sqrt();
                    if r < 1e-9 {
                        continue;
                    }
                    let t = 1.0 - r / h;
                    let gm = (3.0 * t * t) / h / rho0;
                    gx += (gm * dx) / r;
                    gy += (gm * dy) / r;
                    gz += (gm * dz) / r;
                    g2 += gm * gm;
                }
                let c = rho / rho0 - 1.0;
                self.lambda[i] = if c > 0.0 { (-c / (g2 + gx * gx + gy * gy + gz * gz + self.eps)) as f32 } else { 0.0 };
            }
            // position corrections
            for i in 0..n {
                let o = i * 3;
                let pi = self.p(i);
                let (mut ax, mut ay, mut az) = (0.0, 0.0, 0.0);
                let base = i * MAXN;
                let cnt = self.nc[i] as usize;
                let coh = if cnt > 0 { p.cohesion / cnt as f64 } else { 0.0 };
                let li = self.lambda[i] as f64;
                for q in 0..cnt {
                    let j = self.nbr[base + q] as usize;
                    let pj = self.p(j);
                    let (dx, dy, dz) = (pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]);
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 >= self.h2 {
                        continue;
                    }
                    let r = r2.sqrt();
                    if r < 1e-9 {
                        continue;
                    }
                    let t = 1.0 - r / h;
                    let gm = (3.0 * t * t) / h;
                    let s = (-(li + self.lambda[j] as f64) / rho0) * gm;
                    ax += (s * dx) / r;
                    ay += (s * dy) / r;
                    az += (s * dz) / r;
                    if r > d0 && coh > 0.0 {
                        let cc = (coh * 0.5 * (r - d0)) / r;
                        ax -= cc * dx;
                        ay -= cc * dy;
                        az -= cc * dz;
                    }
                }
                let m = js_hypot(&[ax, ay, az]);
                if m > max_step {
                    ax *= max_step / m;
                    ay *= max_step / m;
                    az *= max_step / m;
                }
                self.dp[o] = ax as f32;
                self.dp[o + 1] = ay as f32;
                self.dp[o + 2] = az as f32;
            }
            for i in 0..n {
                let o = i * 3;
                for k in 0..3 {
                    self.pos[o + k] = (self.pos[o + k] as f64 + self.dp[o + k] as f64) as f32;
                }
                env.collide(self, i);
            }
        }

        // velocities, contact friction
        let fr = p.friction;
        let static_v = 1.5 * fr * p.gravity * dt;
        for i in 0..n {
            let o = i * 3;
            let mut vx = (self.pos[o] as f64 - self.prev[o] as f64) / dt;
            let mut vy = (self.pos[o + 1] as f64 - self.prev[o + 1] as f64) / dt;
            let mut vz = (self.pos[o + 2] as f64 - self.prev[o + 2] as f64) / dt;
            if self.contact[i] != CONTACT_NONE {
                let (nx, ny, nz) = (self.cn[o] as f64, self.cn[o + 1] as f64, self.cn[o + 2] as f64);
                let (sx, sy, sz) = (self.sv[o] as f64, self.sv[o + 1] as f64, self.sv[o + 2] as f64);
                let (mut rx, mut ry, mut rz) = (vx - sx, vy - sy, vz - sz);
                let mut vn = rx * nx + ry * ny + rz * nz;
                if vn < 0.0 {
                    rx -= vn * nx;
                    ry -= vn * ny;
                    rz -= vn * nz;
                    vn = 0.0;
                } else if vn > MAX_SEPARATION {
                    // contact push-out is position correction, not a bounce
                    rx -= (vn - MAX_SEPARATION) * nx;
                    ry -= (vn - MAX_SEPARATION) * ny;
                    rz -= (vn - MAX_SEPARATION) * nz;
                    vn = MAX_SEPARATION;
                }
                let (tx, ty, tz) = (rx - vn * nx, ry - vn * ny, rz - vn * nz);
                // static friction: slow sliding sticks
                let ts = (tx * tx + ty * ty + tz * tz).sqrt();
                let kt = if ts < static_v { 0.0 } else { 1.0 - fr };
                vx = sx + vn * nx + tx * kt;
                vy = sy + vn * ny + ty * kt;
                vz = sz + vn * nz + tz * kt;
            }
            // freshly spawned particles start inside solids; their push-out is not real motion
            if self.age[i] < 6 {
                let settled = self.contact[i] != CONTACT_NONE;
                let cl = |v: f64| (-0.2f64).max(0.2f64.min(v));
                vx = if settled { self.sv[o] as f64 } else { cl(vx) };
                vy = if settled { self.sv[o + 1] as f64 } else { cl(vy) };
                vz = if settled { self.sv[o + 2] as f64 } else { cl(vz) };
            }
            self.dp[o] = vx as f32;
            self.dp[o + 1] = vy as f32;
            self.dp[o + 2] = vz as f32;
        }
        // XSPH viscosity
        let visc = p.viscosity;
        for i in 0..n {
            let o = i * 3;
            let pi = self.p(i);
            let (mut ax, mut ay, mut az, mut ws) = (0.0, 0.0, 0.0, 0.0);
            let base = i * MAXN;
            for q in 0..self.nc[i] as usize {
                let j = self.nbr[base + q] as usize;
                let pj = self.p(j);
                let (dx, dy, dz) = (pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]);
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= self.h2 {
                    continue;
                }
                let w = self.w(r2);
                ax += (self.dp[j * 3] as f64 - self.dp[o] as f64) * w;
                ay += (self.dp[j * 3 + 1] as f64 - self.dp[o + 1] as f64) * w;
                az += (self.dp[j * 3 + 2] as f64 - self.dp[o + 2] as f64) * w;
                ws += w;
            }
            let k = if ws > 0.0 { visc / ws.max(1.0) } else { 0.0 };
            self.vel[o] = (self.dp[o] as f64 + ax * k) as f32;
            self.vel[o + 1] = (self.dp[o + 1] as f64 + ay * k) as f32;
            self.vel[o + 2] = (self.dp[o + 2] as f64 + az * k) as f32;
        }
    }
}
