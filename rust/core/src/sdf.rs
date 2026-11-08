//! Analytic signed distance functions for the static scene (bowls, table) and the spoon.
//! Units: metres. Frame: Y up. Negative = inside. Mirrors `src/sim/sdf.js`.

use crate::math::js_round;
use std::sync::OnceLock;

pub const BOWL_R: f64 = 0.075;
pub const BOWL_T: f64 = 0.003;
pub const BOWL_H: f64 = 0.025;
pub const BOWL_SPACING: f64 = 0.18;

#[derive(Clone, Debug)]
pub struct Flavor {
    pub id: &'static str,
    pub label: &'static str,
    pub color: u32,
    pub speck: f64,
    pub speck_color: u32,
}

pub const FLAVORS: [Flavor; 3] = [
    Flavor { id: "vanilla", label: "Vanilla", color: 0xf4e7c3, speck: 0.35, speck_color: 0x5a3b1c },
    Flavor { id: "strawberry", label: "Strawberry", color: 0xf3a9bb, speck: 0.25, speck_color: 0xb0283c },
    Flavor { id: "mint", label: "Mint choc chip", color: 0xbfeed6, speck: 1.0, speck_color: 0x3b2416 },
];

/// A bowl: a "cut hollow sphere" (shell of mid radius r, half-thickness t, kept below y = cy + h).
#[derive(Clone, Debug)]
pub struct Bowl {
    pub index: usize,
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub r: f64,
    pub t: f64,
    pub h: f64,
    /// rim radius (mid surface)
    pub w: f64,
    pub rim_y: f64,
    pub inner_r: f64,
}

pub fn bowls() -> &'static [Bowl; 3] {
    static B: OnceLock<[Bowl; 3]> = OnceLock::new();
    B.get_or_init(|| {
        let mk = |i: usize| {
            let cx = (i as f64 - 1.0) * BOWL_SPACING;
            let cy = BOWL_R + BOWL_T;
            Bowl {
                index: i,
                cx,
                cy,
                cz: 0.0,
                r: BOWL_R,
                t: BOWL_T,
                h: BOWL_H,
                w: (BOWL_R * BOWL_R - BOWL_H * BOWL_H).sqrt(),
                rim_y: cy + BOWL_H,
                inner_r: BOWL_R - BOWL_T,
            }
        };
        [mk(0), mk(1), mk(2)]
    })
}

pub fn sd_bowl(b: &Bowl, x: f64, y: f64, z: f64) -> f64 {
    let px = x - b.cx;
    let py = y - b.cy;
    let pz = z - b.cz;
    let qx = (px * px + pz * pz).sqrt();
    let d = if b.h * qx < b.w * py {
        let dx = qx - b.w;
        let dy = py - b.h;
        (dx * dx + dy * dy).sqrt()
    } else {
        ((qx * qx + py * py).sqrt() - b.r).abs()
    };
    d - b.t
}

/// The bowl whose centre column is closest in x (bowls are laid out along x).
pub fn nearest_bowl(x: f64) -> &'static Bowl {
    let mut i = js_round(x / BOWL_SPACING) + 1.0;
    if i < 0.0 {
        i = 0.0;
    }
    if i > 2.0 {
        i = 2.0;
    }
    &bowls()[i as usize]
}

/// Static environment: table plane (y = 0) and every bowl.
pub fn sd_env(x: f64, y: f64, z: f64) -> f64 {
    let b = nearest_bowl(x);
    y.min(sd_bowl(b, x, y, z))
}

pub const EPS: f64 = 1e-5;

pub fn grad_env(x: f64, y: f64, z: f64) -> [f64; 3] {
    let gx = sd_env(x + EPS, y, z) - sd_env(x - EPS, y, z);
    let gy = sd_env(x, y + EPS, z) - sd_env(x, y - EPS, z);
    let gz = sd_env(x, y, z + EPS) - sd_env(x, y, z - EPS);
    let mut l = (gx * gx + gy * gy + gz * gz).sqrt();
    if l == 0.0 {
        l = 1.0;
    }
    [gx / l, gy / l, gz / l]
}

// ---------------------------------------------------------------- spoon (local frame)
// Origin = centre of the cup opening. +Y = cup opening normal, +X = toward the handle.

pub struct SpoonShape {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub t: f64,
    pub handle_r: f64,
    pub handle: [[f64; 3]; 4],
}

pub const SPOON: SpoonShape = SpoonShape {
    a: 0.028,
    b: 0.011,
    c: 0.019,
    t: 0.0016,
    handle_r: 0.0032,
    handle: [[0.024, 0.001, 0.0], [0.042, 0.0045, 0.0], [0.078, 0.013, 0.0], [0.165, 0.03, 0.0]],
};

pub fn sd_ellipsoid(x: f64, y: f64, z: f64, a: f64, b: f64, c: f64) -> f64 {
    let ax = x / a;
    let by = y / b;
    let cz = z / c;
    let k0 = (ax * ax + by * by + cz * cz).sqrt();
    let ax2 = x / (a * a);
    let by2 = y / (b * b);
    let cz2 = z / (c * c);
    let k1 = (ax2 * ax2 + by2 * by2 + cz2 * cz2).sqrt();
    if k1 < 1e-12 {
        return -a.min(b).min(c);
    }
    (k0 * (k0 - 1.0)) / k1
}

/// The metal of the cup.
pub fn sd_spoon_shell(x: f64, y: f64, z: f64) -> f64 {
    let s = &SPOON;
    let e = sd_ellipsoid(x, y, z, s.a, s.b, s.c);
    (e.abs() - s.t / 2.0).max(y)
}

/// The empty space inside the cup, below the rim plane. Whatever is in here gets scooped.
pub fn sd_hull_interior(x: f64, y: f64, z: f64) -> f64 {
    let s = &SPOON;
    let h = s.t / 2.0;
    sd_ellipsoid(x, y, z, s.a - h, s.b - h, s.c - h).max(y)
}

#[allow(clippy::too_many_arguments)]
fn sd_capsule(x: f64, y: f64, z: f64, ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64, r: f64) -> f64 {
    let pax = x - ax;
    let pay = y - ay;
    let paz = z - az;
    let bax = bx - ax;
    let bay = by - ay;
    let baz = bz - az;
    let mut h = (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz);
    h = if h < 0.0 { 0.0 } else if h > 1.0 { 1.0 } else { h };
    let dx = pax - bax * h;
    let dy = pay - bay * h;
    let dz = paz - baz * h;
    (dx * dx + dy * dy + dz * dz).sqrt() - r
}

pub fn sd_handle(x: f64, y: f64, z: f64) -> f64 {
    let h = &SPOON.handle;
    let mut d = f64::INFINITY;
    for i in 0..h.len() - 1 {
        let a = h[i];
        let b = h[i + 1];
        let di = sd_capsule(x, y, z, a[0], a[1], a[2], b[0], b[1], b[2], SPOON.handle_r);
        if di < d {
            d = di;
        }
    }
    d
}

pub fn sd_spoon_solid(x: f64, y: f64, z: f64) -> f64 {
    sd_spoon_shell(x, y, z).min(sd_handle(x, y, z))
}

pub fn grad_local(f: fn(f64, f64, f64) -> f64, x: f64, y: f64, z: f64) -> [f64; 3] {
    let gx = f(x + EPS, y, z) - f(x - EPS, y, z);
    let gy = f(x, y + EPS, z) - f(x, y - EPS, z);
    let gz = f(x, y, z + EPS) - f(x, y, z - EPS);
    let mut l = (gx * gx + gy * gy + gz * gz).sqrt();
    if l == 0.0 {
        l = 1.0;
    }
    [gx / l, gy / l, gz / l]
}

/// Polynomial smooth minimum (blend width k).
pub fn smin(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return if a < b { a } else { b };
    }
    let mut h = 0.5 + (0.5 * (b - a)) / k;
    h = if h < 0.0 { 0.0 } else if h > 1.0 { 1.0 } else { h };
    b + (a - b) * h - k * h * (1.0 - h)
}
