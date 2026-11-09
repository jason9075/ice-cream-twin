//! Mesh generation. Every mesh is interleaved (position, normal) f32 vertices plus u32 indices. Shapes are
//! built from the same numbers as the simulation's collision SDFs.

use icecream_core::sdf::SpoonShape;
use std::f32::consts::PI;

#[derive(Default, Clone)]
pub struct MeshData {
    /// px py pz nx ny nz per vertex
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
}

impl MeshData {
    fn count(&self) -> u32 {
        (self.vertices.len() / 6) as u32
    }
    fn vert(&mut self, p: [f32; 3], n: [f32; 3]) -> u32 {
        self.vertices.extend_from_slice(&[p[0], p[1], p[2], n[0], n[1], n[2]]);
        self.count() - 1
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }
    /// Grid of (rows+1) x (cols+1) vertices from f(u, v) -> (pos, normal).
    fn grid(&mut self, rows: u32, cols: u32, f: impl Fn(f32, f32) -> ([f32; 3], [f32; 3]), flip: bool) {
        let base = self.count();
        for r in 0..=rows {
            for c in 0..=cols {
                let (p, n) = f(r as f32 / rows as f32, c as f32 / cols as f32);
                self.vert(p, n);
            }
        }
        let w = cols + 1;
        for r in 0..rows {
            for c in 0..cols {
                let a = base + r * w + c;
                let (b, d) = (a + 1, a + w);
                let e = d + 1;
                if flip {
                    self.tri(a, b, d);
                    self.tri(b, e, d);
                } else {
                    self.tri(a, d, b);
                    self.tri(b, d, e);
                }
            }
        }
    }
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let l = if l > 0.0 { l } else { 1.0 };
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Surface of revolution around +Y from a 2D profile [[radius, y], ...].
pub fn lathe(profile: &[[f64; 2]], segments: u32) -> MeshData {
    let mut b = MeshData::default();
    let n = profile.len();
    let pn: Vec<[f32; 2]> = (0..n)
        .map(|k| {
            let a = profile[k.saturating_sub(1)];
            let c = profile[(k + 1).min(n - 1)];
            let (tx, ty) = ((c[0] - a[0]) as f32, (c[1] - a[1]) as f32);
            let l = (tx * tx + ty * ty).sqrt().max(1e-12);
            [ty / l, -tx / l] // tangent rotated -90°
        })
        .collect();
    b.grid(
        n as u32 - 1,
        segments,
        |u, v| {
            let k = (u * (n - 1) as f32).round() as usize;
            let th = v * PI * 2.0;
            let (cs, sn) = (th.cos(), th.sin());
            let (r, y) = (profile[k][0] as f32, profile[k][1] as f32);
            let [nr, ny] = pn[k];
            ([r * cs, y, r * sn], [nr * cs, ny, nr * sn])
        },
        true,
    );
    b
}

pub fn sphere(lat: u32, lon: u32) -> MeshData {
    let mut b = MeshData::default();
    b.grid(
        lat,
        lon,
        |u, v| {
            let (th, ph) = (u * PI, v * PI * 2.0);
            let p = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
            (p, p)
        },
        true,
    );
    b
}

/// Axis-aligned box from min to max corner.
pub fn cuboid(min: [f32; 3], max: [f32; 3]) -> MeshData {
    let mut b = MeshData::default();
    let faces: [(usize, bool, [f32; 3]); 6] = [
        (0, true, [1.0, 0.0, 0.0]),
        (0, false, [-1.0, 0.0, 0.0]),
        (1, true, [0.0, 1.0, 0.0]),
        (1, false, [0.0, -1.0, 0.0]),
        (2, true, [0.0, 0.0, 1.0]),
        (2, false, [0.0, 0.0, -1.0]),
    ];
    for (axis, side, n) in faces {
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let base = b.count();
        for (su, sw) in [(false, false), (true, false), (true, true), (false, true)] {
            let mut p = [0.0; 3];
            p[axis] = if side { max[axis] } else { min[axis] };
            p[u] = if su { max[u] } else { min[u] };
            p[w] = if sw { max[w] } else { min[w] };
            b.vert(p, n);
        }
        if side {
            b.tri(base, base + 1, base + 2);
            b.tri(base, base + 2, base + 3);
        } else {
            b.tri(base, base + 2, base + 1);
            b.tri(base, base + 3, base + 2);
        }
    }
    b
}

/// Spoon: half-ellipsoid cup shell (outer, inner, rim) + polyline handle (cylinders + joints).
pub fn spoon(s: &SpoonShape) -> MeshData {
    let mut b = MeshData::default();
    let (a, bb, c, t) = (s.a as f32, s.b as f32, s.c as f32, s.t as f32);
    let h = t / 2.0;
    let (n_th, n_ph) = (22, 72);
    let cup = |aa: f32, bb: f32, cc: f32, sign: f32| {
        move |u: f32, v: f32| {
            let (th, ph) = (u * PI / 2.0, v * PI * 2.0);
            let p = [aa * th.sin() * ph.cos(), -bb * th.cos(), cc * th.sin() * ph.sin()];
            let n = norm([p[0] / (aa * aa), p[1] / (bb * bb), p[2] / (cc * cc)]);
            (p, [n[0] * sign, n[1] * sign, n[2] * sign])
        }
    };
    b.grid(n_th, n_ph, cup(a + h, bb + h, c + h, 1.0), false);
    b.grid(n_th, n_ph, cup(a - h, bb - h, c - h, -1.0), true);
    // rim strip (y = 0), facing up
    b.grid(
        1,
        n_ph,
        |u, v| {
            let ph = v * PI * 2.0;
            let (aa, cc) = if u > 0.5 { (a - h, c - h) } else { (a + h, c + h) };
            ([aa * ph.cos(), 0.0, cc * ph.sin()], [0.0, 1.0, 0.0])
        },
        true,
    );
    let r = s.handle_r as f32;
    let hp: Vec<[f32; 3]> = s.handle.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    for k in 0..hp.len() - 1 {
        let (p0, p1) = (hp[k], hp[k + 1]);
        let ax = norm([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]);
        let rf = if ax[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let e1 = norm([ax[1] * rf[2] - ax[2] * rf[1], ax[2] * rf[0] - ax[0] * rf[2], ax[0] * rf[1] - ax[1] * rf[0]]);
        let e2 = [ax[1] * e1[2] - ax[2] * e1[1], ax[2] * e1[0] - ax[0] * e1[2], ax[0] * e1[1] - ax[1] * e1[0]];
        b.grid(
            1,
            24,
            |u, v| {
                let ph = v * PI * 2.0;
                let n: [f32; 3] = std::array::from_fn(|i| e1[i] * ph.cos() + e2[i] * ph.sin());
                let o = if u > 0.5 { p1 } else { p0 };
                (std::array::from_fn(|i| o[i] + n[i] * r), n)
            },
            true,
        );
    }
    let sp = sphere(10, 20);
    for p in &hp {
        let base = b.count();
        for v in sp.vertices.as_chunks::<6>().0 {
            let n = [v[0], v[1], v[2]];
            b.vert([p[0] + n[0] * r, p[1] + n[1] * r, p[2] + n[2] * r], n);
        }
        b.indices.extend(sp.indices.iter().map(|i| base + i));
    }
    b
}
