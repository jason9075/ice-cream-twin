//! Pour experiment: scoop, flip the spoon, and report how the loose ice cream holds together.
//!   cargo run --release -p icecream-core --example pour -- [viscosity cohesion lump_stiffness lump_yield]
//!   RELEASE=1 ...: release on a level spoon instead, and report the scoop's shape in the cup
use icecream_core::math::{Quat, Vec3};
use icecream_core::sdf::bowls;
use icecream_core::{Params, Sim};
use std::f64::consts::PI;

fn clusters(pos: &[f32], n: usize, link: f64) -> (usize, usize) {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let l2 = link * link;
    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = (0..3).map(|k| (pos[i * 3 + k] - pos[j * 3 + k]) as f64).map(|v| v * v).sum();
            if d < l2 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut size = vec![0usize; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        size[r] += 1;
    }
    let count = size.iter().filter(|&&s| s > 0).count();
    (count, size.into_iter().max().unwrap_or(0))
}

/// Top of the ice in each (x, z) column of bowl grid k (metres, NaN where empty).
fn heights(s: &Sim, k: usize) -> Vec<f64> {
    let g = &s.grids[k];
    let mut out = vec![f64::NAN; g.nx * g.nz];
    for iz in 0..g.nz {
        for ix in 0..g.nx {
            for iy in (1..g.ny).rev() {
                let a = g.data[ix + g.nx * (iy + g.ny * iz)] as f64;
                let b = g.data[ix + g.nx * (iy - 1 + g.ny * iz)] as f64;
                if a >= 0.0 && b < 0.0 {
                    let t = a / (a - b);
                    out[ix + g.nx * iz] = g.oy + (iy as f64 + 0.5 - t) * g.cell;
                    break;
                }
            }
        }
    }
    out
}

fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let mut p = Params::default();
    if let Some(&v) = a.first() {
        p.fluid.viscosity = v;
    }
    if let Some(&v) = a.get(1) {
        p.fluid.cohesion = v;
    }
    if let Some(&v) = a.get(2) {
        p.fluid.lump_stiffness = v;
    }
    if let Some(&v) = a.get(3) {
        p.fluid.lump_yield = v;
    }
    let mut s = Sim::new(p);
    let b = &bowls()[1];
    let go = |s: &mut Sim, p: Vec3, q: Quat, n: usize, log: bool| {
        s.set_target(Some(p), Some(q));
        for f in 0..n {
            s.step(1.0 / 60.0);
            if log && f % 5 == 0 && f >= 40 && f <= 100 {
                let fl = &s.fluid;
                let n = fl.n;
                let airs: Vec<usize> = (0..n).filter(|&i| fl.contact[i] == 0).collect();
                let ap: Vec<f32> = airs.iter().flat_map(|&i| fl.pos[i * 3..i * 3 + 3].to_vec()).collect();
                let air = airs.len();
                let (c, big) = clusters(&ap, air, 1.25 * fl.d);
                let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                for &i in &airs {
                    let y = fl.pos[i * 3 + 1] as f64;
                    lo = lo.min(y);
                    hi = hi.max(y);
                }
                println!(
                    "f{f:3}  loose {n:4}  air clusters {c:3}  largest {:5.1}%  airborne {air:4}  y-span {:5.1} mm  crumbs {}",
                    if air > 0 { 100.0 * big as f64 / air as f64 } else { 0.0 },
                    if air > 0 { (hi - lo) * 1000.0 } else { 0.0 },
                    s.n_spilled
                );
            }
        }
    };
    go(&mut s, Vec3::new(b.cx, 0.14, 0.0), Quat::IDENTITY, 60, false);
    go(&mut s, Vec3::new(b.cx, 0.085, 0.0), Quat::IDENTITY, 150, false);
    go(&mut s, Vec3::new(b.cx, 0.15, 0.0), Quat::IDENTITY, 120, false);
    println!("scooped {:.1} g", s.state().scooped_grams);
    if std::env::var("RELEASE").is_ok() {
        // release on a level spoon: does the scoop keep its shape in the cup?
        s.release();
        s.set_target(Some(Vec3::new(b.cx, 0.15, 0.0)), Some(Quat::IDENTITY));
        for f in 0..=90 {
            if f % 15 == 0 || f < 12 {
                let fl = &s.fluid;
                let n = fl.n;
                let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
                for i in 0..n {
                    for k in 0..3 {
                        let v = fl.pos[i * 3 + k] as f64;
                        lo[k] = lo[k].min(v);
                        hi[k] = hi[k].max(v);
                    }
                }
                println!(
                    "f{f:3}  below {:4}  loose {n:4}  height {:5.1} mm  width x {:5.1} z {:5.1} mm  on spoon {:.1} g",
                    (0..n).filter(|&i| (fl.pos[i * 3 + 1] as f64) < 0.135).count(),
                    (hi[1] - lo[1]) * 1e3,
                    (hi[0] - lo[0]) * 1e3,
                    (hi[2] - lo[2]) * 1e3,
                    s.state().scooped_grams
                );
            }
            s.step(1.0 / 60.0);
        }
        return;
    }
    let h0 = heights(&s, 1);
    go(&mut s, Vec3::new(b.cx, 0.15, 0.0), Quat::from_euler(PI * 0.9, 0.0, 0.0), 200, true);
    go(&mut s, Vec3::new(b.cx, 0.15, 0.0), Quat::from_euler(PI * 0.9, 0.0, 0.0), 120, false);
    let h1 = heights(&s, 1);
    let c = s.grids[1].cell;
    let (mut area, mut top, mut vol) = (0.0, 0f64, 0.0);
    for (a, b) in h0.iter().zip(&h1) {
        let d = if a.is_nan() || b.is_nan() { 0.0 } else { b - a };
        if d > 0.0005 {
            area += c * c;
            top = top.max(d);
            vol += d * c * c;
        }
    }
    println!(
        "deposit: footprint {:.1} cm²  peak {:.1} mm  volume {:.1} cm³  peak/√area {:.2}  loose left {}",
        area * 1e4,
        top * 1e3,
        vol * 1e6,
        top / area.sqrt(),
        s.fluid.n
    );
}
