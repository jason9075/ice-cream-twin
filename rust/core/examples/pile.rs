//! A cube of particles dropped on a floor: does it keep its shape?
//!   cargo run --release -p icecream-core --example pile -- viscosity cohesion lump_stiffness lump_yield
use icecream_core::fluid::{Collide, Fluid, CONTACT_WORLD};
use icecream_core::Params;

struct Floor;
impl Collide for Floor {
    fn collide(&mut self, f: &mut Fluid, i: usize) {
        let d = f.pos[i * 3 + 1] as f64 - f.r;
        if d < 0.0 {
            f.pos[i * 3 + 1] = f.r as f32;
            f.contact[i] = CONTACT_WORLD;
            f.cn[i * 3..i * 3 + 3].copy_from_slice(&[0.0, 1.0, 0.0]);
            f.sv[i * 3..i * 3 + 3].copy_from_slice(&[0.0, 0.0, 0.0]);
        }
    }
}

fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let mut p = Params::default().fluid;
    for (k, v) in a.iter().enumerate() {
        match k {
            0 => p.viscosity = *v,
            1 => p.cohesion = *v,
            2 => p.lump_stiffness = *v,
            3 => p.lump_yield = *v,
            _ => {}
        }
    }
    let mut f = Fluid::new(4000);
    f.configure(p.particle_radius, p.kernel_scale);
    let d = f.d;
    for z in 0..10 {
        for y in 0..10 {
            for x in 0..10 {
                f.spawn(x as f64 * d, f.r + y as f64 * d, z as f64 * d, 0.0, 0.0, 0.0, [255; 4]);
            }
        }
    }
    let k = f.new_lump();
    for i in 0..f.n {
        f.set_lump(i, k);
    }
    let dt = 1.0 / 60.0 / p.substeps as f64;
    for frame in 0..=120 {
        if frame % 20 == 0 {
            let (mut top, mut w) = (0f64, 0f64);
            for i in 0..f.n {
                top = top.max(f.pos[i * 3 + 1] as f64);
                w = w.max((f.pos[i * 3] as f64 - 4.5 * d).abs());
            }
            println!("f{frame:3}  height {:5.1} mm  half-width {:5.1} mm", (top + f.r) * 1e3, (w + f.r) * 1e3);
        }
        for _ in 0..p.substeps {
            f.step(dt, &p, &mut Floor);
        }
    }
}
