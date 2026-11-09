//! The GPU simulation against the CPU one (icecream-core). Needs a GPU: without one every test says so
//! and passes (CI has none).
use icecream_core::math::{Quat, Vec3};
use icecream_core::sdf::bowls;
use icecream_core::{Params, Sim};
use icecream_gpu::layout::{cnt, st, CAP};
use icecream_gpu::{GpuSim, GpuTwin};
use std::f64::consts::PI;
use std::time::Instant;

fn gpu() -> Option<GpuSim> {
    match GpuSim::new(Params::default()) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            None
        }
    }
}

fn cpu_total(s: &Sim) -> f64 {
    s.total_grams()
}

fn facing(pitch: f64) -> Quat {
    Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI / 2.0).mul(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), pitch))
}

fn drive(g: &mut GpuSim, p: Vec3, q: Quat, frames: usize, mut every: impl FnMut(&GpuSim)) {
    g.cpu.spoon.set_target(Some(p), Some(q));
    for f in 0..frames {
        g.step(1.0 / 60.0);
        if f % 10 == 0 {
            every(g);
        }
    }
}

#[test]
fn prefix_scans_and_volumes() {
    let Some(mut g) = gpu() else { return };
    let n = 100_003u32;
    let u: Vec<u32> = (0..n).map(|i| (i.wrapping_mul(2654435761)) % 7).collect();
    let f: Vec<f32> = (0..n).map(|i| ((i * 40503) % 1000) as f32 / 997.0).collect();
    let (scanu, list, dvd, pd) = (g.l.x.get("SCANU"), g.l.x.get("LIST"), g.l.w.get("DVD"), g.l.w.get("PD"));
    g.debug_write('X', scanu, bytemuck::cast_slice(&u));
    g.debug_write('W', dvd, bytemuck::cast_slice(&f));
    g.debug_run(|g, fr| {
        g.debug_scan(fr, true, scanu, list, n);
        g.debug_scan(fr, false, dvd, pd, n);
        for k in 0..3 {
            g.debug_grid_occ(fr, k, st::VOLUME + k as u32);
        }
    });
    let gu = g.debug_read('X', list, n);
    let gf = g.debug_read_f32(pd, n);
    let (mut au, mut af, mut bad, mut rel) = (0u32, 0f64, 0, 0f64);
    for i in 0..n as usize {
        bad += usize::from(gu[i] != au);
        rel = rel.max((gf[i] as f64 - af).abs() / af.max(1.0));
        au += u[i];
        af += f[i] as f64;
    }
    assert_eq!(bad, 0, "u32 scan");
    assert!(rel < 1e-5, "f32 scan rel error {rel:e}");
    let stv = g.debug_read_f32(g.l.w.get("ST"), 64);
    for k in 0..3 {
        let cpu = g.cpu.grids[k].volume();
        let r = (stv[(st::VOLUME + k as u32) as usize] as f64 - cpu).abs() / cpu;
        assert!(r < 1e-5, "volume {k}: rel {r:e}");
    }
}

#[test]
fn smoothing_matches_the_cpu() {
    let Some(mut g) = gpu() else { return };
    let mut cpu = Sim::default();
    let grid = &mut cpu.grids[1];
    // edits: a few carved and stamped spheres around the surface
    for k in 0..8 {
        let x = -0.03 + 0.008 * k as f64;
        grid.carve_sphere(x, 0.095, 0.0, 0.006);
        grid.stamp_sphere(x + 0.004, 0.092, 0.01, 0.004, 0.0012, None);
    }
    let touched: Vec<u32> = grid.touched_flags().iter().map(|&t| t as u32).collect();
    let n_touched = grid.n_touched;
    let off = g.l.grids[1].off;
    g.debug_write('W', g.l.w.get("D") + off, bytemuck::cast_slice(&grid.data));
    g.debug_write('X', g.l.x.get("FLAG") + off, bytemuck::cast_slice(&touched));
    let sm = cpu.params.ice.smoothing;
    // the spoon sits in the ice there: smoothing keeps out of it on both
    let (sp, sq) = (Vec3::new(-0.01, 0.094, 0.0), Quat::IDENTITY);
    g.cpu.spoon.pos = sp;
    g.cpu.spoon.quat = sq;
    g.debug_globals(1.0 / 60.0);
    let gm = &cpu.grids[1];
    let keep_out = icecream_core::sim::spoon_keep_out(
        icecream_core::math::Pose::new(sp, sq),
        &cpu.content,
        false,
        [gm.ox, gm.oy, gm.oz, gm.cell],
        [gm.nx, gm.ny],
        gm.band,
    );
    let floored = (0..gm.n).filter(|&i| keep_out(i) > -gm.band).count();
    let grid = &mut cpu.grids[1];
    grid.smooth_touched(sm, &keep_out);
    grid.settle(sm, true, &keep_out);
    let resid = grid.take_residual();
    g.debug_run(|g, fr| {
        g.debug_smooth(fr, 1, 0, sm);
        g.debug_smooth(fr, 1, 1, sm);
    });
    let d = g.debug_read_f32(g.l.w.get("D") + off, grid.n as u32);
    let err = d.iter().zip(&grid.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) as f64 / grid.cell;
    let st = g.debug_read_f32(g.l.w.get("ST"), 64);
    eprintln!("smoothing {n_touched} voxels ({floored} under the spoon): max diff {err:.1e} cells, residual {:.4} vs {:.4} mm³", st[(st::RESIDUAL + 1) as usize] as f64 * 1e9, resid * 1e9);
    assert!(err < 1e-3);
    assert!((st[(st::RESIDUAL + 1) as usize] as f64 - resid).abs() < 1e-9);
}

#[test]
fn carving_matches_the_cpu() {
    let Some(mut g) = gpu() else { return };
    let mut cpu = Sim::default();
    cpu.content_floor();
    let mut path = Vec::new();
    for k in 0..=30 {
        let t = k as f64 / 30.0;
        path.push((Vec3::new(0.0, 0.14 - 0.055 * t, 0.02 - 0.005 * t), facing(0.5)));
    }
    for k in 1..=30 {
        let t = k as f64 / 30.0;
        path.push((Vec3::new(0.0, 0.086 + 0.002 * t, 0.015 - 0.04 * t), facing(0.5 * (1.0 - t))));
    }
    for &(p, q) in &path {
        cpu.interact_at(p, q);
    }
    g.debug_globals(1.0 / 60.0);
    g.debug_run(|g, fr| {
        g.debug_frame_start(fr);
        for &(p, q) in &path {
            g.interact(fr, p, q, true);
        }
        g.debug_compact(fr);
    });
    let l = &g.l;
    let bowl = g.debug_read_f32(l.w.get("D") + l.grids[1].off, l.grids[1].count);
    let content = g.debug_read_f32(l.w.get("D") + l.grids[3].off, l.grids[3].count);
    let n = g.debug_read('A', l.a.get("CNT") + cnt::N, 1)[0];
    let bins = g.debug_read('A', l.a.get("BINS"), l.nbins);
    let stv = g.debug_read_f32(l.w.get("ST"), 64);
    let owed: f64 = bins.iter().map(|&b| b as f64 * 1e-14).sum::<f64>() - (0..3).map(|k| stv[(st::NEG_SETTLE + k) as usize] as f64).sum::<f64>();
    let eb = bowl.iter().zip(&cpu.grids[1].data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) as f64 / cpu.grids[1].cell;
    let band = cpu.content.band as f32;
    let differing = content.iter().zip(&cpu.content.data).filter(|(a, b)| (a.min(band) - b.min(band)).abs() as f64 > 1e-3 * cpu.content.cell).count();
    let cv = cpu.content.volume();
    let gcv: f64 = content.iter().map(|&v| (0.5 - v as f64 / cpu.content.cell).clamp(0.0, 1.0)).sum::<f64>() * cpu.content.cell.powi(3);
    eprintln!(
        "{} carve steps: bowl max diff {eb:.1e} cells, scoop voxels differing {differing}, scoop {:.3} vs {:.3} cm³, crumbs {n} vs {}, owed {:.4} vs {:.4} cm³",
        path.len(), gcv * 1e6, cv * 1e6, cpu.fluid.n, owed * 1e6, cpu.debt_volume() * 1e6
    );
    assert!(eb < 1e-3);
    assert!(differing <= 20, "{differing} scoop voxels differ");
    assert!((gcv - cv).abs() < 1e-3 * cv);
    assert!((n as i64 - cpu.fluid.n as i64).abs() <= 2);
    assert!((owed - cpu.debt_volume()).abs() < 0.02e-6 + 0.02 * cpu.debt_volume().abs());

    // then pile what was plowed beside the cut, on both
    let vol_before = cpu.grids[1].volume();
    cpu.plow_pay();
    let (d_at, g1_off, g1_n) = (l.w.get("D"), l.grids[1].off, l.grids[1].count);
    g.debug_run(|g, fr| g.debug_plow(fr));
    let bowl = g.debug_read_f32(d_at + g1_off, g1_n);
    let eb = bowl.iter().zip(&cpu.grids[1].data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) as f64 / cpu.grids[1].cell;
    let gv: f64 = bowl.iter().map(|&v| (0.5 - v as f64 / cpu.grids[1].cell).clamp(0.0, 1.0)).sum::<f64>() * cpu.grids[1].cell.powi(3);
    let cv = cpu.grids[1].volume();
    eprintln!("plowed beside the cut: bowl +{:.3} cm³ (GPU) vs +{:.3} cm³ (CPU), max diff {eb:.1e} cells", (gv - vol_before) * 1e6, (cv - vol_before) * 1e6);
    assert!((gv - cv).abs() < 0.02 * (cv - vol_before).abs() + 1e-9, "the plowed ice landed differently");
}

#[test]
fn scoop_carry_eat_like_the_cpu() {
    let Some(mut g) = gpu() else { return };
    let mut cpu = Sim::default();
    let b = &bowls()[0];
    let dig = Quat::from_euler(0.0, 0.0, 0.5);
    let legs = [
        (Vec3::new(b.cx - 0.03, 0.14, 0.0), dig, 90),
        (Vec3::new(b.cx - 0.02, 0.085, 0.0), dig, 150),
        (Vec3::new(b.cx + 0.025, 0.088, 0.0), Quat::IDENTITY, 150),
        (Vec3::new(b.cx + 0.025, 0.16, 0.0), Quat::IDENTITY, 120),
    ];
    let before = cpu_total(&cpu);
    g.step(1.0 / 60.0);
    let m0 = g.total_grams();
    let mut drift = 0f64;
    let (mut tg, mut tc) = (0.0, 0.0);
    for (p, q, n) in legs {
        cpu.set_target(Some(p), Some(q));
        g.cpu.spoon.set_target(Some(p), Some(q));
        for f in 0..n {
            let t = Instant::now();
            cpu.step(1.0 / 60.0);
            tc += t.elapsed().as_secs_f64();
            let t = Instant::now();
            g.step(1.0 / 60.0);
            tg += t.elapsed().as_secs_f64();
            if f % 10 == 0 {
                drift = drift.max((g.total_grams() - m0).abs());
            }
        }
    }
    let (sg, sc) = (g.state().scooped_grams, cpu.state().scooped_grams);
    let cpu_drift = (cpu_total(&cpu) - before).abs();
    let (eg, ec) = (g.eat(), cpu.eat());
    eprintln!(
        "scoop vanilla: scooped GPU {sg:.1} g vs CPU {sc:.1} g, mass drift {drift:.3} g (CPU {cpu_drift:.3} g), eaten {eg:.1} vs {ec:.1} g, {:.2} vs {:.2} ms/frame",
        tg * 1000.0 / 510.0,
        tc * 1000.0 / 510.0
    );
    // whole runs drift apart (f32, the ridges plowed beside the cut feed back into later carving); the
    // step-by-step comparisons (carving_matches_the_cpu) pin the kernels exactly
    assert!(sg > 3.0 && (sg - sc).abs() < 0.4 * sc);
    assert!(drift < 1.0);
    assert!(eg > 3.0);
}

#[test]
fn tilting_drops_the_scoop_as_particles() {
    let Some(mut g) = gpu() else { return };
    let b = &bowls()[2];
    drive(&mut g, Vec3::new(b.cx, 0.14, 0.0), Quat::IDENTITY, 60, |_| {});
    drive(&mut g, Vec3::new(b.cx, 0.085, 0.0), Quat::IDENTITY, 150, |_| {});
    drive(&mut g, Vec3::new(b.cx, 0.15, 0.0), Quat::IDENTITY, 120, |_| {});
    let scooped = g.state().scooped_grams;
    let m0 = g.total_grams();
    let mut drift = 0f64;
    let mut peak = 0;
    let flip = Quat::from_euler(PI * 0.9, 0.0, 0.0);
    drive(&mut g, Vec3::new(b.cx, 0.15, 0.0), flip, 120, |g| {
        drift = drift.max((g.total_grams() - m0).abs());
        peak = peak.max(g.stats.n);
    });
    let after = g.state().scooped_grams;
    for _ in 0..240 {
        g.step(1.0 / 60.0);
    }
    drift = drift.max((g.total_grams() - m0).abs());
    eprintln!("tip-out: scooped {scooped:.1} g, after the flip {after:.2} g, {peak} particles at the peak, {} loose 4 s later, drift {drift:.3} g", g.stats.n);
    assert!(scooped > 2.0 && after < 0.5 && peak > 100 && drift < 1.0);
}

#[test]
fn plunging_and_eating_keeps_the_mass() {
    let Some(mut g) = gpu() else { return };
    let b = &bowls()[0];
    g.step(1.0 / 60.0);
    let m0 = g.total_grams();
    let mut drift = 0f64;
    let mut eaten = 0.0;
    for _ in 0..3 {
        for (y, n) in [(0.14, 60), (0.0, 240), (0.15, 150)] {
            drive(&mut g, Vec3::new(b.cx, y, 0.0), Quat::IDENTITY, n, |g| drift = drift.max((g.total_grams() - m0).abs()));
        }
        eaten += g.eat();
        drift = drift.max((g.total_grams() - m0).abs());
    }
    eprintln!("3 plunges: eaten {eaten:.1} g of {m0:.1} g, worst drift {drift:.3} g");
    assert!(drift < 1.5 && eaten > 10.0);
}

#[test]
fn the_same_run_twice_is_bit_identical() {
    let Some(mut a) = gpu() else { return };
    let Some(mut b) = gpu() else { return };
    let bw = &bowls()[1];
    let dig = Quat::from_euler(0.0, 0.0, 0.5);
    let grab = |g: &mut GpuSim| {
        drive(g, Vec3::new(bw.cx - 0.03, 0.14, 0.0), dig, 60, |_| {});
        drive(g, Vec3::new(bw.cx - 0.02, 0.085, 0.0), dig, 120, |_| {});
        drive(g, Vec3::new(bw.cx + 0.025, 0.088, 0.0), Quat::IDENTITY, 120, |_| {});
        let l = &g.l;
        (g.debug_read('W', l.w.get("D"), l.n), g.debug_read('W', l.w.get("POS"), 3 * g.stats.n), g.stats.n)
    };
    let (ra, rb) = (grab(&mut a), grab(&mut b));
    eprintln!("two runs: {} vs {} particles", ra.2, rb.2);
    assert!(ra == rb, "the runs differ");
}

#[test]
fn many_loose_particles_frame_time() {
    let Some(mut g) = gpu() else { return };
    let n = 15000usize;
    g.cpu.params.fluid.max_particles = n;
    g.cpu.params.fluid.stop_speed = 0.0;
    let fp = g.cpu.params.fluid.clone();
    g.cpu.fluid.configure(fp.particle_radius, fp.kernel_scale);
    let s = g.cpu.fluid.d * 1.05;
    let side = (n as f64 / 12.0).sqrt().ceil() as usize;
    let mut pos = vec![0f32; 3 * n];
    for k in 0..n {
        let (i, j, l) = (k % side, (k / side) % side, k / side / side);
        pos[3 * k] = (-0.18 + (i as f64 - side as f64 / 2.0) * s) as f32;
        pos[3 * k + 1] = (0.13 + l as f64 * s) as f32;
        pos[3 * k + 2] = ((j as f64 - side as f64 / 2.0) * s) as f32;
    }
    let l = g.l.clone();
    g.debug_write('W', l.w.get("POS"), bytemuck::cast_slice(&pos));
    g.debug_write('W', l.w.get("FPOS"), bytemuck::cast_slice(&pos));
    g.debug_write('W', l.w.get("SLOW"), bytemuck::cast_slice(&vec![1f32; n]));
    g.debug_write('X', l.x.get("AGE"), bytemuck::cast_slice(&vec![10u32; n]));
    g.debug_write('X', l.x.get("RGBA"), bytemuck::cast_slice(&vec![0x00c8_e6f0u32; n]));
    g.debug_write('X', l.x.get("ALIVE"), bytemuck::cast_slice(&vec![1u32; n]));
    g.debug_write('A', l.a.get("CNT") + cnt::N, bytemuck::cast_slice(&[n as u32]));
    let mut t = Vec::new();
    for _ in 0..60 {
        let t0 = Instant::now();
        g.step(1.0 / 60.0);
        t.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("{n} loose particles on {}: {:.2} ms/frame, {} left", g.adapter, t[30], g.stats.n);
    assert_eq!(g.stats.n as usize, n);
    assert!(n <= CAP as usize);
}

#[test]
fn the_arm_carries_a_gpu_scoop() {
    let Some(g) = gpu() else { return };
    let mut t = GpuTwin::from_sim(g);
    let mut worst = 0f64;
    for (p, pitch, n) in [
        (Vec3::new(0.0, 0.14, 0.02), 0.5, 80),
        (Vec3::new(0.0, 0.086, 0.015), 0.5, 150),
        (Vec3::new(0.0, 0.088, -0.025), 0.0, 150),
        (Vec3::new(0.0, 0.15, -0.03), 0.0, 100),
    ] {
        t.sim.cpu.spoon.set_target(Some(p), Some(facing(pitch)));
        for _ in 0..n {
            t.step(1.0 / 60.0);
            worst = worst.max(t.arm.err.mm);
        }
    }
    let s = t.sim.state();
    eprintln!("arm scoop on the GPU: {:.1} g, arm within {worst:.4} mm of the spoon", s.scooped_grams);
    assert!(s.scooped_grams > 10.0 && worst < 0.5);
}

/// Loose particles near the cup (y above `NEAR`): height (2nd to 98th percentile) and lowest point
/// (2nd percentile) in metres, how many, and how many are below (in the bowl). From n x (f32 x, y, z, rgba).
fn loose_extent(bytes: &[u8]) -> (f64, f64, usize, usize) {
    let mut ys: Vec<f64> = bytes.as_chunks::<16>().0.iter().map(|p| f32::from_le_bytes(p[4..8].try_into().unwrap()) as f64).collect();
    let below = ys.iter().filter(|&&y| y < NEAR).count();
    ys.retain(|&y| y >= NEAR);
    ys.sort_by(f64::total_cmp);
    if ys.is_empty() {
        return (0.0, 0.0, 0, below);
    }
    let at = |q: f64| ys[((ys.len() - 1) as f64 * q) as usize];
    (at(0.98) - at(0.02), at(0.02), ys.len(), below)
}
const NEAR: f64 = 0.12;

#[test]
fn a_released_scoop_keeps_its_shape_like_the_cpu() {
    let Some(mut g) = gpu() else { return };
    let mut c = Sim::default();
    // a dragged scoop (scoop_carry_eat_like_the_cpu's motion), lifted level over the bowl
    let b = &bowls()[0];
    let dig = Quat::from_euler(0.0, 0.0, 0.5);
    let script = [
        (Vec3::new(b.cx - 0.03, 0.14, 0.0), dig, 90),
        (Vec3::new(b.cx - 0.02, 0.085, 0.0), dig, 150),
        (Vec3::new(b.cx + 0.025, 0.088, 0.0), Quat::IDENTITY, 150),
        (Vec3::new(b.cx + 0.025, 0.15, 0.0), Quat::IDENTITY, 120),
    ];
    for (p, q, n) in script {
        drive(&mut g, p, q, n, |_| {});
        c.set_target(Some(p), Some(q));
        for _ in 0..n {
            c.step(1.0 / 60.0);
        }
    }
    let cpu_bytes = |c: &Sim| {
        let mut out = Vec::new();
        for i in 0..c.fluid.n {
            for k in 0..3 {
                out.extend_from_slice(&c.fluid.pos[i * 3 + k].to_le_bytes());
            }
            out.extend_from_slice(&[0; 4]);
        }
        out
    };
    g.stream = true;
    g.step(1.0 / 60.0);
    c.step(1.0 / 60.0);
    eprintln!("scooped: GPU {:.1} g, CPU {:.1} g", g.state().scooped_grams, c.state().scooped_grams);
    let g_before = loose_extent(&g.particles).3;
    let c_before = loose_extent(&cpu_bytes(&c)).3;
    // tip it out on a level spoon: the scoop (one lump) should sit in the cup, not slump over the rim
    g.release();
    c.release();
    g.step(1.0 / 60.0);
    c.step(1.0 / 60.0);
    let (gh0, ch0) = (loose_extent(&g.particles).0, loose_extent(&cpu_bytes(&c)).0);
    for _ in 0..19 {
        g.step(1.0 / 60.0);
        c.step(1.0 / 60.0);
    }
    let (gh, glo, gn, gb) = loose_extent(&g.particles);
    let (ch, clo, cn, cbw) = loose_extent(&cpu_bytes(&c));
    eprintln!(
        "released lump after 20 frames: GPU {gn} in the cup, {:.1} mm tall (at release {:.1}), lowest {:.1} mm, {} more in the bowl; CPU {cn}, {:.1} mm, {:.1} mm, {}",
        gh * 1e3,
        gh0 * 1e3,
        glo * 1e3,
        gb as i64 - g_before as i64,
        ch * 1e3,
        clo * 1e3,
        cbw as i64 - c_before as i64
    );
    assert!(gn > 100 && cn > 100);
    // as tall as when it was released, within the plastic squash (a liquid would be a few mm)
    assert!(gh > 0.7 * gh0 && ch > 0.7 * ch0, "the lump flattened: {:.1} -> {:.1} mm (GPU), {:.1} -> {:.1} mm (CPU)", gh0 * 1e3, gh * 1e3, ch0 * 1e3, ch * 1e3);
    assert!(gb <= g_before + 5 && cbw <= c_before + 5, "particles fell through or over the cup");
    assert!((gh - ch).abs() < 0.3 * ch);
}

/// Straight down into each bowl, held there, lifted: GPU and CPU scoop about the same, and holding the spoon
/// still gains nothing (smoothing used to refill the cut next to the spoon, which cut it again: a pump that
/// filled the cup to capacity, or not, depending on chance).
#[test]
fn a_held_spoon_scoops_nothing_more_in_every_bowl() {
    for bi in 0..3 {
        let Some(mut g) = gpu() else { return };
        let mut c = Sim::default();
        let b = &bowls()[bi];
        let legs = [(Vec3::new(b.cx, 0.14, 0.0), 60), (Vec3::new(b.cx, 0.085, 0.0), 150), (Vec3::new(b.cx, 0.15, 0.0), 120)];
        let (mut frame, mut held) = (0, [[0.0; 2]; 2]);
        for (p, n) in legs {
            g.cpu.spoon.set_target(Some(p), Some(Quat::IDENTITY));
            c.set_target(Some(p), Some(Quat::IDENTITY));
            for _ in 0..n {
                g.step(1.0 / 60.0);
                c.step(1.0 / 60.0);
                frame += 1;
                // the spoon reaches the bottom by frame ~140 and stays until 210
                if frame == 160 || frame == 210 {
                    held[usize::from(frame == 210)] = [g.state().scooped_grams, c.state().scooped_grams];
                }
            }
        }
        let (gs, cs) = (g.state().scooped_grams, c.state().scooped_grams);
        eprintln!(
            "bowl {bi}: held at the bottom {:.2} -> {:.2} g (GPU), {:.2} -> {:.2} g (CPU); lifted out {gs:.1} g vs {cs:.1} g",
            held[0][0], held[1][0], held[0][1], held[1][1]
        );
        for (before, after) in held[0].iter().zip(&held[1]) {
            // (crumbs settling onto the scoop add a little; the pump added ~10 g)
            assert!(after - before < 1.0, "bowl {bi}: the held spoon kept scooping");
        }
        // (whole runs drift apart: see scoop_carry_eat_like_the_cpu; and lifting straight out of a hole
        // the spoon pressed brings up little: the ice pushed aside lies beside it, not above the cup)
        assert!((gs - cs).abs() < 0.4 * cs + 2.0, "bowl {bi}: GPU {gs:.1} g vs CPU {cs:.1} g");
    }
}

/// A scoop dropped on the table freezes into the table's SDF on the GPU too (not static crumbs), as much
/// as on the CPU, keeping the mass; then it is scraped back up.
#[test]
fn a_scoop_dropped_on_the_table_freezes_there_like_the_cpu() {
    let Some(mut g) = gpu() else { return };
    let mut c = Sim::default();
    let b = &bowls()[2];
    let level = Quat::IDENTITY;
    let scrape = Quat::from_euler(0.0, 0.0, 1.45); // opening toward -x, handle up
    let legs = [
        (Vec3::new(b.cx, 0.14, 0.0), level, 60),
        (Vec3::new(b.cx, 0.085, 0.0), level, 150),
        (Vec3::new(b.cx, 0.15, 0.0), level, 120),
        (Vec3::new(0.27, 0.12, 0.12), level, 90),
        (Vec3::new(0.27, 0.12, 0.12), Quat::from_euler(PI * 0.9, 0.0, 0.0), 120),
        (Vec3::new(0.27, 0.12, 0.12), Quat::from_euler(PI * 0.9, 0.0, 0.0), 240),
    ];
    g.step(1.0 / 60.0);
    c.step(1.0 / 60.0);
    let (g0, c0) = (g.total_grams(), c.total_grams());
    let mut scooped = (0.0, 0.0);
    for (k, (p, q, n)) in legs.into_iter().enumerate() {
        drive(&mut g, p, q, n, |_| {});
        c.set_target(Some(p), Some(q));
        c.step_frames(n);
        if k == 2 {
            scooped = (g.state().scooped_grams, c.state().scooped_grams);
        }
    }
    c.update_stats(true);
    let (gt, ct) = (g.state().table_grams, c.state().table_grams);
    eprintln!(
        "dropped on the table: GPU {:.1} g -> {gt:.1} g frozen there, {} crumbs, {} loose; CPU {:.1} g -> {ct:.1} g, {} crumbs, {} loose; mass drift {:.3} / {:.3} g",
        scooped.0, g.stats.spilled, g.stats.n, scooped.1, c.n_spilled, c.fluid.n, g.total_grams() - g0, c.total_grams() - c0
    );
    assert!(gt > 0.6 * scooped.0 && ct > 0.6 * scooped.1, "it should freeze onto the table");
    assert!(g.stats.spilled < 20 && c.n_spilled < 20, "not as static crumbs");
    assert!((g.total_grams() - g0).abs() < 0.2 && (c.total_grams() - c0).abs() < 0.2);

    g.eat();
    c.eat();
    for (p, q, n) in [
        (Vec3::new(0.34, 0.09, 0.13), scrape, 120),
        (Vec3::new(0.34, 0.022, 0.13), scrape, 90),
        (Vec3::new(0.22, 0.022, 0.13), scrape, 180),
        (Vec3::new(0.22, 0.09, 0.13), level, 120),
    ] {
        drive(&mut g, p, q, n, |_| {});
        c.set_target(Some(p), Some(q));
        c.step_frames(n);
    }
    let (ga, ca) = (g.state().scooped_grams, c.state().scooped_grams);
    eprintln!("scraped back up: GPU {ga:.1} g, CPU {ca:.1} g; mass drift {:.3} / {:.3} g", g.total_grams() - g0, c.total_grams() - c0);
    assert!(ga > 0.5 && ca > 0.5);
    assert!((g.total_grams() - g0).abs() < 0.2);
}
