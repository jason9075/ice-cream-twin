//! Port of tests/sim.test.js: scooping, eating, tip-off release and mass conservation.

use icecream_core::math::{Quat, Vec3};
use icecream_core::sdf::bowls;
use icecream_core::spoon::PEN_TOL;
use icecream_core::{frames, Sim};
use std::f64::consts::PI;
use std::time::Instant;

fn run_to(sim: &mut Sim, pos: Vec3, q: Quat, frames: usize) {
    sim.set_target(Some(pos), Some(q));
    for _ in 0..frames {
        sim.step(1.0 / 60.0);
        assert!(sim.spoon.penetration(sim.spoon.pos, sim.spoon.quat) <= PEN_TOL + 1e-6);
    }
}

#[test]
fn scoop_vanilla_carry_and_eat() {
    let mut sim = Sim::default();
    sim.update_stats(true);
    let before = sim.total_grams();
    let b = &bowls()[0];
    let level = Quat::IDENTITY;
    let dig = Quat::from_euler(0.0, 0.0, 0.5);
    let t0 = Instant::now();
    run_to(&mut sim, Vec3::new(b.cx - 0.03, 0.14, 0.0), dig, 90);
    run_to(&mut sim, Vec3::new(b.cx - 0.02, 0.085, 0.0), dig, 150);
    run_to(&mut sim, Vec3::new(b.cx + 0.025, 0.088, 0.0), level, 150);
    run_to(&mut sim, Vec3::new(b.cx + 0.025, 0.16, 0.0), level, 120);
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 510.0;
    sim.update_stats(true);
    let s = sim.state();
    println!(
        "scooped {:.1} g, particles {}, vanilla left {:.1} g, {:.2} ms/frame",
        s.scooped_grams, s.particles, s.bowl_grams[0], ms
    );
    assert!(s.scooped_grams > 3.0, "expected a real scoop, got {} g", s.scooped_grams);
    assert!(s.spoon_position[1] > 0.15, "spoon should have lifted out");
    let after = sim.total_grams();
    println!("mass before {before:.1} g, after {after:.1} g");
    assert!((after - before).abs() / before < 0.01, "mass should be conserved");
    let eaten = sim.eat();
    assert!(eaten > 3.0);
    assert_eq!(sim.state().scooped_grams, 0.0);
}

#[test]
fn tilting_past_release_angle_drops_the_scoop() {
    let mut sim = Sim::default();
    let b = &bowls()[2];
    run_to(&mut sim, Vec3::new(b.cx, 0.14, 0.0), Quat::IDENTITY, 60);
    run_to(&mut sim, Vec3::new(b.cx, 0.085, 0.0), Quat::IDENTITY, 150);
    run_to(&mut sim, Vec3::new(b.cx, 0.15, 0.0), Quat::IDENTITY, 120);
    let scooped = sim.state().scooped_grams;
    assert!(scooped > 2.0, "scooped {scooped}");
    let flip = Quat::from_euler(PI * 0.9, 0.0, 0.0);
    run_to(&mut sim, Vec3::new(b.cx, 0.15, 0.0), flip, 120);
    assert!(sim.state().scooped_grams < 0.5, "scoop should have fallen off");
    sim.step_frames(240);
    println!("after release: particles left {}", sim.fluid.n);
}

#[test]
fn plunging_to_the_bottom_keeps_the_mass() {
    let mut sim = Sim::default();
    sim.update_stats(true);
    let before = sim.total_grams();
    let b = &bowls()[0];
    let level = Quat::IDENTITY;
    let t0 = Instant::now();
    let mut grams = Vec::new();
    for _ in 0..3 {
        run_to(&mut sim, Vec3::new(b.cx, 0.14, 0.0), level, 60);
        run_to(&mut sim, Vec3::new(b.cx, 0.0, 0.0), level, 240);
        run_to(&mut sim, Vec3::new(b.cx, 0.15, 0.0), level, 150);
        grams.push(sim.state().scooped_grams);
        sim.eat();
    }
    // the same hole each time: the cut walls stay, so later plunges bring up less
    println!("plunges: {grams:.1?} g");
    assert!(grams[0] > 5.0 && grams.iter().sum::<f64>() > 15.0, "plunges brought up {grams:?} g");
    let after = sim.total_grams();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 1350.0;
    println!("plunge x3: mass {before:.1} -> {after:.1} g, eaten {:.1} g, {ms:.2} ms/frame", sim.state().eaten_grams);
    assert!((after - before).abs() / before < 0.01, "mass must be conserved within 1%");
    assert!(
        sim.spilled_volume <= sim.n_spilled as f64 * sim.particle_volume() * 1.01 + 1e-12,
        "everything counted as spilled is a visible crumb"
    );
}

#[test]
fn ros_frame_roundtrip() {
    let p = [0.1, 0.2, 0.3];
    assert_eq!(frames::pos_from_ros(frames::pos_to_ros(p)), p);
    assert_eq!(frames::pos_to_ros([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], "scene up is ROS z");
    let q = Quat::from_euler(0.3, -0.5, 1.2).to_array();
    let back = frames::quat_from_ros(frames::quat_to_ros(q));
    for k in 0..4 {
        assert!((back[k] - q[k]).abs() < 1e-12);
    }
}
