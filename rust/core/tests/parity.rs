//! Regression against frozen fixtures. They were first written by the JS simulation this project started
//! with (and its SO-101 arm), which the Rust core matched bit for bit; the JS is gone, the fixtures stay, so
//! any change to the numerics shows up here. Behaviour changes on purpose: update the expectations, don't
//! loosen the comparison. `UPDATE_FIXTURES=1 cargo test -p icecream-core --test parity` rewrites scoop,
//! plunge and arm from the Rust core (then run it again without). Rewritten so far: smoothing keeps out of
//! the spoon (scoop, plunge, arm).

use icecream_core::math::{self, Quat, Vec3};
use icecream_core::sdf::{bowls, FLAVORS};
use icecream_core::sim::flavor_rgba;
use icecream_core::robot::Arm;
use icecream_core::Sim;

/// Compare against the fixture, or with UPDATE_FIXTURES set, collect the row to rewrite it.
fn check(name: &str, names: &[&str], rows: &[Vec<f64>], frame: usize, got: &[f64], out: &mut Vec<Vec<f64>>) {
    if update() {
        let mut row = vec![frame as f64];
        row.extend_from_slice(got);
        out.push(row);
        return;
    }
    let r = &rows[frame - 1];
    for k in 0..got.len() {
        assert!(
            got[k] == r[k + 1] || (got[k] - r[k + 1]).abs() <= 1e-15 * r[k + 1].abs(),
            "{name}: frame {frame}: {} now {:e} fixture {:e}",
            names[k],
            got[k],
            r[k + 1]
        );
    }
}

fn update() -> bool {
    std::env::var_os("UPDATE_FIXTURES").is_some()
}

fn finish(name: &str, names: &[&str], rows: &[Vec<f64>], frame: usize, out: Vec<Vec<f64>>) {
    if update() {
        let mut csv = format!("frame,{}\n", names.join(","));
        for r in out {
            csv += &format!("{}\n", r.iter().enumerate().map(|(k, v)| if k == 0 { format!("{}", *v as usize) } else { format!("{v}") }).collect::<Vec<_>>().join(","));
        }
        let path = format!("{}/tests/fixtures/{name}.csv", env!("CARGO_MANIFEST_DIR"));
        std::fs::write(&path, csv).unwrap();
        println!("{name}: wrote {frame} frames to {path}");
        return;
    }
    assert_eq!(frame, rows.len(), "{name}: fixture length");
    println!("{name}: {frame} frames identical to the fixture");
}

enum Step {
    Go([f64; 6], usize),
    Eat,
}

fn parse(csv: &str) -> Vec<Vec<f64>> {
    csv.lines().skip(1).filter(|l| !l.is_empty()).map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect()
}

fn run(name: &str, csv: &str, script: &[Step]) {
    let rows = parse(csv);
    let mut sim = Sim::default();
    let mut frame = 0usize;
    let mut out = Vec::new();
    for s in script {
        match s {
            Step::Eat => {
                sim.eat();
            }
            Step::Go(t, n) => {
                sim.set_target(Some(Vec3::new(t[0], t[1], t[2])), Some(Quat::from_euler(t[3], t[4], t[5])));
                for _ in 0..*n {
                    sim.step(1.0 / 60.0);
                    frame += 1;
                    let sp = &sim.spoon;
                    let got = [
                        sp.pos.x,
                        sp.pos.y,
                        sp.pos.z,
                        sp.quat.x,
                        sp.quat.y,
                        sp.quat.z,
                        sp.quat.w,
                        sim.content.volume(),
                        sim.grids[0].volume(),
                        sim.fluid.n as f64,
                        sim.debt_volume(),
                        sim.cut_force,
                        sim.eaten_volume,
                    ];
                    check(name, &NAMES, &rows, frame, &got, &mut out);
                }
            }
        }
    }
    finish(name, &NAMES, &rows, frame, out);
}

const NAMES: [&str; 13] = ["px", "py", "pz", "qx", "qy", "qz", "qw", "content", "bowl0", "particles", "debt", "cut", "eaten"];

#[test]
fn scoop_matches_fixture() {
    let b = &bowls()[0];
    run(
        "scoop",
        include_str!("fixtures/scoop.csv"),
        &[
            Step::Go([b.cx - 0.03, 0.14, 0.0, 0.0, 0.0, 0.5], 90),
            Step::Go([b.cx - 0.02, 0.085, 0.0, 0.0, 0.0, 0.5], 150),
            Step::Go([b.cx + 0.025, 0.088, 0.0, 0.0, 0.0, 0.0], 150),
            Step::Go([b.cx + 0.025, 0.16, 0.0, 0.0, 0.0, 0.0], 120),
        ],
    );
}

#[test]
fn plunge_matches_fixture() {
    let b = &bowls()[0];
    let mut script = Vec::new();
    for _ in 0..2 {
        script.push(Step::Go([b.cx, 0.14, 0.0, 0.0, 0.0, 0.0], 60));
        script.push(Step::Go([b.cx, 0.0, 0.0, 0.0, 0.0, 0.0], 240));
        script.push(Step::Go([b.cx, 0.15, 0.0, 0.0, 0.0, 0.0], 150));
        script.push(Step::Eat);
    }
    run("plunge", include_str!("fixtures/plunge.csv"), &script);
}

enum ArmStep {
    Go([f64; 4], usize),
    Joints([f64; 5], usize),
}

/// The SO-101 driving the spoon (`arm.constrain` -> `sim.step` -> `arm.follow`), first written by arm.js.
#[test]
fn arm_matches_fixture() {
    use std::f64::consts::PI;
    let script = [
        ArmStep::Go([0.0, 0.14, 0.02, 0.5], 80),
        ArmStep::Go([0.0, 0.086, 0.015, 0.5], 150),
        ArmStep::Go([0.0, 0.088, -0.025, 0.0], 150),
        ArmStep::Go([0.0, 0.15, -0.03, 0.0], 100),
        ArmStep::Joints([0.3, -1.5, 0.6, 1.0, 0.4], 150),
        ArmStep::Go([0.18, 0.15, 0.0, 0.3], 200),
    ];
    let rows = parse(include_str!("fixtures/arm.csv"));
    let mut sim = Sim::default();
    let mut arm = Arm::default();
    arm.reset(Some(&mut sim.spoon));
    let mut frame = 0usize;
    let mut out = Vec::new();
    let names = [
        "px", "py", "pz", "qx", "qy", "qz", "qw", "q0", "q1", "q2", "q3", "q4", "c0", "c1", "c2", "c3", "c4", "errMm", "errDeg", "content",
    ];
    for s in &script {
        let n = match s {
            ArmStep::Go(t, n) => {
                let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI / 2.0).mul(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), t[3]));
                sim.set_target(Some(Vec3::new(t[0], t[1], t[2])), Some(q));
                *n
            }
            ArmStep::Joints(j, n) => {
                arm.set_joints(*j);
                *n
            }
        };
        for _ in 0..n {
            arm.constrain(&mut sim.spoon, 1.0 / 60.0);
            sim.step(1.0 / 60.0);
            arm.follow(&sim.spoon);
            frame += 1;
            let sp = &sim.spoon;
            let mut got = vec![sp.pos.x, sp.pos.y, sp.pos.z, sp.quat.x, sp.quat.y, sp.quat.z, sp.quat.w];
            got.extend(arm.q);
            got.extend(arm.q_cmd);
            got.extend([arm.err.mm, arm.err.deg, sim.content.volume()]);
            check("arm", &names, &rows, frame, &got, &mut out);
        }
    }
    finish("arm", &names, &rows, frame, out);
}

#[test]
fn math_matches_v8() {
    let mut n = 0;
    for r in parse(include_str!("fixtures/math.csv")) {
        let x = r[0];
        let u = x / (std::f64::consts::PI * 4.0);
        let got = [math::cos(x), math::sin(x), math::cbrt(x * 1e-7), math::acos(u), math::atan2(x - 3.0, 1.3), math::asin(u)];
        for k in 0..got.len() {
            assert_eq!(got[k].to_bits(), r[k + 1].to_bits(), "fn #{k} at x={x:e}: rust {:e} js {:e}", got[k], r[k + 1]);
        }
        n += 1;
    }
    assert_eq!(n, 4000);
}

#[test]
fn flavor_colors_match_js() {
    let rows = parse(&format!("h\n{}", include_str!("fixtures/flavors.csv")));
    for (i, f) in FLAVORS.iter().enumerate() {
        let c = flavor_rgba(f.color, f.speck);
        let js: Vec<u8> = rows[i].iter().map(|v| *v as u8).collect();
        assert_eq!(c.to_vec(), js, "flavor {}", f.id);
    }
}
