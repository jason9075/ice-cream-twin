//! Port of tests/collision.test.js: the spoon never penetrates the bowls or the table.

use icecream_core::math::{hypot3, Quat, Vec3};
use icecream_core::params::SpoonParams;
use icecream_core::sdf::{bowls, Bowl};
use icecream_core::spoon::{SpoonController, PEN_TOL};
use std::f64::consts::PI;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 as f64 / 4294967296.0
    }
}

fn random_quat(r: &mut Rng) -> Quat {
    let u1 = r.next();
    let u2 = r.next() * PI * 2.0;
    let u3 = r.next() * PI * 2.0;
    let a = (1.0 - u1).sqrt();
    let b = u1.sqrt();
    Quat::new(a * u2.sin(), a * u2.cos(), b * u3.sin(), b * u3.cos())
}

/// Signed side of the bowl wall for every sample below the rim: -1 inside, +1 outside.
fn wall_sides(sp: &SpoonController, b: &Bowl) -> Vec<i8> {
    sp.samples
        .iter()
        .map(|s| {
            let w = Vec3::new(s[0] as f64, s[1] as f64, s[2] as f64).apply_quat(sp.quat).add(sp.pos);
            if w.y > b.rim_y - 0.004 {
                return 0;
            }
            let d = hypot3(w.x - b.cx, w.y - b.cy, w.z - b.cz) - b.r;
            if d < 0.0 {
                -1
            } else {
                1
            }
        })
        .collect()
}

#[test]
fn spoon_never_penetrates_for_random_targets() {
    let p = SpoonParams { max_lin_speed: 0.6, max_ang_speed: 400.0, ..Default::default() };
    let mut sp = SpoonController::new(p);
    let mut r = Rng(7);
    let mut worst = f64::NEG_INFINITY;
    for trial in 0..120 {
        let b = &bowls()[(r.next() * 3.0).floor() as usize];
        let pos = Vec3::new(b.cx + (r.next() - 0.5) * 0.22, r.next() * 0.16, (r.next() - 0.5) * 0.22);
        let q = random_quat(&mut r);
        sp.set_target(Some(pos), Some(q));
        for f in 0..40 {
            sp.update(1.0 / 60.0, 1.0, None);
            let pen = sp.penetration(sp.pos, sp.quat);
            worst = worst.max(pen);
            assert!(pen <= PEN_TOL + 1e-6, "trial {trial} frame {f}: penetration {pen}");
            assert!((sp.pos.x + sp.quat.w).is_finite());
        }
    }
    println!("worst penetration {:.3} mm (tolerance {} mm)", worst * 1000.0, PEN_TOL * 1000.0);
}

#[test]
fn fast_push_through_the_wall_cannot_tunnel() {
    let p = SpoonParams { max_lin_speed: 3.0, max_ang_speed: 720.0, ..Default::default() };
    let mut sp = SpoonController::new(p);
    let b = &bowls()[1];
    sp.pos = Vec3::new(b.cx, b.cy - 0.03, 0.0);
    sp.quat = Quat::IDENTITY;
    sp.set_target(Some(sp.pos), Some(sp.quat));
    for _ in 0..10 {
        sp.update(1.0 / 60.0, 1.0, None);
    }
    for dir in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.7, -0.7, 0.0]] {
        sp.pos = Vec3::new(b.cx, b.cy - 0.03, 0.0);
        sp.quat = Quat::IDENTITY;
        sp.update(1.0 / 60.0, 1.0, None);
        let mut sides = wall_sides(&sp, b);
        let target = Vec3::new(b.cx + dir[0] * 0.3, b.cy - 0.03 + dir[1] * 0.3, dir[2] * 0.3);
        sp.set_target(Some(target), Some(Quat::from_euler(0.3, 1.0, -0.4)));
        for _ in 0..90 {
            sp.update(1.0 / 60.0, 1.0, None);
            let now = wall_sides(&sp, b);
            for i in 0..now.len() {
                if sides[i] != 0 && now[i] != 0 {
                    assert_eq!(now[i], sides[i], "sample {i} crossed the wall (dir {dir:?})");
                }
            }
            sides = now;
            assert!(sp.penetration(sp.pos, sp.quat) <= PEN_TOL + 1e-6);
        }
    }
}

#[test]
fn spoon_slides_along_the_wall() {
    let mut sp = SpoonController::new(SpoonParams::default());
    let b = &bowls()[0];
    sp.pos = Vec3::new(b.cx, b.cy - 0.03, 0.0);
    sp.quat = Quat::IDENTITY;
    sp.set_target(Some(Vec3::new(b.cx + 0.2, b.cy - 0.03, 0.05)), Some(sp.quat));
    for _ in 0..120 {
        sp.update(1.0 / 60.0, 1.0, None);
    }
    assert!(sp.contacts > 0, "should end in contact with the wall");
    assert!(sp.pos.z > 0.01, "should slide sideways toward z (got z={})", sp.pos.z);
}
