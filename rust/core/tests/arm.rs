//! The SO-101 holding the spoon: port of tests/arm.test.js.
#![allow(clippy::needless_range_loop)]

use icecream_core::math::{Quat, Vec3};
use icecream_core::robot::so101::GRIPPER_LINK;
use icecream_core::robot::tool::{tool, TOOL_NAMES};
use icecream_core::robot::Arm;
use icecream_core::{Twin, PEN_TOL};
use std::f64::consts::PI;

/// Spoon handle toward the robot (it stands at -Z), cup pitched nose-down by `pitch`.
fn facing(pitch: f64) -> Quat {
    Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI / 2.0).mul(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), pitch))
}

fn drive(t: &mut Twin, pos: Vec3, quat: Quat, frames: usize, mut check: impl FnMut(&Twin)) {
    t.sim.set_target(Some(pos), Some(quat));
    for _ in 0..frames {
        t.step(1.0 / 60.0);
        check(t);
    }
}

#[test]
fn fk_ik_round_trip_over_the_joint_range() {
    let arm = Arm::default();
    let lim = arm.limits();
    let mut seed = 7.0f64;
    let mut rnd = || {
        seed = (seed * 16807.0) % 2147483647.0;
        seed / 2147483647.0
    };
    let mut fails = 0;
    for _ in 0..200 {
        let mut q = [0.0; 5];
        for k in 0..5 {
            q[k] = lim[k][0] + (lim[k][1] - lim[k][0]) * rnd();
        }
        let pose = arm.spoon_pose(&q);
        let mut start = [0.0; 5];
        for k in 0..5 {
            start[k] = lim[k][1].min(lim[k][0].max(q[k] + (rnd() - 0.5) * 0.3));
        }
        let e = arm.solve(&mut start, pose.pos, pose.quat, 100);
        if e.mm > 0.05 || e.deg > 0.05 {
            fails += 1;
        }
    }
    assert!(fails <= 2, "{fails} / 200 poses not recovered");
}

#[test]
fn the_tool_mount_is_rigid_and_the_jaw_is_clamped_inside_its_range() {
    for name in TOOL_NAMES {
        let t = tool(name).unwrap();
        let m = t.spoon_in_gripper;
        for a in 0..3 {
            for b in 0..3 {
                let dot = m[a] * m[b] + m[4 + a] * m[4 + b] + m[8 + a] * m[8 + b];
                assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-12, "{name} rotation not orthonormal");
            }
        }
        if let Some(j) = t.jaw {
            assert!((-0.1746..=1.7454).contains(&j));
        }
    }
    // the spoon rim sits a handle's length from gripper_link (world frame, both)
    let arm = Arm::default();
    let s = arm.spoon_pose(&arm.q);
    let w = arm.link_matrices()[GRIPPER_LINK];
    let d = s.pos.distance_to(Vec3::new(w[3], w[7], w[11]));
    assert!(d > 0.05 && d < 0.25, "spoon {d:.3} m from gripper_link");
}

#[test]
fn the_arm_carries_a_scoop_out_of_the_middle_bowl_and_stays_on_the_spoon() {
    let mut t = Twin::default();
    let lim = t.arm.limits();
    let mut worst = 0f64;
    let mut check = |t: &Twin| {
        worst = worst.max(t.arm.err.mm);
        assert!(t.sim.spoon.penetration(t.sim.spoon.pos, t.sim.spoon.quat) <= PEN_TOL + 1e-6);
        for k in 0..5 {
            assert!(t.arm.q[k] >= lim[k][0] - 1e-9 && t.arm.q[k] <= lim[k][1] + 1e-9);
        }
    };
    drive(&mut t, Vec3::new(0.0, 0.14, 0.02), facing(0.5), 80, &mut check);
    drive(&mut t, Vec3::new(0.0, 0.086, 0.015), facing(0.5), 150, &mut check);
    drive(&mut t, Vec3::new(0.0, 0.088, -0.025), facing(0.0), 150, &mut check);
    drive(&mut t, Vec3::new(0.0, 0.15, -0.03), facing(0.0), 100, &mut check);
    let st = t.sim.state();
    assert!(st.scooped_grams > 10.0, "scooped only {:.1} g", st.scooped_grams);
    assert!(worst < 0.5, "arm drifted {worst:.3} mm from the spoon");
}

#[test]
fn moving_between_bowls_goes_through_poses_the_arm_can_hold() {
    let mut t = Twin::default();
    let mut worst = 0f64;
    for x in [-0.18, 0.18, 0.0] {
        drive(&mut t, Vec3::new(x, 0.15, 0.0), facing(0.3), 240, |t| worst = worst.max(t.arm.err.mm));
    }
    assert!(worst < 1.0, "arm drifted {worst:.3} mm from the spoon");
    assert!(t.sim.spoon.pos.distance_to(t.arm.goal.pos) < 0.002);
}

#[test]
fn joint_space_command_settles_on_the_commanded_angles() {
    let mut t = Twin::default();
    let cmd = [0.3, -1.5, 0.6, 1.0, 0.4]; // spoon ~23 cm up, clear of the bowls
    t.arm.set_joints(cmd);
    t.step_frames(300);
    let p = t.arm.spoon_pose(&cmd);
    assert!(t.sim.spoon.pos.distance_to(p.pos) < 0.001);
    assert!((0..5).all(|k| (t.arm.q[k] - cmd[k]).abs() < 0.01), "q = {:?}", t.arm.q);
}

#[test]
fn reset_puts_the_spoon_back_in_the_gripper() {
    let mut t = Twin::default();
    drive(&mut t, Vec3::new(0.18, 0.15, 0.0), facing(0.3), 120, |_| {});
    t.reset();
    let home = t.arm.spoon_pose(&t.arm.q);
    assert!(t.sim.spoon.pos.distance_to(home.pos) < 1e-12);
    t.set_arm_enabled(false);
    t.step_frames(5); // arm off: plain spoon controller
    assert!(!t.arm.enabled);
}
