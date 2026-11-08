//! SO-101 kinematics from the official URDF (`urdf.rs`), first written in JS (`src/robot/so101.js`, removed).
//! Matrices are 4x4 row-major `[f64; 16]`: `m[r * 4 + c]`, in the robot's own Z-up frame.

use super::urdf::SO101_JOINTS;
use crate::math::{cos, sin};

/// One URDF joint: origin (xyz, fixed-axis rpy), axis and limits.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub name: &'static str,
    pub child: &'static str,
    pub xyz: [f64; 3],
    pub rpy: [f64; 3],
    pub axis: [f64; 3],
    pub lower: f64,
    pub upper: f64,
}

pub type M4 = [f64; 16];

/// The five arm joints, base to tip. The gripper jaw is handled separately (the tool module fixes it).
pub const ARM_JOINTS: [&str; 5] = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"];
pub const JAW_JOINT: &str = "gripper";

/// Links in the order [`forward`] fills [`Fk::t`].
pub const LINKS: [&str; 8] = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
    "gripper_frame_link",
];
pub const GRIPPER_LINK: usize = 5;

pub fn joint(name: &str) -> &'static Joint {
    SO101_JOINTS.iter().find(|j| j.name == name).unwrap_or_else(|| panic!("no joint {name}"))
}

/// Joint limits (rad) of the five arm joints.
pub fn limits() -> [[f64; 2]; 5] {
    ARM_JOINTS.map(|n| {
        let j = joint(n);
        [j.lower, j.upper]
    })
}

pub fn jaw_limits() -> [f64; 2] {
    let j = joint(JAW_JOINT);
    [j.lower, j.upper]
}

// ---------------------------------------------------------------- 4x4 helpers
pub mod m4 {
    use super::*;

    pub fn identity() -> M4 {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        m
    }

    pub fn mul(a: &M4, b: &M4) -> M4 {
        let mut out = [0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                out[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
            }
        }
        out
    }

    /// URDF origin: translation xyz, then fixed-axis roll-pitch-yaw (R = Rz(y) Ry(p) Rx(r)).
    pub fn from_xyz_rpy(t: [f64; 3], rpy: [f64; 3]) -> M4 {
        let (cr, sr, cp, sp, cy, sy) = (cos(rpy[0]), sin(rpy[0]), cos(rpy[1]), sin(rpy[1]), cos(rpy[2]), sin(rpy[2]));
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            t[0],
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            t[1],
            -sp,
            cp * sr,
            cp * cr,
            t[2],
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }

    /// Rotation by angle about a unit axis.
    pub fn axis_angle(a: [f64; 3], ang: f64) -> M4 {
        let (x, y, z) = (a[0], a[1], a[2]);
        let (c, s) = (cos(ang), sin(ang));
        let t = 1.0 - c;
        [
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            0.0,
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            0.0,
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }

    pub fn invert_rigid(m: &M4) -> M4 {
        let mut out = [0.0; 16];
        for r in 0..3 {
            for c in 0..3 {
                out[r * 4 + c] = m[c * 4 + r];
            }
        }
        for r in 0..3 {
            out[r * 4 + 3] = -(out[r * 4] * m[3] + out[r * 4 + 1] * m[7] + out[r * 4 + 2] * m[11]);
        }
        out[15] = 1.0;
        out
    }
}

// ---------------------------------------------------------------- chain
/// Forward kinematics result: every link's pose in the robot base frame ([`LINKS`] order), plus
/// each arm joint's world axis and pivot (before its own rotation) for the Jacobian.
#[derive(Clone, Debug)]
pub struct Fk {
    pub t: [M4; 8],
    pub axes: [[f64; 3]; 5],
    pub origins: [[f64; 3]; 5],
}

impl Fk {
    pub fn link(&self, name: &str) -> &M4 {
        &self.t[LINKS.iter().position(|l| *l == name).unwrap_or_else(|| panic!("no link {name}"))]
    }
}

/// Joint origin matrices, computed once (like `ORIGIN` in so101.js).
fn origin(j: &Joint) -> &'static M4 {
    static ORIGINS: std::sync::OnceLock<Vec<M4>> = std::sync::OnceLock::new();
    let all = ORIGINS.get_or_init(|| SO101_JOINTS.iter().map(|j| m4::from_xyz_rpy(j.xyz, j.rpy)).collect());
    &all[SO101_JOINTS.iter().position(|x| x.name == j.name).unwrap()]
}

/// Forward kinematics. `q` = the five arm angles (rad), `jaw` = gripper angle.
pub fn forward(q: &[f64; 5], jaw: f64) -> Fk {
    let mut t = [m4::identity(); 8];
    let mut axes = [[0.0; 3]; 5];
    let mut origins = [[0.0; 3]; 5];
    let mut cur = t[0];
    for k in 0..5 {
        let j = joint(ARM_JOINTS[k]);
        let at = m4::mul(&cur, origin(j));
        let a = j.axis;
        axes[k] = [
            at[0] * a[0] + at[1] * a[1] + at[2] * a[2],
            at[4] * a[0] + at[5] * a[1] + at[6] * a[2],
            at[8] * a[0] + at[9] * a[1] + at[10] * a[2],
        ];
        origins[k] = [at[3], at[7], at[11]];
        cur = m4::mul(&at, &m4::axis_angle(a, q[k]));
        debug_assert_eq!(j.child, LINKS[k + 1]);
        t[k + 1] = cur;
    }
    let g = joint(JAW_JOINT);
    t[6] = m4::mul(&m4::mul(&t[GRIPPER_LINK], origin(g)), &m4::axis_angle(g.axis, jaw));
    t[7] = m4::mul(&t[GRIPPER_LINK], origin(joint("gripper_frame_joint")));
    Fk { t, axes, origins }
}
