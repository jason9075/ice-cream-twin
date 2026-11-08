//! End-of-arm tool modules: how the spoon is held by the SO-101. Port of `src/robot/tool.js`.
//!
//! A module says where the spoon sits in the `gripper_link` frame and what happens to the gripper
//! jaw. Today the stock gripper is clamped shut on the handle. The real arm will carry a different
//! module (later a 3D-printed spoon holder): add an entry to [`tool`] (and to `tool.js`) with its
//! own `spoon_in_gripper` and `jaw: None`.

use super::so101::{m4, M4};
use crate::math::{atan2, cos, sin};
use crate::sdf::SPOON;

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: &'static str,
    pub label: &'static str,
    /// Fixed gripper jaw angle (rad), or `None` when the module replaces the gripper.
    pub jaw: Option<f64>,
    pub hide_jaw: bool,
    /// Spoon pose (rim centre, spoon-local axes) in `gripper_link`.
    pub spoon_in_gripper: M4,
}

pub const TOOL_NAMES: [&str; 1] = ["so101-gripper-clamp"];
pub const DEFAULT_TOOL: &str = "so101-gripper-clamp";

/// Spoon pose in gripper_link from a grip description: the point `grip_x` along the handle's last
/// segment (spoon-local x, metres) is placed at `at` (gripper_link, metres), the handle runs along
/// gripper_link +Z (back toward the wrist) and the cup opening (spoon +Y) faces gripper_link +X.
pub fn handle_grip(grip_x: f64, at: [f64; 3]) -> M4 {
    let h = &SPOON.handle;
    let a = h[h.len() - 2];
    let b = h[h.len() - 1];
    let t = (grip_x - a[0]) / (b[0] - a[0]);
    let g = [grip_x, a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
    let al = atan2(b[1] - a[1], b[0] - a[0]);
    let (c, s) = (cos(al), sin(al));
    // columns = images of spoon x, y, z in gripper_link
    let r = [[-s, c, 0.0], [0.0, 0.0, 1.0], [c, s, 0.0]];
    let mut m = m4::identity();
    for i in 0..3 {
        for k in 0..3 {
            m[i * 4 + k] = r[i][k];
        }
        m[i * 4 + 3] = at[i] - (r[i][0] * g[0] + r[i][1] * g[1] + r[i][2] * g[2]);
    }
    m
}

pub fn tool(name: &str) -> Option<Tool> {
    match name {
        // Stock SO-101 gripper clamped on the handle: the handle lies against the fixed jaw's fingertip
        // pad (inner face at x = -7.9 mm), the jaw closed until it touches (tools/so101_grip.mjs).
        "so101-gripper-clamp" => Some(Tool {
            name: "so101-gripper-clamp",
            label: "SO-101 gripper, jaw clamped on the handle",
            jaw: Some(-0.119),
            hide_jaw: false,
            spoon_in_gripper: handle_grip(0.1, [-0.0079 + SPOON.handle_r, 0.0, -0.09]),
        }),
        _ => None,
    }
}
