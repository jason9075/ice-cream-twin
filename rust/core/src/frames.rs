//! Coordinate frames: the scene is Y-up (three.js style); ROS REP-103 is x forward, y left,
//! z up. Converting is a +90° rotation about X. Mirrors `frames` in `src/bridge.js`.

use crate::math::{Quat, Vec3};

fn q_yup_to_ros() -> Quat {
    Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f64::consts::PI / 2.0)
}

pub fn pos_to_ros(p: [f64; 3]) -> [f64; 3] {
    [p[0], -p[2], p[1]]
}
pub fn pos_from_ros(p: [f64; 3]) -> [f64; 3] {
    [p[0], p[2], -p[1]]
}
pub fn quat_to_ros(q: [f64; 4]) -> [f64; 4] {
    let a = q_yup_to_ros();
    a.mul(Quat::from_array(q)).mul(a.invert()).to_array()
}
pub fn quat_from_ros(q: [f64; 4]) -> [f64; 4] {
    let a = q_yup_to_ros();
    a.invert().mul(Quat::from_array(q)).mul(a).to_array()
}
