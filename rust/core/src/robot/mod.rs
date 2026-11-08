//! The SO-101 arm holding the spoon (port of `src/robot/`): URDF kinematics, tool modules,
//! inverse kinematics and joint-space leading of the spoon. [`crate::Twin`] wires it to the sim.

pub mod arm;
pub mod so101;
pub mod tool;
mod urdf;

pub use arm::{Arm, ArmState, Base, PoseErr, SpoonPose, DEFAULT_BASE};
pub use tool::{Tool, DEFAULT_TOOL};
