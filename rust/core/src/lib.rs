//! # icecream-core
//!
//! The ice-cream scooping digital twin. It started as JavaScript in the browser; this is now the only
//! implementation, and `icecream web` (rust/cli) streams it to the page:
//!
//! - [`spoon`]: deterministic, speed-limited spoon controller that never penetrates the bowls
//!   or the table (sub-stepping + position-based rigid contact projection + rejection).
//! - [`sim`]: Claybook-style ice cream — solid SDF volumes, loose PBF particles that freeze back
//!   into the SDF, scooping into the spoon's own SDF, and a volume ledger that conserves mass.
//! - [`frames`]: Y-up scene frame ⇄ ROS REP-103.
//! - [`robot`]: the SO-101 arm holding the spoon (URDF kinematics, IK, joint-space leading);
//!   [`Twin`] steps the two together.
//!
//! The numerics still follow the JS version operation for operation (f32 storage where it used
//! `Float32Array`, f64 arithmetic, V8's `Math.hypot`, fdlibm-family trig), which is what keeps them on
//! the frozen fixtures in `tests/parity.rs`.

// Written to match the JS bit for bit: `max(a, min(b, v))` is kept instead of `clamp` (different NaN /
// argument-order behaviour), and so on.
#![allow(
    clippy::manual_clamp,
    clippy::should_implement_trait,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::explicit_counter_loop,
    clippy::disallowed_names
)]

pub mod fluid;
pub mod frames;
pub mod grid;
pub mod math;
pub mod params;
pub mod robot;
pub mod sdf;
pub mod sim;
pub mod spoon;
pub mod twin;
mod v8math;

pub use math::{Quat, Vec3};
pub use params::Params;
pub use sim::{Sim, State};
pub use spoon::{SpoonController, PEN_TOL};
pub use twin::Twin;
