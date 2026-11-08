//! # icecream-core
//!
//! Rust port of the ice-cream scooping digital twin (`src/sim/*.js`):
//!
//! - [`spoon`]: deterministic, speed-limited spoon controller that never penetrates the bowls
//!   or the table (sub-stepping + position-based rigid contact projection + rejection).
//! - [`sim`]: Claybook-style ice cream — solid SDF volumes, loose PBF particles that freeze back
//!   into the SDF, scooping into the spoon's own SDF, and a volume ledger that conserves mass.
//! - [`frames`]: Y-up scene frame ⇄ ROS REP-103.
//!
//! The numerics follow the JS version operation for operation (f32 storage where JS uses
//! `Float32Array`, f64 arithmetic, V8's `Math.hypot`, fdlibm-family trig), so a scripted run
//! produces the same trajectory in both (see `tests/parity.rs`).

// The code mirrors the JS operation for operation (bit-for-bit parity): `max(a, min(b, v))`
// is kept instead of `clamp` (different NaN / argument-order behaviour), and so on.
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
pub mod sdf;
pub mod sim;
pub mod spoon;
mod v8math;

pub use math::{Quat, Vec3};
pub use params::Params;
pub use sim::{Sim, State};
pub use spoon::{SpoonController, PEN_TOL};
