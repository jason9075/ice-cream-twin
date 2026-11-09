//! # icecream-gpu
//!
//! The ice-cream simulation resident on the GPU, the way Claybook runs: the four SDF volumes, the
//! particles and the volume ledger live in GPU buffers and every frame is one submission of WGSL
//! compute passes. The CPU (icecream-core) keeps the spoon controller and the SO-101.
pub mod layout;
pub mod sim;
pub mod twin;

pub use sim::{open_gpu, GpuSim, RenderPack, Stats};
pub use wgpu;
pub use twin::GpuTwin;
