//! The ice-cream twin's renderer (wgpu). It knows nothing about windows: the app hands it a texture to
//! draw into, the screenshot tool renders off screen.

pub mod arm;
pub mod camera;
pub mod fluid;
pub mod geometry;
pub mod renderer;
pub mod shaders;

pub use camera::OrbitCamera;
pub use renderer::{FrameState, GridGeo, Light, ParticleMode, Renderer, SceneDesc, Shading, ViewOptions, GRIDS};
