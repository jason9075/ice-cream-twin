//! The ice-cream twin: the JSON protocol, the control server, the window app and the off-screen
//! screenshots, over the simulation on the GPU (icecream-gpu) or the CPU (icecream-core), drawn by
//! icecream-view.
pub mod app;
pub mod control;
pub mod engine;
pub mod protocol;
pub mod scene;
pub mod shot;
pub mod ui;

pub mod recording;

mod preferences;
