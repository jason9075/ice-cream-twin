//! The ice-cream twin as a service: the JSON protocol and the WebSocket server behind `icecream web`, over
//! the simulation on the GPU (icecream-gpu) or the CPU (icecream-core).
pub mod engine;
pub mod protocol;
pub mod web;
