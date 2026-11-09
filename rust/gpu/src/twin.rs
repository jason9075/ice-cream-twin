//! The GPU simulation with the SO-101 holding the spoon, stepped like `icecream_core::Twin`:
//! `arm.constrain` -> `sim.step` -> `arm.follow`.
use crate::sim::GpuSim;
use icecream_core::params::Params;
use icecream_core::robot::Arm;

pub struct GpuTwin {
    pub sim: GpuSim,
    pub arm: Arm,
}

impl GpuTwin {
    pub fn new(params: Params) -> Result<Self, String> {
        let sim = GpuSim::new(params)?;
        Ok(Self::from_sim(sim))
    }

    /// On a device the caller made (shared with a renderer).
    pub fn with_device(params: Params, device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>, adapter: String) -> Self {
        Self::from_sim(GpuSim::with_device(device, queue, adapter, params))
    }

    pub fn from_sim(mut sim: GpuSim) -> Self {
        let mut arm = Arm::default();
        arm.reset(Some(&mut sim.cpu.spoon));
        GpuTwin { sim, arm }
    }

    pub fn step(&mut self, dt: f64) {
        self.arm.constrain(&mut self.sim.cpu.spoon, dt);
        self.sim.step(dt);
        self.arm.follow(&self.sim.cpu.spoon);
    }

    pub fn step_frames(&mut self, n: usize) {
        for _ in 0..n {
            self.step(1.0 / 60.0);
        }
    }

    /// Reset the ice cream, and put the arm and the spoon back home.
    pub fn reset(&mut self) {
        self.sim.reset();
        if self.arm.enabled {
            self.arm.reset(Some(&mut self.sim.cpu.spoon));
        }
        self.sim.refresh();
    }

    pub fn refill(&mut self) {
        self.sim.refill();
        self.sim.refresh();
    }

    pub fn set_arm_enabled(&mut self, on: bool) {
        if on && !self.arm.enabled {
            self.arm.enabled = true;
            self.arm.reset(Some(&mut self.sim.cpu.spoon));
        }
        self.arm.enabled = on;
    }
}
