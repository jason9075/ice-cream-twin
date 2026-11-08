//! The simulation with the SO-101 holding the spoon, stepped the way the web page steps it
//! (`src/main.js`): `arm.constrain` → `sim.step` → `arm.follow`.

use crate::params::Params;
use crate::robot::Arm;
use crate::sim::Sim;

pub struct Twin {
    pub sim: Sim,
    pub arm: Arm,
}

impl Default for Twin {
    fn default() -> Self {
        Twin::new(Params::default())
    }
}

impl Twin {
    pub fn new(params: Params) -> Self {
        let mut sim = Sim::new(params);
        let mut arm = Arm::default();
        arm.reset(Some(&mut sim.spoon));
        Twin { sim, arm }
    }

    /// One frame. With the arm disabled this is just `sim.step`.
    pub fn step(&mut self, dt: f64) {
        self.arm.constrain(&mut self.sim.spoon, dt);
        self.sim.step(dt);
        self.arm.follow(&self.sim.spoon);
    }

    /// Run `n` frames of 1/60 s.
    pub fn step_frames(&mut self, n: usize) {
        for _ in 0..n {
            self.step(1.0 / 60.0);
        }
    }

    /// Reset the ice cream, and put the arm and the spoon back home.
    pub fn reset(&mut self) {
        self.sim.reset();
        if self.arm.enabled {
            self.arm.reset(Some(&mut self.sim.spoon));
        }
    }

    /// Hand the spoon to the arm (it snaps to the arm's home) or take it away.
    pub fn set_arm_enabled(&mut self, on: bool) {
        if on && !self.arm.enabled {
            self.arm.enabled = true;
            self.arm.reset(Some(&mut self.sim.spoon));
        }
        self.arm.enabled = on;
    }
}
