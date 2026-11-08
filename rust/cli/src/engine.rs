//! What the protocol and the server drive: the simulation on the GPU (icecream-gpu, the default) or on
//! the CPU (icecream-core). Both carry the same SO-101 and answer the same commands.

use icecream_core::math::{Quat, Vec3};
use icecream_core::robot::Arm;
use icecream_core::{Params, Sim, State, Twin};
use icecream_gpu::GpuTwin;

/// One SDF volume as the viewer's textures hold it (r8 distances, rgba8 colours).
pub struct GridView<'a> {
    pub dims: [usize; 3],
    pub dist: &'a [u8],
    pub color: &'a [u8],
    pub dist_changed: bool,
    pub color_changed: bool,
}

/// What the server streams after a frame.
pub struct View<'a> {
    pub grids: Vec<GridView<'a>>,
    /// n x (f32 x, y, z, u32 rgba)
    pub particles: Vec<u8>,
    pub spilled_version: u64,
    pub spilled: Vec<u8>,
}

pub trait Engine: Send {
    /// "gpu" or "cpu"
    fn kind(&self) -> &'static str;
    /// The CPU simulation behind it (scene layout, spoon controller, parameters).
    fn sim(&self) -> &Sim;
    fn arm(&self) -> &Arm;
    fn arm_mut(&mut self) -> &mut Arm;
    fn set_arm_enabled(&mut self, on: bool);
    fn set_target(&mut self, pos: Option<Vec3>, quat: Option<Quat>);
    fn step(&mut self, dt: f64);
    fn step_frames(&mut self, n: usize) {
        for _ in 0..n {
            self.step(1.0 / 60.0);
        }
    }
    fn eat(&mut self) -> f64;
    fn release(&mut self);
    fn reset(&mut self);
    fn refill(&mut self);
    fn params(&self) -> &Params {
        &self.sim().params
    }
    fn set_params(&mut self, p: Params);
    fn state(&self) -> State;
    /// Start keeping what `view` returns up to date (the GPU reads it back only when asked to).
    fn set_streaming(&mut self, _on: bool) {}
    fn view(&mut self) -> View<'_>;
}

fn particle_bytes(pos: &[f32], rgba: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 16);
    for i in 0..n {
        for k in 0..3 {
            out.extend_from_slice(&pos[i * 3 + k].to_le_bytes());
        }
        out.extend_from_slice(&rgba[i * 4..i * 4 + 4]);
    }
    out
}

impl Engine for Twin {
    fn kind(&self) -> &'static str {
        "cpu"
    }
    fn sim(&self) -> &Sim {
        &self.sim
    }
    fn arm(&self) -> &Arm {
        &self.arm
    }
    fn arm_mut(&mut self) -> &mut Arm {
        &mut self.arm
    }
    fn set_arm_enabled(&mut self, on: bool) {
        Twin::set_arm_enabled(self, on);
    }
    fn set_target(&mut self, pos: Option<Vec3>, quat: Option<Quat>) {
        self.sim.set_target(pos, quat);
    }
    fn step(&mut self, dt: f64) {
        Twin::step(self, dt);
    }
    fn eat(&mut self) -> f64 {
        self.sim.sync_spoon();
        self.sim.eat()
    }
    fn release(&mut self) {
        self.sim.release();
    }
    fn reset(&mut self) {
        Twin::reset(self);
    }
    fn refill(&mut self) {
        self.sim.refill();
    }
    fn set_params(&mut self, p: Params) {
        self.sim.params = p;
    }
    fn state(&self) -> State {
        self.sim.state()
    }
    fn view(&mut self) -> View<'_> {
        let s = &mut self.sim;
        let mut changed = [(false, false); 4];
        for (k, c) in changed.iter_mut().enumerate() {
            let g = if k < 3 { &mut s.grids[k] } else { &mut s.content };
            if g.dirty {
                g.pack();
                g.dirty = false;
                c.0 = true;
            }
            if g.color_dirty {
                g.color_dirty = false;
                c.1 = true;
            }
        }
        let particles = particle_bytes(&s.fluid.pos, &s.fluid.rgba, s.fluid.n);
        let spilled = particle_bytes(&s.spilled, &s.spilled_rgba, s.n_spilled);
        let s = &self.sim;
        let grids = (0..4)
            .map(|k| {
                let g = if k < 3 { &s.grids[k] } else { &s.content };
                GridView { dims: [g.nx, g.ny, g.nz], dist: &g.bytes, color: &g.color, dist_changed: changed[k].0, color_changed: changed[k].1 }
            })
            .collect();
        View { grids, particles, spilled_version: s.spilled_version, spilled }
    }
}

impl Engine for GpuTwin {
    fn kind(&self) -> &'static str {
        "gpu"
    }
    fn sim(&self) -> &Sim {
        &self.sim.cpu
    }
    fn arm(&self) -> &Arm {
        &self.arm
    }
    fn arm_mut(&mut self) -> &mut Arm {
        &mut self.arm
    }
    fn set_arm_enabled(&mut self, on: bool) {
        GpuTwin::set_arm_enabled(self, on);
    }
    fn set_target(&mut self, pos: Option<Vec3>, quat: Option<Quat>) {
        self.sim.cpu.spoon.set_target(pos, quat);
    }
    fn step(&mut self, dt: f64) {
        GpuTwin::step(self, dt);
    }
    fn eat(&mut self) -> f64 {
        self.sim.eat()
    }
    fn release(&mut self) {
        self.sim.release();
    }
    fn reset(&mut self) {
        GpuTwin::reset(self);
    }
    fn refill(&mut self) {
        GpuTwin::refill(self);
    }
    fn set_params(&mut self, p: Params) {
        self.sim.cpu.params = p;
    }
    fn state(&self) -> State {
        self.sim.state()
    }
    fn set_streaming(&mut self, on: bool) {
        self.sim.stream = on;
        self.sim.refresh(); // so the very first view already has the grids
    }
    fn view(&mut self) -> View<'_> {
        let dist_changed = std::mem::take(&mut self.sim.dist_dirty);
        let color_changed = std::mem::take(&mut self.sim.color_dirty);
        let s = &self.sim;
        let grids = (0..4)
            .map(|k| {
                let g = &s.l.grids[k];
                GridView {
                    dims: [g.n[0] as usize, g.n[1] as usize, g.n[2] as usize],
                    dist: &s.dist[k],
                    color: &s.color[k],
                    dist_changed: dist_changed[k],
                    color_changed: color_changed[k],
                }
            })
            .collect();
        View { grids, particles: s.particles.clone(), spilled_version: s.spilled_version, spilled: s.spilled.clone() }
    }
}

/// The GPU simulation if there is a GPU (unless `cpu`), else the CPU one. Says which on stderr.
pub fn open(cpu: bool) -> Box<dyn Engine> {
    if !cpu {
        match GpuTwin::new(Params::default()) {
            Ok(t) => {
                eprintln!("simulation: GPU ({})", t.sim.adapter);
                return Box::new(t);
            }
            Err(e) => eprintln!("simulation: no GPU ({e}), running on the CPU"),
        }
    } else {
        eprintln!("simulation: CPU");
    }
    Box::new(Twin::default())
}
