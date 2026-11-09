//! What the protocol and the server drive: the simulation on the GPU (icecream-gpu, the default) or on
//! the CPU (icecream-core). Both carry the same SO-101 and answer the same commands.

use icecream_core::math::{Quat, Vec3};
use icecream_core::robot::Arm;
use icecream_core::sim::TABLE;
use icecream_core::{Params, Sim, State, Twin};
use icecream_gpu::{wgpu, GpuSim, GpuTwin};
use std::sync::Arc;

/// One SDF volume as the viewer's textures hold it (r8 distances, rgba8 colours).
pub struct GridView<'a> {
    pub dims: [usize; 3],
    pub dist: &'a [u8],
    pub color: &'a [u8],
    pub dist_changed: bool,
    pub color_changed: bool,
}

/// The CPU simulation's ice for the renderer.
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
    /// The CPU simulation's ice as the renderer's textures hold it (and which grids changed since the
    /// last call). The GPU simulation leaves its ice on the GPU instead (`gpu`).
    fn view(&mut self) -> View<'_>;
    /// The GPU simulation, when that is what runs.
    fn gpu(&mut self) -> Option<&mut GpuSim> {
        None
    }
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
        // renderer order: the bowls, the scoop, the table
        let mut changed = [(false, false); 5];
        for (k, c) in changed.iter_mut().enumerate() {
            let g = match k {
                0..=2 => &mut s.grids[k],
                3 => &mut s.content,
                _ => &mut s.grids[TABLE],
            };
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
        let grids = (0..5)
            .map(|k| {
                let g = match k {
                    0..=2 => &s.grids[k],
                    3 => &s.content,
                    _ => &s.grids[TABLE],
                };
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
    fn gpu(&mut self) -> Option<&mut GpuSim> {
        Some(&mut self.sim)
    }
    fn view(&mut self) -> View<'_> {
        let dist_changed = std::mem::take(&mut self.sim.dist_dirty);
        let color_changed = std::mem::take(&mut self.sim.color_dirty);
        let s = &self.sim;
        let grids = (0..s.l.grids.len())
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
    open_on(cpu, None)
}

/// As `open`, on a device someone else made (the app shares it with its renderer): device, queue, name.
pub fn open_on(cpu: bool, device: Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>, String)>) -> Box<dyn Engine> {
    if let (false, Some((d, q, name))) = (cpu, device.clone()) {
        eprintln!("simulation: GPU ({name})");
        return Box::new(GpuTwin::with_device(Params::default(), d, q, name));
    }
    if !cpu && device.is_none() {
        match GpuTwin::new(Params::default()) {
            Ok(t) => {
                eprintln!("simulation: GPU ({})", t.sim.adapter);
                return Box::new(t);
            }
            Err(e) => eprintln!("simulation: no GPU ({e}), running on the CPU"),
        }
    } else if cpu {
        eprintln!("simulation: CPU");
    }
    Box::new(Twin::default())
}
