//! The frame: which compute passes run, in what order, with what parameters (port of the page's
//! gpu-sim.js, git history up to eba6656). The CPU part is small and sequential: the spoon controller
//! (icecream-core), deciding which passes to run, and reading back one block of statistics per frame.
//!
//!   carve + copy into the scoop (per spoon sub-step) -> carry -> tip-out -> PBF sub-steps -> freeze
//!   (stamp into the SDFs) -> ledger (debts -> stamps) -> smoothing / redistancing -> statistics
//!
//! With `stream` on, a second small submission packs what changed for the viewer (grids that were
//! edited, particles, crumbs) and reads it back.

use crate::layout::{cnt, entry_points, pack_row_words, shader, st, stu, Layout, CAP, MAX_SPILLED, NG, PASS_BYTES};
use icecream_core::fluid::MAX_LUMPS;
use icecream_core::math::{Quat, Vec3};
use icecream_core::params::Params;
use icecream_core::sdf::SPOON;
use icecream_core::spoon::Substep;
use icecream_core::Sim;
use std::collections::HashMap;
use std::sync::Arc;
use std::f64::consts::PI;
use std::time::Instant;

const DISPLACED_PARTICLES_PER_FRAME: u32 = 40;
const MAX_SLOTS: usize = 2048;
const RD_ITERS: u32 = 12;
const WG: u32 = 64;
const STATS_BYTES: u64 = 1024 + 512 + 256;

/// Open the GPU (high-performance adapter). Err when there is none.
pub fn open_gpu() -> Result<(wgpu::Device, wgpu::Queue, String), String> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .ok_or("no GPU adapter")?;
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("icecream"), ..Default::default() }, None)
            .await
            .map_err(|e| format!("request_device: {e}"))?;
        Ok((device, queue, name))
    })
}

/// What `pack_for_render` packed: which grids, how many particles and crumbs.
#[derive(Clone, Copy, Debug)]
pub struct RenderPack {
    pub grids: [bool; NG],
    pub n: u32,
    pub crumbs: u32,
}

/// Per-dispatch parameters (the WGSL `Pass` struct, 256 bytes).
#[derive(Clone, Copy, Default)]
pub struct Pass {
    pub i0: [i32; 4],
    pub i1: [i32; 4],
    pub i2: [i32; 4],
    pub i3: [i32; 4],
    pub f0: [f32; 4],
    pub f1: [f32; 4],
    pub f2: [f32; 4],
    pub f3: [f32; 4],
    pub r0: [f32; 4],
    pub r1: [f32; 4],
    pub r2: [f32; 4],
    pub t: [f32; 4],
    pub q0: [f32; 4],
    pub q1: [f32; 4],
    pub x0: [f32; 4],
    pub x1: [f32; 4],
}

impl Pass {
    pub fn i(i0: [i32; 4]) -> Pass {
        Pass { i0, ..Default::default() }
    }
    fn bytes(&self) -> [u8; 256] {
        let mut b = [0u8; 256];
        let mut o = 0;
        for v in [self.i0, self.i1, self.i2, self.i3] {
            for x in v {
                b[o..o + 4].copy_from_slice(&x.to_le_bytes());
                o += 4;
            }
        }
        for v in [self.f0, self.f1, self.f2, self.f3, self.r0, self.r1, self.r2, self.t, self.q0, self.q1, self.x0, self.x1] {
            for x in v {
                b[o..o + 4].copy_from_slice(&x.to_le_bytes());
                o += 4;
            }
        }
        b
    }
    fn pose(p: &Pose) -> Pass {
        Pass { r0: p.r0, r1: p.r1, r2: p.r2, t: p.t, ..Default::default() }
    }
}

/// Rows (local -> world) of a quaternion's rotation, as Sim's pose builds R, and the position.
#[derive(Clone, Copy)]
struct Pose {
    r0: [f32; 4],
    r1: [f32; 4],
    r2: [f32; 4],
    t: [f32; 4],
    r: [[f64; 3]; 3],
    p: [f64; 3],
}

fn pose_rows(q: Quat, p: Vec3) -> Pose {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let r = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ];
    let row = |k: usize| [r[k][0] as f32, r[k][1] as f32, r[k][2] as f32, 0.0];
    Pose { r0: row(0), r1: row(1), r2: row(2), t: [p.x as f32, p.y as f32, p.z as f32, 0.0], r, p: [p.x, p.y, p.z] }
}

fn local_box_to_world(pose: &Pose, min: [f64; 3], max: [f64; 3]) -> [f64; 6] {
    let mut out = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for m in 0..8 {
        let l = [if m & 1 != 0 { max[0] } else { min[0] }, if m & 2 != 0 { max[1] } else { min[1] }, if m & 4 != 0 { max[2] } else { min[2] }];
        for a in 0..3 {
            let v = pose.r[a][0] * l[0] + pose.r[a][1] * l[1] + pose.r[a][2] * l[2] + pose.p[a];
            out[a] = out[a].min(v);
            out[a + 3] = out[a + 3].max(v);
        }
    }
    out
}

#[derive(Clone, Copy)]
enum Buf {
    W,
    X,
    A,
    Pk,
    Pp,
}

enum Op {
    Run { pipe: usize, groups: u32, slot: u32 },
    Copy { src: Buf, src_off: u64, dst: usize, dst_off: u64, size: u64 },
}

/// One submission's command list: dispatches with their parameters, and copies to staging buffers.
pub struct Frame {
    slots: Vec<u8>,
    ops: Vec<Op>,
}

impl Frame {
    fn new() -> Self {
        Frame { slots: Vec::with_capacity(64 * 256), ops: Vec::new() }
    }
}

/// Statistics read back every frame.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub content_vol: f64,
    /// per GPU grid: the bowls, the scoop, the table
    pub volume: [f64; NG],
    pub removed: f64,
    pub spilled_vol: f64,
    pub eaten_vol: f64,
    pub debt_vol: f64,
    pub loose_owed: f64,
    pub n: u32,
    pub spilled: u32,
    pub n_debts: u32,
    pub settle_frames: [u32; NG],
    /// NG distance, then NG colour marks
    pub dirty: [bool; 2 * NG],
    pub gpu_ms: f64,
}

pub struct GpuSim {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: String,
    pub l: Layout,
    names: HashMap<String, usize>,
    pipes: Vec<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    globals: wgpu::Buffer,
    passbuf: wgpu::Buffer,
    w: wgpu::Buffer,
    x: wgpu::Buffer,
    a: wgpu::Buffer,
    pk: wgpu::Buffer,
    pp: wgpu::Buffer,
    staging: Vec<wgpu::Buffer>,
    pack_off: Vec<u32>,
    pack_size: u32,
    /// the CPU simulation: spoon controller, parameters, the initial ice (never stepped)
    pub cpu: Sim,
    pub stats: Stats,
    pub speed_scale: f64,
    pub cut_force: f64,
    releasing: bool,
    release_queued: bool,
    redist_pending: bool,
    content_edited: bool,
    last_travel: f64,
    last_carve_pos: Vec3,
    last_carve_quat: Quat,
    pub time: f64,
    pub frame: u64,
    /// stream the viewer's data each frame (grids that changed, particles, crumbs)
    pub stream: bool,
    force_pack: bool,
    /// pack every grid for the renderer next time (`pack_for_render`)
    render_force: bool,
    /// per grid: r8 distances and rgba8 colours as the viewer's textures hold them, and whether they
    /// changed since the viewer last cleared the marks
    pub dist: Vec<Vec<u8>>,
    pub color: Vec<Vec<u8>>,
    pub dist_dirty: [bool; NG],
    pub color_dirty: [bool; NG],
    /// n x (f32 x, y, z, u32 rgba)
    pub particles: Vec<u8>,
    pub spilled: Vec<u8>,
    pub spilled_version: u64,
}

impl GpuSim {
    pub fn new(params: Params) -> Result<Self, String> {
        let (device, queue, adapter) = open_gpu()?;
        Ok(Self::with_device(Arc::new(device), Arc::new(queue), adapter, params))
    }

    /// On a device someone else made (the app shares it with the renderer).
    pub fn with_device(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, adapter: String, params: Params) -> Self {
        let cpu = Sim::new(params);
        let l = Layout::new(&cpu);
        let code = shader(&l);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("icecream"), source: wgpu::ShaderSource::Wgsl(code.as_str().into()) });
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buf = |label: &str, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.div_ceil(4) * 4, usage, mapped_at_creation: false });
        let globals = buf("globals", 256, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let passbuf = buf("passes", MAX_SLOTS as u64 * PASS_BYTES, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let w = buf("W", l.w.size as u64 * 4, storage);
        let x = buf("X", l.x.size as u64 * 4, storage);
        let a = buf("A", l.a.size as u64 * 4, storage);
        let (pack_off, pack_size) = l.pack_offsets();
        let pk = buf("PK", pack_size as u64 * 4, storage);
        let pp = buf("PP", (CAP + MAX_SPILLED) as u64 * 16, storage);
        let map = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        // staging: 0 statistics, 1 packed grids, 2 particles + crumbs, 3 debugging reads
        let staging = vec![
            buf("stats", STATS_BYTES, map),
            buf("stream grids", pack_size as u64 * 4, map),
            buf("stream particles", (CAP + MAX_SPILLED) as u64 * 16, map),
            buf("debug", l.w.size.max(l.x.size).max(l.a.size) as u64 * 4, map),
        ];
        let entry = |binding, ty, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: if dynamic { wgpu::BufferSize::new(PASS_BYTES) } else { None } },
            count: None,
        };
        let storage_ty = wgpu::BufferBindingType::Storage { read_only: false };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("icecream"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform, false),
                entry(1, wgpu::BufferBindingType::Uniform, true),
                entry(2, storage_ty, false),
                entry(3, storage_ty, false),
                entry(4, storage_ty, false),
                entry(5, storage_ty, false),
                entry(6, storage_ty, false),
            ],
        });
        fn whole(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
            b.as_entire_binding()
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("icecream"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: whole(&globals) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &passbuf, offset: 0, size: wgpu::BufferSize::new(PASS_BYTES) }) },
                wgpu::BindGroupEntry { binding: 2, resource: whole(&w) },
                wgpu::BindGroupEntry { binding: 3, resource: whole(&x) },
                wgpu::BindGroupEntry { binding: 4, resource: whole(&a) },
                wgpu::BindGroupEntry { binding: 5, resource: whole(&pp) },
                wgpu::BindGroupEntry { binding: 6, resource: whole(&pk) },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("icecream"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let mut names = HashMap::new();
        let mut pipes = Vec::new();
        for name in entry_points(&code) {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&name),
                layout: Some(&pl),
                module: &module,
                entry_point: &name,
                compilation_options: Default::default(),
                cache: None,
            });
            names.insert(name, pipes.len());
            pipes.push(p);
        }
        let mut s = GpuSim {
            device,
            queue,
            adapter,
            names,
            pipes,
            bind,
            globals,
            passbuf,
            w,
            x,
            a,
            pk,
            pp,
            staging,
            pack_off,
            pack_size,
            stats: Stats::default(),
            speed_scale: 1.0,
            cut_force: 0.0,
            releasing: false,
            release_queued: false,
            redist_pending: false,
            content_edited: false,
            last_travel: 0.0,
            last_carve_pos: Vec3::ZERO,
            last_carve_quat: Quat::IDENTITY,
            time: 0.0,
            frame: 0,
            stream: false,
            force_pack: true,
            render_force: true,
            dist: l.grids.iter().map(|g| vec![0; g.count as usize]).collect(),
            color: l.grids.iter().map(|g| vec![0; g.count as usize * 4]).collect(),
            dist_dirty: [true; NG],
            color_dirty: [true; NG],
            particles: Vec::new(),
            spilled: Vec::new(),
            spilled_version: 0,
            l,
            cpu,
        };
        s.reset();
        s
    }

    // ------------------------------------------------------------------ state
    fn upload_grid(&mut self, k: usize) {
        let off = self.l.grids[k].off as u64;
        let g = self.cpu_grid(k);
        self.queue.write_buffer(&self.w, (self.l.w.get("D") as u64 + off) * 4, bytemuck::cast_slice(&g.data));
        self.queue.write_buffer(&self.x, (self.l.x.get("COL") as u64 + off) * 4, &g.color);
        self.queue.write_buffer(&self.x, (self.l.x.get("FLAG") as u64 + off) * 4, &vec![0u8; g.n * 4]);
    }

    /// The CPU simulation's grid behind GPU grid `k` (the bowls, the scoop).
    fn cpu_grid(&self, k: usize) -> &icecream_core::grid::SdfGrid {
        if k < 3 { &self.cpu.grids[k] } else { &self.cpu.content }
    }

    /// Back to the start: the CPU simulation's initial ice, an empty spoon, no particles.
    pub fn reset(&mut self) {
        self.cpu.reset();
        let floor = self.cpu.content_floor();
        for k in 0..NG {
            self.upload_grid(k);
        }
        self.queue.write_buffer(&self.w, self.l.w.get("FLOOR") as u64 * 4, bytemuck::cast_slice(&floor));
        self.queue.write_buffer(&self.w, self.l.w.get("ST") as u64 * 4, &[0u8; 1024]);
        let mut su = vec![0u32; 256];
        su[stu::RNG as usize] = 12345;
        self.queue.write_buffer(&self.x, self.l.x.get("STU") as u64 * 4, bytemuck::cast_slice(&su));
        let mut a = vec![0u32; self.l.a.size as usize];
        a[(self.l.a.get("CNT") + cnt::FIRST_IDX) as usize] = u32::MAX;
        self.queue.write_buffer(&self.a, 0, bytemuck::cast_slice(&a));
        self.time = 0.0;
        self.frame = 0;
        self.speed_scale = 1.0;
        self.cut_force = 0.0;
        self.releasing = false;
        self.release_queued = false;
        self.redist_pending = false;
        self.last_travel = 0.0;
        self.last_carve_pos = self.cpu.spoon.pos;
        self.last_carve_quat = self.cpu.spoon.quat;
        self.stats = Stats::default();
        for k in 0..3 {
            self.stats.volume[k] = self.cpu.grids[k].volume();
        }
        self.particles.clear();
        self.spilled.clear();
        self.spilled_version += 1;
        self.force_pack = true;
        self.render_force = true;
    }

    /// Refill the bowls (the CPU simulation's fill shape).
    pub fn refill(&mut self) {
        self.cpu.refill();
        for k in 0..3 {
            self.upload_grid(k);
        }
        self.force_pack = true;
        self.render_force = true;
    }

    /// Tip everything on the spoon off as particles (next frame).
    pub fn release(&mut self) {
        self.release_queued = true;
    }

    pub fn params(&self) -> &Params {
        &self.cpu.params
    }

    // ------------------------------------------------------------------ recording
    fn pipe(&self, name: &str) -> usize {
        *self.names.get(name).unwrap_or_else(|| panic!("no kernel {name}"))
    }
    fn run(&self, fr: &mut Frame, name: &str, groups: u32, p: Pass) {
        if groups == 0 {
            return;
        }
        let slot = (fr.slots.len() / 256) as u32;
        assert!((slot as usize) < MAX_SLOTS, "too many dispatches in one frame");
        fr.slots.extend_from_slice(&p.bytes());
        fr.ops.push(Op::Run { pipe: self.pipe(name), groups, slot });
    }
    /// A per-particle kernel: dispatched for the whole capacity (threads past the count return).
    fn particles(&self, fr: &mut Frame, name: &str, p: Pass) {
        self.run(fr, name, CAP.div_ceil(WG), p);
    }
    fn copy(&self, fr: &mut Frame, src: Buf, src_off: u64, dst: usize, dst_off: u64, size: u64) {
        fr.ops.push(Op::Copy { src, src_off, dst, dst_off, size });
    }

    fn submit(&self, fr: Frame) {
        if !fr.slots.is_empty() {
            self.queue.write_buffer(&self.passbuf, 0, &fr.slots);
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let mut i = 0;
        while i < fr.ops.len() {
            if let Op::Copy { src, src_off, dst, dst_off, size } = fr.ops[i] {
                let s = match src {
                    Buf::W => &self.w,
                    Buf::X => &self.x,
                    Buf::A => &self.a,
                    Buf::Pk => &self.pk,
                    Buf::Pp => &self.pp,
                };
                enc.copy_buffer_to_buffer(s, src_off, &self.staging[dst], dst_off, size);
                i += 1;
                continue;
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim"), timestamp_writes: None });
            while i < fr.ops.len() {
                let Op::Run { pipe, groups, slot } = fr.ops[i] else { break };
                pass.set_pipeline(&self.pipes[pipe]);
                pass.set_bind_group(0, &self.bind, &[slot * PASS_BYTES as u32]);
                pass.dispatch_workgroups(groups, 1, 1);
                i += 1;
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// Map staging buffer `k`, hand its first `size` bytes to `f`, unmap.
    fn read<R>(&self, k: usize, size: u64, f: impl FnOnce(&[u8]) -> R) -> R {
        let slice = self.staging[k].slice(0..size.div_ceil(4) * 4);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::Maintain::Wait);
        let r = f(&slice.get_mapped_range());
        self.staging[k].unmap();
        r
    }

    fn scan(&self, fr: &mut Frame, unsigned: bool, src: u32, dst: u32, n: u32) {
        let blocks = n.div_ceil(512);
        if blocks == 0 {
            return;
        }
        let p = Pass::i([src as i32, dst as i32, n as i32, i32::from(unsigned)]);
        self.run(fr, if unsigned { "scanBlockU" } else { "scanBlockF" }, blocks, p);
        self.run(fr, "scanSums", 1, p);
        self.run(fr, "scanAdd", blocks, p);
    }

    fn grid_occ(&self, fr: &mut Frame, k: usize, slot: u32, mode: i32) {
        let g = &self.l.grids[k];
        let wgs = g.count.div_ceil(1024);
        self.run(fr, "gridOcc", wgs, Pass::i([k as i32, 0, 0, 0]));
        self.run(fr, "finishSum", 1, Pass { i0: [0, wgs as i32, slot as i32, mode], f0: [g.c.powi(3) as f32, 0.0, 0.0, 0.0], ..Default::default() });
    }

    fn smooth(&self, fr: &mut Frame, k: usize, mode: i32, strength: f64) {
        let total = self.l.grids[k].count;
        let p = Pass { i0: [mode, k as i32, 0, 0], f0: [strength as f32, 0.0, 0.0, 0.0], ..Default::default() };
        self.run(fr, "smFlag", total.div_ceil(256), p);
        if mode == 0 {
            self.run(fr, "smBox", total.div_ceil(256), p);
        }
        self.scan(fr, true, self.l.x.get("SCANU"), self.l.x.get("PRE"), total);
        self.run(fr, "smList", total.div_ceil(256), p);
        self.run(fr, "smWrite", 1, p);
    }

    fn compact(&self, fr: &mut Frame) {
        let groups = CAP.div_ceil(WG);
        self.run(fr, "compactMark", groups, Pass::default());
        self.scan(fr, true, self.l.x.get("SCANU"), self.l.x.get("IDX"), CAP);
        self.run(fr, "compactScatter", groups, Pass::default());
        self.run(fr, "compactCount", 1, Pass::default());
        self.particles(fr, "compactCopy", Pass::default());
    }

    fn content_box(&self, pose: &Pose) -> [f64; 6] {
        let c = &self.cpu.content;
        local_box_to_world(pose, [c.ox, c.oy, c.oz], [c.ox + c.sx, c.oy + c.sy, c.oz + c.sz])
    }

    /// Sim::interact on the CPU side: decide, then record the GPU passes for this spoon pose.
    pub fn interact(&mut self, fr: &mut Frame, pos: Vec3, quat: Quat, force: bool) {
        let moved = self.last_carve_pos.distance_to(pos) + self.last_carve_quat.angle_to(quat) * 0.06;
        if (!force && moved < 0.0009) || moved < 1e-7 {
            return;
        }
        self.last_carve_pos = pos;
        self.last_carve_quat = quat;
        let pose = pose_rows(quat, pos);
        let tilted = Sim::tilt_deg_of(quat) > self.cpu.params.ice.release_angle;
        let hb = self.content_box(&pose);
        let h = &SPOON.handle;
        let ymin = h.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let ymax = h.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
        let hb_handle = local_box_to_world(&pose, [h[0][0] - 0.004, ymin - 0.004, -0.004], [h[h.len() - 1][0] + 0.004, ymax + 0.004, 0.004]);
        let gb: Vec<[f64; 6]> = self.cpu.grids.iter().map(|g| [g.ox, g.oy, g.oz, g.ox + g.sx, g.oy + g.sy, g.oz + g.sz]).collect();
        let hits = |b: &[f64; 6], k: usize| {
            let g = &gb[k];
            !(b[3] < g[0] || b[0] > g[3] || b[4] < g[1] || b[1] > g[4] || b[5] < g[2] || b[2] > g[5])
        };
        let touching: Vec<usize> = (0..3).filter(|&k| hits(&hb, k) || hits(&hb_handle, k)).collect();
        if touching.is_empty() {
            return;
        }
        let base = Pass::pose(&pose);
        let nc = self.l.nc;
        // A. copy ice into the scoop
        if !tilted {
            let wgs = nc.div_ceil(256);
            self.run(fr, "copyMark", wgs, base);
            self.run(fr, "finishSum", 1, Pass { i0: [0, wgs as i32, st::GAINED as i32, 0], f0: [self.cpu.content.cell.powi(3) as f32, 0.0, 0.0, 0.0], ..Default::default() });
            self.run(fr, "copyFill", nc.div_ceil(WG), Pass::default());
            self.run(fr, "copyApply", nc.div_ceil(WG), Pass::default());
            self.content_edited = true;
        } else {
            self.run(fr, "finishSum", 1, Pass::i([0, 0, st::GAINED as i32, 0]));
        }
        self.run(fr, "copyDone", 1, Pass::default());
        // B. carve the bowls, head box then handle box, grid by grid
        for &k in &touching {
            for (bx, handle_only) in [(hb, false), (hb_handle, true)] {
                if !hits(&bx, k) {
                    continue;
                }
                let g = &self.cpu.grids[k];
                let c = g.cell;
                let lo = |v: f64, o: f64| ((v - o) / c).floor().max(0.0) as i32;
                let hi = |v: f64, o: f64, n: usize| ((v - o) / c).floor().min(n as f64 - 1.0) as i32;
                let (i0, i1) = (lo(bx[0], g.ox), hi(bx[3], g.ox, g.nx));
                let (j0, j1) = (lo(bx[1], g.oy), hi(bx[4], g.oy, g.ny));
                let (k0, k1) = (lo(bx[2], g.oz), hi(bx[5], g.oz, g.nz));
                let s = [i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1];
                if s.iter().any(|&v| v <= 0) {
                    continue;
                }
                let rn = (s[0] * s[1] * s[2]) as u32;
                let p = Pass { i0: [k as i32, i0, j0, k0], i1: [s[0], s[1], s[2], i32::from(handle_only) | (i32::from(tilted) << 1)], ..base };
                self.run(fr, "carveMark", rn.div_ceil(256), p);
                self.scan(fr, false, self.l.w.get("DVD"), self.l.w.get("PD"), rn);
                self.scan(fr, false, self.l.w.get("DVC"), self.l.w.get("PC"), rn);
                self.run(fr, "carveSpawn", rn.div_ceil(WG), p);
                self.run(fr, "carveDone", 1, p);
            }
        }
        self.run(fr, "interactDone", 1, Pass::i([touching[0] as i32, 0, 0, 0]));
    }

    fn release_passes(&self, fr: &mut Frame) {
        let step = self.cpu.fluid.d;
        let c = &self.cpu.content;
        let n = |s: f64| if step / 2.0 < s { ((s - step / 2.0) / step).floor() as i32 + 1 } else { 0 };
        let dims = [n(c.sx), n(c.sy), n(c.sz)];
        let rn = (dims[0] * dims[1] * dims[2]) as u32;
        let p = Pass { i0: [dims[0], dims[1], dims[2], 0], f0: [step as f32, 0.0, 0.0, 0.0], ..Default::default() };
        self.run(fr, "releaseMark", rn.div_ceil(WG), p);
        self.scan(fr, true, self.l.x.get("SCANU"), self.l.x.get("LIST"), rn);
        self.run(fr, "releaseSpawn", rn.div_ceil(WG), p);
        self.run(fr, "releaseDone", 1, p);
        self.run(fr, "clearContent", self.l.nc.div_ceil(WG), Pass::i([1, 0, 0, 0]));
    }

    fn write_globals(&mut self, dt_frame: f64, sub: u32, pose: &Pose, tilt: f64) {
        let fp = self.cpu.params.fluid.clone();
        let ip = self.cpu.params.ice.clone();
        self.cpu.fluid.configure(fp.particle_radius, fp.kernel_scale);
        let f = &self.cpu.fluid;
        let dt = dt_frame / sub as f64;
        let r = f.r;
        let sp = &self.cpu.spoon;
        let mut g = [0f32; 64];
        let mut put = |o: usize, v: [f64; 4]| {
            for k in 0..4 {
                g[o + k] = v[k] as f32;
            }
        };
        put(0, [dt, fp.gravity, (1.0 - fp.damping * dt).max(0.0), 1.2]);
        put(4, [r, f.h, f.h2, f.rho0]);
        put(8, [f.eps, f.d, 0.5 * r, fp.cohesion]);
        put(12, [fp.friction, 1.5 * fp.friction * fp.gravity * dt, fp.viscosity, 1.0 / f.h]);
        put(16, [fp.stop_speed, fp.rest_frames as f64, fp.stamp_scale, fp.stamp_smooth]);
        put(20, [ip.smoothing, ip.accretion_dist, ip.capacity * 1e-6, dt_frame]);
        put(24, [8.0 * r * r * r, f64::from(u8::from(fp.stamp_on_contact)), (fp.max_particles as f64).min(CAP as f64), f64::from(u8::from(ip.displaced_to_particles))]);
        put(28, [f64::from(u8::from(tilt < ip.release_angle - 10.0)), f64::from(u8::from(ip.accretion)), ip.release_angle, tilt]);
        put(48, [sp.lin_vel.x, sp.lin_vel.y, sp.lin_vel.z, 0.0]);
        put(52, [sp.ang_vel.x, sp.ang_vel.y, sp.ang_vel.z, 0.0]);
        put(60, [fp.lump_stiffness, fp.lump_yield * r, f64::from(u8::from(ip.smooth_cuts)), 0.0]);
        g[32..36].copy_from_slice(&pose.r0);
        g[36..40].copy_from_slice(&pose.r1);
        g[40..44].copy_from_slice(&pose.r2);
        g[44..48].copy_from_slice(&pose.t);
        let mut bytes: Vec<u8> = bytemuck::cast_slice(&g).to_vec();
        for (k, v) in [self.frame as u32, DISPLACED_PARTICLES_PER_FRAME, 48, 256].iter().enumerate() {
            bytes[(56 + k) * 4..(57 + k) * 4].copy_from_slice(&v.to_le_bytes());
        }
        self.queue.write_buffer(&self.globals, 0, &bytes);
    }

    /// Set the per-frame uniforms for the spoon's current pose (debug passes that need them).
    pub fn debug_globals(&mut self, dt: f64) {
        let sp = &self.cpu.spoon;
        let pose = pose_rows(sp.quat, sp.pos);
        let tilt = Sim::tilt_deg_of(sp.quat);
        let sub = self.cpu.params.fluid.substeps.max(1);
        self.write_globals(dt, sub, &pose, tilt);
    }

    fn copy_stats(&self, fr: &mut Frame) {
        self.copy(fr, Buf::W, self.l.w.get("ST") as u64 * 4, 0, 0, 1024);
        self.copy(fr, Buf::X, self.l.x.get("STU") as u64 * 4, 0, 1024, 512);
        self.copy(fr, Buf::A, self.l.a.get("CNT") as u64 * 4, 0, 1536, 256);
    }

    fn read_stats(&mut self) {
        let s = self.read(0, STATS_BYTES, |b| {
            let f = |k: u32| f32::from_le_bytes(b[k as usize * 4..k as usize * 4 + 4].try_into().unwrap()) as f64;
            let u = |k: u32| u32::from_le_bytes(b[1024 + k as usize * 4..1028 + k as usize * 4].try_into().unwrap());
            let c = |k: u32| u32::from_le_bytes(b[1536 + k as usize * 4..1540 + k as usize * 4].try_into().unwrap());
            Stats {
                content_vol: f(st::CONTENT_VOL),
                volume: std::array::from_fn(|k| f(st::VOLUME + k as u32)),
                removed: f(st::REMOVED),
                spilled_vol: f(st::SPILLED_VOL),
                eaten_vol: f(st::EATEN_VOL),
                debt_vol: f(st::DEBT_VOL),
                loose_owed: f(st::LOOSE_OWED),
                n: c(cnt::N),
                spilled: c(cnt::SPILLED),
                n_debts: u(stu::N_DEBTS),
                settle_frames: std::array::from_fn(|k| u(stu::SETTLE_FRAMES + k as u32)),
                dirty: std::array::from_fn(|k| u(stu::DIRTY + k as u32) != 0),
                gpu_ms: 0.0,
            }
        });
        self.stats = s;
    }

    // ------------------------------------------------------------------ the frame
    /// One frame of `dt` (clamped to 1/30 s), carried out on the GPU; returns when it is done.
    pub fn step(&mut self, dt: f64) {
        let t0 = Instant::now();
        let dt = dt.min(1.0 / 30.0);
        let start_pos = self.cpu.spoon.pos;
        let start_quat = self.cpu.spoon.quat;
        let mut fr = Frame::new();
        self.content_edited = false;
        self.run(&mut fr, "frameStart", 1, Pass::default());

        // 1. spoon motion, carving the ice along the swept path
        self.cpu.spoon.p = self.cpu.params.spoon.clone();
        let mut subs: Vec<Substep> = Vec::new();
        let ss = self.speed_scale;
        self.cpu.spoon.update(dt, ss, Some(&mut subs));
        for s in &subs {
            self.interact(&mut fr, s.pos, s.quat, false);
        }
        let (sp_pos, sp_quat) = (self.cpu.spoon.pos, self.cpu.spoon.quat);
        self.interact(&mut fr, sp_pos, sp_quat, true);
        self.compact(&mut fr);

        // cutting force from the volume the last frame's carving removed (read back a frame late)
        let travel = sp_pos.distance_to(start_pos) + start_quat.angle_to(sp_quat) * 0.03;
        let area = (self.stats.removed / self.last_travel.max(0.0005)).min(PI * SPOON.a * SPOON.c);
        self.last_travel = travel;
        let p = &self.cpu.params;
        let force = p.ice.hardness * 1000.0 * area;
        self.cut_force += (force - self.cut_force) * 0.3;
        let max_cut = p.spoon.max_cut_force;
        let target = if self.cut_force > max_cut { max_cut / self.cut_force } else { 1.0 };
        self.speed_scale += (0.05f64.max(target) - self.speed_scale) * 0.3;

        // 2. carry loose ice in / on the spoon
        let pose = pose_rows(sp_quat, sp_pos);
        let tilt = Sim::tilt_deg_of(sp_quat);
        let release_angle = self.cpu.params.ice.release_angle;
        let tilted = tilt > release_angle;
        let sub = self.cpu.params.fluid.substeps.max(1);
        let iterations = self.cpu.params.fluid.iterations;
        self.write_globals(dt, sub, &pose, tilt);
        if !tilted && !(start_pos.distance_to(sp_pos) < 1e-9 && start_quat.angle_to(sp_quat) < 1e-9) {
            let qi = start_quat.invert();
            let qd = sp_quat.mul(qi);
            let v4 = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
            let q4 = |q: Quat| [q.x as f32, q.y as f32, q.z as f32, q.w as f32];
            self.particles(&mut fr, "carry", Pass { q0: q4(qi), q1: q4(qd), x0: v4(start_pos), x1: v4(sp_pos), ..Default::default() });
        }
        // 3. tilted past the release angle: the scoop slides off as particles
        let mut release = std::mem::take(&mut self.release_queued);
        if tilted && !self.releasing {
            release = true;
            self.releasing = true;
        } else if tilt < release_angle - 10.0 {
            self.releasing = false;
        }
        if release {
            self.release_passes(&mut fr);
        }

        // 4. particles
        self.particles(&mut fr, "markFrame", Pass::default());
        let table_groups = (1u32 << 16).div_ceil(WG);
        for _ in 0..sub {
            self.particles(&mut fr, "predict", Pass::default());
            self.run(&mut fr, "clearCells", table_groups + 1, Pass::default());
            self.particles(&mut fr, "countCells", Pass::default());
            self.run(&mut fr, "scanCells", 1, Pass::default());
            self.run(&mut fr, "clearCells", table_groups + 1, Pass::default());
            self.particles(&mut fr, "scatter", Pass::default());
            self.run(&mut fr, "sortCells", table_groups, Pass::default());
            self.particles(&mut fr, "neighbors", Pass::default());
            let lumps = self.cpu.params.fluid.lump_stiffness > 0.0;
            let plastic = Pass { i0: [1, 0, 0, 0], ..Default::default() };
            for _ in 0..iterations {
                self.particles(&mut fr, "lambdas", Pass::default());
                self.particles(&mut fr, "deltas", Pass::default());
                if lumps {
                    // corrections, then each lump back toward its shape, then the solids
                    self.particles(&mut fr, "applyDeltas", Pass::default());
                    self.run(&mut fr, "lumpMatch", MAX_LUMPS as u32, Pass::default());
                    self.particles(&mut fr, "lumpPull", Pass::default());
                }
                self.particles(&mut fr, "applyCollide", Pass::default());
            }
            if lumps {
                self.run(&mut fr, "lumpMatch", MAX_LUMPS as u32, Pass::default());
                self.particles(&mut fr, "lumpPull", plastic);
            }
            self.particles(&mut fr, "velocities", Pass::default());
            self.particles(&mut fr, "xsph", Pass::default());
        }
        // 5. freeze resting particles into the SDFs
        let cap_groups = CAP.div_ceil(WG);
        self.run(&mut fr, "freezeMark", cap_groups, Pass::default());
        self.scan(&mut fr, true, self.l.x.get("SCANU"), self.l.x.get("IDX"), CAP);
        self.run(&mut fr, "freezeList", cap_groups, Pass::default());
        self.run(&mut fr, "freezeStamp", 1, Pass::default());
        self.compact(&mut fr);
        // 6. ledger: owed ice back onto the heap
        let cup = self.content_box(&pose);
        let cup_pass = Pass { x0: [cup[0] as f32, cup[1] as f32, cup[2] as f32, 0.0], x1: [cup[3] as f32, cup[4] as f32, cup[5] as f32, 0.0], ..Default::default() };
        self.run(&mut fr, "ledgerBins", 1, Pass::default());
        self.run(&mut fr, "ledgerPay", 1, cup_pass);
        // 7. surface tension: volume-preserving smoothing of what was edited, then settling
        let sm = self.cpu.params.ice.smoothing;
        let bsm = self.cpu.params.ice.bowl_smoothing;
        for k in 0..3 {
            self.smooth(&mut fr, k, 0, bsm);
            // settling smooths the whole recently edited box for 30 frames: that would round the cuts off
            // again, so only with smooth_cuts
            if self.cpu.params.ice.smooth_cuts {
                self.smooth(&mut fr, k, 1, bsm);
            }
        }
        self.smooth(&mut fr, 3, 0, sm);
        if self.content_edited || self.stats.settle_frames[3] > 0 {
            self.redist_pending = true;
        }
        let nc = self.l.nc;
        let c3 = (self.cpu.content.cell.powi(3)) as f32;
        if self.redist_pending && self.frame.is_multiple_of(3) && self.stats.content_vol > 0.0 {
            let wgs = nc.div_ceil(1024);
            self.run(&mut fr, "gridOcc", wgs, Pass::i([3, 0, 0, 0]));
            self.run(&mut fr, "finishSum", 1, Pass { i0: [0, wgs as i32, (st::RESIDUAL + 3) as i32, 1], f0: [c3, 0.0, 0.0, 0.0], ..Default::default() });
            let g64 = nc.div_ceil(WG);
            self.run(&mut fr, "rdInit", g64, Pass::default());
            for it in 0..RD_ITERS {
                self.run(&mut fr, "rdIter", g64, Pass::i([(it & 1) as i32, 0, 0, 0]));
            }
            // RD_ITERS is even: the result is back in RU0
            self.run(&mut fr, "rdFinish", g64, Pass::default());
            self.run(&mut fr, "gridOcc", wgs, Pass::i([3, 0, 0, 0]));
            self.run(&mut fr, "finishSum", 1, Pass { i0: [0, wgs as i32, (st::RESIDUAL + 3) as i32, 2], f0: [c3, 0.0, 0.0, 0.0], ..Default::default() });
            self.redist_pending = self.stats.settle_frames[3] > 0;
        }
        for k in 0..3 {
            self.smooth(&mut fr, 3, if k == 0 { 1 } else { 2 }, sm);
        }
        let cw = nc.div_ceil(256);
        self.run(&mut fr, "clampContent", cw, Pass::default());
        self.run(&mut fr, "finishSum", 1, Pass { i0: [0, cw as i32, (st::RESIDUAL + 3) as i32, 1], f0: [c3, 0.0, 0.0, 0.0], ..Default::default() });
        self.run(&mut fr, "ledgerResiduals", 1, Pass::default());
        // 8. statistics
        for k in 0..NG {
            self.grid_occ(&mut fr, k, st::VOLUME + k as u32, 0);
        }
        self.grid_occ(&mut fr, 3, st::CONTENT_VOL, 0);
        self.run(&mut fr, "ledgerStats", 1, Pass::default());
        self.copy_stats(&mut fr);
        self.submit(fr);
        self.read_stats();
        if self.stream {
            self.stream_out();
        }
        self.stats.gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.time += dt;
        self.frame += 1;
    }

    /// Bring the streamed data up to date without a frame (after reset / refill while paused).
    pub fn refresh(&mut self) {
        if self.stream {
            self.stream_out();
        }
    }

    /// Pack and read back what the viewer needs: grids that changed, particles, crumbs.
    /// For a renderer on the same device: pack the grids that changed since the last call (r8 distance and
    /// rgba8 colour rows of 256 bytes, ready for copy_buffer_to_texture) and the particles + crumbs, all on
    /// the GPU; nothing is read back. Then copy from `render_buffers`.
    pub fn pack_for_render(&mut self) -> RenderPack {
        let mut fr = Frame::new();
        let force = std::mem::take(&mut self.render_force);
        let mut grids = [false; NG];
        for (k, packed) in grids.iter_mut().enumerate() {
            if !(self.stats.dirty[k] || self.stats.dirty[NG + k] || force) {
                continue;
            }
            let g = &self.l.grids[k];
            let (dw, cw) = pack_row_words(g.n[0]);
            self.run(&mut fr, "packGrid", g.n[1] * g.n[2], Pass::i([k as i32, self.pack_off[k] as i32, dw as i32, cw as i32]));
            *packed = true;
        }
        self.particles(&mut fr, "packParticles", Pass::i([(CAP * 4) as i32, 0, 0, 0]));
        self.submit(fr);
        if !self.stream {
            // nobody else consumes the dirty marks
            self.queue.write_buffer(&self.x, (self.l.x.get("STU") + stu::DIRTY) as u64 * 4, &[0u8; 8 * NG]);
            self.stats.dirty = [false; 2 * NG];
        }
        RenderPack { grids, n: self.stats.n.min(CAP), crumbs: self.stats.spilled.min(MAX_SPILLED) }
    }

    /// Packed grids, per grid (byte offset in it, distance row bytes, colour row bytes), particles +
    /// crumbs, byte offset of the crumbs.
    pub fn render_buffers(&self) -> (&wgpu::Buffer, Vec<[u64; 3]>, &wgpu::Buffer, u64) {
        let rows = self.l.grids.iter().zip(&self.pack_off).map(|(g, &o)| {
            let (dw, cw) = pack_row_words(g.n[0]);
            [o as u64 * 4, dw as u64 * 4, cw as u64 * 4]
        });
        (&self.pk, rows.collect(), &self.pp, CAP as u64 * 16)
    }

    fn stream_out(&mut self) {
        let mut fr = Frame::new();
        let force = std::mem::take(&mut self.force_pack);
        let mut packed = Vec::new();
        for k in 0..4 {
            let (d, c) = (self.stats.dirty[k] || force, self.stats.dirty[NG + k] || force);
            // accumulate until the viewer takes them (several frames may pass between two views)
            self.dist_dirty[k] |= d;
            self.color_dirty[k] |= c;
            if !(d || c) {
                continue;
            }
            let g = &self.l.grids[k];
            let rows = g.n[1] * g.n[2];
            let (dw, cw) = pack_row_words(g.n[0]);
            self.run(&mut fr, "packGrid", rows, Pass::i([k as i32, self.pack_off[k] as i32, dw as i32, cw as i32]));
            let off = self.pack_off[k] as u64 * 4;
            self.copy(&mut fr, Buf::Pk, off, 1, off, rows as u64 * (dw + cw) as u64 * 4);
            packed.push(k);
        }
        let n = self.stats.n.min(CAP) as u64;
        let ns = self.stats.spilled.min(MAX_SPILLED) as u64;
        let spill_changed = ns as usize * 16 != self.spilled.len() || force;
        self.particles(&mut fr, "packParticles", Pass::i([(CAP * 4) as i32, 0, 0, 0]));
        if n > 0 {
            self.copy(&mut fr, Buf::Pp, 0, 2, 0, n * 16);
        }
        if spill_changed && ns > 0 {
            self.copy(&mut fr, Buf::Pp, CAP as u64 * 16, 2, CAP as u64 * 16, ns * 16);
        }
        self.submit(fr);
        // the dirty marks are consumed: clear them before the next frame
        self.queue.write_buffer(&self.x, (self.l.x.get("STU") + stu::DIRTY) as u64 * 4, &[0u8; 8 * NG]);
        if !packed.is_empty() {
            let grids = self.l.grids.clone();
            let offs = self.pack_off.clone();
            let size = self.pack_size as u64 * 4;
            let (mut dist, mut color) = (std::mem::take(&mut self.dist), std::mem::take(&mut self.color));
            self.read(1, size, |b| {
                for &k in &packed {
                    let g = &grids[k];
                    let (nx, rows) = (g.n[0] as usize, (g.n[1] * g.n[2]) as usize);
                    let o = offs[k] as usize * 4;
                    let (dw, cw) = pack_row_words(g.n[0]);
                    let (db, cb) = (dw as usize * 4, cw as usize * 4);
                    for r in 0..rows {
                        dist[k][r * nx..r * nx + nx].copy_from_slice(&b[o + r * db..o + r * db + nx]);
                        let oc = o + rows * db + r * cb;
                        color[k][r * nx * 4..(r + 1) * nx * 4].copy_from_slice(&b[oc..oc + nx * 4]);
                    }
                }
            });
            self.dist = dist;
            self.color = color;
        }
        let total = if spill_changed { (CAP as u64 + ns) * 16 } else { n * 16 };
        if total > 0 {
            let (p, s) = self.read(2, total.max(16), |b| {
                let p = b[..n as usize * 16].to_vec();
                let s = if spill_changed { Some(b[CAP as usize * 16..CAP as usize * 16 + ns as usize * 16].to_vec()) } else { None };
                (p, s)
            });
            self.particles = p;
            if let Some(s) = s {
                self.spilled = s;
                self.spilled_version += 1;
            }
        } else {
            self.particles.clear();
            if spill_changed {
                self.spilled.clear();
                self.spilled_version += 1;
            }
        }
    }

    /// "Eat": empty the spoon, including loose bits riding on it. Returns grams eaten.
    pub fn eat(&mut self) -> f64 {
        let mut fr = Frame::new();
        let (q, p) = (self.cpu.spoon.quat, self.cpu.spoon.pos);
        let pose = pose_rows(q, p);
        let sub = self.cpu.params.fluid.substeps.max(1);
        self.write_globals(1.0 / 60.0, sub, &pose, Sim::tilt_deg_of(q));
        let before = self.stats.eaten_vol;
        let nc = self.l.nc;
        let wgs = nc.div_ceil(1024);
        let c3 = self.cpu.content.cell.powi(3) as f32;
        self.run(&mut fr, "gridOcc", wgs, Pass::i([3, 0, 0, 0]));
        self.run(&mut fr, "finishSum", 1, Pass { i0: [0, wgs as i32, st::EATEN_VOL as i32, 1], f0: [c3, 0.0, 0.0, 0.0], ..Default::default() });
        self.run(&mut fr, "clearContent", nc.div_ceil(WG), Pass::i([0, 0, 0, 0]));
        self.particles(&mut fr, "eatMark", Pass::default());
        self.run(&mut fr, "eatDone", 1, Pass::default());
        self.compact(&mut fr);
        self.grid_occ(&mut fr, 3, st::CONTENT_VOL, 0);
        self.grid_occ(&mut fr, 3, st::VOLUME + 3, 0);
        self.run(&mut fr, "ledgerStats", 1, Pass::default());
        self.copy_stats(&mut fr);
        self.submit(fr);
        let dirty_before = self.stats.dirty;
        self.read_stats();
        for (d, b) in self.stats.dirty.iter_mut().zip(dirty_before) {
            *d |= b;
        }
        if self.stream {
            self.stream_out();
        }
        (self.stats.eaten_vol - before) * 1e6 * self.cpu.params.ice.density
    }

    // ------------------------------------------------------------------ readouts
    pub fn grams(&self, vol: f64) -> f64 {
        vol * 1e6 * self.cpu.params.ice.density
    }
    pub fn particle_volume(&self) -> f64 {
        self.cpu.particle_volume()
    }
    /// Everything accounted for (g): ice in bowls and on the spoon, loose, owed, eaten, spilled.
    pub fn total_grams(&self) -> f64 {
        let s = &self.stats;
        let vol = s.volume[0] + s.volume[1] + s.volume[2] + s.content_vol + s.n as f64 * self.particle_volume() + s.debt_vol + s.eaten_vol + s.spilled_vol;
        self.grams(vol)
    }

    /// The CPU simulation's state shape, filled from the GPU's statistics.
    pub fn state(&self) -> icecream_core::State {
        let sp = &self.cpu.spoon;
        let s = &self.stats;
        icecream_core::State {
            time: self.time,
            spoon_position: sp.pos.to_array(),
            spoon_quaternion: sp.quat.to_array(),
            spoon_euler_xyz_deg: sp.quat.to_euler_xyz().map(|a| a.to_degrees()),
            linear_velocity: sp.lin_vel.to_array(),
            angular_velocity: sp.ang_vel.to_array(),
            tilt_deg: Sim::tilt_deg_of(sp.quat),
            target_position: sp.target_pos.to_array(),
            target_quaternion: sp.target_quat.to_array(),
            contact: sp.contacts > 0,
            contact_samples: sp.contacts,
            blocked: sp.blocked,
            speed_scale: self.speed_scale,
            cut_force_n: self.cut_force,
            scooped_grams: self.grams(s.content_vol),
            eaten_grams: self.grams(s.eaten_vol),
            spilled_grams: self.grams(s.spilled_vol),
            bowl_grams: [0, 1, 2].map(|k| self.grams(s.volume[k])),
            particles: s.n as usize,
        }
    }

    // ------------------------------------------------------------------ debugging / tests
    /// Record passes, submit, wait.
    pub fn debug_run(&mut self, build: impl FnOnce(&mut GpuSim, &mut Frame)) {
        let mut fr = Frame::new();
        build(self, &mut fr);
        self.submit(fr);
        self.device.poll(wgpu::Maintain::Wait);
    }
    /// Read `count` elements at `offset` of buffer 'W', 'X' or 'A' as raw u32 words.
    pub fn debug_read(&self, name: char, offset: u32, count: u32) -> Vec<u32> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        let src = match name {
            'W' => &self.w,
            'X' => &self.x,
            _ => &self.a,
        };
        enc.copy_buffer_to_buffer(src, offset as u64 * 4, &self.staging[3], 0, count as u64 * 4);
        self.queue.submit([enc.finish()]);
        self.read(3, count as u64 * 4, |b| bytemuck::cast_slice(b).to_vec())
    }
    pub fn debug_read_f32(&self, offset: u32, count: u32) -> Vec<f32> {
        self.debug_read('W', offset, count).into_iter().map(f32::from_bits).collect()
    }
    pub fn debug_write(&self, name: char, offset: u32, data: &[u8]) {
        let dst = match name {
            'W' => &self.w,
            'X' => &self.x,
            _ => &self.a,
        };
        self.queue.write_buffer(dst, offset as u64 * 4, data);
    }
    pub fn debug_scan(&self, fr: &mut Frame, unsigned: bool, src: u32, dst: u32, n: u32) {
        self.scan(fr, unsigned, src, dst, n);
    }
    pub fn debug_grid_occ(&self, fr: &mut Frame, k: usize, slot: u32) {
        self.grid_occ(fr, k, slot, 0);
    }
    pub fn debug_smooth(&self, fr: &mut Frame, k: usize, mode: i32, strength: f64) {
        self.smooth(fr, k, mode, strength);
    }
    pub fn debug_frame_start(&self, fr: &mut Frame) {
        self.run(fr, "frameStart", 1, Pass::default());
    }
    pub fn debug_compact(&self, fr: &mut Frame) {
        self.compact(fr);
    }
}
