//! The ice-cream simulation: three bowls of SDF ice cream, a spoon that scoops it into its own
//! SDF, loose PBF particles that freeze back into solid ice, and a volume ledger that keeps
//! the total mass constant. Ported from the JS version this project started with (removed; `tests/parity.rs` pins the port to frozen fixtures from it).

use crate::fluid::{Collide, Fluid, CONTACT_NONE, CONTACT_SPOON, CONTACT_WORLD};
use crate::grid::SdfGrid;
use crate::math::{acos, cbrt, cos, hypot2, hypot3, js_round, pow, rad_to_deg, sin, Pose, Quat, Vec3};
use crate::params::{FillShape, Params};
use crate::grid::NO_FLOOR;
use crate::sdf::{
    bowls, grad_local, nearest_bowl, sd_bowl, sd_ellipsoid, sd_env, sd_handle, sd_hull_interior, sd_spoon_shell,
    sd_spoon_solid, smin, FLAVORS, SPOON,
};
use crate::spoon::{point_velocity, SpoonController, Substep};
use std::f64::consts::PI;

const WORLD_CELL: f64 = 0.0025;
const WORLD_DIMS: [usize; 3] = [60, 60, 60];
const CONTENT_ORIGIN: [f64; 3] = [-0.032, -0.014, -0.022];
const CONTENT_CELL: f64 = 0.0016;
const CONTENT_DIMS: [usize; 3] = [40, 25, 28];
const MAX_SPILLED: usize = 8000;
/// re-deposit stamps per frame (keeps the frame time bounded)
const DEBT_STAMPS_PER_FRAME: u32 = 48;
const DEBT_MAX_ENTRIES: usize = 256;
const BOWL_T_HALF: f64 = 0.0015;
/// displaced ice becomes at most this many loose particles per frame; the rest goes
/// straight onto the heap through the ledger
const DISPLACED_PARTICLES_PER_FRAME: i32 = 40;

fn srgb_to_linear(v: f64) -> f64 {
    let v = v / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        pow((v + 0.055) / 1.055, 2.4)
    }
}

/// Flavour colour -> RGBA8 (rgb linear, a = speck amount).
pub fn flavor_rgba(color: u32, speck: f64) -> [u8; 4] {
    [
        js_round(srgb_to_linear(((color >> 16) & 255) as f64) * 255.0) as u8,
        js_round(srgb_to_linear(((color >> 8) & 255) as f64) * 255.0) as u8,
        js_round(srgb_to_linear((color & 255) as f64) * 255.0) as u8,
        js_round(speck * 255.0) as u8,
    ]
}

/// Which SDF volume a ledger entry belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridId {
    Bowl(usize),
    Content,
}

#[derive(Clone, Debug)]
struct Debt {
    g: GridId,
    x: f64,
    y: f64,
    z: f64,
    vol: f64,
    rgba: [u8; 4],
    tries: u32,
}

/// Spoon pose and velocity as seen by the ice-cream code (during sub-step replay this is the
/// sub-step pose with the previous frame's velocities, exactly what the JS callback sees).
#[derive(Clone, Copy, Debug)]
struct SpoonView {
    pos: Vec3,
    quat: Quat,
    lin: Vec3,
    ang: Vec3,
}

/// Snapshot of the simulation for the digital-twin API (Y-up frame, metres, grams).
#[derive(Clone, Debug)]
pub struct State {
    pub time: f64,
    pub spoon_position: [f64; 3],
    pub spoon_quaternion: [f64; 4],
    pub spoon_euler_xyz_deg: [f64; 3],
    pub linear_velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub tilt_deg: f64,
    pub target_position: [f64; 3],
    pub target_quaternion: [f64; 4],
    pub contact: bool,
    pub contact_samples: usize,
    pub blocked: bool,
    pub speed_scale: f64,
    pub cut_force_n: f64,
    pub scooped_grams: f64,
    pub eaten_grams: f64,
    pub spilled_grams: f64,
    pub bowl_grams: [f64; 3],
    pub particles: usize,
}

fn grid_at(grids: &[SdfGrid], x: f64, y: f64, z: f64) -> Option<usize> {
    let i = nearest_bowl(x).index;
    if grids[i].contains(x, y, z, 0.0) {
        Some(i)
    } else {
        None
    }
}

fn box_hits_grid(b: &[f64; 6], g: &SdfGrid) -> bool {
    !(b[3] < g.ox || b[0] > g.ox + g.sx || b[4] < g.oy || b[1] > g.oy + g.sy || b[5] < g.oz || b[2] > g.oz + g.sz)
}

fn local_box_to_world(pose: &Pose, min: [f64; 3], max: [f64; 3]) -> [f64; 6] {
    let mut out = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for m in 0..8 {
        let x = if m & 1 != 0 { max[0] } else { min[0] };
        let y = if m & 2 != 0 { max[1] } else { min[1] };
        let z = if m & 4 != 0 { max[2] } else { min[2] };
        let w = pose.to_world(x, y, z);
        for a in 0..3 {
            if w[a] < out[a] {
                out[a] = w[a];
            }
            if w[a] > out[a + 3] {
                out[a + 3] = w[a];
            }
        }
    }
    out
}

pub struct Sim {
    pub params: Params,
    pub flavor_rgba: [[u8; 4]; 3],
    pub grids: Vec<SdfGrid>,
    pub content: SdfGrid,
    pub spoon: SpoonController,
    pub fluid: Fluid,
    pose: Pose,
    pub spilled: Vec<f32>,
    pub spilled_rgba: Vec<u8>,
    pub n_spilled: usize,
    pub spilled_version: u64,
    pub time: f64,
    pub frame: u64,
    pub speed_scale: f64,
    pub cut_force: f64,
    pub eaten_volume: f64,
    pub spilled_volume: f64,
    pub content_volume: f64,
    pub bowl_volumes: [f64; 3],
    releasing: bool,
    debts: Vec<Debt>,
    credit: Vec<(GridId, f64)>,
    loose_owed: f64,
    squeezed: Vec<u8>,
    rng: u32,
    last_carve_pos: Vec3,
    last_carve_quat: Quat,
    displaced_acc: f64,
    pub debt_spilled: f64,
    redist_pending: bool,
    content_floor: Option<Vec<f32>>,
    cup_box: [f64; 6],
    capture_budget: f64,
    spawn_budget: i32,
    removed: f64,
    scratch_c: [u8; 4],
    cur: SpoonView,
    substeps: Vec<Substep>,
}

impl Default for Sim {
    fn default() -> Self {
        Sim::new(Params::default())
    }
}

impl Sim {
    pub fn new(params: Params) -> Self {
        let flavor_rgba = [
            flavor_rgba(FLAVORS[0].color, FLAVORS[0].speck),
            flavor_rgba(FLAVORS[1].color, FLAVORS[1].speck),
            flavor_rgba(FLAVORS[2].color, FLAVORS[2].speck),
        ];
        let grids = bowls().iter().map(|b| SdfGrid::new([b.cx - 0.075, -0.006, b.cz - 0.075], WORLD_CELL, WORLD_DIMS, 4.0)).collect();
        let spoon = SpoonController::new(params.spoon.clone());
        let fluid = Fluid::new(16000);
        let cap = fluid.cap;
        let mut s = Sim {
            params,
            flavor_rgba,
            grids,
            content: SdfGrid::new(CONTENT_ORIGIN, CONTENT_CELL, CONTENT_DIMS, 4.0),
            spoon,
            fluid,
            pose: Pose::default(),
            spilled: vec![0.0; MAX_SPILLED * 3],
            spilled_rgba: vec![0; MAX_SPILLED * 4],
            n_spilled: 0,
            spilled_version: 0,
            time: 0.0,
            frame: 0,
            speed_scale: 1.0,
            cut_force: 0.0,
            eaten_volume: 0.0,
            spilled_volume: 0.0,
            content_volume: 0.0,
            bowl_volumes: [0.0; 3],
            releasing: false,
            debts: Vec::new(),
            credit: Vec::new(),
            loose_owed: 0.0,
            squeezed: vec![0; cap],
            rng: 12345,
            last_carve_pos: Vec3::ZERO,
            last_carve_quat: Quat::IDENTITY,
            displaced_acc: 0.0,
            debt_spilled: 0.0,
            redist_pending: false,
            content_floor: None,
            cup_box: [0.0; 6],
            capture_budget: 0.0,
            spawn_budget: DISPLACED_PARTICLES_PER_FRAME,
            removed: 0.0,
            scratch_c: [0; 4],
            cur: SpoonView { pos: Vec3::ZERO, quat: Quat::IDENTITY, lin: Vec3::ZERO, ang: Vec3::ZERO },
            substeps: Vec::new(),
        };
        s.reset();
        s
    }

    // ------------------------------------------------------------------ setup
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.frame = 0;
        self.speed_scale = 1.0;
        self.cut_force = 0.0;
        self.eaten_volume = 0.0;
        self.spilled_volume = 0.0;
        self.n_spilled = 0;
        self.spilled_version = 0;
        self.releasing = false;
        self.debts.clear();
        self.credit.clear();
        self.loose_owed = 0.0;
        self.squeezed.iter_mut().for_each(|v| *v = 0);
        self.rng = 12345;
        self.fluid.clear();
        self.content.clear();
        self.content.fill_color(self.flavor_rgba[0]);
        self.spoon.reset();
        self.sync_view();
        let shape = self.params.ice.shape;
        for i in 0..3 {
            self.init_bowl(i, shape);
        }
        self.last_carve_pos = self.spoon.pos;
        self.last_carve_quat = self.spoon.quat;
        self.update_stats(true);
    }

    pub fn refill(&mut self) {
        let shape = self.params.ice.shape;
        for i in 0..3 {
            self.init_bowl(i, shape);
        }
        self.update_stats(true);
    }

    fn init_bowl(&mut self, index: usize, shape: FillShape) {
        let b = &bowls()[index];
        let fill = b.rim_y - if shape == FillShape::Flat { 0.006 } else { 0.012 };
        let inner = b.inner_r - 0.0006;
        let mut balls: Vec<[f64; 4]> = Vec::new();
        match shape {
            FillShape::Scoops => {
                for k in 0..3 {
                    let a = (k as f64 / 3.0) * PI * 2.0 + index as f64 * 0.7;
                    balls.push([b.cx + cos(a) * 0.027, fill + 0.012, b.cz + sin(a) * 0.027, 0.026]);
                }
            }
            FillShape::Mound => balls.push([b.cx, fill - 0.035, b.cz, 0.062]),
            FillShape::Flat => {}
        }
        let g = &mut self.grids[index];
        let c = g.cell;
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let x = g.ox + (i as f64 + 0.5) * c;
                    let y = g.oy + (j as f64 + 0.5) * c;
                    let z = g.oz + (k as f64 + 0.5) * c;
                    let d_in = hypot3(x - b.cx, y - b.cy, z - b.cz) - inner;
                    let mut d = d_in.max(y - fill);
                    for bl in &balls {
                        let db = hypot3(x - bl[0], y - bl[1], z - bl[2]) - bl[3];
                        d = smin(d, db, 0.008);
                    }
                    // never let the blend bulge into the bowl wall
                    if y < b.rim_y {
                        d = d.max(d_in);
                    }
                    let idx = i + g.nx * (j + g.ny * k);
                    g.data[idx] = ((-g.band).max(g.band.min(d))) as f32;
                }
            }
        }
        g.fill_color(self.flavor_rgba[index]);
        g.dirty = true;
    }

    // ------------------------------------------------------------------ helpers
    fn sync_view(&mut self) {
        self.cur = SpoonView { pos: self.spoon.pos, quat: self.spoon.quat, lin: self.spoon.lin_vel, ang: self.spoon.ang_vel };
    }

    fn set_pose(&mut self, pos: Vec3, quat: Quat) {
        self.pose = Pose::new(pos, quat);
    }

    pub fn tilt_deg_of(quat: Quat) -> f64 {
        let uy = 1.0 - 2.0 * (quat.x * quat.x + quat.z * quat.z);
        rad_to_deg(acos((-1.0f64).max(1.0f64.min(uy))))
    }

    pub fn tilt_deg(&self) -> f64 {
        Sim::tilt_deg_of(self.spoon.quat)
    }

    fn grid(&self, g: GridId) -> &SdfGrid {
        match g {
            GridId::Bowl(i) => &self.grids[i],
            GridId::Content => &self.content,
        }
    }

    fn grid_mut(&mut self, g: GridId) -> &mut SdfGrid {
        match g {
            GridId::Bowl(i) => &mut self.grids[i],
            GridId::Content => &mut self.content,
        }
    }

    fn content_box(&self) -> [f64; 6] {
        local_box_to_world(
            &self.pose,
            CONTENT_ORIGIN,
            [CONTENT_ORIGIN[0] + self.content.sx, CONTENT_ORIGIN[1] + self.content.sy, CONTENT_ORIGIN[2] + self.content.sz],
        )
    }

    // ------------------------------------------------------------------ main step
    pub fn step(&mut self, dt: f64) {
        let dt = dt.min(1.0 / 30.0);
        let start_pos = self.spoon.pos;
        let start_quat = self.spoon.quat;

        // 1. spoon motion, carving the ice along the swept path
        self.removed = 0.0;
        self.spawn_budget = DISPLACED_PARTICLES_PER_FRAME;
        self.spoon.p = self.params.spoon.clone();
        let (old_lin, old_ang) = (self.spoon.lin_vel, self.spoon.ang_vel);
        let mut subs = std::mem::take(&mut self.substeps);
        subs.clear();
        self.spoon.update(dt, self.speed_scale, Some(&mut subs));
        for s in &subs {
            self.cur = SpoonView { pos: s.pos, quat: s.quat, lin: old_lin, ang: old_ang };
            self.interact(s.pos, s.quat, s.dt, false);
        }
        self.substeps = subs;
        self.sync_view();
        self.interact(self.spoon.pos, self.spoon.quat, dt, true);

        let sp_pos = self.spoon.pos;
        let sp_quat = self.spoon.quat;
        let travel = sp_pos.distance_to(start_pos) + start_quat.angle_to(sp_quat) * 0.03;
        let area = (self.removed / travel.max(0.0005)).min(PI * SPOON.a * SPOON.c);
        let force = self.params.ice.hardness * 1000.0 * area;
        self.cut_force += (force - self.cut_force) * 0.3;
        let max_cut = self.params.spoon.max_cut_force;
        let target = if self.cut_force > max_cut { max_cut / self.cut_force } else { 1.0 };
        self.speed_scale += (0.05f64.max(target) - self.speed_scale) * 0.3;

        // 2. carry loose ice that sits in / on the spoon along with it
        self.set_pose(sp_pos, sp_quat);
        let tilt = self.tilt_deg();
        let release = self.params.ice.release_angle;
        let tilted = tilt > release;
        if !tilted {
            self.carry_particles(start_pos, start_quat, sp_pos, sp_quat);
        }

        // 3. tilted past the release angle: the scoop slides off as particles
        if tilted && !self.releasing {
            self.release_content();
            self.releasing = true;
        } else if tilt < release - 10.0 {
            self.releasing = false;
        }

        // 4. particles
        let fp = self.params.fluid.clone();
        let sub = fp.substeps.max(1);
        self.fluid.mark_frame();
        for _ in 0..sub {
            let mut env = Env {
                grids: &self.grids,
                content: &self.content,
                pose: self.pose,
                sp: self.cur,
                squeezed: &mut self.squeezed,
            };
            self.fluid.step(dt / sub as f64, &fp, &mut env);
        }
        self.freeze_resting(tilt < release - 10.0, dt);
        self.pay_debts();
        let sm = self.params.ice.smoothing;
        let bsm = self.params.ice.bowl_smoothing;
        let tilted_now = tilt > release;
        for i in 0..3 {
            let g = &self.grids[i];
            {
                let keep_out = spoon_keep_out(self.pose, &self.content, tilted_now, [g.ox, g.oy, g.oz, g.cell], [g.nx, g.ny], g.band);
                self.grids[i].smooth_touched(bsm, &keep_out);
                // settling smooths the whole recently edited box for 30 frames: that would round the cuts
                // off again, so only with smooth_cuts
                if self.params.ice.smooth_cuts {
                    self.grids[i].settle(bsm, true, &keep_out);
                }
            }
            self.settle_residual(GridId::Bowl(i));
        }
        let content_edited = self.content.n_touched > 0;
        self.content.smooth_touched(sm, NO_FLOOR);
        if content_edited {
            self.redist_pending = true;
        }
        if self.redist_pending && self.frame % 3 == 0 && self.content_volume > 0.0 {
            self.content.redistance();
            self.redist_pending = self.content.settle_frames > 0;
        }
        for k in 0..3 {
            self.content.settle(sm, k == 0, NO_FLOOR);
        }
        self.clamp_content();
        self.settle_residual(GridId::Content);

        self.time += dt;
        self.frame += 1;
        self.update_stats(self.frame % 15 == 0);
    }

    /// Run `n` frames of 1/60 s.
    pub fn step_frames(&mut self, n: usize) {
        for _ in 0..n {
            self.step(1.0 / 60.0);
        }
    }

    // ------------------------------------------------------------------ spoon <-> ice
    /// Called along the spoon path: metal displaces ice (Cut -> particles), the cup interior
    /// copies ice into the spoon's SDF (scooped), and ice on the spoon accretes more (curl).
    fn interact(&mut self, pos: Vec3, quat: Quat, sub_dt: f64, force: bool) {
        let moved = self.last_carve_pos.distance_to(pos) + self.last_carve_quat.angle_to(quat) * 0.06;
        if !force && moved < 0.0009 {
            return;
        }
        if moved < 1e-7 {
            return;
        }
        self.last_carve_pos = pos;
        self.last_carve_quat = quat;

        let ip = self.params.ice.clone();
        self.set_pose(pos, quat);
        let pose = self.pose;
        let tilted = Sim::tilt_deg_of(quat) > ip.release_angle;
        let cap = ip.capacity * 1e-6;
        let accrete = ip.accretion && !tilted && self.content_volume < cap;

        let hb = self.content_box();
        let hdl = &SPOON.handle;
        let ymin = hdl.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let ymax = hdl.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
        let hb_handle = local_box_to_world(&pose, [hdl[0][0] - 0.004, ymin - 0.004, -0.004], [hdl[3][0] + 0.004, ymax + 0.004, 0.004]);

        let touching: Vec<usize> =
            (0..3).filter(|&i| box_hits_grid(&hb, &self.grids[i]) || box_hits_grid(&hb_handle, &self.grids[i])).collect();
        if touching.is_empty() {
            return;
        }

        // A. copy ice into the spoon's SDF (cup interior + accretion shell)
        if !tilted {
            let s = &SPOON;
            let delta = ip.accretion_dist;
            let content = &mut self.content;
            let c = content.cell;
            let mut changed = false;
            let mut was_empty = self.content_volume < 1e-9;
            let mut gained = 0.0;
            let (cnx, cny, cnz) = (content.nx, content.ny, content.nz);
            for k in 0..cnz {
                let lz = content.oz + (k as f64 + 0.5) * c;
                for j in 0..cny {
                    let ly = content.oy + (j as f64 + 0.5) * c;
                    let mut idx = cnx * (j + cny * k);
                    for i in 0..cnx {
                        let lx = content.ox + (i as f64 + 0.5) * c;
                        let cv = content.data[idx] as f64;
                        let in_hull_box = lx > -s.a && lx < s.a && ly > -s.b && ly < 0.0 && lz > -s.c && lz < s.c;
                        // accretion grows the scoop, but never on the outside bottom of the cup
                        let below = ly < 0.0 && sd_ellipsoid(lx, ly, lz, s.a + s.t, s.b + s.t, s.c + s.t) > 0.0;
                        // keep the scoop off the walls of its grid
                        let near_wall = i < 3 || j < 2 || k < 3 || i >= cnx - 3 || j >= cny - 3 || k >= cnz - 3;
                        let mut d_r = if accrete && !below && !near_wall { cv - delta } else { f64::INFINITY };
                        if in_hull_box {
                            d_r = d_r.min(sd_hull_interior(lx, ly, lz));
                        }
                        if d_r >= c {
                            idx += 1;
                            continue;
                        }
                        let w3 = pose.to_world(lx, ly, lz);
                        let Some(gi) = grid_at(&self.grids, w3[0], w3[1], w3[2]) else {
                            idx += 1;
                            continue;
                        };
                        let g = &self.grids[gi];
                        let w = g.sample(w3[0], w3[1], w3[2]);
                        if w >= g.band * 0.999 {
                            idx += 1;
                            continue;
                        }
                        // never inside the metal (keeps the rim clean)
                        let nv = w.max(d_r).max(-sd_spoon_shell(lx, ly, lz));
                        if nv < cv {
                            gained += content.occ(nv) - content.occ(cv);
                            content.data[idx] = nv as f32;
                            content.touch(idx);
                            changed = true;
                            if let Some(vi) = g.voxel_at(w3[0], w3[1], w3[2]) {
                                if nv < 2.0 * c {
                                    let mut col = [0u8; 4];
                                    col.copy_from_slice(&g.color[vi * 4..vi * 4 + 4]);
                                    self.scratch_c = col;
                                    // first ice on an empty spoon: its flavour becomes the base colour
                                    if was_empty {
                                        content.fill_color(col);
                                        was_empty = false;
                                    }
                                    content.blend_color(idx, col, if cv > 0.0 { 1.0 } else { 0.5 });
                                }
                            }
                        }
                        idx += 1;
                    }
                }
            }
            if changed {
                content.dirty = true;
                content.color_dirty = true;
            }
            self.capture_budget = gained * c * c * c;
        } else {
            self.capture_budget = 0.0;
        }

        // B. carve the bowls
        for &gi in &touching {
            self.carve_grid(gi, &hb, sub_dt, tilted, accrete, false);
            self.carve_grid(gi, &hb_handle, sub_dt, tilted, accrete, true);
        }
        // the spoon picked up more than the bowls gave up (resampling): the bowl owes it back
        if self.capture_budget > 0.0 {
            let c = self.scratch_c;
            self.settle(GridId::Bowl(touching[0]), 0.0, 0.0, 0.0, -self.capture_budget, c);
        }
        self.capture_budget = 0.0;
    }

    fn carve_grid(&mut self, gi: usize, bx: &[f64; 6], _sub_dt: f64, tilted: bool, accrete: bool, handle_only: bool) -> f64 {
        if !box_hits_grid(bx, &self.grids[gi]) {
            return 0.0;
        }
        let pose = self.pose;
        let (ox, oy, oz, c, nx, ny, nz, band) = {
            let g = &self.grids[gi];
            (g.ox, g.oy, g.oz, g.cell, g.nx, g.ny, g.nz, g.band)
        };
        let c3 = c * c * c;
        let pad = 0.4 * c;
        let fl = |v: f64, n: usize| -> i64 { (v.floor() as i64).min(n as i64 - 1) };
        let i0 = 0i64.max(((bx[0] - ox) / c).floor() as i64);
        let i1 = fl((bx[3] - ox) / c, nx);
        let j0 = 0i64.max(((bx[1] - oy) / c).floor() as i64);
        let j1 = fl((bx[4] - oy) / c, ny);
        let k0 = 0i64.max(((bx[2] - oz) / c).floor() as i64);
        let k1 = fl((bx[5] - oz) / c, nz);
        let to_particles = self.params.ice.displaced_to_particles;
        let smooth_cuts = self.params.ice.smooth_cuts;
        let pr = if self.fluid.r > 0.0 { self.fluid.r } else { self.params.fluid.particle_radius };
        let vp = 8.0 * pr * pr * pr;
        let mut displaced = 0.0;
        let mut changed = false;
        let occ = |d: f64| -> f64 {
            let o = 0.5 - d / c;
            if o < 0.0 {
                0.0
            } else if o > 1.0 {
                1.0
            } else {
                o
            }
        };
        let mut k = k0;
        while k <= k1 {
            let z = oz + (k as f64 + 0.5) * c - pose.t[2];
            let mut j = j0;
            while j <= j1 {
                let y = oy + (j as f64 + 0.5) * c - pose.t[1];
                let mut i = i0;
                let mut idx = (i0 + nx as i64 * (j + ny as i64 * k)) as usize;
                while i <= i1 {
                    let w = self.grids[gi].data[idx] as f64;
                    if w >= band * 0.999 {
                        i += 1;
                        idx += 1;
                        continue;
                    }
                    let x = ox + (i as f64 + 0.5) * c - pose.t[0];
                    let r = &pose.r;
                    let lx = r[0] * x + r[3] * y + r[6] * z;
                    let ly = r[1] * x + r[4] * y + r[7] * z;
                    let lz = r[2] * x + r[5] * y + r[8] * z;
                    let ds;
                    let mut di = f64::INFINITY;
                    let mut dc = f64::INFINITY;
                    if handle_only {
                        ds = sd_handle(lx, ly, lz) - pad;
                    } else {
                        ds = sd_spoon_shell(lx, ly, lz) - pad;
                        if !tilted {
                            di = sd_hull_interior(lx, ly, lz);
                        }
                        dc = self.content.sample(lx, ly, lz);
                    }
                    let d_carve = ds.min(di).min(dc);
                    let mut nw = -d_carve;
                    if nw <= w {
                        i += 1;
                        idx += 1;
                        continue;
                    }
                    if nw > band {
                        nw = band;
                    }
                    let dv = (occ(w) - occ(nw)) * c3;
                    self.grids[gi].data[idx] = nw as f32;
                    // a cut keeps its edge: smoothing only melts in what is stamped (unless smooth_cuts)
                    if smooth_cuts {
                        self.grids[gi].touch(idx);
                    }
                    changed = true;
                    self.removed += dv;
                    let is_metal = ds <= di && ds <= dc;
                    let is_content_no_accrete = !is_metal && dc < di && !accrete;
                    if dv > 0.0 {
                        if is_metal || is_content_no_accrete {
                            displaced += dv;
                            if to_particles {
                                self.displaced_acc += dv;
                                while self.displaced_acc >= vp {
                                    self.displaced_acc -= vp;
                                    if !self.spawn_displaced(lx, ly, lz, gi, idx, handle_only, pr) {
                                        self.owe_at(gi, idx, vp);
                                    }
                                }
                            } else {
                                self.owe_at(gi, idx, dv);
                            }
                        } else {
                            // swept by the cup: only what was actually copied onto the spoon counts
                            // as scooped; a full cup pushing deeper shoves the rest aside
                            let took = dv.min(self.capture_budget);
                            self.capture_budget -= took;
                            if dv > took {
                                self.owe_at(gi, idx, dv - took);
                            }
                        }
                    }
                    i += 1;
                    idx += 1;
                }
                j += 1;
            }
            k += 1;
        }
        if changed {
            self.grids[gi].dirty = true;
        }
        displaced
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_displaced(&mut self, lx: f64, ly: f64, lz: f64, gi: usize, idx: usize, handle_only: bool, r: f64) -> bool {
        if self.fluid.n >= self.params.fluid.max_particles || self.spawn_budget <= 0 {
            return false;
        }
        self.spawn_budget -= 1;
        let f: fn(f64, f64, f64) -> f64 = if handle_only { sd_handle } else { sd_spoon_solid };
        let d = f(lx, ly, lz);
        let n = grad_local(f, lx, ly, lz);
        let push = -d + r * 1.3;
        let w = self.pose.to_world(lx + n[0] * push, ly + n[1] * push, lz + n[2] * push);
        // pushed out of the spoon's underside at the bottom of a bowl, the spawn point can land
        // behind the bowl wall: that ice stays in the bowl (ledger) instead of leaking outside
        if sd_env(w[0], w[1], w[2]) < r || grid_at(&self.grids, w[0], w[1], w[2]).is_none() {
            return false;
        }
        let v = point_velocity(self.cur.pos, self.cur.lin, self.cur.ang, w[0], w[1], w[2]);
        let mut col = [0u8; 4];
        col.copy_from_slice(&self.grids[gi].color[idx * 4..idx * 4 + 4]);
        self.scratch_c = col;
        self.fluid.spawn(w[0], w[1], w[2], v[0], v[1], v[2], col);
        true
    }

    // ------------------------------------------------------------------ volume ledger
    /// Record `vol` m³ of ice from bowl voxel `idx` that still has to go somewhere.
    fn owe_at(&mut self, gi: usize, idx: usize, vol: f64) {
        let g = &self.grids[gi];
        let mut col = [0u8; 4];
        col.copy_from_slice(&g.color[idx * 4..idx * 4 + 4]);
        self.scratch_c = col;
        let p = g.center(idx);
        self.owe(GridId::Bowl(gi), p[0], p[1], p[2], vol, col);
    }

    /// Map a spoon-local point to the bowl grid under it (if any).
    fn content_to_bowl(&mut self, x: f64, y: f64, z: f64) -> ([f64; 3], Option<usize>) {
        self.set_pose(self.cur.pos, self.cur.quat);
        let w = self.pose.to_world(x, y, z);
        let bi = nearest_bowl(w[0]).index;
        let bg = &self.grids[bi];
        let ok = bg.contains(w[0], bg.oy + bg.sy * 0.5, w[2], 2.0 * bg.cell);
        (w, if ok { Some(bi) } else { None })
    }

    /// Record a debt at (x, y, z) in grid `g`'s coordinates (spoon-local for the content grid).
    fn owe(&mut self, mut g: GridId, mut x: f64, mut y: f64, mut z: f64, mut vol: f64, rgba: [u8; 4]) {
        if g == GridId::Content {
            // never pile owed ice onto the scoop: hand it to the bowl under the spoon, or let it
            // fall as loose particles
            let (w, bowl) = self.content_to_bowl(x, y, z);
            if let Some(bi) = bowl {
                let bg = &self.grids[bi];
                g = GridId::Bowl(bi);
                x = w[0];
                y = w[1].min(bg.oy + bg.sy - 2.0 * bg.cell);
                z = w[2];
            } else {
                self.loose_owed += vol;
                let vp = self.particle_volume();
                while self.loose_owed >= vp && self.fluid.n < self.params.fluid.max_particles {
                    self.loose_owed -= vp;
                    let r = self.fluid.r;
                    self.fluid.spawn(w[0], w[1] + r * 2.0, w[2], 0.0, 0.0, 0.0, rgba);
                }
                return;
            }
        }
        // a previous stamp that added a little more than it owed pays for this first
        if let Some(e) = self.credit.iter_mut().find(|e| e.0 == g) {
            if e.1 > 0.0 {
                let used = e.1.min(vol);
                e.1 -= used;
                vol -= used;
                if vol <= 0.0 {
                    return;
                }
            }
        }
        let near = 3.0 * self.grid(g).cell;
        let len = self.debts.len();
        let lo = len.saturating_sub(8);
        for n in (lo..len).rev() {
            let e = &mut self.debts[n];
            if e.g == g && (e.x - x).abs() < near && (e.y - y).abs() < near && (e.z - z).abs() < near {
                e.vol += vol;
                return;
            }
        }
        if len >= DEBT_MAX_ENTRIES {
            self.debts[len - 1].vol += vol;
            return;
        }
        self.debts.push(Debt { g, x, y, z, vol, rgba, tries: 0 });
    }

    /// The per-voxel floor of the content grid (built on first use; also clamps the content to it).
    pub fn content_floor(&mut self) -> Vec<f32> {
        self.clamp_content();
        self.content_floor.clone().unwrap()
    }

    /// Where the scoop may never be (spoon coordinates): under the outside of the cup, and
    /// within two cells of its grid's walls. Content values are clamped to this floor.
    fn clamp_content(&mut self) {
        if self.content_floor.is_none() {
            let c = &self.content;
            let mut fl = vec![0f32; c.n];
            let s = &SPOON;
            let m = 2.0 * c.cell;
            for k in 0..c.nz {
                let lz = c.oz + (k as f64 + 0.5) * c.cell;
                for j in 0..c.ny {
                    let ly = c.oy + (j as f64 + 0.5) * c.cell;
                    for i in 0..c.nx {
                        let lx = c.ox + (i as f64 + 0.5) * c.cell;
                        let wall = (lx - (c.ox + m))
                            .min(c.ox + c.sx - m - lx)
                            .min(ly - (c.oy + m))
                            .min(c.oy + c.sy - m - ly)
                            .min(lz - (c.oz + m))
                            .min(c.oz + c.sz - m - lz);
                        let outer = sd_ellipsoid(lx, ly, lz, s.a + s.t, s.b + s.t, s.c + s.t);
                        let sd_below = ly.max(-outer);
                        fl[i + c.nx * (j + c.ny * k)] = (-wall).max(-sd_below) as f32;
                    }
                }
            }
            self.content_floor = Some(fl);
        }
        let fl = self.content_floor.as_ref().unwrap();
        let c = &mut self.content;
        let mut removed = 0.0;
        for q in 0..c.n {
            if c.data[q] < fl[q] {
                removed += c.occ(c.data[q] as f64) - c.occ(fl[q] as f64);
                c.data[q] = fl[q];
            }
        }
        if removed > 0.0 {
            c.residual += removed * c.cell * c.cell * c.cell;
            c.dirty = true;
        }
        self.settle_residual(GridId::Content);
    }

    /// Smoothing that couldn't keep its volume hands the difference to the ledger.
    fn settle_residual(&mut self, g: GridId) {
        let gr = self.grid_mut(g);
        let r = gr.take_residual();
        if r.abs() < 1e-12 {
            return;
        }
        let idx = gr.residual_idx;
        let mut col = [0u8; 4];
        col.copy_from_slice(&gr.color[idx * 4..idx * 4 + 4]);
        let p = gr.center(idx);
        self.scratch_c = col;
        self.settle(g, p[0], p[1], p[2], r, col);
    }

    /// Signed balance: positive = still owed, negative = a stamp added more than it should.
    fn settle(&mut self, mut g: GridId, x: f64, y: f64, z: f64, vol: f64, rgba: [u8; 4]) {
        if g == GridId::Content && vol < 0.0 {
            // extra ice on the scoop: the bowl under the spoon gives up that much instead
            let (_, bowl) = self.content_to_bowl(x, y, z);
            match bowl {
                Some(bi) => g = GridId::Bowl(bi),
                None => {
                    self.loose_owed += vol;
                    return;
                }
            }
        }
        if vol > 0.0 {
            self.owe(g, x, y, z, vol, rgba);
            return;
        }
        // extra ice first cancels debts still pending in this grid
        let mut extra = -vol;
        let mut n = self.debts.len();
        while n > 0 && extra > 0.0 {
            n -= 1;
            if self.debts[n].g != g {
                continue;
            }
            let used = extra.min(self.debts[n].vol);
            self.debts[n].vol -= used;
            extra -= used;
            if self.debts[n].vol <= 0.0 {
                self.debts.swap_remove(n);
            }
        }
        if extra > 0.0 {
            match self.credit.iter_mut().find(|e| e.0 == g) {
                Some(e) => e.1 += extra,
                None => self.credit.push((g, extra)),
            }
        }
    }

    fn rand(&mut self) -> f64 {
        self.rng = (self.rng as u64 * 1664525 + 1013904223) as u32;
        self.rng as f64 / 4294967296.0
    }

    /// Is (x, y, z) at least `m` away from the spoon (metal and cup opening)? World coords.
    fn clear_of_spoon(&self, x: f64, y: f64, z: f64, m: f64) -> bool {
        let l = self.pose.to_local(x, y, z);
        sd_spoon_solid(l[0], l[1], l[2]) > m && sd_hull_interior(l[0], l[1], l[2]) > m
    }

    /// Is (x, z) under (or inside) the cup + scoop box, above height y?
    fn under_cup(&self, x: f64, y: f64, z: f64, m: f64) -> bool {
        let b = &self.cup_box;
        x > b[0] - m && x < b[3] + m && z > b[2] - m && z < b[5] + m && b[4] > y
    }

    /// Top ice surface of the column through (x, z), or NaN when the column has no ice.
    fn column_top(g: &SdfGrid, x: f64, z: f64) -> f64 {
        let c = g.cell;
        let mut y = g.oy + g.sy - c;
        while y > g.oy + c {
            if g.sample(x, y, z) < 0.0 {
                return y;
            }
            y -= 0.5 * c;
        }
        f64::NAN
    }

    /// Put owed ice back on the lowest free ice surface near where it went missing, clear of
    /// the spoon; each stamp's real added volume is measured, overshoot becomes credit.
    fn pay_debts(&mut self) {
        self.set_pose(self.cur.pos, self.cur.quat);
        self.cup_box = self.content_box();
        let mut stamps = self.pay_credit();
        if self.debts.is_empty() {
            return;
        }
        let k = self.params.fluid.stamp_smooth;
        let mut n = 0usize;
        while n < self.debts.len() && stamps < DEBT_STAMPS_PER_FRAME {
            let e = self.debts[n].clone();
            let g = e.g;
            let c = self.grid(g).cell;
            let rs = (2.5 * c).min((1.2 * c).max(cbrt((e.vol * 3.0) / (2.0 * PI))));
            // wait until there is enough owed for a stamp of the minimum size
            if e.vol < 0.25 * (2.0 / 3.0) * PI * rs * rs * rs && e.tries == 0 && stamps > 0 {
                n += 1;
                continue;
            }
            let (mut x, mut y, mut z) = (f64::NAN, f64::NAN, f64::NAN);
            match g {
                GridId::Content => {
                    let gr = &self.content;
                    let mut yy = e.y;
                    while yy < gr.oy + gr.sy - rs {
                        if gr.contains(e.x, yy, e.z, rs) && gr.sample(e.x, yy, e.z) >= 0.0 && sd_spoon_shell(e.x, yy, e.z) > rs {
                            x = e.x;
                            y = yy;
                            z = e.z;
                            break;
                        }
                        yy += 0.5 * c;
                    }
                }
                GridId::Bowl(bi) => {
                    let a = self.rand() * PI * 2.0;
                    let (dx, dz) = (cos(a), sin(a));
                    let bw = &bowls()[bi];
                    let gr = &self.grids[bi];
                    // stay inside the rim opening
                    let rim_in = bw.w - BOWL_T_HALF - rs - c;
                    for t in 0..16 {
                        let xx = e.x + dx * t as f64 * c;
                        let zz = e.z + dz * t as f64 * c;
                        if !gr.contains(xx, gr.oy + gr.sy * 0.5, zz, rs + c) {
                            break;
                        }
                        if hypot2(xx - bw.cx, zz - bw.cz) > rim_in {
                            continue;
                        }
                        let top = Sim::column_top(gr, xx, zz);
                        if top.is_nan() || top > gr.oy + gr.sy - 2.0 * rs - c {
                            continue;
                        }
                        if !(top < y) && !y.is_nan() {
                            continue;
                        }
                        if self.under_cup(xx, top, zz, rs) {
                            continue;
                        }
                        if self.clear_of_spoon(xx, top, zz, rs + 0.002) {
                            x = xx;
                            y = top;
                            z = zz;
                        }
                    }
                }
            }
            stamps += 1;
            self.debts[n].tries += 1;
            // stuck: jump to a random spot inside the bowl that isn't under the spoon
            if y.is_nan() && self.debts[n].tries > 8 {
                if let GridId::Bowl(bi) = g {
                    let b = &bowls()[bi];
                    for _ in 0..8 {
                        let a2 = self.rand() * PI * 2.0;
                        let r2 = self.rand().sqrt() * (b.w - 0.01);
                        let xx = b.cx + cos(a2) * r2;
                        let zz = b.cz + sin(a2) * r2;
                        if !self.under_cup(xx, -1.0, zz, rs) {
                            self.debts[n].x = xx;
                            self.debts[n].z = zz;
                            break;
                        }
                    }
                }
            }
            if !y.is_nan() {
                let added = self.grid_mut(g).stamp_sphere(x, y, z, rs, k, Some(e.rgba));
                self.debts[n].vol -= added;
            }
            if self.debts[n].vol <= 0.0 {
                let over = -self.debts[n].vol;
                let rgba = self.debts[n].rgba;
                self.debts.swap_remove(n);
                if over > 0.0 {
                    self.settle(g, 0.0, 0.0, 0.0, -over, rgba);
                }
            } else if self.debts[n].tries > 600 {
                // nowhere to put it (e.g. the bowl is full to the brim): count it as spilled
                let v = self.debts[n].vol;
                self.debt_spilled += v;
                self.spilled_volume += v;
                self.debts.swap_remove(n);
            } else {
                n += 1;
            }
        }
    }

    /// Credit = ice that exists in a grid but shouldn't. Shave it off the highest ice top
    /// that is clear of the spoon. Returns stamps used.
    fn pay_credit(&mut self) -> u32 {
        let mut stamps = 0;
        let mut ci = 0;
        while ci < self.credit.len() {
            let (g, cr) = self.credit[ci];
            ci += 1;
            let c = self.grid(g).cell;
            let rs = 1.5 * c;
            let min_vol = 0.5 * (2.0 / 3.0) * PI * rs * rs * rs;
            let mut credit = cr;
            if credit < min_vol || self.debts.iter().any(|e| e.g == g) {
                continue;
            }
            let mut tries = 0;
            while tries < 6 && credit >= min_vol {
                tries += 1;
                let (mut x, mut y, mut z) = (f64::NAN, f64::NAN, f64::NAN);
                match g {
                    GridId::Content => {
                        for _ in 0..12 {
                            let (sx, sz) = (self.content.sx, self.content.sz);
                            let xx = self.content.ox + rs + self.rand() * (sx - 2.0 * rs);
                            let zz = self.content.oz + rs + self.rand() * (sz - 2.0 * rs);
                            let gr = &self.content;
                            let mut yy = gr.oy + gr.sy - rs;
                            while yy > gr.oy + rs {
                                if gr.sample(xx, yy, zz) < 0.0 {
                                    if !(yy <= y) {
                                        x = xx;
                                        y = yy;
                                        z = zz;
                                    }
                                    break;
                                }
                                yy -= 0.5 * c;
                            }
                        }
                    }
                    GridId::Bowl(bi) => {
                        for _ in 0..12 {
                            let bw = &bowls()[bi];
                            let a2 = self.rand() * PI * 2.0;
                            let r2 = self.rand().sqrt() * (bw.w - 0.006);
                            let xx = bw.cx + cos(a2) * r2;
                            let zz = bw.cz + sin(a2) * r2;
                            let top = Sim::column_top(&self.grids[bi], xx, zz);
                            if top.is_nan() || top <= y {
                                continue;
                            }
                            if self.under_cup(xx, top, zz, rs) {
                                continue;
                            }
                            if self.clear_of_spoon(xx, top, zz, rs + 0.002) {
                                x = xx;
                                y = top;
                                z = zz;
                            }
                        }
                    }
                }
                stamps += 1;
                if y.is_nan() {
                    continue;
                }
                credit -= self.grid_mut(g).carve_sphere(x, y + 0.5 * rs, z, rs);
            }
            let slot = ci - 1;
            if credit < 0.0 {
                self.credit[slot].1 = 0.0;
                self.scratch_c = [255; 4];
                let col = match g {
                    GridId::Bowl(bi) => self.flavor_rgba[bi],
                    GridId::Content => [255; 4],
                };
                let gr = self.grid(g);
                let (cx, cy, cz) = (gr.ox + gr.sx * 0.5, gr.oy + gr.sy * 0.5, gr.oz + gr.sz * 0.5);
                self.owe(g, cx, cy, cz, -credit, col);
            } else {
                self.credit[slot].1 = credit;
            }
        }
        stamps
    }

    /// Ice owed but not yet re-deposited (m³), for mass bookkeeping.
    pub fn debt_volume(&self) -> f64 {
        let mut v = 0.0;
        for e in &self.debts {
            v += e.vol;
        }
        for e in &self.credit {
            v -= e.1;
        }
        v + self.loose_owed
    }

    // ------------------------------------------------------------------ particles
    /// Rigidly move particles that sit in the cup or touch the spoon with the spoon's motion.
    fn carry_particles(&mut self, p0: Vec3, q0: Quat, p1: Vec3, q1: Quat) {
        if p0.distance_to(p1) < 1e-9 && q0.angle_to(q1) < 1e-9 {
            return;
        }
        let f = &mut self.fluid;
        let qi = q0.invert();
        let qd = q1.mul(qi);
        for i in 0..f.n {
            let o = i * 3;
            let mut carry = f.contact[i] == CONTACT_SPOON;
            if !carry {
                let l = Vec3::new(f.pos[o] as f64 - p0.x, f.pos[o + 1] as f64 - p0.y, f.pos[o + 2] as f64 - p0.z).apply_quat(qi);
                carry = sd_hull_interior(l.x, l.y, l.z) < 0.0 || self.content.sample(l.x, l.y, l.z) < 2.0 * f.r;
            }
            if !carry {
                continue;
            }
            for a in [&mut f.pos, &mut f.prev] {
                let v = Vec3::new(a[o] as f64 - p0.x, a[o + 1] as f64 - p0.y, a[o + 2] as f64 - p0.z).apply_quat(qd).add(p1);
                a[o] = v.x as f32;
                a[o + 1] = v.y as f32;
                a[o + 2] = v.z as f32;
            }
        }
    }

    /// Remove particle i (the last one moves into its slot, flags included).
    fn remove_particle(&mut self, i: usize) {
        let last = self.fluid.n - 1;
        self.squeezed[i] = self.squeezed[last];
        self.squeezed[last] = 0;
        self.fluid.remove_at(i);
    }

    /// Claybook's DieWhenStops + StampSDFWhenDies: resting particles freeze into an SDF.
    fn freeze_resting(&mut self, allow_spoon: bool, dt: f64) {
        let fp = self.params.fluid.clone();
        let rs = fp.stamp_scale * self.fluid.r;
        let vp = self.particle_volume();
        let mut i = self.fluid.n;
        while i > 0 {
            i -= 1;
            let o = i * 3;
            let [x, y, z] = self.fluid.p(i);
            if y < -0.2 || x.abs() > 1.0 || z.abs() > 1.0 || !(x + y + z).is_finite() {
                self.remove_particle(i);
                continue;
            }
            let mut col = [0u8; 4];
            if self.squeezed[i] != 0 {
                self.squeezed[i] = 0;
                let gs = grid_at(&self.grids, x, y, z).unwrap_or(nearest_bowl(x).index);
                col.copy_from_slice(&self.fluid.rgba[i * 4..i * 4 + 4]);
                self.scratch_c = col;
                let gy = (y).max(self.grids[gs].oy + self.grids[gs].cell);
                self.owe(GridId::Bowl(gs), x, gy, z, vp, col);
                self.remove_particle(i);
                continue;
            }
            let f = &mut self.fluid;
            let ty = f.contact[i];
            if ty == CONTACT_NONE {
                f.rest[i] = 0;
                continue;
            }
            // net displacement over the whole frame, relative to the surface it rests on
            let vx = (x - f.frame_pos[o] as f64) / dt - f.sv[o] as f64;
            let vy = (y - f.frame_pos[o + 1] as f64) / dt - f.sv[o + 1] as f64;
            let vz = (z - f.frame_pos[o + 2] as f64) / dt - f.sv[o + 2] as f64;
            let speed = (vx * vx + vy * vy + vz * vz).sqrt();
            f.slow[i] = (f.slow[i] as f64 * 0.6 + speed * 0.4) as f32;
            f.rest[i] = if (f.slow[i] as f64) < fp.stop_speed { f.rest[i].wrapping_add(1) } else { 0 };
            let instant = fp.stamp_on_contact && ty == CONTACT_WORLD && f.age[i] > 3;
            let slow = f.slow[i] as f64;
            let rest = f.rest[i];
            let embedded = slow < 3.0 * fp.stop_speed && self.embedded(ty, x, y, z);
            if (rest as u32) < fp.rest_frames && !instant && !embedded {
                continue;
            }
            col.copy_from_slice(&self.fluid.rgba[i * 4..i * 4 + 4]);
            self.scratch_c = col;

            if ty == CONTACT_SPOON {
                if !allow_spoon {
                    continue;
                }
                let l = self.pose.to_local(x, y, z);
                let s = &SPOON;
                let under_cup = l[1] < 0.0 && sd_ellipsoid(l[0], l[1], l[2], s.a + s.t, s.b + s.t, s.c + s.t) > 0.0;
                if !under_cup && self.content.contains(l[0], l[1], l[2], rs + 3.0 * self.content.cell) {
                    let added = self.content.stamp_sphere(l[0], l[1], l[2], rs, fp.stamp_smooth, Some(col));
                    self.settle(GridId::Content, l[0], l[1], l[2], vp - added, col);
                    self.remove_particle(i);
                }
                continue;
            }
            let g = grid_at(&self.grids, x, y, z);
            match g {
                Some(gi) if self.grids[gi].contains(x, y, z, rs + self.grids[gi].cell) => {
                    let added = self.grids[gi].stamp_sphere(x, y, z, rs, fp.stamp_smooth, Some(col));
                    // merging into ice that is already there adds nothing: re-deposit the difference
                    self.settle(GridId::Bowl(gi), x, y, z, vp - added, col);
                    self.remove_particle(i);
                }
                _ => {
                    // on the table, outside every bowl volume: keep it as a static crumb
                    if self.n_spilled < MAX_SPILLED {
                        let s = self.n_spilled;
                        self.n_spilled += 1;
                        self.spilled[s * 3] = x as f32;
                        self.spilled[s * 3 + 1] = y as f32;
                        self.spilled[s * 3 + 2] = z as f32;
                        self.spilled_rgba[s * 4..s * 4 + 4].copy_from_slice(&col);
                        self.spilled_version += 1;
                    }
                    self.spilled_volume += self.particle_volume();
                    self.remove_particle(i);
                }
            }
        }
    }

    fn embedded(&self, ty: u8, x: f64, y: f64, z: f64) -> bool {
        let r = self.fluid.r;
        if ty == CONTACT_WORLD {
            return match grid_at(&self.grids, x, y, z) {
                Some(gi) => self.grids[gi].sample(x, y, z) < 0.5 * r,
                None => false,
            };
        }
        let l = self.pose.to_local(x, y, z);
        self.content.sample(l[0], l[1], l[2]) < 0.5 * r
    }

    pub fn particle_volume(&self) -> f64 {
        let r = if self.fluid.r > 0.0 { self.fluid.r } else { self.params.fluid.particle_radius };
        8.0 * r * r * r
    }

    /// Carve and copy at one spoon pose, as a frame does along the spoon's path (for comparisons:
    /// the GPU simulation's checks run the same sequence of poses through both).
    pub fn interact_at(&mut self, pos: Vec3, quat: Quat) {
        self.interact(pos, quat, 1.0 / 60.0, true);
    }

    /// Take the spoon's current pose as the one eat / release work at (between frames).
    pub fn sync_spoon(&mut self) {
        self.sync_view();
    }

    /// Tip everything on the spoon off as particles now (the "Tip off" button), at the spoon's pose.
    pub fn release(&mut self) {
        self.sync_view();
        self.release_content();
        self.update_stats(false);
    }

    /// Turn everything on the spoon into particles (tilt release). Lattice points hugging the cup are
    /// moved inside it first (see `into_cup`).
    fn release_content(&mut self) {
        if self.content_volume <= 0.0 {
            return;
        }
        let fp = self.params.fluid.clone();
        self.fluid.configure(fp.particle_radius, fp.kernel_scale);
        let step = self.fluid.d;
        let lump = self.fluid.new_lump();
        let (ox, oy, oz, sx, sy, sz) = (self.content.ox, self.content.oy, self.content.oz, self.content.sx, self.content.sy, self.content.sz);
        let mut lz = oz + step / 2.0;
        while lz < oz + sz {
            let mut ly = oy + step / 2.0;
            while ly < oy + sy {
                let mut lx = ox + step / 2.0;
                while lx < ox + sx {
                    if self.content.sample(lx, ly, lz) >= 0.0 {
                        lx += step;
                        continue;
                    }
                    if self.fluid.n >= fp.max_particles {
                        break;
                    }
                    let [px, py, pz] = into_cup(lx, ly, lz, fp.particle_radius);
                    let w = self.pose.to_world(px, py, pz);
                    let v = point_velocity(self.cur.pos, self.cur.lin, self.cur.ang, w[0], w[1], w[2]);
                    let mut col = [0u8; 4];
                    if let Some(vi) = self.content.voxel_at(lx, ly, lz) {
                        col.copy_from_slice(&self.content.color[vi * 4..vi * 4 + 4]);
                    }
                    self.scratch_c = col;
                    if let Some(i) = self.fluid.spawn(w[0], w[1], w[2], v[0], v[1], v[2], col) {
                        self.fluid.set_lump(i, lump);
                    }
                    lx += step;
                }
                ly += step;
            }
            lz += step;
        }
        self.content.clear();
    }

    /// "Eat": empty the spoon, including loose bits riding on it. Returns grams eaten.
    pub fn eat(&mut self) -> f64 {
        self.set_pose(self.cur.pos, self.cur.quat);
        let mut vol = self.content.volume();
        self.content.clear();
        let mut i = self.fluid.n;
        while i > 0 {
            i -= 1;
            let [x, y, z] = self.fluid.p(i);
            let l = self.pose.to_local(x, y, z);
            let on_spoon = self.fluid.contact[i] == CONTACT_SPOON
                || sd_hull_interior(l[0], l[1], l[2]) < self.fluid.r
                || (self.content.contains(l[0], l[1], l[2], 0.0) && sd_spoon_solid(l[0], l[1], l[2]) < 0.02);
            if on_spoon {
                vol += self.particle_volume();
                self.remove_particle(i);
            }
        }
        self.eaten_volume += vol;
        self.update_stats(false);
        vol * 1e6 * self.params.ice.density
    }

    /// Set the spoon's target pose (Y-up world frame).
    pub fn set_target(&mut self, pos: Option<Vec3>, quat: Option<Quat>) {
        self.spoon.set_target(pos, quat);
    }

    // ------------------------------------------------------------------ stats
    pub fn update_stats(&mut self, full: bool) {
        self.content_volume = self.content.volume();
        if full {
            for i in 0..3 {
                self.bowl_volumes[i] = self.grids[i].volume();
            }
        }
    }

    pub fn grams(&self, vol: f64) -> f64 {
        vol * 1e6 * self.params.ice.density
    }

    /// Total ice in the system (g): bowls + spoon + particles + eaten + spilled + owed.
    pub fn total_grams(&self) -> f64 {
        let bowls: f64 = self.grids.iter().map(|g| g.volume()).sum();
        self.grams(
            bowls + self.content.volume() + self.fluid.n as f64 * self.particle_volume() + self.eaten_volume + self.spilled_volume + self.debt_volume(),
        )
    }

    pub fn state(&self) -> State {
        let sp = &self.spoon;
        let e = sp.quat.to_euler_xyz();
        State {
            time: self.time,
            spoon_position: sp.pos.to_array(),
            spoon_quaternion: sp.quat.to_array(),
            spoon_euler_xyz_deg: [rad_to_deg(e[0]), rad_to_deg(e[1]), rad_to_deg(e[2])],
            linear_velocity: sp.lin_vel.to_array(),
            angular_velocity: sp.ang_vel.to_array(),
            tilt_deg: self.tilt_deg(),
            target_position: sp.target_pos.to_array(),
            target_quaternion: sp.target_quat.to_array(),
            contact: sp.contacts > 0,
            contact_samples: sp.contacts,
            blocked: sp.blocked,
            speed_scale: self.speed_scale,
            cut_force_n: self.cut_force,
            scooped_grams: self.grams(self.content_volume),
            eaten_grams: self.grams(self.eaten_volume),
            spilled_grams: self.grams(self.spilled_volume),
            bowl_grams: [self.grams(self.bowl_volumes[0]), self.grams(self.bowl_volumes[1]), self.grams(self.bowl_volumes[2])],
            particles: self.fluid.n,
        }
    }
}

/// Particle collision environment (table, bowls, solid ice, spoon metal, spoon contents).
struct Env<'a> {
    grids: &'a [SdfGrid],
    content: &'a SdfGrid,
    pose: Pose,
    sp: SpoonView,
    squeezed: &'a mut [u8],
}

impl Collide for Env<'_> {
    fn collide(&mut self, f: &mut Fluid, i: usize) {
        let o = i * 3;
        let r = f.r;
        let [mut x, mut y, mut z] = f.p(i);
        let mut best = 0.0f64;
        let mut ty = CONTACT_NONE;
        let (mut nx, mut ny, mut nz) = (0.0, 1.0, 0.0);

        let mut d = y - r;
        if d < 0.0 {
            y -= d;
            best = -d;
            ty = CONTACT_WORLD;
        }
        let b = nearest_bowl(x);
        d = sd_bowl(b, x, y, z) - r;
        if d < 0.0 {
            let e = 1e-5;
            let mut gx = sd_bowl(b, x + e, y, z) - sd_bowl(b, x - e, y, z);
            let mut gy = sd_bowl(b, x, y + e, z) - sd_bowl(b, x, y - e, z);
            let mut gz = sd_bowl(b, x, y, z + e) - sd_bowl(b, x, y, z - e);
            let mut l = hypot3(gx, gy, gz);
            if l == 0.0 {
                l = 1.0;
            }
            gx /= l;
            gy /= l;
            gz /= l;
            x -= d * gx;
            y -= d * gy;
            z -= d * gz;
            if -d >= best {
                best = -d;
                ty = CONTACT_WORLD;
                nx = gx;
                ny = gy;
                nz = gz;
            }
        }
        let g = &self.grids[b.index];
        d = g.sample(x, y, z) - r;
        if d < 0.0 {
            let n = g.grad(x, y, z);
            x -= d * n[0];
            y -= d * n[1];
            z -= d * n[2];
            if -d >= best {
                best = -d;
                ty = CONTACT_WORLD;
                nx = n[0];
                ny = n[1];
                nz = n[2];
            }
        }

        let (x0, y0, z0) = (x, y, z);
        let t = self.pose.t;
        let (mut rx, mut ry, mut rz) = (x - t[0], y - t[1], z - t[2]);
        if rx * rx + ry * ry + rz * rz < 0.04 {
            let rm = &self.pose.r;
            for pass in 0..2 {
                let lx = rm[0] * rx + rm[3] * ry + rm[6] * rz;
                let ly = rm[1] * rx + rm[4] * ry + rm[7] * rz;
                let lz = rm[2] * rx + rm[5] * ry + rm[8] * rz;
                let (dl, n) = if pass == 0 {
                    let dl = sd_spoon_solid(lx, ly, lz) - r;
                    (dl, if dl < 0.0 { grad_local(sd_spoon_solid, lx, ly, lz) } else { [0.0; 3] })
                } else {
                    let dl = self.content.sample(lx, ly, lz) - r;
                    (dl, if dl < 0.0 { self.content.grad(lx, ly, lz) } else { [0.0; 3] })
                };
                if dl >= 0.0 {
                    continue;
                }
                let wn = self.pose.rot(n);
                x -= dl * wn[0];
                y -= dl * wn[1];
                z -= dl * wn[2];
                rx = x - t[0];
                ry = y - t[1];
                rz = z - t[2];
                if -dl >= best {
                    best = -dl;
                    ty = CONTACT_SPOON;
                    nx = wn[0];
                    ny = wn[1];
                    nz = wn[2];
                }
            }
        }

        // squeezed between the spoon and the bowl/table: keep it on the bowl's side and merge it
        // into the ice there (freeze_resting)
        if ty == CONTACT_SPOON && sd_env(x, y, z) < 0.5 * r {
            x = x0;
            y = y0;
            z = z0;
            self.squeezed[i] = 1;
        }
        f.pos[o] = x as f32;
        f.pos[o + 1] = y as f32;
        f.pos[o + 2] = z as f32;
        if ty != CONTACT_NONE {
            f.contact[i] = ty;
            f.cn[o] = nx as f32;
            f.cn[o + 1] = ny as f32;
            f.cn[o + 2] = nz as f32;
            if ty == CONTACT_SPOON {
                let v = point_velocity(self.sp.pos, self.sp.lin, self.sp.ang, x, y, z);
                f.sv[o] = v[0] as f32;
                f.sv[o + 1] = v[1] as f32;
                f.sv[o + 2] = v[2] as f32;
            } else {
                f.sv[o] = 0.0;
                f.sv[o + 1] = 0.0;
                f.sv[o + 2] = 0.0;
            }
        }
    }
}

/// A particle spawned from the scoop where it touches the cup (spoon-local point, within a particle
/// radius of the metal, below the rim) goes to the inside of the cup, a radius clear of the wall.
/// Otherwise its push-out can pick the cup's outside (the wall is thinner than a particle) and it falls
/// through the spoon.
pub fn into_cup(x: f64, y: f64, z: f64, r: f64) -> [f64; 3] {
    if y >= 0.0 || sd_spoon_solid(x, y, z) >= r {
        return [x, y, z];
    }
    let d = sd_hull_interior(x, y, z);
    if d <= -r {
        return [x, y, z];
    }
    let g = grad_local(sd_hull_interior, x, y, z);
    let k = d + r;
    [x - k * g[0], y - k * g[1], z - k * g[2]]
}

/// The lowest distance a bowl voxel may keep: what carving cuts away (spoon metal, the cup interior unless
/// tilted, the ice on the spoon), as `-distance` to that. Smoothing a bowl must not refill it, or the next
/// carve cuts that ice again and a spoon held still keeps pumping ice into the cup. Voxel `idx` of a grid
/// with origin + cell `geo` and `n` = (nx, ny).
pub fn spoon_keep_out<'a>(pose: Pose, content: &'a SdfGrid, tilted: bool, geo: [f64; 4], n: [usize; 2], band: f64) -> impl Fn(usize) -> f64 + 'a {
    let [ox, oy, oz, c] = geo;
    let [nx, ny] = n;
    let pad = 0.4 * c;
    let cg = content;
    let (lo, hi) = keep_out_box([cg.ox, cg.oy, cg.oz], [cg.nx as f64 * cg.cell, cg.ny as f64 * cg.cell, cg.nz as f64 * cg.cell], band);
    move |idx: usize| {
        let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
        let (x, y, z) = (ox + (i as f64 + 0.5) * c, oy + (j as f64 + 0.5) * c, oz + (k as f64 + 0.5) * c);
        let [lx, ly, lz] = pose.to_local(x, y, z);
        // further than a band from all of it: the floor would sit below -band, where values are clamped anyway
        if lx < lo[0] || ly < lo[1] || lz < lo[2] || lx > hi[0] || ly > hi[1] || lz > hi[2] {
            return f64::NEG_INFINITY;
        }
        let mut d = (sd_spoon_shell(lx, ly, lz) - pad).min(sd_handle(lx, ly, lz) - pad);
        let cs = content.sample(lx, ly, lz);
        if cs < content.band * 0.999 {
            // (at the band: outside the scoop's grid, or nothing near; not a distance)
            d = d.min(cs);
        }
        if !tilted {
            d = d.min(sd_hull_interior(lx, ly, lz));
        }
        -d
    }
}
/// Spoon-local box around the spoon and the scoop's grid (origin `co`, size `cs`), `pad` wider on every side.
pub fn keep_out_box(co: [f64; 3], cs: [f64; 3], pad: f64) -> ([f64; 3], [f64; 3]) {
    let s = &SPOON;
    let r = s.handle_r;
    let mut lo = [-s.a - s.t, -s.b - s.t, -s.c - s.t];
    let mut hi = [s.a + s.t, s.t, s.c + s.t];
    for p in &s.handle {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k] - r);
            hi[k] = hi[k].max(p[k] + r);
        }
    }
    for k in 0..3 {
        lo[k] = lo[k].min(co[k]) - pad;
        hi[k] = hi[k].max(co[k] + cs[k]) + pad;
    }
    (lo, hi)
}
