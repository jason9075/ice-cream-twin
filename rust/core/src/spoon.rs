//! The spoon controller: follows a target pose like a speed-limited robot tool, but can never
//! penetrate the bowls or the table. Ported from the former JS `spoon.js`.
//!
//!  1. Each frame the step toward the target is clamped to max linear / angular speed.
//!  2. The step is split into sub-steps so no collision sample moves more than `SUBSTEP_MAX`
//!     (well under half the 6 mm bowl wall) — this prevents tunnelling.
//!  3. After every sub-step the pose is projected out of the environment with position-based
//!     rigid contact corrections (translation + rotation), so the spoon slides along walls.
//!  4. A sub-step that still penetrates more than `PEN_TOL` is rejected (`blocked`).

use crate::math::{deg_to_rad, hypot3, Quat, Vec3};
use crate::params::SpoonParams;
use crate::sdf::{grad_env, sd_env, SPOON};
use std::f64::consts::PI;

/// m of sample travel per sub-step (bowl wall is 6 mm thick)
pub const SUBSTEP_MAX: f64 = 0.0008;
/// m of residual penetration allowed after projection
pub const PEN_TOL: f64 = 0.0002;
/// samples are slightly fatter than the mesh
pub const SAMPLE_MARGIN: f64 = 0.0002;
const CONTACT_DIST: f64 = 0.0004;
const MASS: f64 = 1.0;
/// m — weights rotation vs translation in contact corrections
const GYRATION: f64 = 0.05;
const INERTIA: f64 = MASS * GYRATION * GYRATION;

pub const HOME_POS: Vec3 = Vec3::new(0.0, 0.17, 0.02);
pub const HOME_QUAT: Quat = Quat::IDENTITY;

/// Collision samples (x, y, z, radius) in the spoon frame. Stored as f32 like the JS
/// `Float32Array`, so both versions use bit-identical sample positions.
pub fn build_spoon_samples(spacing: f64) -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    let (a, b, c, t) = (SPOON.a, SPOON.b, SPOON.c, SPOON.t);
    let rr = t / 2.0 + SAMPLE_MARGIN;
    let avg = (a + c) / 2.0;
    let n_th = ((PI / 2.0) * (a + b) / 2.0 / spacing).ceil() as usize;
    for i in 0..=n_th {
        let th = (i as f64 / n_th as f64) * (PI / 2.0);
        let s = crate::math::sin(th);
        let co = crate::math::cos(th);
        let n_ph = ((2.0 * PI * s * avg) / spacing).ceil().max(1.0) as usize;
        for j in 0..n_ph {
            let ph = (j as f64 / n_ph as f64) * 2.0 * PI;
            out.push([
                (a * s * crate::math::cos(ph)) as f32,
                (-b * co) as f32,
                (c * s * crate::math::sin(ph)) as f32,
                rr as f32,
            ]);
        }
    }
    let h = &SPOON.handle;
    let hr = SPOON.handle_r + SAMPLE_MARGIN;
    for i in 0..h.len() - 1 {
        let p = h[i];
        let q = h[i + 1];
        let len = hypot3(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
        let n = (len / spacing).ceil() as usize;
        for k in 0..n {
            let u = k as f64 / n as f64;
            out.push([
                (p[0] + (q[0] - p[0]) * u) as f32,
                (p[1] + (q[1] - p[1]) * u) as f32,
                (p[2] + (q[2] - p[2]) * u) as f32,
                hr as f32,
            ]);
        }
    }
    let e = h[h.len() - 1];
    out.push([e[0] as f32, e[1] as f32, e[2] as f32, hr as f32]);
    out
}

/// q <- normalize(q + 0.5 * [dθ, 0] ⊗ q): apply a small world-space rotation vector.
pub fn apply_rot_vec(q: Quat, x: f64, y: f64, z: f64) -> Quat {
    let t = Quat::new(x, y, z, 0.0).mul(q);
    Quat::new(q.x + 0.5 * t.x, q.y + 0.5 * t.y, q.z + 0.5 * t.z, q.w + 0.5 * t.w).normalize()
}

/// Rotation vector (axis * angle, world frame) taking quaternion a to b, shortest way.
pub fn rot_vec_between(a: Quat, b: Quat) -> Vec3 {
    let mut qd = b.mul(a.invert());
    if qd.w < 0.0 {
        qd = Quat::new(-qd.x, -qd.y, -qd.z, -qd.w);
    }
    let w = 1.0f64.min((-1.0f64).max(qd.w));
    let angle = 2.0 * crate::math::acos(w);
    let s = (1.0 - w * w).sqrt();
    if s < 1e-9 {
        return Vec3::ZERO;
    }
    Vec3::new(qd.x / s, qd.y / s, qd.z / s).scale(angle)
}

/// One accepted sub-step: the pose the spoon moved to and the sub-step duration.
#[derive(Clone, Copy, Debug)]
pub struct Substep {
    pub pos: Vec3,
    pub quat: Quat,
    pub dt: f64,
}

#[derive(Clone, Debug)]
pub struct SpoonController {
    pub p: SpoonParams,
    pub samples: Vec<[f32; 4]>,
    pub sample_contact: Vec<u8>,
    pub max_reach: f64,
    pub pos: Vec3,
    pub quat: Quat,
    pub target_pos: Vec3,
    pub target_quat: Quat,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3,
    pub blocked: bool,
    pub contacts: usize,
    pub correction: f64,
    pub substeps: usize,
}

impl SpoonController {
    pub fn new(p: SpoonParams) -> Self {
        let samples = build_spoon_samples(0.0022);
        let mut reach = 0.0f64;
        for s in &samples {
            reach = reach.max(hypot3(s[0] as f64, s[1] as f64, s[2] as f64) + s[3] as f64);
        }
        let n = samples.len();
        let mut c = SpoonController {
            p,
            samples,
            sample_contact: vec![0; n],
            max_reach: reach,
            pos: Vec3::ZERO,
            quat: Quat::IDENTITY,
            target_pos: Vec3::ZERO,
            target_quat: Quat::IDENTITY,
            lin_vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            blocked: false,
            contacts: 0,
            correction: 0.0,
            substeps: 0,
        };
        c.reset();
        c
    }

    pub fn reset(&mut self) {
        self.pos = HOME_POS;
        self.quat = HOME_QUAT;
        self.target_pos = HOME_POS;
        self.target_quat = HOME_QUAT;
        self.lin_vel = Vec3::ZERO;
        self.ang_vel = Vec3::ZERO;
        self.blocked = false;
        self.contacts = 0;
        self.correction = 0.0;
        self.substeps = 0;
    }

    /// Set the target pose (either part optional). Position is clamped to the work box.
    pub fn set_target(&mut self, pos: Option<Vec3>, quat: Option<Quat>) {
        if let Some(p) = pos {
            self.target_pos = Vec3::new(p.x.min(0.45).max(-0.45), p.y.min(0.45).max(0.0), p.z.min(0.45).max(-0.45));
        }
        if let Some(q) = quat {
            self.target_quat = q.normalize();
        }
    }

    #[inline]
    fn sample_world(&self, i: usize, pos: Vec3, quat: Quat) -> Vec3 {
        let s = &self.samples[i];
        Vec3::new(s[0] as f64, s[1] as f64, s[2] as f64).apply_quat(quat).add(pos)
    }

    /// Largest penetration depth (m) of any collision sample at a pose; <= 0 means free.
    pub fn penetration(&self, pos: Vec3, quat: Quat) -> f64 {
        let mut max_pen = f64::NEG_INFINITY;
        for i in 0..self.samples.len() {
            let w = self.sample_world(i, pos, quat);
            let pen = self.samples[i][3] as f64 - sd_env(w.x, w.y, w.z);
            if pen > max_pen {
                max_pen = pen;
            }
        }
        max_pen
    }

    /// Position-based rigid contact projection (Gauss-Seidel over samples).
    pub fn project(&self, mut pos: Vec3, mut quat: Quat, iterations: usize) -> (Vec3, Quat) {
        for _ in 0..iterations {
            let mut max_pen = 0.0f64;
            for i in 0..self.samples.len() {
                let w = self.sample_world(i, pos, quat);
                let d = sd_env(w.x, w.y, w.z) - self.samples[i][3] as f64;
                if d >= 0.0 {
                    continue;
                }
                if -d > max_pen {
                    max_pen = -d;
                }
                let n = grad_env(w.x, w.y, w.z);
                let arm = w.sub(pos);
                let rxn = Vec3::cross(arm, Vec3::new(n[0], n[1], n[2]));
                let wg = 1.0 / MASS + rxn.length_sq() / INERTIA;
                let lam = -d / wg;
                pos.x += (n[0] * lam) / MASS;
                pos.y += (n[1] * lam) / MASS;
                pos.z += (n[2] * lam) / MASS;
                let k = lam / INERTIA;
                quat = apply_rot_vec(quat, rxn.x * k, rxn.y * k, rxn.z * k);
            }
            if max_pen < 1e-6 {
                break;
            }
        }
        (pos, quat)
    }

    pub fn count_contacts(&mut self) -> usize {
        let mut n = 0;
        for i in 0..self.samples.len() {
            let w = self.sample_world(i, self.pos, self.quat);
            let d = sd_env(w.x, w.y, w.z) - self.samples[i][3] as f64;
            let c = (d < CONTACT_DIST) as u8;
            self.sample_contact[i] = c;
            n += c as usize;
        }
        n
    }

    /// Advance toward the target. Every accepted sub-step is appended to `out` (if given), so
    /// the caller can carve ice cream along the swept path.
    pub fn update(&mut self, dt: f64, speed_scale: f64, mut out: Option<&mut Vec<Substep>>) {
        let p = &self.p;
        let start_pos = self.pos;
        let start_quat = self.quat;

        let mut dp = self.target_pos.sub(self.pos);
        let max_l = p.max_lin_speed * speed_scale * dt;
        if dp.length() > max_l {
            dp = dp.set_length(max_l);
        }

        let mut axis = rot_vec_between(self.quat, self.target_quat);
        let mut ang = axis.length();
        let max_a = deg_to_rad(p.max_ang_speed) * speed_scale * dt;
        if ang > max_a {
            axis = axis.scale(max_a / ang);
            ang = max_a;
        }

        let disp = dp.length() + ang * self.max_reach;
        let n = ((disp / SUBSTEP_MAX).ceil().max(1.0)).min(64.0) as usize;
        let sub_dt = dt / n as f64;
        let qs = if ang > 1e-12 { Quat::from_axis_angle(axis.normalize(), ang / n as f64) } else { Quat::IDENTITY };

        self.blocked = false;
        self.correction = 0.0;
        self.substeps = 0;
        for _ in 0..n {
            let cp0 = self.pos.add_scaled(dp, 1.0 / n as f64);
            let cq0 = qs.mul(self.quat).normalize();
            let (cp, cq) = self.project(cp0, cq0, 12);
            self.correction += cp0.distance_to(cp);
            if self.penetration(cp, cq) > PEN_TOL {
                self.blocked = true;
                break;
            }
            self.pos = cp;
            self.quat = cq;
            self.substeps += 1;
            if let Some(o) = out.as_deref_mut() {
                o.push(Substep { pos: cp, quat: cq, dt: sub_dt });
            }
        }

        self.lin_vel = self.pos.sub(start_pos).div(dt);
        self.ang_vel = rot_vec_between(start_quat, self.quat).div(dt);
        self.contacts = self.count_contacts();
    }

    /// World-space velocity of a material point of the spoon.
    pub fn point_velocity(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        point_velocity(self.pos, self.lin_vel, self.ang_vel, x, y, z)
    }
}

/// Velocity of a point rigidly attached to a body at `pos` moving with (lin, ang).
pub fn point_velocity(pos: Vec3, lin: Vec3, w: Vec3, x: f64, y: f64, z: f64) -> [f64; 3] {
    let rx = x - pos.x;
    let ry = y - pos.y;
    let rz = z - pos.z;
    [lin.x + (w.y * rz - w.z * ry), lin.y + (w.z * rx - w.x * rz), lin.z + (w.x * ry - w.y * rx)]
}
