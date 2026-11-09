//! The SO-101 holding the spoon: base placement, inverse kinematics, joint-space leading.
//! Ported operation for operation from the former JS `arm.js` (see `tests/parity.rs`).
//!
//! The spoon controller stays in charge of contact and the no-penetration guarantee. The arm wraps it:
//! 1. [`Arm::constrain`] (before each step): a new spoon target becomes the arm's goal, the closest
//!    pose the arm can reach (IK, then FK), and the spoon is led there in joint space;
//! 2. [`Arm::follow`] (after each step): the arm is solved for the spoon's actual pose, which gives
//!    the joint angles that are reported.

use super::so101::{forward, limits, m4, M4, ARM_JOINTS, GRIPPER_LINK};
use super::tool::{tool, Tool, DEFAULT_TOOL};
use crate::math::{acos, atan2, hypot3, Quat, Vec3};
use crate::spoon::SpoonController;
use std::f64::consts::PI;

/// Where the robot stands: a position in the scene (Y-up, metres) and a yaw about the scene's Y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Base {
    pub position: [f64; 3],
    pub yaw_deg: f64,
}

/// Behind the middle bowl, facing +Z (toward the camera).
pub const DEFAULT_BASE: Base = Base { position: [0.0, 0.0, -0.28], yaw_deg: -90.0 };

/// Spoon pose the arm goes home to: above the middle bowl, handle toward the robot, cup 10° nose-down.
pub const ARM_HOME_POS: Vec3 = Vec3::new(0.0, 0.15, 0.0);
pub fn arm_home_quat() -> Quat {
    Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI / 2.0).mul(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), (10.0 * PI) / 180.0))
}

/// Rotation error of 1 rad weighs like this many metres of position error in the IK.
const ROT_SCALE: f64 = 0.04;
const LAMBDA: f64 = 0.004;
const SEED: [f64; 5] = [0.0, -0.6, 0.9, 0.9, 0.0];
const SEEDS: [[f64; 5]; 5] =
    [SEED, [0.0, 0.3, 0.3, 1.2, 0.0], [0.0, -1.2, 1.5, 0.3, 0.0], [0.0, 0.8, -0.5, 1.5, 0.0], [0.0, -0.3, 0.2, 1.4, 0.0]];
/// The joint command waits while the spoon is this far behind it.
const LEAD_LIN: f64 = 0.012;
fn lead_ang() -> f64 {
    (20.0 * PI) / 180.0
}

/// Remaining pose error of an IK solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PoseErr {
    pub mm: f64,
    pub deg: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpoonPose {
    pub pos: Vec3,
    pub quat: Quat,
}

/// world <- robot: the URDF is Z-up, the scene is Y-up; then a yaw about the scene's Y.
fn base_matrix(b: &Base) -> M4 {
    let yaw = (b.yaw_deg * PI) / 180.0;
    let z_up_to_y_up = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut m = m4::mul(&m4::axis_angle([0.0, 1.0, 0.0], yaw), &z_up_to_y_up);
    m[3] = b.position[0];
    m[7] = b.position[1];
    m[11] = b.position[2];
    m
}

fn pose_matrix(pos: Vec3, q: Quat) -> M4 {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        pos.x,
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        pos.y,
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
        pos.z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

fn matrix_to_quat(m: &M4) -> Quat {
    let (m11, m12, m13, m21, m22, m23, m31, m32, m33) = (m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
    let tr = m11 + m22 + m33;
    let q = if tr > 0.0 {
        let s = 0.5 / (tr + 1.0).sqrt();
        Quat::new((m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s)
    } else if m11 > m22 && m11 > m33 {
        let s = 2.0 * (1.0 + m11 - m22 - m33).sqrt();
        Quat::new(0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s)
    } else if m22 > m33 {
        let s = 2.0 * (1.0 + m22 - m11 - m33).sqrt();
        Quat::new((m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s)
    } else {
        let s = 2.0 * (1.0 + m33 - m11 - m22).sqrt();
        Quat::new((m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s)
    };
    q.normalize()
}

/// Rotation vector of Ra * Rb^T (3x3 blocks of 4x4s).
fn rot_error(ra: &M4, rb: &M4) -> [f64; 3] {
    let mut e = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            e[r * 3 + c] = ra[r * 4] * rb[c * 4] + ra[r * 4 + 1] * rb[c * 4 + 1] + ra[r * 4 + 2] * rb[c * 4 + 2];
        }
    }
    let cos = 1f64.min((-1f64).max((e[0] + e[4] + e[8] - 1.0) / 2.0));
    let ang = acos(cos);
    let v = [e[7] - e[5], e[2] - e[6], e[3] - e[1]];
    let s = hypot3(v[0], v[1], v[2]);
    if s < 1e-12 {
        if ang < 1e-6 {
            return [0.0; 3];
        }
        // 180°: axis from the largest diagonal entry
        let i = if e[0] > e[4] && e[0] > e[8] {
            0
        } else if e[4] > e[8] {
            1
        } else {
            2
        };
        let mut a = [0.0; 3];
        a[i] = 0f64.max((e[i * 4] + 1.0) / 2.0).sqrt();
        for k in 0..3 {
            if k != i {
                a[k] = (e[i * 3 + k] + e[k * 3 + i]) / (4.0 * a[i]);
            }
        }
        return a.map(|x| x * ang);
    }
    let k = ang / s;
    [v[0] * k, v[1] * k, v[2] * k]
}

/// Solve the 5x5 system A x = b in place (Gaussian elimination with partial pivoting).
fn solve_lin(a: &mut [f64; 25], b: &mut [f64; 5]) {
    const N: usize = 5;
    for c in 0..N {
        let mut p = c;
        for r in c + 1..N {
            if a[r * N + c].abs() > a[p * N + c].abs() {
                p = r;
            }
        }
        if p != c {
            for k in 0..N {
                a.swap(c * N + k, p * N + k);
            }
            b.swap(c, p);
        }
        let d = a[c * N + c];
        for r in c + 1..N {
            let f = a[r * N + c] / d;
            for k in c..N {
                a[r * N + k] -= f * a[c * N + k];
            }
            b[r] -= f * b[c];
        }
    }
    for r in (0..N).rev() {
        let mut s = b[r];
        for k in r + 1..N {
            s -= a[r * N + k] * b[k];
        }
        b[r] = s / a[r * N + r];
    }
}

fn clamp_q(q: &mut [f64; 5], lim: &[[f64; 2]; 5]) {
    for k in 0..5 {
        q[k] = lim[k][1].min(lim[k][0].max(q[k]));
    }
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub base: Base,
    b: M4,
    b_inv: M4,
    pub tool: Tool,
    pub jaw: f64,
    lim: [[f64; 2]; 5],
    /// Solved from the spoon's actual pose (what is drawn and reported).
    pub q: [f64; 5],
    /// Where the arm is heading.
    pub q_goal: [f64; 5],
    /// The joint-space command this frame.
    pub q_cmd: [f64; 5],
    /// How well `q` reproduces the actual spoon pose.
    pub err: PoseErr,
    /// How far the requested target was from reachable.
    pub goal_err: PoseErr,
    pub enabled: bool,
    pub constrain_target: bool,
    /// deg/s, fastest joint.
    pub joint_speed: f64,
    pub home: SpoonPose,
    pub goal: SpoonPose,
    last_cmd: SpoonPose,
}

impl Default for Arm {
    fn default() -> Self {
        Arm::new(DEFAULT_BASE, DEFAULT_TOOL)
    }
}

impl Arm {
    pub fn new(base: Base, tool_name: &str) -> Self {
        let t = tool(tool_name).unwrap_or_else(|| panic!("unknown tool module: {tool_name}"));
        let home = SpoonPose { pos: ARM_HOME_POS, quat: arm_home_quat() };
        let b = base_matrix(&base);
        let mut arm = Arm {
            base,
            b,
            b_inv: m4::invert_rigid(&b),
            jaw: t.jaw.unwrap_or(0.0),
            tool: t,
            lim: limits(),
            q: SEED,
            q_goal: SEED,
            q_cmd: SEED,
            err: PoseErr::default(),
            goal_err: PoseErr::default(),
            enabled: true,
            constrain_target: true,
            joint_speed: 90.0,
            home,
            goal: home,
            last_cmd: home,
        };
        arm.reset(None);
        arm
    }

    pub fn set_base(&mut self, base: Base) {
        self.base = base;
        self.b = base_matrix(&base);
        self.b_inv = m4::invert_rigid(&self.b);
    }

    pub fn set_tool(&mut self, name: &str) -> Result<(), String> {
        let t = tool(name).ok_or_else(|| format!("unknown tool module: {name}"))?;
        self.jaw = t.jaw.unwrap_or(0.0);
        self.tool = t;
        Ok(())
    }

    pub fn limits(&self) -> [[f64; 2]; 5] {
        self.lim
    }

    /// World pose of the spoon (rim centre) for joint angles q.
    pub fn spoon_pose(&self, q: &[f64; 5]) -> SpoonPose {
        let fk = forward(q, self.jaw);
        let w = m4::mul(&self.b, &m4::mul(&fk.t[GRIPPER_LINK], &self.tool.spoon_in_gripper));
        SpoonPose { pos: Vec3::new(w[3], w[7], w[11]), quat: matrix_to_quat(&w) }
    }

    /// Damped least squares on the 6-D spoon pose error (position + ROT_SCALE * rotation),
    /// warm-started from q (modified in place). Returns the remaining error.
    pub fn solve(&self, q: &mut [f64; 5], pos: Vec3, quat: Quat, iters: usize) -> PoseErr {
        const N: usize = 5;
        let d = m4::mul(&self.b_inv, &pose_matrix(pos, quat)); // desired spoon pose, robot frame
        let lim = &self.lim;
        let mut jac = [0.0; 6 * N];
        let mut a = [0.0; N * N];
        let mut g = [0.0; N];
        let mut ep = 0.0;
        let mut er = 0.0;
        for it in 0..=iters {
            let fk = forward(q, self.jaw);
            let s = m4::mul(&fk.t[GRIPPER_LINK], &self.tool.spoon_in_gripper);
            let p = [s[3], s[7], s[11]];
            let re = rot_error(&d, &s);
            let mut e = [d[3] - p[0], d[7] - p[1], d[11] - p[2], re[0], re[1], re[2]];
            ep = hypot3(e[0], e[1], e[2]);
            er = hypot3(e[3], e[4], e[5]);
            if it == iters || (ep < 1e-7 && er < 1e-6) {
                break;
            }
            for k in 0..3 {
                e[3 + k] *= ROT_SCALE;
            }
            for j in 0..N {
                let ax = fk.axes[j];
                let o = fk.origins[j];
                let dd = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
                jac[j] = ax[1] * dd[2] - ax[2] * dd[1];
                jac[N + j] = ax[2] * dd[0] - ax[0] * dd[2];
                jac[2 * N + j] = ax[0] * dd[1] - ax[1] * dd[0];
                jac[3 * N + j] = ax[0] * ROT_SCALE;
                jac[4 * N + j] = ax[1] * ROT_SCALE;
                jac[5 * N + j] = ax[2] * ROT_SCALE;
            }
            // (J^T J + λ² I) dq = J^T e, with joints pinned at a limit (and pushing past it) taken out
            let mut locked = [false; N];
            for _pass in 0..3 {
                for r in 0..N {
                    let mut sum = 0.0;
                    for k in 0..6 {
                        sum += if locked[r] { 0.0 } else { jac[k * N + r] * e[k] };
                    }
                    g[r] = sum;
                    for c in 0..N {
                        let mut t = 0.0;
                        if !locked[r] && !locked[c] {
                            for k in 0..6 {
                                t += jac[k * N + r] * jac[k * N + c];
                            }
                        }
                        a[r * N + c] = t + if r == c { if locked[r] { 1.0 } else { LAMBDA * LAMBDA } } else { 0.0 };
                    }
                }
                solve_lin(&mut a, &mut g);
                let mut again = false;
                for j in 0..N {
                    if locked[j] {
                        continue;
                    }
                    if (q[j] <= lim[j][0] + 1e-9 && g[j] < 0.0) || (q[j] >= lim[j][1] - 1e-9 && g[j] > 0.0) {
                        locked[j] = true;
                        again = true;
                    }
                }
                if !again {
                    break;
                }
            }
            let dq = g;
            let mut big = 0f64;
            for j in 0..N {
                big = big.max(dq[j].abs());
            }
            let sc = if big > 0.25 { 0.25 / big } else { 1.0 };
            let mut moved = 0f64;
            for j in 0..N {
                let before = q[j];
                q[j] = lim[j][1].min(lim[j][0].max(q[j] + dq[j] * sc));
                moved = moved.max((q[j] - before).abs());
            }
            if moved < 1e-9 {
                break;
            }
        }
        PoseErr { mm: ep * 1000.0, deg: (er * 180.0) / PI }
    }

    /// Put the arm (and the spoon, and its target) at the home pose.
    pub fn reset(&mut self, spoon: Option<&mut SpoonController>) {
        let mut q = SEED;
        self.solve(&mut q, self.home.pos, self.home.quat, 200);
        self.q = q;
        self.q_goal = q;
        self.q_cmd = q;
        let p = self.spoon_pose(&self.q);
        self.goal = p;
        self.last_cmd = p;
        self.err = PoseErr::default();
        if let Some(sp) = spoon {
            sp.pos = p.pos;
            sp.quat = p.quat;
            sp.set_target(Some(p.pos), Some(p.quat));
            sp.lin_vel = Vec3::ZERO;
            sp.ang_vel = Vec3::ZERO;
        }
    }

    /// Set the goal from a pose: the reachable pose closest to it (IK, then FK).
    fn set_goal(&mut self, pos: Vec3, quat: Quat) {
        // seeds: the last goal (smooth while jogging), the current command (big jumps), then a few
        // canonical postures if neither fits. Keep the best fit; on a tie, the one closest to the arm.
        fn fit(e: &PoseErr) -> f64 {
            e.mm + e.deg * 0.5
        }
        let try_from = |arm: &Arm, best: &mut Option<([f64; 5], PoseErr)>, seed: [f64; 5], iters: usize| {
            let dist = |q: &[f64; 5]| {
                let mut s = 0.0;
                for k in 0..5 {
                    s += (q[k] - arm.q_cmd[k]).abs();
                }
                s
            };
            let mut q = seed;
            let e = arm.solve(&mut q, pos, quat, iters);
            let better = match best {
                None => true,
                Some((bq, be)) => fit(&e) < fit(be) - 0.05 || ((fit(&e) - fit(be)).abs() <= 0.05 && dist(&q) < dist(bq)),
            };
            if better {
                *best = Some((q, e));
            }
        };
        let mut best = None;
        try_from(self, &mut best, self.q_goal, 60);
        try_from(self, &mut best, self.q_cmd, 150);
        if fit(&best.unwrap().1) > 0.2 {
            let pan = atan2(-(pos.x - self.b[3]), pos.z - self.b[11]); // rough heading toward the goal
            for s in SEEDS {
                try_from(self, &mut best, [pan, s[1], s[2], s[3], s[4]], 150);
            }
        }
        let (q, e) = best.unwrap();
        self.q_goal = q;
        self.goal_err = e;
        self.goal = self.spoon_pose(&self.q_goal);
    }

    /// Before the sim step. A new spoon target becomes the arm's goal, projected onto what the arm
    /// can reach. The spoon is then led there in joint space: the command moves toward the goal at
    /// `joint_speed` and waits whenever the spoon falls behind, so the spoon only ever moves through
    /// poses the arm can hold.
    pub fn constrain(&mut self, sp: &mut SpoonController, dt: f64) {
        if !self.enabled {
            return;
        }
        let l = self.last_cmd;
        let (tp, tq) = (sp.target_pos, sp.target_quat);
        let changed = tp.x != l.pos.x
            || tp.y != l.pos.y
            || tp.z != l.pos.z
            || tq.x != l.quat.x
            || tq.y != l.quat.y
            || tq.z != l.quat.z
            || tq.w != l.quat.w;
        if changed {
            if !self.constrain_target {
                // raw mode: hand the target to the spoon as is, the arm just follows
                self.goal = SpoonPose { pos: tp, quat: tq };
                self.last_cmd = self.goal;
                return;
            }
            self.set_goal(tp, tq);
        } else if !self.constrain_target {
            return;
        }

        // A collision can leave the old command permanently beyond the lead limit.
        // Fresh input must be able to retreat: rebase its path on the joints solved
        // from the actual spoon pose, rather than waiting for the unreachable old command.
        if changed {
            let old = self.spoon_pose(&self.q_cmd);
            if old.pos.distance_to(sp.pos) > LEAD_LIN || old.quat.angle_to(sp.quat) > lead_ang() {
                self.q_cmd = self.q;
            }
        }
        let cmd = self.spoon_pose(&self.q_cmd);
        let lagging = cmd.pos.distance_to(sp.pos) > LEAD_LIN || cmd.quat.angle_to(sp.quat) > lead_ang();
        if !lagging {
            let mut big = 0f64;
            for k in 0..5 {
                big = big.max((self.q_goal[k] - self.q_cmd[k]).abs());
            }
            let max_step = ((self.joint_speed * PI) / 180.0) * dt;
            let f = if big > max_step { max_step / big } else { 1.0 };
            for k in 0..5 {
                self.q_cmd[k] += (self.q_goal[k] - self.q_cmd[k]) * f;
            }
        }
        let p = self.spoon_pose(&self.q_cmd);
        sp.set_target(Some(p.pos), Some(p.quat));
        self.last_cmd = SpoonPose { pos: sp.target_pos, quat: sp.target_quat };
    }

    /// After the sim step: joint angles for the spoon's actual pose.
    pub fn follow(&mut self, sp: &SpoonController) {
        if !self.enabled {
            return;
        }
        let (pos, quat) = (sp.pos, sp.quat);
        let mut q = self.q;
        self.err = self.solve(&mut q, pos, quat, 16);
        self.q = q;
        if self.err.mm > 0.05 || self.err.deg > 0.05 {
            let mut alt = self.q_cmd;
            let e = self.solve(&mut alt, pos, quat, 80);
            if e.mm + e.deg < self.err.mm + self.err.deg {
                self.q = alt;
                self.err = e;
            } else {
                let mut q = self.q;
                self.err = self.solve(&mut q, pos, quat, 80);
                self.q = q;
            }
        }
    }

    /// Joint-space command: the goal becomes these joint angles (clamped to the limits).
    pub fn set_joints(&mut self, joints: [f64; 5]) {
        let mut q = joints;
        clamp_q(&mut q, &self.lim);
        self.q_goal = q;
        self.goal_err = PoseErr::default();
        self.goal = self.spoon_pose(&q);
    }

    /// World matrices (row-major 4x4) of every link, in [`super::so101::LINKS`] order.
    pub fn link_matrices(&self) -> [M4; 8] {
        forward(&self.q, self.jaw).t.map(|t| m4::mul(&self.b, &t))
    }

    pub fn state(&self) -> ArmState {
        let r2d = |v: f64| (v * 180.0) / PI;
        ArmState {
            enabled: self.enabled,
            tool: self.tool.name,
            joint_names: ARM_JOINTS,
            joints: self.q,
            joints_deg: self.q.map(r2d),
            goal_joints: self.q_goal,
            command_joints: self.q_cmd,
            goal_position: self.goal.pos.to_array(),
            goal_quaternion: self.goal.quat.to_array(),
            goal_unreachable_mm: self.goal_err.mm,
            goal_unreachable_deg: self.goal_err.deg,
            jaw: self.jaw,
            tracking_error_mm: self.err.mm,
            tracking_error_deg: self.err.deg,
            base: self.base,
        }
    }
}

/// The `arm` block of the twin's state (same fields as `Arm.state()` in arm.js).
#[derive(Clone, Debug)]
pub struct ArmState {
    pub enabled: bool,
    pub tool: &'static str,
    pub joint_names: [&'static str; 5],
    pub joints: [f64; 5],
    pub joints_deg: [f64; 5],
    pub goal_joints: [f64; 5],
    pub command_joints: [f64; 5],
    pub goal_position: [f64; 3],
    pub goal_quaternion: [f64; 4],
    pub goal_unreachable_mm: f64,
    pub goal_unreachable_deg: f64,
    pub jaw: f64,
    pub tracking_error_mm: f64,
    pub tracking_error_deg: f64,
    pub base: Base,
}
