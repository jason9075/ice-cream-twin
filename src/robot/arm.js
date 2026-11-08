// The SO-101 holding the spoon: base placement, inverse kinematics, and the link poses for rendering.
//
// The spoon controller (src/sim/spoon.js) stays in charge of contact and the no-penetration guarantee.
// The arm wraps it:
//   1. before each step the spoon target is projected onto what the 5-DOF arm can reach (IK, then FK),
//      so the spoon is never asked for a pose the arm can't hold;
//   2. after each step the arm is solved for the spoon's actual pose, which gives the joint angles that
//      are rendered and reported to the twin.
import * as M from '../math.js';
import { ARM_JOINTS, LIMITS, forward, m4 } from './so101.js';
import { TOOLS, DEFAULT_TOOL } from './tool.js';

/** Where the robot stands: behind the middle bowl, facing +Z (toward the camera). */
export const DEFAULT_BASE = { position: [0, 0, -0.28], yawDeg: -90 };

/** Spoon pose the arm goes home to: above the middle bowl, handle toward the robot, cup tipped 10° nose-down. */
export const ARM_HOME_POS = new M.Vector3(0, 0.15, 0);
export const ARM_HOME_QUAT = new M.Quaternion()
  .setFromAxisAngle(new M.Vector3(0, 1, 0), Math.PI / 2)
  .multiply(new M.Quaternion().setFromAxisAngle(new M.Vector3(0, 0, 1), (10 * Math.PI) / 180));

/** Rotation error of 1 rad weighs like this many metres of position error in the IK. */
const ROT_SCALE = 0.04;
const LAMBDA = 0.004;
const SEED = [0, -0.6, 0.9, 0.9, 0];
const SEEDS = [SEED, [0, 0.3, 0.3, 1.2, 0], [0, -1.2, 1.5, 0.3, 0], [0, 0.8, -0.5, 1.5, 0], [0, -0.3, 0.2, 1.4, 0]];
/** The joint command waits while the spoon is this far behind it. */
const LEAD_LIN = 0.012;
const LEAD_ANG = (20 * Math.PI) / 180;

/** world <- robot: the URDF is Z-up, the scene is Y-up; then a yaw about the scene's Y. */
function baseMatrix({ position, yawDeg }) {
  const yaw = (yawDeg * Math.PI) / 180;
  const zUpToYUp = Float64Array.from([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]);
  const B = m4.mul(m4.axisAngle([0, 1, 0], yaw), zUpToYUp);
  B[3] = position[0];
  B[7] = position[1];
  B[11] = position[2];
  return B;
}

function poseMatrix(pos, quat) {
  const { x, y, z, w } = quat;
  return Float64Array.from([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), pos.x,
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), pos.y,
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), pos.z,
    0, 0, 0, 1,
  ]);
}

function matrixToQuat(m, out = new M.Quaternion()) {
  const [m11, m12, m13, , m21, m22, m23, , m31, m32, m33] = m;
  const tr = m11 + m22 + m33;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    out.set((m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s);
  } else if (m11 > m22 && m11 > m33) {
    const s = 2 * Math.sqrt(1 + m11 - m22 - m33);
    out.set(0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s);
  } else if (m22 > m33) {
    const s = 2 * Math.sqrt(1 + m22 - m11 - m33);
    out.set((m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s);
  } else {
    const s = 2 * Math.sqrt(1 + m33 - m11 - m22);
    out.set((m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s);
  }
  return out.normalize();
}

/** Rotation vector of Ra * Rb^T (3x3 blocks of 4x4s). */
function rotError(Ra, Rb) {
  const E = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) E[r * 3 + c] = Ra[r * 4] * Rb[c * 4] + Ra[r * 4 + 1] * Rb[c * 4 + 1] + Ra[r * 4 + 2] * Rb[c * 4 + 2];
  const cos = Math.min(1, Math.max(-1, (E[0] + E[4] + E[8] - 1) / 2));
  const ang = Math.acos(cos);
  const v = [E[7] - E[5], E[2] - E[6], E[3] - E[1]];
  const s = Math.hypot(v[0], v[1], v[2]);
  if (s < 1e-12) {
    if (ang < 1e-6) return [0, 0, 0];
    // 180°: axis from the largest diagonal entry
    const i = E[0] > E[4] && E[0] > E[8] ? 0 : E[4] > E[8] ? 1 : 2;
    const a = [0, 0, 0];
    a[i] = Math.sqrt(Math.max(0, (E[i * 4] + 1) / 2));
    for (let k = 0; k < 3; k++) if (k !== i) a[k] = (E[i * 3 + k] + E[k * 3 + i]) / (4 * a[i]);
    return a.map((x) => x * ang);
  }
  const k = ang / s;
  return [v[0] * k, v[1] * k, v[2] * k];
}

/** Solve the n x n system A x = b in place (Gaussian elimination with partial pivoting). */
function solve(A, b, n) {
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[p * n + c])) p = r;
    if (p !== c) {
      for (let k = 0; k < n; k++) [A[c * n + k], A[p * n + k]] = [A[p * n + k], A[c * n + k]];
      [b[c], b[p]] = [b[p], b[c]];
    }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
    b[r] = s / A[r * n + r];
  }
  return b;
}

const clampQ = (q) => q.forEach((v, k) => (q[k] = Math.min(LIMITS[k][1], Math.max(LIMITS[k][0], v))));

export class Arm {
  constructor({ base = DEFAULT_BASE, tool = DEFAULT_TOOL } = {}) {
    this.setBase(base);
    this.setTool(tool);
    this.q = Float64Array.from(SEED); // solved from the spoon's actual pose (what is drawn and reported)
    this.qGoal = Float64Array.from(SEED); // where the arm is heading
    this.qCmd = Float64Array.from(SEED); // the joint-space command this frame
    this.err = { mm: 0, deg: 0 }; // how well q reproduces the actual spoon pose
    this.goalErr = { mm: 0, deg: 0 }; // how far the requested target was from reachable
    this.enabled = true;
    this.constrainTarget = true;
    this.jointSpeed = 90; // deg/s, fastest joint
    this.home = { pos: ARM_HOME_POS.clone(), quat: ARM_HOME_QUAT.clone() };
    this.reset();
  }

  setBase(base) {
    this.base = { position: [...base.position], yawDeg: base.yawDeg };
    this.B = baseMatrix(this.base);
    this.Binv = m4.invertRigid(this.B);
  }

  setTool(name) {
    const t = TOOLS[name];
    if (!t) throw new Error(`unknown tool module: ${name}`);
    this.toolName = name;
    this.tool = t;
    this.jaw = t.jaw ?? 0;
  }

  /** World pose of the spoon (rim centre) for joint angles q. */
  spoonPose(q, out = { pos: new M.Vector3(), quat: new M.Quaternion() }) {
    const W = m4.mul(this.B, m4.mul(forward(q, this.jaw).T.gripper_link, this.tool.spoonInGripper));
    out.pos.set(W[3], W[7], W[11]);
    matrixToQuat(W, out.quat);
    return out;
  }

  /**
   * Damped least squares on the 6-D spoon pose error (position + ROT_SCALE * rotation), warm-started
   * from q (modified in place). Returns the remaining error.
   */
  solve(q, pos, quat, iters = 40) {
    const D = m4.mul(this.Binv, poseMatrix(pos, quat)); // desired spoon pose, robot frame
    const n = ARM_JOINTS.length;
    const J = new Float64Array(6 * n);
    const A = new Float64Array(n * n);
    const g = new Float64Array(n);
    let ep = 0;
    let er = 0;
    for (let it = 0; it <= iters; it++) {
      const fk = forward(q, this.jaw);
      const S = m4.mul(fk.T.gripper_link, this.tool.spoonInGripper);
      const p = [S[3], S[7], S[11]];
      const e = [D[3] - p[0], D[7] - p[1], D[11] - p[2], ...rotError(D, S)];
      ep = Math.hypot(e[0], e[1], e[2]);
      er = Math.hypot(e[3], e[4], e[5]);
      if (it === iters || (ep < 1e-7 && er < 1e-6)) break;
      for (let k = 0; k < 3; k++) e[3 + k] *= ROT_SCALE;
      for (let j = 0; j < n; j++) {
        const a = fk.axes[j];
        const o = fk.origins[j];
        const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        J[0 * n + j] = a[1] * d[2] - a[2] * d[1];
        J[1 * n + j] = a[2] * d[0] - a[0] * d[2];
        J[2 * n + j] = a[0] * d[1] - a[1] * d[0];
        J[3 * n + j] = a[0] * ROT_SCALE;
        J[4 * n + j] = a[1] * ROT_SCALE;
        J[5 * n + j] = a[2] * ROT_SCALE;
      }
      // (J^T J + λ² I) dq = J^T e, with joints pinned at a limit (and pushing past it) taken out
      const locked = [false, false, false, false, false];
      let dq;
      for (let pass = 0; pass < 3; pass++) {
        for (let r = 0; r < n; r++) {
          let s = 0;
          for (let k = 0; k < 6; k++) s += locked[r] ? 0 : J[k * n + r] * e[k];
          g[r] = s;
          for (let c = 0; c < n; c++) {
            let t = 0;
            if (!locked[r] && !locked[c]) for (let k = 0; k < 6; k++) t += J[k * n + r] * J[k * n + c];
            A[r * n + c] = t + (r === c ? (locked[r] ? 1 : LAMBDA * LAMBDA) : 0);
          }
        }
        dq = solve(A, g, n);
        let again = false;
        for (let j = 0; j < n; j++) {
          if (locked[j]) continue;
          if ((q[j] <= LIMITS[j][0] + 1e-9 && dq[j] < 0) || (q[j] >= LIMITS[j][1] - 1e-9 && dq[j] > 0)) {
            locked[j] = true;
            again = true;
          }
        }
        if (!again) break;
      }
      let big = 0;
      for (let j = 0; j < n; j++) big = Math.max(big, Math.abs(dq[j]));
      const s = big > 0.25 ? 0.25 / big : 1;
      let moved = 0;
      for (let j = 0; j < n; j++) {
        const before = q[j];
        q[j] = Math.min(LIMITS[j][1], Math.max(LIMITS[j][0], q[j] + dq[j] * s));
        moved = Math.max(moved, Math.abs(q[j] - before));
      }
      if (moved < 1e-9) break;
    }
    return { mm: ep * 1000, deg: (er * 180) / Math.PI };
  }

  /** Put the arm (and the spoon, and its target) at the home pose. */
  reset(sim) {
    this.q.set(SEED);
    this.solve(this.q, this.home.pos, this.home.quat, 200);
    this.qGoal.set(this.q);
    this.qCmd.set(this.q);
    const p = this.spoonPose(this.q);
    this.goal = p;
    this._lastCmd = { pos: p.pos.clone(), quat: p.quat.clone() };
    this.err = { mm: 0, deg: 0 };
    if (sim) {
      const sp = sim.spoon;
      sp.pos.copy(p.pos);
      sp.quat.copy(p.quat);
      sp.setTarget(p.pos, p.quat);
      sp.linVel.set(0, 0, 0);
      sp.angVel.set(0, 0, 0);
    }
  }

  /** Set the goal from a pose: the reachable pose closest to it (IK, then FK). */
  _setGoal(pos, quat) {
    // seeds: the last goal (smooth while jogging), the current command (big jumps), then a few
    // canonical postures if neither fits. Keep the best fit; on a tie, the one closest to the arm.
    const fit = (e) => e.mm + e.deg * 0.5;
    const dist = (q) => q.reduce((s, v, k) => s + Math.abs(v - this.qCmd[k]), 0);
    let best = null;
    const tryFrom = (seed, iters) => {
      const q = Float64Array.from(seed);
      const e = this.solve(q, pos, quat, iters);
      if (!best || fit(e) < fit(best.e) - 0.05 || (Math.abs(fit(e) - fit(best.e)) <= 0.05 && dist(q) < dist(best.q)))
        best = { q, e };
    };
    tryFrom(this.qGoal, 60);
    tryFrom(this.qCmd, 150);
    if (fit(best.e) > 0.2) {
      const pan = Math.atan2(-(pos.x - this.B[3]), pos.z - this.B[11]); // rough heading toward the goal
      for (const s of SEEDS) tryFrom([pan, ...s.slice(1)], 150);
    }
    this.qGoal.set(best.q);
    this.goalErr = best.e;
    this.goal = this.spoonPose(this.qGoal);
  }

  /**
   * Before sim.step. A new spoon target (from the mouse, jog keys, the bridge…) becomes the arm's goal,
   * projected onto what the arm can reach. The spoon is then led there in joint space: the command
   * moves toward the goal at `jointSpeed` and waits whenever the spoon falls behind, so the spoon
   * only ever moves through poses the arm can hold.
   */
  constrain(sim, dt = 1 / 60) {
    if (!this.enabled) return;
    const sp = sim.spoon;
    const L = this._lastCmd;
    const tp = sp.targetPos;
    const tq = sp.targetQuat;
    const changed =
      tp.x !== L.pos.x || tp.y !== L.pos.y || tp.z !== L.pos.z ||
      tq.x !== L.quat.x || tq.y !== L.quat.y || tq.z !== L.quat.z || tq.w !== L.quat.w;
    if (changed) {
      if (!this.constrainTarget) {
        // raw mode: hand the target to the spoon as is, the arm just follows
        this.goal = { pos: sp.targetPos.clone(), quat: sp.targetQuat.clone() };
        this._lastCmd = { pos: sp.targetPos.clone(), quat: sp.targetQuat.clone() };
        return;
      }
      this._setGoal(sp.targetPos, sp.targetQuat);
    } else if (!this.constrainTarget) return;

    const cmd = this.spoonPose(this.qCmd);
    const lagging = cmd.pos.distanceTo(sp.pos) > LEAD_LIN || cmd.quat.angleTo(sp.quat) > LEAD_ANG;
    if (!lagging) {
      let big = 0;
      for (let k = 0; k < 5; k++) big = Math.max(big, Math.abs(this.qGoal[k] - this.qCmd[k]));
      const maxStep = ((this.jointSpeed * Math.PI) / 180) * dt;
      const f = big > maxStep ? maxStep / big : 1;
      for (let k = 0; k < 5; k++) this.qCmd[k] += (this.qGoal[k] - this.qCmd[k]) * f;
    }
    const p = this.spoonPose(this.qCmd);
    sp.setTarget(p.pos, p.quat);
    this._lastCmd = { pos: sp.targetPos.clone(), quat: sp.targetQuat.clone() };
  }

  /** After sim.step: joint angles for the spoon's actual pose. */
  follow(sim) {
    if (!this.enabled) return;
    const { pos, quat } = sim.spoon;
    this.err = this.solve(this.q, pos, quat, 16);
    if (this.err.mm > 0.05 || this.err.deg > 0.05) {
      const alt = Float64Array.from(this.qCmd);
      const e = this.solve(alt, pos, quat, 80);
      if (e.mm + e.deg < this.err.mm + this.err.deg) {
        this.q.set(alt);
        this.err = e;
      } else this.err = this.solve(this.q, pos, quat, 80);
    }
  }

  /** Joint-space command: the goal becomes these joint angles (clamped to the limits). */
  setJoints(sim, joints) {
    const q = Float64Array.from(joints.slice(0, 5));
    clampQ(q);
    this.qGoal.set(q);
    this.goalErr = { mm: 0, deg: 0 };
    this.goal = this.spoonPose(q);
  }

  /** World matrices (row-major 4x4) of every link, for rendering. */
  linkMatrices() {
    const { T } = forward(this.q, this.jaw);
    const out = {};
    for (const [k, v] of Object.entries(T)) out[k] = m4.mul(this.B, v);
    return out;
  }

  state() {
    const r2d = (v) => (v * 180) / Math.PI;
    return {
      enabled: this.enabled,
      tool: this.toolName,
      jointNames: ARM_JOINTS,
      joints: [...this.q],
      jointsDeg: [...this.q].map(r2d),
      goalJoints: [...this.qGoal],
      commandJoints: [...this.qCmd],
      goal: { position: this.goal.pos.toArray(), quaternion: this.goal.quat.toArray() },
      goalUnreachableMm: this.goalErr.mm,
      goalUnreachableDeg: this.goalErr.deg,
      jaw: this.jaw,
      trackingErrorMm: this.err.mm,
      trackingErrorDeg: this.err.deg,
      base: this.base,
    };
  }
}
