// SO-101 kinematics from the official URDF (src/robot/so101-urdf.js).
// Matrices are 4x4 row-major Float64Array: M[r*4+c]. Everything here is in the robot's own Z-up frame.
import { SO101_JOINTS } from './so101-urdf.js';

/** The five arm joints, base to tip. The gripper jaw is handled separately (the tool module fixes it). */
export const ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'];
export const JAW_JOINT = 'gripper';

const byName = Object.fromEntries(SO101_JOINTS.map((j) => [j.name, j]));
export const LIMITS = ARM_JOINTS.map((n) => [byName[n].lower, byName[n].upper]);
export const JAW_LIMITS = [byName[JAW_JOINT].lower, byName[JAW_JOINT].upper];

// ---------------------------------------------------------------- 4x4 helpers
export const m4 = {
  identity() {
    const m = new Float64Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
  },
  mul(a, b, out = new Float64Array(16)) {
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        out[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
    return out;
  },
  /** URDF origin: translation xyz, then fixed-axis roll-pitch-yaw (R = Rz(y) Ry(p) Rx(r)). */
  fromXyzRpy([x, y, z], [r, p, yw]) {
    const cr = Math.cos(r), sr = Math.sin(r), cp = Math.cos(p), sp = Math.sin(p), cy = Math.cos(yw), sy = Math.sin(yw);
    return Float64Array.from([
      cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x,
      sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y,
      -sp, cp * sr, cp * cr, z,
      0, 0, 0, 1,
    ]);
  },
  /** Rotation by angle about a unit axis. */
  axisAngle([x, y, z], a) {
    const c = Math.cos(a), s = Math.sin(a), t = 1 - c;
    return Float64Array.from([
      t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
      t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
      t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
      0, 0, 0, 1,
    ]);
  },
  invertRigid(m, out = new Float64Array(16)) {
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 4 + c] = m[c * 4 + r];
    for (let r = 0; r < 3; r++) out[r * 4 + 3] = -(out[r * 4] * m[3] + out[r * 4 + 1] * m[7] + out[r * 4 + 2] * m[11]);
    out[12] = out[13] = out[14] = 0;
    out[15] = 1;
    return out;
  },
  /** Column-major Float32Array for the GPU. */
  toGPU(m, out = new Float32Array(16)) {
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = m[r * 4 + c];
    return out;
  },
};

// ---------------------------------------------------------------- chain
const ORIGIN = Object.fromEntries(SO101_JOINTS.map((j) => [j.name, m4.fromXyzRpy(j.xyz, j.rpy)]));

/**
 * Forward kinematics. q = the five arm angles (rad), jaw = gripper angle.
 * Returns { link name -> 4x4 } in the robot base frame, plus `joint` axes/origins for the Jacobian.
 */
export function forward(q, jaw = 0) {
  const T = { base_link: m4.identity() };
  const axes = [];
  const origins = [];
  let cur = T.base_link;
  for (let k = 0; k < ARM_JOINTS.length; k++) {
    const j = byName[ARM_JOINTS[k]];
    const at = m4.mul(cur, ORIGIN[j.name]);
    // world axis and pivot of this joint (before its own rotation)
    const a = j.axis;
    axes.push([
      at[0] * a[0] + at[1] * a[1] + at[2] * a[2],
      at[4] * a[0] + at[5] * a[1] + at[6] * a[2],
      at[8] * a[0] + at[9] * a[1] + at[10] * a[2],
    ]);
    origins.push([at[3], at[7], at[11]]);
    cur = m4.mul(at, m4.axisAngle(a, q[k]));
    T[j.child] = cur;
  }
  const g = byName[JAW_JOINT];
  T[g.child] = m4.mul(m4.mul(T.gripper_link, ORIGIN[g.name]), m4.axisAngle(g.axis, jaw));
  T.gripper_frame_link = m4.mul(T.gripper_link, ORIGIN.gripper_frame_joint);
  return { T, axes, origins };
}
