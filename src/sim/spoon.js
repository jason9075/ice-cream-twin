import * as M from '../math.js';
import { SPOON, sdEnv, gradEnv } from './sdf.js';

/**
 * The spoon follows a target pose (position + quaternion of the cup-centre frame) like a
 * speed-limited robot tool, but its motion is constrained so it can never penetrate the bowls
 * or the table:
 *
 *  1. Each frame the step toward the target is clamped to maxLinSpeed / maxAngSpeed.
 *  2. The step is split into sub-steps so no collision sample moves more than SUBSTEP_MAX
 *     (well under half the bowl wall thickness) — this is what prevents tunnelling.
 *  3. After every sub-step the pose is projected out of the environment with position-based
 *     rigid-body contact corrections (translation + rotation), which makes the spoon slide
 *     along walls instead of stopping dead.
 *  4. If a sub-step still penetrates after projection, it is rejected and the spoon stays at
 *     the last valid pose for this frame (reported as `blocked`).
 *
 * Ice cream never blocks the spoon; it slows it down through `speedScale` (cutting force).
 */

export const SUBSTEP_MAX = 0.0008; // m of sample travel per sub-step (bowl wall is 6 mm thick)
export const PEN_TOL = 0.0002; // m of residual penetration allowed after projection
export const SAMPLE_MARGIN = 0.0002; // samples are slightly fatter than the mesh
const CONTACT_DIST = 0.0004;
const MASS = 1;
const GYRATION = 0.05; // m — weights rotation vs translation in contact corrections
const INERTIA = MASS * GYRATION * GYRATION;

export function buildSpoonSamples(spacing = 0.0022) {
  const out = [];
  const { a, b, c, t } = SPOON;
  const rr = t / 2 + SAMPLE_MARGIN;
  const avg = (a + c) / 2;
  const nTh = Math.ceil(((Math.PI / 2) * (a + b)) / 2 / spacing);
  for (let i = 0; i <= nTh; i++) {
    const th = (i / nTh) * (Math.PI / 2);
    const s = Math.sin(th);
    const co = Math.cos(th);
    const nPh = Math.max(1, Math.ceil((2 * Math.PI * s * avg) / spacing));
    for (let j = 0; j < nPh; j++) {
      const ph = (j / nPh) * 2 * Math.PI;
      out.push(a * s * Math.cos(ph), -b * co, c * s * Math.sin(ph), rr);
    }
  }
  const H = SPOON.handle;
  const hr = SPOON.handleR + SAMPLE_MARGIN;
  for (let i = 0; i < H.length - 1; i++) {
    const p = H[i];
    const q = H[i + 1];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
    const n = Math.ceil(len / spacing);
    for (let k = 0; k < n; k++) {
      const u = k / n;
      out.push(p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u, p[2] + (q[2] - p[2]) * u, hr);
    }
  }
  const e = H[H.length - 1];
  out.push(e[0], e[1], e[2], hr);
  return new Float32Array(out);
}

export const HOME_POS = new M.Vector3(0, 0.17, 0.02);
export const HOME_QUAT = new M.Quaternion();

const _w = new M.Vector3();
const _arm = new M.Vector3();
const _rxn = new M.Vector3();
const _n = [0, 0, 0];
const _tmpQ = new M.Quaternion();
const _dp = new M.Vector3();
const _qd = new M.Quaternion();
const _qi = new M.Quaternion();
const _qs = new M.Quaternion();
const _axis = new M.Vector3();
const _unitAxis = new M.Vector3();
const _cp = new M.Vector3();
const _cq = new M.Quaternion();
const _before = new M.Vector3();
const _startPos = new M.Vector3();
const _startQuat = new M.Quaternion();

/** q <- normalize(q + 0.5 * [dθ, 0] ⊗ q) : apply a small world-space rotation vector. */
export function applyRotVec(q, x, y, z) {
  _tmpQ.set(x, y, z, 0).multiply(q);
  q.x += 0.5 * _tmpQ.x;
  q.y += 0.5 * _tmpQ.y;
  q.z += 0.5 * _tmpQ.z;
  q.w += 0.5 * _tmpQ.w;
  q.normalize();
  return q;
}

/** Rotation vector (axis * angle, world frame) taking quaternion a to b, shortest way. */
export function rotVecBetween(a, b, out) {
  _qd.copy(b).multiply(_qi.copy(a).invert());
  if (_qd.w < 0) _qd.set(-_qd.x, -_qd.y, -_qd.z, -_qd.w);
  const w = Math.min(1, Math.max(-1, _qd.w));
  const angle = 2 * Math.acos(w);
  const s = Math.sqrt(1 - w * w);
  if (s < 1e-9) return out.set(0, 0, 0);
  return out.set(_qd.x / s, _qd.y / s, _qd.z / s).multiplyScalar(angle);
}

export class SpoonController {
  constructor(params) {
    this.p = params;
    this.samples = buildSpoonSamples();
    this.nSamples = this.samples.length / 4;
    this.sampleContact = new Uint8Array(this.nSamples);
    let reach = 0;
    for (let i = 0; i < this.nSamples; i++) {
      const s = this.samples;
      reach = Math.max(reach, Math.hypot(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]) + s[i * 4 + 3]);
    }
    this.maxReach = reach;
    this.pos = new M.Vector3();
    this.quat = new M.Quaternion();
    this.targetPos = new M.Vector3();
    this.targetQuat = new M.Quaternion();
    this.linVel = new M.Vector3();
    this.angVel = new M.Vector3();
    this.reset();
  }

  reset() {
    this.pos.copy(HOME_POS);
    this.quat.copy(HOME_QUAT);
    this.targetPos.copy(HOME_POS);
    this.targetQuat.copy(HOME_QUAT);
    this.linVel.set(0, 0, 0);
    this.angVel.set(0, 0, 0);
    this.blocked = false;
    this.contacts = 0;
    this.correction = 0;
    this.substeps = 0;
  }

  setTarget(pos, quat) {
    if (pos) {
      this.targetPos.copy(pos);
      this.targetPos.x = Math.max(-0.45, Math.min(0.45, this.targetPos.x));
      this.targetPos.y = Math.max(0.0, Math.min(0.45, this.targetPos.y));
      this.targetPos.z = Math.max(-0.45, Math.min(0.45, this.targetPos.z));
    }
    if (quat) this.targetQuat.copy(quat).normalize();
  }

  /** Largest penetration depth (m) of any collision sample at a pose; <= 0 means free. */
  penetration(pos, quat) {
    const s = this.samples;
    let maxPen = -Infinity;
    for (let i = 0; i < this.nSamples; i++) {
      _w.set(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]).applyQuaternion(quat).add(pos);
      const pen = s[i * 4 + 3] - sdEnv(_w.x, _w.y, _w.z);
      if (pen > maxPen) maxPen = pen;
    }
    return maxPen;
  }

  /** Position-based rigid contact projection (Gauss-Seidel over samples). Mutates pos/quat. */
  project(pos, quat, iterations = 12) {
    const s = this.samples;
    for (let it = 0; it < iterations; it++) {
      let maxPen = 0;
      for (let i = 0; i < this.nSamples; i++) {
        _w.set(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]).applyQuaternion(quat).add(pos);
        const d = sdEnv(_w.x, _w.y, _w.z) - s[i * 4 + 3];
        if (d >= 0) continue;
        if (-d > maxPen) maxPen = -d;
        gradEnv(_w.x, _w.y, _w.z, _n);
        _arm.subVectors(_w, pos);
        _rxn.set(_n[0], _n[1], _n[2]);
        _rxn.crossVectors(_arm, _rxn);
        const wg = 1 / MASS + _rxn.lengthSq() / INERTIA;
        const lam = -d / wg;
        pos.x += (_n[0] * lam) / MASS;
        pos.y += (_n[1] * lam) / MASS;
        pos.z += (_n[2] * lam) / MASS;
        const k = lam / INERTIA;
        applyRotVec(quat, _rxn.x * k, _rxn.y * k, _rxn.z * k);
      }
      if (maxPen < 1e-6) break;
    }
  }

  countContacts() {
    const s = this.samples;
    let n = 0;
    for (let i = 0; i < this.nSamples; i++) {
      _w.set(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]).applyQuaternion(this.quat).add(this.pos);
      const d = sdEnv(_w.x, _w.y, _w.z) - s[i * 4 + 3];
      const c = d < CONTACT_DIST ? 1 : 0;
      this.sampleContact[i] = c;
      n += c;
    }
    return n;
  }

  /**
   * Advance toward the target. `onSubstep(prevPos, prevQuat, pos, quat, subDt)` is called after
   * every accepted sub-step so the caller can carve ice cream along the swept path.
   */
  update(dt, speedScale, onSubstep) {
    const p = this.p;
    _startPos.copy(this.pos);
    _startQuat.copy(this.quat);

    _dp.subVectors(this.targetPos, this.pos);
    const maxL = p.maxLinSpeed * speedScale * dt;
    if (_dp.length() > maxL) _dp.setLength(maxL);

    rotVecBetween(this.quat, this.targetQuat, _axis);
    let ang = _axis.length();
    const maxA = M.MathUtils.degToRad(p.maxAngSpeed) * speedScale * dt;
    if (ang > maxA) {
      _axis.multiplyScalar(maxA / ang);
      ang = maxA;
    }

    const disp = _dp.length() + ang * this.maxReach;
    const n = Math.min(64, Math.max(1, Math.ceil(disp / SUBSTEP_MAX)));
    const subDt = dt / n;
    if (ang > 1e-12) _qs.setFromAxisAngle(_unitAxis.copy(_axis).normalize(), ang / n);
    else _qs.identity();

    this.blocked = false;
    this.correction = 0;
    this.substeps = 0;
    const prevPos = new M.Vector3();
    const prevQuat = new M.Quaternion();
    for (let s = 0; s < n; s++) {
      _cp.copy(this.pos).addScaledVector(_dp, 1 / n);
      _cq.copy(_qs).multiply(this.quat).normalize();
      _before.copy(_cp);
      this.project(_cp, _cq);
      this.correction += _before.distanceTo(_cp);
      if (this.penetration(_cp, _cq) > PEN_TOL) {
        this.blocked = true;
        break;
      }
      prevPos.copy(this.pos);
      prevQuat.copy(this.quat);
      this.pos.copy(_cp);
      this.quat.copy(_cq);
      this.substeps++;
      if (onSubstep) onSubstep(prevPos, prevQuat, this.pos, this.quat, subDt);
    }

    this.linVel.subVectors(this.pos, _startPos).divideScalar(dt);
    rotVecBetween(_startQuat, this.quat, this.angVel).divideScalar(dt);
    this.contacts = this.countContacts();
    return this;
  }

  /** World-space velocity of a material point of the spoon. */
  pointVelocity(x, y, z, out) {
    const rx = x - this.pos.x;
    const ry = y - this.pos.y;
    const rz = z - this.pos.z;
    const w = this.angVel;
    out[0] = this.linVel.x + (w.y * rz - w.z * ry);
    out[1] = this.linVel.y + (w.z * rx - w.x * rz);
    out[2] = this.linVel.z + (w.x * ry - w.y * rx);
    return out;
  }
}
