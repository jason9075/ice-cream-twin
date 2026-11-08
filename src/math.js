// Minimal vector / quaternion / 4x4 matrix math (no dependencies).
// Vector3 and Quaternion keep three.js-style method names; matrices are column-major
// Float32Array(16) as WGSL expects, with WebGPU's 0..1 clip-space depth.

export const MathUtils = {
  degToRad: (d) => (d * Math.PI) / 180,
  radToDeg: (r) => (r * 180) / Math.PI,
  clamp: (v, a, b) => Math.max(a, Math.min(b, v)),
};

export class Vector3 {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  set(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  copy(v) {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }
  clone() {
    return new Vector3(this.x, this.y, this.z);
  }
  add(v) {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }
  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }
  addScaledVector(v, s) {
    this.x += v.x * s;
    this.y += v.y * s;
    this.z += v.z * s;
    return this;
  }
  subVectors(a, b) {
    return this.set(a.x - b.x, a.y - b.y, a.z - b.z);
  }
  multiplyScalar(s) {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }
  divideScalar(s) {
    return this.multiplyScalar(1 / s);
  }
  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  length() {
    return Math.sqrt(this.lengthSq());
  }
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  setLength(l) {
    return this.normalize().multiplyScalar(l);
  }
  distanceTo(v) {
    return Math.hypot(this.x - v.x, this.y - v.y, this.z - v.z);
  }
  crossVectors(a, b) {
    return this.set(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  }
  applyQuaternion(q) {
    const { x, y, z } = this;
    const tx = 2 * (q.y * z - q.z * y);
    const ty = 2 * (q.z * x - q.x * z);
    const tz = 2 * (q.x * y - q.y * x);
    this.x = x + q.w * tx + q.y * tz - q.z * ty;
    this.y = y + q.w * ty + q.z * tx - q.x * tz;
    this.z = z + q.w * tz + q.x * ty - q.y * tx;
    return this;
  }
  getComponent(i) {
    return i === 0 ? this.x : i === 1 ? this.y : this.z;
  }
  setComponent(i, v) {
    if (i === 0) this.x = v;
    else if (i === 1) this.y = v;
    else this.z = v;
    return this;
  }
  toArray() {
    return [this.x, this.y, this.z];
  }
}

export class Quaternion {
  constructor(x = 0, y = 0, z = 0, w = 1) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }
  set(x, y, z, w) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    return this;
  }
  copy(q) {
    return this.set(q.x, q.y, q.z, q.w);
  }
  clone() {
    return new Quaternion(this.x, this.y, this.z, this.w);
  }
  identity() {
    return this.set(0, 0, 0, 1);
  }
  invert() {
    return this.set(-this.x, -this.y, -this.z, this.w); // unit quaternions only
  }
  normalize() {
    const l = Math.hypot(this.x, this.y, this.z, this.w) || 1;
    return this.set(this.x / l, this.y / l, this.z / l, this.w / l);
  }
  /** this = this * q */
  multiply(q) {
    const a = this;
    return this.set(
      a.x * q.w + a.w * q.x + a.y * q.z - a.z * q.y,
      a.y * q.w + a.w * q.y + a.z * q.x - a.x * q.z,
      a.z * q.w + a.w * q.z + a.x * q.y - a.y * q.x,
      a.w * q.w - a.x * q.x - a.y * q.y - a.z * q.z,
    );
  }
  setFromAxisAngle(axis, angle) {
    const s = Math.sin(angle / 2);
    return this.set(axis.x * s, axis.y * s, axis.z * s, Math.cos(angle / 2));
  }
  /** Intrinsic XYZ Euler angles (radians), same convention as three.js 'XYZ'. */
  setFromEuler(e) {
    const c1 = Math.cos(e.x / 2);
    const c2 = Math.cos(e.y / 2);
    const c3 = Math.cos(e.z / 2);
    const s1 = Math.sin(e.x / 2);
    const s2 = Math.sin(e.y / 2);
    const s3 = Math.sin(e.z / 2);
    return this.set(
      s1 * c2 * c3 + c1 * s2 * s3,
      c1 * s2 * c3 - s1 * c2 * s3,
      c1 * c2 * s3 + s1 * s2 * c3,
      c1 * c2 * c3 - s1 * s2 * s3,
    );
  }
  dot(q) {
    return this.x * q.x + this.y * q.y + this.z * q.z + this.w * q.w;
  }
  angleTo(q) {
    return 2 * Math.acos(Math.min(1, Math.abs(this.dot(q))));
  }
  slerp(qb, t) {
    let cos = this.dot(qb);
    let bx = qb.x;
    let by = qb.y;
    let bz = qb.z;
    let bw = qb.w;
    if (cos < 0) {
      cos = -cos;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    if (cos > 0.9995) {
      return this.set(
        this.x + (bx - this.x) * t,
        this.y + (by - this.y) * t,
        this.z + (bz - this.z) * t,
        this.w + (bw - this.w) * t,
      ).normalize();
    }
    const th = Math.acos(cos);
    const s = Math.sin(th);
    const ka = Math.sin((1 - t) * th) / s;
    const kb = Math.sin(t * th) / s;
    return this.set(this.x * ka + bx * kb, this.y * ka + by * kb, this.z * ka + bz * kb, this.w * ka + bw * kb);
  }
  toArray() {
    return [this.x, this.y, this.z, this.w];
  }
}

/** XYZ Euler angles; `setFromQuaternion` matches three.js Euler.setFromQuaternion(q, 'XYZ'). */
export class Euler {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  setFromQuaternion(q) {
    const m11 = 1 - 2 * (q.y * q.y + q.z * q.z);
    const m12 = 2 * (q.x * q.y - q.z * q.w);
    const m13 = 2 * (q.x * q.z + q.y * q.w);
    const m22 = 1 - 2 * (q.x * q.x + q.z * q.z);
    const m23 = 2 * (q.y * q.z - q.x * q.w);
    const m32 = 2 * (q.y * q.z + q.x * q.w);
    const m33 = 1 - 2 * (q.x * q.x + q.y * q.y);
    this.y = Math.asin(MathUtils.clamp(m13, -1, 1));
    if (Math.abs(m13) < 0.9999999) {
      this.x = Math.atan2(-m23, m33);
      this.z = Math.atan2(-m12, m11);
    } else {
      this.x = Math.atan2(m32, m22);
      this.z = 0;
    }
    return this;
  }
}

// ------------------------------------------------------------------ mat4 (column-major)
export const mat4 = {
  identity(out = new Float32Array(16)) {
    out.fill(0);
    out[0] = out[5] = out[10] = out[15] = 1;
    return out;
  },
  multiply(a, b, out = new Float32Array(16)) {
    const r = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let rI = 0; rI < 4; rI++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + rI] * b[c * 4 + k];
        r[c * 4 + rI] = s;
      }
    out.set(r);
    return out;
  },
  /** Rigid transform (+ uniform or per-axis scale) from position and quaternion. */
  compose(p, q, s = [1, 1, 1], out = new Float32Array(16)) {
    const { x, y, z, w } = q;
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    out[0] = (1 - (yy + zz)) * s[0];
    out[1] = (xy + wz) * s[0];
    out[2] = (xz - wy) * s[0];
    out[3] = 0;
    out[4] = (xy - wz) * s[1];
    out[5] = (1 - (xx + zz)) * s[1];
    out[6] = (yz + wx) * s[1];
    out[7] = 0;
    out[8] = (xz + wy) * s[2];
    out[9] = (yz - wx) * s[2];
    out[10] = (1 - (xx + yy)) * s[2];
    out[11] = 0;
    out[12] = p.x;
    out[13] = p.y;
    out[14] = p.z;
    out[15] = 1;
    return out;
  },
  translation(x, y, z, out = new Float32Array(16)) {
    mat4.identity(out);
    out[12] = x;
    out[13] = y;
    out[14] = z;
    return out;
  },
  invert(m, out = new Float32Array(16)) {
    const a = m;
    const b00 = a[0] * a[5] - a[1] * a[4];
    const b01 = a[0] * a[6] - a[2] * a[4];
    const b02 = a[0] * a[7] - a[3] * a[4];
    const b03 = a[1] * a[6] - a[2] * a[5];
    const b04 = a[1] * a[7] - a[3] * a[5];
    const b05 = a[2] * a[7] - a[3] * a[6];
    const b06 = a[8] * a[13] - a[9] * a[12];
    const b07 = a[8] * a[14] - a[10] * a[12];
    const b08 = a[8] * a[15] - a[11] * a[12];
    const b09 = a[9] * a[14] - a[10] * a[13];
    const b10 = a[9] * a[15] - a[11] * a[13];
    const b11 = a[10] * a[15] - a[11] * a[14];
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    det = 1 / det;
    const r = [
      (a[5] * b11 - a[6] * b10 + a[7] * b09) * det,
      (a[2] * b10 - a[1] * b11 - a[3] * b09) * det,
      (a[13] * b05 - a[14] * b04 + a[15] * b03) * det,
      (a[10] * b04 - a[9] * b05 - a[11] * b03) * det,
      (a[6] * b08 - a[4] * b11 - a[7] * b07) * det,
      (a[0] * b11 - a[2] * b08 + a[3] * b07) * det,
      (a[14] * b02 - a[12] * b05 - a[15] * b01) * det,
      (a[8] * b05 - a[10] * b02 + a[11] * b01) * det,
      (a[4] * b10 - a[5] * b08 + a[7] * b06) * det,
      (a[1] * b08 - a[0] * b10 - a[3] * b06) * det,
      (a[12] * b04 - a[13] * b02 + a[15] * b00) * det,
      (a[9] * b02 - a[8] * b04 - a[11] * b00) * det,
      (a[5] * b07 - a[4] * b09 - a[6] * b06) * det,
      (a[0] * b09 - a[1] * b07 + a[2] * b06) * det,
      (a[13] * b01 - a[12] * b03 - a[14] * b00) * det,
      (a[8] * b03 - a[9] * b01 + a[10] * b00) * det,
    ];
    out.set(r);
    return out;
  },
  /** Perspective projection with WebGPU depth range [0, 1]. */
  perspective(fovy, aspect, near, far, out = new Float32Array(16)) {
    const f = 1 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far / (near - far);
    out[11] = -1;
    out[14] = (near * far) / (near - far);
    return out;
  },
  ortho(l, r, b, t, near, far, out = new Float32Array(16)) {
    out.fill(0);
    out[0] = 2 / (r - l);
    out[5] = 2 / (t - b);
    out[10] = 1 / (near - far);
    out[12] = -(r + l) / (r - l);
    out[13] = -(t + b) / (t - b);
    out[14] = near / (near - far);
    out[15] = 1;
    return out;
  },
  lookAt(eye, target, up, out = new Float32Array(16)) {
    const z = new Vector3().subVectors(eye, target).normalize();
    const x = new Vector3().crossVectors(up, z).normalize();
    const y = new Vector3().crossVectors(z, x);
    out[0] = x.x;
    out[1] = y.x;
    out[2] = z.x;
    out[3] = 0;
    out[4] = x.y;
    out[5] = y.y;
    out[6] = z.y;
    out[7] = 0;
    out[8] = x.z;
    out[9] = y.z;
    out[10] = z.z;
    out[11] = 0;
    out[12] = -x.dot(eye);
    out[13] = -y.dot(eye);
    out[14] = -z.dot(eye);
    out[15] = 1;
    return out;
  },
};
