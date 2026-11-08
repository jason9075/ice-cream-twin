//! Minimal vector / quaternion math, mirroring `src/math.js` operation for operation so the
//! Rust and JS simulations produce the same numbers (f64 arithmetic, V8's `Math.hypot`,
//! fdlibm-family trig from the `libm` crate).

use std::f64::consts::PI;

pub fn deg_to_rad(d: f64) -> f64 {
    (d * PI) / 180.0
}
pub fn rad_to_deg(r: f64) -> f64 {
    (r * 180.0) / PI
}
pub fn clamp(v: f64, a: f64, b: f64) -> f64 {
    a.max(b.min(v))
}

/// JavaScript `Math.round` (half rounds toward +infinity).
pub fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// V8's `Math.hypot`: scale by the largest magnitude, Kahan-compensated sum of squares.
pub fn js_hypot(vals: &[f64]) -> f64 {
    let mut max = 0.0f64;
    for &v in vals {
        let a = v.abs();
        if a.is_infinite() {
            return f64::INFINITY;
        }
        if a.is_nan() {
            return f64::NAN;
        }
        if a > max {
            max = a;
        }
    }
    if max == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in vals {
        let n = v.abs() / max;
        let summand = n * n - comp;
        let prelim = sum + summand;
        comp = (prelim - sum) - summand;
        sum = prelim;
    }
    sum.sqrt() * max
}
pub fn hypot2(a: f64, b: f64) -> f64 {
    js_hypot(&[a, b])
}
pub fn hypot3(a: f64, b: f64, c: f64) -> f64 {
    js_hypot(&[a, b, c])
}

pub fn sin(x: f64) -> f64 {
    crate::v8math::sin(x)
}
pub fn cos(x: f64) -> f64 {
    crate::v8math::cos(x)
}
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}
pub fn cbrt(x: f64) -> f64 {
    crate::v8math::cbrt(x)
}
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    pub fn add(self, v: Vec3) -> Vec3 {
        Vec3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
    pub fn sub(self, v: Vec3) -> Vec3 {
        Vec3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
    pub fn add_scaled(self, v: Vec3, s: f64) -> Vec3 {
        Vec3::new(self.x + v.x * s, self.y + v.y * s, self.z + v.z * s)
    }
    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    /// three.js `divideScalar` = multiply by the reciprocal.
    pub fn div(self, s: f64) -> Vec3 {
        self.scale(1.0 / s)
    }
    pub fn dot(self, v: Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        self.div(if l == 0.0 { 1.0 } else { l })
    }
    pub fn set_length(self, l: f64) -> Vec3 {
        self.normalize().scale(l)
    }
    pub fn distance_to(self, v: Vec3) -> f64 {
        hypot3(self.x - v.x, self.y - v.y, self.z - v.z)
    }
    pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
        Vec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }
    pub fn apply_quat(self, q: Quat) -> Vec3 {
        let (x, y, z) = (self.x, self.y, self.z);
        let tx = 2.0 * (q.y * z - q.z * y);
        let ty = 2.0 * (q.z * x - q.x * z);
        let tz = 2.0 * (q.x * y - q.y * x);
        Vec3::new(
            x + q.w * tx + q.y * tz - q.z * ty,
            y + q.w * ty + q.z * tx - q.x * tz,
            z + q.w * tz + q.x * ty - q.y * tx,
        )
    }
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    pub fn from_array(a: [f64; 3]) -> Vec3 {
        Vec3::new(a[0], a[1], a[2])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Quat { x, y, z, w }
    }
    /// Conjugate (unit quaternions only), like `invert()` in math.js.
    pub fn invert(self) -> Quat {
        Quat::new(-self.x, -self.y, -self.z, self.w)
    }
    pub fn normalize(self) -> Quat {
        let mut l = js_hypot(&[self.x, self.y, self.z, self.w]);
        if l == 0.0 {
            l = 1.0;
        }
        Quat::new(self.x / l, self.y / l, self.z / l, self.w / l)
    }
    /// self * q
    pub fn mul(self, q: Quat) -> Quat {
        let a = self;
        Quat::new(
            a.x * q.w + a.w * q.x + a.y * q.z - a.z * q.y,
            a.y * q.w + a.w * q.y + a.z * q.x - a.x * q.z,
            a.z * q.w + a.w * q.z + a.x * q.y - a.y * q.x,
            a.w * q.w - a.x * q.x - a.y * q.y - a.z * q.z,
        )
    }
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Quat {
        let s = sin(angle / 2.0);
        Quat::new(axis.x * s, axis.y * s, axis.z * s, cos(angle / 2.0))
    }
    /// Intrinsic XYZ Euler angles (radians), three.js 'XYZ'.
    pub fn from_euler(x: f64, y: f64, z: f64) -> Quat {
        let (c1, c2, c3) = (cos(x / 2.0), cos(y / 2.0), cos(z / 2.0));
        let (s1, s2, s3) = (sin(x / 2.0), sin(y / 2.0), sin(z / 2.0));
        Quat::new(
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 + s1 * s2 * c3,
            c1 * c2 * c3 - s1 * s2 * s3,
        )
    }
    pub fn dot(self, q: Quat) -> f64 {
        self.x * q.x + self.y * q.y + self.z * q.z + self.w * q.w
    }
    pub fn angle_to(self, q: Quat) -> f64 {
        2.0 * acos(1.0f64.min(self.dot(q).abs()))
    }
    pub fn to_array(self) -> [f64; 4] {
        [self.x, self.y, self.z, self.w]
    }
    pub fn from_array(a: [f64; 4]) -> Quat {
        Quat::new(a[0], a[1], a[2], a[3])
    }
    /// XYZ Euler angles (radians), three.js `Euler.setFromQuaternion(q, 'XYZ')`.
    pub fn to_euler_xyz(self) -> [f64; 3] {
        let q = self;
        let m11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let m12 = 2.0 * (q.x * q.y - q.z * q.w);
        let m13 = 2.0 * (q.x * q.z + q.y * q.w);
        let m22 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z);
        let m23 = 2.0 * (q.y * q.z - q.x * q.w);
        let m32 = 2.0 * (q.y * q.z + q.x * q.w);
        let m33 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let y = asin(clamp(m13, -1.0, 1.0));
        if m13.abs() < 0.9999999 {
            [atan2(-m23, m33), y, atan2(-m12, m11)]
        } else {
            [atan2(m32, m22), y, 0.0]
        }
    }
}

/// Rotation matrix (row-major, R[r*3+c]) + translation of a pose, as `Sim._setPose` builds it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pose {
    pub r: [f64; 9],
    pub t: [f64; 3],
}

impl Pose {
    pub fn new(pos: Vec3, q: Quat) -> Pose {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        Pose {
            r: [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            t: [pos.x, pos.y, pos.z],
        }
    }
    /// local -> world
    #[inline]
    pub fn to_world(&self, lx: f64, ly: f64, lz: f64) -> [f64; 3] {
        let r = &self.r;
        [
            r[0] * lx + r[1] * ly + r[2] * lz + self.t[0],
            r[3] * lx + r[4] * ly + r[5] * lz + self.t[1],
            r[6] * lx + r[7] * ly + r[8] * lz + self.t[2],
        ]
    }
    /// world -> local (R^T (p - t))
    #[inline]
    pub fn to_local(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let r = &self.r;
        let (rx, ry, rz) = (x - self.t[0], y - self.t[1], z - self.t[2]);
        [
            r[0] * rx + r[3] * ry + r[6] * rz,
            r[1] * rx + r[4] * ry + r[7] * rz,
            r[2] * rx + r[5] * ry + r[8] * rz,
        ]
    }
    /// rotate a local direction into world
    #[inline]
    pub fn rot(&self, v: [f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [
            r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
            r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
            r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hypot_matches_plain_for_simple_values() {
        assert_eq!(hypot3(3.0, 4.0, 12.0), 13.0);
        assert_eq!(js_hypot(&[0.0, 0.0]), 0.0);
    }
    #[test]
    fn euler_roundtrip() {
        let q = Quat::from_euler(0.3, -0.7, 1.1);
        let e = q.to_euler_xyz();
        assert!((e[0] - 0.3).abs() < 1e-12 && (e[1] + 0.7).abs() < 1e-12 && (e[2] - 1.1).abs() < 1e-12);
    }
}
