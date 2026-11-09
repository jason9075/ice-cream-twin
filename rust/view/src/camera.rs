//! Orbit camera: left drag orbits, right drag (or shift + left) pans, the wheel zooms. The app maps the
//! mouse onto these; `ray` turns a pixel into a world ray (for dragging the spoon target).

use glam::{Mat4, Vec3};

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// around Y, 0 = looking from +Z
    pub azimuth: f32,
    /// from +Y
    pub polar: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera { target: Vec3::new(0.0, 0.08, -0.07), distance: 0.74, azimuth: 0.24, polar: 0.98, fov: 38f32.to_radians(), near: 0.005, far: 10.0 }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let s = self.polar.sin();
        self.target + self.distance * Vec3::new(s * self.azimuth.sin(), self.polar.cos(), s * self.azimuth.cos())
    }

    /// (forward, right, up)
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        (fwd, right, right.cross(fwd))
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * 0.006;
        self.polar = (self.polar - dy * 0.006).clamp(0.05, std::f32::consts::PI - 0.05);
    }

    /// Pan by a mouse delta in pixels, for a view `height` pixels tall.
    pub fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        let s = self.distance * (self.fov / 2.0).tan() * 2.0 / height;
        let (_, right, up) = self.basis();
        self.target += right * (-dx * s) + up * (dy * s);
    }

    /// Zoom by wheel "pixels" (positive: away).
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (delta * 0.001).exp()).clamp(0.08, 3.0);
    }

    /// Place the camera at an eye position looking at a target.
    pub fn look_from(&mut self, eye: Vec3, target: Vec3) {
        self.target = target;
        let d = eye - target;
        self.distance = d.length();
        self.polar = (d.y / self.distance).acos();
        self.azimuth = d.x.atan2(d.z);
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, self.near, self.far)
    }

    /// World ray (origin, direction) through pixel (x, y) of a `w` x `h` view.
    pub fn ray(&self, x: f32, y: f32, w: f32, h: f32) -> (Vec3, Vec3) {
        let nx = x / w * 2.0 - 1.0;
        let ny = 1.0 - y / h * 2.0;
        let (fwd, right, up) = self.basis();
        let th = (self.fov / 2.0).tan();
        (self.eye(), (fwd + right * (nx * th * w / h) + up * (ny * th)).normalize())
    }
}
