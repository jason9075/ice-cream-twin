//! `icecream shot [DIR]`: a scripted session rendered off screen to PNGs (no window, no browser): the SO-101
//! scoops strawberry, close-ups, the scoop rolled off into the bowl, and the same frame through the View
//! panel's debug options. Prints a JSON summary.

use crate::engine::{open_on, Engine};
use crate::scene::{scene_desc, Sync};
use glam::Vec3;
use icecream_core::math::{Quat, Vec3 as V3};
use icecream_gpu::wgpu;
use icecream_view::{Light, OrbitCamera, ParticleMode, Renderer, Shading, ViewOptions};
use serde_json::json;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

const W: u32 = 1400;
const H: u32 = 860;

struct Shot {
    engine: Box<dyn Engine>,
    renderer: Renderer,
    sync: Sync,
    camera: OrbitCamera,
    view: ViewOptions,
    dir: std::path::PathBuf,
}

impl Shot {
    fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.engine.step(1.0 / 60.0);
        }
    }

    fn go(&mut self, p: [f64; 3], q: Quat, n: usize) {
        self.engine.set_target(Some(V3::new(p[0], p[1], p[2])), Some(q));
        self.steps(n);
    }

    fn save(&mut self, name: &str) {
        let mut enc = self.renderer.device.create_command_encoder(&Default::default());
        self.sync.frame(&mut self.renderer, self.engine.as_mut(), &mut enc);
        self.renderer.queue.submit([enc.finish()]);
        let px = self.renderer.capture(W, H, &self.camera, &Light::default(), &self.view);
        let file = std::fs::File::create(self.dir.join(name)).expect("create png");
        let mut e = png::Encoder::new(std::io::BufWriter::new(file), W, H);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().and_then(|mut w| w.write_image_data(&px)).expect("write png");
    }
}

/// The spoon with its handle toward the SO-101 (90° about Y), cup pitched by `pitch` about its own Z.
fn facing(pitch: f64) -> Quat {
    Quat::from_axis_angle(V3::new(0.0, 1.0, 0.0), PI / 2.0).mul(Quat::from_axis_angle(V3::new(0.0, 0.0, 1.0), pitch))
}

/// `params`: a set_params patch applied before anything happens (A/B comparisons).
pub fn run(dir: &Path, cpu: bool, params: Option<&serde_json::Value>, glossy: bool) {
    std::fs::create_dir_all(dir).expect("output directory");
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no GPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("icecream"), ..Default::default() }, None)).expect("device");
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let errors = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(Box::new(move |e| sink.lock().unwrap().push(e.to_string())));
    let engine = open_on(cpu, Some((device.clone(), queue.clone(), format!("{} ({:?})", info.name, info.backend))));
    let renderer = Renderer::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, &scene_desc(engine.as_ref()));
    let mut engine = engine;
    if let Some(patch) = params {
        let mut p = engine.params().clone();
        crate::protocol::apply_params(&mut p, patch).expect("--params");
        engine.set_params(p);
    }
    let mut s = Shot { engine, renderer, sync: Sync::default(), camera: OrbitCamera::default(), view: ViewOptions { ice_matte: !glossy, ..Default::default() }, dir: dir.to_path_buf() };

    s.steps(30);
    s.save("01-start.png");
    // the SO-101 scoops strawberry (the middle bowl)
    s.go([0.0, 0.14, 0.02], facing(0.5), 80);
    s.go([0.0, 0.086, 0.015], facing(0.5), 150);
    s.go([0.0, 0.088, -0.025], facing(0.0), 150);
    s.go([0.0, 0.15, -0.03], facing(0.0), 100);
    s.steps(48);
    let scooped = s.engine.state();
    s.save("02-scooped.png");
    s.camera.look_from(Vec3::new(0.2, 0.26, 0.16), Vec3::new(0.0, 0.13, -0.08));
    s.save("03-closeup.png");
    let gl = icecream_core::robot::so101::LINKS.iter().position(|l| *l == "gripper_link").expect("gripper_link");
    let g = s.engine.arm().link_matrices()[gl];
    let at = Vec3::new((g[3] - g[2] * 0.08) as f32, (g[7] - g[6] * 0.08) as f32, (g[11] - g[10] * 0.08) as f32);
    s.camera.look_from(at + Vec3::new(0.09, 0.05, 0.08), at);
    s.save("04-gripper.png");

    // roll the spoon over the bowl: past the slide-off tilt the scoop drops off in one piece
    let mut p = s.engine.params().clone();
    p.spoon.max_ang_speed = 360.0;
    s.engine.set_params(p);
    let roll = Quat::from_axis_angle(V3::new(0.0, 0.0, 1.0), 150f64.to_radians()).mul(facing(0.0));
    s.engine.set_target(Some(V3::new(0.0, 0.15, -0.03)), Some(roll));
    let mut rolled = 0;
    while s.engine.state().scooped_grams > 0.5 && rolled < 300 {
        s.steps(5);
        rolled += 5;
    }
    s.camera.look_from(Vec3::new(0.05, 0.27, 0.17), Vec3::new(0.0, 0.1, -0.03));
    let mut drops = Vec::new();
    for (name, n) in [("05-drop.png", 4), ("06-falling.png", 4), ("07-landed.png", 60)] {
        s.steps(n);
        s.save(name);
        let st = s.engine.state();
        drops.push(json!({ "shot": name, "particles": st.particles, "onSpoonGrams": st.scooped_grams, "tiltDeg": st.tilt_deg }));
    }
    // the same frame through the debug options
    type Edit = fn(&mut ViewOptions);
    let views: [(&str, Edit); 4] = [
        ("08-balls.png", |v| v.particles = ParticleMode::Balls),
        ("09-debug-tint.png", |v| {
            v.particles = ParticleMode::Surface;
            v.tint_fluid = true;
            v.grid_boxes = true;
        }),
        ("10-debug-normals.png", |v| {
            v.tint_fluid = false;
            v.grid_boxes = false;
            v.shading = Shading::Normals;
        }),
        ("11-debug-steps.png", |v| v.shading = Shading::MarchSteps),
    ];
    for (name, set) in views {
        set(&mut s.view);
        s.save(name);
    }
    // a scoop of mint dropped on the table behind the bowls: it freezes into the table's SDF
    s.view = ViewOptions { ice_matte: !glossy, ..Default::default() };
    let b = icecream_core::sdf::bowls()[2].cx;
    s.go([b, 0.14, 0.02], facing(0.5), 80);
    s.go([b, 0.086, 0.015], facing(0.5), 150);
    s.go([b, 0.088, -0.025], facing(0.0), 150);
    s.go([b, 0.15, -0.03], facing(0.0), 100);
    let spot = [0.29, 0.12, -0.13];
    s.go(spot, facing(0.0), 120);
    s.engine.set_target(Some(V3::new(spot[0], spot[1], spot[2])), Some(roll));
    let mut n = 0;
    while s.engine.state().scooped_grams > 0.5 && n < 300 {
        s.steps(5);
        n += 5;
    }
    s.steps(90);
    s.go([0.0, 0.17, 0.02], facing(0.0), 90);
    s.camera.look_from(Vec3::new(0.42, 0.2, 0.08), Vec3::new(0.28, 0.0, -0.13));
    s.save("12-table-drop.png");
    s.view.grid_boxes = true;
    s.save("13-table-grid.png");
    let table = s.engine.state();
    let errors = errors.lock().unwrap().clone();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "simulation": s.engine.kind(),
            "scoopedGrams": scooped.scooped_grams,
            "armTrackingErrorMm": s.engine.arm().err.mm,
            "rolledFrames": rolled,
            "drop": drops,
            "table": { "grams": table.table_grams, "crumbs": table.spilled_grams, "loose": table.particles },
            "gpuErrors": errors,
        }))
        .unwrap()
    );
    if !errors.is_empty() {
        std::process::exit(1);
    }
}
