//! The ice-cream digital twin.
//!
//! ```text
//! icecream app [--port 8765 | --no-control] [--paused] [--cpu]
//!                       the twin in a window: simulation and drawing on one GPU, a side panel, mouse
//!                       and keys; the control server (WebSocket, control.rs) runs alongside
//! icecream server [--port 8765] [--paused] [--cpu]
//!                       headless: the simulation in real time, controlled over the WebSocket only
//! icecream shot [DIR] [--cpu] [--params JSON] [--glossy]
//!                       a scripted session rendered off screen to PNGs (default DIR: shots); --params
//!                       patches the parameters first (as set_params), for A/B comparisons
//! icecream serve [--gpu] JSON lines on stdin -> JSON lines on stdout (same commands; for ROS nodes,
//!                       Python, test rigs); CPU by default
//! icecream bench        the scoop + plunge scenarios on the CPU, and a scoop on the GPU if there is one
//! ```
//!
//! The simulation runs on the GPU when there is one (wgpu); --cpu forces the CPU version.
//!
//! Commands are listed in protocol.rs. `serve` also takes `{"type":"config","frame":"ros","arm":true}`:
//! the default frame for replies and targets, and whether the SO-101 holds the spoon (on by default).

use icecream_core::math::{Quat, Vec3};
use icecream_core::sdf::bowls;
use icecream_core::{Sim, Twin};
use icecream_cli::protocol::{frame_of, handle, twin_state, Frame};
use icecream_cli::{app, control, shot};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn serve(gpu: bool) {
    let mut twin = icecream_cli::engine::open(!gpu);
    let mut default_frame = Frame::Yup;
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let hello = json!({ "type": "hello", "sim": "icecream-core", "version": env!("CARGO_PKG_VERSION"), "dt": 1.0 / 60.0 });
    writeln!(out, "{hello}").ok();
    out.flush().ok();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) if msg.get("type").and_then(|t| t.as_str()) == Some("config") => {
                default_frame = frame_of(&msg, default_frame);
                if let Some(on) = msg.get("arm").and_then(|a| a.as_bool()) {
                    twin.set_arm_enabled(on);
                }
                Some(twin_state(twin.as_ref(), default_frame))
            }
            Ok(msg) => match handle(twin.as_mut(), &msg, default_frame) {
                Ok(r) => r,
                Err(e) => Some(json!({ "type": "error", "message": e })),
            },
            Err(e) => Some(json!({ "type": "error", "message": format!("bad JSON: {e}") })),
        };
        if let Some(r) = reply {
            writeln!(out, "{r}").ok();
            out.flush().ok();
        }
    }
}

fn bench() {
    let b = &bowls()[0];
    let level = Quat::IDENTITY;
    let dig = Quat::from_euler(0.0, 0.0, 0.5);
    let mut sim = Sim::default();
    let before = sim.total_grams();
    let t0 = Instant::now();
    let mut frames = 0;
    let mut go = |sim: &mut Sim, p: Vec3, q: Quat, n: usize| {
        sim.set_target(Some(p), Some(q));
        sim.step_frames(n);
        frames += n;
    };
    go(&mut sim, Vec3::new(b.cx - 0.03, 0.14, 0.0), dig, 90);
    go(&mut sim, Vec3::new(b.cx - 0.02, 0.085, 0.0), dig, 150);
    go(&mut sim, Vec3::new(b.cx + 0.025, 0.088, 0.0), level, 150);
    go(&mut sim, Vec3::new(b.cx + 0.025, 0.16, 0.0), level, 120);
    let scooped = sim.state().scooped_grams;
    sim.eat();
    for _ in 0..3 {
        go(&mut sim, Vec3::new(b.cx, 0.14, 0.0), level, 60);
        go(&mut sim, Vec3::new(b.cx, 0.0, 0.0), level, 240);
        go(&mut sim, Vec3::new(b.cx, 0.15, 0.0), level, 150);
        sim.eat();
    }
    let dt = t0.elapsed().as_secs_f64();
    let after = sim.total_grams();
    println!("frames {frames}, {:.2} ms/frame ({:.0}x real time)", dt * 1000.0 / frames as f64, frames as f64 / 60.0 / dt);
    println!("first scoop {scooped:.1} g, eaten {:.1} g, mass {before:.1} -> {after:.1} g", sim.state().eaten_grams);

    // the same kind of scoop with the SO-101 holding the spoon (IK every frame)
    let face = |pitch: f64| {
        Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::PI / 2.0).mul(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), pitch))
    };
    let mut twin = Twin::default();
    let t0 = Instant::now();
    let mut n = 0;
    let mut worst = 0f64;
    for (p, pitch, frames) in [
        (Vec3::new(0.0, 0.14, 0.02), 0.5, 80),
        (Vec3::new(0.0, 0.086, 0.015), 0.5, 150),
        (Vec3::new(0.0, 0.088, -0.025), 0.0, 150),
        (Vec3::new(0.0, 0.15, -0.03), 0.0, 100),
    ] {
        twin.sim.set_target(Some(p), Some(face(pitch)));
        for _ in 0..frames {
            twin.step(1.0 / 60.0);
            worst = worst.max(twin.arm.err.mm);
        }
        n += frames;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "arm scoop: frames {n}, {:.2} ms/frame, scooped {:.1} g, arm within {worst:.4} mm of the spoon",
        dt * 1000.0 / n as f64,
        twin.sim.state().scooped_grams
    );

    // the same on the GPU simulation
    match icecream_gpu::GpuTwin::new(icecream_core::Params::default()) {
        Err(e) => println!("GPU: none ({e})"),
        Ok(mut g) => {
            let t0 = Instant::now();
            let mut worst = 0f64;
            for (p, pitch, frames) in [
                (Vec3::new(0.0, 0.14, 0.02), 0.5, 80),
                (Vec3::new(0.0, 0.086, 0.015), 0.5, 150),
                (Vec3::new(0.0, 0.088, -0.025), 0.0, 150),
                (Vec3::new(0.0, 0.15, -0.03), 0.0, 100),
            ] {
                g.sim.cpu.spoon.set_target(Some(p), Some(face(pitch)));
                for _ in 0..frames {
                    g.step(1.0 / 60.0);
                    worst = worst.max(g.arm.err.mm);
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "arm scoop on the GPU ({}): {:.2} ms/frame, scooped {:.1} g, arm within {worst:.4} mm",
                g.sim.adapter,
                dt * 1000.0 / 480.0,
                g.sim.state().scooped_grams
            );
        }
    }
}

fn flag_port(args: &[String]) -> Option<u16> {
    let k = args.iter().position(|a| a == "--port")?;
    Some(args.get(k + 1).and_then(|p| p.parse().ok()).expect("--port N"))
}

fn server(args: &[String]) {
    let opts = control::Options { port: flag_port(args).unwrap_or(8765), paused: args.iter().any(|a| a == "--paused"), cpu: args.iter().any(|a| a == "--cpu") };
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let (addr, server) = control::start(opts).await.expect("bind");
        eprintln!("icecream server: ws://localhost:{}/ws", addr.port());
        server.await.expect("server");
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("app") => {
            let rest = &args[2..];
            let port = if rest.iter().any(|a| a == "--no-control") { None } else { Some(flag_port(rest).unwrap_or(8765)) };
            app::run(app::Options { port, paused: rest.iter().any(|a| a == "--paused"), cpu: rest.iter().any(|a| a == "--cpu") });
        }
        Some("server") => server(&args[2..]),
        Some("shot") => {
            let rest = &args[2..];
            let params = rest.iter().position(|a| a == "--params").map(|k| serde_json::from_str::<Value>(rest.get(k + 1).expect("--params JSON")).expect("--params: bad JSON"));
            let dir = rest.iter().enumerate().find(|(k, a)| !a.starts_with("--") && (*k == 0 || rest[k - 1] != "--params")).map(|(_, a)| a.clone()).unwrap_or_else(|| "shots".into());
            shot::run(std::path::Path::new(&dir), rest.iter().any(|a| a == "--cpu"), params.as_ref(), rest.iter().any(|a| a == "--glossy"));
        }
        Some("serve") => serve(args.iter().any(|a| a == "--gpu")),
        Some("bench") => bench(),
        _ => {
            eprintln!("usage: icecream app [--port N | --no-control] [--paused] [--cpu] | server [--port N] [--paused] [--cpu] | shot [DIR] [--cpu] | serve [--gpu] | bench");
            std::process::exit(2);
        }
    }
}
