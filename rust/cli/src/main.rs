//! Headless ice-cream digital twin.
//!
//! ```text
//! icecream serve        JSON lines on stdin -> JSON lines on stdout (same messages as the
//!                       web page's WebSocket bridge; for ROS nodes, Python, test rigs)
//! icecream bench        run the scoop + plunge scenarios, print ms/frame and mass drift
//! ```
//!
//! `serve` messages (one JSON object per line):
//! - `{"type":"set_target","position":[x,y,z],"quaternion":[x,y,z,w],"frame":"ros"}` (either part optional);
//!   with the arm on, the spoon goes to the closest pose the SO-101 can reach
//! - `{"type":"set_joints","joints":[pan,lift,elbow,wrist_flex,wrist_roll],"unit":"deg"}` SO-101 joint command
//!   (`unit` "rad" by default)
//! - `{"type":"step","frames":60}`        advance the simulation (default 1 frame of 1/60 s), replies with state
//! - `{"type":"get_state","frame":"ros"}` reply with the state
//! - `{"type":"eat"}` `{"type":"release"}` `{"type":"reset"}` `{"type":"refill"}`
//! - `{"type":"config","frame":"ros","arm":true}` default frame for replies and targets ("yup" or "ros");
//!   `arm` hands the spoon to the SO-101 (on by default, like the web page) or frees it
//!
//! Every reply is `{"type":"state", ...}` or `{"type":"error","message":...}`.

use icecream_core::math::{Quat, Vec3};
use icecream_core::sdf::{bowls, FLAVORS};
use icecream_core::robot::ArmState;
use icecream_core::{frames, Sim, State, Twin};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum Frame {
    Yup,
    Ros,
}

fn frame_of(v: &Value, default: Frame) -> Frame {
    match v.get("frame").and_then(|f| f.as_str()) {
        Some("ros") => Frame::Ros,
        Some("yup") => Frame::Yup,
        _ => default,
    }
}

fn arr<const N: usize>(v: Option<&Value>) -> Option<[f64; N]> {
    let a = v?.as_array()?;
    if a.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.as_f64()?;
    }
    Some(out)
}

fn state_json(s: &State, arm: Option<ArmState>, f: Frame) -> Value {
    let (pos, quat, lin, ang, tp, tq) = if f == Frame::Ros {
        (
            frames::pos_to_ros(s.spoon_position),
            frames::quat_to_ros(s.spoon_quaternion),
            frames::pos_to_ros(s.linear_velocity),
            frames::pos_to_ros(s.angular_velocity),
            frames::pos_to_ros(s.target_position),
            frames::quat_to_ros(s.target_quaternion),
        )
    } else {
        (s.spoon_position, s.spoon_quaternion, s.linear_velocity, s.angular_velocity, s.target_position, s.target_quaternion)
    };
    let bowl_grams: serde_json::Map<String, Value> = FLAVORS.iter().zip(s.bowl_grams).map(|(f, g)| (f.id.to_string(), json!(g))).collect();
    let arm = arm.map(|a| {
        let (gp, gq, bp) = if f == Frame::Ros {
            (frames::pos_to_ros(a.goal_position), frames::quat_to_ros(a.goal_quaternion), frames::pos_to_ros(a.base.position))
        } else {
            (a.goal_position, a.goal_quaternion, a.base.position)
        };
        json!({
            "enabled": a.enabled, "tool": a.tool, "jointNames": a.joint_names,
            "joints": a.joints, "jointsDeg": a.joints_deg, "goalJoints": a.goal_joints, "commandJoints": a.command_joints,
            "goal": { "position": gp, "quaternion": gq },
            "goalUnreachableMm": a.goal_unreachable_mm, "goalUnreachableDeg": a.goal_unreachable_deg,
            "jaw": a.jaw, "trackingErrorMm": a.tracking_error_mm, "trackingErrorDeg": a.tracking_error_deg,
            "base": { "position": bp, "yawDeg": a.base.yaw_deg },
        })
    });
    json!({
        "type": "state",
        "frame": if f == Frame::Ros { "ros" } else { "yup" },
        "time": s.time,
        "spoon": {
            "position": pos, "quaternion": quat,
            "linearVelocity": lin, "angularVelocity": ang,
            "eulerXYZdeg": s.spoon_euler_xyz_deg, "tiltDeg": s.tilt_deg,
        },
        "target": { "position": tp, "quaternion": tq },
        "contact": s.contact, "contactSamples": s.contact_samples, "blocked": s.blocked,
        "speedScale": s.speed_scale, "cutForceN": s.cut_force_n,
        "scoopedGrams": s.scooped_grams, "eatenGrams": s.eaten_grams, "spilledGrams": s.spilled_grams,
        "bowlGrams": bowl_grams, "particles": s.particles,
        "arm": arm,
    })
}

fn reply(t: &Twin, f: Frame) -> Value {
    state_json(&t.sim.state(), Some(t.arm.state()), f)
}

fn handle(t: &mut Twin, msg: &Value, default_frame: &mut Frame) -> Result<Option<Value>, String> {
    let ty = msg.get("type").and_then(|t| t.as_str()).ok_or("missing \"type\"")?;
    let f = frame_of(msg, *default_frame);
    match ty {
        "config" => {
            *default_frame = f;
            if let Some(on) = msg.get("arm").and_then(|a| a.as_bool()) {
                t.set_arm_enabled(on);
            }
            Ok(Some(reply(t, f)))
        }
        "set_target" => {
            let p = arr::<3>(msg.get("position")).map(|p| if f == Frame::Ros { frames::pos_from_ros(p) } else { p });
            let q = arr::<4>(msg.get("quaternion")).map(|q| if f == Frame::Ros { frames::quat_from_ros(q) } else { q });
            if p.is_none() && q.is_none() {
                return Err("set_target needs \"position\" [x,y,z] and/or \"quaternion\" [x,y,z,w]".into());
            }
            t.sim.set_target(p.map(Vec3::from_array), q.map(Quat::from_array));
            Ok(None)
        }
        "set_joints" => {
            let j = arr::<5>(msg.get("joints")).ok_or("set_joints needs \"joints\": 5 angles (pan, lift, elbow, wrist_flex, wrist_roll)")?;
            let k = match msg.get("unit").and_then(|u| u.as_str()).unwrap_or("rad") {
                "rad" => 1.0,
                "deg" => std::f64::consts::PI / 180.0,
                u => return Err(format!("unknown unit {u:?} (\"rad\" or \"deg\")")),
            };
            if !t.arm.enabled {
                return Err("the arm is off (config {\"arm\": true})".into());
            }
            t.arm.set_joints(j.map(|v| v * k));
            Ok(None)
        }
        "step" => {
            let n = msg.get("frames").and_then(|n| n.as_u64()).unwrap_or(1) as usize;
            t.step_frames(n);
            Ok(Some(reply(t, f)))
        }
        "get_state" => Ok(Some(reply(t, f))),
        "eat" => {
            t.sim.eat();
            Ok(Some(reply(t, f)))
        }
        "release" => {
            // tip the spoon over in place; the scoop slides off over the next frames
            let q = Quat::from_euler(std::f64::consts::PI * 0.9, 0.0, 0.0);
            t.sim.set_target(None, Some(q));
            Ok(None)
        }
        "reset" => {
            t.reset();
            Ok(Some(reply(t, f)))
        }
        "refill" => {
            t.sim.refill();
            Ok(Some(reply(t, f)))
        }
        other => Err(format!("unknown message type {other:?}")),
    }
}

fn serve() {
    let mut twin = Twin::default();
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
            Ok(msg) => match handle(&mut twin, &msg, &mut default_frame) {
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
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("serve") => serve(),
        Some("bench") => bench(),
        _ => {
            eprintln!("usage: icecream serve | bench");
            std::process::exit(2);
        }
    }
}
