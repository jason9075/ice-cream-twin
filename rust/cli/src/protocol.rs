//! The JSON message protocol, shared by `icecream serve` (stdin/stdout) and `icecream web` (WebSocket).
//!
//! Commands (one JSON object each; `frame` is "yup" (scene, default) or "ros" (REP-103)):
//! - `set_target {position?, quaternion?, frame?}`  spoon target; with the arm on, the closest reachable pose
//! - `set_joints {joints: [pan, lift, elbow, wrist_flex, wrist_roll], unit?: "rad" | "deg"}`
//! - `home`, `eat`, `release` (tip the scoop off), `reset`, `refill`
//! - `arm {enabled?, constrainTarget?, jointSpeed?}`
//! - `set_params {params: {fluid: {...}, ice: {...}, spoon: {...}}}`, `get_params`
//! - `step {frames?}` advance (1/60 s each) and reply with the state, `get_state`
//!
//! Replies: `state`, `params`, `eaten {grams}`, `error {message}`; commands that only change the target
//! reply nothing.

use icecream_core::math::{Quat, Vec3};
use icecream_core::params::FillShape;
use icecream_core::robot::so101::{LINKS, ARM_JOINTS};
use icecream_core::robot::ArmState;
use icecream_core::sdf::{bowl_profile, bowls, FLAVORS, SPOON};
use icecream_core::spoon::{HOME_POS, HOME_QUAT};
use crate::engine::Engine;
use icecream_core::{frames, Params, State};
use serde_json::{json, Map, Value};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Frame {
    Yup,
    Ros,
}

pub fn frame_of(v: &Value, default: Frame) -> Frame {
    match v.get("frame").and_then(|f| f.as_str()) {
        Some("ros") => Frame::Ros,
        Some("yup") => Frame::Yup,
        _ => default,
    }
}

pub fn arr<const N: usize>(v: Option<&Value>) -> Option<[f64; N]> {
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

pub fn arm_json(a: &ArmState, f: Frame) -> Value {
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
}

pub fn state_json(s: &State, arm: Option<&ArmState>, f: Frame) -> Value {
    state_json_kind(s, arm, f, "cpu")
}

pub fn state_json_kind(s: &State, arm: Option<&ArmState>, f: Frame, kind: &str) -> Value {
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
    let bowl_grams: Map<String, Value> = FLAVORS.iter().zip(s.bowl_grams).map(|(f, g)| (f.id.to_string(), json!(g))).collect();
    json!({
        "type": "state",
        "frame": if f == Frame::Ros { "ros" } else { "yup" },
        "time": s.time,
        "spoon": {
            "position": pos, "quaternion": quat,
            "linearVelocity": lin, "angularVelocity": ang,
            "eulerXYZdeg": s.spoon_euler_xyz_deg, "tiltDeg": s.tilt_deg,
        },
        "target": {
            "position": tp, "quaternion": tq,
            "eulerXYZdeg": Quat::from_array(s.target_quaternion).to_euler_xyz().map(|a| a.to_degrees()),
        },
        "contact": s.contact, "contactSamples": s.contact_samples, "blocked": s.blocked,
        "speedScale": s.speed_scale, "cutForceN": s.cut_force_n,
        "scoopedGrams": s.scooped_grams, "eatenGrams": s.eaten_grams, "spilledGrams": s.spilled_grams,
        "bowlGrams": bowl_grams, "particles": s.particles,
        "arm": arm.map(|a| arm_json(a, f)),
        "simulation": kind,
    })
}

pub fn twin_state(t: &dyn Engine, f: Frame) -> Value {
    let arm = t.arm().state();
    state_json_kind(&t.state(), Some(&arm), f, t.kind())
}

// ---------------------------------------------------------------- parameters (same names as the panel)
pub fn params_json(p: &Params) -> Value {
    let (fl, ice, sp) = (&p.fluid, &p.ice, &p.spoon);
    json!({
        "fluid": {
            "particleRadius": fl.particle_radius, "kernelScale": fl.kernel_scale, "substeps": fl.substeps,
            "iterations": fl.iterations, "gravity": fl.gravity, "viscosity": fl.viscosity, "cohesion": fl.cohesion,
            "lumpStiffness": fl.lump_stiffness, "lumpYield": fl.lump_yield,
            "damping": fl.damping, "friction": fl.friction, "maxParticles": fl.max_particles, "stopSpeed": fl.stop_speed,
            "restFrames": fl.rest_frames, "stampScale": fl.stamp_scale, "stampSmooth": fl.stamp_smooth,
            "stampOnContact": fl.stamp_on_contact,
        },
        "ice": {
            "hardness": ice.hardness, "accretion": ice.accretion, "accretionDist": ice.accretion_dist,
            "capacity": ice.capacity, "releaseAngle": ice.release_angle, "displacedToParticles": ice.displaced_to_particles,
            "density": ice.density, "smoothing": ice.smoothing,
            "shape": match ice.shape { FillShape::Scoops => "scoops", FillShape::Flat => "flat", FillShape::Mound => "mound" },
        },
        "spoon": {
            "maxLinSpeed": sp.max_lin_speed, "maxAngSpeed": sp.max_ang_speed, "jogLinSpeed": sp.jog_lin_speed,
            "jogAngSpeed": sp.jog_ang_speed, "maxCutForce": sp.max_cut_force,
        },
    })
}

/// Merge a (partial) parameter object into `p`. Unknown names are an error.
pub fn apply_params(p: &mut Params, patch: &Value) -> Result<(), String> {
    let groups = patch.as_object().ok_or("params must be an object")?;
    for (group, vals) in groups {
        let vals = vals.as_object().ok_or_else(|| format!("params.{group} must be an object"))?;
        for (k, v) in vals {
            let num = || v.as_f64().ok_or_else(|| format!("{group}.{k} must be a number"));
            let flag = || v.as_bool().ok_or_else(|| format!("{group}.{k} must be true or false"));
            let (fl, ice, sp) = (&mut p.fluid, &mut p.ice, &mut p.spoon);
            match (group.as_str(), k.as_str()) {
                ("fluid", "particleRadius") => fl.particle_radius = num()?,
                ("fluid", "kernelScale") => fl.kernel_scale = num()?,
                ("fluid", "substeps") => fl.substeps = num()?.max(1.0) as u32,
                ("fluid", "iterations") => fl.iterations = num()?.max(0.0) as u32,
                ("fluid", "gravity") => fl.gravity = num()?,
                ("fluid", "viscosity") => fl.viscosity = num()?,
                ("fluid", "cohesion") => fl.cohesion = num()?,
                ("fluid", "lumpStiffness") => fl.lump_stiffness = num()?.clamp(0.0, 1.0),
                ("fluid", "lumpYield") => fl.lump_yield = num()?.max(0.0),
                ("fluid", "damping") => fl.damping = num()?,
                ("fluid", "friction") => fl.friction = num()?,
                ("fluid", "maxParticles") => fl.max_particles = num()?.clamp(0.0, 16000.0) as usize,
                ("fluid", "stopSpeed") => fl.stop_speed = num()?,
                ("fluid", "restFrames") => fl.rest_frames = num()?.max(0.0) as u32,
                ("fluid", "stampScale") => fl.stamp_scale = num()?,
                ("fluid", "stampSmooth") => fl.stamp_smooth = num()?,
                ("fluid", "stampOnContact") => fl.stamp_on_contact = flag()?,
                ("ice", "hardness") => ice.hardness = num()?,
                ("ice", "accretion") => ice.accretion = flag()?,
                ("ice", "accretionDist") => ice.accretion_dist = num()?,
                ("ice", "capacity") => ice.capacity = num()?,
                ("ice", "releaseAngle") => ice.release_angle = num()?,
                ("ice", "displacedToParticles") => ice.displaced_to_particles = flag()?,
                ("ice", "density") => ice.density = num()?,
                ("ice", "smoothing") => ice.smoothing = num()?,
                ("ice", "shape") => {
                    ice.shape = match v.as_str() {
                        Some("scoops") => FillShape::Scoops,
                        Some("flat") => FillShape::Flat,
                        Some("mound") => FillShape::Mound,
                        _ => return Err("ice.shape must be \"scoops\", \"flat\" or \"mound\"".into()),
                    }
                }
                ("spoon", "maxLinSpeed") => sp.max_lin_speed = num()?,
                ("spoon", "maxAngSpeed") => sp.max_ang_speed = num()?,
                ("spoon", "jogLinSpeed") => sp.jog_lin_speed = num()?,
                ("spoon", "jogAngSpeed") => sp.jog_ang_speed = num()?,
                ("spoon", "maxCutForce") => sp.max_cut_force = num()?,
                _ => return Err(format!("unknown parameter {group}.{k}")),
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- scene (for a viewer)
/// Everything a viewer needs to build the scene once: grids, bowls, spoon shape, flavours, arm links.
pub fn scene_json(t: &dyn Engine) -> Value {
    let s = t.sim();
    let grid = |g: &icecream_core::grid::SdfGrid| {
        json!({ "origin": [g.ox, g.oy, g.oz], "cell": g.cell, "dims": [g.nx, g.ny, g.nz], "band": g.band })
    };
    json!({
        "grids": s.grids.iter().map(grid).chain(std::iter::once(grid(&s.content))).collect::<Vec<_>>(),
        "bowls": bowls().iter().map(|b| json!({ "cx": b.cx, "cz": b.cz, "profile": bowl_profile(b, 48) })).collect::<Vec<_>>(),
        "spoon": { "a": SPOON.a, "b": SPOON.b, "c": SPOON.c, "t": SPOON.t, "handleR": SPOON.handle_r, "handle": SPOON.handle },
        "flavors": FLAVORS.iter().map(|f| json!({ "id": f.id, "label": f.label, "color": format!("#{:06x}", f.color), "speck": f.speck })).collect::<Vec<_>>(),
        "arm": { "links": LINKS, "jointNames": ARM_JOINTS, "jawLink": LINKS[6], "hideJaw": t.arm().tool.hide_jaw, "tool": t.arm().tool.label },
        "particleCap": s.fluid.cap,
        "simulation": t.kind(),
    })
}

// ---------------------------------------------------------------- commands
/// Apply one command to the twin. `Ok(Some(reply))` when it answers.
pub fn handle(t: &mut dyn Engine, msg: &Value, default_frame: Frame) -> Result<Option<Value>, String> {
    let ty = msg.get("type").and_then(|t| t.as_str()).ok_or("missing \"type\"")?;
    let f = frame_of(msg, default_frame);
    match ty {
        "set_target" => {
            let p = arr::<3>(msg.get("position")).map(|p| if f == Frame::Ros { frames::pos_from_ros(p) } else { p });
            let q = arr::<4>(msg.get("quaternion")).map(|q| if f == Frame::Ros { frames::quat_from_ros(q) } else { q });
            if p.is_none() && q.is_none() {
                return Err("set_target needs \"position\" [x,y,z] and/or \"quaternion\" [x,y,z,w]".into());
            }
            t.set_target(p.map(Vec3::from_array), q.map(Quat::from_array));
            Ok(None)
        }
        "set_joints" => {
            let j = arr::<5>(msg.get("joints")).ok_or("set_joints needs \"joints\": 5 angles (pan, lift, elbow, wrist_flex, wrist_roll)")?;
            let k = match msg.get("unit").and_then(|u| u.as_str()).unwrap_or("rad") {
                "rad" => 1.0,
                "deg" => std::f64::consts::PI / 180.0,
                u => return Err(format!("unknown unit {u:?} (\"rad\" or \"deg\")")),
            };
            if !t.arm().enabled {
                return Err("the arm is off (arm {\"enabled\": true})".into());
            }
            t.arm_mut().set_joints(j.map(|v| v * k));
            Ok(None)
        }
        "home" => {
            if t.arm().enabled {
                let h = t.arm().home;
                t.set_target(Some(h.pos), Some(h.quat));
            } else {
                t.set_target(Some(HOME_POS), Some(HOME_QUAT));
            }
            Ok(None)
        }
        "arm" => {
            if let Some(on) = msg.get("enabled").and_then(|v| v.as_bool()) {
                t.set_arm_enabled(on);
            }
            if let Some(c) = msg.get("constrainTarget").and_then(|v| v.as_bool()) {
                t.arm_mut().constrain_target = c;
            }
            if let Some(s) = msg.get("jointSpeed").and_then(|v| v.as_f64()) {
                t.arm_mut().joint_speed = s.max(1.0);
            }
            Ok(Some(twin_state(t, f)))
        }
        "step" => {
            let n = msg.get("frames").and_then(|n| n.as_u64()).unwrap_or(1) as usize;
            t.step_frames(n);
            Ok(Some(twin_state(t, f)))
        }
        "get_state" => Ok(Some(twin_state(t, f))),
        "eat" => {
            let g = t.eat();
            Ok(Some(json!({ "type": "eaten", "grams": g })))
        }
        "release" => {
            t.release();
            Ok(None)
        }
        "reset" => {
            t.reset();
            Ok(Some(twin_state(t, f)))
        }
        "refill" => {
            t.refill();
            Ok(None)
        }
        "set_params" => {
            let patch = msg.get("params").ok_or("set_params needs \"params\"")?;
            let mut p = t.params().clone();
            apply_params(&mut p, patch)?;
            t.set_params(p);
            Ok(Some(json!({ "type": "params", "params": params_json(t.params()) })))
        }
        "get_params" => Ok(Some(json!({ "type": "params", "params": params_json(t.params()) }))),
        other => Err(format!("unknown message type {other:?}")),
    }
}
