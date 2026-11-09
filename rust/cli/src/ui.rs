//! The side panel (egui): actions, jog, go to pose, readouts, every simulation parameter, the SO-101's
//! joints, and the View options. Anything that changes the simulation goes through the same commands as
//! the control server (`protocol::handle`).

use crate::engine::Engine;
use crate::protocol::{handle, Frame};
use egui::{Color32, RichText};
use icecream_core::math::{Quat, Vec3};
use icecream_core::params::FillShape;
use icecream_view::{ParticleMode, Shading, ViewOptions};
use serde_json::{json, Value};
use std::collections::{HashSet, BTreeMap};

/// Jog axes: (label, key, linear?, axis, sign)
pub const JOG: [(&str, char, bool, usize, f64); 12] = [
    ("X−", 'a', true, 0, -1.0),
    ("X+", 'd', true, 0, 1.0),
    ("Y+", 'r', true, 1, 1.0),
    ("Y−", 'f', true, 1, -1.0),
    ("Z−", 'w', true, 2, -1.0),
    ("Z+", 's', true, 2, 1.0),
    ("Rx−", 'u', false, 0, -1.0),
    ("Rx+", 'o', false, 0, 1.0),
    ("Ry−", 'j', false, 1, -1.0),
    ("Ry+", 'l', false, 1, 1.0),
    ("Rz−", 'k', false, 2, -1.0),
    ("Rz+", 'i', false, 2, 1.0),
];
/// how far the target may run ahead of the spoon while jogging (m, rad)
const LEAD_LIN: f64 = 0.012;
const LEAD_ANG: f64 = 20.0 * std::f64::consts::PI / 180.0;

fn slerp(a: Quat, b: Quat, t: f64) -> Quat {
    let mut d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    let b = if d < 0.0 {
        d = -d;
        Quat { x: -b.x, y: -b.y, z: -b.z, w: -b.w }
    } else {
        b
    };
    if d > 0.9995 {
        return Quat { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t, w: a.w + (b.w - a.w) * t }.normalize();
    }
    let th = d.clamp(-1.0, 1.0).acos();
    let (s0, s1) = (((1.0 - t) * th).sin() / th.sin(), (t * th).sin() / th.sin());
    Quat { x: a.x * s0 + b.x * s1, y: a.y * s0 + b.y * s1, z: a.z * s0 + b.z * s1, w: a.w * s0 + b.w * s1 }
}

pub struct Panel {
    pub sections: BTreeMap<String, bool>,
    pub tool_frame: bool,
    /// jog axes held (keys or buttons)
    pub held_keys: HashSet<usize>,
    held_buttons: HashSet<usize>,
    pose: [f64; 6],
    pub toast: Option<(String, std::time::Instant)>,
}

impl Default for Panel {
    fn default() -> Self {
        Panel { sections: BTreeMap::new(), tool_frame: false, held_keys: HashSet::new(), held_buttons: HashSet::new(), pose: [0.0, 170.0, 20.0, 0.0, 0.0, 0.0], toast: None }
    }
}

/// Run a protocol command from the panel; errors become a toast.
fn cmd(t: &mut dyn Engine, msg: Value) -> Option<Value> {
    match handle(t, &msg, Frame::Yup) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            None
        }
    }
}

impl Panel {
    pub fn eat(&mut self, t: &mut dyn Engine) {
        let g = cmd(t, json!({ "type": "eat" })).and_then(|v| v["grams"].as_f64()).unwrap_or(0.0);
        self.say(if g > 0.05 { format!("Yum! {g:.1} g eaten") } else { "Spoon is empty".into() });
    }

    pub fn home(&mut self, t: &mut dyn Engine) {
        cmd(t, json!({ "type": "home" }));
    }

    fn say(&mut self, s: String) {
        self.toast = Some((s, std::time::Instant::now()));
    }

    /// Move the target by the held jog axes over `dt` seconds (on a short leash, so a blocked spoon
    /// doesn't wind up a huge error).
    pub fn apply_jog(&self, t: &mut dyn Engine, dt: f64) {
        let held: Vec<usize> = self.held_keys.union(&self.held_buttons).copied().collect();
        if held.is_empty() {
            return;
        }
        let sp = &t.sim().spoon;
        let p = t.params().spoon.clone();
        let (mut lin, mut ang) = ([0.0f64; 3], [0.0f64; 3]);
        for k in held {
            let (_, _, is_lin, axis, sign) = JOG[k];
            if is_lin {
                lin[axis] += sign;
            } else {
                ang[axis] += sign;
            }
        }
        let (mut lin, mut ang) = (Vec3::from_array(lin), Vec3::from_array(ang));
        if self.tool_frame {
            lin = lin.apply_quat(sp.quat);
            ang = ang.apply_quat(sp.quat);
        }
        let mut tp = sp.target_pos.add(lin.scale(p.jog_lin_speed * dt));
        let lead = tp.sub(sp.pos);
        if lead.length() > LEAD_LIN {
            tp = sp.pos.add(lead.scale(LEAD_LIN / lead.length()));
        }
        let mut tq = sp.target_quat;
        if ang.length() > 0.0 {
            let a = p.jog_ang_speed.to_radians() * dt;
            tq = Quat::from_axis_angle(ang.scale(1.0 / ang.length()), a).mul(tq);
            let err = sp.quat.angle_to(tq);
            if err > LEAD_ANG {
                tq = slerp(sp.quat, tq, LEAD_ANG / err);
            }
        }
        t.set_target(Some(tp), Some(tq));
    }

    pub fn show(&mut self, ctx: &egui::Context, t: &mut dyn Engine, view: &mut ViewOptions, paused: &mut bool, fps: f64, control: Option<u16>) {
        egui::SidePanel::left("panel").resizable(true).default_width(330.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("🍨 Ice Cream Twin");
                let recording = crate::recording::active();
                if ui.button(if recording { "Stop recording operations" } else { "Record operations" }).clicked() {
                    if recording {
                        crate::recording::stop();
                    } else {
                        crate::recording::start(json!({
                            "state":crate::protocol::twin_state(t, Frame::Yup),
                            "params":crate::protocol::params_json(t.params()),
                            "paused":*paused,
                            "note":"Diagnostic journal; not a complete simulation checkpoint"
                        }));
                    }
                }
                let status = crate::recording::status();
                if !status.is_empty() { ui.small(status); }
                if let Some(path) = crate::recording::path() {
                    if ui.button("Copy recording path").clicked() {
                        ctx.copy_text(path.display().to_string());
                        self.say("Recording path copied".into());
                    }
                }

                ui.label(RichText::new(format!(
                    "simulation: {}{}",
                    t.kind(),
                    control.map(|p| format!(" · control ws://localhost:{p}/ws")).unwrap_or_default()
                )).small());
                ui.horizontal(|ui| {
                    if ui.button("Eat 😋").clicked() {
                        self.eat(t);
                    }
                    if ui.button("Tip off").on_hover_text("tip the scoop off").clicked() {
                        cmd(t, json!({ "type": "release" }));
                    }
                    if ui.button("Home").clicked() {
                        self.home(t);
                    }
                    if ui.button("Reset").clicked() {
                        cmd(t, json!({ "type": "reset" }));
                    }
                    ui.checkbox(paused, "pause");
                });
                if let Some((msg, at)) = &self.toast {
                    if at.elapsed().as_secs_f64() < 2.0 {
                        ui.label(RichText::new(msg).strong());
                    }
                }
                ui.separator();
                self.jog_ui(ui, t);
                ui.separator();
                self.pose_ui(ui, t);
                ui.separator();
                status_ui(ui, t, fps);
                ui.separator();
                params_ui(ui, t, &mut self.sections);
                arm_ui(ui, t, &mut self.sections);
                view_ui(ui, view, &mut self.sections);
                ui.separator();
                ui.label(RichText::new(
                    "Mouse: left-drag orbit · right-drag or Shift+drag pan · wheel zoom · Ctrl+drag moves the target \
                     on its horizontal plane · Ctrl+wheel its height. Keys: A/D X · R/F Y · W/S Z · U/O Rx · J/L Ry · \
                     K/I Rz · E eat · B home · Space pause.",
                ).small().weak());
            });
        });
    }

    fn jog_ui(&mut self, ui: &mut egui::Ui, t: &mut dyn Engine) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Jog").strong());
            ui.selectable_value(&mut self.tool_frame, false, "World");
            ui.selectable_value(&mut self.tool_frame, true, "Tool");
        });
        self.held_buttons.clear();
        for row in [0..6, 6..12] {
            ui.horizontal(|ui| {
                for k in row {
                    let (label, key, ..) = JOG[k];
                    let on = self.held_keys.contains(&k);
                    let b = egui::Button::new(format!("{label} {}", key.to_ascii_uppercase())).min_size(egui::vec2(46.0, 24.0));
                    let b = if on { b.fill(Color32::from_rgb(70, 110, 170)) } else { b };
                    if ui.add(b).is_pointer_button_down_on() {
                        self.held_buttons.insert(k);
                    }
                }
            });
        }
        let mut p = t.params().clone();
        let mut lin = p.spoon.jog_lin_speed * 1000.0;
        let mut ang = p.spoon.jog_ang_speed;
        let a = ui.add(egui::Slider::new(&mut lin, 5.0..=150.0).text("lin (mm/s)"));
        let b = ui.add(egui::Slider::new(&mut ang, 5.0..=180.0).text("ang (°/s)"));
        if a.changed() || b.changed() {
            p.spoon.jog_lin_speed = lin / 1000.0;
            p.spoon.jog_ang_speed = ang;
            t.set_params(p);
        }
    }

    fn pose_ui(&mut self, ui: &mut egui::Ui, t: &mut dyn Engine) {
        ui.label(RichText::new("Go to pose (mm / deg, XYZ Euler)").strong());
        ui.horizontal(|ui| {
            for (k, n) in ["x", "y", "z", "rx", "ry", "rz"].iter().enumerate() {
                ui.label(*n);
                ui.add(egui::DragValue::new(&mut self.pose[k]).speed(1.0).max_decimals(0));
                if k == 2 {
                    ui.end_row();
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Go").clicked() {
                let p = &self.pose;
                let q = Quat::from_euler(p[3].to_radians(), p[4].to_radians(), p[5].to_radians());
                t.set_target(Some(Vec3::new(p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0)), Some(q));
            }
            if ui.button("Use current").clicked() {
                let s = t.state();
                for k in 0..3 {
                    self.pose[k] = (s.spoon_position[k] * 1000.0).round();
                    self.pose[3 + k] = s.spoon_euler_xyz_deg[k].round();
                }
            }
        });
    }
}

fn status_ui(ui: &mut egui::Ui, t: &mut dyn Engine, fps: f64) {
    let s = t.state();
    ui.horizontal(|ui| {
        let chip = |ui: &mut egui::Ui, on: bool, label: &str| {
            let c = if on { Color32::from_rgb(230, 160, 40) } else { Color32::GRAY };
            ui.label(RichText::new(label).color(c));
        };
        ui.label(RichText::new("Status").strong());
        chip(ui, s.contact, "contact");
        chip(ui, s.blocked, "blocked");
        chip(ui, s.cut_force_n > 0.5, "cutting");
    });
    egui::Grid::new("pose readout").striped(true).show(ui, |ui| {
        for h in ["", "x", "y", "z", "rx", "ry", "rz"] {
            ui.label(h);
        }
        ui.end_row();
        let tq = Quat::from_array(s.target_quaternion);
        let te = tq.to_euler_xyz();
        for (name, p, e) in [("actual", s.spoon_position, s.spoon_euler_xyz_deg), ("target", s.target_position, [te[0].to_degrees(), te[1].to_degrees(), te[2].to_degrees()])] {
            ui.label(name);
            for v in p {
                ui.label(format!("{:.1}", v * 1000.0));
            }
            for v in e {
                ui.label(format!("{v:.1}"));
            }
            ui.end_row();
        }
    });
    egui::Grid::new("readout").num_columns(4).show(ui, |ui| {
        let g = |v: f64| format!("{v:.1} g");
        let rows = [
            ("on spoon", g(s.scooped_grams)),
            ("eaten", g(s.eaten_grams)),
            ("spilled", g(s.spilled_grams)),
            ("cut force", format!("{:.1} N", s.cut_force_n)),
            ("speed scale", format!("{:.0} %", s.speed_scale * 100.0)),
            ("tilt", format!("{:.0}°", s.tilt_deg)),
            ("vanilla", g(s.bowl_grams[0])),
            ("strawberry", g(s.bowl_grams[1])),
            ("mint", g(s.bowl_grams[2])),
            ("particles", s.particles.to_string()),
            ("fps", format!("{fps:.0}")),
        ];
        for (k, (name, v)) in rows.iter().enumerate() {
            ui.label(RichText::new(*name).weak());
            ui.label(v);
            if k % 2 == 1 {
                ui.end_row();
            }
        }
    });
}

fn params_ui(ui: &mut egui::Ui, t: &mut dyn Engine, sections: &mut BTreeMap<String, bool>) {
    let mut p = t.params().clone();
    let before = format!("{p:?}");
    section(ui, sections, "Fluid (loose ice cream)", false, |ui| {
        let f = &mut p.fluid;
        ui.add(egui::Slider::new(&mut f.particle_radius, 0.0008..=0.0025).text("particle radius (m)"));
        ui.add(egui::Slider::new(&mut f.viscosity, 0.0..=1.0).text("viscosity (XSPH)"));
        ui.add(egui::Slider::new(&mut f.cohesion, 0.0..=1.0).text("cohesion"));
        ui.add(egui::Slider::new(&mut f.lump_stiffness, 0.0..=1.0).text("lump stiffness (0 = flows)"));
        ui.add(egui::Slider::new(&mut f.lump_yield, 0.0..=4.0).text("lump yield (× radius)"));
        ui.add(egui::Slider::new(&mut f.friction, 0.0..=1.0).text("friction"));
        ui.add(egui::Slider::new(&mut f.damping, 0.0..=5.0).text("damping (1/s)"));
        ui.add(egui::Slider::new(&mut f.gravity, 0.0..=20.0).text("gravity (m/s²)"));
        ui.add(egui::Slider::new(&mut f.iterations, 1..=8).text("solver iterations"));
        ui.add(egui::Slider::new(&mut f.substeps, 1..=6).text("substeps"));
        ui.add(egui::Slider::new(&mut f.kernel_scale, 1.4..=2.5).text("kernel scale"));
        ui.add(egui::Slider::new(&mut f.max_particles, 500..=15000).text("max particles"));
        ui.label(RichText::new("Freezing (particle → solid)").strong());
        ui.add(egui::Slider::new(&mut f.stop_speed, 0.005..=0.15).text("rest speed (m/s)"));
        ui.add(egui::Slider::new(&mut f.rest_frames, 1..=60).text("rest frames"));
        ui.add(egui::Slider::new(&mut f.stamp_scale, 0.8..=2.5).text("stamp radius ×"));
        ui.add(egui::Slider::new(&mut f.stamp_smooth, 0.0..=0.004).text("stamp smoothing (m)"));
        ui.checkbox(&mut f.stamp_on_contact, "freeze on first contact");
    });
    let mut refill = false;
    section(ui, sections, "Ice cream", false, |ui| {
        let i = &mut p.ice;
        ui.add(egui::Slider::new(&mut i.hardness, 0.0..=120.0).text("hardness (kPa)"));
        ui.checkbox(&mut i.accretion, "scoop curls (accretion)");
        ui.add(egui::Slider::new(&mut i.accretion_dist, 0.0002..=0.003).text("accretion step (m)"));
        ui.add(egui::Slider::new(&mut i.capacity, 5.0..=60.0).text("capacity (cm³)"));
        ui.add(egui::Slider::new(&mut i.release_angle, 60.0..=180.0).text("slide-off tilt (°)"));
        ui.checkbox(&mut i.displaced_to_particles, "displaced → crumbs");
        ui.add(egui::Slider::new(&mut i.plow, 0.0..=1.0).text("plowed aside (rest: crumbs)"));
        ui.add(egui::Slider::new(&mut i.smoothing, 0.0..=1.0).text("scoop smoothing"));
        ui.add(egui::Slider::new(&mut i.bowl_smoothing, 0.0..=1.0).text("bowl smoothing"));
        ui.checkbox(&mut i.smooth_cuts, "smooth cut edges too");
        ui.add(egui::Slider::new(&mut i.density, 0.3..=1.2).text("density (g/cm³)"));
        egui::ComboBox::from_label("refill shape").selected_text(format!("{:?}", i.shape)).show_ui(ui, |ui| {
            ui.selectable_value(&mut i.shape, FillShape::Scoops, "Scoops");
            ui.selectable_value(&mut i.shape, FillShape::Flat, "Flat");
            ui.selectable_value(&mut i.shape, FillShape::Mound, "Mound");
        });
        refill = ui.button("Refill bowls").clicked();
    });
    section(ui, sections, "Spoon controller", false, |ui| {
        let s = &mut p.spoon;
        ui.add(egui::Slider::new(&mut s.max_lin_speed, 0.01..=0.5).text("max speed (m/s)"));
        ui.add(egui::Slider::new(&mut s.max_ang_speed, 10.0..=360.0).text("max ang speed (°/s)"));
        ui.add(egui::Slider::new(&mut s.max_cut_force, 1.0..=100.0).text("max cut force (N)"));
    });
    if format!("{p:?}") != before {
        t.set_params(p);
    }
    if refill {
        cmd(t, json!({ "type": "refill" }));
    }
}

fn arm_ui(ui: &mut egui::Ui, t: &mut dyn Engine, sections: &mut BTreeMap<String, bool>) {
    section(ui, sections, "Robot arm (SO-101)", true, |ui| {
        let (mut on, mut limit) = (t.arm().enabled, t.arm().constrain_target);
        if ui.checkbox(&mut on, "arm holds the spoon").changed() {
            cmd(t, json!({ "type": "arm", "enabled": on }));
        }
        if ui.checkbox(&mut limit, "limit target to reach").changed() {
            cmd(t, json!({ "type": "arm", "constrainTarget": limit }));
        }
        let st = t.arm().state();
        // joint limits (deg) of the SO-101 URDF
        let limits = [(-110.0, 110.0), (-100.0, 100.0), (-96.8, 96.8), (-95.0, 95.0), (-157.2, 162.8)];
        let mut q = st.joints_deg;
        let mut changed = false;
        for k in 0..5 {
            changed |= ui.add_enabled(on, egui::Slider::new(&mut q[k], limits[k].0..=limits[k].1).text(format!("{} (°)", st.joint_names[k]))).changed();
        }
        if changed {
            cmd(t, json!({ "type": "set_joints", "joints": q, "unit": "deg" }));
        }
        ui.label(format!("tracking error {:.2} mm / {:.2}°", st.tracking_error_mm, st.tracking_error_deg));
    });
}

fn view_ui(ui: &mut egui::Ui, v: &mut ViewOptions, sections: &mut BTreeMap<String, bool>) {
    section(ui, sections, "View", false, |ui| {
        egui::ComboBox::from_label("loose particles").selected_text(format!("{:?}", v.particles)).show_ui(ui, |ui| {
            for (m, n) in [(ParticleMode::Surface, "surface"), (ParticleMode::Balls, "balls"), (ParticleMode::Both, "both"), (ParticleMode::Hidden, "hidden")] {
                ui.selectable_value(&mut v.particles, m, n);
            }
        });
        ui.label(RichText::new("Particle surface (× particle radius)").strong());
        ui.add(egui::Slider::new(&mut v.surface_radius, 0.5..=3.0).text("radius"));
        ui.add(egui::Slider::new(&mut v.surface_kernel, 1.5..=6.0).text("blend kernel"));
        ui.add(egui::Slider::new(&mut v.surface_voxel, 0.3..=2.0).text("voxel"));
        ui.add(egui::Slider::new(&mut v.surface_smooth, 0.0..=4.0).text("smoothing"));
        ui.label(RichText::new("Show").strong());
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut v.crumbs, "crumbs");
            ui.checkbox(&mut v.bowl_ice, "ice in bowls");
            ui.checkbox(&mut v.spoon_ice, "ice on spoon");
            ui.checkbox(&mut v.bowls, "bowls");
            ui.checkbox(&mut v.spoon, "spoon");
            ui.checkbox(&mut v.arm, "SO-101");
            ui.checkbox(&mut v.table, "table");
            ui.checkbox(&mut v.ghost, "target ghost");
            ui.checkbox(&mut v.shadows, "shadows");
        });
        ui.label(RichText::new("Debug").strong());
        egui::ComboBox::from_label("shading").selected_text(format!("{:?}", v.shading)).show_ui(ui, |ui| {
            for (m, n) in [(Shading::Shaded, "shaded"), (Shading::Normals, "normals"), (Shading::Albedo, "albedo (unlit)"), (Shading::MarchSteps, "march steps")] {
                ui.selectable_value(&mut v.shading, m, n);
            }
        });
        ui.checkbox(&mut v.ice_matte, "matte ice cream (off: the older glossy look)");
        ui.checkbox(&mut v.tint_fluid, "tint particle surface");
        ui.checkbox(&mut v.grid_boxes, "grid bounds");
    });
}

/// Persist only the header's open state, not egui's other widget memory.
fn section(ui: &mut egui::Ui, sections: &mut BTreeMap<String, bool>, label: &str, default: bool, body: impl FnOnce(&mut egui::Ui)) {
    let open = sections.get(label).copied().unwrap_or(default);
    let response = egui::CollapsingHeader::new(label).default_open(open).show(ui, body);
    if let Some(state) = egui::collapsing_header::CollapsingState::load(ui.ctx(), response.header_response.id) {
        sections.insert(label.to_owned(), state.is_open());
    }
}

#[cfg(test)]
mod preference_tests {
    use super::*;

    #[test]
    fn fresh_ui_restores_sections_independently_of_default_open() {
        let ctx = egui::Context::default();
        let mut sections = BTreeMap::from([("View".into(), true), ("Robot arm (SO-101)".into(), false)]);
        let (mut view_body, mut arm_body) = (false, false);
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                section(ui, &mut sections, "View", false, |_| view_body = true);
                section(ui, &mut sections, "Robot arm (SO-101)", true, |_| arm_body = true);
                section(ui, &mut sections, "Ice cream", false, |_| {});
            });
        });
        assert!(view_body);
        assert!(!arm_body);
        assert_eq!(sections["View"], true);
        assert_eq!(sections["Robot arm (SO-101)"], false);
        assert_eq!(sections["Ice cream"], false);
    }
}
