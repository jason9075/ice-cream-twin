//! `icecream app`: the twin in a window. The simulation and the renderer share one wgpu device, so with
//! the GPU simulation the ice goes from the compute kernels to the screen without leaving the GPU. The
//! control server (`control.rs`) runs alongside, so ROS nodes and scripts drive the same simulation.

use crate::control::{self, Control, DT};
use crate::engine::{open_on, Engine};
use crate::scene::{scene_desc, Sync};
use crate::ui::{Panel, JOG};
use icecream_core::math::Vec3 as V3;
use icecream_gpu::wgpu;
use icecream_view::{Light, OrbitCamera, Renderer, ViewOptions};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

pub struct Options {
    /// the control server's port (None: no server)
    pub port: Option<u16>,
    pub paused: bool,
    pub cpu: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Orbit,
    Pan,
    Target,
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    sync: Sync,
    engine: Box<dyn Engine>,
    control: Option<(u16, Control)>,
    paused: bool,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    camera: OrbitCamera,
    light: Light,
    view: ViewOptions,
    panel: Panel,
    mods: ModifiersState,
    cursor: (f32, f32),
    drag: Option<Drag>,
    drag_plane_y: f64,
    /// where the scene is drawn: the window right of the panel (x, y, w, h in pixels)
    viewport: [f32; 4],
    last: Instant,
    acc: f64,
    fps: f64,
}

struct App {
    opts: Options,
    gfx: Option<Gfx>,
}

pub fn run(opts: Options) {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { opts, gfx: None };
    event_loop.run_app(&mut app).expect("run");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let window = Arc::new(el.create_window(Window::default_attributes().with_title("Ice Cream Twin").with_inner_size(winit::dpi::LogicalSize::new(1400, 860))).expect("window"));
        self.gfx = Some(Gfx::new(window, &self.opts));
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(g) = self.gfx.as_mut() else { return };
        let consumed = g.egui_state.on_window_event(&g.window, &event).consumed;
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => g.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                g.frame();
                g.window.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => g.mods = m.state(),
            WindowEvent::CursorMoved { position, .. } => g.cursor_moved(position.x as f32, position.y as f32),
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Released {
                    g.drag = None;
                } else if !consumed && !g.egui_ctx.is_pointer_over_area() {
                    g.mouse_down(button);
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !consumed && !g.egui_ctx.is_pointer_over_area() => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 100.0,
                    MouseScrollDelta::PixelDelta(p) => -p.y as f32,
                };
                if g.mods.control_key() {
                    let p = g.engine.sim().spoon.target_pos;
                    g.engine.set_target(Some(V3::new(p.x, p.y - dy.signum() as f64 * 0.004, p.z)), None);
                } else {
                    g.camera.zoom(dy);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if g.egui_ctx.wants_keyboard_input() {
                    return;
                }
                let down = event.state == ElementState::Pressed;
                match &event.logical_key {
                    Key::Character(c) => {
                        let ch = c.chars().next().unwrap_or(' ').to_ascii_lowercase();
                        if let Some(k) = JOG.iter().position(|j| j.1 == ch) {
                            if down && !g.mods.control_key() {
                                g.panel.held_keys.insert(k);
                            } else {
                                g.panel.held_keys.remove(&k);
                            }
                        } else if down && !event.repeat {
                            match ch {
                                'e' => g.panel.eat(g.engine.as_mut()),
                                'h' => g.panel.home(g.engine.as_mut()),
                                _ => {}
                            }
                        }
                    }
                    Key::Named(NamedKey::Space) if down && !event.repeat => g.paused = !g.paused,
                    _ => {}
                }
            }
            WindowEvent::Focused(false) => g.panel.held_keys.clear(),
            _ => {}
        }
    }
}

impl Gfx {
    fn new(window: Arc<Window>, opts: &Options) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no GPU adapter that can draw to this window");
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("icecream"), ..Default::default() }, None)).expect("device");
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        device.on_uncaptured_error(Box::new(|e| eprintln!("wgpu: {e}")));
        // the shaders do their own gamma: draw into a linear (non-sRGB) format
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, // vsync: no point drawing faster than the screen
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let engine = open_on(opts.cpu, Some((device.clone(), queue.clone(), name)));
        let renderer = Renderer::new(device.clone(), queue.clone(), format, &scene_desc(engine.as_ref()));
        let control = opts.port.and_then(|port| match control::spawn(port, engine.as_ref(), opts.paused) {
            Ok((addr, c)) => {
                eprintln!("control: ws://localhost:{}/ws", addr.port());
                Some((addr.port(), c))
            }
            Err(e) => {
                eprintln!("control server not started: {e}");
                None
            }
        });
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);
        window.request_redraw();
        Gfx {
            window,
            surface,
            config,
            device,
            queue,
            renderer,
            sync: Sync::default(),
            engine,
            control,
            paused: opts.paused,
            egui_ctx,
            egui_state,
            egui_renderer,
            camera: OrbitCamera::default(),
            light: Light::default(),
            view: ViewOptions::default(),
            panel: Panel::default(),
            mods: ModifiersState::empty(),
            cursor: (0.0, 0.0),
            drag: None,
            drag_plane_y: 0.0,
            viewport: [0.0, 0.0, size.width as f32, size.height as f32],
            last: Instant::now(),
            acc: 0.0,
            fps: 60.0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn mouse_down(&mut self, button: MouseButton) {
        self.drag = match button {
            MouseButton::Left if self.mods.control_key() => {
                self.drag_plane_y = self.engine.sim().spoon.target_pos.y;
                Some(Drag::Target)
            }
            MouseButton::Left if self.mods.shift_key() => Some(Drag::Pan),
            MouseButton::Left => Some(Drag::Orbit),
            MouseButton::Right => Some(Drag::Pan),
            _ => None,
        };
        if self.drag == Some(Drag::Target) {
            self.drag_target();
        }
    }

    fn cursor_moved(&mut self, x: f32, y: f32) {
        let (dx, dy) = (x - self.cursor.0, y - self.cursor.1);
        self.cursor = (x, y);
        match self.drag {
            Some(Drag::Orbit) => self.camera.orbit(dx, dy),
            Some(Drag::Pan) => self.camera.pan(dx, dy, self.viewport[3]),
            Some(Drag::Target) => self.drag_target(),
            None => {}
        }
    }

    /// Move the spoon target to where the mouse ray meets its horizontal plane.
    fn drag_target(&mut self) {
        let v = self.viewport;
        let (o, d) = self.camera.ray(self.cursor.0 - v[0], self.cursor.1 - v[1], v[2], v[3]);
        if d.y.abs() < 1e-4 {
            return;
        }
        let t = (self.drag_plane_y as f32 - o.y) / d.y;
        if t > 0.0 {
            let p = o + d * t;
            self.engine.set_target(Some(V3::new(p.x as f64, self.drag_plane_y, p.z as f64)), None);
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f64().min(0.1);
        self.last = now;
        self.fps += (1.0 / dt.max(1e-3) - self.fps) * 0.05;

        // commands from the control server, the jog, then the simulation at a fixed 60 Hz
        if let Some((_, c)) = &mut self.control {
            let before = c.paused;
            c.poll(self.engine.as_mut());
            if c.paused != before {
                self.paused = c.paused; // a pause command
            }
        }
        self.panel.apply_jog(self.engine.as_mut(), dt);
        if !self.paused {
            self.acc += dt;
            let mut steps = 0;
            while self.acc >= DT && steps < 4 {
                self.engine.step(DT);
                self.acc -= DT;
                steps += 1;
            }
            if steps == 4 {
                self.acc = 0.0; // fell behind: don't try to catch up
            }
        }
        if let Some((_, c)) = &mut self.control {
            c.paused = self.paused;
            c.publish(self.engine.as_ref());
        }

        // the panel
        let raw = self.egui_state.take_egui_input(&self.window);
        let port = self.control.as_ref().map(|c| c.0);
        let (engine, view, paused, panel, fps) = (&mut self.engine, &mut self.view, &mut self.paused, &mut self.panel, self.fps);
        let out = self.egui_ctx.run(raw, |ctx| panel.show(ctx, engine.as_mut(), view, paused, fps, port));
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        let free = self.egui_ctx.available_rect();
        let ppp = out.pixels_per_point;
        self.viewport = [free.min.x * ppp, free.min.y * ppp, free.width() * ppp, free.height() * ppp];
        let prims = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);

        // draw
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let target = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.sync.frame(&mut self.renderer, self.engine.as_mut(), &mut enc);
        let (w, h) = (self.config.width, self.config.height);
        self.renderer.render(&mut enc, &target, w, h, Some(self.viewport), &self.camera, &self.light, &self.view);

        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: out.pixels_per_point };
        for (id, delta) in &out.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut enc, &prims, &screen);
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("panel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer.render(&mut pass.forget_lifetime(), &prims, &screen);
        }
        for id in &out.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.queue.submit([enc.finish()]);
        frame.present();
    }
}
