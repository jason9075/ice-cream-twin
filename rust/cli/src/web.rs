//! `icecream web`: the simulation runs here, browsers only draw it.
//!
//! - One thread owns the [`Twin`] and steps it in real time (60 Hz) unless paused. After every step (or
//!   command, while paused) it publishes a [`Snapshot`]: state, poses, particles and the SDF grids
//!   quantised the way the renderer's 3D textures store them.
//! - `GET /ws` is a WebSocket. Commands are the JSON messages of [`crate::protocol`] plus `config`
//!   (`stream`: "render" | "state" | "none", `frame`, `rateHz`) and `pause {paused}`. A request's `id` is
//!   echoed in its reply.
//! - A "render" client gets one binary frame per snapshot (latest wins: a slow client skips frames).
//!   Grids are sent as patches: the bounding box of what changed since *that client's* last frame.
//! - Everything else is the built page (`--dist`), so `icecream web --dist dist` is the whole app.
//!
//! Binary frame, little-endian, 4-byte aligned:
//! ```text
//! u32 magic "ICE1" | u32 seq | u32 json length, state JSON, pad
//! f32[7] spoon pose (x y z qx qy qz qw) | f32[7] ghost pose (arm goal, or the raw target) | u32 arm on
//! f32[8 * 16] link world matrices, row-major (protocol scene.arm.links order)
//! u32 n, n x (f32 x, y, z, u32 rgba) loose particles
//! u32 spilled changed (0/1) [, u32 n, n x (f32 x, y, z, u32 rgba)] crumbs on the table
//! u32 patches, each: u32 grid, u32 kind (0 distance r8, 1 colour rgba8), u32 x0 y0 z0, u32 sx sy sz, bytes, pad
//! ```

use crate::protocol::{frame_of, handle, params_json, scene_json, state_json, Frame};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State as AxState;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use icecream_core::robot::ArmState;
use icecream_core::{State, Twin};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc as tmpsc, oneshot, watch};

pub const MAGIC: u32 = u32::from_le_bytes(*b"ICE1");
const DT: f64 = 1.0 / 60.0;

pub struct GridSnap {
    pub dims: [usize; 3],
    pub dist_ver: u64,
    pub dist: Arc<Vec<u8>>,
    pub col_ver: u64,
    pub color: Arc<Vec<u8>>,
}

pub struct Snapshot {
    pub seq: u64,
    pub paused: bool,
    pub state: State,
    pub arm: ArmState,
    pub spoon: [f32; 7],
    pub ghost: [f32; 7],
    pub links: [[f32; 16]; 8],
    pub particles: Arc<Vec<u8>>,
    pub spilled_ver: u64,
    pub spilled: Arc<Vec<u8>>,
    pub grids: Vec<GridSnap>,
}

struct Cmd {
    msg: Value,
    frame: Frame,
    reply: oneshot::Sender<Result<Option<Value>, String>>,
}

struct Shared {
    cmd: mpsc::Sender<Cmd>,
    snap: watch::Receiver<Arc<Snapshot>>,
    scene: Value,
}

// ---------------------------------------------------------------- simulation thread
struct Publisher {
    seq: u64,
    grids: Vec<GridSnap>,
    spilled_ver: u64,
    spilled: Arc<Vec<u8>>,
}

fn pose7(p: [f64; 3], q: [f64; 4]) -> [f32; 7] {
    [p[0] as f32, p[1] as f32, p[2] as f32, q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
}

fn particle_bytes(pos: &[f32], rgba: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 16);
    for i in 0..n {
        for k in 0..3 {
            out.extend_from_slice(&pos[i * 3 + k].to_le_bytes());
        }
        out.extend_from_slice(&rgba[i * 4..i * 4 + 4]);
    }
    out
}

impl Publisher {
    fn new(t: &mut Twin) -> Self {
        let mut p = Publisher { seq: 0, grids: Vec::new(), spilled_ver: u64::MAX, spilled: Arc::new(Vec::new()) };
        for k in 0..4 {
            let g = if k < 3 { &mut t.sim.grids[k] } else { &mut t.sim.content };
            g.pack();
            g.dirty = false;
            g.color_dirty = false;
            p.grids.push(GridSnap { dims: [g.nx, g.ny, g.nz], dist_ver: 0, dist: Arc::new(g.bytes.clone()), col_ver: 0, color: Arc::new(g.color.clone()) });
        }
        p
    }

    fn snapshot(&mut self, t: &mut Twin, paused: bool) -> Arc<Snapshot> {
        self.seq += 1;
        for k in 0..4 {
            let g = if k < 3 { &mut t.sim.grids[k] } else { &mut t.sim.content };
            let s = &mut self.grids[k];
            if g.dirty {
                g.pack();
                g.dirty = false;
                s.dist = Arc::new(g.bytes.clone());
                s.dist_ver += 1;
            }
            if g.color_dirty {
                g.color_dirty = false;
                s.color = Arc::new(g.color.clone());
                s.col_ver += 1;
            }
        }
        let sim = &t.sim;
        if sim.spilled_version != self.spilled_ver {
            self.spilled_ver = sim.spilled_version;
            self.spilled = Arc::new(particle_bytes(&sim.spilled, &sim.spilled_rgba, sim.n_spilled));
        }
        let state = sim.state();
        let arm = t.arm.state();
        let ghost = if t.arm.enabled {
            pose7(t.arm.goal.pos.to_array(), t.arm.goal.quat.to_array())
        } else {
            pose7(sim.spoon.target_pos.to_array(), sim.spoon.target_quat.to_array())
        };
        let mut links = [[0f32; 16]; 8];
        for (k, m) in t.arm.link_matrices().iter().enumerate() {
            for (o, v) in links[k].iter_mut().zip(m.iter()) {
                *o = *v as f32;
            }
        }
        Arc::new(Snapshot {
            seq: self.seq,
            paused,
            spoon: pose7(state.spoon_position, state.spoon_quaternion),
            ghost,
            links,
            particles: Arc::new(particle_bytes(&sim.fluid.pos, &sim.fluid.rgba, sim.fluid.n)),
            spilled_ver: self.spilled_ver,
            spilled: self.spilled.clone(),
            grids: self
                .grids
                .iter()
                .map(|g| GridSnap { dims: g.dims, dist_ver: g.dist_ver, dist: g.dist.clone(), col_ver: g.col_ver, color: g.color.clone() })
                .collect(),
            state,
            arm,
        })
    }
}

fn run_sim(mut twin: Twin, rx: mpsc::Receiver<Cmd>, tx: watch::Sender<Arc<Snapshot>>, mut pub_: Publisher, mut paused: bool) {
    let dt = Duration::from_secs_f64(DT);
    let mut next = Instant::now() + dt;
    loop {
        let wait = if paused { Duration::from_millis(200) } else { next.saturating_duration_since(Instant::now()) };
        match rx.recv_timeout(wait) {
            Ok(cmd) => {
                let ty = cmd.msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let r = if ty == "pause" {
                    paused = cmd.msg.get("paused").and_then(|v| v.as_bool()).unwrap_or(true);
                    next = Instant::now() + dt;
                    Ok(Some(json!({ "type": "paused", "paused": paused })))
                } else {
                    handle(&mut twin, &cmd.msg, cmd.frame)
                };
                // while running the next tick publishes; while paused every command shows at once
                let r = r.map(|v| {
                    v.map(|mut v| {
                        if let Some(o) = v.as_object_mut() {
                            o.insert("seq".into(), json!(pub_.seq + 1)); // the snapshot that shows this
                        }
                        v
                    })
                });
                if paused {
                    let _ = tx.send(pub_.snapshot(&mut twin, paused));
                }
                let _ = cmd.reply.send(r);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
        if !paused && Instant::now() >= next {
            twin.step(DT);
            next += dt;
            if Instant::now() > next + Duration::from_millis(100) {
                next = Instant::now() + dt; // fell behind: don't try to catch up
            }
            let _ = tx.send(pub_.snapshot(&mut twin, paused));
        }
    }
}

// ---------------------------------------------------------------- frames for one client
#[derive(Default)]
struct ClientView {
    dist: Vec<Option<(u64, Arc<Vec<u8>>)>>,
    color: Vec<Option<(u64, Arc<Vec<u8>>)>>,
    spilled_ver: Option<u64>,
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f32s(b: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
}
fn pad4(b: &mut Vec<u8>) {
    while !b.len().is_multiple_of(4) {
        b.push(0);
    }
}

/// Bounding box (x0, y0, z0, x1, y1, z1 inclusive) of the voxels whose `bpp` bytes differ.
fn diff_box(a: &[u8], b: &[u8], dims: [usize; 3], bpp: usize) -> Option<[usize; 6]> {
    let [nx, ny, nz] = dims;
    let mut bx = [usize::MAX, usize::MAX, usize::MAX, 0, 0, 0];
    let row = nx * bpp;
    for z in 0..nz {
        for y in 0..ny {
            let o = (y + ny * z) * row;
            let (ra, rb) = (&a[o..o + row], &b[o..o + row]);
            if ra == rb {
                continue;
            }
            let first = ra.iter().zip(rb).position(|(p, q)| p != q).unwrap() / bpp;
            let last = (row - 1 - ra.iter().rev().zip(rb.iter().rev()).position(|(p, q)| p != q).unwrap()) / bpp;
            bx[0] = bx[0].min(first);
            bx[3] = bx[3].max(last);
            bx[1] = bx[1].min(y);
            bx[4] = bx[4].max(y);
            bx[2] = bx[2].min(z);
            bx[5] = bx[5].max(z);
        }
    }
    (bx[0] != usize::MAX).then_some(bx)
}

fn put_patch(out: &mut Vec<u8>, grid: usize, kind: u32, data: &[u8], dims: [usize; 3], bx: [usize; 6], bpp: usize) {
    let s = [bx[3] - bx[0] + 1, bx[4] - bx[1] + 1, bx[5] - bx[2] + 1];
    for v in [grid as u32, kind, bx[0] as u32, bx[1] as u32, bx[2] as u32, s[0] as u32, s[1] as u32, s[2] as u32] {
        put_u32(out, v);
    }
    for z in bx[2]..=bx[5] {
        for y in bx[1]..=bx[4] {
            let o = (bx[0] + dims[0] * (y + dims[1] * z)) * bpp;
            out.extend_from_slice(&data[o..o + s[0] * bpp]);
        }
    }
    pad4(out);
}

fn render_frame(s: &Snapshot, view: &mut ClientView) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 * 1024);
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, s.seq as u32);
    let mut st = state_json(&s.state, Some(&s.arm), Frame::Yup);
    st["seq"] = json!(s.seq);
    st["paused"] = json!(s.paused);
    let js = st.to_string();
    put_u32(&mut out, js.len() as u32);
    out.extend_from_slice(js.as_bytes());
    pad4(&mut out);
    put_f32s(&mut out, &s.spoon);
    put_f32s(&mut out, &s.ghost);
    put_u32(&mut out, u32::from(s.arm.enabled));
    for m in &s.links {
        put_f32s(&mut out, m);
    }
    put_u32(&mut out, (s.particles.len() / 16) as u32);
    out.extend_from_slice(&s.particles);
    if view.spilled_ver != Some(s.spilled_ver) {
        view.spilled_ver = Some(s.spilled_ver);
        put_u32(&mut out, 1);
        put_u32(&mut out, (s.spilled.len() / 16) as u32);
        out.extend_from_slice(&s.spilled);
    } else {
        put_u32(&mut out, 0);
    }
    if view.dist.is_empty() {
        view.dist = vec![None; s.grids.len()];
        view.color = vec![None; s.grids.len()];
    }
    let count_at = out.len();
    put_u32(&mut out, 0);
    let mut patches = 0u32;
    for (k, g) in s.grids.iter().enumerate() {
        let full = [0, 0, 0, g.dims[0] - 1, g.dims[1] - 1, g.dims[2] - 1];
        for (kind, data, ver, last, bpp) in [(0u32, &g.dist, g.dist_ver, &mut view.dist[k], 1usize), (1, &g.color, g.col_ver, &mut view.color[k], 4)] {
            let bx = match last {
                Some((v, _)) if *v == ver => continue,
                Some((_, prev)) => diff_box(prev, data, g.dims, bpp),
                None => Some(full),
            };
            if let Some(bx) = bx {
                put_patch(&mut out, k, kind, data, g.dims, bx, bpp);
                patches += 1;
            }
            *last = Some((ver, data.clone()));
        }
    }
    out[count_at..count_at + 4].copy_from_slice(&patches.to_le_bytes());
    out
}

// ---------------------------------------------------------------- WebSocket clients
#[derive(Clone, Copy, PartialEq)]
enum Stream {
    None,
    Render,
    State,
}

enum Out {
    Text(String),
    Config(Stream, Frame, f64),
}

async fn ws_handler(ws: WebSocketUpgrade, AxState(sh): AxState<Arc<Shared>>) -> impl IntoResponse {
    ws.max_message_size(1 << 20).on_upgrade(move |socket| client(socket, sh))
}

async fn client(socket: WebSocket, sh: Arc<Shared>) {
    let (mut sink, mut stream) = socket.split();
    let (out_tx, mut out_rx) = tmpsc::unbounded_channel::<Out>();
    let hello = {
        let s = sh.snap.borrow().clone();
        json!({ "type": "hello", "sim": "icecream-core", "version": env!("CARGO_PKG_VERSION"), "dt": DT,
                "paused": s.paused, "scene": sh.scene, "params": Value::Null })
    };
    // the parameters come from the simulation thread
    let params = request(&sh, json!({ "type": "get_params" }), Frame::Yup).await.ok().flatten();
    let mut hello = hello;
    hello["params"] = params.map(|p| p["params"].clone()).unwrap_or(Value::Null);
    if sink.send(Message::Text(hello.to_string())).await.is_err() {
        return;
    }

    let reader_sh = sh.clone();
    let reader_tx = out_tx.clone();
    let reader = tokio::spawn(async move {
        let mut frame = Frame::Yup;
        while let Some(Ok(msg)) = stream.next().await {
            let Message::Text(text) = msg else { continue };
            let reply = match serde_json::from_str::<Value>(&text) {
                Err(e) => Some(json!({ "type": "error", "message": format!("bad JSON: {e}") })),
                Ok(m) if m.get("type").and_then(|t| t.as_str()) == Some("config") => {
                    frame = frame_of(&m, frame);
                    let stream = match m.get("stream").and_then(|s| s.as_str()) {
                        Some("render") => Stream::Render,
                        Some("state") => Stream::State,
                        _ => Stream::None,
                    };
                    let rate = m.get("rateHz").and_then(|r| r.as_f64()).unwrap_or(30.0).clamp(1.0, 120.0);
                    let _ = reader_tx.send(Out::Config(stream, frame, rate));
                    None
                }
                Ok(m) => {
                    let id = m.get("id").cloned();
                    let r = request(&reader_sh, m, frame).await;
                    let r = match r {
                        Ok(v) => v,
                        Err(e) => Some(json!({ "type": "error", "message": e })),
                    };
                    r.map(|mut v| {
                        if let (Some(id), Some(o)) = (id, v.as_object_mut()) {
                            o.insert("id".into(), id);
                        }
                        v
                    })
                }
            };
            if let Some(r) = reply {
                let _ = reader_tx.send(Out::Text(r.to_string()));
            }
        }
    });

    let mut snap_rx = sh.snap.clone();
    let mut mode = Stream::None;
    let mut frame = Frame::Yup;
    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / 30.0));
    let mut view = ClientView::default();
    loop {
        tokio::select! {
            out = out_rx.recv() => match out {
                Some(Out::Text(t)) => { if sink.send(Message::Text(t)).await.is_err() { break; } }
                Some(Out::Config(s, f, rate)) => {
                    mode = s;
                    frame = f;
                    tick = tokio::time::interval(Duration::from_secs_f64(1.0 / rate));
                    snap_rx.mark_changed();
                }
                None => break,
            },
            r = snap_rx.changed(), if mode == Stream::Render => {
                if r.is_err() { break; }
                let s = snap_rx.borrow_and_update().clone();
                let bytes = render_frame(&s, &mut view);
                if sink.send(Message::Binary(bytes)).await.is_err() { break; }
            },
            _ = tick.tick(), if mode == Stream::State => {
                let s = snap_rx.borrow().clone();
                let mut st = state_json(&s.state, Some(&s.arm), frame);
                st["seq"] = json!(s.seq);
                if sink.send(Message::Text(st.to_string())).await.is_err() { break; }
            },
        }
    }
    reader.abort();
}

async fn request(sh: &Shared, msg: Value, frame: Frame) -> Result<Option<Value>, String> {
    let (tx, rx) = oneshot::channel();
    sh.cmd.send(Cmd { msg, frame, reply: tx }).map_err(|_| "simulation stopped".to_string())?;
    rx.await.map_err(|_| "simulation stopped".to_string())?
}

// ---------------------------------------------------------------- server
pub struct Options {
    pub port: u16,
    pub dist: Option<PathBuf>,
    pub paused: bool,
}

/// Bind, start the simulation thread, and return the address and the server future.
pub async fn start(opts: Options) -> std::io::Result<(std::net::SocketAddr, impl std::future::Future<Output = std::io::Result<()>>)> {
    let mut twin = Twin::default();
    let scene = scene_json(&twin);
    let _ = params_json(&twin.sim.params);
    let mut publisher = Publisher::new(&mut twin);
    let first = publisher.snapshot(&mut twin, opts.paused);
    let (snap_tx, snap_rx) = watch::channel(first);
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let paused = opts.paused;
    std::thread::Builder::new().name("simulation".into()).spawn(move || run_sim(twin, cmd_rx, snap_tx, publisher, paused))?;
    let shared = Arc::new(Shared { cmd: cmd_tx, snap: snap_rx, scene });
    let mut app = Router::new().route("/ws", get(ws_handler)).with_state(shared);
    if let Some(dir) = opts.dist {
        app = app.fallback_service(tower_http::services::ServeDir::new(dir));
    }
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port)).await?;
    let addr = listener.local_addr()?;
    Ok((addr, async move { axum::serve(listener, app).await }))
}
