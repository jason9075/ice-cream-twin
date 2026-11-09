//! The control server: the JSON commands of [`crate::protocol`] and a state stream over a WebSocket
//! (`ws://host:port/ws`), for ROS nodes, scripts and test rigs. Nothing is drawn here.
//!
//! Whoever owns the simulation drives it: the app's window loop (`icecream app`) or the headless loop
//! ([`start`], `icecream server`). Each frame it calls [`Control::poll`] (apply what arrived) and
//! [`Control::publish`] (the state the stream sends).
//!
//! - On connect: `{"type":"hello", "scene":..., "params":..., "paused":..., "dt":...}`.
//! - `{"type":"config", "stream":"state"|"none", "frame":"yup"|"ros", "rateHz":30}` per client.
//! - `{"type":"pause", "paused":true}` stops the real-time loop (`step` still works).
//! - A request's `id` is echoed in its reply.

use crate::engine::{open, Engine};
use crate::protocol::{frame_of, handle, scene_json, state_json_kind, Frame};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State as AxState;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use icecream_core::robot::ArmState;
use icecream_core::State;
use serde_json::{json, Value};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc as tmpsc, oneshot, watch};

pub const DT: f64 = 1.0 / 60.0;

/// What the state stream sends.
pub struct Snapshot {
    pub seq: u64,
    pub kind: &'static str,
    pub paused: bool,
    pub state: State,
    pub arm: ArmState,
}

impl Snapshot {
    fn of(t: &dyn Engine, seq: u64, paused: bool) -> Self {
        Snapshot { seq, kind: t.kind(), paused, state: t.state(), arm: t.arm().state() }
    }
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

/// The simulation owner's end.
pub struct Control {
    rx: mpsc::Receiver<Cmd>,
    tx: watch::Sender<Arc<Snapshot>>,
    pub seq: u64,
    pub paused: bool,
}

impl Control {
    fn apply(&mut self, cmd: Cmd, t: &mut dyn Engine) {
        let ty = cmd.msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let r = if ty == "pause" {
            self.paused = cmd.msg.get("paused").and_then(|v| v.as_bool()).unwrap_or(true);
            Ok(Some(json!({ "type": "paused", "paused": self.paused })))
        } else {
            handle(t, &cmd.msg, cmd.frame)
        };
        let r = r.map(|v| {
            v.map(|mut v| {
                if let Some(o) = v.as_object_mut() {
                    o.insert("seq".into(), json!(self.seq + 1)); // the snapshot that shows this
                }
                v
            })
        });
        let _ = cmd.reply.send(r);
    }

    /// Apply every command that arrived; true if there were any.
    pub fn poll(&mut self, t: &mut dyn Engine) -> bool {
        let mut any = false;
        while let Ok(cmd) = self.rx.try_recv() {
            self.apply(cmd, t);
            any = true;
        }
        any
    }

    /// Wait up to `timeout` for a command and apply it (the headless loop). False when the server is gone.
    fn wait(&mut self, t: &mut dyn Engine, timeout: Duration) -> Result<bool, ()> {
        match self.rx.recv_timeout(timeout) {
            Ok(cmd) => {
                self.apply(cmd, t);
                Ok(true)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(false),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(()),
        }
    }

    pub fn publish(&mut self, t: &dyn Engine) {
        self.seq += 1;
        let _ = self.tx.send(Arc::new(Snapshot::of(t, self.seq, self.paused)));
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Stream {
    None,
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
    let paused = sh.snap.borrow().paused;
    // the parameters come from the simulation's owner
    let params = request(&sh, json!({ "type": "get_params" }), Frame::Yup).await.ok().flatten();
    let hello = json!({ "type": "hello", "sim": "icecream-core", "version": env!("CARGO_PKG_VERSION"), "dt": DT,
                        "paused": paused, "scene": sh.scene, "params": params.map(|p| p["params"].clone()).unwrap_or(Value::Null) });
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
                    let stream = if m.get("stream").and_then(|s| s.as_str()) == Some("state") { Stream::State } else { Stream::None };
                    let rate = m.get("rateHz").and_then(|r| r.as_f64()).unwrap_or(30.0).clamp(1.0, 120.0);
                    let _ = reader_tx.send(Out::Config(stream, frame, rate));
                    None
                }
                Ok(m) => {
                    let id = m.get("id").cloned();
                    let r = match request(&reader_sh, m, frame).await {
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

    let snap_rx = sh.snap.clone();
    let mut mode = Stream::None;
    let mut frame = Frame::Yup;
    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / 30.0));
    loop {
        tokio::select! {
            out = out_rx.recv() => match out {
                Some(Out::Text(t)) => { if sink.send(Message::Text(t)).await.is_err() { break; } }
                Some(Out::Config(s, f, rate)) => {
                    mode = s;
                    frame = f;
                    tick = tokio::time::interval(Duration::from_secs_f64(1.0 / rate));
                }
                None => break,
            },
            _ = tick.tick(), if mode == Stream::State => {
                let s = snap_rx.borrow().clone();
                let mut st = state_json_kind(&s.state, Some(&s.arm), frame, s.kind);
                st["seq"] = json!(s.seq);
                st["paused"] = json!(s.paused);
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

/// Bind `port` (0: any free one) for the simulation `t`. Returns the address, the owner's end and the
/// server future (run it on a tokio runtime).
pub async fn bind(port: u16, t: &dyn Engine, paused: bool) -> std::io::Result<(std::net::SocketAddr, Control, impl std::future::Future<Output = std::io::Result<()>>)> {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (snap_tx, snap_rx) = watch::channel(Arc::new(Snapshot::of(t, 0, paused)));
    let shared = Arc::new(Shared { cmd: cmd_tx, snap: snap_rx, scene: scene_json(t) });
    let app = Router::new().route("/ws", get(ws_handler)).with_state(shared);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    let addr = listener.local_addr()?;
    Ok((addr, Control { rx: cmd_rx, tx: snap_tx, seq: 0, paused }, async move { axum::serve(listener, app).await }))
}

/// Start the control server on its own thread (with its own tokio runtime) for a simulation this thread
/// keeps driving (the app). Returns the address and the owner's end.
pub fn spawn(port: u16, t: &dyn Engine, paused: bool) -> std::io::Result<(std::net::SocketAddr, Control)> {
    let rt = tokio::runtime::Runtime::new()?;
    let (addr, control, server) = rt.block_on(bind(port, t, paused))?;
    std::thread::Builder::new().name("control server".into()).spawn(move || {
        if let Err(e) = rt.block_on(server) {
            eprintln!("control server: {e}");
        }
    })?;
    Ok((addr, control))
}

pub struct Options {
    pub port: u16,
    pub paused: bool,
    /// run the simulation on the CPU even when there is a GPU
    pub cpu: bool,
}

/// Headless: the simulation on its own thread in real time (60 Hz unless paused), controlled over the
/// WebSocket. Returns the address and the server future.
pub async fn start(opts: Options) -> std::io::Result<(std::net::SocketAddr, impl std::future::Future<Output = std::io::Result<()>>)> {
    let mut twin = open(opts.cpu);
    let (addr, mut control, server) = bind(opts.port, twin.as_ref(), opts.paused).await?;
    std::thread::Builder::new().name("simulation".into()).spawn(move || {
        let dt = Duration::from_secs_f64(DT);
        let mut next = Instant::now() + dt;
        loop {
            let wait = if control.paused { Duration::from_millis(200) } else { next.saturating_duration_since(Instant::now()) };
            match control.wait(twin.as_mut(), wait) {
                Err(()) => {
                    // the server is gone, so is the process soon: tearing the GPU device down now would race
                    // the driver's own exit-time cleanup (seen as a double free at exit). Leave it.
                    std::mem::forget(twin);
                    return;
                }
                Ok(true) => {
                    next = next.max(Instant::now());
                    if control.paused {
                        control.publish(twin.as_ref());
                    }
                }
                Ok(false) => {}
            }
            if !control.paused && Instant::now() >= next {
                twin.step(DT);
                next += dt;
                if Instant::now() > next + Duration::from_millis(100) {
                    next = Instant::now() + dt; // fell behind: don't try to catch up
                }
                control.publish(twin.as_ref());
            }
        }
    })?;
    Ok((addr, server))
}
