//! `icecream web` end to end: a real server on a free port, a WebSocket client. The viewer test runs
//! the GPU simulation when there is a GPU (else the CPU one), the controller test the CPU one.
use futures_util::{SinkExt, StreamExt};
use icecream_cli::web::{start, Options, MAGIC};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn next_text(ws: &mut Ws) -> Value {
    loop {
        match ws.next().await.expect("socket open").expect("message") {
            Message::Text(t) => return serde_json::from_str(&t).unwrap(),
            _ => continue,
        }
    }
}

async fn next_binary(ws: &mut Ws) -> Vec<u8> {
    loop {
        if let Message::Binary(b) = ws.next().await.expect("socket open").expect("message") {
            return b;
        }
    }
}

struct Frame {
    seq: u32,
    state: Value,
    particles: u32,
    patches: Vec<[u32; 8]>,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn parse(b: &[u8]) -> Frame {
    assert_eq!(u32_at(b, 0), MAGIC);
    let seq = u32_at(b, 4);
    let jl = u32_at(b, 8) as usize;
    let state: Value = serde_json::from_slice(&b[12..12 + jl]).unwrap();
    let mut o = (12 + jl).div_ceil(4) * 4;
    o += 7 * 4 + 7 * 4 + 4 + 8 * 16 * 4;
    let particles = u32_at(b, o);
    o += 4 + particles as usize * 16;
    if u32_at(b, o) == 1 {
        let n = u32_at(b, o + 4) as usize;
        o += 8 + n * 16;
    } else {
        o += 4;
    }
    let np = u32_at(b, o);
    o += 4;
    let mut patches = Vec::new();
    for _ in 0..np {
        let h: [u32; 8] = std::array::from_fn(|k| u32_at(b, o + 4 * k));
        let bpp = if h[1] == 0 { 1 } else { 4 };
        o += 32 + (h[5] * h[6] * h[7]) as usize * bpp;
        o = o.div_ceil(4) * 4;
        patches.push(h);
    }
    assert_eq!(o, b.len(), "frame length");
    Frame { seq, state, particles, patches }
}

#[tokio::test(flavor = "multi_thread")]
async fn viewer_gets_the_scene_full_grids_then_patches() {
    let (addr, server) = start(Options { port: 0, dist: None, paused: true, cpu: false }).await.unwrap();
    tokio::spawn(server);
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.unwrap();

    let hello = next_text(&mut ws).await;
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["scene"]["grids"].as_array().unwrap().len(), 4);
    assert_eq!(hello["scene"]["arm"]["links"].as_array().unwrap().len(), 8);
    assert_eq!(hello["params"]["fluid"]["maxParticles"], 5000);

    ws.send(Message::Text(json!({ "type": "config", "stream": "render" }).to_string())).await.unwrap();
    let f0 = parse(&next_binary(&mut ws).await);
    // first frame: every grid in full, distances and colours
    assert_eq!(f0.patches.len(), 8);
    assert!(f0.patches.iter().all(|p| p[2] == 0 && p[3] == 0 && p[4] == 0));
    assert_eq!(f0.state["paused"], true);

    // scoop in the middle bowl while paused: step replies, then a frame with only the changed boxes
    let dig = [0.0, 0.2474, 0.0, 0.9689]; // 0.5 rad about Z
    ws.send(Message::Text(json!({ "type": "arm", "enabled": false }).to_string())).await.unwrap();
    assert_eq!(next_text(&mut ws).await["arm"]["enabled"], false);
    ws.send(Message::Text(json!({ "type": "set_target", "position": [-0.02, 0.085, 0.0], "quaternion": dig }).to_string())).await.unwrap();
    ws.send(Message::Text(json!({ "type": "step", "frames": 240, "id": 7 }).to_string())).await.unwrap();
    let mut reply = None;
    let mut last = None;
    while reply.is_none() || last.as_ref().map(|f: &Frame| (f.seq as u64) < reply.as_ref().map(|r: &Value| r["seq"].as_u64().unwrap()).unwrap()).unwrap_or(true) {
        match ws.next().await.unwrap().unwrap() {
            Message::Text(t) => {
                let v: Value = serde_json::from_str(&t).unwrap();
                if v["id"] == 7 {
                    reply = Some(v);
                }
            }
            Message::Binary(b) => last = Some(parse(&b)),
            _ => {}
        }
    }
    let r = reply.unwrap();
    assert_eq!(r["type"], "state");
    let f1 = last.unwrap();
    assert!(r["scoopedGrams"].as_f64().unwrap() > 1.0, "scooped {}", r["scoopedGrams"]);
    let middle: Vec<_> = f1.patches.iter().filter(|p| p[0] == 1 && p[1] == 0).collect();
    assert!(!middle.is_empty(), "the middle bowl changed");
    assert!(middle.iter().all(|p| p[5] * p[6] * p[7] < 60 * 60 * 60), "a patch, not the whole grid");
    assert!(!f1.patches.iter().any(|p| p[0] == 0 || p[0] == 2), "the other bowls are untouched");
    // the scoop's colour changed somewhere in those 240 frames: it must reach the viewer in this one frame
    assert!(f1.patches.iter().any(|p| p[0] == 3 && p[1] == 1), "the scoop's colour patch");
    assert!(f1.particles > 0 || r["particles"].as_u64().unwrap() == 0);

    // eat replies with the grams
    ws.send(Message::Text(json!({ "type": "eat", "id": 8 }).to_string())).await.unwrap();
    loop {
        let v = next_text(&mut ws).await;
        if v["id"] == 8 {
            assert_eq!(v["type"], "eaten");
            assert!(v["grams"].as_f64().unwrap() > 1.0);
            break;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn controller_gets_state_in_ros_frame() {
    let (addr, server) = start(Options { port: 0, dist: None, paused: false, cpu: true }).await.unwrap();
    tokio::spawn(server);
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.unwrap();
    assert_eq!(next_text(&mut ws).await["type"], "hello");
    ws.send(Message::Text(json!({ "type": "config", "stream": "state", "frame": "ros", "rateHz": 60 }).to_string())).await.unwrap();
    let s = next_text(&mut ws).await;
    assert_eq!(s["type"], "state");
    assert_eq!(s["frame"], "ros");
    // the arm base stands at -0.28 m in the scene's Z, +0.28 m in ROS y
    assert!((s["arm"]["base"]["position"][1].as_f64().unwrap() - 0.28).abs() < 1e-9);
    ws.send(Message::Text(json!({ "type": "set_params", "params": { "fluid": { "viscosity": 0.5 } }, "id": 1 }).to_string())).await.unwrap();
    loop {
        let v = next_text(&mut ws).await;
        if v["id"] == 1 {
            assert_eq!(v["params"]["fluid"]["viscosity"], 0.5);
            break;
        }
    }
    ws.send(Message::Text(json!({ "type": "set_params", "params": { "fluid": { "nope": 1 } }, "id": 2 }).to_string())).await.unwrap();
    loop {
        let v = next_text(&mut ws).await;
        if v["id"] == 2 {
            assert_eq!(v["type"], "error");
            break;
        }
    }
}
