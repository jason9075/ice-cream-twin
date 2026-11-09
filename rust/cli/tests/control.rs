//! The control server end to end (`icecream server`): a real server on a free port, a WebSocket client.
//! The scoop test runs the GPU simulation when there is a GPU (else the CPU one), the state test the CPU.
use futures_util::{SinkExt, StreamExt};
use icecream_cli::control::{start, Options};
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

async fn reply(ws: &mut Ws, id: i64) -> Value {
    loop {
        let v = next_text(ws).await;
        if v["id"] == id {
            return v;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_controller_scoops_and_eats() {
    let (addr, server) = start(Options { port: 0, paused: true, cpu: false }).await.unwrap();
    tokio::spawn(server);
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.unwrap();

    let hello = next_text(&mut ws).await;
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["paused"], true);
    assert_eq!(hello["scene"]["grids"].as_array().unwrap().len(), 5); // three bowls, the scoop, the table
    assert_eq!(hello["scene"]["arm"]["links"].as_array().unwrap().len(), 8);
    assert_eq!(hello["params"]["fluid"]["maxParticles"], 5000);

    // while paused: above the ice, then drag the cup down through it, opening first (dig tilts the
    // opening toward -x); pressing straight down would only cut
    let dig = [0.0, 0.2474, 0.0, 0.9689]; // 0.5 rad about Z
    ws.send(Message::Text(json!({ "type": "arm", "enabled": false, "id": 1 }).to_string())).await.unwrap();
    assert_eq!(reply(&mut ws, 1).await["arm"]["enabled"], false);
    ws.send(Message::Text(json!({ "type": "set_target", "position": [0.03, 0.15, 0.0], "quaternion": dig }).to_string())).await.unwrap();
    ws.send(Message::Text(json!({ "type": "step", "frames": 120, "id": 2 }).to_string())).await.unwrap();
    assert_eq!(reply(&mut ws, 2).await["scoopedGrams"], 0.0);
    ws.send(Message::Text(json!({ "type": "set_target", "position": [-0.03, 0.088, 0.0], "quaternion": dig }).to_string())).await.unwrap();
    ws.send(Message::Text(json!({ "type": "step", "frames": 240, "id": 3 }).to_string())).await.unwrap();
    let r = reply(&mut ws, 3).await;
    assert_eq!(r["type"], "state");
    assert!(r["scoopedGrams"].as_f64().unwrap() > 1.0, "scooped {}", r["scoopedGrams"]);
    assert!(r["bowlGrams"]["strawberry"].as_f64().unwrap() < 690.0, "{}", r["bowlGrams"]);

    // eat replies with the grams
    ws.send(Message::Text(json!({ "type": "eat", "id": 8 }).to_string())).await.unwrap();
    let v = reply(&mut ws, 8).await;
    assert_eq!(v["type"], "eaten");
    assert!(v["grams"].as_f64().unwrap() > 1.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn controller_gets_state_in_ros_frame() {
    let (addr, server) = start(Options { port: 0, paused: false, cpu: true }).await.unwrap();
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
