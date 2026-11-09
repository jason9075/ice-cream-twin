# Justfile for Ice Cream Twin（挖冰淇淋數位孿生：Rust 原生程式，模擬與繪圖在同一張 GPU 上）

set shell := ["bash", "-uc"]

# 預設動作：列出所有可用指令
default:
    @just --list

# 開啟視窗程式（GPU 模擬 + 繪圖 + 控制面板；控制伺服器 ws://localhost:8765/ws 同時開著）。加 --cpu 改用 CPU 模擬
app *flags:
    cd rust && cargo run --release -q -p icecream-cli -- app {{flags}}

# 無頭模擬伺服器（不開視窗，只有 WebSocket 控制；給外部控制器或 CI 用）
server *flags:
    cd rust && cargo run --release -q -p icecream-cli -- server --port 8765 {{flags}}

# Rust 測試：碰撞不穿模、質量守恆、手臂、凍結對照資料回歸、GPU 對 CPU 比對、繪圖、控制伺服器端到端（沒有 GPU 會略過）
test:
    cd rust && cargo test --release

# 離屏繪圖：手臂挖一匙、特寫、翻轉倒下、debug 視圖，輸出 PNG 到 shots/
screenshots:
    cd rust && cargo run --release -q -p icecream-cli -- shot ../shots

# WebSocket 範例控制器（ROS 座標）：連到模擬挖兩匙（請先 just app 或 just server）
ws-example url="ws://localhost:8765/ws":
    uv run --with websockets tools/ws_example_client.py {{url}}

# 重新下載並轉換 SO-101 手臂模型（TheRobotStudio/SO-ARM100，Apache-2.0）到 rust/view/assets/so101/，並產生 Rust 的關節表
so101-assets:
    uv run --with trimesh --with pymeshlab --with numpy --with scipy --with networkx tools/so101_assets.py

# 效能測試：CPU 版挖取 + 三次下挖 + 手臂挖取，再跑一次 GPU 版的手臂挖取
bench:
    cd rust && cargo run --release -q -p icecream-cli -- bench

# 無頭數位孿生：stdin/stdout JSON lines（和 WebSocket 同一套指令，給 ROS node 當子程序用）
stdio:
    cd rust && cargo run --release -q -p icecream-cli -- serve

# 執行完整驗證流程 (Rust 測試 + 截圖)
verify: test screenshots
    @echo "✅ 測試與截圖全部完成！"

# 清理產物
clean:
    rm -rf shots
    cd rust && cargo clean
    @echo "🧹 已清除 shots/ 與 rust/target/。"
