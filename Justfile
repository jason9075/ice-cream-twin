# Justfile for Ice Cream Twin（挖冰淇淋數位孿生：Rust 後端跑模擬，網頁只負責畫面）

set shell := ["bash", "-uc"]

# 預設動作：列出所有可用指令
default:
    @just --list

# 安裝前端相依套件
install:
    npm install

# 開發：啟動 Rust 模擬伺服器（ws://localhost:8765/ws）+ Vite 前端（http://localhost:5173，/ws 代理到伺服器）
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    (cd rust && cargo build --release -q -p icecream-cli)
    ./rust/target/release/icecream web --port 8765 &
    pid=$!
    trap 'kill $pid' EXIT
    npm run dev

# 只啟動 Rust 模擬伺服器（給外部控制器或另開的前端用）
server port="8765":
    cd rust && cargo run --release -q -p icecream-cli -- web --port {{port}}

# 用開啟 WebGPU 的獨立 Chrome profile 開啟頁面（Linux 預設關閉 WebGPU；請先在另一個終端機跑 just dev）
chrome url="http://localhost:5173/":
    google-chrome --user-data-dir="${XDG_CACHE_HOME:-$HOME/.cache}/ice-cream-sim-chrome" \
        --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan "{{url}}" > /dev/null 2>&1 &

# 建置前端 (dist/)
build:
    npm run build

# 單一程式提供整個應用：建置前端後由 Rust 伺服器同時送出頁面與模擬（http://localhost:8765）
serve port="8765": build
    cd rust && cargo run --release -q -p icecream-cli -- web --port {{port}} --dist ../dist

# Rust 測試：碰撞不穿模、質量守恆、手臂、凍結對照資料回歸、WebSocket 伺服器端到端
test:
    cd rust && cargo test --release

# 無頭 Chrome（真 GPU）連到 Rust 伺服器跑一次挖取，截圖輸出到 shots/
screenshots: build
    #!/usr/bin/env bash
    set -euo pipefail
    (cd rust && cargo build --release -q -p icecream-cli)
    ./rust/target/release/icecream web --port 4173 --dist dist > /dev/null 2>&1 &
    pid=$!
    trap 'kill $pid' EXIT
    for _ in $(seq 50); do curl -sf http://localhost:4173/ > /dev/null && break; sleep 0.2; done
    node tools/shot.mjs http://localhost:4173/ shots

# WebSocket 範例控制器（ROS 座標）：連到 Rust 伺服器挖兩匙（請先 just dev 或 just server）
ws-example url="ws://localhost:8765/ws":
    uv run --with websockets tools/ws_example_client.py {{url}}

# 重新下載並轉換 SO-101 手臂模型（TheRobotStudio/SO-ARM100，Apache-2.0）到 public/so101/，並產生 Rust 的關節表
so101-assets:
    uv run --with trimesh --with pymeshlab --with numpy --with scipy --with networkx tools/so101_assets.py

# Rust 效能測試（挖取 + 三次下挖 + 手臂挖取，印出 ms/frame 與質量）
bench:
    cd rust && cargo run --release -q -p icecream-cli -- bench

# 無頭數位孿生：stdin/stdout JSON lines（和 WebSocket 同一套指令，給 ROS node 當子程序用）
stdio:
    cd rust && cargo run --release -q -p icecream-cli -- serve

# 執行完整驗證流程 (Rust 測試 + 前端建置 + 截圖)
verify: test build screenshots
    @echo "✅ 測試、建置與截圖全部完成！"

# 清理建置產物
clean:
    rm -rf dist shots
    @echo "🧹 已清除 dist/ 與 shots/。"
