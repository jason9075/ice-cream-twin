# Justfile for Ice Cream Twin（WebGPU 挖冰淇淋數位孿生）

set shell := ["bash", "-uc"]

# 預設動作：列出所有可用指令
default:
    @just --list

# 安裝專案相依套件
install:
    npm install

# 啟動本機 Vite 開發伺服器 (http://localhost:5173，需要支援 WebGPU 的瀏覽器)
dev:
    npm run dev

# 用開啟 WebGPU 的獨立 Chrome profile 開啟頁面（Linux 預設關閉 WebGPU；請先在另一個終端機跑 just dev）
chrome url="http://localhost:5173/":
    google-chrome --user-data-dir="${XDG_CACHE_HOME:-$HOME/.cache}/ice-cream-sim-chrome" \
        --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan "{{url}}" > /dev/null 2>&1 &

# 執行單元測試（碰撞不穿模、挖取質量守恆、ROS 座標轉換；node，不需 GPU）
test:
    npm test

# 建置生產環境打包成品 (dist/)
build:
    npm run build

# 預覽生產環境打包成品 (http://localhost:4173)
preview: build
    npx vite preview --port 4173

# 無頭 Chrome（真 GPU）跑一次挖取並輸出截圖到 shots/
screenshots: build
    #!/usr/bin/env bash
    set -euo pipefail
    ./node_modules/.bin/vite preview --port 4173 --strictPort > /dev/null 2>&1 &
    pid=$!
    trap 'kill $pid' EXIT
    for _ in $(seq 50); do curl -sf http://localhost:4173/ > /dev/null && break; sleep 0.2; done
    node tools/shot.mjs http://localhost:4173/ shots

# 啟動 WebSocket 範例控制器（ROS 座標；頁面按 Connect 或 sim.connect()）
ws-example:
    uv run --with websockets tools/ws_example_server.py

# 重新下載並轉換 SO-101 手臂模型（TheRobotStudio/SO-ARM100，Apache-2.0）到 public/so101/，並產生 JS 與 Rust 的關節表
so101-assets:
    uv run --with trimesh --with pymeshlab --with numpy --with scipy --with networkx tools/so101_assets.py

# 重新計算夾爪夾住湯匙柄的角度（結果寫在 src/robot/tool.js）
so101-grip:
    node tools/so101_grip.mjs

# Rust 核心：單元測試 + 不穿模測試 + 手臂測試 + 與 JS 逐幀比對（bit-identical，含手臂）
rust-test:
    cd rust && cargo test --release

# 重新產生 JS 對照資料再跑比對（改了 JS 或 Rust 模擬後執行）
parity:
    node tools/parity-fixture.mjs
    cd rust && cargo test --release --test parity

# Rust 效能測試（挖取 + 三次下挖 + 手臂挖取，印出 ms/frame 與質量）
rust-bench:
    cd rust && cargo run --release -q -p icecream-cli -- bench

# 無頭數位孿生：stdin/stdout JSON lines（和 WebSocket 同一套訊息，含 SO-101 的 set_joints）
rust-serve:
    cd rust && cargo run --release -q -p icecream-cli -- serve

# 執行完整驗證流程 (JS 測試 + Rust 測試 + 生產建置 + 截圖)
verify: test rust-test build screenshots
    @echo "✅ 測試、建置與截圖全部完成！"

# 清理建置產物
clean:
    rm -rf dist shots
    @echo "🧹 已清除 dist/ 與 shots/。"
