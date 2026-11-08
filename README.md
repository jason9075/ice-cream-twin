# Ice Cream Twin

挖冰淇淋的互動網頁，給機器手臂數位孿生用。三碗冰淇淋（香草、草莓、薄荷巧克力碎片），湯匙可用按鍵、滑鼠或外部程式（WebSocket／ROS 座標）控制，**湯匙不會穿過碗或桌面**，挖起來的冰淇淋可以按 **Eat** 清空。

- 渲染：原生 **WebGPU**（手寫 WGSL，沒有 three.js），SDF 光線步進 + shadow map + MSAA。
- 模擬：純 JS（CPU，確定性），演算法參考 Claybook（見 `~/docs/study/claybook/`）。
- 依賴：只有 `lil-gui`（參數面板）。`puppeteer-core` 只給 `tools/` 的無頭測試用。

```bash
just install
just dev           # http://localhost:5173 （需要支援 WebGPU 的瀏覽器：Chrome / Edge 113+）
just chrome        # Linux：Chrome 預設關閉 WebGPU，用這個開一個帶旗標的獨立 profile
just test          # 碰撞、挖取、座標轉換的單元測試（node，不需要 GPU）
just screenshots   # 建置 + 無頭跑一次挖取並輸出截圖到 shots/
just verify        # test + build + screenshots
just               # 列出所有指令
```

推上 GitHub 的 `main` 會由 `.github/workflows/deploy.yml` 跑測試、建置並部署到 GitHub Pages（repo 的 Settings → Pages → Source 要選 **GitHub Actions**）。

## 演算法（Claybook 式）

冰淇淋同時有兩種形態，互相轉換：

| 形態 | 存放 | 什麼時候 |
|---|---|---|
| 固體 | 每碗一個 SDF 體積（2.5 mm voxel，60×56×60），湯匙上另有一個跟著湯匙走的 SDF（1.6 mm） | 靜止的冰淇淋 |
| 流體 | PBF 粒子（Position Based Fluids + 凝聚力 + XSPH 黏性 + 靜摩擦） | 被湯匙擠開的碎屑、傾倒時滑落的那一坨 |

- **挖**：湯匙金屬部分對碗的 SDF 做 Cut；杯內空間的冰淇淋**複製**進湯匙的 SDF（所以壓進去、拿起來就是一匙）；湯匙上已有的冰淇淋會沿著前進方向再黏上更多（accretion，產生捲起的效果）。被金屬擠開的體積變成粒子。
- **凝固**：粒子相對接觸面靜止幾幀（或一半陷進冰淇淋裡）就 smooth-min 蓋回 SDF（Claybook 的 `DieWhenStops` + `StampSDFWhenDies`）。
- **表面張力**：每幀對被編輯的 voxel 做一次**保體積**的 Laplacian 平滑，之後再「沉澱」30 幀，挖起來的冰淇淋會變成圓潤的一球，而且質量守恆。
- **傾倒**：湯匙傾斜超過 `slide-off tilt`，湯匙上的冰淇淋轉成粒子掉下去。
- **渲染**：每個 SDF 是一個 3D texture（r8 距離 + rgba8 顏色／碎片量），fragment shader 在代理方塊內 sphere tracing，寫 `frag_depth` 和網格正確遮擋；同一支 shader 也畫進 shadow map。

## 湯匙控制與「不穿模」保證（`src/sim/spoon.js`）

湯匙像速度受限的機械手臂工具端，追蹤一個目標姿態（藍色半透明的 ghost）：

1. 每幀朝目標前進，線速度／角速度有上限；切冰淇淋時依估計切削力減速。
2. 這一步切成多個子步，任何碰撞取樣點每子步移動 ≤ 0.8 mm（碗壁 6 mm 厚）→ 不會穿隧。
3. 每個子步後以 position-based 剛體接觸修正（平移 + 旋轉）把湯匙推出碗和桌面 → 會**沿著碗壁滑動**而不是卡死。
4. 修正後仍穿透 > 0.2 mm 的子步直接拒絕，湯匙停在上一個合法姿態，回報 `blocked`。取樣點本身比網格胖 0.2 mm，所以畫面上不會看到穿模。

測試（`tests/collision.test.js`）：120 組隨機目標姿態（含 0.6 m/s、400°/s）、以 3 m/s 朝五個方向硬推穿碗壁，檢查每一幀都沒有任何取樣點越過碗壁。

## 操作

| | |
|---|---|
| 按住按鈕或鍵盤 | **A/D** X · **R/F** Y · **W/S** Z · **U/O** Rx · **J/L** Ry · **K/I** Rz（World／Tool 座標系可切換） |
| 滑鼠 | 左鍵拖曳旋轉視角 · 右鍵或 Shift+拖曳平移 · 滾輪縮放 · **Ctrl+拖曳**移動目標（水平面）· **Ctrl+滾輪**目標高度 |
| 其他 | **E** Eat（清空湯匙）· **H** 回原位 · Tip off（倒掉）· Go to pose（mm／度，XYZ Euler） |

右側面板可即時調整：流體（粒子半徑、黏性、凝聚力、摩擦、阻尼、重力、迭代次數…）、凝固條件、冰淇淋（硬度 kPa、accretion、容量、滑落角度、表面平滑、密度、重新裝填形狀）、湯匙控制器（最大速度、最大切削力）。

## 數位孿生介面

座標：預設是場景的 Y-up（公尺）；指定 `frame: "ros"` 會改用 ROS REP-103（x 前、y 左、z 上）。
湯匙姿態的原點是**杯口中心**，+X 指向握把，+Y 是杯口朝向。

**JS API**（`window.sim`）：

```js
sim.setTarget({ position: [x, y, z], quaternion: [x, y, z, w], frame: 'ros' });
sim.getState('ros');   // 實際／目標姿態、速度、contact、blocked、cutForceN、scoopedGrams、eatenGrams、各碗剩餘克數…
sim.eat(); sim.release(); sim.reset(); sim.refill();
sim.setParams({ fluid: { viscosity: 0.5 }, ice: { hardness: 40 } });
sim.pause(true); sim.stepFrames(120);    // 不經過即時迴圈，直接跑模擬
sim.connect('ws://localhost:8765', 'ros');
window.__view.capturePNG(1280, 800);      // 離屏渲染一張畫面（虛擬相機）
```

**WebSocket**：頁面是 client，外部控制器（例如 ROS bridge node）是 server。

- 送進來：`set_target`、`eat`、`release`、`reset`、`refill`、`set_params`、`get_state`、`config {frame, rateHz}`
- 送出去：連上時送 `hello`（含參數），之後以 `rateHz` 送 `state`

範例：`uv run --with websockets tools/ws_example_server.py`，然後在頁面按 Connect（或 `sim.connect()`），它會用 ROS 座標挖香草 → 吃 → 挖薄荷 → 吃。

## 檔案

```
src/math.js            向量／四元數／矩陣（WebGPU 0..1 深度）
src/sim/sdf.js         碗、湯匙、桌面的解析 SDF；三種口味
src/sim/grid.js        窄帶 SDF 體積 + 顏色、smooth-min 蓋印、保體積平滑
src/sim/spoon.js       湯匙控制器（子步、剛體接觸投影、blocked）
src/sim/fluid.js       PBF 粒子
src/sim/sim.js         挖取／複製／凝固／傾倒／Eat、統計
src/gpu/               WebGPU renderer、WGSL、幾何、軌道相機
src/ui/panel.js        控制面板
src/bridge.js          window.sim 與 WebSocket bridge
tools/                 無頭截圖、WebSocket 範例控制器
```

## 已知限制

- 模擬在 CPU 上（約 4–6 ms/frame）。粒子上限預設 5000；要更大規模可以把 `fluid.js` 搬到 WebGPU compute shader。
- 冰淇淋不會擋住湯匙，只會讓它減速（切削力是依硬度 × 掃過面積估計的，不是真實力學）。
- 固體冰淇淋凝固後不會慢慢下垂。
- 無頭 Chrome 需要用真 GPU：`--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan`（這版 Chrome 的 SwiftShader WebGPU 不能 readback）。
