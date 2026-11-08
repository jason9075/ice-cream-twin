# Ice Cream Twin

An interactive ice-cream scooping digital twin for a robot arm. Three bowls (vanilla, strawberry, mint
chocolate chip) sit on a table, and an **SO-101 arm** behind them holds the spoon. The spoon can be driven by
keys, the mouse, joint sliders or an external program (WebSocket, optionally in ROS coordinates). **The
spoon never passes through a bowl or the table**, and **Eat** empties whatever is on the spoon.

**The simulation runs in a Rust server, on its GPU (wgpu compute shaders, Claybook-style); the web page
only draws it.**

```
 icecream web (Rust, rust/cli)                                     browser (src/)
 ┌─────────────────────────────────────────────┐   /ws (binary)   ┌──────────────────────────────┐
 │ simulation thread, 60 Hz: the ice cream on  │ ───────────────▶ │ WebGPU renderer: SDF ray     │
 │ the GPU (rust/gpu), spoon + SO-101 on the   │  frames: state,  │ marching, shadows, the arm   │
 │ CPU (rust/core)                             │                  │                              │
 │   └ snapshot → per-client frame (patches)   │  poses, grid     │ meshes, particles            │
 │ WebSocket + static files (--dist)           │  patches         │ panel, keys, mouse, window.sim│
 └─────────────────────────────────────────────┘ ◀─────────────── └──────────────────────────────┘
          ▲  same JSON commands                     JSON commands
          └── controllers: ROS bridge, Python (tools/ws_example_client.py), `icecream serve` over stdio
```

- Simulation: Rust. The ice cream (SDF volumes, particles, the volume ledger) runs as WGSL compute shaders
  through wgpu (`rust/gpu`, ~2 ms/frame while scooping on an RTX 3080 Ti); the spoon controller and the
  SO-101 run on the CPU (`rust/core`). Without a GPU, or with `--cpu`, the same simulation runs on one CPU
  core (~6 ms/frame). The algorithm follows Claybook (notes in `~/docs/study/claybook/`).
- Viewer: native **WebGPU** (hand-written WGSL, no three.js). Dependencies: only `lil-gui`;
  `puppeteer-core` is used by the headless screenshot tool.

```bash
just install       # npm install (the viewer)
just dev           # Rust server on :8765 + Vite on http://localhost:5173 (proxies /ws to the server)
just chrome        # Linux: Chrome ships with WebGPU off; opens a separate profile with the flags on
just serve         # the whole app from one binary: http://localhost:8765 (built page + simulation)
just test          # Rust: collision, mass, arm, fixtures, GPU vs CPU (skipped without a GPU), WebSocket server
just screenshots   # build, run `icecream web --dist dist`, drive a scoop headless, write shots/
just verify        # test + build + screenshots
just               # list all recipes
```

The page used to simulate everything itself (first in JavaScript on the CPU, then as WebGPU compute shaders,
see git history up to `eba6656`). The Rust core was a bit-for-bit port of that JavaScript; the server's GPU
simulation runs the very same WGSL, moved over from the page.

### Browser support

WebGPU works out of the box in Chrome / Edge on Windows, macOS, ChromeOS and Android, and in Safari 26.
**Chrome on Linux** (notably with NVIDIA GPUs) keeps it behind flags, and a page can't change that. Either
run `just chrome`, or open `chrome://flags`, enable **Unsafe WebGPU Support** and **Vulkan**, and relaunch.
`chrome://gpu` should then say "WebGPU: Hardware accelerated".

## Algorithm (Claybook-style)

Ice cream exists in two forms that convert into each other:

| Form | Stored as | When |
|---|---|---|
| Solid | one SDF volume per bowl (2.5 mm voxels, 60×56×60), plus one SDF that rides on the spoon (1.6 mm) | ice cream at rest |
| Loose | PBF particles (Position Based Fluids + cohesion + XSPH viscosity + static friction) | crumbs pushed off by the spoon, a scoop sliding off a tilted spoon |

- **Scooping**: the spoon's metal cuts the bowl SDF. Ice inside the cup is **copied** into the spoon's SDF,
  so pressing in and lifting gives a scoop. Ice already on the spoon picks up more ice in the direction of
  travel (accretion), which gives the curl. Ice pushed aside by the metal becomes particles.
- **Freezing**: a particle that stays still relative to what it rests on for a few frames, or sinks halfway
  into ice, is smooth-min stamped back into the SDF (Claybook's `DieWhenStops` + `StampSDFWhenDies`).
- **Surface tension**: every frame the edited voxels get one **volume-preserving** Laplacian smoothing pass,
  then keep "settling" for 30 frames, so a scoop rounds into a ball without losing mass. While the scoop
  changes, its SDF is rebuilt as a true distance field every few frames (fast sweeping), so crumbs that
  freeze on melt in instead of leaving lumps, and it may never grow under the outside of the cup.
- **Tipping**: tilt the spoon past `slide-off tilt` and the scoop turns into particles and falls.
- **Rendering**: each SDF is a 3D texture (r8 distance + rgba8 colour / speck amount). A fragment shader
  sphere-traces it inside a proxy box and writes `frag_depth`, so it occludes meshes correctly. The same
  shader renders into the shadow map.

### On the GPU (`rust/gpu`)

The whole ice-cream state lives on the GPU: the four SDF volumes (distance + colour), the particles and the
volume ledger sit in a few storage buffers, and each frame is **one submission** of compute passes:

> carve the bowls + copy ice into the scoop (once per spoon sub-step) → carry loose ice with the spoon →
> tip-out → PBF sub-steps (hash grid, density, collision with every SDF) → freeze resting particles
> (smooth-min stamps) → ledger (owed ice stamped back onto the heap, extra ice shaved off) →
> volume-preserving smoothing and settling → redistancing the scoop → statistics

The CPU keeps what is small and sequential: the spoon controller (it never penetrates the analytic bowls),
the SO-101, and deciding which passes to run. It reads back a 1.8 kB block of statistics per frame; for the
viewer a second small submission packs the grids that changed (dirty marks set by the kernels), the
particles and the crumbs, and reads those back too. The cutting force uses the previous frame's carving.

- **Same algorithm, in f32.** Where the CPU walks voxels or particles in order and keeps a running sum,
  the GPU runs every element at once and an exclusive **prefix scan** over the same order tells each one
  what the sum was when it was reached. That covers displaced ice turning into one crumb per particle
  volume, the scoop's capture budget, freezing order and compaction. Owed ice from parallel kernels goes
  into **fixed-point bins** (3×3×3 voxels, atomic adds). Sequential decisions (paying debts, searching for
  a free ice top) run in one workgroup that evaluates candidates and stamps together.
- **Deterministic.** The same run twice gives bit-identical grids and particle positions on a given GPU:
  the atomics only add integers, every list comes out of a scan, and hash cells are sorted by particle
  index. The WGSL gives the same numbers under wgpu/naga (Vulkan) as it did in Chrome.
- **Checked against the CPU simulation** (`rust/gpu/tests/gpu.rs`, skipped without a GPU):

  | check | result |
  |---|---|
  | prefix scans, volume reduction | exact (u32), < 10⁻⁵ (f32) |
  | volume-preserving smoothing + settling of 2,111 edited voxels | max difference 6·10⁻⁷ cell |
  | 61 carve steps of a scoop | bowl grid within 7·10⁻⁶ cell; scoop volume, crumbs and owed ice identical |
  | scoop, carry, eat (510 frames) | 17.8 g scooped (CPU 16.9 g), mass drift 0.009 g |
  | tip-out | scoop gone, 1,827 particles, all frozen back 4 s later, drift 0.04 g |
  | three plunges + eat | 52.4 g eaten, drift 0.01 g |
  | the SO-101 carrying a GPU scoop | 17.7 g, arm within 0.32 mm of the spoon |
  | two identical runs | bit-identical |

- **Speed** (RTX 3080 Ti, Vulkan): a scoop takes ~2 ms/frame with the arm (CPU ~3–4 ms), 15,000 loose
  particles 3.2 ms/frame. `just bench` prints both.

### Mass conservation

Every transfer is measured in voxel occupancy, and anything that can't be placed at once goes into a
**volume ledger**:

- ice the cup sweeps through beyond what it actually holds (a full spoon pushed deeper);
- displaced ice beyond the per-frame particle budget;
- a particle freezing into ice that is already there;
- whatever a smoothing pass can't restore.

Each frame, owed ice is stamped onto the lowest free ice surface near where it went missing, clear of the
spoon, so displaced ice heaps up around the spoon. Stamps that add too much become credit and are shaved off
the highest clear peak. Plunging straight to the bottom of a bowl three times conserves the total
(bowls + spoon + particles + eaten + spilled) to within 0.1 g of 2075 g (`rust/core/tests/sim.rs`). Everything
counted as "spilled" is a visible crumb on the table.

## Spoon control and the no-penetration guarantee (`rust/core/src/spoon.rs`)

The spoon behaves like a speed-limited robot tool tracking a target pose (the translucent blue ghost):

1. Each frame it moves toward the target with capped linear and angular speed, slowing down when the
   estimated cutting force is high.
2. That motion is split into substeps so that no collision sample moves more than 0.8 mm per substep (the
   bowl wall is 6 mm thick), so it can't tunnel.
3. After every substep a position-based rigid contact projection (translation + rotation) pushes the spoon
   out of the bowls and the table, so it **slides along the wall** instead of sticking.
4. A substep that still penetrates more than 0.2 mm after projection is rejected: the spoon stays at the
   last valid pose and reports `blocked`. Collision samples are 0.2 mm fatter than the mesh, so no
   interpenetration is ever visible.

Tested in `rust/core/tests/collision.rs`: 120 random target poses (up to 0.6 m/s and 400°/s), and ramming the
bowl wall at 3 m/s from five directions, checking every frame that no sample crosses the wall.

## Robot arm: SO-101 (`rust/core/src/robot/`)

The arm is the official SO-101 model from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`so101_new_calib.urdf`, Apache-2.0).
`just so101-assets` downloads it at a pinned commit, decimates the meshes to 1.6 MB
(`public/so101/`, licence included; the page draws them, `src/robot/render.js`) and generates the kinematic
chain (`rust/core/src/robot/urdf.rs`).

- **Placement**: the base stands on the table behind the middle bowl at (0, 0, −0.28) m, facing the
  bowls. In ROS coordinates that is (0, 0.28, 0).
- **Tool module** (`rust/core/src/robot/tool.rs`): how the spoon is held. Right now the stock gripper's jaw is
  **locked shut on the spoon handle** at −0.119 rad (−6.8°). The handle lies along the fingers against the
  fixed jaw's fingertip pad, 10 cm from the handle's end, with the cup opening toward the moving jaw.
  The angle is where the jaw mesh first touches the 6.4 mm handle. The real arm will carry a different
  module (later a 3D-printed spoon holder). Swapping it in means adding one entry to `tool()`: the spoon's
  pose in `gripper_link`, `jaw: None`, `hide_jaw: true` (and its meshes for the viewer). Nothing else
  depends on the gripper.
- **Inverse kinematics**: damped least squares on the spoon's 6-D pose with joint limits (active set).
  The SO-101 has 5 joints, so some spoon poses can't be reached. In practice the handle has to point
  roughly at the arm. In exchange, the scooping curl (wrist flex) and the sideways tilt (wrist roll) are
  both native.
- **How it drives the spoon**: the spoon controller still owns contact and the no-penetration guarantee.
  The arm wraps it:
  1. Any new target (mouse, jog keys, Go to pose, a controller) becomes the arm's **goal**. This is the
     closest pose the arm can reach, and it is what the blue ghost shows.
  2. The spoon is led to the goal **in joint space** (90°/s by default). The command waits while the
     spoon is held back by ice. This way the spoon only moves through poses the arm can hold. A straight
     Cartesian line between two reachable poses often isn't one.
  3. After every step the arm is solved for the spoon's actual pose. Those are the joint angles drawn
     and reported. The tests check the arm stays within 0.5 mm of the spoon.
- **Panel**: *Robot arm (SO-101)* has the joint sliders (joint-space command), a switch for the arm, a
  switch for limiting targets to reach, and the tracking error.

## Controls

| | |
|---|---|
| Hold a button or key | **A/D** X · **R/F** Y · **W/S** Z · **U/O** Rx · **J/L** Ry · **K/I** Rz (World / Tool frame toggle) |
| Mouse | left-drag orbit · right-drag or Shift+drag pan · wheel zoom · **Ctrl+drag** move target (horizontal plane) · **Ctrl+wheel** target height |
| Other | **E** Eat (empty the spoon) · **H** home · Tip off · Go to pose (mm / degrees, XYZ Euler) |

The side panel adjusts everything live: fluid (particle radius, viscosity, cohesion, friction, damping,
gravity, iterations…), freezing, ice cream (hardness in kPa, accretion, capacity, slide-off angle, surface
smoothing, density, refill shape) and the spoon controller (max speeds, max cutting force); every change
goes to the server (`set_params`). **View** is drawing only, nothing goes to the server:
- **loose particles**: `surface` (default) melts them into one smooth surface: each frame a compute pass
  splats them into a grid around them and builds the smoothed-particle distance of Zhu & Bridson, which the
  ice cream's sphere tracer then draws (`src/gpu/fluid.js`). `balls`, `both` or `hidden` instead. The
  surface's radius, blend kernel and voxel size are sliders (× particle radius), with the grid it used.
- **Show**: crumbs, ice cream in the bowls and on the spoon, bowls, spoon, SO-101, table, target ghost,
  shadows.
- **Debug**: shading (`normals`, `albedo (unlit)`, `march steps` heat map), a blue tint on the particle
  surface, and translucent boxes for each SDF grid (orange: bowls, green: spoon, blue: particle surface).

From the console, `__view.setView({shading: 'normals', gridBoxes: true})` sets the same options.

## Digital-twin interface

Coordinates default to the scene's Y-up frame in metres. Pass `frame: "ros"` for ROS REP-103 (x forward,
y left, z up). The spoon pose's origin is the **centre of the cup's rim**; +X points toward the handle and
+Y is the direction the cup opens.

### The server (`icecream web`, `rust/cli/src/web.rs`)

`GET /ws` is a WebSocket; everything else is the built page when started with `--dist`. On connect the
server sends `hello` (the scene: grids, bowls, spoon shape, flavours, arm links; and the parameters). Then:

- **Commands** (JSON text, `rust/cli/src/protocol.rs`): `set_target {position?, quaternion?, frame?}`,
  `set_joints {joints, unit: "rad"|"deg"}`, `home`, `eat` → `eaten {grams}`, `release`, `reset`, `refill`,
  `arm {enabled?, constrainTarget?, jointSpeed?}`, `set_params {params}` / `get_params` → `params`,
  `get_state` → `state`, `pause {paused}`, `step {frames}` → `state` (with `seq`). A request's `id` comes back
  in its reply.
- `hello` and every `state` say which simulation runs (`"simulation": "gpu"` or `"cpu"`).
- **Streams**, chosen with `config {stream, frame, rateHz}`:
  - `"state"`: `state` JSON at `rateHz` (controllers);
  - `"render"`: one binary frame per simulation frame (the page). A frame has the state, the spoon and
    ghost poses, the 8 SO-101 link matrices, the loose particles, the crumbs when they change, and **grid
    patches**: for each SDF volume, the bounding box of what changed since *that client's* last frame,
    already quantised like the textures (r8 distance, rgba8 colour). A slow client skips frames, never
    patches. A scoop streams ~100 kB/frame (6 MB/s at 60 fps); the first frame carries every grid (3.3 MB).

`state` holds the actual and target pose, velocity, contact, blocked, cutForceN, scoopedGrams, eatenGrams,
grams left per bowl, particles, and `arm` (joints in rad and deg, goal pose, goalUnreachableMm/Deg,
trackingErrorMm/Deg, base).

Example controller: `just ws-example` (`tools/ws_example_client.py`) connects like the page does, asks for
a ROS-frame state stream and scoops vanilla → eats → scoops mint → eats, with the handle pointed at the arm,
then parks the arm with a `set_joints`. Open the page alongside to watch.

**`icecream serve`** (`just stdio`) takes the same commands as JSON lines on stdin and answers on stdout, for
a ROS node or a script that runs the twin as a subprocess (no real-time loop: it advances on `step`).

### In the page (`window.sim`)

```js
sim.setTarget({ position: [x, y, z], quaternion: [x, y, z, w], frame: 'ros' });
sim.setJoints({ joints: [0, -90, 50, 70, 0], unit: 'deg' });  // SO-101: pan, lift, elbow, wrist flex, wrist roll
sim.getState();                 // the latest streamed state (Y-up); sim.getState('ros') is a Promise of a fresh one
await sim.eat();                // grams eaten
sim.release(); sim.reset(); sim.refill(); sim.home();
await sim.setParams({ fluid: { viscosity: 0.5 }, ice: { hardness: 40 } });
await sim.arm({ enabled: false });
await sim.pause(true); await sim.stepFrames(120);  // step the server; resolves once the page has drawn it
window.__view.capturePNG(1280, 800);               // offscreen render of one frame (virtual camera)
```

### Tests

```bash
just test     # cargo test: collision, mass, arm, fixture regression, the WebSocket server end to end
just bench    # CPU: ~6 ms/frame, mass 2075.3 -> 2075.2 g; arm-driven scoop ~3 ms/frame; the same scoop on the GPU
```

`rust/core/tests/parity.rs` replays scripted runs (a scoop, plunge-and-eat cycles, an arm-driven run) against
fixtures the JavaScript simulation wrote when the Rust core was its bit-for-bit port. The JavaScript is gone;
the fixtures stay as a regression net: any change to the numerics shows up there. (To match it the core
keeps f32 storage where JS had `Float32Array`, V8's `Math.hypot`, V8's fdlibm `sin` / `cos` / `cbrt` in
`v8math.rs`, and the same RNG.) `rust/cli/tests/web.rs` starts a real server and checks the hello, the full
first frame, that a scoop sends only patches of the bowl it touched (and the scoop's colour, even when it
changed frames earlier), `step` / `eat` replies, and a ROS-frame state stream. `rust/gpu/tests/gpu.rs` runs
the GPU simulation against the CPU one (table above).

## Files

```
rust/core/             the simulation (lib icecream_core): spoon controller, SDF grids, PBF particles,
                       scooping, volume ledger, SO-101 kinematics / IK / tool modules; tests + fixtures
rust/gpu/              the same simulation on the GPU (lib icecream_gpu, wgpu): layout.rs (buffers, the
                       generated WGSL header), sim.rs (the frame), twin.rs (+ SO-101), src/wgsl/ (kernels)
rust/cli/              `icecream`: web (WebSocket server + static files), serve (stdio JSON lines), bench
  src/engine.rs          the GPU or CPU simulation behind one interface
  src/protocol.rs        the JSON commands, state, parameters, scene
  src/web.rs             simulation thread, snapshots, per-client binary frames with grid patches
src/main.js            the viewer: builds the scene from `hello`, draws frames, panel, window.sim
src/net/remote.js      WebSocket client, frame decoding, the spoon / params the panel uses
src/gpu/               WebGPU renderer, WGSL, particle surface (fluid.js), geometry, orbit camera
src/robot/render.js    loads public/so101/ and poses the link meshes from the streamed matrices
src/ui/panel.js        control panel (jog, go to pose, readouts)
public/so101/          decimated SO-101 meshes + licence (tools/so101_assets.py)
tools/                 headless screenshots (shot.mjs), WebSocket example controller, SO-101 asset conversion
```

## Known limitations

- The GPU and CPU simulations agree closely but not bit for bit (f32, the cutting force a frame late, a few
  voxels exactly on a test boundary), so long runs drift apart while staying physically the same. The
  fixture regression and the CPU tests pin the CPU version; the GPU one is checked against it with
  tolerances. Particles are capped at 5000 by default (*max particles*, up to 15,000).
- Each GPU frame waits for its read-back (~0.3 ms on this machine); overlapping frame N's read-back with
  frame N+1 would take that off.
- The render stream is uncompressed: ~6 MB/s while scooping, fine on localhost, heavy over a network.
  The scoop's grid changes nearly every frame while it settles; compressing patches (or sending the colour
  only when it changes) would cut most of it.
- One simulation per server, shared by every connected page and controller.
- Only the spoon is collision-checked. The arm's links aren't, so a deep scoop near the arm can push the
  wrist through a bowl rim.
- Joint angles follow the URDF convention (radians, zero at the URDF's zero pose). `so101_new_calib` is meant
  to line up with LeRobot's calibration, but check each joint's zero and direction on the real arm before
  streaming angles to it.
- Ice cream doesn't stop the spoon, only slows it down (the cutting force is estimated from hardness × swept
  area, not real mechanics).
- Solid ice cream doesn't slowly sag once frozen.
- Headless Chrome needs a real GPU: `--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan`
  (this Chrome's SwiftShader WebGPU can't read back).
