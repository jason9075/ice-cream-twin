# Ice Cream Twin

An interactive ice-cream scooping page, built as a digital-twin environment for a robot arm. Three bowls
(vanilla, strawberry, mint chocolate chip) sit on a table, and an **SO-101 arm** behind them holds the spoon.
The spoon can be driven by keys, the mouse, joint sliders or an external program (WebSocket, optionally in
ROS coordinates). **The spoon never passes through a bowl or the
table**, and **Eat** empties whatever is on the spoon.

- Rendering: native **WebGPU** (hand-written WGSL, no three.js), SDF ray marching, shadow map, MSAA.
- Simulation: plain JavaScript on the CPU (deterministic). The algorithm follows Claybook (notes in
  `~/docs/study/claybook/`).
- Dependencies: only `lil-gui` (parameter panel). `puppeteer-core` is used only by the headless tools in
  `tools/`.

```bash
just install
just dev           # http://localhost:5173 (needs a WebGPU browser: Chrome / Edge 113+)
just chrome        # Linux: Chrome ships with WebGPU off; opens a separate profile with the flags on
just test          # unit tests: collision, scooping, mass conservation, frames, SO-101 IK (node, no GPU)
just screenshots   # build, run a scripted scoop headless, write screenshots to shots/
just verify        # test + build + screenshots
just               # list all recipes
```

Pushing to `main` runs `.github/workflows/deploy.yml`: tests, build, deploy to GitHub Pages (set the repo's
Settings → Pages → Source to **GitHub Actions**).

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

### Everything on the GPU (`src/gpu/sim/`)

In the page the whole simulation lives on the GPU, as in Claybook: the four SDF volumes (distance +
colour), the particles and the volume ledger sit in GPU buffers, and each frame is **one submission** of
compute passes:

> carve the bowls + copy ice into the scoop (once per spoon sub-step) → carry loose ice with the spoon →
> tip-out → PBF sub-steps (hash grid, density, collision with every SDF) → freeze resting particles
> (smooth-min stamps) → ledger (owed ice stamped back onto the heap, extra ice shaved off) →
> volume-preserving smoothing and settling → redistancing the scoop → statistics → render textures and
> particle instances

The CPU keeps what is small and sequential: the spoon controller (it never penetrates the analytic bowls),
the SO-101, and deciding which passes to run. It reads back one 1.5 kB block of statistics per frame,
a frame late. The CPU version reads `contentVolume` at the same point, so nothing waits longer than it
did on the CPU, except the cutting force, which lags by one frame.

- **Same algorithm, in f32.** Where the CPU walks voxels or particles in order and keeps a running sum,
  the GPU runs every element at once and an exclusive **prefix scan** over the same order tells each one
  what the sum was when it was reached. That covers displaced ice turning into one crumb per particle
  volume, the scoop's capture budget, freezing order and compaction. Owed ice from parallel kernels goes
  into **fixed-point bins** (3×3×3 voxels, atomic adds). Sequential decisions (paying debts, searching for
  a free ice top) run in one workgroup that evaluates candidates and stamps together.
- **Deterministic.** The same run twice gives bit-identical grids and particle positions on a given GPU.
  The atomics only add integers, neighbour lists and every other list come out of scans, and hash cells
  are sorted by particle index.
- **Checked against the CPU** (`just gpu-sim-check`, headless Chrome with a real GPU):

  | check | result |
  |---|---|
  | prefix scans, volume reduction | exact (u32), 2·10⁻⁷ (f32) |
  | volume-preserving smoothing + settling of 13,837 edited voxels | max difference 6·10⁻⁷ cell |
  | 61 carve steps of a scoop | bowl grid within 7·10⁻⁶ cell; scoop volume, crumbs and owed ice identical |
  | the CPU tests' scoop, carry, eat (510 frames) | 16.6 g scooped (CPU 18.1 g), mass drift 0.010 g |
  | tip-out | scoop gone, 1,800 particles, all frozen back 4 s later, drift 0.04 g |
  | three plunges + eat | 52.4 g eaten (CPU 53.3 g), drift 0.01 g |
  | two identical runs | bit-identical |

- **Speed** (RTX-class GPU): ~1 ms/frame on average over a scoop against ~5 ms on the CPU. Busy frames
  take ~4 ms (2.5 ms of it GPU work). 15,000 loose particles take 4.2 ms/frame.
- The CPU simulation (`src/sim/`) stays the reference: node tests and the Rust port follow it bit for
  bit. Run it in the page with `?sim=cpu` or the *simulation* switch at the top of the panel.

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
(bowls + spoon + particles + eaten + spilled) to within 0.1 g of 2075 g (`tests/sim.test.js`). Everything
counted as "spilled" is a visible crumb on the table.

## Spoon control and the no-penetration guarantee (`src/sim/spoon.js`)

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

Tested in `tests/collision.test.js`: 120 random target poses (up to 0.6 m/s and 400°/s), and ramming the
bowl wall at 3 m/s from five directions, checking every frame that no sample crosses the wall.

## Robot arm: SO-101 (`src/robot/`)

The arm is the official SO-101 model from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`so101_new_calib.urdf`, Apache-2.0).
`just so101-assets` downloads it at a pinned commit, decimates the meshes to 1.6 MB
(`public/so101/`, licence included) and generates the kinematic chain (`src/robot/so101-urdf.js`, and the same
table for Rust in `rust/core/src/robot/urdf.rs`).

- **Placement**: the base stands on the table behind the middle bowl at (0, 0, −0.28) m, facing the
  bowls. In ROS coordinates that is (0, 0.28, 0).
- **Tool module** (`src/robot/tool.js`): how the spoon is held. Right now the stock gripper's jaw is
  **locked shut on the spoon handle** at −0.119 rad (−6.8°). The handle lies along the fingers against the
  fixed jaw's fingertip pad, 10 cm from the handle's end, with the cup opening toward the moving jaw.
  `just so101-grip` recomputes the jaw angle from the meshes. The real arm will carry a different module
  (later a 3D-printed spoon holder). Swapping it in means adding one entry to `TOOLS`: the spoon's pose in
  `gripper_link`, `jaw: null`, `hideJaw: true` and its meshes. Nothing else depends on the gripper.
- **Inverse kinematics**: damped least squares on the spoon's 6-D pose with joint limits (active set).
  The SO-101 has 5 joints, so some spoon poses can't be reached. In practice the handle has to point
  roughly at the arm. In exchange, the scooping curl (wrist flex) and the sideways tilt (wrist roll) are
  both native.
- **How it drives the spoon**: the spoon controller still owns contact and the no-penetration guarantee.
  The arm wraps it:
  1. Any new target (mouse, jog keys, Go to pose, bridge) becomes the arm's **goal**. This is the
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
smoothing, density, refill shape) and the spoon controller (max speeds, max cutting force). Under **View**
you can show the target ghost, the collision samples and the loose particles (hidden by default).

## Digital-twin interface

Coordinates default to the scene's Y-up frame in metres. Pass `frame: "ros"` for ROS REP-103 (x forward,
y left, z up). The spoon pose's origin is the **centre of the cup's rim**; +X points toward the handle and
+Y is the direction the cup opens.

**JS API** (`window.sim`):

```js
sim.setTarget({ position: [x, y, z], quaternion: [x, y, z, w], frame: 'ros' });
sim.setJoints({ joints: [0, -90, 50, 70, 0], unit: 'deg' });  // SO-101: pan, lift, elbow, wrist flex, wrist roll
sim.getState('ros');   // actual / target pose, velocity, contact, blocked, cutForceN, scoopedGrams, eatenGrams, grams left per bowl,
                       // particles, simulation ('gpu' | 'cpu'), gpuMs…
                       // + arm: joints (rad), jointsDeg, goal pose, goalUnreachableMm/Deg, trackingErrorMm…
await sim.eat();       // grams eaten (a Promise: the GPU simulation reads it back)
sim.release(); sim.reset(); sim.refill();
sim.setParams({ fluid: { viscosity: 0.5 }, ice: { hardness: 40 } });
sim.pause(true); await sim.stepFrames(120);  // run outside the real-time loop (a Promise of the state)
sim.setSimulation('cpu');                   // reload on the CPU reference simulation ('gpu' = compute shaders)
sim.connect('ws://localhost:8765', 'ros');
window.__view.capturePNG(1280, 800);      // offscreen render of one frame (virtual camera)
```

**WebSocket**: the page is the client; the external controller (e.g. a ROS bridge node) is the server.

- Incoming: `set_target`, `set_joints {joints, unit: "rad"|"deg"}`, `eat`, `release`, `reset`, `refill`, `set_params`, `get_state`, `config {frame, rateHz}`
- Outgoing: `hello` (with the parameters) on connect, then `state` at `rateHz`

Example: `just ws-example` (or `uv run --with websockets tools/ws_example_server.py`), then press Connect
on the page (or call `sim.connect()`). Using ROS coordinates, it scoops vanilla → eats → scoops mint → eats,
with the handle pointed at the arm, then parks the arm with a `set_joints`.

## Rust core (`rust/`)

The simulation is also available as a Rust library, `icecream-core`, plus a headless CLI. This is the base
for a native digital-twin app: direct ROS 2, batch and parallel runs, no browser.

- `core/` is a port of `src/sim/*.js`, `src/robot/*.js` and `src/math.js`: the spoon controller, the SDF
  grids, the PBF particles, scooping, the volume ledger, the frames and the SO-101 (kinematics, tool
  modules, IK, joint-space leading). `Twin` steps the arm and the sim together like the page does
  (`arm.constrain` → `sim.step` → `arm.follow`). The WebGPU renderer stays in JS for now.
- **Bit-for-bit parity with the JS version.** A scripted scoop (510 frames) and two plunge-and-eat cycles
  (900 frames) produce identical spoon poses, ice volumes, particle counts and ledger values in both. So
  does an arm-driven run (830 frames: a scoop, a joint command, a move to the next bowl), down to every
  joint angle, joint command and IK error. To get
  there, the Rust code:
  - stores f32 wherever JS uses a `Float32Array` and does the arithmetic in f64;
  - copies V8's `Math.hypot` algorithm;
  - ports V8's fdlibm `sin`, `cos` and `cbrt` (`v8math.rs`; the `libm` crate differs in the last bit for
    about 1% of inputs);
  - uses the same RNG and hash wrapping.
- The fixtures are written by `node tools/parity-fixture.mjs` and checked by `cargo test`. CI regenerates
  them with Node 22 (V8 12.4) and fails if they no longer match.
- The collision tests (no penetration, no tunnelling, sliding along the wall), the mass-conservation tests
  and the arm tests are ported too.

```bash
just rust-test     # 20 tests: collision, mass, arm, parity with JS
just parity        # regenerate the JS fixtures, then compare
just rust-bench    # ~6 ms/frame single-threaded, mass 2075.3 -> 2075.2 g; arm-driven scoop ~3 ms/frame
just rust-serve    # headless twin over stdin/stdout
```

**`icecream serve`** speaks the WebSocket bridge's messages as JSON lines (one object per line), so a ROS node
or a Python script can run it as a subprocess. The SO-101 holds the spoon by default, as on the page:
`set_target` goes to the closest pose the arm can reach, `set_joints` commands the joints, and every state
carries the `arm` block. `{"type":"config","arm":false}` frees the spoon.

```text
-> {"type":"config","frame":"ros"}
-> {"type":"set_target","position":[-0.18,0,0.085],"quaternion":[0,0,0,1]}
-> {"type":"step","frames":200}
<- {"type":"state","spoon":{...},"scoopedGrams":11.7,"bowlGrams":{...},"blocked":false,...}
-> {"type":"eat"}
-> {"type":"set_joints","joints":[0,-90,50,70,0],"unit":"deg"}
-> {"type":"step","frames":120}
<- {"type":"state",...,"arm":{"jointsDeg":[0.0,-90.0,50.0,70.0,0.0],"trackingErrorMm":3e-6,...}}
```

## Files

```
src/math.js            vectors / quaternions / matrices (WebGPU 0..1 depth)
src/sim/sdf.js         analytic SDFs for the bowls, spoon and table; the three flavours
src/sim/grid.js        narrow-band SDF volume + colour, smooth-min stamping, volume-preserving smoothing
src/sim/spoon.js       spoon controller (substeps, rigid contact projection, blocked)
src/sim/fluid.js       PBF particles
src/sim/sim.js         scoop / copy / freeze / tip / Eat, volume ledger, stats (the CPU reference)
src/gpu/sim/           the same simulation resident on the GPU (default in the page):
  layout.js              buffer map, shared WGSL (grids, SDFs, reductions, ledger primitives)
  k-world.js             carving, copying into the scoop, crumbs, carrying, tip-out, eating, compaction
  k-fluid.js             PBF particles
  k-freeze.js            freezing particles back into the SDFs
  k-ledger.js            owed ice: bins -> debts -> stamps, credit, residuals
  k-grid.js              scans, volumes, smoothing / settling, redistancing, render textures
  gpu-sim.js             the frame: which passes, in what order; statistics read-back
src/gpu/               WebGPU renderer, WGSL, geometry, orbit camera
src/robot/so101.js     SO-101 forward kinematics (chain generated from the URDF in so101-urdf.js)
src/robot/arm.js       placement, IK, goal projection, joint-space leading of the spoon
src/robot/tool.js      end-of-arm tool modules (today: gripper clamped on the handle)
src/robot/render.js    loads public/so101/ and poses the link meshes
public/so101/          decimated SO-101 meshes + licence (tools/so101_assets.py)
src/ui/panel.js        control panel
src/bridge.js          window.sim and the WebSocket bridge
tools/                 headless screenshots, GPU-vs-CPU checks (gpu-sim-check.mjs + gpu-sim-tests.js),
                       WebSocket example controller, parity fixtures, SO-101 asset conversion
                       (so101_assets.py) and jaw-angle sweep (so101_grip.mjs)
rust/core/             Rust port of the simulation and the arm (lib icecream_core) + tests and JS fixtures
rust/cli/              `icecream serve` (JSON lines) and `icecream bench`
```

## Known limitations

- The GPU simulation agrees with the CPU closely but not bit for bit (f32, a one-frame lag in the cutting
  force, a few voxels that sit exactly on a test boundary), so a long run drifts apart from the CPU one
  while staying physically the same. The Rust core runs the CPU reference only. The WGSL could run there
  through wgpu, and would be checked with tolerances like `tools/gpu-sim-tests.js` does.
- Particles are capped at 5000 by default (*max particles*, up to 15,000 on both).
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
