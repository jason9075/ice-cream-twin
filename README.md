# Ice Cream Twin

An interactive ice-cream scooping digital twin for a robot arm. Three bowls (vanilla, strawberry, mint
chocolate chip) sit on a table, and an **SO-101 arm** behind them holds the spoon. The spoon can be driven by
keys, the mouse, joint sliders or an external program (WebSocket, optionally in ROS coordinates). **The
spoon never passes through a bowl or the table**, and **Eat** empties whatever is on the spoon.

**A native Rust application: the simulation and the drawing run on the same GPU (wgpu), Claybook-style. The
ice cream goes from the compute kernels to the screen without leaving the GPU.**

```
 icecream app (rust/cli)
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ window loop, 60 Hz            one wgpu device                                      │
 │  commands ──▶ spoon + SO-101 (CPU, rust/core)                                      │
 │               ice cream: SDF volumes, particles, ledger (compute, rust/gpu) ──┐    │
 │               pack changed grids + particles (compute) ──▶ copy to textures ◀─┘    │
 │               renderer (rust/view): sphere-traced SDFs, particle surface,          │
 │               shadows, SO-101 meshes ──▶ window + side panel (egui)                │
 │  control server (WebSocket /ws, own thread) ── same JSON commands, state stream    │
 └────────────────────────────────────────────────────────────────────────────────────┘
          ▲
          └── controllers: ROS bridge, Python (tools/ws_example_client.py), `icecream serve` over stdio
```

- Simulation: the ice cream (SDF volumes, particles, the volume ledger) runs as WGSL compute shaders
  (`rust/gpu`, ~2 ms/frame while scooping on an RTX 3080 Ti); the spoon controller and the SO-101 run on
  the CPU (`rust/core`). Without a GPU, or with `--cpu`, the same simulation runs on one CPU core
  (~6 ms/frame) and its ice is uploaded to the renderer instead. The algorithm follows Claybook (notes in
  `~/docs/study/claybook/`).
- Drawing: hand-written WGSL through wgpu (`rust/view`); the window and input are winit, the panel egui.
  Per frame only a 1.8 kB block of statistics is read back from the GPU.

```bash
just app           # the window (GPU simulation, panel; control server on ws://localhost:8765/ws)
just app --cpu     # the same with the CPU simulation
just server        # headless: simulation + control server, no window
just test          # collision, mass, arm, fixtures, GPU vs CPU, rendering, control server (GPU tests skip without one)
just screenshots   # a scripted session rendered off screen to shots/*.png
just verify        # test + screenshots
just               # list all recipes
```

History: the simulation started as JavaScript in a web page, then WebGPU compute shaders in the page (git
history up to `eba6656`), then a Rust server the page drew from. The Rust core was a bit-for-bit port of the
JavaScript; the GPU simulation runs the very same WGSL; the renderer's shaders came over from the page too.

## Algorithm (Claybook-style)

Ice cream exists in two forms that convert into each other:

| Form | Stored as | When |
|---|---|---|
| Solid | one SDF volume per bowl (2.5 mm voxels, 60×96×60), one for the table around the bowls (72 × 5 × 40 cm, 288×20×160), plus one SDF that rides on the spoon (1.6 mm) | ice cream at rest, in a bowl or on the table |
| Loose | PBF particles (Position Based Fluids + cohesion + XSPH viscosity + static friction); a released scoop is one shape-matched **lump** | crumbs pushed off by the spoon, a scoop sliding off a tilted spoon |

- **Scooping**: the spoon's metal cuts the bowl SDF. Ice inside the cup is **copied** into the spoon's SDF,
  so pressing in and lifting gives a scoop. Ice already on the spoon picks up more ice in the direction of
  travel (accretion), which gives the curl. Ice pushed aside by the metal is mostly **plowed**: gathered
  per 3×3×3-voxel bin, then each bin walks horizontally away from the spoon to the first ice top where a
  ball of its volume is clear of the spoon, and is stamped there, so it piles up beside the cut as a ridge.
  Only ice pushed aside near the surface piles up (a ridge never builds on itself); what the spoon presses
  down deep is packed away through the ledger (real ice cream is half air and compresses). The rest
  (`ice.plow` = 0.9 plowed, 10 % by default) flies off as crumbs.
- **Freezing**: bowl grids extend to 23.4 cm above the table so scoops dropped onto an existing pile
  can freeze as solid ice instead of becoming permanent static crumbs at the old 14.4 cm ceiling. A particle that stays still relative to what it rests on for a few frames, or sinks halfway
  into ice, is smooth-min stamped back into the SDF (Claybook's `DieWhenStops` + `StampSDFWhenDies`). That
  works anywhere there is a grid: in a bowl, or **on the table** (Claybook's whole world is one SDF; here
  the table around the bowls has its own grid), so dropped ice cream becomes solid ice on the table that
  particles land on and the spoon can scoop again. Only beyond the table grid does a particle become a
  static crumb. Stamps hold a little more than a particle; in a bowl the ledger shaves the difference off
  the heap, on the table the table's own ice gives it back through its next volume-preserving smoothing.
- **Surface tension**: the edited voxels get a **volume-preserving** Laplacian smoothing pass. On the spoon
  it keeps "settling" for 30 frames, so a scoop rounds into a ball without losing mass; while the scoop
  changes, its SDF is rebuilt as a true distance field every few frames (fast sweeping), so crumbs that
  freeze on melt in instead of leaving lumps, and it may never grow under the outside of the cup. **In the
  bowls only what is stamped is smoothed** (frozen crumbs, repaid ice), once, and never into what the spoon
  occupies; **what the spoon cuts keeps its edge** (`ice.smoothCuts` turns the old behaviour back on).
  Smoothing everything for 30 frames made every cut melt into a soft valley, the main reason the ice
  looked watery; it also let a spoon held still pump ice into the cup (smoothing refilled the cut, the next
  carve cut it again), which is why GPU and CPU once scooped 16.8 vs 12.3 g from the same bowl. Crumbs melt
  in through the stamp's smooth-min instead (`stamp smoothing`, 2.5 mm), as Claybook's `k_smooth`. Pressing
  straight down scoops nothing (the metal cuts the ice before the cup gets there), and plunging into the
  same hole again brings up less (21, 6.6, 1.4 g): dragging or lifting through fresh ice is what scoops.
- **Tipping**: tilt the spoon past `slide-off tilt` and the scoop turns into particles and falls.
- **Lumps**: PBF particles alone are a liquid: the density constraint pushes the edges of any pile outward
  every sub-step, so a released scoop slumped into a puddle and ran over the rim (neither particle friction
  nor a yield on XSPH stops that; both were tried, `rust/core/examples/pile.rs` shows it). The particles of
  one release therefore form a lump held by shape matching (Müller et al. 2005): each solver iteration finds
  the rotation and translation that best map the lump's rest shape onto its particles and pulls them
  `lump stiffness` of the way back; a particle pushed more than `lump yield` from its place moves its place
  (plastic flow), so the scoop squashes a little when it lands instead of bouncing back. A dropped scoop
  falls as a ball and lands as a mound; on a level spoon it stays in the cup. Once a lump has touched the
  bowls, the table or their ice for 6 frames it freezes whole, in the shape it landed in (Claybook's
  `DieWhenHitGround`, for a lump): a rigid fit pressed onto a flat table creeps instead of coming to rest.
  Crumbs stay fluid.
- **Rendering**: each SDF is a 3D texture (r8 distance + rgba8 colour / speck amount). A fragment shader
  sphere-traces it inside a proxy box and writes `frag_depth`, so it occludes meshes correctly. The same
  shader renders into the shadow map. The ice cream is **matte frozen foam**: sub-millimetre relief from
  noise perturbs the normal (ice crystals, pores), light wraps and comes back saturated in the shade (a
  little subsurface scattering), and there is no sharp highlight, only a soft sheen at grazing angles
  (View → Debug turns the older glossy look back on).

### On the GPU (`rust/gpu`)

The whole ice-cream state lives on the GPU: the four SDF volumes (distance + colour), the particles and the
volume ledger sit in a few storage buffers, and each frame is **one submission** of compute passes:

> carve the bowls + copy ice into the scoop (once per spoon sub-step) → carry loose ice with the spoon →
> tip-out → PBF sub-steps (hash grid, density, lump shape matching, collision with every SDF) → freeze resting particles
> (smooth-min stamps) → ledger (owed ice stamped back onto the heap, extra ice shaved off) →
> volume-preserving smoothing and settling → redistancing the scoop → statistics

The CPU keeps what is small and sequential: the spoon controller (it never penetrates the analytic bowls),
the SO-101, and deciding which passes to run. It reads back a 1.8 kB block of statistics per frame. For the
renderer a second small submission packs the grids that changed (dirty marks set by the kernels) into rows
of 256 bytes, and the particles and crumbs, all on the GPU; the renderer copies them into its 3D textures and
instance buffer on the same device (`GpuSim::pack_for_render`, `rust/cli/src/scene.rs`). The cutting force
uses the previous frame's carving.

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
  | volume-preserving smoothing + settling of 2,111 edited voxels, the spoon in them | max difference 5·10⁻⁶ cell |
  | 61 carve steps of a scoop | bowl grid within 7·10⁻⁶ cell; scoop volume, crumbs and owed ice identical |
  | 61 carve steps, then one plow pass | the plowed ice adds 53.7 cm³ to the bowl (CPU 54.0) |
  | scoop, carry, eat (510 frames) | 6.6 g scooped (CPU 6.3 g), mass drift 0.009 g |
  | straight down into each bowl, held, lifted | held: no gain; lifted out 7.5 / 0.7 / 4.5 g (CPU 5.4 / 0.1 / 5.6 g) |
  | tip-out | scoop gone, ~558 particles, frozen back 4 s later, drift 0.001 g |
  | the same dragged scoop geometry released on a level spoon | stays in the cup as one lump (16.6 → 13.7 mm tall after 20 frames) |
  | a scoop dropped on the table | freezes into the table grid (GPU 5.8 g, no crumbs), scraped back up 1.2 g (CPU 1.2) |
  | three plunges into the same hole + eat | 18.0 g eaten, drift 0.01 g |
  | the SO-101 carrying a GPU scoop (70 mm deep pass) | ~12 g, arm within 0.31 mm of the spoon |
  | near-edge deep dig, then lift | peak rise 10 mm (previously 80 mm); mass drift 0.009 g |
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

Each frame, owed ice is stamped onto a low clear surface among 64 candidates spread around the
displacement, reaching across the bowl. Searching only a short ray concentrated deep cuts into thin towers;
the area search lets displaced ice spread into lower parts of the bowl without a height cap. A distance
cost favors nearby surfaces, and candidate spacing is denser close to the displacement. This is a
volume redistribution approximation, rather than a pressure or plastic deformation solver.
Stamps that add too much become credit and are shaved off
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
(`rust/view/assets/so101/`, licence included, built into the binary) and generates the kinematic
chain (`rust/core/src/robot/urdf.rs`).

- **Placement**: the base stands on the table behind the middle bowl at (0, 0, −0.28) m, facing the
  bowls. In ROS coordinates that is (0, 0.28, 0).
- **Tool module** (`rust/core/src/robot/tool.rs`): how the spoon is held. Right now the stock gripper's jaw is
  **locked shut on the spoon handle** at −0.119 rad (−6.8°). The handle lies along the fingers against the
  fixed jaw's fingertip pad, 10 cm from the handle's end, with the cup opening toward the moving jaw.
  The angle is where the jaw mesh first touches the 6.4 mm handle. The real arm will carry a different
  module (later a 3D-printed spoon holder). Swapping it in means adding one entry to `tool()`: the spoon's
  pose in `gripper_link`, `jaw: None`, `hide_jaw: true` (and its meshes for the renderer). Nothing else
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
| Other | **E** Eat (empty the spoon) · **B** home · **Space** pause · Tip off · Reset · Go to pose (mm / degrees, XYZ Euler) |

The side panel adjusts everything live: fluid (particle radius, viscosity, cohesion, friction, damping,
gravity, iterations, lump stiffness / yield…), freezing, ice cream (hardness in kPa, accretion, capacity,
slide-off angle, scoop and bowl smoothing, smooth cut edges, density, refill shape) and the spoon controller (max speeds, max cutting force), with the same
commands a controller sends (`rust/cli/src/ui.rs`). **View** is drawing only:
- **loose particles**: `surface` (default) melts them into one smooth surface: each frame compute passes
  find their bounding box, fit a grid to it, splat them and build the smoothed-particle distance of Zhu &
  Bridson, blur it, and the ice cream's sphere tracer draws it (`rust/view/src/wgsl/fluid_splat.wgsl`);
  the grid and the dispatch sizes are worked out on the GPU (indirect dispatch), nothing is read back.
  `balls`, `both` or `hidden` instead. The surface's radius, blend kernel, voxel size and smoothing are
  sliders (× particle radius).
- **Show**: crumbs, ice cream in the bowls (and on the table) and on the spoon, bowls, spoon, SO-101, table, target ghost,
  shadows.
- **Debug**: shading (`normals`, `albedo (unlit)`, `march steps` heat map, `voxels`), matte or the older glossy ice
  cream, a blue tint on the particle surface, and translucent boxes for each SDF grid (orange: bowls,
  green: spoon, purple: table). `voxels` draws occupied cells as cubes with face outlines, using the
  native grid spacing (2.5 mm in bowls/on the table, 1.6 mm on the spoon). The particle surface uses its
  rendering grid, whose spacing is controlled by the particle surface's voxel slider. Cell traversal
  replaces interpolation for both visible surfaces and shadows; simulation data stays unchanged.

## Digital-twin interface

Coordinates default to the scene's Y-up frame in metres. Pass `frame: "ros"` for ROS REP-103 (x forward,
y left, z up). The spoon pose's origin is the **centre of the cup's rim**; +X points toward the handle and
+Y is the direction the cup opens.

### The control server (`rust/cli/src/control.rs`)

`icecream app` (alongside the window; `--port N`, `--no-control`) and `icecream server` (headless) listen on
`ws://localhost:8765/ws`. On connect the server sends `hello` (the scene: grids, bowls, spoon shape,
flavours, arm links; and the parameters). Then:

- **Commands** (JSON text, `rust/cli/src/protocol.rs`): `set_target {position?, quaternion?, frame?}`,
  `set_joints {joints, unit: "rad"|"deg"}`, `home`, `eat` → `eaten {grams}`, `release`, `reset`, `refill`,
  `arm {enabled?, constrainTarget?, jointSpeed?}`, `set_params {params}` / `get_params` → `params`,
  `get_state` → `state`, `pause {paused}`, `step {frames}` → `state`. A request's `id` comes back in its reply.
- `hello` and every `state` say which simulation runs (`"simulation": "gpu"` or `"cpu"`).
- **State stream**: `config {stream: "state", frame, rateHz}` sends `state` JSON at `rateHz`.

`state` holds the actual and target pose, velocity, contact, blocked, cutForceN, scoopedGrams, eatenGrams,
grams left per bowl, grams frozen on the table (`tableGrams`), particles, and `arm` (joints in rad and deg, goal pose, goalUnreachableMm/Deg,
trackingErrorMm/Deg, base).

Example controller: `just ws-example` (`tools/ws_example_client.py`) asks for a ROS-frame state stream and
scoops vanilla → eats → scoops mint → eats, with the handle pointed at the arm, then parks the arm with a
`set_joints`. Run it next to `just app` to watch.

**`icecream serve`** (`just stdio`) takes the same commands as JSON lines on stdin and answers on stdout, for
a ROS node or a script that runs the twin as a subprocess (no real-time loop: it advances on `step`).

**`icecream shot [DIR] [--params JSON] [--glossy]`** (`just screenshots`) renders a scripted session off
screen (`--params` patches the parameters first, for A/B comparisons): the SO-101 scoops
strawberry, close-ups of the spoon and the gripper, the scoop rolled off into the bowl, and the same frame
through the debug views; it prints a JSON summary and fails on any GPU validation error.

### Tests

```bash
just test     # cargo test: collision, mass, arm, fixture regression, GPU vs CPU, rendering, the control server
just bench    # CPU: mass 2075.3 -> 2075.3 g; arm-driven scoop ~2 ms/frame; the same scoop on the GPU ~1.4
```

`rust/core/tests/parity.rs` replays scripted runs (a scoop, plunge-and-eat cycles, an arm-driven run) against
fixtures the JavaScript simulation wrote when the Rust core was its bit-for-bit port. The JavaScript is gone;
the fixtures stay as a regression net: any change to the numerics shows up there. When a change is meant to
alter the numbers, `UPDATE_FIXTURES=1` rewrites them from the Rust core (done for: smoothing keeping out of
the spoon; cuts kept crisp, crumbs blended at the stamp; taller bowl grids to retain stacked scoops;
displaced volume repaid over an area to prevent towers). (To match it the core
keeps f32 storage where JS had `Float32Array`, V8's `Math.hypot`, V8's fdlibm `sin` / `cos` / `cbrt` in
`v8math.rs`, and the same RNG.) `rust/cli/tests/control.rs` starts a real headless server and checks the
hello, a scoop driven by `step` commands, `eat`, and a ROS-frame state stream. `rust/cli/tests/render.rs`
renders the scene off screen fed by the GPU simulation and by the CPU one: the two pictures agree (mean
difference 0.006 per channel). `rust/view` validates its shaders with naga. `rust/gpu/tests/gpu.rs` runs the
GPU simulation against the CPU one (table above).

Experiments (`cargo run --release -p icecream-core --example ...`): `pile` drops a cube of particles on a
floor and prints its height (lumps on or off); `pour` scoops, tips the scoop out and prints how the loose ice
holds together and the shape of what lands (`RELEASE=1`: released on a level spoon instead).

## Files

```
rust/core/             the simulation (lib icecream_core): spoon controller, SDF grids, PBF particles and
                       lumps, scooping, volume ledger, SO-101 kinematics / IK / tool modules; tests,
                       fixtures, examples/ (pile, pour experiments)
rust/gpu/              the same simulation on the GPU (lib icecream_gpu, wgpu): layout.rs (buffers, the
                       generated WGSL header), sim.rs (the frame, packing for the renderer), twin.rs
                       (+ SO-101), src/wgsl/ (kernels)
rust/view/             the renderer (lib icecream_view, no window code): renderer.rs (pipelines, scene,
                       shadow map, off-screen capture), fluid.rs + wgsl/fluid_splat.wgsl (particle surface),
                       wgsl/ (mesh, instances, SDF volume tracer), geometry, orbit camera, arm.rs +
                       assets/so101/ (decimated SO-101 meshes + licence, tools/so101_assets.py)
rust/cli/              `icecream`: app, server, shot, serve (stdio JSON lines), bench
  src/app.rs             the window: shared device, 60 Hz loop, mouse and keys, egui
  src/ui.rs              the side panel
  src/scene.rs           simulation → renderer each frame (GPU: copies on the device; CPU: uploads)
  src/control.rs         the WebSocket control server
  src/engine.rs          the GPU or CPU simulation behind one interface
  src/protocol.rs        the JSON commands, state, parameters, scene
  src/shot.rs            the scripted off-screen screenshots
tools/                 WebSocket example controller, SO-101 asset conversion
```

## Known limitations

- The GPU and CPU simulations agree closely but not bit for bit (f32, the cutting force a frame late, a few
  voxels exactly on a test boundary), so long runs drift apart while staying physically the same; ridges
  plowed beside a cut feed back into later carving, so whole scoops can differ by a third. The
  step-by-step comparisons (carving, plowing, smoothing) agree to a few cells. The
  fixture regression and the CPU tests pin the CPU version; the GPU one is checked against it with
  tolerances. Particles are capped at 5000 by default (*max particles*, up to 15,000).
- Each GPU frame waits for its read-back (~0.3 ms on this machine); overlapping frame N's read-back with
  frame N+1 would take that off.
- One simulation per process, shared by the window and every connected controller. Watching from another
  machine needs a remote desktop (or a video stream, not built).
- Only the spoon is collision-checked. The arm's links aren't, so a deep scoop near the arm can push the
  wrist through a bowl rim.
- Joint angles follow the URDF convention (radians, zero at the URDF's zero pose). `so101_new_calib` is meant
  to line up with LeRobot's calibration, but check each joint's zero and direction on the real arm before
  streaming angles to it.
- What scoops is the cup's opening moving into ice (lifting, or pushing opening first); moving the cup's
  bottom first only cuts. A low pile on the table is therefore hard to pick up (tilt the spoon to ~80°,
  rim along the table, opening toward the pile): about 1–3 g of a 15 g pile per pass; a taller lump scoops
  like a bowl.
- Ice cream doesn't stop the spoon, only slows it down (the cutting force is estimated from hardness × swept
  area, not real mechanics).
- Solid ice cream doesn't slowly sag once frozen.
- `icecream shot` and the rendering test need a GPU adapter (no software fallback is set up); without one
  the test skips.

## Operation recording

Use **Record operations** / **Stop recording operations** in the side panel to save a diagnostic
JSONL journal (no video). The full path appears below the button; **Copy recording path** copies it, including after recording stops. Files are created under
`recordings/` in the working directory (`rust/recordings/` when launched with `just app`).
The journal includes an initial state and parameters, protocol command attempts (including WebSocket
commands), window input, and per-frame state, parameters, camera, view settings and timing.
A `frame_begin` without a corresponding `frame_end` helps locate an interrupted frame.

Each line is written immediately without application buffering, so completed writes remain available
if the application crashes; stopping normally also syncs the file. After a crash, ignore a possibly
incomplete final JSON line. This protects against process crashes, not power loss or disk failure.
Recording must be started before the incident; it does not recover earlier actions. It is a diagnostic
journal, not a full simulation checkpoint or an automatic deterministic replay. It may contain input
text and can grow substantially during long sessions. Recording errors are displayed in the panel.

## UI preferences

Section expansion (Fluid, Ice cream, Spoon controller, Robot arm, View) and all settings inside View
are saved automatically to `$XDG_CONFIG_HOME/ice-cream-sim/ui.json`, or
`~/.config/ice-cream-sim/ui.json` when XDG_CONFIG_HOME is unset. Changes are written through a temporary
file and renamed into place. Invalid files fall back to defaults. Simulation parameters, arm settings,
poses, camera position, pause state and jog settings are not restored between runs.
