# Ice Cream Twin

An interactive ice-cream scooping page, built as a digital-twin environment for a robot arm. Three bowls
(vanilla, strawberry, mint chocolate chip) sit on a table. The spoon can be driven by keys, the mouse or an
external program (WebSocket, optionally in ROS coordinates). **The spoon never passes through a bowl or the
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
just test          # unit tests: collision, scooping, mass conservation, frames (node, no GPU)
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
sim.getState('ros');   // actual / target pose, velocity, contact, blocked, cutForceN, scoopedGrams, eatenGrams, grams left per bowl…
sim.eat(); sim.release(); sim.reset(); sim.refill();
sim.setParams({ fluid: { viscosity: 0.5 }, ice: { hardness: 40 } });
sim.pause(true); sim.stepFrames(120);    // run the simulation directly, outside the real-time loop
sim.connect('ws://localhost:8765', 'ros');
window.__view.capturePNG(1280, 800);      // offscreen render of one frame (virtual camera)
```

**WebSocket**: the page is the client; the external controller (e.g. a ROS bridge node) is the server.

- Incoming: `set_target`, `eat`, `release`, `reset`, `refill`, `set_params`, `get_state`, `config {frame, rateHz}`
- Outgoing: `hello` (with the parameters) on connect, then `state` at `rateHz`

Example: `just ws-example` (or `uv run --with websockets tools/ws_example_server.py`), then press Connect
on the page (or call `sim.connect()`). Using ROS coordinates, it scoops vanilla → eats → scoops mint → eats.

## Rust core (`rust/`)

The simulation is also available as a Rust library, `icecream-core`, plus a headless CLI. This is the base
for a native digital-twin app: direct ROS 2, batch and parallel runs, no browser.

- `core/` is a port of `src/sim/*.js` and `src/math.js`: the spoon controller, the SDF grids, the PBF
  particles, scooping, the volume ledger and the frames. The WebGPU renderer stays in JS for now.
- **Bit-for-bit parity with the JS version.** A scripted scoop (510 frames) and two plunge-and-eat cycles
  (900 frames) produce identical spoon poses, ice volumes, particle counts and ledger values in both. To get
  there, the Rust code:
  - stores f32 wherever JS uses a `Float32Array` and does the arithmetic in f64;
  - copies V8's `Math.hypot` algorithm;
  - ports V8's fdlibm `sin`, `cos` and `cbrt` (`v8math.rs`; the `libm` crate differs in the last bit for
    about 1% of inputs);
  - uses the same RNG and hash wrapping.
- The fixtures are written by `node tools/parity-fixture.mjs` and checked by `cargo test`. CI regenerates
  them with Node 22 (V8 12.4) and fails if they no longer match.
- The collision tests (no penetration, no tunnelling, sliding along the wall) and the mass-conservation tests
  are ported too.

```bash
just rust-test     # 13 tests: collision, mass, parity with JS
just parity        # regenerate the JS fixtures, then compare
just rust-bench    # ~6 ms/frame single-threaded, mass 2075.3 -> 2075.2 g
just rust-serve    # headless twin over stdin/stdout
```

**`icecream serve`** speaks the WebSocket bridge's messages as JSON lines (one object per line), so a ROS node
or a Python script can run it as a subprocess:

```text
-> {"type":"config","frame":"ros"}
-> {"type":"set_target","position":[-0.18,0,0.085],"quaternion":[0,0,0,1]}
-> {"type":"step","frames":200}
<- {"type":"state","spoon":{...},"scoopedGrams":11.7,"bowlGrams":{...},"blocked":false,...}
-> {"type":"eat"}
```

## Files

```
src/math.js            vectors / quaternions / matrices (WebGPU 0..1 depth)
src/sim/sdf.js         analytic SDFs for the bowls, spoon and table; the three flavours
src/sim/grid.js        narrow-band SDF volume + colour, smooth-min stamping, volume-preserving smoothing
src/sim/spoon.js       spoon controller (substeps, rigid contact projection, blocked)
src/sim/fluid.js       PBF particles
src/sim/sim.js         scoop / copy / freeze / tip / Eat, volume ledger, stats
src/gpu/               WebGPU renderer, WGSL, geometry, orbit camera
src/ui/panel.js        control panel
src/bridge.js          window.sim and the WebSocket bridge
tools/                 headless screenshots, WebSocket example controller, parity fixtures
rust/core/             Rust port of the simulation (lib icecream_core) + tests and JS fixtures
rust/cli/              `icecream serve` (JSON lines) and `icecream bench`
```

## Known limitations

- The simulation runs on the CPU (about 4–8 ms/frame). Particles are capped at 5000 by default; for much
  larger scenes `fluid.js` could move to a WebGPU compute shader.
- Ice cream doesn't stop the spoon, only slows it down (the cutting force is estimated from hardness × swept
  area, not real mechanics).
- Solid ice cream doesn't slowly sag once frozen.
- Headless Chrome needs a real GPU: `--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan`
  (this Chrome's SwiftShader WebGPU can't read back).
