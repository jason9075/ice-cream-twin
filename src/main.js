// The viewer. The simulation (spoon, ice cream, particles, SO-101) runs in the Rust server
// (`icecream web`); this page draws what it streams and sends it commands (src/net/remote.js).
import GUI from 'lil-gui';
import { Vector3, Quaternion, mat4 } from './math.js';
import { createRenderer, createFluidSurface } from './gpu/renderer.js';
import { OrbitCamera } from './gpu/camera.js';
import * as geo from './gpu/geometry.js';
import { Panel } from './ui/panel.js';
import { Remote, RemoteSim } from './net/remote.js';
import { createArmView } from './robot/render.js';
import './style.css';

const canvas = document.getElementById('gpu');

function fatal(msg, hint) {
  const el = document.getElementById('fatal');
  el.style.display = 'flex';
  el.innerHTML = `<div><b>${msg}</b><br><br>${hint}</div>`;
}

function webgpuHint() {
  const linux = /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);
  return linux
    ? `On Linux, Chrome keeps WebGPU behind flags. Either:<br>
       • open <code>chrome://flags</code>, enable <b>Unsafe WebGPU Support</b> and <b>Vulkan</b>, then relaunch; or<br>
       • start Chrome with <code>--enable-unsafe-webgpu --enable-features=Vulkan</code>
         (in this repo: <code>just chrome</code>).<br>
       Check <code>chrome://gpu</code> → “WebGPU: Hardware accelerated”.`
    : 'Use a recent Chrome / Edge (or Firefox / Safari with WebGPU enabled).';
}

/** ws://<this host>/ws (Vite proxies it to the server in development), or ?server=ws://host:port/ws. */
function serverUrl() {
  const q = new URLSearchParams(location.search).get('server');
  if (q) return q;
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
}

async function main() {
  let r;
  try {
    r = await createRenderer(canvas);
  } catch (e) {
    fatal(`Cannot start WebGPU<br>${e.message}`, webgpuHint());
    throw e;
  }

  // ------------------------------------------------------------------ connect
  const remote = new Remote(serverUrl());
  const hello = await new Promise((resolve) => {
    const t = setTimeout(
      () => fatal('Waiting for the simulation server…', `Start it with <code>just dev</code> (or <code>icecream web</code>), connecting to <code>${remote.url}</code>.`),
      3000,
    );
    remote.onHello = (h) => {
      clearTimeout(t);
      document.getElementById('fatal').style.display = 'none';
      resolve(h);
    };
    remote.connect();
  });
  const sim = new RemoteSim(remote, hello);
  const S = hello.scene;

  // ------------------------------------------------------------------ scene
  const sphereMesh = r.createMesh(geo.sphere(8, 12));
  const spoonMesh = r.createMesh(geo.spoon(S.spoon));
  const table = r.createObject(r.createMesh(geo.box([-0.7, -0.04, -0.45], [0.7, 0, 0.45])), { kind: 1, rough: 0.6 });
  const bowlMesh = r.createMesh(geo.lathe(S.bowls[0].profile));
  const tints = [
    [0.96, 0.93, 0.88],
    [0.97, 0.9, 0.91],
    [0.88, 0.95, 0.91],
  ];
  const bowls = S.bowls.map((B, i) => {
    const o = r.createObject(bowlMesh, { color: tints[i], rough: 0.12 });
    o.setModel(mat4.translation(B.cx, 0, B.cz));
    return o;
  });
  const spoon = r.createObject(spoonMesh, { color: [0.86, 0.88, 0.91], metal: 1, rough: 0.18 });
  const ghost = r.createObject(spoonMesh, { color: [0.16, 0.6, 0.96], rough: 0.5, opacity: 0.25 });

  // SDF volumes: the server streams their textures (quantised distance + colour) as patches
  const gridMeta = S.grids.map((g) => ({
    ox: g.origin[0], oy: g.origin[1], oz: g.origin[2], cell: g.cell, band: g.band,
    nx: g.dims[0], ny: g.dims[1], nz: g.dims[2],
    sx: g.dims[0] * g.cell, sy: g.dims[1] * g.cell, sz: g.dims[2] * g.cell,
  }));
  const g0 = gridMeta[0];
  const worldBox = r.createMesh(geo.box([0, 0, 0], [g0.sx, g0.sy, g0.sz]));
  const bowlVols = gridMeta.slice(0, 3).map((g) => r.createVolume(g, worldBox, { external: true }));
  const c = gridMeta[3];
  const contentVol = r.createVolume(c, r.createMesh(geo.box([0, 0, 0], [c.sx, c.sy, c.sz])), { external: true });
  const volumes = [...bowlVols, contentVol];

  const particles = r.createInstances(sphereMesh, S.particleCap);
  const spilled = r.createInstances(sphereMesh, 8000);
  // loose particles as one smooth surface (the balls themselves are a debug view)
  const fluid = createFluidSurface(r, particles);

  let armView = null;
  try {
    armView = await createArmView(r, S.arm);
  } catch (e) {
    console.error('SO-101 meshes failed to load:', e);
  }
  const armObjects = armView ? armView.objects : [];

  const scene = {
    objects: [table, ...bowls, spoon, ...armObjects],
    ghosts: [ghost],
    instances: [particles, spilled],
    overlays: [],
    volumes,
    fluids: [fluid],
    shadowCasters: { objects: [...bowls, spoon, ...armObjects], instances: [particles, spilled], volumes, fluids: [fluid] },
    debug: 0,
    shadows: true,
  };
  const light = { dir: new Vector3(0.45, 1.0, 0.35).normalize(), center: new Vector3(0, 0.08, -0.08), extent: 0.5 };

  // ------------------------------------------------------------------ frames from the server
  const radius = () => sim.params.fluid.particleRadius * 1.15;
  /** Fills the instance buffer; returns the bounds [lo, hi] of the n particles. */
  function writeInstances(inst, list) {
    const d = inst.data;
    const n = Math.min(list.n, inst.cap);
    const rad = radius();
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < n; k++) {
      const o = k * 16;
      for (let a = 0; a < 3; a++) {
        const v = list.data.getFloat32(o + a * 4, true);
        d[k * 8 + a] = v;
        if (v < lo[a]) lo[a] = v;
        if (v > hi[a]) hi[a] = v;
      }
      d[k * 8 + 3] = rad;
      d[k * 8 + 4] = list.data.getUint8(o + 12) / 255;
      d[k * 8 + 5] = list.data.getUint8(o + 13) / 255;
      d[k * 8 + 6] = list.data.getUint8(o + 14) / 255;
      d[k * 8 + 7] = list.data.getUint8(o + 15) / 255;
    }
    inst.upload(n);
    return [lo, hi];
  }
  let latest = null;
  remote.onFrame = (f) => {
    // grid patches apply in order as they arrive; poses and particles only matter for the next draw
    for (const p of f.patches) {
      const vol = volumes[p.grid];
      r.device.queue.writeTexture(
        { texture: p.kind === 0 ? vol.sdfTex : vol.colTex, origin: { x: p.x, y: p.y, z: p.z } },
        p.data,
        { bytesPerRow: p.sx * (p.kind === 0 ? 1 : 4), rowsPerImage: p.sy },
        { width: p.sx, height: p.sy, depthOrArrayLayers: p.sz },
      );
    }
    if (f.spilled) writeInstances(spilled, f.spilled);
    sim._frame(f);
    latest = f;
  };
  remote.startStream();
  remote.onStatus = (s) => {
    if (s === 'disconnected') fatal('Lost the simulation server', 'Reconnecting… (is <code>icecream web</code> still running?)');
    else document.getElementById('fatal').style.display = 'none';
  };
  await remote.waitSeq(1);

  // ------------------------------------------------------------------ camera + mouse target
  let dragPlaneY = 0;
  const camera = new OrbitCamera(canvas, {
    onTargetDrag(phase, ray) {
      if (phase === 'start') dragPlaneY = sim.spoon.targetPos.y;
      if (!ray || Math.abs(ray.dir.y) < 1e-4) return;
      const t = (dragPlaneY - ray.origin.y) / ray.dir.y;
      if (t <= 0) return;
      const p = new Vector3().copy(ray.origin).addScaledVector(ray.dir, t);
      sim.spoon.setTarget(new Vector3(p.x, dragPlaneY, p.z), null);
    },
    onTargetHeight(dy) {
      const p = sim.spoon.targetPos.clone();
      p.y += dy;
      sim.spoon.setTarget(p, null);
    },
  });

  // ------------------------------------------------------------------ actions, panel, API
  const toast = document.getElementById('toast');
  const showToast = (msg) => {
    toast.textContent = msg;
    toast.classList.add('on');
    clearTimeout(showToast.t);
    showToast.t = setTimeout(() => toast.classList.remove('on'), 1600);
  };
  const actions = {
    async eat() {
      const g = await sim.eat();
      showToast(g > 0.05 ? `Yum! ${g.toFixed(1)} g eaten` : 'Spoon is empty');
      return g;
    },
    release: () => sim.releaseContent(),
    reset: () => sim.reset(),
    home: () => sim.home(),
  };
  const panel = new Panel(sim, actions);
  let gui;
  let armFolder = null;

  /** window.sim: drive the server from the console or a headless test (see the README). */
  window.sim = {
    setTarget({ position, quaternion, frame = 'yup' } = {}) {
      remote.send({ type: 'set_target', position, quaternion, frame });
    },
    setJoints({ joints, unit = 'rad' } = {}) {
      remote.send({ type: 'set_joints', joints, unit });
    },
    /** The latest state (Y-up), or a Promise of a fresh one in `frame` ('yup' | 'ros'). */
    getState(frame) {
      return frame ? remote.request({ type: 'get_state', frame }) : sim.getState();
    },
    eat: () => actions.eat(),
    release: () => sim.releaseContent(),
    reset: () => sim.reset(),
    refill: () => sim.refill(),
    home: () => sim.home(),
    async setParams(patch) {
      await sim.setParams(patch);
      gui?.controllersRecursive().forEach((x) => x.updateDisplay());
      return sim.params;
    },
    getParams: () => JSON.parse(JSON.stringify(sim.params)),
    /** Arm settings: {enabled, constrainTarget, jointSpeed}. */
    arm: (cfg) => remote.request({ type: 'arm', ...cfg }),
    /** Pause the server's real-time loop (stepFrames still works). */
    pause: (v = true) => remote.request({ type: 'pause', paused: v }),
    /** Advance n frames on the server; resolves with the state once the page has drawn them. */
    async stepFrames(n = 1) {
      const st = await remote.request({ type: 'step', frames: n });
      await remote.waitSeq(st.seq);
      return sim.getState();
    },
  };

  gui = new GUI({ title: 'Parameters', width: 300 });
  {
    // every edit goes to the server (set_params); the server's answer is what the sliders show
    const P = sim.params;
    const ctl = (folder, group, prop, ...args) =>
      folder.add(P[group], prop, ...args).onFinishChange((v) => window.sim.setParams({ [group]: { [prop]: v } }));
    const f = gui.addFolder('Fluid (loose ice cream)');
    ctl(f, 'fluid', 'particleRadius', 0.0008, 0.0025, 0.0001).name('particle radius (m)');
    ctl(f, 'fluid', 'viscosity', 0, 1, 0.01).name('viscosity (XSPH)');
    ctl(f, 'fluid', 'cohesion', 0, 1, 0.01).name('cohesion / stickiness');
    ctl(f, 'fluid', 'friction', 0, 1, 0.01);
    ctl(f, 'fluid', 'damping', 0, 5, 0.05).name('damping (1/s)');
    ctl(f, 'fluid', 'gravity', 0, 20, 0.1).name('gravity (m/s²)');
    ctl(f, 'fluid', 'iterations', 1, 8, 1).name('solver iterations');
    ctl(f, 'fluid', 'substeps', 1, 6, 1);
    ctl(f, 'fluid', 'kernelScale', 1.4, 2.5, 0.05).name('kernel scale');
    ctl(f, 'fluid', 'maxParticles', 500, 15000, 100).name('max particles');
    const fr = f.addFolder('Freezing (particle → solid)');
    ctl(fr, 'fluid', 'stopSpeed', 0.005, 0.15, 0.005).name('rest speed (m/s)');
    ctl(fr, 'fluid', 'restFrames', 1, 60, 1).name('rest frames');
    ctl(fr, 'fluid', 'stampScale', 0.8, 2.5, 0.05).name('stamp radius ×');
    ctl(fr, 'fluid', 'stampSmooth', 0, 0.004, 0.0001).name('stamp smoothing (m)');
    ctl(fr, 'fluid', 'stampOnContact').name('freeze on first contact');
    fr.close();
    f.close(); // keep the arm's joint sliders in view
    const i = gui.addFolder('Ice cream');
    ctl(i, 'ice', 'hardness', 0, 120, 1).name('hardness (kPa)');
    ctl(i, 'ice', 'accretion').name('scoop curls (accretion)');
    ctl(i, 'ice', 'accretionDist', 0.0002, 0.003, 0.0001).name('accretion step (m)');
    ctl(i, 'ice', 'capacity', 5, 60, 1).name('capacity (cm³)');
    ctl(i, 'ice', 'releaseAngle', 60, 180, 1).name('slide-off tilt (°)');
    ctl(i, 'ice', 'displacedToParticles').name('displaced → crumbs');
    ctl(i, 'ice', 'smoothing', 0, 1, 0.05).name('surface smoothing');
    ctl(i, 'ice', 'density', 0.3, 1.2, 0.01).name('density (g/cm³)');
    ctl(i, 'ice', 'shape', ['scoops', 'flat', 'mound']).name('refill shape');
    i.add({ refill: () => sim.refill() }, 'refill').name('Refill bowls');
    const s = gui.addFolder('Spoon controller');
    ctl(s, 'spoon', 'maxLinSpeed', 0.01, 0.5, 0.01).name('max speed (m/s)');
    ctl(s, 'spoon', 'maxAngSpeed', 10, 360, 5).name('max ang speed (°/s)');
    ctl(s, 'spoon', 'maxCutForce', 1, 100, 1).name('max cut force (N)');

    const a = gui.addFolder('Robot arm (SO-101)');
    const armUI = { enabled: true, constrainTarget: true };
    a.add(armUI, 'enabled').name('arm holds the spoon').onChange((v) => window.sim.arm({ enabled: v }));
    a.add(armUI, 'constrainTarget').name('limit target to reach').onChange((v) => window.sim.arm({ constrainTarget: v }));
    const names = S.arm.jointNames;
    // joint limits (deg) of the SO-101 URDF, for the sliders
    const limits = [[-110, 110], [-100, 100], [-96.8, 96.8], [-95, 95], [-157.2, 162.8]];
    const jointUI = Object.fromEntries(names.map((n) => [n, 0]));
    let jointEdited = 0; // while a joint slider is dragged, don't overwrite it with the actual angle
    const jointCtl = names.map((n, k) =>
      a
        .add(jointUI, n, limits[k][0], limits[k][1], 0.5)
        .name(`${n} (°)`)
        .onChange(() => {
          jointEdited = performance.now();
          window.sim.setJoints({ joints: names.map((m) => jointUI[m]), unit: 'deg' });
        }),
    );
    const armInfo = { error: '' };
    const errCtl = a.add(armInfo, 'error').name('tracking error').disable();
    a.syncJoints = (st) => {
      if (!st?.arm) return;
      if (armUI.enabled !== st.arm.enabled) {
        armUI.enabled = st.arm.enabled;
        a.controllers[0].updateDisplay();
      }
      if (performance.now() - jointEdited > 1500)
        names.forEach((n, k) => {
          jointUI[n] = Math.round(st.arm.jointsDeg[k] * 10) / 10;
          jointCtl[k].updateDisplay();
        });
      armInfo.error = `${st.arm.trackingErrorMm.toFixed(2)} mm / ${st.arm.trackingErrorDeg.toFixed(2)}°`;
      errCtl.updateDisplay();
    };
    armFolder = a;

  }

  // ------------------------------------------------------------------ View panel (rendering + debug)
  const view = {
    particles: 'surface', // surface | balls | both | hidden
    surfaceRadius: 1.3, // × particle radius
    surfaceKernel: 3.0, // × particle radius
    surfaceVoxel: 0.8, // × particle radius
    tintFluid: false,
    crumbs: true,
    bowlIce: true,
    spoonIce: true,
    bowls: true,
    spoon: true,
    arm: true,
    table: true,
    ghost: true,
    shadows: true,
    shading: 'shaded',
    gridBoxes: false,
  };
  const SHADING = { shaded: 0, normals: 1, 'albedo (unlit)': 2, 'march steps': 3 };
  const fluidInfo = { status: '' };

  function updateFluid(n, lo, hi) {
    const pr = sim.params.fluid.particleRadius;
    fluid.update(n, lo, hi, { r: view.surfaceRadius * pr, R: view.surfaceKernel * pr, voxel: view.surfaceVoxel * pr, tint: view.tintFluid });
    const i = fluid.info;
    fluidInfo.status = n ? `${n} particles, ${i.dims.join('×')} @ ${(i.cell * 1000).toFixed(2)} mm` : 'no loose particles';
  }
  function applyView() {
    const m = view.particles;
    particles.visible = m === 'balls' || m === 'both';
    fluid.vol.visible = m === 'surface' || m === 'both';
    spilled.visible = view.crumbs;
    bowlVols.forEach((x) => (x.visible = view.bowlIce));
    contentVol.visible = view.spoonIce;
    bowls.forEach((x) => (x.visible = view.bowls));
    spoon.visible = view.spoon;
    armObjects.forEach((x) => (x.visible = view.arm));
    table.visible = view.table;
    ghost.visible = view.ghost;
    scene.shadows = view.shadows;
    scene.debug = SHADING[view.shading];
    gridBoxes.forEach((b) => (b.visible = view.gridBoxes));
    // re-fit with the new surface settings even while paused
    if (latest) {
      const [lo, hi] = writeInstances(particles, latest.particles);
      updateFluid(latest.particles.n, lo, hi);
    }
  }

  // translucent boxes showing what each SDF grid (and the fluid surface's grid) covers
  const unitCube = r.createMesh(geo.box([0, 0, 0], [1, 1, 1]));
  const boxColors = [[1, 0.6, 0.2], [1, 0.6, 0.2], [1, 0.6, 0.2], [0.3, 0.9, 0.3], [0.2, 0.5, 1]];
  const gridBoxes = boxColors.map((color) => r.createObject(unitCube, { color, rough: 1, opacity: 0.15 }));
  scene.ghosts.push(...gridBoxes);
  function syncBoxes() {
    if (!view.gridBoxes) return;
    const f = latest;
    gridMeta.forEach((g, k) => {
      const box = mat4.multiply(mat4.translation(g.ox, g.oy, g.oz), scale(g.sx, g.sy, g.sz));
      if (k < 3) gridBoxes[k].setModel(box);
      else if (f) {
        const sm = mat4.compose(new Vector3(f.spoon[0], f.spoon[1], f.spoon[2]), new Quaternion(f.spoon[3], f.spoon[4], f.spoon[5], f.spoon[6]));
        gridBoxes[3].setModel(mat4.multiply(sm, box));
      }
    });
    const fb = fluid.box;
    gridBoxes[4].visible = !!fb;
    if (fb) gridBoxes[4].setModel(mat4.multiply(mat4.translation(...fb.origin), scale(...fb.size)));
  }
  function scale(x, y, z) {
    const m = mat4.identity();
    m[0] = x;
    m[5] = y;
    m[10] = z;
    return m;
  }

  function viewFolder(v) {
    const on = () => applyView();
    v.add(view, 'particles', ['surface', 'balls', 'both', 'hidden']).name('loose particles').onChange(on);
    const sf = v.addFolder('Particle surface');
    sf.add(view, 'surfaceRadius', 0.5, 3, 0.05).name('radius × r').onChange(on);
    sf.add(view, 'surfaceKernel', 1.5, 6, 0.1).name('blend kernel × r').onChange(on);
    sf.add(view, 'surfaceVoxel', 0.3, 2, 0.05).name('voxel × r').onChange(on);
    sf.add(fluidInfo, 'status').name('grid').disable().listen();
    const show = v.addFolder('Show');
    show.add(view, 'crumbs').name('crumbs (spilled)').onChange(on);
    show.add(view, 'bowlIce').name('ice cream in bowls').onChange(on);
    show.add(view, 'spoonIce').name('ice cream on spoon').onChange(on);
    show.add(view, 'bowls').onChange(on);
    show.add(view, 'spoon').onChange(on);
    show.add(view, 'arm').name('SO-101').onChange(on);
    show.add(view, 'table').onChange(on);
    show.add(view, 'ghost').name('target ghost').onChange(on);
    show.add(view, 'shadows').onChange(on);
    const dbg = v.addFolder('Debug');
    dbg.add(view, 'shading', Object.keys(SHADING)).name('shading').onChange(on);
    dbg.add(view, 'tintFluid').name('tint particle surface').onChange(on);
    dbg.add(view, 'gridBoxes').name('grid bounds').onChange(on);
    dbg.add({ reset: () => actions.reset() }, 'reset').name('Reset everything');
    v.close();
  }
  viewFolder(gui.addFolder('View'));
  applyView();

  // ------------------------------------------------------------------ loop
  let last = performance.now();
  let fps = 60;
  let uiTimer = 0;
  let drawn = -1;

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    fps += (1 / Math.max(dt, 1e-3) - fps) * 0.05;
    panel.applyJog(dt);
    const f = latest;
    if (f && f.seq !== drawn) {
      drawn = f.seq;
      armView?.update(f.links, f.armOn);
      const spoonModel = mat4.compose(new Vector3(f.spoon[0], f.spoon[1], f.spoon[2]), new Quaternion(f.spoon[3], f.spoon[4], f.spoon[5], f.spoon[6]));
      spoon.setModel(spoonModel);
      // the ghost shows where the spoon is heading: the arm's goal (reachable), or the raw target
      ghost.setModel(mat4.compose(new Vector3(f.ghost[0], f.ghost[1], f.ghost[2]), new Quaternion(f.ghost[3], f.ghost[4], f.ghost[5], f.ghost[6])));
      contentVol.setModel(mat4.multiply(spoonModel, mat4.translation(c.ox, c.oy, c.oz)));
      const [lo, hi] = writeInstances(particles, f.particles);
      updateFluid(f.particles.n, lo, hi);
    }
    syncBoxes();
    r.render(scene, camera, light);
    uiTimer += dt;
    if (uiTimer > 0.1 && sim.getState()) {
      uiTimer = 0;
      panel.update(fps);
      armFolder?.syncJoints(sim.getState());
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  window.__view = {
    camera,
    renderer: r,
    remote,
    /** How loose particles are drawn: 'surface' | 'balls' | 'both' | 'hidden' (true = 'surface'). */
    showParticles(v = true) {
      const mode = v === true ? 'surface' : v === false ? 'hidden' : v;
      gui.controllersRecursive().find((x) => x.property === 'particles')?.setValue(mode);
    },
    /** Change View settings (same names as the panel's `view` object), e.g. {shading: 'normals'}. */
    setView(patch) {
      Object.assign(view, patch);
      gui.controllersRecursive().forEach((x) => x.updateDisplay());
      applyView();
      return { ...view };
    },
    fluid,
    /** The SO-101's gripper_link world matrix (row-major), from the last frame. */
    linkMatrix(name) {
      const k = S.arm.links.indexOf(name);
      return latest ? Array.from(latest.links.subarray(k * 16, k * 16 + 16)) : null;
    },
    /** Offscreen render -> PNG data URL (headless tests, virtual camera). */
    async capturePNG(width = 1280, height = 800) {
      const img = await r.capture(scene, camera, light, width, height);
      const cv = document.createElement('canvas');
      cv.width = width;
      cv.height = height;
      cv.getContext('2d').putImageData(new ImageData(img.pixels, width, height), 0, 0);
      return cv.toDataURL('image/png');
    },
  };
}

main();
