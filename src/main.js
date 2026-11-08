import GUI from 'lil-gui';
import { Vector3, Quaternion, mat4 } from './math.js';
import { Sim } from './sim/sim.js';
import { defaultParams } from './sim/params.js';
import { BOWLS, bowlProfile } from './sim/sdf.js';
import { createRenderer } from './gpu/renderer.js';
import { OrbitCamera } from './gpu/camera.js';
import * as geo from './gpu/geometry.js';
import { Panel } from './ui/panel.js';
import { installApi, WsBridge } from './bridge.js';
import { Arm } from './robot/arm.js';
import { createArmView } from './robot/render.js';
import { ARM_JOINTS, LIMITS } from './robot/so101.js';
import { HOME_POS, HOME_QUAT } from './sim/spoon.js';
import './style.css';

const canvas = document.getElementById('gpu');

function fatal(msg) {
  const el = document.getElementById('fatal');
  el.style.display = 'flex';
  const linux = /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);
  const hint = linux
    ? `On Linux, Chrome keeps WebGPU behind flags. Either:<br>
       • open <code>chrome://flags</code>, enable <b>Unsafe WebGPU Support</b> and <b>Vulkan</b>, then relaunch; or<br>
       • start Chrome with <code>--enable-unsafe-webgpu --enable-features=Vulkan</code>
         (in this repo: <code>just chrome</code>).<br>
       Check <code>chrome://gpu</code> → “WebGPU: Hardware accelerated”.`
    : 'Use a recent Chrome / Edge (or Firefox / Safari with WebGPU enabled).';
  el.innerHTML = `<div><b>Cannot start WebGPU</b><br>${msg}<br><br>${hint}</div>`;
}

async function main() {
  let r;
  try {
    r = await createRenderer(canvas);
  } catch (e) {
    fatal(e.message);
    throw e;
  }

  // ------------------------------------------------------------------ simulation
  const params = defaultParams();
  const sim = new Sim(params);
  // The SO-101 holds the spoon: targets are projected onto what it can reach, and it follows the spoon.
  const arm = new Arm();
  arm.reset(sim);
  const step = (dt) => {
    arm.constrain(sim);
    sim.step(dt);
    arm.follow(sim);
  };

  // ------------------------------------------------------------------ scene
  const sphereMesh = r.createMesh(geo.sphere(8, 12));
  const spoonMesh = r.createMesh(geo.spoon());
  const table = r.createObject(r.createMesh(geo.box([-0.7, -0.04, -0.45], [0.7, 0, 0.45])), { kind: 1, rough: 0.6 });
  const bowlMesh = r.createMesh(geo.lathe(bowlProfile({ ...BOWLS[0], cx: 0 })));
  const tints = [
    [0.96, 0.93, 0.88],
    [0.97, 0.9, 0.91],
    [0.88, 0.95, 0.91],
  ];
  const bowls = BOWLS.map((B, i) => {
    const o = r.createObject(bowlMesh, { color: tints[i], rough: 0.12 });
    o.setModel(mat4.translation(B.cx, 0, B.cz));
    return o;
  });
  const spoon = r.createObject(spoonMesh, { color: [0.86, 0.88, 0.91], metal: 1, rough: 0.18 });
  const ghost = r.createObject(spoonMesh, { color: [0.16, 0.6, 0.96], rough: 0.5, opacity: 0.25 });

  const g0 = sim.grids[0];
  const worldBox = r.createMesh(geo.box([0, 0, 0], [g0.sx, g0.sy, g0.sz]));
  const bowlVols = sim.grids.map((g) => r.createVolume(g, worldBox));
  const c = sim.content;
  const contentVol = r.createVolume(c, r.createMesh(geo.box([0, 0, 0], [c.sx, c.sy, c.sz])));

  const particles = r.createInstances(sphereMesh, sim.fluid.cap);
  const spilled = r.createInstances(sphereMesh, sim.spilled.length / 3);
  const samples = r.createInstances(sphereMesh, sim.spoon.nSamples, { unlit: true });
  samples.visible = false;

  let armView = null;
  try {
    armView = await createArmView(r, arm);
  } catch (e) {
    console.error('SO-101 meshes failed to load:', e);
  }
  const armObjects = armView ? armView.objects : [];

  const scene = {
    objects: [table, ...bowls, spoon, ...armObjects],
    ghosts: [ghost],
    instances: [particles, spilled],
    overlays: [samples],
    volumes: [...bowlVols, contentVol],
    shadowCasters: { objects: [...bowls, spoon, ...armObjects], instances: [particles, spilled], volumes: [...bowlVols, contentVol] },
  };
  const light = { dir: new Vector3(0.45, 1.0, 0.35).normalize(), center: new Vector3(0, 0.08, -0.08), extent: 0.5 };

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

  // ------------------------------------------------------------------ actions, UI, bridge
  let paused = false;
  const toast = document.getElementById('toast');
  const showToast = (msg) => {
    toast.textContent = msg;
    toast.classList.add('on');
    clearTimeout(showToast.t);
    showToast.t = setTimeout(() => toast.classList.remove('on'), 1600);
  };
  const actions = {
    eat() {
      const g = sim.eat();
      showToast(g > 0.05 ? `Yum! ${g.toFixed(1)} g eaten` : 'Spoon is empty');
    },
    release: () => sim.releaseContent(),
    reset: () => {
      sim.reset();
      if (arm.enabled) arm.reset(sim);
    },
    home: () => (arm.enabled ? sim.spoon.setTarget(arm.home.pos, arm.home.quat) : sim.spoon.setTarget(HOME_POS, HOME_QUAT)),
    pause: (v) => (paused = v),
    step,
    arm,
  };
  const panel = new Panel(sim, actions);
  let gui;
  let armFolder = null;
  const api = installApi(sim, { ...actions, paramsChanged: () => gui.controllersRecursive().forEach((x) => x.updateDisplay()) });
  const bridge = new WsBridge(api);
  /** Connect to an external controller, e.g. sim.connect('ws://localhost:8765', 'ros'). */
  api.connect = (url = 'ws://localhost:8765', frame = 'yup') => {
    bridge.frame = frame;
    bridge.connect(url);
  };
  api.disconnect = () => bridge.disconnect();

  gui = new GUI({ title: 'Parameters', width: 300 });
  {
    const f = gui.addFolder('Fluid (loose ice cream)');
    const p = params.fluid;
    f.add(p, 'particleRadius', 0.0008, 0.0025, 0.0001).name('particle radius (m)');
    f.add(p, 'viscosity', 0, 1, 0.01).name('viscosity (XSPH)');
    f.add(p, 'cohesion', 0, 1, 0.01).name('cohesion / stickiness');
    f.add(p, 'friction', 0, 1, 0.01);
    f.add(p, 'damping', 0, 5, 0.05).name('damping (1/s)');
    f.add(p, 'gravity', 0, 20, 0.1).name('gravity (m/s²)');
    f.add(p, 'iterations', 1, 8, 1).name('solver iterations');
    f.add(p, 'substeps', 1, 6, 1);
    f.add(p, 'kernelScale', 1.4, 2.5, 0.05).name('kernel scale');
    f.add(p, 'maxParticles', 500, 15000, 100).name('max particles');
    const fr = f.addFolder('Freezing (particle → solid)');
    fr.add(p, 'stopSpeed', 0.005, 0.15, 0.005).name('rest speed (m/s)');
    fr.add(p, 'restFrames', 1, 60, 1).name('rest frames');
    fr.add(p, 'stampScale', 0.8, 2.5, 0.05).name('stamp radius ×');
    fr.add(p, 'stampSmooth', 0, 0.004, 0.0001).name('stamp smoothing (m)');
    fr.add(p, 'stampOnContact').name('freeze on first contact');
    fr.close();
    f.close(); // keep the arm's joint sliders in view
    const i = gui.addFolder('Ice cream');
    const ip = params.ice;
    i.add(ip, 'hardness', 0, 120, 1).name('hardness (kPa)');
    i.add(ip, 'accretion').name('scoop curls (accretion)');
    i.add(ip, 'accretionDist', 0.0002, 0.003, 0.0001).name('accretion step (m)');
    i.add(ip, 'capacity', 5, 60, 1).name('capacity (cm³)');
    i.add(ip, 'releaseAngle', 60, 180, 1).name('slide-off tilt (°)');
    i.add(ip, 'displacedToParticles').name('displaced → crumbs');
    i.add(ip, 'smoothing', 0, 1, 0.05).name('surface smoothing');
    i.add(ip, 'density', 0.3, 1.2, 0.01).name('density (g/cm³)');
    i.add(ip, 'shape', ['scoops', 'flat', 'mound']).name('refill shape');
    i.add({ refill: () => sim.refill() }, 'refill').name('Refill bowls');
    const s = gui.addFolder('Spoon controller');
    const sp = params.spoon;
    s.add(sp, 'maxLinSpeed', 0.01, 0.5, 0.01).name('max speed (m/s)');
    s.add(sp, 'maxAngSpeed', 10, 360, 5).name('max ang speed (°/s)');
    s.add(sp, 'maxCutForce', 1, 100, 1).name('max cut force (N)');
    const a = gui.addFolder('Robot arm (SO-101)');
    a.add(arm, 'enabled').name('arm holds the spoon').onChange((x) => {
      if (x) arm.reset(sim);
    });
    a.add(arm, 'constrainTarget').name('limit target to reach');
    const deg = (v) => (v * 180) / Math.PI;
    const jointUI = Object.fromEntries(ARM_JOINTS.map((n, k) => [n, deg(arm.q[k])]));
    let jointEdited = 0; // while the user drags a joint slider, don't overwrite it with the actual angle
    const jointCtl = ARM_JOINTS.map((n, k) =>
      a
        .add(jointUI, n, deg(LIMITS[k][0]), deg(LIMITS[k][1]), 0.5)
        .name(`${n} (°)`)
        .onChange(() => {
          jointEdited = performance.now();
          arm.setJoints(sim, ARM_JOINTS.map((m) => (jointUI[m] * Math.PI) / 180));
        }),
    );
    const armInfo = { error: '' };
    const errCtl = a.add(armInfo, 'error').name('tracking error').disable();
    a.syncJoints = () => {
      if (performance.now() - jointEdited > 1500)
        ARM_JOINTS.forEach((n, k) => {
          jointUI[n] = Math.round(deg(arm.q[k]) * 10) / 10;
          jointCtl[k].updateDisplay();
        });
      armInfo.error = `${arm.err.mm.toFixed(2)} mm / ${arm.err.deg.toFixed(2)}°`;
      errCtl.updateDisplay();
    };
    armFolder = a;
    const v = gui.addFolder('View');
    const view = { ghost: true, samples: false, particles: false };
    particles.visible = view.particles;
    v.add(view, 'ghost').name('target ghost').onChange((x) => (ghost.visible = x));
    v.add(view, 'samples').name('collision samples').onChange((x) => (samples.visible = x));
    v.add(view, 'particles').onChange((x) => (particles.visible = x));
    v.add({ reset: () => actions.reset() }, 'reset').name('Reset everything');
    v.close();
    const b = gui.addFolder('Digital-twin bridge (WebSocket)');
    const bs = { url: 'ws://localhost:8765', frame: 'yup', status: 'disconnected' };
    b.add(bs, 'url');
    b.add(bs, 'frame', ['yup', 'ros']).onChange((x) => (bridge.frame = x));
    b.add({ connect: () => bridge.connect(bs.url) }, 'connect').name('Connect');
    b.add({ disconnect: () => bridge.disconnect() }, 'disconnect').name('Disconnect');
    const st = b.add(bs, 'status').disable();
    bridge.onStatus = (x) => {
      bs.status = x;
      st.updateDisplay();
    };
    b.close();
  }

  // ------------------------------------------------------------------ loop
  const STEP = 1 / 60;
  let last = performance.now();
  let acc = 0;
  let fps = 60;
  let uiTimer = 0;
  let spilledVersion = -1;
  const w = new Vector3();
  const q = new Quaternion();

  function writeParticles(inst, n, pos, rgba, radius) {
    const d = inst.data;
    for (let k = 0; k < n; k++) {
      d[k * 8] = pos[k * 3];
      d[k * 8 + 1] = pos[k * 3 + 1];
      d[k * 8 + 2] = pos[k * 3 + 2];
      d[k * 8 + 3] = radius;
      d[k * 8 + 4] = rgba[k * 4] / 255;
      d[k * 8 + 5] = rgba[k * 4 + 1] / 255;
      d[k * 8 + 6] = rgba[k * 4 + 2] / 255;
    }
    inst.upload(n);
  }

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    fps += (1 / Math.max(dt, 1e-3) - fps) * 0.05;
    if (!paused) {
      acc += dt;
      let n = 0;
      while (acc >= STEP && n < 2) {
        panel.applyJog(STEP);
        step(STEP);
        acc -= STEP;
        n++;
      }
      if (n === 2) acc = 0; // fall behind gracefully instead of spiralling
    }

    armView?.update();
    const sp = sim.spoon;
    const spoonModel = mat4.compose(sp.pos, sp.quat);
    spoon.setModel(spoonModel);
    // the ghost shows where the spoon is heading: the arm's goal (reachable), or the raw target
    if (arm.enabled) ghost.setModel(mat4.compose(arm.goal.pos, arm.goal.quat));
    else ghost.setModel(mat4.compose(sp.targetPos, sp.targetQuat));
    contentVol.setModel(mat4.multiply(spoonModel, mat4.translation(c.ox, c.oy, c.oz)));
    bowlVols.forEach((v) => v.sync());
    contentVol.sync();

    const f = sim.fluid;
    const rad = (f.r > 0 ? f.r : params.fluid.particleRadius) * 1.15;
    writeParticles(particles, f.n, f.pos, f.rgba, rad);
    if (sim.spilledVersion !== spilledVersion) {
      spilledVersion = sim.spilledVersion;
      writeParticles(spilled, sim.nSpilled, sim.spilled, sim.spilledRGBA, rad);
    }
    if (samples.visible) {
      const s = sp.samples;
      const d = samples.data;
      q.copy(sp.quat);
      for (let k = 0; k < sp.nSamples; k++) {
        w.set(s[k * 4], s[k * 4 + 1], s[k * 4 + 2]).applyQuaternion(q).add(sp.pos);
        const hit = sp.sampleContact[k];
        d.set([w.x, w.y, w.z, 0.0007, hit ? 1 : 0.1, hit ? 0.15 : 0.8, hit ? 0.1 : 1, 0], k * 8);
      }
      samples.upload(sp.nSamples);
    }

    r.render(scene, camera, light);
    uiTimer += dt;
    if (uiTimer > 0.1) {
      uiTimer = 0;
      panel.update(fps);
      armFolder?.syncJoints();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  window.__view = {
    camera,
    arm,
    renderer: r,
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
