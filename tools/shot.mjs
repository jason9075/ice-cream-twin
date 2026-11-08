// Headless smoke test: load the page served by the Rust server, drive a scripted scoop through window.sim
// (commands go to the server, frames come back), save screenshots.
//   just screenshots      (builds the page, runs `icecream web --dist dist`, then this script)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const url = process.argv[2] || 'http://localhost:4173/';
const out = process.argv[3] || 'shots';
fs.mkdirSync(out, { recursive: true });
const chrome = process.env.CHROME || '/usr/bin/google-chrome';

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,PdfOopif', '--use-angle=vulkan', '--window-size=1400,860', '--no-sandbox'],
  // puppeteer's own --enable-features would override ours (last switch wins)
  ignoreDefaultArgs: ['--enable-features=PdfOopif'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 860 });
const errors = [];
// WebGPU canvases are not captured by headless screenshots, so grab an offscreen render instead
async function shot(name) {
  const url = await page.evaluate(() => window.__view.capturePNG(1400, 860));
  fs.writeFileSync(`${out}/${name}`, Buffer.from(url.split(',')[1], 'base64'));
  await page.screenshot({ path: `${out}/ui-${name}` });
}
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${m.text()}`);
});
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.sim && window.__view, { timeout: 30000 });
await new Promise((r) => setTimeout(r, 1500));
await shot('01-start.png');

// scripted scoop in the strawberry bowl (middle), run headless-fast with stepFrames. The SO-101 stands
// behind the bowls, so the handle points toward -Z (rotate +90° about Y), then pitch the cup nose-down.
const state = await page.evaluate(async () => {
  const s = window.sim;
  await s.pause(true);
  const q = (pitch) => {
    const a = Math.PI / 4; // half of 90° about Y
    const b = pitch / 2; // about the spoon's own Z
    return [Math.sin(a) * Math.sin(b), Math.sin(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.cos(a) * Math.cos(b)];
  };
  const go = (p, quat, n) => {
    s.setTarget({ position: p, quaternion: quat });
    return s.stepFrames(n);
  };
  // stepFrames resolves once the server has stepped and the page has drawn the result
  await go([0, 0.14, 0.02], q(0.5), 80);
  await go([0, 0.086, 0.015], q(0.5), 150);
  await go([0, 0.088, -0.025], q(0), 150);
  return go([0, 0.15, -0.03], q(0), 100);
});
await page.evaluate(() => window.sim.pause(false));
await new Promise((r) => setTimeout(r, 800));
await shot('02-scooped.png');

// close-up of the spoon
await page.evaluate(() => {
  const { camera } = window.__view;
  camera.lookFrom({ x: 0.2, y: 0.26, z: 0.16 }, { x: 0, y: 0.13, z: -0.08 });
});
await new Promise((r) => setTimeout(r, 500));
await shot('03-closeup.png');

// close-up of the gripper clamping the handle
await page.evaluate(() => {
  const { camera } = window.__view;
  const g = window.__view.linkMatrix('gripper_link');
  const at = { x: g[3] - g[2] * 0.08, y: g[7] - g[6] * 0.08, z: g[11] - g[10] * 0.08 }; // ~fingertips
  camera.lookFrom({ x: at.x + 0.09, y: at.y + 0.05, z: at.z + 0.08 }, at);
});
await new Promise((r) => setTimeout(r, 500));
await shot('04-gripper.png');

// roll the spoon over the bowl (quickly): past the slide-off tilt the scoop drops off in one piece
await page.evaluate(async () => {
  await window.sim.pause(true);
  await window.sim.setParams({ spoon: { maxAngSpeed: 360 } });
});
const rollAbout = (deg) => {
  const h = (deg * Math.PI) / 360; // about world Z (the handle points toward -Z)
  return [0, 0, Math.sin(h), Math.cos(h)];
};
const rolled = await page.evaluate(async (q) => {
  const s = window.sim;
  const [ax, ay, az, aw] = q;
  const b = [0, Math.SQRT1_2, 0, Math.SQRT1_2]; // q(0): 90° about Y
  const m = [aw * b[0] + ax * b[3] + ay * b[2] - az * b[1], aw * b[1] - ax * b[2] + ay * b[3] + az * b[0], aw * b[2] + ax * b[1] - ay * b[0] + az * b[3], aw * b[3] - ax * b[0] - ay * b[1] - az * b[2]];
  s.setTarget({ position: [0, 0.15, -0.03], quaternion: m });
  // the arm turns at its joint speed: step until the scoop slides off
  let st;
  let frames = 0;
  do {
    st = await s.stepFrames(5);
    frames += 5;
  } while (st.scoopedGrams > 0.5 && frames < 300);
  return frames;
}, rollAbout(150));
console.log(`the scoop slid off after ${rolled} frames of rolling`);
await page.evaluate(() => window.__view.camera.lookFrom({ x: 0.05, y: 0.27, z: 0.17 }, { x: 0, y: 0.1, z: -0.03 }));
for (const [name, frames] of [['05-drop.png', 4], ['06-falling.png', 4], ['07-landed.png', 60]]) {
  const st = await page.evaluate((n) => window.sim.stepFrames(n), frames);
  await new Promise((r) => setTimeout(r, 300));
  await shot(name);
  console.log(`${name}: ${st.particles} loose particles, ${st.scoopedGrams.toFixed(1)} g on the spoon, tilt ${(st.spoon?.tiltDeg ?? st.tiltDeg)?.toFixed(0)}°`);
}


// the same frame through the View panel's debug options
const views = [
  ['08-balls.png', { particles: 'balls' }],
  ['09-debug-tint.png', { particles: 'surface', tintFluid: true, gridBoxes: true }],
  ['10-debug-normals.png', { tintFluid: false, gridBoxes: false, shading: 'normals' }],
  ['11-debug-steps.png', { shading: 'march steps' }],
];
for (const [name, patch] of views) {
  await page.evaluate((p) => window.__view.setView(p), patch);
  await new Promise((r) => setTimeout(r, 300));
  await shot(name);
}
await page.evaluate(() => window.__view.setView({ shading: 'shaded' }));
console.log(`fluid grid: ${await page.evaluate(() => JSON.stringify(window.__view.fluid.info))}`);

console.log(JSON.stringify({ arm: { joints: state.arm?.jointsDeg?.map((v) => +v.toFixed(1)), errMm: state.arm?.trackingErrorMm }, scoopedGrams: state.scoopedGrams, particles: state.particles, bowls: state.bowlGrams, blocked: state.blocked }, null, 1));
console.log(errors.length ? errors.join('\n') : 'no console errors');
await browser.close();
