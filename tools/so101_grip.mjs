// Find the clamped jaw angle for the 'so101-gripper-clamp' tool module (src/robot/tool.js).
//   node tools/so101_grip.mjs
// The handle (radius SPOON.handleR) lies along gripper_link Z against the fixed jaw's fingertip pad.
// Sweep the moving jaw closed from open until it first touches the handle.
import fs from 'node:fs';
import { forward, m4 } from '../src/robot/so101.js';
import { SPOON } from '../src/sim/sdf.js';

const dir = new URL('../public/so101/', import.meta.url);
const meta = JSON.parse(fs.readFileSync(new URL('so101.json', dir)));
const bin = fs.readFileSync(new URL('so101.bin', dir));
const points = (link) => {
  const e = meta.links[link].find((x) => x.material === '3d_printed');
  const a = new Float32Array(bin.buffer.slice(bin.byteOffset + e.vOffset, bin.byteOffset + e.vOffset + e.vBytes));
  const out = [];
  for (let i = 0; i < a.length; i += 6) out.push([a[i], a[i + 1], a[i + 2]]);
  return out;
};
const r = SPOON.handleR;
const fingers = (p) => Math.abs(p[1]) < 0.006 && p[2] < -0.065 && p[2] > -0.102;

// fixed jaw (part of gripper_link): the inner face is the largest x below 0 in the fingertip region
const pad = Math.max(...points('gripper_link').filter((p) => fingers(p) && p[0] < 0 && p[2] < -0.093).map((p) => p[0]));
const axisX = pad + r;
console.log(`fixed fingertip pad x = ${(pad * 1000).toFixed(1)} mm -> handle axis x = ${(axisX * 1000).toFixed(1)} mm`);

const jaw = points('moving_jaw_so101_v1_link');
const q = [0, 0, 0, 0, 0];
for (let a = 0.6; a > -0.18; a -= 0.0005) {
  const { T } = forward(q, a);
  const M = m4.mul(m4.invertRigid(T.gripper_link), T.moving_jaw_so101_v1_link);
  let gap = Infinity;
  for (const v of jaw) {
    const x = M[0] * v[0] + M[1] * v[1] + M[2] * v[2] + M[3];
    const y = M[4] * v[0] + M[5] * v[1] + M[6] * v[2] + M[7];
    const z = M[8] * v[0] + M[9] * v[1] + M[10] * v[2] + M[11];
    if (Math.abs(y) < r && z < -0.06 && z > -0.102) gap = Math.min(gap, Math.abs(x - axisX));
  }
  if (gap <= r) {
    console.log(`jaw touches the handle at ${a.toFixed(4)} rad (${((a * 180) / Math.PI).toFixed(2)}°)`);
    break;
  }
}
