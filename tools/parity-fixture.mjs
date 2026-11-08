// Writes the fixtures for rust/core/tests/parity.rs from the JS simulation:
//   node tools/parity-fixture.mjs
// - scoop.csv / plunge.csv: per-frame spoon pose, ice volumes, particle count, ledger, cut force
// - arm.csv: the SO-101 holding the spoon (src/robot/arm.js): spoon pose, joint angles, command, error
// - math.csv: Math.sin/cos/cbrt/acos/atan2/asin samples (the Rust port must match V8 bit for bit)
import fs from 'node:fs';
import * as M from '../src/math.js';
import { Sim, flavorRGBA } from '../src/sim/sim.js';
import { BOWLS, FLAVORS } from '../src/sim/sdf.js';
import { Arm } from '../src/robot/arm.js';

const OUT = new URL('../rust/core/tests/fixtures/', import.meta.url);
fs.mkdirSync(OUT, { recursive: true });
const num = (v) => (Number.isInteger(v) ? String(v) : v.toPrecision(17));

// Scenario steps: [x, y, z, ex, ey, ez, frames] (target position + XYZ euler) or ['eat'].
// The same lists are hard-coded in parity.rs.
const B0 = BOWLS[0];
const SCENARIOS = {
  scoop: [
    [B0.cx - 0.03, 0.14, 0, 0, 0, 0.5, 90],
    [B0.cx - 0.02, 0.085, 0, 0, 0, 0.5, 150],
    [B0.cx + 0.025, 0.088, 0, 0, 0, 0, 150],
    [B0.cx + 0.025, 0.16, 0, 0, 0, 0, 120],
  ],
  plunge: [0, 1].flatMap(() => [
    [B0.cx, 0.14, 0, 0, 0, 0, 60],
    [B0.cx, 0.0, 0, 0, 0, 0, 240],
    [B0.cx, 0.15, 0, 0, 0, 0, 150],
    ['eat'],
  ]),
};

for (const [name, script] of Object.entries(SCENARIOS)) {
  const sim = new Sim();
  const rows = ['frame,px,py,pz,qx,qy,qz,qw,content,bowl0,particles,debt,cut,eaten'];
  let frame = 0;
  for (const s of script) {
    if (s[0] === 'eat') {
      sim.eat();
      continue;
    }
    sim.spoon.setTarget(new M.Vector3(s[0], s[1], s[2]), new M.Quaternion().setFromEuler(new M.Euler(s[3], s[4], s[5])));
    for (let f = 0; f < s[6]; f++) {
      sim.step(1 / 60);
      frame++;
      const sp = sim.spoon;
      rows.push([frame, sp.pos.x, sp.pos.y, sp.pos.z, sp.quat.x, sp.quat.y, sp.quat.z, sp.quat.w,
        sim.content.volume(), sim.grids[0].volume(), sim.fluid.n, sim.debtVolume(), sim.cutForce, sim.eatenVolume].map(num).join(','));
    }
  }
  fs.writeFileSync(new URL(`${name}.csv`, OUT), rows.join('\n') + '\n');
  console.log(`${name}: ${frame} frames`);
}

// The arm drives the spoon the way the page does: arm.constrain -> sim.step -> arm.follow.
// Steps: ['go', x, y, z, pitch, frames] (handle toward the robot, cup pitched nose-down) or
// ['joints', [5 angles], frames]. The same list is hard-coded in parity.rs.
{
  const ARM_SCRIPT = [
    ['go', 0, 0.14, 0.02, 0.5, 80],
    ['go', 0, 0.086, 0.015, 0.5, 150],
    ['go', 0, 0.088, -0.025, 0, 150],
    ['go', 0, 0.15, -0.03, 0, 100],
    ['joints', [0.3, -1.5, 0.6, 1.0, 0.4], 150],
    ['go', 0.18, 0.15, 0, 0.3, 200],
  ];
  const sim = new Sim();
  const arm = new Arm();
  arm.reset(sim);
  const rows = ['frame,px,py,pz,qx,qy,qz,qw,q0,q1,q2,q3,q4,c0,c1,c2,c3,c4,errMm,errDeg,content'];
  let frame = 0;
  for (const s of ARM_SCRIPT) {
    if (s[0] === 'go') {
      const q = new M.Quaternion()
        .setFromAxisAngle(new M.Vector3(0, 1, 0), Math.PI / 2)
        .multiply(new M.Quaternion().setFromAxisAngle(new M.Vector3(0, 0, 1), s[4]));
      sim.spoon.setTarget(new M.Vector3(s[1], s[2], s[3]), q);
    } else arm.setJoints(sim, s[1]);
    for (let f = 0; f < s[s.length - 1]; f++) {
      arm.constrain(sim);
      sim.step(1 / 60);
      arm.follow(sim);
      frame++;
      const sp = sim.spoon;
      rows.push([frame, sp.pos.x, sp.pos.y, sp.pos.z, sp.quat.x, sp.quat.y, sp.quat.z, sp.quat.w,
        ...arm.q, ...arm.qCmd, arm.err.mm, arm.err.deg, sim.content.volume()].map(num).join(','));
    }
  }
  fs.writeFileSync(new URL('arm.csv', OUT), rows.join('\n') + '\n');
  console.log(`arm: ${frame} frames`);
}

let s = 12345 >>> 0;
const rows = ['x,cos,sin,cbrt,acos,atan2,asin'];
for (let i = 0; i < 4000; i++) {
  s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
  const x = (s / 4294967296) * Math.PI * 8 - Math.PI * 4;
  const u = x / (Math.PI * 4);
  rows.push([x, Math.cos(x), Math.sin(x), Math.cbrt(x * 1e-7), Math.acos(u), Math.atan2(x - 3, 1.3), Math.asin(u)].map(num).join(','));
}
fs.writeFileSync(new URL('math.csv', OUT), rows.join('\n') + '\n');
fs.writeFileSync(new URL('flavors.csv', OUT), FLAVORS.map((f) => flavorRGBA(f).join(',')).join('\n') + '\n');
console.log('math: 4000 samples, flavors: 3');
