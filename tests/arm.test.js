import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/math.js';
import { Sim } from '../src/sim/sim.js';
import { PEN_TOL } from '../src/sim/spoon.js';
import { Arm } from '../src/robot/arm.js';
import { LIMITS, forward } from '../src/robot/so101.js';
import { TOOLS } from '../src/robot/tool.js';

// spoon handle toward the robot (it stands at -Z), cup pitched nose-down by `pitch`
const facing = (pitch, yaw = 0) =>
  new M.Quaternion()
    .setFromAxisAngle(new M.Vector3(0, 1, 0), Math.PI / 2 + yaw)
    .multiply(new M.Quaternion().setFromAxisAngle(new M.Vector3(0, 0, 1), pitch));

function drive(sim, arm, pos, quat, frames, check) {
  sim.spoon.setTarget(pos, quat);
  for (let f = 0; f < frames; f++) {
    arm.constrain(sim);
    sim.step(1 / 60);
    arm.follow(sim);
    check?.();
  }
}

test('FK -> IK round trip over the joint range', () => {
  const arm = new Arm();
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  let fails = 0;
  for (let i = 0; i < 200; i++) {
    const q = LIMITS.map(([a, b]) => a + (b - a) * rnd());
    const pose = arm.spoonPose(q);
    const start = q.map((v, k) => Math.min(LIMITS[k][1], Math.max(LIMITS[k][0], v + (rnd() - 0.5) * 0.3)));
    const e = arm.solve(start, pose.pos, pose.quat, 100);
    if (e.mm > 0.05 || e.deg > 0.05) fails++;
  }
  assert.ok(fails <= 2, `${fails} / 200 poses not recovered`);
});

test('the tool mount is a rigid transform and the jaw is clamped inside its range', () => {
  for (const [name, t] of Object.entries(TOOLS)) {
    const m = t.spoonInGripper;
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) {
        const dot = m[a] * m[b] + m[4 + a] * m[4 + b] + m[8 + a] * m[8 + b];
        assert.ok(Math.abs(dot - (a === b ? 1 : 0)) < 1e-12, `${name} rotation not orthonormal`);
      }
    if (t.jaw != null) assert.ok(t.jaw >= -0.1746 && t.jaw <= 1.7454);
  }
  // the handle point the gripper holds sits between the fingertips (gripper_link -Z, ~9 cm out)
  const arm = new Arm();
  const { T } = forward(arm.q, arm.jaw);
  const g = T.gripper_link;
  const s = arm.spoonPose(arm.q);
  assert.ok(s.pos.distanceTo(new M.Vector3(g[3], g[7], g[11])) < 0.3);
});

test('the arm carries a scoop out of the middle bowl and stays on the spoon', () => {
  const sim = new Sim();
  const arm = new Arm();
  arm.reset(sim);
  let worst = 0;
  const check = () => {
    worst = Math.max(worst, arm.err.mm);
    assert.ok(sim.spoon.penetration(sim.spoon.pos, sim.spoon.quat) <= PEN_TOL + 1e-6);
    arm.q.forEach((v, k) => assert.ok(v >= LIMITS[k][0] - 1e-9 && v <= LIMITS[k][1] + 1e-9));
  };
  drive(sim, arm, new M.Vector3(0, 0.14, 0.02), facing(0.5), 80, check);
  drive(sim, arm, new M.Vector3(0, 0.086, 0.015), facing(0.5), 150, check);
  drive(sim, arm, new M.Vector3(0, 0.088, -0.025), facing(0), 150, check);
  drive(sim, arm, new M.Vector3(0, 0.15, -0.03), facing(0), 100, check);
  const st = sim.getState();
  assert.ok(st.scoopedGrams > 10, `scooped only ${st.scoopedGrams.toFixed(1)} g`);
  assert.ok(worst < 0.5, `arm drifted ${worst.toFixed(3)} mm from the spoon`);
});

test('moving between bowls goes through poses the arm can hold', () => {
  const sim = new Sim();
  const arm = new Arm();
  arm.reset(sim);
  let worst = 0;
  const check = () => (worst = Math.max(worst, arm.err.mm));
  for (const x of [-0.18, 0.18, 0]) drive(sim, arm, new M.Vector3(x, 0.15, 0), facing(0.3), 240, check);
  assert.ok(worst < 1, `arm drifted ${worst.toFixed(3)} mm from the spoon`);
  assert.ok(sim.spoon.pos.distanceTo(arm.goal.pos) < 0.002);
});

test('joint-space command: the arm settles on the commanded angles', () => {
  const sim = new Sim();
  const arm = new Arm();
  arm.reset(sim);
  const cmd = [0.3, -1.5, 0.6, 1.0, 0.4]; // spoon ~23 cm up, clear of the bowls
  arm.setJoints(sim, cmd);
  for (let f = 0; f < 300; f++) {
    arm.constrain(sim);
    sim.step(1 / 60);
    arm.follow(sim);
  }
  const p = arm.spoonPose(cmd);
  assert.ok(sim.spoon.pos.distanceTo(p.pos) < 0.001);
  assert.ok(arm.q.every((v, k) => Math.abs(v - cmd[k]) < 0.01), `q = ${[...arm.q].map((v) => v.toFixed(3))}`);
});
