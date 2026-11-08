import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/math.js';
import { SpoonController, PEN_TOL } from '../src/sim/spoon.js';
import { BOWLS } from '../src/sim/sdf.js';
import { defaultParams } from '../src/sim/params.js';

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function randomQuat(r) {
  const u1 = r();
  const u2 = r() * Math.PI * 2;
  const u3 = r() * Math.PI * 2;
  const a = Math.sqrt(1 - u1);
  const b = Math.sqrt(u1);
  return new M.Quaternion(a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3));
}

/** Signed side of the bowl wall for every sample that is below the rim: -1 inside, +1 outside. */
function wallSides(sp, B) {
  const s = sp.samples;
  const out = new Int8Array(sp.nSamples);
  const w = new M.Vector3();
  for (let i = 0; i < sp.nSamples; i++) {
    w.set(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]).applyQuaternion(sp.quat).add(sp.pos);
    if (w.y > B.rimY - 0.004) continue;
    const d = Math.hypot(w.x - B.cx, w.y - B.cy, w.z - B.cz) - B.r;
    out[i] = d < 0 ? -1 : 1;
  }
  return out;
}

test('spoon never penetrates bowls or table for random targets', () => {
  const p = defaultParams().spoon;
  p.maxLinSpeed = 0.6;
  p.maxAngSpeed = 400;
  const sp = new SpoonController(p);
  const r = rng(7);
  let worst = -Infinity;
  for (let trial = 0; trial < 120; trial++) {
    const B = BOWLS[Math.floor(r() * 3)];
    const pos = new M.Vector3(B.cx + (r() - 0.5) * 0.22, r() * 0.16, (r() - 0.5) * 0.22);
    sp.setTarget(pos, randomQuat(r));
    for (let f = 0; f < 40; f++) {
      sp.update(1 / 60, 1);
      const pen = sp.penetration(sp.pos, sp.quat);
      worst = Math.max(worst, pen);
      assert.ok(pen <= PEN_TOL + 1e-6, `trial ${trial} frame ${f}: penetration ${pen}`);
      assert.ok(Number.isFinite(sp.pos.x + sp.quat.w));
    }
  }
  console.log(`worst penetration ${(worst * 1000).toFixed(3)} mm (tolerance ${PEN_TOL * 1000} mm)`);
});

test('fast push through the bowl wall cannot tunnel', () => {
  const p = defaultParams().spoon;
  p.maxLinSpeed = 3; // 5 cm per frame, much more than the 6 mm wall
  p.maxAngSpeed = 720;
  const sp = new SpoonController(p);
  const B = BOWLS[1];
  // start inside the middle bowl, cup level, low down
  sp.pos.set(B.cx, B.cy - 0.03, 0);
  sp.quat.identity();
  sp.setTarget(sp.pos, sp.quat);
  for (let f = 0; f < 10; f++) sp.update(1 / 60, 1);
  for (const dir of [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
    [0.7, -0.7, 0],
  ]) {
    sp.pos.set(B.cx, B.cy - 0.03, 0);
    sp.quat.identity();
    sp.update(1 / 60, 1);
    let sides = wallSides(sp, B);
    const target = new M.Vector3(B.cx + dir[0] * 0.3, B.cy - 0.03 + dir[1] * 0.3, dir[2] * 0.3);
    sp.setTarget(target, new M.Quaternion().setFromEuler(new M.Euler(0.3, 1.0, -0.4)));
    for (let f = 0; f < 90; f++) {
      sp.update(1 / 60, 1);
      const now = wallSides(sp, B);
      for (let i = 0; i < now.length; i++) {
        if (sides[i] && now[i]) assert.equal(now[i], sides[i], `sample ${i} crossed the wall (dir ${dir})`);
      }
      sides = now;
      assert.ok(sp.penetration(sp.pos, sp.quat) <= PEN_TOL + 1e-6);
    }
  }
});

test('spoon slides along the wall instead of stopping dead', () => {
  const sp = new SpoonController(defaultParams().spoon);
  const B = BOWLS[0];
  sp.pos.set(B.cx, B.cy - 0.03, 0);
  sp.quat.identity();
  sp.setTarget(new M.Vector3(B.cx + 0.2, B.cy - 0.03, 0.05), sp.quat);
  for (let f = 0; f < 120; f++) sp.update(1 / 60, 1);
  assert.ok(sp.contacts > 0, 'should end in contact with the wall');
  assert.ok(sp.pos.z > 0.01, `should slide sideways toward z (got z=${sp.pos.z})`);
});
