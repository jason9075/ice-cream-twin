import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/math.js';
import { frames } from '../src/bridge.js';

const close = (a, b, eps = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) < eps);

test('ROS <-> Y-up position conversion', () => {
  // three.js up (+Y) is ROS up (+Z); three.js +Z (toward viewer) is ROS -Y (right)
  assert.ok(close(frames.posToRos([0, 1, 0]), [0, 0, 1]));
  assert.ok(close(frames.posToRos([0, 0, 1]), [0, -1, 0]));
  assert.ok(close(frames.posFromRos(frames.posToRos([0.1, 0.2, 0.3])), [0.1, 0.2, 0.3]));
});

test('ROS <-> Y-up quaternion conversion keeps the physical rotation', () => {
  const q = new M.Quaternion().setFromEuler(new M.Euler(0.3, -0.7, 1.1));
  const qr = frames.quatToRos(q.toArray());
  assert.ok(close(frames.quatFromRos(qr), q.toArray()) || close(frames.quatFromRos(qr).map((v) => -v), q.toArray()));
  // rotating a vector in either frame must agree after converting the vector
  const v = new M.Vector3(0.2, -0.1, 0.4);
  const a = frames.posToRos(v.clone().applyQuaternion(q).toArray());
  const b = new M.Vector3(...frames.posToRos(v.toArray())).applyQuaternion(new M.Quaternion(...qr)).toArray();
  assert.ok(close(a, b, 1e-9), `${a} vs ${b}`);
});
