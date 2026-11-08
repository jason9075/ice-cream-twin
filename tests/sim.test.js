import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/math.js';
import { Sim } from '../src/sim/sim.js';
import { BOWLS } from '../src/sim/sdf.js';
import { PEN_TOL } from '../src/sim/spoon.js';

function totalGrams(sim) {
  const s = sim.getState();
  const fluid = sim.grams(sim.fluid.n * sim.particleVolume());
  const bowls = Object.values(s.bowlGrams).reduce((a, b) => a + b, 0);
  return bowls + s.scoopedGrams + s.eatenGrams + s.spilledGrams + fluid + sim.grams(sim.debtVolume());
}

function runTo(sim, pos, quat, frames) {
  sim.spoon.setTarget(pos, quat);
  for (let f = 0; f < frames; f++) {
    sim.step(1 / 60);
    assert.ok(sim.spoon.penetration(sim.spoon.pos, sim.spoon.quat) <= PEN_TOL + 1e-6);
  }
}

test('scoop vanilla, carry it, eat it', () => {
  const sim = new Sim();
  sim.updateStats(true);
  const before = totalGrams(sim);
  const B = BOWLS[0];
  const level = new M.Quaternion();
  const dig = new M.Quaternion().setFromEuler(new M.Euler(0, 0, 0.5)); // tip the cup forward
  const t0 = performance.now();
  runTo(sim, new M.Vector3(B.cx - 0.03, 0.14, 0), dig, 90);
  runTo(sim, new M.Vector3(B.cx - 0.02, 0.085, 0), dig, 150); // push in
  runTo(sim, new M.Vector3(B.cx + 0.025, 0.088, 0), level, 150); // drag + level
  runTo(sim, new M.Vector3(B.cx + 0.025, 0.16, 0), level, 120); // lift
  const ms = (performance.now() - t0) / 510;
  sim.updateStats(true);
  const s = sim.getState();
  console.log(
    `scooped ${s.scoopedGrams.toFixed(1)} g, particles ${s.particles}, vanilla left ${s.bowlGrams.vanilla.toFixed(1)} g, ` +
      `cut force peak-ish ${s.cutForceN.toFixed(1)} N, ${ms.toFixed(2)} ms/frame`,
  );
  assert.ok(s.scoopedGrams > 3, `expected a real scoop, got ${s.scoopedGrams} g`);
  assert.ok(s.spoon.position[1] > 0.15, 'spoon should have lifted out');

  const after = totalGrams(sim);
  console.log(`mass before ${before.toFixed(1)} g, after ${after.toFixed(1)} g`);
  assert.ok(Math.abs(after - before) / before < 0.15, 'mass should be roughly conserved');

  const eaten = sim.eat();
  assert.ok(eaten > 3);
  assert.equal(sim.getState().scoopedGrams, 0);
});

test('tilting past the release angle drops the scoop back as particles', () => {
  const sim = new Sim();
  const B = BOWLS[2];
  runTo(sim, new M.Vector3(B.cx, 0.14, 0), new M.Quaternion(), 60);
  runTo(sim, new M.Vector3(B.cx, 0.085, 0), new M.Quaternion(), 150);
  runTo(sim, new M.Vector3(B.cx, 0.15, 0), new M.Quaternion(), 120);
  const scooped = sim.getState().scoopedGrams;
  assert.ok(scooped > 2, `scooped ${scooped}`);
  const flip = new M.Quaternion().setFromEuler(new M.Euler(Math.PI * 0.9, 0, 0));
  runTo(sim, new M.Vector3(B.cx, 0.15, 0), flip, 120);
  assert.ok(sim.getState().scoopedGrams < 0.5, 'scoop should have fallen off');
  for (let f = 0; f < 240; f++) sim.step(1 / 60);
  console.log(`after release: particles left ${sim.fluid.n}`);
});

test('plunging straight down to the bottom, again and again, keeps the mass', () => {
  const sim = new Sim();
  sim.updateStats(true);
  const before = totalGrams(sim);
  const B = BOWLS[0];
  const level = new M.Quaternion();
  const t0 = performance.now();
  let frames = 0;
  for (let k = 0; k < 3; k++) {
    runTo(sim, new M.Vector3(B.cx, 0.14, 0), level, 60);
    runTo(sim, new M.Vector3(B.cx, 0.0, 0), level, 240); // as deep as the bowl lets it
    runTo(sim, new M.Vector3(B.cx, 0.15, 0), level, 150);
    frames += 450;
    assert.ok(sim.getState().scoopedGrams > 5, 'each plunge should bring up a scoop');
    sim.eat();
  }
  const after = totalGrams(sim);
  const ms = (performance.now() - t0) / frames;
  console.log(`plunge x3: mass ${before.toFixed(1)} -> ${after.toFixed(1)} g, eaten ${sim.getState().eatenGrams.toFixed(1)} g, ${ms.toFixed(2)} ms/frame`);
  assert.ok(Math.abs(after - before) / before < 0.01, 'mass must be conserved within 1%');
  assert.ok(sim.spilledVolume <= sim.nSpilled * sim.particleVolume() * 1.01 + 1e-12, 'everything counted as spilled is a visible crumb');
});
