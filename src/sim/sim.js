import * as M from '../math.js';
import { SdfGrid } from './grid.js';
import { Fluid, CONTACT_NONE, CONTACT_WORLD, CONTACT_SPOON } from './fluid.js';
import { SpoonController } from './spoon.js';
import {
  BOWLS,
  FLAVORS,
  SPOON,
  nearestBowl,
  sdBowl,
  sdSpoonShell,
  sdSpoonSolid,
  sdHullInterior,
  sdHandle,
  sdEllipsoid,
  gradLocal,
  smin,
} from './sdf.js';
import { defaultParams } from './params.js';

const WORLD_CELL = 0.0025;
const WORLD_DIMS = [60, 56, 60];
const CONTENT_ORIGIN = [-0.032, -0.014, -0.022];
const CONTENT_CELL = 0.0016;
const CONTENT_DIMS = [40, 25, 28];
const MAX_SPILLED = 8000;

function srgbToLinear(v) {
  v /= 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** Flavour -> RGBA8 (rgb linear, a = speck amount). */
export function flavorRGBA(f) {
  const hex = parseInt(f.color.slice(1), 16);
  return [
    Math.round(srgbToLinear((hex >> 16) & 255) * 255),
    Math.round(srgbToLinear((hex >> 8) & 255) * 255),
    Math.round(srgbToLinear(hex & 255) * 255),
    Math.round(f.speck * 255),
  ];
}

const _n = [0, 0, 0];
const _v = [0, 0, 0];
const _c = [0, 0, 0, 0];

export class Sim {
  constructor(params = defaultParams()) {
    this.params = params;
    this.flavorRGBA = FLAVORS.map(flavorRGBA);
    this.grids = BOWLS.map(
      (B) => new SdfGrid([B.cx - 0.075, 0.004, B.cz - 0.075], WORLD_CELL, WORLD_DIMS, 4),
    );
    this.content = new SdfGrid(CONTENT_ORIGIN, CONTENT_CELL, CONTENT_DIMS, 4);
    this.spoon = new SpoonController(params.spoon);
    this.fluid = new Fluid(16000);
    this.R = new Float64Array(9);
    this.spilled = new Float32Array(MAX_SPILLED * 3);
    this.spilledRGBA = new Uint8Array(MAX_SPILLED * 4);
    this.reset();
  }

  // ------------------------------------------------------------------ setup
  reset() {
    this.time = 0;
    this.frame = 0;
    this.speedScale = 1;
    this.cutForce = 0;
    this.eatenVolume = 0;
    this.spilledVolume = 0;
    this.nSpilled = 0;
    this.spilledVersion = 0;
    this.releasing = false;
    this.fluid.clear();
    this.content.clear();
    // neutral base colour so linear filtering never bleeds black into the scoop's edges
    this.content.fillColor(this.flavorRGBA[0]);
    this.spoon.reset();
    this.grids.forEach((g, i) => this.initBowl(i, this.params.ice.shape));
    this._lastCarvePos = this.spoon.pos.clone();
    this._lastCarveQuat = this.spoon.quat.clone();
    this.updateStats(true);
  }

  refill() {
    this.grids.forEach((g, i) => this.initBowl(i, this.params.ice.shape));
    this.updateStats(true);
  }

  initBowl(index, shape) {
    const g = this.grids[index];
    const B = BOWLS[index];
    const fill = B.rimY - (shape === 'flat' ? 0.006 : 0.012);
    const inner = B.innerR - 0.0006;
    const balls = [];
    if (shape === 'scoops') {
      for (let k = 0; k < 3; k++) {
        const a = (k / 3) * Math.PI * 2 + index * 0.7;
        balls.push([B.cx + Math.cos(a) * 0.027, fill + 0.012, B.cz + Math.sin(a) * 0.027, 0.026]);
      }
    } else if (shape === 'mound') {
      balls.push([B.cx, fill - 0.035, B.cz, 0.062]);
    }
    const c = g.cell;
    for (let k = 0; k < g.nz; k++)
      for (let j = 0; j < g.ny; j++)
        for (let i = 0; i < g.nx; i++) {
          const x = g.ox + (i + 0.5) * c;
          const y = g.oy + (j + 0.5) * c;
          const z = g.oz + (k + 0.5) * c;
          const dIn = Math.hypot(x - B.cx, y - B.cy, z - B.cz) - inner;
          let d = Math.max(dIn, y - fill);
          for (const b of balls) {
            const db = Math.hypot(x - b[0], y - b[1], z - b[2]) - b[3];
            d = smin(d, db, 0.008);
          }
          // never let the blend bulge into the bowl wall
          if (y < B.rimY) d = Math.max(d, dIn);
          g.data[i + g.nx * (j + g.ny * k)] = Math.max(-g.band, Math.min(g.band, d));
        }
    g.fillColor(this.flavorRGBA[index]);
    g.dirty = true;
  }

  // ------------------------------------------------------------------ helpers
  _setPose(pos, quat) {
    const { x, y, z, w } = quat;
    const R = this.R;
    R[0] = 1 - 2 * (y * y + z * z);
    R[1] = 2 * (x * y - z * w);
    R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w);
    R[4] = 1 - 2 * (x * x + z * z);
    R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);
    R[7] = 2 * (y * z + x * w);
    R[8] = 1 - 2 * (x * x + y * y);
    this.tx = pos.x;
    this.ty = pos.y;
    this.tz = pos.z;
  }

  tiltDeg(quat = this.spoon.quat) {
    const uy = 1 - 2 * (quat.x * quat.x + quat.z * quat.z);
    return M.MathUtils.radToDeg(Math.acos(Math.max(-1, Math.min(1, uy))));
  }

  gridAt(x, y, z) {
    const g = this.grids[nearestBowl(x).index];
    return g.contains(x, y, z) ? g : null;
  }

  // ------------------------------------------------------------------ main step
  step(dt) {
    dt = Math.min(dt, 1 / 30);
    const P = this.params;
    const sp = this.spoon;
    const startPos = sp.pos.clone();
    const startQuat = sp.quat.clone();

    // 1. spoon motion, carving the ice along the swept path
    this._removed = 0;
    sp.p = P.spoon;
    sp.update(dt, this.speedScale, (pp, pq, np, nq, subDt) => this.interact(np, nq, subDt));
    this.interact(sp.pos, sp.quat, dt, true);

    const travel = sp.pos.distanceTo(startPos) + startQuat.angleTo(sp.quat) * 0.03;
    // frontal area swept through the ice; bounded by the cup's projected area so tiny
    // travel (sliding along a wall) can't blow the estimate up
    const area = Math.min(this._removed / Math.max(travel, 0.0005), Math.PI * SPOON.a * SPOON.c);
    const force = P.ice.hardness * 1000 * area;
    this.cutForce += (force - this.cutForce) * 0.3;
    const target = this.cutForce > P.spoon.maxCutForce ? P.spoon.maxCutForce / this.cutForce : 1;
    this.speedScale += (Math.max(0.05, target) - this.speedScale) * 0.3;

    // 2. carry loose ice that sits in / on the spoon along with it
    this._setPose(sp.pos, sp.quat);
    const tilt = this.tiltDeg();
    const tilted = tilt > P.ice.releaseAngle;
    if (!tilted) this.carryParticles(startPos, startQuat, sp.pos, sp.quat);

    // 3. tilted past the release angle: the scoop slides off as particles
    if (tilted && !this.releasing) {
      this.releaseContent();
      this.releasing = true;
    } else if (tilt < P.ice.releaseAngle - 10) {
      this.releasing = false;
    }

    // 4. particles
    const fp = P.fluid;
    const sub = Math.max(1, fp.substeps | 0);
    this.fluid.markFrame();
    for (let s = 0; s < sub; s++) this.fluid.step(dt / sub, fp, this);
    this.freezeResting(tilt < P.ice.releaseAngle - 10, dt);
    const sm = P.ice.smoothing;
    for (const g of this.grids) {
      g.smoothTouched(sm);
      g.settle(sm);
    }
    this.content.smoothTouched(sm);
    this.content.settle(sm);

    this.time += dt;
    this.frame++;
    this.updateStats(this.frame % 15 === 0);
  }

  // ------------------------------------------------------------------ spoon <-> ice
  /**
   * Called along the spoon path. Implements three Claybook-style transfers:
   *  - metal (shell, handle) displaces ice  -> SDF Cut, displaced volume -> particles
   *  - cup interior                         -> ice copied into the spoon's own SDF (scooped)
   *  - ice already on the spoon             -> accretes more ice in front of it (curl)
   */
  interact(pos, quat, subDt, force = false) {
    const lp = this._lastCarvePos;
    const moved = lp.distanceTo(pos) + this._lastCarveQuat.angleTo(quat) * 0.06;
    if (!force && moved < 0.0009) return;
    if (moved < 1e-7) return;
    lp.copy(pos);
    this._lastCarveQuat.copy(quat);

    const P = this.params.ice;
    this._setPose(pos, quat);
    const R = this.R;
    const tx = this.tx;
    const ty = this.ty;
    const tz = this.tz;
    const tilted = this.tiltDeg(quat) > P.releaseAngle;
    const content = this.content;
    const cap = P.capacity * 1e-6;
    const accrete = P.accretion && !tilted && this.contentVolume < cap;

    // world AABB of the spoon head + content box
    const hb = this._localBoxToWorld(CONTENT_ORIGIN, [
      CONTENT_ORIGIN[0] + content.sx,
      CONTENT_ORIGIN[1] + content.sy,
      CONTENT_ORIGIN[2] + content.sz,
    ]);
    const hdl = SPOON.handle;
    const hbHandle = this._localBoxToWorld(
      [hdl[0][0] - 0.004, Math.min(...hdl.map((p) => p[1])) - 0.004, -0.004],
      [hdl[hdl.length - 1][0] + 0.004, Math.max(...hdl.map((p) => p[1])) + 0.004, 0.004],
    );

    const touching = this.grids.filter(
      (g) => this._boxHitsGrid(hb, g) || this._boxHitsGrid(hbHandle, g),
    );
    if (!touching.length) return;

    // A. copy ice into the spoon's SDF (cup interior + accretion shell)
    if (!tilted) {
      const c = content.cell;
      const S = SPOON;
      const δ = P.accretionDist;
      let changed = false;
      let wasEmpty = this.contentVolume < 1e-9;
      for (let k = 0; k < content.nz; k++) {
        const lz = content.oz + (k + 0.5) * c;
        for (let j = 0; j < content.ny; j++) {
          const ly = content.oy + (j + 0.5) * c;
          let idx = content.nx * (j + content.ny * k);
          for (let i = 0; i < content.nx; i++, idx++) {
            const lx = content.ox + (i + 0.5) * c;
            const cv = content.data[idx];
            const inHullBox = lx > -S.a && lx < S.a && ly > -S.b && ly < 0 && lz > -S.c && lz < S.c;
            // accretion grows the scoop, but never on the outside bottom of the cup
            const below = ly < 0 && sdEllipsoid(lx, ly, lz, S.a + S.t, S.b + S.t, S.c + S.t) > 0;
            let dR = accrete && !below ? cv - δ : Infinity;
            if (inHullBox) dR = Math.min(dR, sdHullInterior(lx, ly, lz));
            if (dR >= c) continue;
            const wx = R[0] * lx + R[1] * ly + R[2] * lz + tx;
            const wy = R[3] * lx + R[4] * ly + R[5] * lz + ty;
            const wz = R[6] * lx + R[7] * ly + R[8] * lz + tz;
            const g = this.gridAt(wx, wy, wz);
            if (!g) continue;
            const w = g.sample(wx, wy, wz);
            if (w >= g.band * 0.999) continue;
            // never inside the metal (keeps the rim clean)
            const nv = Math.max(w, dR, -sdSpoonShell(lx, ly, lz));
            if (nv < cv) {
              content.data[idx] = nv;
              content.touch(idx);
              changed = true;
              const vi = g.voxelAt(wx, wy, wz);
              if (vi >= 0 && nv < 2 * c) {
                for (let q = 0; q < 4; q++) _c[q] = g.color[vi * 4 + q];
                // first ice on an empty spoon: its flavour becomes the base colour everywhere,
                // so later smoothing never reveals an uncoloured voxel
                if (wasEmpty) {
                  content.fillColor(_c);
                  wasEmpty = false;
                }
                content.blendColor(idx, _c, cv > 0 ? 1 : 0.5);
              }
            }
          }
        }
      }
      if (changed) {
        content.dirty = true;
        content.colorDirty = true;
      }
    }

    // B. carve the bowls
    let displaced = 0;
    for (const g of touching) {
      displaced += this._carveGrid(g, hb, subDt, tilted, accrete, false);
      displaced += this._carveGrid(g, hbHandle, subDt, tilted, accrete, true);
    }
  }

  _carveGrid(g, box, subDt, tilted, accrete, handleOnly) {
    if (!this._boxHitsGrid(box, g)) return 0;
    const R = this.R;
    const tx = this.tx;
    const ty = this.ty;
    const tz = this.tz;
    const c = g.cell;
    const c3 = c * c * c;
    const pad = 0.4 * c;
    const content = this.content;
    const i0 = Math.max(0, Math.floor((box[0] - g.ox) / c));
    const i1 = Math.min(g.nx - 1, Math.floor((box[3] - g.ox) / c));
    const j0 = Math.max(0, Math.floor((box[1] - g.oy) / c));
    const j1 = Math.min(g.ny - 1, Math.floor((box[4] - g.oy) / c));
    const k0 = Math.max(0, Math.floor((box[2] - g.oz) / c));
    const k1 = Math.min(g.nz - 1, Math.floor((box[5] - g.oz) / c));
    const band = g.band;
    const fluidP = this.params.fluid;
    const toParticles = this.params.ice.displacedToParticles;
    const pr = this.fluid.r > 0 ? this.fluid.r : fluidP.particleRadius;
    const vp = 8 * pr * pr * pr;
    let displacedVol = 0;
    let changed = false;
    for (let k = k0; k <= k1; k++) {
      const z = g.oz + (k + 0.5) * c - tz;
      for (let j = j0; j <= j1; j++) {
        const y = g.oy + (j + 0.5) * c - ty;
        let idx = i0 + g.nx * (j + g.ny * k);
        for (let i = i0; i <= i1; i++, idx++) {
          const w = g.data[idx];
          if (w >= band * 0.999) continue;
          const x = g.ox + (i + 0.5) * c - tx;
          // world -> spoon local (R^T)
          const lx = R[0] * x + R[3] * y + R[6] * z;
          const ly = R[1] * x + R[4] * y + R[7] * z;
          const lz = R[2] * x + R[5] * y + R[8] * z;
          let dS;
          let dI = Infinity;
          let dC = Infinity;
          if (handleOnly) {
            dS = sdHandle(lx, ly, lz) - pad;
          } else {
            dS = sdSpoonShell(lx, ly, lz) - pad;
            if (!tilted) dI = sdHullInterior(lx, ly, lz);
            dC = content.sample(lx, ly, lz);
          }
          const dCarve = Math.min(dS, dI, dC);
          let nw = -dCarve;
          if (nw <= w) continue;
          if (nw > band) nw = band;
          const dv = (g.occ(w) - g.occ(nw)) * c3;
          g.data[idx] = nw;
          g.touch(idx);
          changed = true;
          this._removed += dv;
          const isMetal = dS <= dI && dS <= dC;
          const isContentNoAccrete = !isMetal && dC < dI && !accrete;
          if ((isMetal || isContentNoAccrete) && dv > 0) {
            displacedVol += dv;
            if (toParticles) {
              this._displacedAcc = (this._displacedAcc || 0) + dv;
              while (this._displacedAcc >= vp) {
                this._displacedAcc -= vp;
                this._spawnDisplaced(lx, ly, lz, g, idx, handleOnly, pr);
              }
            }
          }
        }
      }
    }
    if (changed) g.dirty = true;
    return displacedVol;
  }

  _spawnDisplaced(lx, ly, lz, g, idx, handleOnly, r) {
    if (this.fluid.n >= this.params.fluid.maxParticles) return;
    const fn = handleOnly ? sdHandle : sdSpoonSolid;
    const d = fn(lx, ly, lz);
    gradLocal(fn, lx, ly, lz, _n);
    const push = -d + r * 1.3;
    const px = lx + _n[0] * push;
    const py = ly + _n[1] * push;
    const pz = lz + _n[2] * push;
    const R = this.R;
    const wx = R[0] * px + R[1] * py + R[2] * pz + this.tx;
    const wy = R[3] * px + R[4] * py + R[5] * pz + this.ty;
    const wz = R[6] * px + R[7] * py + R[8] * pz + this.tz;
    this.spoon.pointVelocity(wx, wy, wz, _v);
    for (let q = 0; q < 4; q++) _c[q] = g.color[idx * 4 + q];
    this.fluid.spawn(wx, wy, wz, _v[0], _v[1], _v[2], _c);
  }

  _localBoxToWorld(min, max) {
    const R = this.R;
    const out = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let m = 0; m < 8; m++) {
      const x = m & 1 ? max[0] : min[0];
      const y = m & 2 ? max[1] : min[1];
      const z = m & 4 ? max[2] : min[2];
      const wx = R[0] * x + R[1] * y + R[2] * z + this.tx;
      const wy = R[3] * x + R[4] * y + R[5] * z + this.ty;
      const wz = R[6] * x + R[7] * y + R[8] * z + this.tz;
      if (wx < out[0]) out[0] = wx;
      if (wy < out[1]) out[1] = wy;
      if (wz < out[2]) out[2] = wz;
      if (wx > out[3]) out[3] = wx;
      if (wy > out[4]) out[4] = wy;
      if (wz > out[5]) out[5] = wz;
    }
    return out;
  }

  _boxHitsGrid(b, g) {
    return !(
      b[3] < g.ox ||
      b[0] > g.ox + g.sx ||
      b[4] < g.oy ||
      b[1] > g.oy + g.sy ||
      b[5] < g.oz ||
      b[2] > g.oz + g.sz
    );
  }

  // ------------------------------------------------------------------ particles
  /** Rigidly move particles that sit in the cup or touch the spoon with the spoon's motion. */
  carryParticles(p0, q0, p1, q1) {
    if (p0.distanceTo(p1) < 1e-9 && q0.angleTo(q1) < 1e-9) return;
    const f = this.fluid;
    const qi = q0.clone().invert();
    const qd = q1.clone().multiply(qi); // world rotation from the old pose to the new one
    const v = new M.Vector3();
    const l = new M.Vector3();
    for (let i = 0; i < f.n; i++) {
      const o = i * 3;
      let carry = f.contact[i] === CONTACT_SPOON;
      if (!carry) {
        l.set(f.pos[o] - p0.x, f.pos[o + 1] - p0.y, f.pos[o + 2] - p0.z).applyQuaternion(qi);
        carry = sdHullInterior(l.x, l.y, l.z) < 0 || this.content.sample(l.x, l.y, l.z) < 2 * f.r;
      }
      if (!carry) continue;
      for (const A of [f.pos, f.prev]) {
        v.set(A[o] - p0.x, A[o + 1] - p0.y, A[o + 2] - p0.z).applyQuaternion(qd).add(p1);
        A[o] = v.x;
        A[o + 1] = v.y;
        A[o + 2] = v.z;
      }
    }
  }

  /** Project particle i out of the table, bowls, solid ice, spoon metal and spoon contents. */
  collideParticle(i) {
    const f = this.fluid;
    const P = f.pos;
    const o = i * 3;
    const r = f.r;
    let x = P[o];
    let y = P[o + 1];
    let z = P[o + 2];
    let best = 0;
    let type = CONTACT_NONE;
    let nx = 0;
    let ny = 1;
    let nz = 0;

    let d = y - r;
    if (d < 0) {
      y -= d;
      best = -d;
      type = CONTACT_WORLD;
    }
    const B = nearestBowl(x);
    d = sdBowl(B, x, y, z) - r;
    if (d < 0) {
      const e = 1e-5;
      let gx = sdBowl(B, x + e, y, z) - sdBowl(B, x - e, y, z);
      let gy = sdBowl(B, x, y + e, z) - sdBowl(B, x, y - e, z);
      let gz = sdBowl(B, x, y, z + e) - sdBowl(B, x, y, z - e);
      const l = Math.hypot(gx, gy, gz) || 1;
      gx /= l;
      gy /= l;
      gz /= l;
      x -= d * gx;
      y -= d * gy;
      z -= d * gz;
      if (-d >= best) {
        best = -d;
        type = CONTACT_WORLD;
        nx = gx;
        ny = gy;
        nz = gz;
      }
    }
    const g = this.grids[B.index];
    d = g.sample(x, y, z) - r;
    if (d < 0) {
      g.grad(x, y, z, _n);
      x -= d * _n[0];
      y -= d * _n[1];
      z -= d * _n[2];
      if (-d >= best) {
        best = -d;
        type = CONTACT_WORLD;
        nx = _n[0];
        ny = _n[1];
        nz = _n[2];
      }
    }

    const R = this.R;
    let rx = x - this.tx;
    let ry = y - this.ty;
    let rz = z - this.tz;
    if (rx * rx + ry * ry + rz * rz < 0.04) {
      for (let pass = 0; pass < 2; pass++) {
        const lx = R[0] * rx + R[3] * ry + R[6] * rz;
        const ly = R[1] * rx + R[4] * ry + R[7] * rz;
        const lz = R[2] * rx + R[5] * ry + R[8] * rz;
        let dl;
        if (pass === 0) {
          dl = sdSpoonSolid(lx, ly, lz) - r;
          if (dl < 0) gradLocal(sdSpoonSolid, lx, ly, lz, _n);
        } else {
          dl = this.content.sample(lx, ly, lz) - r;
          if (dl < 0) this.content.grad(lx, ly, lz, _n);
        }
        if (dl >= 0) continue;
        const wnx = R[0] * _n[0] + R[1] * _n[1] + R[2] * _n[2];
        const wny = R[3] * _n[0] + R[4] * _n[1] + R[5] * _n[2];
        const wnz = R[6] * _n[0] + R[7] * _n[1] + R[8] * _n[2];
        x -= dl * wnx;
        y -= dl * wny;
        z -= dl * wnz;
        rx = x - this.tx;
        ry = y - this.ty;
        rz = z - this.tz;
        if (-dl >= best) {
          best = -dl;
          type = CONTACT_SPOON;
          nx = wnx;
          ny = wny;
          nz = wnz;
        }
      }
    }

    P[o] = x;
    P[o + 1] = y;
    P[o + 2] = z;
    if (type !== CONTACT_NONE) {
      f.contact[i] = type;
      f.cn[o] = nx;
      f.cn[o + 1] = ny;
      f.cn[o + 2] = nz;
      if (type === CONTACT_SPOON) {
        this.spoon.pointVelocity(x, y, z, _v);
        f.sv[o] = _v[0];
        f.sv[o + 1] = _v[1];
        f.sv[o + 2] = _v[2];
      } else {
        f.sv[o] = f.sv[o + 1] = f.sv[o + 2] = 0;
      }
    }
  }

  /** Claybook's DieWhenStops + StampSDFWhenDies: resting particles freeze into an SDF. */
  freezeResting(allowSpoon, dt) {
    const f = this.fluid;
    const fp = this.params.fluid;
    const rs = fp.stampScale * f.r;
    const R = this.R;
    for (let i = f.n - 1; i >= 0; i--) {
      const o = i * 3;
      const x = f.pos[o];
      const y = f.pos[o + 1];
      const z = f.pos[o + 2];
      if (y < -0.2 || Math.abs(x) > 1 || Math.abs(z) > 1 || !Number.isFinite(x + y + z)) {
        f.removeAt(i);
        continue;
      }
      const type = f.contact[i];
      if (type === CONTACT_NONE) {
        f.rest[i] = 0;
        continue;
      }
      // net displacement over the whole frame, relative to the surface it rests on,
      // smoothed: solver jitter inside a pile averages out, real flow does not
      const vx = (x - f.framePos[o]) / dt - f.sv[o];
      const vy = (y - f.framePos[o + 1]) / dt - f.sv[o + 1];
      const vz = (z - f.framePos[o + 2]) / dt - f.sv[o + 2];
      const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      f.slow[i] = f.slow[i] * 0.6 + speed * 0.4;
      f.rest[i] = f.slow[i] < fp.stopSpeed ? f.rest[i] + 1 : 0;
      const instant = fp.stampOnContact && type === CONTACT_WORLD && f.age[i] > 3;
      // a slow particle already half-sunk into ice cream simply merges with it
      // (the edited SDF is not an exact distance, so the solver can't fully push it out)
      const embedded = f.slow[i] < 3 * fp.stopSpeed && this._embedded(i, type, x, y, z);
      if (f.rest[i] < fp.restFrames && !instant && !embedded) continue;
      for (let q = 0; q < 4; q++) _c[q] = f.rgba[i * 4 + q];

      if (type === CONTACT_SPOON) {
        if (!allowSpoon) continue;
        const rx = x - this.tx;
        const ry = y - this.ty;
        const rz = z - this.tz;
        const lx = R[0] * rx + R[3] * ry + R[6] * rz;
        const ly = R[1] * rx + R[4] * ry + R[7] * rz;
        const lz = R[2] * rx + R[5] * ry + R[8] * rz;
        if (this.content.contains(lx, ly, lz, rs)) {
          this.content.stampSphere(lx, ly, lz, rs, fp.stampSmooth, _c);
          f.removeAt(i);
        }
        continue;
      }
      const g = this.gridAt(x, y, z);
      if (g && g.contains(x, y, z, rs + g.cell)) {
        g.stampSphere(x, y, z, rs, fp.stampSmooth, _c);
        f.removeAt(i);
      } else {
        // on the table, outside every bowl volume: keep it as a static crumb
        if (this.nSpilled < MAX_SPILLED) {
          const s = this.nSpilled++;
          this.spilled[s * 3] = x;
          this.spilled[s * 3 + 1] = y;
          this.spilled[s * 3 + 2] = z;
          for (let q = 0; q < 4; q++) this.spilledRGBA[s * 4 + q] = _c[q];
          this.spilledVersion++;
        }
        this.spilledVolume += this.particleVolume();
        f.removeAt(i);
      }
    }
  }

  _embedded(i, type, x, y, z) {
    const r = this.fluid.r;
    if (type === CONTACT_WORLD) {
      const g = this.gridAt(x, y, z);
      return !!g && g.sample(x, y, z) < 0.5 * r;
    }
    const R = this.R;
    const rx = x - this.tx;
    const ry = y - this.ty;
    const rz = z - this.tz;
    return (
      this.content.sample(
        R[0] * rx + R[3] * ry + R[6] * rz,
        R[1] * rx + R[4] * ry + R[7] * rz,
        R[2] * rx + R[5] * ry + R[8] * rz,
      ) <
      0.5 * r
    );
  }

  particleVolume() {
    const r = this.fluid.r > 0 ? this.fluid.r : this.params.fluid.particleRadius;
    return 8 * r * r * r;
  }

  /** Turn everything on the spoon into particles (tilt release). */
  releaseContent() {
    const content = this.content;
    if (this.contentVolume <= 0) return;
    const f = this.fluid;
    f.configure(this.params.fluid.particleRadius, this.params.fluid.kernelScale);
    const step = f.d;
    const R = this.R;
    for (let lz = content.oz + step / 2; lz < content.oz + content.sz; lz += step)
      for (let ly = content.oy + step / 2; ly < content.oy + content.sy; ly += step)
        for (let lx = content.ox + step / 2; lx < content.ox + content.sx; lx += step) {
          if (content.sample(lx, ly, lz) >= 0) continue;
          if (f.n >= this.params.fluid.maxParticles) break;
          const wx = R[0] * lx + R[1] * ly + R[2] * lz + this.tx;
          const wy = R[3] * lx + R[4] * ly + R[5] * lz + this.ty;
          const wz = R[6] * lx + R[7] * ly + R[8] * lz + this.tz;
          this.spoon.pointVelocity(wx, wy, wz, _v);
          const vi = content.voxelAt(lx, ly, lz);
          for (let q = 0; q < 4; q++) _c[q] = content.color[vi * 4 + q];
          f.spawn(wx, wy, wz, _v[0], _v[1], _v[2], _c);
        }
    content.clear();
  }

  /** "Eat": empty the spoon, including loose bits riding on it. Returns grams eaten. */
  eat() {
    this._setPose(this.spoon.pos, this.spoon.quat);
    let vol = this.content.volume();
    this.content.clear();
    const f = this.fluid;
    const R = this.R;
    for (let i = f.n - 1; i >= 0; i--) {
      const o = i * 3;
      const rx = f.pos[o] - this.tx;
      const ry = f.pos[o + 1] - this.ty;
      const rz = f.pos[o + 2] - this.tz;
      const lx = R[0] * rx + R[3] * ry + R[6] * rz;
      const ly = R[1] * rx + R[4] * ry + R[7] * rz;
      const lz = R[2] * rx + R[5] * ry + R[8] * rz;
      const onSpoon =
        f.contact[i] === CONTACT_SPOON ||
        sdHullInterior(lx, ly, lz) < f.r ||
        (this.content.contains(lx, ly, lz) && sdSpoonSolid(lx, ly, lz) < 0.02);
      if (onSpoon) {
        vol += this.particleVolume();
        f.removeAt(i);
      }
    }
    this.eatenVolume += vol;
    this.updateStats(false);
    return vol * 1e6 * this.params.ice.density;
  }

  // ------------------------------------------------------------------ stats
  updateStats(full) {
    this.contentVolume = this.content.volume();
    if (full || !this.bowlVolumes) this.bowlVolumes = this.grids.map((g) => g.volume());
  }

  grams(vol) {
    return vol * 1e6 * this.params.ice.density;
  }

  getState() {
    const sp = this.spoon;
    const e = new M.Euler().setFromQuaternion(sp.quat, 'XYZ');
    const te = new M.Euler().setFromQuaternion(sp.targetQuat, 'XYZ');
    return {
      time: this.time,
      spoon: {
        position: sp.pos.toArray(),
        quaternion: sp.quat.toArray(),
        eulerXYZdeg: [e.x, e.y, e.z].map(M.MathUtils.radToDeg),
        linearVelocity: sp.linVel.toArray(),
        angularVelocity: sp.angVel.toArray(),
        tiltDeg: this.tiltDeg(),
      },
      target: {
        position: sp.targetPos.toArray(),
        quaternion: sp.targetQuat.toArray(),
        eulerXYZdeg: [te.x, te.y, te.z].map(M.MathUtils.radToDeg),
      },
      contact: sp.contacts > 0,
      contactSamples: sp.contacts,
      blocked: sp.blocked,
      speedScale: this.speedScale,
      cutForceN: this.cutForce,
      scoopedGrams: this.grams(this.contentVolume),
      eatenGrams: this.grams(this.eatenVolume),
      spilledGrams: this.grams(this.spilledVolume),
      bowlGrams: Object.fromEntries(BOWLS.map((b, i) => [b.flavor.id, this.grams(this.bowlVolumes[i])])),
      particles: this.fluid.n,
    };
  }
}
