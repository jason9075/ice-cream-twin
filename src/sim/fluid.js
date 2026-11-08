/**
 * Position Based Fluids (Macklin & Müller 2013), the same family of solver Claybook's
 * FSimulateFluidParticlesPass1/Pass3 shaders point to, on the CPU:
 *
 *   predict -> hash-grid neighbours -> [density constraint (repulsive only) + cohesion
 *   -> collide with SDFs] x iterations -> velocity from positions -> friction -> XSPH viscosity
 *
 * Only the "loose" ice cream lives here; once a particle comes to rest the Sim stamps it back
 * into an SDF (Claybook's StampSDFWhenDies), so the particle count stays small.
 */

const MAXN = 48;
const MAX_SEPARATION = 0.05; // m/s of outward velocity a contact may produce
const TABLE_BITS = 16;
const TABLE = 1 << TABLE_BITS;
const MASK = TABLE - 1;

export const CONTACT_NONE = 0;
export const CONTACT_WORLD = 1; // table, bowl, solid ice cream
export const CONTACT_SPOON = 2; // spoon metal or the ice cream already on the spoon

export class Fluid {
  constructor(cap = 16000) {
    this.cap = cap;
    this.n = 0;
    this.pos = new Float32Array(cap * 3);
    this.prev = new Float32Array(cap * 3);
    this.vel = new Float32Array(cap * 3);
    this.dp = new Float32Array(cap * 3);
    this.lambda = new Float32Array(cap);
    this.rest = new Uint16Array(cap);
    this.age = new Uint16Array(cap);
    this.contact = new Uint8Array(cap);
    this.cn = new Float32Array(cap * 3); // contact normal
    this.sv = new Float32Array(cap * 3); // velocity of the surface in contact
    this.rgba = new Uint8Array(cap * 4);
    this.framePos = new Float32Array(cap * 3); // position at the start of the frame
    this.slow = new Float32Array(cap); // smoothed net speed over frames (rest detection)
    this.nbr = new Int32Array(cap * MAXN);
    this.nc = new Uint8Array(cap);
    this.cellCount = new Int32Array(TABLE + 1);
    this.cellStart = new Int32Array(TABLE + 1);
    this.sorted = new Int32Array(cap);
    this.cellOf = new Int32Array(cap);
    this.r = -1;
  }

  configure(radius, kernelScale) {
    if (radius === this.r && kernelScale === this.ks) return;
    this.r = radius;
    this.ks = kernelScale;
    this.d = 2 * radius;
    this.h = kernelScale * this.d;
    this.h2 = this.h * this.h;
    // rest density = kernel sum over a simple cubic lattice at spacing d
    let rho = 0;
    for (let i = -4; i <= 4; i++)
      for (let j = -4; j <= 4; j++)
        for (let k = -4; k <= 4; k++) {
          const r2 = (i * i + j * j + k * k) * this.d * this.d;
          if (r2 < this.h2) rho += this.W(r2);
        }
    this.rho0 = rho;
    const g = 3 / this.h / this.rho0;
    this.eps = 2 * g * g;
  }

  W(r2) {
    const x = 1 - r2 / this.h2;
    return x * x * x;
  }

  clear() {
    this.n = 0;
  }

  spawn(x, y, z, vx, vy, vz, rgba) {
    if (this.n >= this.cap) return -1;
    const i = this.n++;
    const o = i * 3;
    this.pos[o] = this.prev[o] = x;
    this.pos[o + 1] = this.prev[o + 1] = y;
    this.pos[o + 2] = this.prev[o + 2] = z;
    this.vel[o] = vx;
    this.vel[o + 1] = vy;
    this.vel[o + 2] = vz;
    this.rest[i] = 0;
    this.age[i] = 0;
    this.slow[i] = 1;
    this.framePos[o] = x;
    this.framePos[o + 1] = y;
    this.framePos[o + 2] = z;
    this.contact[i] = CONTACT_NONE;
    const c = i * 4;
    this.rgba[c] = rgba[0];
    this.rgba[c + 1] = rgba[1];
    this.rgba[c + 2] = rgba[2];
    this.rgba[c + 3] = rgba[3];
    return i;
  }

  removeAt(i) {
    const last = --this.n;
    if (i === last) return;
    const a = i * 3;
    const b = last * 3;
    for (let k = 0; k < 3; k++) {
      this.pos[a + k] = this.pos[b + k];
      this.prev[a + k] = this.prev[b + k];
      this.vel[a + k] = this.vel[b + k];
      this.cn[a + k] = this.cn[b + k];
      this.sv[a + k] = this.sv[b + k];
      this.framePos[a + k] = this.framePos[b + k];
    }
    this.slow[i] = this.slow[last];
    for (let k = 0; k < 4; k++) this.rgba[i * 4 + k] = this.rgba[last * 4 + k];
    this.rest[i] = this.rest[last];
    this.age[i] = this.age[last];
    this.contact[i] = this.contact[last];
  }

  markFrame() {
    this.framePos.set(this.pos.subarray(0, this.n * 3));
  }

  _hash(ix, iy, iz) {
    return (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) & MASK;
  }

  buildNeighbors() {
    const n = this.n;
    const P = this.pos;
    const inv = 1 / this.h;
    const cc = this.cellCount;
    cc.fill(0);
    for (let i = 0; i < n; i++) {
      const c = this._hash(
        Math.floor(P[i * 3] * inv),
        Math.floor(P[i * 3 + 1] * inv),
        Math.floor(P[i * 3 + 2] * inv),
      );
      this.cellOf[i] = c;
      cc[c]++;
    }
    const cs = this.cellStart;
    let acc = 0;
    for (let c = 0; c < TABLE; c++) {
      cs[c] = acc;
      acc += cc[c];
    }
    cs[TABLE] = acc;
    cc.fill(0);
    for (let i = 0; i < n; i++) {
      const c = this.cellOf[i];
      this.sorted[cs[c] + cc[c]++] = i;
    }
    const h2 = this.h2;
    for (let i = 0; i < n; i++) {
      const x = P[i * 3];
      const y = P[i * 3 + 1];
      const z = P[i * 3 + 2];
      const ix = Math.floor(x * inv);
      const iy = Math.floor(y * inv);
      const iz = Math.floor(z * inv);
      let cnt = 0;
      const base = i * MAXN;
      for (let dz = -1; dz <= 1 && cnt < MAXN; dz++)
        for (let dy = -1; dy <= 1 && cnt < MAXN; dy++)
          for (let dx = -1; dx <= 1 && cnt < MAXN; dx++) {
            const c = this._hash(ix + dx, iy + dy, iz + dz);
            const e = cs[c] + cc[c];
            for (let s = cs[c]; s < e; s++) {
              const j = this.sorted[s];
              if (j === i) continue;
              const ex = P[j * 3] - x;
              const ey = P[j * 3 + 1] - y;
              const ez = P[j * 3 + 2] - z;
              if (ex * ex + ey * ey + ez * ez < h2) {
                // a hash collision can list a cell twice; skip duplicates
                let dup = false;
                for (let q = 0; q < cnt; q++)
                  if (this.nbr[base + q] === j) {
                    dup = true;
                    break;
                  }
                if (dup) continue;
                this.nbr[base + cnt++] = j;
                if (cnt >= MAXN) break;
              }
            }
          }
      this.nc[i] = cnt;
    }
  }

  /** One sub-step. `env.collideParticle(i)` projects particle i out of every solid. */
  step(dt, p, env) {
    const n = this.n;
    if (!n) return;
    this.configure(p.particleRadius, p.kernelScale);
    const P = this.pos;
    const Q = this.prev;
    const V = this.vel;
    const D = this.dp;
    const L = this.lambda;
    const damp = Math.max(0, 1 - p.damping * dt);
    const vmax = 1.2;

    for (let i = 0; i < n; i++) {
      const o = i * 3;
      V[o + 1] -= p.gravity * dt;
      V[o] *= damp;
      V[o + 1] *= damp;
      V[o + 2] *= damp;
      const sp = Math.hypot(V[o], V[o + 1], V[o + 2]);
      if (sp > vmax) {
        V[o] *= vmax / sp;
        V[o + 1] *= vmax / sp;
        V[o + 2] *= vmax / sp;
      }
      Q[o] = P[o];
      Q[o + 1] = P[o + 1];
      Q[o + 2] = P[o + 2];
      P[o] += V[o] * dt;
      P[o + 1] += V[o + 1] * dt;
      P[o + 2] += V[o + 2] * dt;
      this.contact[i] = CONTACT_NONE;
      if (this.age[i] < 65535) this.age[i]++;
    }

    this.buildNeighbors();
    const h = this.h;
    const rho0 = this.rho0;
    const d0 = this.d;
    const maxStep = 0.5 * this.r;

    for (let it = 0; it < p.iterations; it++) {
      // density constraint multipliers
      for (let i = 0; i < n; i++) {
        const o = i * 3;
        let rho = 1;
        let gx = 0;
        let gy = 0;
        let gz = 0;
        let g2 = 0;
        const base = i * MAXN;
        const cnt = this.nc[i];
        for (let q = 0; q < cnt; q++) {
          const j = this.nbr[base + q];
          const dx = P[o] - P[j * 3];
          const dy = P[o + 1] - P[j * 3 + 1];
          const dz = P[o + 2] - P[j * 3 + 2];
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= this.h2) continue;
          rho += this.W(r2);
          const r = Math.sqrt(r2);
          if (r < 1e-9) continue;
          const t = 1 - r / h;
          const gm = (3 * t * t) / h / rho0;
          gx += (gm * dx) / r;
          gy += (gm * dy) / r;
          gz += (gm * dz) / r;
          g2 += gm * gm;
        }
        const C = rho / rho0 - 1;
        L[i] = C > 0 ? -C / (g2 + gx * gx + gy * gy + gz * gz + this.eps) : 0;
      }
      // position corrections
      for (let i = 0; i < n; i++) {
        const o = i * 3;
        let ax = 0;
        let ay = 0;
        let az = 0;
        const base = i * MAXN;
        const cnt = this.nc[i];
        const coh = cnt ? p.cohesion / cnt : 0;
        for (let q = 0; q < cnt; q++) {
          const j = this.nbr[base + q];
          const dx = P[o] - P[j * 3];
          const dy = P[o + 1] - P[j * 3 + 1];
          const dz = P[o + 2] - P[j * 3 + 2];
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= this.h2) continue;
          const r = Math.sqrt(r2);
          if (r < 1e-9) continue;
          const t = 1 - r / h;
          const gm = (3 * t * t) / h;
          // dp_i = 1/rho0 * sum (λi+λj) ∇W ; ∇W points from j to i with negative magnitude
          const s = (-(L[i] + L[j]) / rho0) * gm;
          ax += (s * dx) / r;
          ay += (s * dy) / r;
          az += (s * dz) / r;
          if (r > d0 && coh > 0) {
            const c = (coh * 0.5 * (r - d0)) / r;
            ax -= c * dx;
            ay -= c * dy;
            az -= c * dz;
          }
        }
        const m = Math.hypot(ax, ay, az);
        if (m > maxStep) {
          ax *= maxStep / m;
          ay *= maxStep / m;
          az *= maxStep / m;
        }
        D[o] = ax;
        D[o + 1] = ay;
        D[o + 2] = az;
      }
      for (let i = 0; i < n; i++) {
        const o = i * 3;
        P[o] += D[o];
        P[o + 1] += D[o + 1];
        P[o + 2] += D[o + 2];
        env.collideParticle(i);
      }
    }

    // velocities, contact friction
    const fr = p.friction;
    const staticV = 1.5 * fr * p.gravity * dt;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      let vx = (P[o] - Q[o]) / dt;
      let vy = (P[o + 1] - Q[o + 1]) / dt;
      let vz = (P[o + 2] - Q[o + 2]) / dt;
      if (this.contact[i] !== CONTACT_NONE) {
        const nx = this.cn[o];
        const ny = this.cn[o + 1];
        const nz = this.cn[o + 2];
        const sx = this.sv[o];
        const sy = this.sv[o + 1];
        const sz = this.sv[o + 2];
        let rx = vx - sx;
        let ry = vy - sy;
        let rz = vz - sz;
        let vn = rx * nx + ry * ny + rz * nz;
        if (vn < 0) {
          rx -= vn * nx;
          ry -= vn * ny;
          rz -= vn * nz;
          vn = 0;
        } else if (vn > MAX_SEPARATION) {
          // contact push-out is position correction, not a bounce: ice cream has no restitution
          rx -= (vn - MAX_SEPARATION) * nx;
          ry -= (vn - MAX_SEPARATION) * ny;
          rz -= (vn - MAX_SEPARATION) * nz;
          vn = MAX_SEPARATION;
        }
        const tx = rx - vn * nx;
        const ty = ry - vn * ny;
        const tz = rz - vn * nz;
        // static friction: slow sliding sticks (otherwise gravity feeds an endless creep)
        const ts = Math.sqrt(tx * tx + ty * ty + tz * tz);
        const kt = ts < staticV ? 0 : 1 - fr;
        vx = sx + vn * nx + tx * kt;
        vy = sy + vn * ny + ty * kt;
        vz = sz + vn * nz + tz * kt;
      }
      // freshly spawned particles start inside solids; their push-out is not real motion
      // (otherwise it turns into a large velocity and crumbs fly across the table)
      if (this.age[i] < 6) {
        const settled = this.contact[i] !== CONTACT_NONE;
        vx = settled ? this.sv[o] : Math.max(-0.2, Math.min(0.2, vx));
        vy = settled ? this.sv[o + 1] : Math.max(-0.2, Math.min(0.2, vy));
        vz = settled ? this.sv[o + 2] : Math.max(-0.2, Math.min(0.2, vz));
      }
      D[o] = vx;
      D[o + 1] = vy;
      D[o + 2] = vz;
    }
    // XSPH viscosity
    const visc = p.viscosity;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      let ax = 0;
      let ay = 0;
      let az = 0;
      let ws = 0;
      const base = i * MAXN;
      const cnt = this.nc[i];
      for (let q = 0; q < cnt; q++) {
        const j = this.nbr[base + q];
        const dx = P[o] - P[j * 3];
        const dy = P[o + 1] - P[j * 3 + 1];
        const dz = P[o + 2] - P[j * 3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= this.h2) continue;
        const w = this.W(r2);
        ax += (D[j * 3] - D[o]) * w;
        ay += (D[j * 3 + 1] - D[o + 1]) * w;
        az += (D[j * 3 + 2] - D[o + 2]) * w;
        ws += w;
      }
      const k = ws > 0 ? visc / Math.max(ws, 1) : 0;
      V[o] = D[o] + ax * k;
      V[o + 1] = D[o + 1] + ay * k;
      V[o + 2] = D[o + 2] + az * k;
    }
  }
}
