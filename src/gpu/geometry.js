// Mesh generation. Every mesh is { vertices: Float32Array [px,py,pz,nx,ny,nz]*, indices: Uint32Array }.
// Shapes are built from the same parameters as the server's collision SDFs (scene in the hello message).

class Builder {
  constructor() {
    this.v = [];
    this.i = [];
  }
  get count() {
    return this.v.length / 6;
  }
  vert(p, n) {
    this.v.push(p[0], p[1], p[2], n[0], n[1], n[2]);
    return this.count - 1;
  }
  tri(a, b, c) {
    this.i.push(a, b, c);
  }
  /** Grid of (rows+1) x (cols+1) vertices from f(u, v) -> [pos, normal]. */
  grid(rows, cols, f, flip = false) {
    const base = this.count;
    for (let r = 0; r <= rows; r++)
      for (let c = 0; c <= cols; c++) {
        const [p, n] = f(r / rows, c / cols);
        this.vert(p, n);
      }
    const w = cols + 1;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const a = base + r * w + c;
        const b = a + 1;
        const d = a + w;
        const e = d + 1;
        if (flip) {
          this.tri(a, b, d);
          this.tri(b, e, d);
        } else {
          this.tri(a, d, b);
          this.tri(b, d, e);
        }
      }
  }
  build() {
    return { vertices: new Float32Array(this.v), indices: new Uint32Array(this.i) };
  }
}

const norm = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};

/** Surface of revolution around +Y from a 2D profile [[radius, y], ...]. */
export function lathe(profile, segments = 96) {
  const b = new Builder();
  const n = profile.length;
  const pn = profile.map((p, k) => {
    const a = profile[Math.max(0, k - 1)];
    const c = profile[Math.min(n - 1, k + 1)];
    const tx = c[0] - a[0];
    const ty = c[1] - a[1];
    const l = Math.hypot(tx, ty) || 1;
    return [ty / l, -tx / l]; // tangent rotated -90°
  });
  b.grid(n - 1, segments, (u, v) => {
    const k = Math.round(u * (n - 1));
    const th = v * Math.PI * 2;
    const cs = Math.cos(th);
    const sn = Math.sin(th);
    const [r, y] = profile[k];
    const [nr, ny] = pn[k];
    return [
      [r * cs, y, r * sn],
      [nr * cs, ny, nr * sn],
    ];
  }, true);
  return b.build();
}

export function sphere(lat = 10, lon = 16) {
  const b = new Builder();
  b.grid(lat, lon, (u, v) => {
    const th = u * Math.PI;
    const ph = v * Math.PI * 2;
    const p = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
    return [p, p];
  }, true);
  return b.build();
}

/** Axis-aligned box from min to max corner. */
export function box(min, max) {
  const b = new Builder();
  const faces = [
    [0, 1, [1, 0, 0]],
    [0, 0, [-1, 0, 0]],
    [1, 1, [0, 1, 0]],
    [1, 0, [0, -1, 0]],
    [2, 1, [0, 0, 1]],
    [2, 0, [0, 0, -1]],
  ];
  for (const [axis, side, n] of faces) {
    const u = (axis + 1) % 3;
    const w = (axis + 2) % 3;
    const base = b.count;
    for (const [su, sw] of [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ]) {
      const p = [0, 0, 0];
      p[axis] = side ? max[axis] : min[axis];
      p[u] = su ? max[u] : min[u];
      p[w] = sw ? max[w] : min[w];
      b.vert(p, n);
    }
    if (side) {
      b.tri(base, base + 1, base + 2);
      b.tri(base, base + 2, base + 3);
    } else {
      b.tri(base, base + 2, base + 1);
      b.tri(base, base + 3, base + 2);
    }
  }
  return b.build();
}

/**
 * Spoon: half-ellipsoid cup shell (outer, inner, rim) + polyline handle (cylinders + joints).
 * `S` = {a, b, c, t, handleR, handle} (the server's scene.spoon).
 */
export function spoon(S) {
  const b = new Builder();
  const { a, c, t } = S;
  const bb = S.b;
  const h = t / 2;
  const nTh = 22;
  const nPh = 72;
  const cup = (A, B, C, sign) => (u, v) => {
    const th = u * (Math.PI / 2);
    const ph = v * Math.PI * 2;
    const p = [A * Math.sin(th) * Math.cos(ph), -B * Math.cos(th), C * Math.sin(th) * Math.sin(ph)];
    const n = norm([p[0] / (A * A), p[1] / (B * B), p[2] / (C * C)]);
    return [p, [n[0] * sign, n[1] * sign, n[2] * sign]];
  };
  b.grid(nTh, nPh, cup(a + h, bb + h, c + h, 1), false);
  b.grid(nTh, nPh, cup(a - h, bb - h, c - h, -1), true);
  // rim strip (y = 0), facing up
  b.grid(1, nPh, (u, v) => {
    const ph = v * Math.PI * 2;
    const A = u ? a - h : a + h;
    const C = u ? c - h : c + h;
    return [
      [A * Math.cos(ph), 0, C * Math.sin(ph)],
      [0, 1, 0],
    ];
  }, true);

  const R = S.handleR;
  const H = S.handle;
  for (let k = 0; k < H.length - 1; k++) {
    const p0 = H[k];
    const p1 = H[k + 1];
    const ax = norm([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]);
    const ref = Math.abs(ax[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const e1 = norm([ax[1] * ref[2] - ax[2] * ref[1], ax[2] * ref[0] - ax[0] * ref[2], ax[0] * ref[1] - ax[1] * ref[0]]);
    const e2 = [ax[1] * e1[2] - ax[2] * e1[1], ax[2] * e1[0] - ax[0] * e1[2], ax[0] * e1[1] - ax[1] * e1[0]];
    b.grid(1, 24, (u, v) => {
      const ph = v * Math.PI * 2;
      const n = [0, 1, 2].map((i) => e1[i] * Math.cos(ph) + e2[i] * Math.sin(ph));
      const o = u ? p1 : p0;
      return [[0, 1, 2].map((i) => o[i] + n[i] * R), n];
    }, true);
  }
  const s = sphere(10, 20);
  for (const p of H) {
    const base = b.count;
    for (let i = 0; i < s.vertices.length; i += 6) {
      const n = [s.vertices[i], s.vertices[i + 1], s.vertices[i + 2]];
      b.vert([p[0] + n[0] * R, p[1] + n[1] * R, p[2] + n[2] * R], n);
    }
    for (const idx of s.indices) b.i.push(base + idx);
  }
  return b.build();
}
