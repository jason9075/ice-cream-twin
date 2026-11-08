// Analytic signed distance functions for the static scene (bowls, table) and the spoon.
// Units: metres. Frame: three.js world, Y up. Negative = inside.

// ---------------------------------------------------------------- bowls
// A bowl is a "cut hollow sphere" (Inigo Quilez): a spherical shell of mid radius r and
// half-thickness t, kept below the plane y = cy + h, resting on the table at y = 0.
const BOWL_R = 0.075;
const BOWL_T = 0.003;
const BOWL_H = 0.025;

export const FLAVORS = [
  { id: 'vanilla', label: 'Vanilla', color: '#f4e7c3', speck: 0.35, speckColor: '#5a3b1c' },
  { id: 'strawberry', label: 'Strawberry', color: '#f3a9bb', speck: 0.25, speckColor: '#b0283c' },
  { id: 'mint', label: 'Mint choc chip', color: '#bfeed6', speck: 1.0, speckColor: '#3b2416' },
];

export const BOWL_SPACING = 0.18;

export const BOWLS = FLAVORS.map((f, i) => {
  const cx = (i - 1) * BOWL_SPACING;
  const cy = BOWL_R + BOWL_T;
  return {
    index: i,
    flavor: f,
    cx,
    cy,
    cz: 0,
    r: BOWL_R,
    t: BOWL_T,
    h: BOWL_H,
    w: Math.sqrt(BOWL_R * BOWL_R - BOWL_H * BOWL_H), // rim radius (mid surface)
    rimY: cy + BOWL_H,
    innerR: BOWL_R - BOWL_T,
  };
});

export function sdBowl(B, x, y, z) {
  const px = x - B.cx;
  const py = y - B.cy;
  const pz = z - B.cz;
  const qx = Math.sqrt(px * px + pz * pz);
  let d;
  if (B.h * qx < B.w * py) {
    const dx = qx - B.w;
    const dy = py - B.h;
    d = Math.sqrt(dx * dx + dy * dy);
  } else {
    d = Math.abs(Math.sqrt(qx * qx + py * py) - B.r);
  }
  return d - B.t;
}

/** Index of the bowl whose centre column is closest in x (bowls are laid out along x). */
export function nearestBowl(x) {
  let i = Math.round(x / BOWL_SPACING) + 1;
  if (i < 0) i = 0;
  if (i > BOWLS.length - 1) i = BOWLS.length - 1;
  return BOWLS[i];
}

/** Static environment: table plane (y = 0) and every bowl. */
export function sdEnv(x, y, z) {
  const b = nearestBowl(x);
  return Math.min(y, sdBowl(b, x, y, z));
}

const EPS = 1e-5;
export function gradEnv(x, y, z, out) {
  const gx = sdEnv(x + EPS, y, z) - sdEnv(x - EPS, y, z);
  const gy = sdEnv(x, y + EPS, z) - sdEnv(x, y - EPS, z);
  const gz = sdEnv(x, y, z + EPS) - sdEnv(x, y, z - EPS);
  const l = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
  out[0] = gx / l;
  out[1] = gy / l;
  out[2] = gz / l;
  return out;
}

/** Lathe profile (radius, height) matching sdBowl exactly, for the render mesh. */
export function bowlProfile(B, segs = 48) {
  const pts = [];
  const alpha = Math.atan2(B.h, B.w);
  const ro = B.r + B.t;
  const ri = B.r - B.t;
  for (let i = 0; i <= segs; i++) {
    const a = -Math.PI / 2 + (alpha + Math.PI / 2) * (i / segs);
    pts.push([ro * Math.cos(a), B.cy + ro * Math.sin(a)]);
  }
  for (let i = 1; i < 16; i++) {
    const a = alpha + Math.PI * (i / 16);
    pts.push([B.w + B.t * Math.cos(a), B.cy + B.h + B.t * Math.sin(a)]);
  }
  for (let i = 0; i <= segs; i++) {
    const a = alpha - (alpha + Math.PI / 2) * (i / segs);
    pts.push([Math.max(0, ri * Math.cos(a)), B.cy + ri * Math.sin(a)]);
  }
  pts[0][0] = 0;
  return pts;
}

// ---------------------------------------------------------------- spoon (local frame)
// Origin = centre of the cup opening. +Y = cup opening normal, +X = toward the handle.
// The cup is the lower half (y <= 0) of an ellipsoid shell with mid semi-axes a, b, c.
export const SPOON = {
  a: 0.028,
  b: 0.011,
  c: 0.019,
  t: 0.0016,
  handleR: 0.0032,
  handle: [
    [0.024, 0.001, 0],
    [0.042, 0.0045, 0],
    [0.078, 0.013, 0],
    [0.165, 0.03, 0],
  ],
};

export function sdEllipsoid(x, y, z, a, b, c) {
  const ax = x / a;
  const by = y / b;
  const cz = z / c;
  const k0 = Math.sqrt(ax * ax + by * by + cz * cz);
  const ax2 = x / (a * a);
  const by2 = y / (b * b);
  const cz2 = z / (c * c);
  const k1 = Math.sqrt(ax2 * ax2 + by2 * by2 + cz2 * cz2);
  if (k1 < 1e-12) return -Math.min(a, b, c);
  return (k0 * (k0 - 1)) / k1;
}

/** The metal of the cup. */
export function sdSpoonShell(x, y, z) {
  const S = SPOON;
  const e = sdEllipsoid(x, y, z, S.a, S.b, S.c);
  return Math.max(Math.abs(e) - S.t / 2, y);
}

/** The empty space inside the cup, below the rim plane. Whatever is in here gets scooped. */
export function sdHullInterior(x, y, z) {
  const S = SPOON;
  const h = S.t / 2;
  return Math.max(sdEllipsoid(x, y, z, S.a - h, S.b - h, S.c - h), y);
}

function sdCapsule(x, y, z, ax, ay, az, bx, by, bz, r) {
  const pax = x - ax;
  const pay = y - ay;
  const paz = z - az;
  const bax = bx - ax;
  const bay = by - ay;
  const baz = bz - az;
  let h = (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz);
  h = h < 0 ? 0 : h > 1 ? 1 : h;
  const dx = pax - bax * h;
  const dy = pay - bay * h;
  const dz = paz - baz * h;
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
}

export function sdHandle(x, y, z) {
  const H = SPOON.handle;
  let d = Infinity;
  for (let i = 0; i < H.length - 1; i++) {
    const a = H[i];
    const b = H[i + 1];
    const di = sdCapsule(x, y, z, a[0], a[1], a[2], b[0], b[1], b[2], SPOON.handleR);
    if (di < d) d = di;
  }
  return d;
}

export function sdSpoonSolid(x, y, z) {
  return Math.min(sdSpoonShell(x, y, z), sdHandle(x, y, z));
}

export function gradLocal(fn, x, y, z, out) {
  const gx = fn(x + EPS, y, z) - fn(x - EPS, y, z);
  const gy = fn(x, y + EPS, z) - fn(x, y - EPS, z);
  const gz = fn(x, y, z + EPS) - fn(x, y, z - EPS);
  const l = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
  out[0] = gx / l;
  out[1] = gy / l;
  out[2] = gz / l;
  return out;
}

export function smin(a, b, k) {
  if (k <= 0) return a < b ? a : b;
  let h = 0.5 + (0.5 * (b - a)) / k;
  h = h < 0 ? 0 : h > 1 ? 1 : h;
  return b + (a - b) * h - k * h * (1 - h);
}
