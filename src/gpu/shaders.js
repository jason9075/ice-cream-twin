// WGSL shaders. All world units are metres, Y up.

const common = /* wgsl */ `
struct Frame {
  viewProj : mat4x4f,
  lightVP  : mat4x4f,
  camPos   : vec4f,   // w = 1: orthographic pass (shadow map)
  lightDir : vec4f,   // xyz: unit vector toward the sun
  rayDir   : vec4f,   // xyz: ray direction of the orthographic pass
};
@group(0) @binding(0) var<uniform> F : Frame;
@group(0) @binding(1) var shadowMap : texture_depth_2d;
@group(0) @binding(2) var shadowSamp : sampler_comparison;

const SUN    = vec3f(1.00, 0.94, 0.86) * 2.1;
const SKY    = vec3f(0.62, 0.68, 0.80) * 0.75;
const GROUND = vec3f(0.42, 0.34, 0.27) * 0.55;

fn shadowAt(p : vec3f, n : vec3f) -> f32 {
  let lp = F.lightVP * vec4f(p + n * 0.0015, 1.0);
  let uv = lp.xy * vec2f(0.5, -0.5) + vec2f(0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || lp.z > 1.0) { return 1.0; }
  let d = lp.z - 0.0006;
  let texel = 1.0 / f32(textureDimensions(shadowMap).x);
  var s = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(x), f32(y)) * texel * 1.5, d);
    }
  }
  return s / 9.0;
}

fn envColor(r : vec3f, rough : f32) -> vec3f {
  let t = 0.5 + 0.5 * r.y;
  var c = mix(vec3f(0.32, 0.27, 0.23), vec3f(0.92, 0.93, 0.98), smoothstep(0.35, 0.75, t));
  let w = rough * 0.35;
  c += smoothstep(0.86 - w, 0.97, dot(r, normalize(vec3f(0.45, 1.0, 0.35)))) * 3.0;
  c += smoothstep(0.90 - w, 0.98, dot(r, normalize(vec3f(-0.8, 0.45, -0.3)))) * 1.4;
  return c;
}

fn aces(x : vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
fn toScreen(c : vec3f) -> vec3f { return pow(aces(c * 0.85), vec3f(1.0 / 2.2)); }

fn hash3(q : vec3f) -> f32 {
  var p = fract(q * 0.3183099 + vec3f(0.1));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
fn noise3(x : vec3f) -> f32 {
  let i = floor(x);
  var f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3f(1,0,0)), f.x), mix(hash3(i + vec3f(0,1,0)), hash3(i + vec3f(1,1,0)), f.x), f.y),
             mix(mix(hash3(i + vec3f(0,0,1)), hash3(i + vec3f(1,0,1)), f.x), mix(hash3(i + vec3f(0,1,1)), hash3(i + vec3f(1,1,1)), f.x), f.y), f.z);
}

fn shade(alb : vec3f, nIn : vec3f, p : vec3f, metal : f32, rough : f32) -> vec3f {
  let V = normalize(F.camPos.xyz - p);
  var n = nIn;
  if (dot(n, V) < 0.0) { n = -n; }
  let L = F.lightDir.xyz;
  let sh = shadowAt(p, n);
  let ndl = max(dot(n, L), 0.0);
  let H = normalize(L + V);
  let sp = mix(500.0, 10.0, rough);
  let spec = pow(max(dot(n, H), 0.0), sp) * (sp + 8.0) / 25.0;
  let hemi = mix(GROUND, SKY, 0.5 + 0.5 * n.y);
  let env = envColor(reflect(-V, n), rough);
  let fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
  let diel = alb * (hemi + SUN * ndl * sh) + env * fres * 0.35 * (1.0 - rough) + SUN * spec * 0.04 * sh;
  let met = alb * (env * 0.85 + SUN * spec * sh * 0.3);
  return mix(diel, met, metal);
}
`;

export const meshWGSL = /* wgsl */ `
${common}
struct Obj { model : mat4x4f, color : vec4f, mat : vec4f };  // mat: metal, rough, opacity, kind
@group(1) @binding(0) var<uniform> O : Obj;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp : vec3f,
  @location(1) n : vec3f,
};

@vertex fn vs(@location(0) p : vec3f, @location(1) n : vec3f) -> VOut {
  var o : VOut;
  let w = O.model * vec4f(p, 1.0);
  o.pos = F.viewProj * w;
  o.wp = w.xyz;
  o.n = (O.model * vec4f(n, 0.0)).xyz;
  return o;
}

fn wood(p : vec3f) -> vec3f {
  let g = p.z * 38.0 + noise3(p * vec3f(2.0, 2.0, 25.0)) * 2.0 + noise3(p * 160.0) * 0.25;
  let ring = smoothstep(0.1, 0.5, abs(fract(g) - 0.5));
  let base = mix(vec3f(0.23, 0.12, 0.055), vec3f(0.30, 0.17, 0.085), ring);
  return base * (0.92 + 0.16 * noise3(p * vec3f(6.0, 6.0, 70.0)));
}

@fragment fn fs(i : VOut) -> @location(0) vec4f {
  var alb = O.color.rgb;
  if (O.mat.w > 0.5) { alb = wood(i.wp); }
  let c = shade(alb, normalize(i.n), i.wp, O.mat.x, O.mat.y);
  return vec4f(toScreen(c), O.mat.z);
}
`;

export const instWGSL = /* wgsl */ `
${common}
struct Obj { model : mat4x4f, color : vec4f, mat : vec4f };
struct Inst { p : vec4f, c : vec4f };   // xyz + radius, linear rgb
@group(1) @binding(0) var<uniform> O : Obj;
@group(1) @binding(1) var<storage, read> I : array<Inst>;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp : vec3f,
  @location(1) n : vec3f,
  @location(2) col : vec3f,
};

@vertex fn vs(@location(0) p : vec3f, @location(1) n : vec3f, @builtin(instance_index) ii : u32) -> VOut {
  var o : VOut;
  let it = I[ii];
  let w = it.p.xyz + p * it.p.w;
  o.pos = F.viewProj * vec4f(w, 1.0);
  o.wp = w;
  o.n = n;
  o.col = it.c.rgb;
  return o;
}

@fragment fn fs(i : VOut) -> @location(0) vec4f {
  if (O.mat.w > 0.5) { return vec4f(i.col, 1.0); }   // unlit (debug samples)
  let c = shade(i.col, normalize(i.n), i.wp, O.mat.x, O.mat.y);
  return vec4f(toScreen(c), 1.0);
}
`;

/**
 * Sphere-traces an 8-bit SDF volume inside its proxy box (back faces), like Claybook's
 * FRenderPrimarySDFCS. The fragment writes frag_depth so it composites with the meshes and
 * the same code renders into the shadow map (orthographic rays).
 */
export const volumeWGSL = /* wgsl */ `
${common}
struct Vol { model : mat4x4f, inv : mat4x4f, size : vec4f, prm : vec4f };  // size.w = band, prm.x = cell
@group(1) @binding(0) var<uniform> U : Vol;
@group(1) @binding(1) var sdfTex : texture_3d<f32>;
@group(1) @binding(2) var colTex : texture_3d<f32>;
@group(1) @binding(3) var samp : sampler;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) lp : vec3f,
};

@vertex fn vs(@location(0) p : vec3f) -> VOut {
  var o : VOut;
  o.pos = F.viewProj * (U.model * vec4f(p, 1.0));
  o.lp = p;
  return o;
}

fn map(p : vec3f) -> f32 {
  return (textureSampleLevel(sdfTex, samp, p / U.size.xyz, 0.0).r * 2.0 - 1.0) * U.size.w;
}

struct Hit { ok : bool, p : vec3f };

fn march(lp : vec3f) -> Hit {
  var ro : vec3f;
  var rd : vec3f;
  if (F.camPos.w > 0.5) {
    rd = normalize((U.inv * vec4f(F.rayDir.xyz, 0.0)).xyz);
    ro = lp - rd * 2.0;
  } else {
    ro = (U.inv * vec4f(F.camPos.xyz, 1.0)).xyz;
    rd = normalize(lp - ro);
  }
  let inv = 1.0 / rd;
  let t0 = -ro * inv;
  let t1 = (U.size.xyz - ro) * inv;
  let tmn = min(t0, t1);
  let tmx = max(t0, t1);
  let tn = max(max(max(tmn.x, tmn.y), tmn.z), 0.0);
  let tf = min(min(tmx.x, tmx.y), tmx.z);
  var h : Hit;
  h.ok = false;
  if (tn >= tf) { return h; }
  let cell = U.prm.x;
  var t = tn;
  for (var i = 0; i < 220; i++) {
    let d = map(ro + rd * t);
    if (d < cell * 0.05) { h.ok = true; break; }
    t += max(d * 0.85, cell * 0.06);
    if (t > tf) { break; }
  }
  if (h.ok) {
    // secant refinement for a crisper silhouette
    let tb = max(t - cell * 0.12, tn);
    let da = map(ro + rd * tb);
    let db = map(ro + rd * t);
    if (da > 0.0 && db < da) { t = tb + (t - tb) * da / (da - db); }
    h.p = ro + rd * t;
  }
  return h;
}

fn depthOf(lp : vec3f) -> f32 {
  let c = F.viewProj * (U.model * vec4f(lp, 1.0));
  return clamp(c.z / c.w, 0.0, 1.0);
}

struct FOut { @location(0) color : vec4f, @builtin(frag_depth) depth : f32 };

@fragment fn fs(i : VOut) -> FOut {
  let h = march(i.lp);
  if (!h.ok) { discard; }
  let p = h.p;
  let cell = U.prm.x;
  let e = vec2f(cell * 0.6, 0.0);
  let n = normalize(vec3f(map(p + e.xyy) - map(p - e.xyy),
                          map(p + e.yxy) - map(p - e.yxy),
                          map(p + e.yyx) - map(p - e.yyx)));
  let nW = normalize((U.model * vec4f(n, 0.0)).xyz);
  let wp = (U.model * vec4f(p, 1.0)).xyz;
  let V = normalize(F.camPos.xyz - wp);
  let L = F.lightDir.xyz;

  var ao = 0.0;
  for (var k = 1; k <= 4; k++) {
    let hh = cell * 1.3 * f32(k);
    ao += (hh - map(p + n * hh)) / hh;
  }
  ao = clamp(1.0 - 0.2 * ao, 0.3, 1.0);
  let sh = shadowAt(wp, nW);

  let col = textureSampleLevel(colTex, samp, p / U.size.xyz, 0.0);
  var alb = col.rgb * (0.92 + 0.14 * (noise3(p * 700.0) * 0.6 + noise3(p * 2100.0) * 0.4));
  // specks: alpha sets size and density (vanilla bean dots ... mint chocolate chips)
  let a = col.a;
  let sn = noise3(p * mix(1900.0, 520.0, a) + vec3f(17.0));
  let th = 1.0 - 0.17 * a;
  var speck = 0.0;
  if (a > 0.02) { speck = smoothstep(th, th + 0.03, sn); }
  alb = mix(alb, vec3f(0.08, 0.04, 0.02) * (0.6 + 0.8 * alb), speck);

  let diff = clamp((dot(nW, L) + 0.35) / 1.35, 0.0, 1.0);
  let H = normalize(L + V);
  let spec = pow(max(dot(nW, H), 0.0), 40.0) * 0.16 * sh * (1.0 - speck * 0.5);
  let back = pow(clamp(dot(V, -L), 0.0, 1.0), 3.0) * 0.2;
  let hemi = mix(GROUND, SKY, 0.5 + 0.5 * nW.y);
  var c = alb * (hemi * ao + SUN * diff * sh * 0.8) + alb * back * 0.3 + vec3f(spec);
  c += alb * pow(1.0 - max(dot(nW, V), 0.0), 3.0) * 0.1;

  var o : FOut;
  o.color = vec4f(toScreen(c), 1.0);
  o.depth = depthOf(p);
  return o;
}

@fragment fn fsShadow(i : VOut) -> @builtin(frag_depth) f32 {
  let h = march(i.lp);
  if (!h.ok) { discard; }
  return depthOf(h.p);
}
`;
