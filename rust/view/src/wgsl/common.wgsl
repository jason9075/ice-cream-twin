// Shared by every render shader. All world units are metres, Y up.
struct Frame {
  viewProj : mat4x4f,
  lightVP  : mat4x4f,
  camPos   : vec4f,   // w = 1: orthographic pass (shadow map)
  lightDir : vec4f,   // xyz: unit vector toward the sun
  rayDir   : vec4f,   // xyz: ray direction of the orthographic pass; w = 1: matte ice cream
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

// debug view (F.lightDir.w): 0 shaded, 1 normals, 2 albedo (unlit), 3 sphere-tracing steps, 4 voxels
fn debugMode() -> i32 { return i32(F.lightDir.w + 0.5); }
fn heat(t : f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  return clamp(vec3f(1.5 - abs(4.0 * x - 3.0), 1.5 - abs(4.0 * x - 2.0), 1.5 - abs(4.0 * x - 1.0)), vec3f(0.0), vec3f(1.0));
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
