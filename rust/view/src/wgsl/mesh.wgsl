// Lit meshes: table (procedural wood), bowls, spoon, SO-101, translucent ghosts and grid boxes.
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
  let m = debugMode();
  if (m == 1) { return vec4f(normalize(i.n) * 0.5 + 0.5, O.mat.z); }
  if (m == 2) { return vec4f(pow(alb, vec3f(1.0 / 2.2)), O.mat.z); }
  let c = shade(alb, normalize(i.n), i.wp, O.mat.x, O.mat.y);
  return vec4f(toScreen(c), O.mat.z);
}
