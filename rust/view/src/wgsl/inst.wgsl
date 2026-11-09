
// Instanced spheres for particles and crumbs, read straight from the simulation's packed layout:
// per instance (f32 x, y, z, u32 rgba). O.mat = (metal, rough, radius, unlit).
struct Obj { model : mat4x4f, color : vec4f, mat : vec4f };
@group(1) @binding(0) var<uniform> O : Obj;
@group(1) @binding(1) var<storage, read> I : array<vec4u>;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) wp : vec3f,
  @location(1) n : vec3f,
  @location(2) col : vec3f,
};

@vertex fn vs(@location(0) p : vec3f, @location(1) n : vec3f, @builtin(instance_index) ii : u32) -> VOut {
  var o : VOut;
  let it = I[ii];
  let w = bitcast<vec3f>(it.xyz) + p * O.mat.z;
  o.pos = F.viewProj * vec4f(w, 1.0);
  o.wp = w;
  o.n = n;
  o.col = unpack4x8unorm(it.w).rgb;
  return o;
}

@fragment fn fs(i : VOut) -> @location(0) vec4f {
  if (O.mat.w > 0.5) { return vec4f(i.col, 1.0); }   // unlit (debug samples)
  let m = debugMode();
  if (m == 1) { return vec4f(normalize(i.n) * 0.5 + 0.5, 1.0); }
  if (m == 2) { return vec4f(pow(i.col, vec3f(1.0 / 2.2)), 1.0); }
  let c = shade(i.col, normalize(i.n), i.wp, O.mat.x, O.mat.y);
  return vec4f(toScreen(c), 1.0);
}
