
// Sphere-traces an SDF volume inside its proxy box (back faces), like Claybook's FRenderPrimarySDFCS;
// writes frag_depth so it composites with the meshes, and renders into the shadow map the same way.
// FLUID (prepended by the renderer): the loose-particle surface (fluid_splat.wgsl): metric distance in
// rgba16float .r, a unit box scaled to the region in use, the corner prm.yzw texels wide of a larger
// texture, U written by the GPU (fluid setup), size.w = 1 tints it (debug).
struct Vol { model : mat4x4f, inv : mat4x4f, size : vec4f, prm : vec4f };  // size.w = band, prm.x = cell
#ifdef FLUID
@group(1) @binding(0) var<storage, read> U : Vol;
#else
@group(1) @binding(0) var<uniform> U : Vol;
#endif
@group(1) @binding(1) var sdfTex : texture_3d<f32>;
@group(1) @binding(2) var colTex : texture_3d<f32>;
@group(1) @binding(3) var samp : sampler;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) lp : vec3f,
};

@vertex fn vs(@location(0) p : vec3f) -> VOut {
  var o : VOut;
  var lp = p;
  if (FLUID) { lp = p * U.size.xyz; }
  o.pos = F.viewProj * (U.model * vec4f(lp, 1.0));
  o.lp = lp;
  return o;
}

// texture coordinate of a local point (fluid: clamped inside the region, the texels past it are stale)
fn uvw(p : vec3f) -> vec3f {
  if (FLUID) {
    let c = U.prm.x;
    return clamp(p, vec3f(0.5 * c), U.size.xyz - 0.5 * c) / (c * U.prm.yzw);
  }
  return p / U.size.xyz;
}

fn map(p : vec3f) -> f32 {
  if (FLUID) { return textureSampleLevel(sdfTex, samp, uvw(p), 0.0).r; }
  return (textureSampleLevel(sdfTex, samp, p / U.size.xyz, 0.0).r * 2.0 - 1.0) * U.size.w;
}

struct Hit { ok : bool, p : vec3f, steps : i32 };

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
    h.steps = i;
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

  let col = textureSampleLevel(colTex, samp, uvw(p), 0.0);
  var alb = col.rgb * (0.92 + 0.14 * (noise3(p * 700.0) * 0.6 + noise3(p * 2100.0) * 0.4));
  // specks: alpha sets size and density (vanilla bean dots ... mint chocolate chips)
  let a = col.a;
  let sn = noise3(p * mix(1900.0, 520.0, a) + vec3f(17.0));
  let th = 1.0 - 0.17 * a;
  var speck = 0.0;
  if (a > 0.02) { speck = smoothstep(th, th + 0.03, sn); }
  alb = mix(alb, vec3f(0.08, 0.04, 0.02) * (0.6 + 0.8 * alb), speck);
  if (FLUID && U.size.w > 0.5) { alb = mix(alb, vec3f(0.1, 0.45, 1.0), 0.6); }

  var o : FOut;
  o.depth = depthOf(p);
  let m = debugMode();
  if (m == 1) { o.color = vec4f(nW * 0.5 + 0.5, 1.0); return o; }
  if (m == 2) { o.color = vec4f(pow(alb, vec3f(1.0 / 2.2)), 1.0); return o; }
  if (m == 3) { o.color = vec4f(heat(f32(h.steps) / 80.0), 1.0); return o; }

  let diff = clamp((dot(nW, L) + 0.35) / 1.35, 0.0, 1.0);
  let H = normalize(L + V);
  let spec = pow(max(dot(nW, H), 0.0), 40.0) * 0.16 * sh * (1.0 - speck * 0.5);
  let back = pow(clamp(dot(V, -L), 0.0, 1.0), 3.0) * 0.2;
  let hemi = mix(GROUND, SKY, 0.5 + 0.5 * nW.y);
  var c = alb * (hemi * ao + SUN * diff * sh * 0.8) + alb * back * 0.3 + vec3f(spec);
  c += alb * pow(1.0 - max(dot(nW, V), 0.0), 3.0) * 0.1;

  o.color = vec4f(toScreen(c), 1.0);
  return o;
}

@fragment fn fsShadow(i : VOut) -> @builtin(frag_depth) f32 {
  let h = march(i.lp);
  if (!h.ok) { discard; }
  return depthOf(h.p);
}
