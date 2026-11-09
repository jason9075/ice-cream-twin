
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

struct Hit { ok : bool, p : vec3f, steps : i32, n : vec3f, voxel : vec3i };

// Traverse actual cells, testing the stored SDF sign without interpolation. This shows the
// union of occupied cubes, including their true silhouette and depth, rather than a grid overlay.
fn voxelMarch(ro : vec3f, rd : vec3f, tn : f32, tf : f32, entry : vec3f) -> Hit {
  var h : Hit;
  let cell = U.prm.x;
  let dims = vec3i(round(U.size.xyz / cell));
  var t = tn;
  var v = clamp(vec3i(floor((ro + rd * (tn + cell * 0.0001)) / cell)), vec3i(0), dims - 1);
  let step = select(vec3i(-1), vec3i(1), rd >= vec3f(0.0));
  let safeDir = vec3f(step) * max(abs(rd), vec3f(1e-9));
  let boundary = (vec3f(v) + select(vec3f(0.0), vec3f(1.0), rd >= vec3f(0.0))) * cell;
  var next = select(vec3f(1e30), (boundary - ro) / safeDir, abs(rd) > vec3f(1e-9));
  let delta = cell / max(abs(rd), vec3f(1e-9));
  var n = vec3f(0.0, 0.0, -f32(step.z));
  if (entry.x >= entry.y && entry.x >= entry.z) { n = vec3f(-f32(step.x), 0.0, 0.0); }
  else if (entry.y >= entry.z) { n = vec3f(0.0, -f32(step.y), 0.0); }
  for (var i = 0; i < dims.x + dims.y + dims.z; i++) {
    if (t > tf || any(v < vec3i(0)) || any(v >= dims)) { break; }
    let d = textureLoad(sdfTex, v, 0).r;
    if (d < select(0.5, 0.0, FLUID)) {
      h.ok = true;
      h.p = ro + rd * t;
      h.steps = i;
      h.n = n;
      h.voxel = v;
      return h;
    }
    t = min(min(next.x, next.y), next.z);
    let crossed = next <= vec3f(t + cell * 1e-5);
    if (crossed.x) { n = vec3f(-f32(step.x), 0.0, 0.0); }
    else if (crossed.y) { n = vec3f(0.0, -f32(step.y), 0.0); }
    else { n = vec3f(0.0, 0.0, -f32(step.z)); }
    v += select(vec3i(0), step, crossed);
    next += select(vec3f(0.0), delta, crossed);
  }
  return h;
}

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
  let inv = select(vec3f(-1.0), vec3f(1.0), rd >= vec3f(0.0)) / max(abs(rd), vec3f(1e-9));
  let t0 = -ro * inv;
  let t1 = (U.size.xyz - ro) * inv;
  let tmn = min(t0, t1);
  let tmx = max(t0, t1);
  let tn = max(max(max(tmn.x, tmn.y), tmn.z), 0.0);
  let tf = min(min(tmx.x, tmx.y), tmx.z);
  var h : Hit;
  h.ok = false;
  if (tn >= tf) { return h; }
  if (debugMode() == 4) { return voxelMarch(ro, rd, tn, tf, tmn); }
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

// local-space normal with sub-millimetre relief from two octaves of value noise (ice crystals, pores)
fn microNormal(p : vec3f, n : vec3f) -> vec3f {
  var g = vec3f(0.0);
  for (var k = 0; k < 2; k++) {
    let f = select(1400.0, 3800.0, k == 1);
    let a = select(0.25, 0.14, k == 1);
    let q = p * f;
    let d = vec3f(noise3(q + vec3f(0.35, 0.0, 0.0)) - noise3(q - vec3f(0.35, 0.0, 0.0)),
                  noise3(q + vec3f(0.0, 0.35, 0.0)) - noise3(q - vec3f(0.0, 0.35, 0.0)),
                  noise3(q + vec3f(0.0, 0.0, 0.35)) - noise3(q - vec3f(0.0, 0.0, 0.35)));
    g += d * a;
  }
  // keep only the part across the surface
  return normalize(n - (g - n * dot(g, n)));
}

@fragment fn fs(i : VOut) -> FOut {
  let h = march(i.lp);
  if (!h.ok) { discard; }
  let p = h.p;
  let cell = U.prm.x;
  if (debugMode() == 4) {
    let nW = normalize((U.model * vec4f(h.n, 0.0)).xyz);
    let wp = (U.model * vec4f(p, 1.0)).xyz;
    var alb = textureLoad(colTex, h.voxel, 0).rgb;
    if (FLUID && U.size.w > 0.5) { alb = mix(alb, vec3f(0.1, 0.45, 1.0), 0.6); }
    // Outline the two tangential edges of each face, keeping the native cell size visible.
    let f = fract(p / cell);
    let edges = select(min(f, 1.0 - f), vec3f(1.0), abs(h.n) > vec3f(0.5));
    let edge = min(min(edges.x, edges.y), edges.z);
    let width = max(fwidth(edge), 0.005);
    let face = smoothstep(0.035 - width, 0.035 + width, edge);
    alb *= mix(0.45, 1.0, face);
    let hemi = mix(GROUND, SKY, 0.5 + 0.5 * nW.y);
    let lit = hemi + SUN * max(dot(nW, F.lightDir.xyz), 0.0) * shadowAt(wp, nW) * 0.8;
    var o : FOut;
    o.depth = depthOf(p);
    o.color = vec4f(toScreen(alb * lit), 1.0);
    return o;
  }
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

  var c : vec3f;
  if (F.rayDir.w > 0.5) {
    // matte ice cream: frozen foam, not a wet surface. Micro relief (ice crystals, air pockets) perturbs
    // the normal, light scatters a little under the surface (wrapped, slightly saturated in the shade),
    // no sharp highlight, only a soft sheen at grazing angles.
    let nm = microNormal(p, n);
    let nM = normalize((U.model * vec4f(nm, 0.0)).xyz);
    let pore = 0.86 + 0.14 * smoothstep(0.2, 0.7, noise3(p * 3200.0 + vec3f(5.0)));
    let albM = alb * pore;
    let ndl = dot(nM, L);
    let wrap = clamp((ndl + 0.5) / 1.5, 0.0, 1.0);
    let scatter = albM * albM * 1.6; // light that went under the surface comes out in the flavour's colour
    let lit = mix(scatter, albM, clamp(ndl * 2.0, 0.0, 1.0));
    let hemi = mix(GROUND, SKY, 0.5 + 0.5 * nM.y);
    let H = normalize(L + V);
    let sheen = pow(1.0 - max(dot(nM, V), 0.0), 4.0) * 0.12;
    let soft = pow(max(dot(nM, H), 0.0), 6.0) * 0.035 * sh * (1.0 - speck);
    c = albM * hemi * ao + lit * SUN * wrap * sh * 0.82 + albM * sheen + vec3f(soft);
  } else {
    let diff = clamp((dot(nW, L) + 0.35) / 1.35, 0.0, 1.0);
    let H = normalize(L + V);
    let spec = pow(max(dot(nW, H), 0.0), 40.0) * 0.16 * sh * (1.0 - speck * 0.5);
    let back = pow(clamp(dot(V, -L), 0.0, 1.0), 3.0) * 0.2;
    let hemi = mix(GROUND, SKY, 0.5 + 0.5 * nW.y);
    c = alb * (hemi * ao + SUN * diff * sh * 0.8) + alb * back * 0.3 + vec3f(spec);
    c += alb * pow(1.0 - max(dot(nW, V), 0.0), 3.0) * 0.1;
  }

  o.color = vec4f(toScreen(c), 1.0);
  return o;
}

@fragment fn fsShadow(i : VOut) -> @builtin(frag_depth) f32 {
  let h = march(i.lp);
  if (!h.ok) { discard; }
  return depthOf(h.p);
}
