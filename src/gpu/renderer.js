import { meshWGSL, instWGSL, volumeWGSL } from './shaders.js';
export { createFluidSurface } from './fluid.js';
import { mat4, Vector3 } from '../math.js';

const MSAA = 4;
const SHADOW_SIZE = 2048;
const DEPTH_FORMAT = 'depth24plus';
const SHADOW_FORMAT = 'depth32float';
const VERTEX_LAYOUT = [
  {
    arrayStride: 24,
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
      { shaderLocation: 1, offset: 12, format: 'float32x3' },
    ],
  },
];
const VERTEX_LAYOUT_POS = [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }];

export async function createRenderer(canvas) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter =
    (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
    (await navigator.gpu.requestAdapter());
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const device = await adapter.requestDevice();
  device.lost.then((info) => console.error('WebGPU device lost:', info.message));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', e.error.message));
  return new Renderer(canvas, device);
}

export class Renderer {
  constructor(canvas, device) {
    this.canvas = canvas;
    this.device = device;
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.width = 0;
    this.height = 0;
    this.clearColor = { r: 0.83, g: 0.8, b: 0.76, a: 1 };

    const d = device;
    const V = GPUShaderStage.VERTEX;
    const Fr = GPUShaderStage.FRAGMENT;
    this.bglFrame = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | Fr, buffer: {} },
        { binding: 1, visibility: Fr, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: Fr, sampler: { type: 'comparison' } },
      ],
    });
    this.bglObj = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: V | Fr, buffer: {} }] });
    this.bglInst = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | Fr, buffer: {} },
        { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.bglVol = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | Fr, buffer: {} },
        { binding: 1, visibility: Fr, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: Fr, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: Fr, sampler: { type: 'filtering' } },
      ],
    });

    // frame uniforms: one for the camera pass, one for the shadow pass
    this.frameData = new Float32Array(64);
    this.frameBuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shadowFrameBuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shadowTex = d.createTexture({
      size: [SHADOW_SIZE, SHADOW_SIZE],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const dummyDepth = d.createTexture({ size: [1, 1], format: SHADOW_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING });
    const cmpSampler = d.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.linearSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.frameBG = d.createBindGroup({
      layout: this.bglFrame,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuf } },
        { binding: 1, resource: this.shadowTex.createView() },
        { binding: 2, resource: cmpSampler },
      ],
    });
    this.shadowFrameBG = d.createBindGroup({
      layout: this.bglFrame,
      entries: [
        { binding: 0, resource: { buffer: this.shadowFrameBuf } },
        { binding: 1, resource: dummyDepth.createView() },
        { binding: 2, resource: cmpSampler },
      ],
    });

    const meshMod = d.createShaderModule({ code: meshWGSL, label: 'mesh' });
    const instMod = d.createShaderModule({ code: instWGSL, label: 'inst' });
    const volMod = d.createShaderModule({ code: volumeWGSL(false), label: 'volume' });
    const fluidMod = d.createShaderModule({ code: volumeWGSL(true), label: 'fluid surface' });
    for (const m of [meshMod, instMod, volMod, fluidMod]) {
      m.getCompilationInfo().then((info) => {
        for (const msg of info.messages) console[msg.type === 'error' ? 'error' : 'warn'](`WGSL ${m.label}: ${msg.lineNum}:${msg.linePos} ${msg.message}`);
      });
    }

    const layout = (g1) => d.createPipelineLayout({ bindGroupLayouts: [this.bglFrame, g1] });
    const main = (opts) => ({
      multisample: { count: MSAA },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      ...opts,
    });
    const shadow = (opts) => ({
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 1.5 },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      ...opts,
    });
    const color = [{ format: this.format }];
    const blended = [
      {
        format: this.format,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      },
    ];

    this.pipes = {
      mesh: d.createRenderPipeline(main({
        layout: layout(this.bglObj),
        vertex: { module: meshMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT },
        fragment: { module: meshMod, entryPoint: 'fs', targets: color },
      })),
      ghost: d.createRenderPipeline(main({
        layout: layout(this.bglObj),
        vertex: { module: meshMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT },
        fragment: { module: meshMod, entryPoint: 'fs', targets: blended },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
      })),
      inst: d.createRenderPipeline(main({
        layout: layout(this.bglInst),
        vertex: { module: instMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT },
        fragment: { module: instMod, entryPoint: 'fs', targets: color },
      })),
      overlay: d.createRenderPipeline(main({
        layout: layout(this.bglInst),
        vertex: { module: instMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT },
        fragment: { module: instMod, entryPoint: 'fs', targets: color },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
      })),
      volume: d.createRenderPipeline(main({
        layout: layout(this.bglVol),
        vertex: { module: volMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT_POS },
        fragment: { module: volMod, entryPoint: 'fs', targets: color },
        primitive: { topology: 'triangle-list', cullMode: 'front' },
      })),
      fluid: d.createRenderPipeline(main({
        layout: layout(this.bglVol),
        vertex: { module: fluidMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT_POS },
        fragment: { module: fluidMod, entryPoint: 'fs', targets: color },
        primitive: { topology: 'triangle-list', cullMode: 'front' },
      })),
      fluidShadow: d.createRenderPipeline(shadow({
        layout: layout(this.bglVol),
        vertex: { module: fluidMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT_POS },
        fragment: { module: fluidMod, entryPoint: 'fsShadow', targets: [] },
        primitive: { topology: 'triangle-list', cullMode: 'front' },
      })),
      meshShadow: d.createRenderPipeline(shadow({
        layout: layout(this.bglObj),
        vertex: { module: meshMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT },
      })),
      instShadow: d.createRenderPipeline(shadow({
        layout: layout(this.bglInst),
        vertex: { module: instMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT },
      })),
      volumeShadow: d.createRenderPipeline(shadow({
        layout: layout(this.bglVol),
        vertex: { module: volMod, entryPoint: 'vs', buffers: VERTEX_LAYOUT_POS },
        fragment: { module: volMod, entryPoint: 'fsShadow', targets: [] },
        primitive: { topology: 'triangle-list', cullMode: 'front' },
      })),
    };
  }

  // ---------------------------------------------------------------- resources
  createMesh(geo) {
    const d = this.device;
    const vb = d.createBuffer({ size: geo.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(vb, 0, geo.vertices);
    const ib = d.createBuffer({ size: geo.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ib, 0, geo.indices);
    return { vb, ib, count: geo.indices.length };
  }

  /** A drawable mesh instance with its own uniforms. */
  createObject(mesh, { color = [1, 1, 1], metal = 0, rough = 0.5, opacity = 1, kind = 0 } = {}) {
    const d = this.device;
    const buf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new Float32Array(24);
    mat4.identity(data.subarray(0, 16));
    data.set([...color, 1], 16);
    data.set([metal, rough, opacity, kind], 20);
    const obj = {
      mesh,
      buf,
      data,
      visible: true,
      bg: d.createBindGroup({ layout: this.bglObj, entries: [{ binding: 0, resource: { buffer: buf } }] }),
      setModel: (m) => {
        data.set(m, 0);
        d.queue.writeBuffer(buf, 0, data);
      },
    };
    d.queue.writeBuffer(buf, 0, data);
    return obj;
  }

  /** Instanced spheres; data layout per instance: x, y, z, radius, r, g, b, _. */
  createInstances(mesh, cap, { metal = 0, rough = 0.55, unlit = false } = {}) {
    const d = this.device;
    const ubuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const u = new Float32Array(24);
    mat4.identity(u.subarray(0, 16));
    u.set([1, 1, 1, 1], 16);
    u.set([metal, rough, 1, unlit ? 1 : 0], 20);
    d.queue.writeBuffer(ubuf, 0, u);
    const sbuf = d.createBuffer({ size: cap * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    return {
      mesh,
      cap,
      count: 0,
      visible: true,
      data: new Float32Array(cap * 8),
      sbuf,
      bg: d.createBindGroup({
        layout: this.bglInst,
        entries: [
          { binding: 0, resource: { buffer: ubuf } },
          { binding: 1, resource: { buffer: sbuf } },
        ],
      }),
      upload(n) {
        this.count = Math.min(n, cap);
        if (this.count) d.queue.writeBuffer(sbuf, 0, this.data, 0, this.count * 8);
      },
    };
  }

  /**
   * GPU side of an SdfGrid: r8unorm distance texture + rgba8unorm colour texture. With
   * `external`, the textures are written elsewhere (streamed patches, or `textures` = [distance, colour]
   * filled by a compute pass) and sync() does nothing.
   */
  createVolume(grid, boxMesh, { external = false, textures = null } = {}) {
    const d = this.device;
    const size = [grid.nx, grid.ny, grid.nz];
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    const sdf = textures ? textures[0] : d.createTexture({ size, dimension: '3d', format: 'r8unorm', usage });
    const col = textures ? textures[1] : d.createTexture({ size, dimension: '3d', format: 'rgba8unorm', usage });
    const buf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new Float32Array(40);
    data.set([grid.sx, grid.sy, grid.sz, grid.band, grid.cell, 0, 0, 0], 32);
    const vol = {
      grid,
      sdfTex: sdf,
      colTex: col,
      mesh: boxMesh,
      buf,
      data,
      visible: true,
      bg: d.createBindGroup({
        layout: this.bglVol,
        entries: [
          { binding: 0, resource: { buffer: buf } },
          { binding: 1, resource: sdf.createView({ dimension: '3d' }) },
          { binding: 2, resource: col.createView({ dimension: '3d' }) },
          { binding: 3, resource: this.linearSampler },
        ],
      }),
      /** model = transform of the grid's local (origin-at-corner) frame */
      setModel: (m) => {
        data.set(m, 0);
        mat4.invert(m, data.subarray(16, 32));
        d.queue.writeBuffer(buf, 0, data);
      },
      sync: () => {
        if (external) return;
        if (grid.dirty) {
          grid.pack();
          d.queue.writeTexture({ texture: sdf }, grid.bytes, { bytesPerRow: grid.nx, rowsPerImage: grid.ny }, size);
          grid.dirty = false;
        }
        if (grid.colorDirty) {
          d.queue.writeTexture({ texture: col }, grid.color, { bytesPerRow: grid.nx * 4, rowsPerImage: grid.ny }, size);
          grid.colorDirty = false;
        }
      },
    };
    if (!external) {
      grid.dirty = true;
      grid.colorDirty = true;
      vol.sync();
    }
    vol.setModel(mat4.translation(grid.ox, grid.oy, grid.oz));
    return vol;
  }

  // ---------------------------------------------------------------- frame
  resize() {
    if (this._noResize) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (w === this.width && h === this.height) return;
    this.width = this.canvas.width = w;
    this.height = this.canvas.height = h;
    this.msaaTex?.destroy();
    this.depthTex?.destroy();
    this.msaaTex = this.device.createTexture({ size: [w, h], sampleCount: MSAA, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTex = this.device.createTexture({ size: [w, h], sampleCount: MSAA, format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  _writeFrame(buf, viewProj, lightVP, camPos, ortho, lightDir, rayDir, debug = 0) {
    const f = this.frameData;
    f.set(viewProj, 0);
    f.set(lightVP, 16);
    f.set([camPos.x, camPos.y, camPos.z, ortho ? 1 : 0], 32);
    f.set([lightDir.x, lightDir.y, lightDir.z, debug], 36);
    f.set([rayDir.x, rayDir.y, rayDir.z, 0], 40);
    this.device.queue.writeBuffer(buf, 0, f, 0, 44);
  }

  /**
   * scene = { objects: [..opaque], ghosts: [..], instances: [..], overlays: [..], volumes: [..],
   *           fluids: [..createFluidSurface()], shadowCasters: { objects, instances, volumes, fluids },
   *           debug: 0 shaded | 1 normals | 2 albedo | 3 march steps, shadows: bool }
   */
  render(scene, camera, light) {
    this.resize();
    const aspect = this.width / this.height;
    const viewProj = mat4.multiply(camera.projection(aspect), camera.view());
    // directional light: orthographic box around the table area
    const lc = light.center;
    const eye = new Vector3().copy(lc).addScaledVector(light.dir, 1.0);
    const lightView = mat4.lookAt(eye, lc, new Vector3(0, 1, 0));
    const e = light.extent;
    const lightVP = mat4.multiply(mat4.ortho(-e, e, -e * 0.7, e * 0.7, 0.2, 2.0), lightView);
    const rayDir = new Vector3().copy(light.dir).multiplyScalar(-1);
    this._writeFrame(this.frameBuf, viewProj, lightVP, camera.eye(), false, light.dir, rayDir, scene.debug ?? 0);
    this._writeFrame(this.shadowFrameBuf, lightVP, lightVP, eye, true, light.dir, rayDir);

    const enc = this.device.createCommandEncoder();
    const fluids = (scene.fluids ?? []).filter((f) => f.vol.visible && f.box);
    for (const f of fluids) f.encode(enc);
    // shadow pass
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: { view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      pass.setBindGroup(0, this.shadowFrameBG);
      const sc = scene.shadowCasters;
      if (scene.shadows !== false) {
        this._drawObjects(pass, this.pipes.meshShadow, sc.objects);
        this._drawInstances(pass, this.pipes.instShadow, sc.instances);
        this._drawVolumes(pass, this.pipes.volumeShadow, sc.volumes);
        this._drawVolumes(pass, this.pipes.fluidShadow, fluids.filter((f) => (sc.fluids ?? []).includes(f)).map((f) => f.vol));
      }
      pass.end();
    }
    // camera pass
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [
          {
            view: this.msaaTex.createView(),
            resolveTarget: this.context.getCurrentTexture().createView(),
            clearValue: this.clearColor,
            loadOp: 'clear',
            storeOp: 'discard',
          },
        ],
        depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      pass.setBindGroup(0, this.frameBG);
      this._drawObjects(pass, this.pipes.mesh, scene.objects);
      this._drawVolumes(pass, this.pipes.volume, scene.volumes);
      this._drawVolumes(pass, this.pipes.fluid, fluids.map((f) => f.vol));
      this._drawInstances(pass, this.pipes.inst, scene.instances);
      this._drawObjects(pass, this.pipes.ghost, scene.ghosts);
      this._drawInstances(pass, this.pipes.overlay, scene.overlays);
      pass.end();
    }
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Render one frame into an offscreen texture and read it back as RGBA8 pixels
   * (for headless tests and as a virtual camera for the digital twin).
   */
  async capture(scene, camera, light, width = this.width, height = this.height) {
    const d = this.device;
    const tex = d.createTexture({ size: [width, height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const saved = { w: this.width, h: this.height, msaa: this.msaaTex, depth: this.depthTex, ctx: this.context };
    this.msaaTex = d.createTexture({ size: [width, height], sampleCount: MSAA, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTex = d.createTexture({ size: [width, height], sampleCount: MSAA, format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.width = width;
    this.height = height;
    this.context = { getCurrentTexture: () => tex };
    this._noResize = true;
    try {
      this.render(scene, camera, light);
    } finally {
      this.msaaTex.destroy();
      this.depthTex.destroy();
      Object.assign(this, { width: saved.w, height: saved.h, msaaTex: saved.msaa, depthTex: saved.depth, context: saved.ctx, _noResize: false });
    }
    const bpr = Math.ceil((width * 4) / 256) * 256;
    const buf = d.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [width, height]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8ClampedArray(width * height * 4);
    const bgra = this.format.startsWith('bgra');
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const s = y * bpr + x * 4;
        const o = (y * width + x) * 4;
        out[o] = src[s + (bgra ? 2 : 0)];
        out[o + 1] = src[s + 1];
        out[o + 2] = src[s + (bgra ? 0 : 2)];
        out[o + 3] = 255;
      }
    buf.unmap();
    buf.destroy();
    tex.destroy();
    return { width, height, pixels: out };
  }

  _drawObjects(pass, pipe, list) {
    pass.setPipeline(pipe);
    for (const o of list) {
      if (!o.visible) continue;
      pass.setBindGroup(1, o.bg);
      pass.setVertexBuffer(0, o.mesh.vb);
      pass.setIndexBuffer(o.mesh.ib, 'uint32');
      pass.drawIndexed(o.mesh.count);
    }
  }

  _drawInstances(pass, pipe, list) {
    pass.setPipeline(pipe);
    for (const o of list) {
      if (!o.visible || (!o.count && !o.indirect)) continue;
      pass.setBindGroup(1, o.bg);
      pass.setVertexBuffer(0, o.mesh.vb);
      pass.setIndexBuffer(o.mesh.ib, 'uint32');
      // GPU-written instances (src/gpu/sim) draw as many as the simulation says
      if (o.indirect) pass.drawIndexedIndirect(o.indirect.buffer, o.indirect.offset);
      else pass.drawIndexed(o.mesh.count, o.count);
    }
  }

  _drawVolumes(pass, pipe, list) {
    pass.setPipeline(pipe);
    for (const v of list) {
      if (!v.visible) continue;
      pass.setBindGroup(1, v.bg);
      pass.setVertexBuffer(0, v.mesh.vb);
      pass.setIndexBuffer(v.mesh.ib, 'uint32');
      pass.drawIndexed(v.mesh.count);
    }
  }
}
