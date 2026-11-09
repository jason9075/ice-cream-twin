//! The renderer: lit meshes (table, bowls, spoon, SO-101), the SDF volumes sphere-traced in their proxy
//! boxes, the loose particles as one surface (fluid.rs) or as balls, crumbs, a shadow map, 4x MSAA.
//!
//! It draws whatever it was last given. The ice comes either straight from the GPU simulation's buffers
//! (`copy_from_gpu`: a few buffer -> texture copies on the same device, nothing goes through the CPU) or
//! from the CPU simulation (`upload_grid`, `upload_particles`).

use crate::camera::OrbitCamera;
use crate::fluid::FluidSurface;
use crate::geometry::{self, MeshData};
use crate::shaders;
use glam::{Mat4, Quat, Vec3};
use icecream_core::sdf::{bowl_profile, bowls, SPOON};
use std::sync::Arc;

pub const MSAA: u32 = 4;
const SHADOW_SIZE: u32 = 2048;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const SHADOW: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CLEAR: wgpu::Color = wgpu::Color { r: 0.83, g: 0.8, b: 0.76, a: 1.0 };

/// One SDF grid: origin (m), cell (m), voxels, band (m).
#[derive(Clone, Copy, Debug)]
pub struct GridGeo {
    pub origin: [f64; 3],
    pub cell: f64,
    pub dims: [usize; 3],
    pub band: f64,
}

/// SDF grids: the three bowls, the scoop (spoon-local), the table.
pub const GRIDS: usize = 5;

pub struct SceneDesc {
    /// the three bowls, the scoop (spoon-local), the table
    pub grids: [GridGeo; GRIDS],
    pub particle_cap: u32,
    pub crumb_cap: u32,
    pub hide_jaw: bool,
}

/// What moves between frames.
#[derive(Clone, Debug, Default)]
pub struct FrameState {
    /// position xyz, quaternion xyzw
    pub spoon: [f32; 7],
    /// where the spoon is heading (the arm's goal, or the raw target)
    pub ghost: [f32; 7],
    /// world matrices of the 8 SO-101 links (protocol order), row-major
    pub links: [[f64; 16]; 8],
    pub arm_on: bool,
    pub particles: u32,
    pub crumbs: u32,
    pub particle_radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ParticleMode {
    Surface,
    Balls,
    Both,
    Hidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Shading {
    Shaded = 0,
    Normals = 1,
    Albedo = 2,
    MarchSteps = 3,
    Voxels = 4,
}

/// The View panel: drawing only, the simulation never sees it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ViewOptions {
    pub particles: ParticleMode,
    /// × particle radius
    pub surface_radius: f32,
    pub surface_kernel: f32,
    pub surface_voxel: f32,
    pub surface_smooth: f32,
    pub tint_fluid: bool,
    pub crumbs: bool,
    pub bowl_ice: bool,
    pub spoon_ice: bool,
    pub bowls: bool,
    pub spoon: bool,
    pub arm: bool,
    pub table: bool,
    pub ghost: bool,
    pub shadows: bool,
    pub shading: Shading,
    pub grid_boxes: bool,
    /// the ice cream's material: matte frozen foam (true) or the older glossy look
    pub ice_matte: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        ViewOptions {
            particles: ParticleMode::Surface,
            surface_radius: 1.3,
            surface_kernel: 3.0,
            surface_voxel: 0.8,
            surface_smooth: 1.6,
            tint_fluid: false,
            crumbs: true,
            bowl_ice: true,
            spoon_ice: true,
            bowls: true,
            spoon: true,
            arm: true,
            table: true,
            ghost: true,
            shadows: true,
            shading: Shading::Shaded,
            grid_boxes: false,
            ice_matte: true,
        }
    }
}

pub struct Light {
    pub dir: Vec3,
    pub center: Vec3,
    pub extent: f32,
}

impl Default for Light {
    fn default() -> Self {
        Light { dir: Vec3::new(0.45, 1.0, 0.35).normalize(), center: Vec3::new(0.0, 0.08, -0.08), extent: 0.5 }
    }
}

struct Mesh {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    count: u32,
}

struct Object {
    mesh: usize,
    buf: wgpu::Buffer,
    data: [f32; 24],
    bg: wgpu::BindGroup,
    visible: bool,
}

struct Volume {
    sdf: wgpu::Texture,
    col: wgpu::Texture,
    mesh: usize,
    buf: wgpu::Buffer,
    data: [f32; 40],
    bg: wgpu::BindGroup,
    visible: bool,
    dims: [u32; 3],
    origin: Vec3,
}

struct Instances {
    buf: wgpu::Buffer,
    data: [f32; 24],
    bg: wgpu::BindGroup,
    count: u32,
    visible: bool,
}

struct Pipes {
    mesh: wgpu::RenderPipeline,
    ghost: wgpu::RenderPipeline,
    inst: wgpu::RenderPipeline,
    volume: wgpu::RenderPipeline,
    fluid: wgpu::RenderPipeline,
    mesh_shadow: wgpu::RenderPipeline,
    inst_shadow: wgpu::RenderPipeline,
    volume_shadow: wgpu::RenderPipeline,
    fluid_shadow: wgpu::RenderPipeline,
}

/// Indices into `objects` of the scene's parts.
struct Parts {
    table: usize,
    bowls: [usize; 3],
    spoon: usize,
    ghost: usize,
    arm: Vec<(usize, usize, bool)>, // (object, link slot, is the jaw)
    boxes: [usize; GRIDS],
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub format: wgpu::TextureFormat,
    pipes: Pipes,
    bgl_obj: wgpu::BindGroupLayout,
    bgl_vol: wgpu::BindGroupLayout,
    frame_buf: wgpu::Buffer,
    shadow_frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    shadow_frame_bg: wgpu::BindGroup,
    shadow_tex: wgpu::Texture,
    linear: wgpu::Sampler,
    meshes: Vec<Mesh>,
    objects: Vec<Object>,
    volumes: Vec<Volume>,
    particles: Instances,
    crumbs: Instances,
    /// particles (from 0) and crumbs (from particle_cap * 16), (f32 x, y, z, u32 rgba) each
    pub particle_buf: wgpu::Buffer,
    particle_cap: u32,
    crumb_cap: u32,
    fluid: FluidSurface,
    parts: Parts,
    hide_jaw: bool,
    targets: Option<(u32, u32, wgpu::Texture, wgpu::Texture)>,
    frame: FrameState,
    sphere_mesh: usize,
}

fn uniform_buf(d: &wgpu::Device, label: &str) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 256, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

fn make_instances(d: &wgpu::Device, layout: &wgpu::BindGroupLayout, particle_buf: &wgpu::Buffer, offset: u64, cap: u32) -> Instances {
    let buf = uniform_buf(d, "instances");
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: particle_buf, offset, size: wgpu::BufferSize::new(cap as u64 * 16) }),
            },
        ],
    });
    let mut data = [0f32; 24];
    data[..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
    data[16..20].copy_from_slice(&[1.0; 4]);
    data[20..24].copy_from_slice(&[0.0, 0.55, 0.0015, 0.0]);
    Instances { buf, data, bg, count: 0, visible: true }
}


/// Column-major model matrix from a pose (position xyz, quaternion xyzw).
pub fn pose_matrix(p: &[f32; 7]) -> Mat4 {
    Mat4::from_rotation_translation(Quat::from_xyzw(p[3], p[4], p[5], p[6]), Vec3::new(p[0], p[1], p[2]))
}

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, format: wgpu::TextureFormat, desc: &SceneDesc) -> Self {
        let d = &*device;
        let (vf, ff) = (wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::FRAGMENT);
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None };
        let ubuf = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let sbuf = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let tex3 = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
        let bgl = |label, entries: &[wgpu::BindGroupLayoutEntry]| d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let bgl_frame = bgl(
            "frame",
            &[
                entry(0, vf | ff, ubuf),
                entry(1, ff, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                entry(2, ff, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)),
            ],
        );
        let bgl_obj = bgl("object", &[entry(0, vf | ff, ubuf)]);
        let bgl_inst = bgl("instances", &[entry(0, vf | ff, ubuf), entry(1, vf, sbuf)]);
        let samp = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let bgl_vol = bgl("volume", &[entry(0, vf | ff, ubuf), entry(1, ff, tex3), entry(2, ff, tex3), entry(3, ff, samp)]);
        let bgl_fluid = bgl("fluid surface", &[entry(0, vf | ff, sbuf), entry(1, ff, tex3), entry(2, ff, tex3), entry(3, ff, samp)]);

        let frame_buf = uniform_buf(d, "frame");
        let shadow_frame_buf = uniform_buf(d, "shadow frame");
        let tex = |label, size: [u32; 3], format, usage| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] },
                mip_level_count: 1,
                sample_count: 1,
                dimension: if size[2] > 1 { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                format,
                usage,
                view_formats: &[],
            })
        };
        let shadow_tex = tex("shadow map", [SHADOW_SIZE, SHADOW_SIZE, 1], SHADOW, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let dummy = tex("no shadow", [1, 1, 1], SHADOW, wgpu::TextureUsages::TEXTURE_BINDING);
        let cmp = d.create_sampler(&wgpu::SamplerDescriptor { compare: Some(wgpu::CompareFunction::Less), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let linear = d.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let frame_bg_of = |buf: &wgpu::Buffer, t: &wgpu::Texture| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl_frame,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&t.create_view(&Default::default())) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&cmp) },
                ],
            })
        };
        let frame_bg = frame_bg_of(&frame_buf, &shadow_tex);
        let shadow_frame_bg = frame_bg_of(&shadow_frame_buf, &dummy);

        // ---------------------------------------------------------------- pipelines
        let module = |label, code: String| d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
        let mesh_mod = module("mesh", shaders::mesh());
        let inst_mod = module("instances", shaders::inst());
        let vol_mod = module("volume", shaders::volume(false));
        let fluid_mod = module("fluid surface", shaders::volume(true));
        let layout = |g1: &wgpu::BindGroupLayout| d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl_frame, g1], push_constant_ranges: &[] });
        let vlayout_full = [wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            ],
        }];
        let vlayout_pos = [wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }],
        }];
        let opaque = [Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let blended = [Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })];
        struct P<'a> {
            label: &'a str,
            layout: &'a wgpu::PipelineLayout,
            module: &'a wgpu::ShaderModule,
            buffers: &'a [wgpu::VertexBufferLayout<'a>],
            fs: Option<&'a str>,
            targets: &'a [Option<wgpu::ColorTargetState>],
            cull: Option<wgpu::Face>,
            shadow: bool,
            depth_write: bool,
            always: bool,
        }
        let pipe = |p: P| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(p.label),
                layout: Some(p.layout),
                vertex: wgpu::VertexState { module: p.module, entry_point: "vs", compilation_options: Default::default(), buffers: p.buffers },
                fragment: p.fs.map(|fs| wgpu::FragmentState { module: p.module, entry_point: fs, compilation_options: Default::default(), targets: p.targets }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: p.cull, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: if p.shadow { SHADOW } else { DEPTH },
                    depth_write_enabled: p.depth_write,
                    depth_compare: if p.always { wgpu::CompareFunction::Always } else { wgpu::CompareFunction::Less },
                    stencil: Default::default(),
                    bias: if p.shadow { wgpu::DepthBiasState { constant: 2, slope_scale: 1.5, clamp: 0.0 } } else { Default::default() },
                }),
                multisample: wgpu::MultisampleState { count: if p.shadow { 1 } else { MSAA }, ..Default::default() },
                multiview: None,
                cache: None,
            })
        };
        let (l_obj, l_inst, l_vol, l_fluid) = (layout(&bgl_obj), layout(&bgl_inst), layout(&bgl_vol), layout(&bgl_fluid));
        let base = |label, layout, module, buffers, fs, targets| P { label, layout, module, buffers, fs, targets, cull: None, shadow: false, depth_write: true, always: false };
        let pipes = Pipes {
            mesh: pipe(base("mesh", &l_obj, &mesh_mod, &vlayout_full, Some("fs"), &opaque)),
            ghost: pipe(P { depth_write: false, ..base("ghost", &l_obj, &mesh_mod, &vlayout_full, Some("fs"), &blended) }),
            inst: pipe(base("instances", &l_inst, &inst_mod, &vlayout_full, Some("fs"), &opaque)),
            volume: pipe(P { cull: Some(wgpu::Face::Front), ..base("volume", &l_vol, &vol_mod, &vlayout_pos, Some("fs"), &opaque) }),
            fluid: pipe(P { cull: Some(wgpu::Face::Front), ..base("fluid surface", &l_fluid, &fluid_mod, &vlayout_pos, Some("fs"), &opaque) }),
            mesh_shadow: pipe(P { shadow: true, ..base("mesh shadow", &l_obj, &mesh_mod, &vlayout_full, None, &[]) }),
            inst_shadow: pipe(P { shadow: true, ..base("instances shadow", &l_inst, &inst_mod, &vlayout_full, None, &[]) }),
            volume_shadow: pipe(P { shadow: true, cull: Some(wgpu::Face::Front), ..base("volume shadow", &l_vol, &vol_mod, &vlayout_pos, Some("fsShadow"), &[]) }),
            fluid_shadow: pipe(P { shadow: true, cull: Some(wgpu::Face::Front), ..base("fluid shadow", &l_fluid, &fluid_mod, &vlayout_pos, Some("fsShadow"), &[]) }),
        };

        let particle_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles + crumbs"),
            size: (desc.particle_cap as u64 + desc.crumb_cap as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fluid = FluidSurface::new(d, &particle_buf, &bgl_fluid, &linear);
        let particles = make_instances(d, &bgl_inst, &particle_buf, 0, desc.particle_cap);
        let crumbs = make_instances(d, &bgl_inst, &particle_buf, desc.particle_cap as u64 * 16, desc.crumb_cap);

        let mut r = Renderer {
            device: device.clone(),
            queue,
            format,
            pipes,
            bgl_obj,
            bgl_vol,
            frame_buf,
            shadow_frame_buf,
            frame_bg,
            shadow_frame_bg,
            shadow_tex,
            linear,
            meshes: Vec::new(),
            objects: Vec::new(),
            volumes: Vec::new(),
            particles,
            crumbs,
            particle_buf,
            particle_cap: desc.particle_cap,
            crumb_cap: desc.crumb_cap,
            fluid,
            parts: Parts { table: 0, bowls: [0; 3], spoon: 0, ghost: 0, arm: Vec::new(), boxes: [0; GRIDS] },
            hide_jaw: desc.hide_jaw,
            targets: None,
            frame: FrameState::default(),
            sphere_mesh: 0,
        };
        r.build_scene(desc);
        r
    }

    fn mesh(&mut self, m: &MeshData) -> usize {
        use wgpu::util::DeviceExt;
        let d = &self.device;
        let vb = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&m.vertices), usage: wgpu::BufferUsages::VERTEX });
        let ib = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&m.indices), usage: wgpu::BufferUsages::INDEX });
        self.meshes.push(Mesh { vb, ib, count: m.indices.len() as u32 });
        self.meshes.len() - 1
    }

    /// color, metal, rough, opacity, kind (1 = wood)
    fn object(&mut self, mesh: usize, color: [f32; 3], metal: f32, rough: f32, opacity: f32, kind: f32) -> usize {
        let buf = uniform_buf(&self.device, "object");
        let mut data = [0f32; 24];
        data[..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
        data[16..20].copy_from_slice(&[color[0], color[1], color[2], 1.0]);
        data[20..24].copy_from_slice(&[metal, rough, opacity, kind]);
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.bgl_obj, entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
        self.objects.push(Object { mesh, buf, data, bg, visible: true });
        self.objects.len() - 1
    }

    fn set_model(&mut self, o: usize, m: Mat4) {
        let ob = &mut self.objects[o];
        ob.data[..16].copy_from_slice(&m.to_cols_array());
        self.queue.write_buffer(&ob.buf, 0, bytemuck::cast_slice(&ob.data));
    }

    fn volume(&mut self, g: &GridGeo, mesh: usize) -> usize {
        let d = &self.device;
        let size = wgpu::Extent3d { width: g.dims[0] as u32, height: g.dims[1] as u32, depth_or_array_layers: g.dims[2] as u32 };
        let tex = |format| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some("sdf volume"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (sdf, col) = (tex(wgpu::TextureFormat::R8Unorm), tex(wgpu::TextureFormat::Rgba8Unorm));
        let buf = uniform_buf(d, "volume");
        let mut data = [0f32; 40];
        let s = [g.dims[0] as f64 * g.cell, g.dims[1] as f64 * g.cell, g.dims[2] as f64 * g.cell];
        data[32..40].copy_from_slice(&[s[0] as f32, s[1] as f32, s[2] as f32, g.band as f32, g.cell as f32, 0.0, 0.0, 0.0]);
        let v3 = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D3), ..Default::default() });
        let (sdf_view, col_view) = (v3(&sdf), v3(&col));
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_vol,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sdf_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&col_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.linear) },
            ],
        });
        let origin = Vec3::new(g.origin[0] as f32, g.origin[1] as f32, g.origin[2] as f32);
        self.volumes.push(Volume { sdf, col, mesh, buf, data, bg, visible: true, dims: [g.dims[0] as u32, g.dims[1] as u32, g.dims[2] as u32], origin });
        let k = self.volumes.len() - 1;
        self.set_volume_model(k, Mat4::from_translation(origin));
        k
    }

    fn set_volume_model(&mut self, k: usize, m: Mat4) {
        let v = &mut self.volumes[k];
        v.data[..16].copy_from_slice(&m.to_cols_array());
        v.data[16..32].copy_from_slice(&m.inverse().to_cols_array());
        self.queue.write_buffer(&v.buf, 0, bytemuck::cast_slice(&v.data));
    }

    fn build_scene(&mut self, desc: &SceneDesc) {
        let table_mesh = self.mesh(&geometry::cuboid([-0.7, -0.04, -0.45], [0.7, 0.0, 0.45]));
        self.parts.table = self.object(table_mesh, [1.0; 3], 0.0, 0.6, 1.0, 1.0);
        let bowl_mesh = self.mesh(&geometry::lathe(&bowl_profile(&bowls()[0], 48), 96));
        let tints = [[0.96, 0.93, 0.88], [0.97, 0.9, 0.91], [0.88, 0.95, 0.91]];
        for (k, b) in bowls().iter().enumerate() {
            let o = self.object(bowl_mesh, tints[k], 0.0, 0.12, 1.0, 0.0);
            self.set_model(o, Mat4::from_translation(Vec3::new(b.cx as f32, 0.0, b.cz as f32)));
            self.parts.bowls[k] = o;
        }
        let spoon_mesh = self.mesh(&geometry::spoon(&SPOON));
        self.parts.spoon = self.object(spoon_mesh, [0.86, 0.88, 0.91], 1.0, 0.18, 1.0, 0.0);
        self.parts.ghost = self.object(spoon_mesh, [0.16, 0.6, 0.96], 0.0, 0.5, 0.25, 0.0);

        // SO-101: one object per (link, material), from the decimated meshes in assets/so101
        let looks = |m: &str| if m == "sts3215" { ([0.07, 0.07, 0.08], 0.35) } else { ([0.96, 0.8, 0.22], 0.55) };
        for part in crate::arm::parts() {
            let mesh = self.mesh(&part.mesh);
            let (c, rough) = looks(&part.material);
            let o = self.object(mesh, c, 0.0, rough, 1.0, 0.0);
            let slot = icecream_core::robot::so101::LINKS.iter().position(|l| *l == part.link).unwrap_or(usize::MAX);
            self.parts.arm.push((o, slot, slot == 6));
        }

        // the SDF volumes: three bowls (one shared world box), the scoop
        let g0 = desc.grids[0];
        let s0 = [(g0.dims[0] as f64 * g0.cell) as f32, (g0.dims[1] as f64 * g0.cell) as f32, (g0.dims[2] as f64 * g0.cell) as f32];
        let world_box = self.mesh(&geometry::cuboid([0.0; 3], s0));
        for g in &desc.grids[..3] {
            self.volume(g, world_box);
        }
        let c = desc.grids[3];
        let sc = [(c.dims[0] as f64 * c.cell) as f32, (c.dims[1] as f64 * c.cell) as f32, (c.dims[2] as f64 * c.cell) as f32];
        let content_box = self.mesh(&geometry::cuboid([0.0; 3], sc));
        self.volume(&c, content_box);
        let t = desc.grids[4];
        let st = [(t.dims[0] as f64 * t.cell) as f32, (t.dims[1] as f64 * t.cell) as f32, (t.dims[2] as f64 * t.cell) as f32];
        let table_box = self.mesh(&geometry::cuboid([0.0; 3], st));
        self.volume(&t, table_box);

        // translucent boxes showing what each SDF grid covers (debug)
        let unit = self.mesh(&geometry::cuboid([0.0; 3], [1.0; 3]));
        let colors = [[1.0, 0.6, 0.2], [1.0, 0.6, 0.2], [1.0, 0.6, 0.2], [0.3, 0.9, 0.3], [0.7, 0.4, 1.0]];
        for (k, col) in colors.iter().enumerate() {
            let o = self.object(unit, *col, 0.0, 1.0, 0.15, 0.0);
            let g = desc.grids[k];
            let size = Vec3::new((g.dims[0] as f64 * g.cell) as f32, (g.dims[1] as f64 * g.cell) as f32, (g.dims[2] as f64 * g.cell) as f32);
            self.set_model(o, Mat4::from_translation(self.volumes[k].origin) * Mat4::from_scale(size));
            self.parts.boxes[k] = o;
        }
        self.sphere_mesh = self.mesh(&geometry::sphere(8, 12));
    }


    // ------------------------------------------------------------------ ice data in
    /// A whole grid from the CPU: r8 distances and rgba8 colours, x fastest.
    pub fn upload_grid(&self, k: usize, dist: Option<&[u8]>, color: Option<&[u8]>) {
        let v = &self.volumes[k];
        let [nx, ny, nz] = v.dims;
        let size = wgpu::Extent3d { width: nx, height: ny, depth_or_array_layers: nz };
        let write = |tex: &wgpu::Texture, data: &[u8], bpp: u32| {
            self.queue.write_texture(
                wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(nx * bpp), rows_per_image: Some(ny) },
                size,
            );
        };
        if let Some(d) = dist {
            write(&v.sdf, d, 1);
        }
        if let Some(c) = color {
            write(&v.col, c, 4);
        }
    }

    /// Particles and (when given) crumbs from the CPU, n x (f32 x, y, z, u32 rgba).
    pub fn upload_particles(&self, particles: &[u8], crumbs: Option<&[u8]>) {
        let n = (particles.len() as u64).min(self.particle_cap as u64 * 16);
        if n > 0 {
            self.queue.write_buffer(&self.particle_buf, 0, &particles[..n as usize]);
        }
        if let Some(c) = crumbs {
            let m = (c.len() as u64).min(self.crumb_cap as u64 * 16);
            if m > 0 {
                self.queue.write_buffer(&self.particle_buf, self.particle_cap as u64 * 16, &c[..m as usize]);
            }
        }
    }

    /// The GPU simulation's packed grids and particles, copied on the device. `pk`: per grid, `rows[k]` =
    /// (byte offset, distance row bytes, colour row bytes): `ny * nz` distance rows then as many colour
    /// rows, each a multiple of 256 bytes (what copy_buffer_to_texture wants); `grids` says which were
    /// packed. `pp`: particles from 0, crumbs from
    /// `crumb_at` bytes, 16 bytes each.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_from_gpu(&self, enc: &mut wgpu::CommandEncoder, pk: &wgpu::Buffer, rows: &[[u64; 3]], grids: [bool; GRIDS], pp: &wgpu::Buffer, n: u32, crumb_at: u64, crumbs: u32) {
        for (k, v) in self.volumes.iter().enumerate() {
            if !grids[k] {
                continue;
            }
            let [nx, ny, nz] = v.dims;
            let n_rows = (ny * nz) as u64;
            let [off, db, cb] = rows[k];
            let size = wgpu::Extent3d { width: nx, height: ny, depth_or_array_layers: nz };
            for (tex, at, bpr) in [(&v.sdf, off, db), (&v.col, off + n_rows * db, cb)] {
                enc.copy_buffer_to_texture(
                    wgpu::ImageCopyBuffer { buffer: pk, layout: wgpu::ImageDataLayout { offset: at, bytes_per_row: Some(bpr as u32), rows_per_image: Some(ny) } },
                    wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    size,
                );
            }
        }
        let n = n.min(self.particle_cap) as u64;
        if n > 0 {
            enc.copy_buffer_to_buffer(pp, 0, &self.particle_buf, 0, n * 16);
        }
        let m = crumbs.min(self.crumb_cap) as u64;
        if m > 0 {
            enc.copy_buffer_to_buffer(pp, crumb_at, &self.particle_buf, self.particle_cap as u64 * 16, m * 16);
        }
    }

    /// Poses and counts for the next draw.
    pub fn set_frame(&mut self, f: &FrameState) {
        self.frame = f.clone();
        let spoon = pose_matrix(&f.spoon);
        self.set_model(self.parts.spoon, spoon);
        self.set_model(self.parts.ghost, pose_matrix(&f.ghost));
        let c = self.volumes[3].origin;
        self.set_volume_model(3, spoon * Mat4::from_translation(c));
        let content_size = Vec3::new(self.volumes[3].data[32], self.volumes[3].data[33], self.volumes[3].data[34]);
        self.set_model(self.parts.boxes[3], spoon * Mat4::from_translation(c) * Mat4::from_scale(content_size));
        for k in 0..self.parts.arm.len() {
            let (o, slot, _) = self.parts.arm[k];
            if slot < 8 {
                let m = f.links[slot];
                // row-major -> column-major
                let cols: [f32; 16] = std::array::from_fn(|i| m[(i % 4) * 4 + i / 4] as f32);
                self.set_model(o, Mat4::from_cols_array(&cols));
            }
        }
        self.particles.count = f.particles.min(self.particle_cap);
        self.crumbs.count = f.crumbs.min(self.crumb_cap);
        for inst in [&mut self.particles, &mut self.crumbs] {
            inst.data[22] = f.particle_radius * 1.15;
            self.queue.write_buffer(&inst.buf, 0, bytemuck::cast_slice(&inst.data));
        }
    }

    fn apply_options(&mut self, o: &ViewOptions) {
        let f = &self.frame;
        self.objects[self.parts.table].visible = o.table;
        for k in self.parts.bowls {
            self.objects[k].visible = o.bowls;
        }
        self.objects[self.parts.spoon].visible = o.spoon;
        self.objects[self.parts.ghost].visible = o.ghost;
        for &(obj, slot, jaw) in &self.parts.arm {
            self.objects[obj].visible = o.arm && f.arm_on && slot < 8 && !(jaw && self.hide_jaw);
        }
        for k in self.parts.boxes {
            self.objects[k].visible = o.grid_boxes;
        }
        for k in [0, 1, 2, 4] {
            self.volumes[k].visible = o.bowl_ice;
        }
        self.volumes[3].visible = o.spoon_ice;
        self.particles.visible = matches!(o.particles, ParticleMode::Balls | ParticleMode::Both);
        self.crumbs.visible = o.crumbs;
    }

    // ------------------------------------------------------------------ drawing
    fn ensure_targets(&mut self, w: u32, h: u32) {
        if matches!(self.targets, Some((tw, th, _, _)) if tw == w && th == h) {
            return;
        }
        let mk = |format, label| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: MSAA,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        self.targets = Some((w, h, mk(self.format, "msaa colour"), mk(DEPTH, "msaa depth")));
    }

    #[allow(clippy::too_many_arguments)]
    fn write_frame(&self, buf: &wgpu::Buffer, view_proj: Mat4, light_vp: Mat4, cam: Vec3, ortho: bool, light: &Light, debug: f32, matte: bool) {
        let mut f = [0f32; 44];
        f[..16].copy_from_slice(&view_proj.to_cols_array());
        f[16..32].copy_from_slice(&light_vp.to_cols_array());
        f[32..36].copy_from_slice(&[cam.x, cam.y, cam.z, if ortho { 1.0 } else { 0.0 }]);
        f[36..40].copy_from_slice(&[light.dir.x, light.dir.y, light.dir.z, debug]);
        let rd = -light.dir;
        f[40..44].copy_from_slice(&[rd.x, rd.y, rd.z, if matte { 1.0 } else { 0.0 }]);
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&f));
    }

    /// Record one frame into `target` (w x h, the renderer's format, single-sampled), the scene inside
    /// `viewport` (x, y, width, height in pixels; None: all of it).
    #[allow(clippy::too_many_arguments)]
    pub fn render(&mut self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, w: u32, h: u32, viewport: Option<[f32; 4]>, camera: &OrbitCamera, light: &Light, opts: &ViewOptions) {
        self.ensure_targets(w, h);
        self.apply_options(opts);
        let vp = viewport.unwrap_or([0.0, 0.0, w as f32, h as f32]);
        let view_proj = camera.projection(vp[2] / vp[3].max(1.0)) * camera.view();
        let eye = light.center + light.dir * 1.0;
        let light_view = Mat4::look_at_rh(eye, light.center, Vec3::Y);
        let e = light.extent;
        let light_vp = Mat4::orthographic_rh(-e, e, -e * 0.7, e * 0.7, 0.2, 2.0) * light_view;
        self.write_frame(&self.frame_buf, view_proj, light_vp, camera.eye(), false, light, opts.shading as u32 as f32, opts.ice_matte);
        let shadow_mode = if opts.shading == Shading::Voxels { Shading::Voxels as u32 as f32 } else { 0.0 };
        self.write_frame(&self.shadow_frame_buf, light_vp, light_vp, eye, true, light, shadow_mode, false);

        // the particle surface (compute), when there is one to draw
        let surface = matches!(opts.particles, ParticleMode::Surface | ParticleMode::Both) && self.frame.particles > 0;
        if surface {
            let r = self.frame.particle_radius;
            self.fluid.encode(&self.queue, enc, self.frame.particles.min(self.particle_cap), r * opts.surface_radius, r * opts.surface_kernel, r * opts.surface_voxel, r * opts.surface_smooth, opts.tint_fluid);
        }

        // shadow map
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_tex.create_view(&Default::default()),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if opts.shadows {
                pass.set_bind_group(0, &self.shadow_frame_bg, &[]);
                let casters: Vec<usize> = self.parts.bowls.iter().copied().chain([self.parts.spoon]).chain(self.parts.arm.iter().map(|a| a.0)).collect();
                self.draw_objects(&mut pass, &self.pipes.mesh_shadow, &casters);
                self.draw_instances(&mut pass, &self.pipes.inst_shadow);
                self.draw_volumes(&mut pass, &self.pipes.volume_shadow);
                if surface {
                    self.draw_fluid(&mut pass, &self.pipes.fluid_shadow);
                }
            }
        }
        // camera
        let (_, _, msaa, depth) = self.targets.as_ref().unwrap();
        let (msaa_view, depth_view) = (msaa.create_view(&Default::default()), depth.create_view(&Default::default()));
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("camera"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_view,
                    resolve_target: Some(target),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(CLEAR), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(vp[0], vp[1], vp[2].max(1.0), vp[3].max(1.0), 0.0, 1.0);
            pass.set_bind_group(0, &self.frame_bg, &[]);
            let opaque: Vec<usize> = [self.parts.table].into_iter().chain(self.parts.bowls).chain([self.parts.spoon]).chain(self.parts.arm.iter().map(|a| a.0)).collect();
            self.draw_objects(&mut pass, &self.pipes.mesh, &opaque);
            self.draw_volumes(&mut pass, &self.pipes.volume);
            if surface {
                self.draw_fluid(&mut pass, &self.pipes.fluid);
            }
            self.draw_instances(&mut pass, &self.pipes.inst);
            let ghosts: Vec<usize> = [self.parts.ghost].into_iter().chain(self.parts.boxes).collect();
            self.draw_objects(&mut pass, &self.pipes.ghost, &ghosts);
        }
    }

    fn draw_objects<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, pipe: &'a wgpu::RenderPipeline, list: &[usize]) {
        pass.set_pipeline(pipe);
        for &k in list {
            let o = &self.objects[k];
            if !o.visible {
                continue;
            }
            let m = &self.meshes[o.mesh];
            pass.set_bind_group(1, &o.bg, &[]);
            pass.set_vertex_buffer(0, m.vb.slice(..));
            pass.set_index_buffer(m.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.count, 0, 0..1);
        }
    }

    fn draw_instances<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, pipe: &'a wgpu::RenderPipeline) {
        pass.set_pipeline(pipe);
        let m = &self.meshes[self.sphere_mesh];
        for inst in [&self.particles, &self.crumbs] {
            if !inst.visible || inst.count == 0 {
                continue;
            }
            pass.set_bind_group(1, &inst.bg, &[]);
            pass.set_vertex_buffer(0, m.vb.slice(..));
            pass.set_index_buffer(m.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.count, 0, 0..inst.count);
        }
    }

    fn draw_volumes<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, pipe: &'a wgpu::RenderPipeline) {
        pass.set_pipeline(pipe);
        for v in &self.volumes {
            if !v.visible {
                continue;
            }
            let m = &self.meshes[v.mesh];
            pass.set_bind_group(1, &v.bg, &[]);
            pass.set_vertex_buffer(0, m.vb.slice(..));
            pass.set_index_buffer(m.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.count, 0, 0..1);
        }
    }

    fn draw_fluid<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, pipe: &'a wgpu::RenderPipeline) {
        let m = &self.meshes[self.fluid_box()];
        pass.set_pipeline(pipe);
        pass.set_bind_group(1, &self.fluid.render_bg, &[]);
        pass.set_vertex_buffer(0, m.vb.slice(..));
        pass.set_index_buffer(m.ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..m.count, 0, 0..1);
    }

    /// the unit cube (the grid boxes' mesh) also bounds the particle surface, scaled in its shader
    fn fluid_box(&self) -> usize {
        self.objects[self.parts.boxes[0]].mesh
    }

    /// Render one frame off screen and read it back as RGBA8 (screenshots, a virtual camera).
    pub fn capture(&mut self, w: u32, h: u32, camera: &OrbitCamera, light: &Light, opts: &ViewOptions) -> Vec<u8> {
        let d = self.device.clone();
        let tex = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bpr = (w * 4).div_ceil(256) * 256;
        let buf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("capture read"), size: (bpr * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = d.create_command_encoder(&Default::default());
        self.render(&mut enc, &tex.create_view(&Default::default()), w, h, None, camera, light, opts);
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        d.poll(wgpu::Maintain::Wait);
        let src = slice.get_mapped_range();
        let bgra = matches!(self.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let mut out = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let s = y * bpr as usize + x * 4;
                let o = (y * w as usize + x) * 4;
                out[o] = src[s + if bgra { 2 } else { 0 }];
                out[o + 1] = src[s + 1];
                out[o + 2] = src[s + if bgra { 0 } else { 2 }];
                out[o + 3] = 255;
            }
        }
        out
    }
}
