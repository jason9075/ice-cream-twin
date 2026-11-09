//! Loose particles drawn as one smooth surface (wgsl/fluid_splat.wgsl), entirely on the GPU: the bounding
//! box, the grid that fits it, the dispatch sizes and the parameters the render pass reads are all worked
//! out by kernels, so nothing is read back.

const WG: u32 = 64;
const CHANNELS: u64 = 8;
pub const MAX_DIM: u32 = 128;
pub const MAX_VOXELS: u32 = 1_500_000;

pub struct FluidSurface {
    params: wgpu::Buffer,
    box_buf: wgpu::Buffer,
    /// setup writes the dispatch sizes here; copied to `indirect` (a buffer can't be storage and indirect
    /// within one dispatch)
    args: wgpu::Buffer,
    indirect: wgpu::Buffer,
    bg: wgpu::BindGroup,
    bounds: wgpu::ComputePipeline,
    setup: wgpu::ComputePipeline,
    splat: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    blurs: [wgpu::ComputePipeline; 3],
    /// group 1 of the surface's render pipelines (volume shader, FLUID)
    pub render_bg: wgpu::BindGroup,
}

impl FluidSurface {
    pub fn new(d: &wgpu::Device, particles: &wgpu::Buffer, render_layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler) -> Self {
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fluid splat"), source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/fluid_splat.wgsl").into()) });
        let c = wgpu::ShaderStages::COMPUTE;
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let st = |format| wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D3 };
        let e = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty, count: None };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid splat"),
            entries: &[
                e(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                e(1, storage(true)),
                e(2, storage(false)),
                e(3, st(wgpu::TextureFormat::Rgba16Float)),
                e(4, st(wgpu::TextureFormat::Rgba8Unorm)),
                e(5, storage(false)),
                e(6, storage(false)),
                e(7, storage(false)),
                e(8, storage(false)),
                e(9, storage(false)),
                e(10, storage(false)),
            ],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let pipe = |entry| d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&pl), module: &module, entry_point: entry, compilation_options: Default::default(), cache: None });
        let tex = |format| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some("particle surface"),
                size: wgpu::Extent3d { width: MAX_DIM, height: MAX_DIM, depth_or_array_layers: MAX_DIM },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let (phi, col) = (tex(wgpu::TextureFormat::Rgba16Float), tex(wgpu::TextureFormat::Rgba8Unorm));
        let buf = |label, size: u64, usage| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let sto = wgpu::BufferUsages::STORAGE;
        let params = buf("surface params", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let acc = buf("surface sums", MAX_VOXELS as u64 * CHANNELS * 4, sto);
        let phi_a = buf("surface distance A", MAX_VOXELS as u64 * 4, sto);
        let phi_b = buf("surface distance B", MAX_VOXELS as u64 * 4, sto);
        let box_buf = buf("surface box", 24, sto | wgpu::BufferUsages::COPY_DST);
        let grid = buf("surface grid", 32, sto);
        let args = buf("surface dispatch sizes", 24, sto | wgpu::BufferUsages::COPY_SRC);
        let indirect = buf("surface dispatch", 24, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST);
        let vol = buf("surface volume", 160, sto);
        let v3 = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D3), ..Default::default() });
        let (phi_view, col_view) = (v3(&phi), v3(&col));
        let b = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid splat"),
            layout: &layout,
            entries: &[
                b(0, params.as_entire_binding()),
                b(1, particles.as_entire_binding()),
                b(2, acc.as_entire_binding()),
                b(3, wgpu::BindingResource::TextureView(&phi_view)),
                b(4, wgpu::BindingResource::TextureView(&col_view)),
                b(5, phi_a.as_entire_binding()),
                b(6, phi_b.as_entire_binding()),
                b(7, box_buf.as_entire_binding()),
                b(8, grid.as_entire_binding()),
                b(9, args.as_entire_binding()),
                b(10, vol.as_entire_binding()),
            ],
        });
        let render_bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle surface"),
            layout: render_layout,
            entries: &[
                b(0, vol.as_entire_binding()),
                b(1, wgpu::BindingResource::TextureView(&phi_view)),
                b(2, wgpu::BindingResource::TextureView(&col_view)),
                b(3, wgpu::BindingResource::Sampler(sampler)),
            ],
        });
        FluidSurface {
            params,
            box_buf,
            args,
            indirect,
            bg,
            bounds: pipe("bounds"),
            setup: pipe("setup"),
            splat: pipe("splat"),
            resolve: pipe("resolve"),
            blurs: [pipe("blurX"), pipe("blurY"), pipe("blurZ")],
            render_bg,
        }
    }

    /// Record the surface of the first `n` particles. r: surface radius, big_r: blend kernel, voxel:
    /// preferred cell, smooth: blur radius (metres).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, q: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, n: u32, r: f32, big_r: f32, voxel: f32, smooth: f32, tint: bool) {
        let mut p = [0u8; 32];
        for (k, v) in [n, MAX_DIM, MAX_VOXELS, u32::from(tint)].iter().enumerate() {
            p[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (k, v) in [voxel, big_r, r, smooth].iter().enumerate() {
            p[16 + k * 4..20 + k * 4].copy_from_slice(&v.to_le_bytes());
        }
        q.write_buffer(&self.params, 0, &p);
        // empty box for the atomics: lo = max, hi = min (ordered u32)
        q.write_buffer(&self.box_buf, 0, bytemuck::cast_slice(&[u32::MAX, u32::MAX, u32::MAX, 0, 0, 0]));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("particle surface: grid, splat"), timestamp_writes: None });
            pass.set_bind_group(0, &self.bg, &[]);
            pass.set_pipeline(&self.bounds);
            pass.dispatch_workgroups(n.div_ceil(WG), 1, 1);
            pass.set_pipeline(&self.setup);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.splat);
            pass.dispatch_workgroups(n.div_ceil(WG), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.args, 0, &self.indirect, 0, 24);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("particle surface: resolve, blur"), timestamp_writes: None });
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_pipeline(&self.resolve);
        pass.dispatch_workgroups_indirect(&self.indirect, 0);
        for b in &self.blurs {
            pass.set_pipeline(b);
            pass.dispatch_workgroups_indirect(&self.indirect, 12);
        }
    }
}
