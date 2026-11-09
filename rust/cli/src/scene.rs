//! Between the simulation and the renderer: the scene's layout, and each frame's poses, grids and
//! particles. With the GPU simulation the ice never leaves the GPU (packed there, copied into the
//! renderer's textures on the same device); with the CPU one it is uploaded.

use crate::engine::Engine;
use icecream_core::grid::SdfGrid;
use icecream_core::sim::TABLE;
use icecream_gpu::layout::{CAP, MAX_SPILLED};
use icecream_gpu::wgpu;
use icecream_view::{FrameState, GridGeo, Renderer, SceneDesc};

pub fn scene_desc(t: &dyn Engine) -> SceneDesc {
    let s = t.sim();
    let geo = |g: &SdfGrid| GridGeo { origin: [g.ox, g.oy, g.oz], cell: g.cell, dims: [g.nx, g.ny, g.nz], band: g.band };
    SceneDesc {
        grids: [geo(&s.grids[0]), geo(&s.grids[1]), geo(&s.grids[2]), geo(&s.content), geo(&s.grids[TABLE])],
        particle_cap: CAP,
        crumb_cap: MAX_SPILLED,
        hide_jaw: t.arm().tool.hide_jaw,
    }
}

fn pose7(p: [f64; 3], q: [f64; 4]) -> [f32; 7] {
    [p[0] as f32, p[1] as f32, p[2] as f32, q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
}

/// Keeps the renderer in step with the simulation.
pub struct Sync {
    /// upload every CPU grid next time (first frame, after a reset)
    pub full: bool,
}

impl Default for Sync {
    fn default() -> Self {
        Sync { full: true }
    }
}

impl Sync {
    /// Hand the renderer this frame's ice and poses (copies recorded into `enc` for the GPU simulation).
    pub fn frame(&mut self, r: &mut Renderer, t: &mut dyn Engine, enc: &mut wgpu::CommandEncoder) {
        let st = t.state();
        let arm = t.arm();
        let ghost = if arm.enabled { pose7(arm.goal.pos.to_array(), arm.goal.quat.to_array()) } else { pose7(st.target_position, st.target_quaternion) };
        let mut f = FrameState {
            spoon: pose7(st.spoon_position, st.spoon_quaternion),
            ghost,
            links: arm.link_matrices(),
            arm_on: arm.enabled,
            particles: 0,
            crumbs: 0,
            particle_radius: t.params().fluid.particle_radius as f32,
        };
        if let Some(g) = t.gpu() {
            let pack = g.pack_for_render();
            let (pk, rows, pp, crumb_at) = g.render_buffers();
            r.copy_from_gpu(enc, pk, &rows, pack.grids, pp, pack.n, crumb_at, pack.crumbs);
            f.particles = pack.n;
            f.crumbs = pack.crumbs;
        } else {
            let full = std::mem::take(&mut self.full);
            let v = t.view();
            for (k, g) in v.grids.iter().enumerate() {
                let dist = (g.dist_changed || full).then_some(g.dist);
                let color = (g.color_changed || full).then_some(g.color);
                if dist.is_some() || color.is_some() {
                    r.upload_grid(k, dist, color);
                }
            }
            r.upload_particles(&v.particles, Some(&v.spilled));
            f.particles = (v.particles.len() / 16) as u32;
            f.crumbs = (v.spilled.len() / 16) as u32;
        }
        r.set_frame(&f);
    }
}
