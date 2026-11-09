//! The renderer off screen, fed both ways: from the GPU simulation (packed and copied on the device) and
//! from the CPU one (uploaded). Before anything moves the two are the same scene, so the pictures must
//! agree. Needs a GPU: without one it says so and passes (CI has none).
use icecream_cli::engine::open_on;
use icecream_cli::scene::{scene_desc, Sync};
use icecream_gpu::wgpu;
use icecream_view::{Light, OrbitCamera, Renderer, ViewOptions};
use std::sync::Arc;

const W: u32 = 320;
const H: u32 = 200;

fn picture(cpu: bool) -> Option<(Vec<u8>, Vec<String>)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).ok()?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let errors = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(Box::new(move |e| sink.lock().unwrap().push(e.to_string())));
    let mut engine = open_on(cpu, Some((device.clone(), queue.clone(), "test".into())));
    let mut r = Renderer::new(device.clone(), queue.clone(), wgpu::TextureFormat::Rgba8Unorm, &scene_desc(engine.as_ref()));
    engine.step_frames(5);
    let mut enc = device.create_command_encoder(&Default::default());
    Sync::default().frame(&mut r, engine.as_mut(), &mut enc);
    queue.submit([enc.finish()]);
    let px = r.capture(W, H, &OrbitCamera::default(), &Light::default(), &ViewOptions::default());
    let e = errors.lock().unwrap().clone();
    Some((px, e))
}

#[test]
fn gpu_and_cpu_simulations_draw_the_same_scene() {
    let Some((g, ge)) = picture(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (c, ce) = picture(true).unwrap();
    assert!(ge.is_empty() && ce.is_empty(), "GPU errors: {ge:?} {ce:?}");
    // not just the background: the table, bowls and ice are there
    let bg = [212u8, 204, 194];
    let busy = g.as_chunks::<4>().0.iter().filter(|p| (0..3).any(|k| (p[k] as i32 - bg[k] as i32).abs() > 12)).count();
    assert!(busy > (W * H) as usize / 2, "only {busy} pixels drawn");
    let diff = g.iter().zip(&c).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum::<u64>() as f64 / g.len() as f64;
    eprintln!("{busy} of {} pixels drawn; GPU vs CPU feed: mean difference {diff:.3} per channel", W * H);
    assert!(diff < 2.0, "the two feeds differ by {diff:.2} per channel");
}

/// A single negative SDF sample must draw its whole native cell, not an interpolated blob.
#[test]
fn voxel_mode_draws_the_native_cell_bounds() {
    use glam::Vec3;
    use icecream_core::Twin;
    use icecream_view::Shading;
    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else { return };
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let errors = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(Box::new(move |e| sink.lock().unwrap().push(e.to_string())));
    let mut twin = Twin::default();
    for grid in &mut twin.sim.grids { grid.clear(); }
    let grid = &mut twin.sim.grids[1];
    let idx = 30 + grid.nx * (50 + grid.ny * 30);
    grid.data[idx] = -grid.band as f32;
    grid.color[idx*4..idx*4+4].copy_from_slice(&[255, 255, 255, 0]);
    let center = Vec3::from_array(grid.center(idx).map(|v| v as f32));
    let half = grid.cell as f32 * 0.5;
    let mut camera = OrbitCamera::default();
    camera.look_from(center + Vec3::new(0.02, 0.015, 0.025), center);
    let opts = ViewOptions {
        shading: Shading::Voxels, bowls: false, spoon: false, spoon_ice: false,
        arm: false, table: false, ghost: false, shadows: false, ..Default::default()
    };
    let mut renderer = Renderer::new(device.clone(), queue.clone(), wgpu::TextureFormat::Rgba8Unorm, &scene_desc(&twin));
    let mut enc = device.create_command_encoder(&Default::default());
    Sync::default().frame(&mut renderer, &mut twin, &mut enc);
    queue.submit([enc.finish()]);
    let px = renderer.capture(W, H, &camera, &Light::default(), &opts);
    let errors = errors.lock().unwrap().clone();
    assert!(errors.is_empty(), "{errors:?}");
    let bg = &px[..3];
    let mut bounds = [W as f32, H as f32, 0.0f32, 0.0f32];
    let mut count = 0;
    for (i, p) in px.chunks_exact(4).enumerate() {
        if (0..3).any(|k| (p[k] as i32 - bg[k] as i32).abs() > 8) {
            let (x, y) = ((i as u32 % W) as f32, (i as u32 / W) as f32);
            bounds = [bounds[0].min(x), bounds[1].min(y), bounds[2].max(x), bounds[3].max(y)];
            count += 1;
        }
    }
    assert!(count > 100, "occupied cell is missing");
    let vp = camera.projection(W as f32 / H as f32) * camera.view();
    let mut expected = [W as f32, H as f32, 0.0f32, 0.0f32];
    for corner in 0..8 {
        let point = center + Vec3::new(
            if corner & 1 == 0 { -half } else { half },
            if corner & 2 == 0 { -half } else { half },
            if corner & 4 == 0 { -half } else { half },
        );
        let ndc = vp.project_point3(point);
        let (x, y) = ((ndc.x * 0.5 + 0.5) * W as f32, (0.5 - ndc.y * 0.5) * H as f32);
        expected = [expected[0].min(x), expected[1].min(y), expected[2].max(x), expected[3].max(y)];
    }
    for k in 0..4 {
        assert!((bounds[k]-expected[k]).abs() < 2.0, "cell silhouette {bounds:?}, projected cube {expected:?}");
    }
}
