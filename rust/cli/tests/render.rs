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
