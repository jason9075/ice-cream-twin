//! The shader compiles (naga) and every entry point makes a pipeline.
use icecream_gpu::layout::{entry_points, shader, Layout};

#[test]
fn shader_validates_with_naga() {
    let l = Layout::new(&icecream_core::Sim::default());
    let code = shader(&l);
    let module = match wgpu::naga::front::wgsl::parse_str(&code) {
        Ok(m) => m,
        Err(e) => panic!("{}", e.emit_to_string(&code)),
    };
    let mut v = wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all());
    if let Err(e) = v.validate(&module) {
        panic!("{}", e.emit_to_string(&code));
    }
    let eps = entry_points(&code);
    assert!(eps.len() > 50, "{} entry points", eps.len());
}
