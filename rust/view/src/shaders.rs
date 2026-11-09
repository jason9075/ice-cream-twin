//! The render shaders (src/wgsl), assembled: each starts with the shared frame uniforms and lighting.

const COMMON: &str = include_str!("wgsl/common.wgsl");

pub fn mesh() -> String {
    format!("{COMMON}\n{}", include_str!("wgsl/mesh.wgsl"))
}

pub fn inst() -> String {
    format!("{COMMON}\n{}", include_str!("wgsl/inst.wgsl"))
}

/// The SDF volume tracer; `fluid` for the particle surface (its parameters come from a storage buffer the
/// GPU writes). The source marks the two bindings with `#ifdef FLUID` / `#else` / `#endif`.
pub fn volume(fluid: bool) -> String {
    let src = include_str!("wgsl/volume.wgsl");
    let mut out = String::new();
    let mut keep = true;
    for line in src.lines() {
        match line.trim() {
            "#ifdef FLUID" => keep = fluid,
            "#else" => keep = !fluid,
            "#endif" => keep = true,
            _ if keep => {
                out += line;
                out.push('\n');
            }
            _ => {}
        }
    }
    format!("{COMMON}\nconst FLUID = {fluid};\n{out}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_shader_validates() {
        for (name, code) in [
            ("mesh", super::mesh()),
            ("inst", super::inst()),
            ("volume", super::volume(false)),
            ("fluid surface", super::volume(true)),
            ("fluid splat", include_str!("wgsl/fluid_splat.wgsl").to_string()),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
