//! The SO-101's meshes: decimated by tools/so101_assets.py into assets/so101 (one interleaved vertex +
//! index block per link and material), built into the binary.

use crate::geometry::MeshData;

const META: &str = include_str!("../assets/so101/so101.json");
const BIN: &[u8] = include_bytes!("../assets/so101/so101.bin");

pub struct ArmPart {
    pub link: String,
    pub material: String,
    pub mesh: MeshData,
}

pub fn parts() -> Vec<ArmPart> {
    let meta: serde_json::Value = serde_json::from_str(META).expect("so101.json");
    let mut out = Vec::new();
    let Some(links) = meta["links"].as_object() else { return out };
    for (link, entries) in links {
        for e in entries.as_array().into_iter().flatten() {
            let u = |k: &str| e[k].as_u64().unwrap_or(0) as usize;
            let (vo, vb, io, ic) = (u("vOffset"), u("vBytes"), u("iOffset"), u("iCount"));
            let vertices = BIN[vo..vo + vb].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            let indices = BIN[io..io + ic * 4].as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
            out.push(ArmPart { link: link.clone(), material: e["material"].as_str().unwrap_or("3d_printed").to_string(), mesh: MeshData { vertices, indices } });
        }
    }
    out
}
