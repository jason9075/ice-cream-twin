// Draws the SO-101: loads the decimated meshes written by tools/so101_assets.py and keeps one
// renderer object per (link, material), posed from Arm.linkMatrices() every frame.
import { m4 } from './so101.js';

const LOOK = {
  '3d_printed': { color: [0.96, 0.8, 0.22], rough: 0.55, metal: 0 },
  sts3215: { color: [0.07, 0.07, 0.08], rough: 0.35, metal: 0 },
};

export async function createArmView(renderer, arm, base = 'so101/') {
  const [meta, bin] = await Promise.all([
    fetch(`${base}so101.json`).then((r) => r.json()),
    fetch(`${base}so101.bin`).then((r) => r.arrayBuffer()),
  ]);
  const parts = [];
  for (const [link, entries] of Object.entries(meta.links)) {
    for (const e of entries) {
      const mesh = renderer.createMesh({
        vertices: new Float32Array(bin, e.vOffset, e.vBytes / 4),
        indices: new Uint32Array(bin, e.iOffset, e.iCount),
      });
      const obj = renderer.createObject(mesh, LOOK[e.material] ?? LOOK['3d_printed']);
      obj.link = link;
      parts.push(obj);
    }
  }
  const jawLink = 'moving_jaw_so101_v1_link';
  const gpu = new Float32Array(16);
  return {
    objects: parts,
    meta,
    update() {
      const T = arm.linkMatrices();
      for (const o of parts) {
        o.visible = arm.enabled && !(o.link === jawLink && arm.tool.hideJaw);
        if (o.visible) o.setModel(m4.toGPU(T[o.link], gpu));
      }
    },
  };
}
