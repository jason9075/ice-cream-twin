// Draws the SO-101: loads the decimated meshes written by tools/so101_assets.py and keeps one
// renderer object per (link, material), posed by the link matrices the server streams every frame.

const LOOK = {
  '3d_printed': { color: [0.96, 0.8, 0.22], rough: 0.55, metal: 0 },
  sts3215: { color: [0.07, 0.07, 0.08], rough: 0.35, metal: 0 },
};

/**
 * `arm` = the server's scene.arm: {links (frame order), jawLink, hideJaw}. update(links, enabled) takes
 * the frame's 8 x 16 row-major link world matrices.
 */
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
  const gpu = new Float32Array(16);
  for (const o of parts) o.slot = arm.links.indexOf(o.link);
  return {
    objects: parts,
    meta,
    update(links, enabled) {
      for (const o of parts) {
        o.visible = enabled && o.slot >= 0 && !(o.link === arm.jawLink && arm.hideJaw);
        if (!o.visible) continue;
        // row-major -> column-major for the GPU
        const b = o.slot * 16;
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) gpu[c * 4 + r] = links[b + r * 4 + c];
        o.setModel(gpu);
      }
    },
  };
}
