"""Fetch the SO-101 arm (TheRobotStudio/SO-ARM100, Apache-2.0) and convert it for the web page.

    uv run --with trimesh --with pymeshlab --with numpy --with scipy --with networkx tools/so101_assets.py

Writes
  public/so101/so101.bin   decimated meshes, one per (link, material): f32 [px py pz nx ny nz]* + u32 indices
  public/so101/so101.json  where each mesh lives in the .bin, plus the source commit and licence
  public/so101/LICENSE     the SO-ARM100 licence (Apache-2.0), shipped next to the meshes
  src/robot/so101-urdf.js  the kinematic chain (joints, origins, axes, limits) read from the URDF

Visual origins are baked into the vertices, so every mesh is already in its link's frame.
"""

import json
import math
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh

SHA = "a758567c3978dfeefe282ede0500085a48fe8f78"
BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{SHA}/Simulation/SO101/"
URDF = "so101_new_calib.urdf"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "so101"
TARGET_FACES = {"sts3215": 1200, "3d_printed": 3500}  # per visual


def fetch(rel: str) -> Path:
    p = CACHE / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print("fetch", rel, file=sys.stderr)
        urllib.request.urlretrieve(BASE + rel, p)
    return p


def floats(s, n=3):
    v = [float(x) for x in (s or "").split()]
    return v if len(v) == n else [0.0] * n


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def main():
    root = ET.parse(fetch(URDF)).getroot()

    # ------------------------------------------------------------ meshes
    blobs, index, offset = [], {}, 0
    for link in root.findall("link"):
        name = link.get("name")
        groups = {}
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            mat = vis.find("material").get("name")
            o = vis.find("origin")
            T = np.eye(4)
            T[:3, :3] = rpy_matrix(*floats(o.get("rpy")))
            T[:3, 3] = floats(o.get("xyz"))
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(fetch(mesh_el.get("filename"))))
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=TARGET_FACES[mat], preservenormal=True, planarquadric=True
            )
            cm = ms.current_mesh()
            m = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
            m.apply_transform(T)
            groups.setdefault(mat, []).append(m)
        entries = []
        for mat, ms in groups.items():
            m = trimesh.util.concatenate(ms)
            m = trimesh.graph.smooth_shade(m, angle=math.radians(35))
            verts = np.hstack([m.vertices, m.vertex_normals]).astype(np.float32)
            idx = m.faces.astype(np.uint32).ravel()
            vb, ib = verts.tobytes(), idx.tobytes()
            entries.append(
                {"material": mat, "vOffset": offset, "vBytes": len(vb), "iOffset": offset + len(vb), "iCount": len(idx)}
            )
            blobs += [vb, ib]
            offset += len(vb) + len(ib)
            print(f"{name:28s} {mat:10s} {len(m.faces):6d} tris", file=sys.stderr)
        if entries:
            index[name] = entries

    out = ROOT / "public" / "so101"
    out.mkdir(parents=True, exist_ok=True)
    (out / "so101.bin").write_bytes(b"".join(blobs))
    meta = {
        "source": f"https://github.com/TheRobotStudio/SO-ARM100/tree/{SHA}/Simulation/SO101/{URDF}",
        "license": "Apache-2.0 (c) TheRobotStudio; meshes decimated by tools/so101_assets.py",
        "links": index,
    }
    (out / "so101.json").write_text(json.dumps(meta, indent=1) + "\n")
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{SHA}/LICENSE", out / "LICENSE")

    # ------------------------------------------------------------ kinematic chain
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        lim = j.find("limit")
        ax = j.find("axis")
        joints.append(
            {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": floats(o.get("xyz")),
                "rpy": floats(o.get("rpy")),
                "axis": floats(ax.get("xyz")) if ax is not None else [0, 0, 0],
                "lower": float(lim.get("lower")) if lim is not None else 0.0,
                "upper": float(lim.get("upper")) if lim is not None else 0.0,
            }
        )
    js = (
        "// Generated by tools/so101_assets.py from TheRobotStudio/SO-ARM100 (Apache-2.0),\n"
        f"// {meta['source']}\n"
        "// Do not edit by hand.\n"
        f"export const SO101_JOINTS = {json.dumps(joints, indent=2)};\n"
    )
    (ROOT / "src" / "robot").mkdir(exist_ok=True)
    (ROOT / "src" / "robot" / "so101-urdf.js").write_text(js)
    print(f"wrote {offset / 1e6:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
