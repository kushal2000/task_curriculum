"""Visual meshes of a URDF, attached to the bodies a recording stores (fixed joints folded into their parent),
decimated for a browser: one .npz the viser replay loads with --visual.

    python build_visual_cache.py assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf <recording.npz> <out.npz>

Needs trimesh + fast_simplification (not part of the Isaac Lab venv).
"""
import sys, os
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

urdf, rec, out = sys.argv[1:4]
budget = {"arm": 12000, "hand": 2500}
root = ET.parse(urdf).getroot()
base = os.path.dirname(urdf)
bodies = [str(b).rsplit("/", 1)[-1] for b in np.load(rec, allow_pickle=True)["body_label"]]


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = rpy_matrix(*[float(x) for x in (o.get("rpy") or "0 0 0").split()])
        T[:3, 3] = [float(x) for x in (o.get("xyz") or "0 0 0").split()]
    return T


parent_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
outv = {}
for link in root.findall("link"):
    name = link.get("name")
    T, cur = np.eye(4), name
    while cur not in bodies:
        j = parent_joint.get(cur)
        if j is None or j.get("type") != "fixed":
            T = None
            break
        T = origin(j) @ T
        cur = j.find("parent").get("link")
    if T is None:
        continue
    for k, vis in enumerate(link.findall("visual")):
        m = vis.find("geometry/mesh")
        if m is None:
            continue
        kind = "hand" if name.startswith("r_") else "arm"
        loaded = trimesh.load(os.path.join(base, m.get("filename")), process=True)
        # One piece per material (the Flexiv OBJs: a white cover and a grey body); an STL is one piece.
        parts = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        total = sum(len(p.faces) for p in parts)
        for pi, mesh in enumerate(parts):
            if m.get("scale"):
                mesh.apply_scale([float(x) for x in m.get("scale").split()])
            mesh.apply_transform(T @ origin(vis))
            target = max(200, int(budget[kind] * len(mesh.faces) / max(total, 1)))
            if len(mesh.faces) > target:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
            rgb = None
            mat = getattr(getattr(mesh, "visual", None), "material", None)
            if kind == "arm" and mat is not None:
                d = getattr(mat, "diffuse", None)
                if d is None and hasattr(mat, "main_color"):
                    d = mat.main_color
                if d is not None:
                    rgb = tuple(int(c) for c in np.asarray(d)[:3])
            if rgb is None:
                if kind == "hand":  # wuji_hand2 MJCF: links 0.75 grey, fingertip pads 0.55 0.78 0.95
                    rgb = (140, 199, 242) if "tip_sensor" in name else (191, 191, 191)
                else:
                    rgb = (226, 228, 232)
            outv[f"{cur}|{name}|{k}.{pi}"] = (np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32), rgb)
            print(f"{name:32s} -> {cur:28s} {len(mesh.faces):6d} faces {rgb}")
np.savez_compressed(out, keys=np.array(list(outv)), **{f"v{i}": v[0] for i, v in enumerate(outv.values())},
                    **{f"f{i}": v[1] for i, v in enumerate(outv.values())},
                    rgb=np.array([v[2] for v in outv.values()]))
print(f"{len(outv)} visual meshes, {sum(len(v[1]) for v in outv.values())} faces -> {out}")
