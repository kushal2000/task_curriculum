"""Render recorded episodes (.npz from render_vbd.py --record or render_rigid_cloth.py) to MP4 with one look for
every simulator: the Flexiv arm and Wuji hand as their URDF visual meshes in their own colours, a wooden table,
the same floor, lighting and camera. Only the sheet's colour differs, to tell the worlds apart: VBD cloth blue,
PhysX rigid plates orange.

    .venv/isaaclab3/bin/python experiments/nov27_controlled_evals/render_recording.py \
        outputs/renders/default_pin_64live_v3 outputs/renders/rigid_pin_64live_v3 [--out <dir>]

Each recording's MP4 is written next to it as <name>.mp4 (or under --out). Headless (Newton's GL viewer
with EGL); needs no simulator.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "1")

import imageio.v2 as imageio
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
VISUAL = ROOT / "outputs/renders/visual_flexiv_wuji.npz"

TABLE_SIZE = (0.475, 0.4, 0.3)          # assets/urdf/table_narrow.urdf
TABLE_RGB = (0.82, 0.56, 0.35)          # its "wood" material
FLOOR_RGB = (0.80, 0.81, 0.83)
CLOTH_RGB = {"vbd": (0.35, 0.59, 0.84), "physx": (0.52, 0.27, 0.80)}   # cloth blue, plates purple / green
PLATE2_RGB = (0.16, 0.68, 0.38)         # the rigid plates' second half (each plate its own colour)
SMOOTH = 4                              # cloth surface: Catmull-Rom samples per grid interval


def catmull_rom_weights(n: int, k: int) -> np.ndarray:
    ts = np.linspace(0.0, n - 1.0, k * (n - 1) + 1)
    W = np.zeros((len(ts), n))
    for a, t in enumerate(ts):
        i = min(int(np.floor(t)), n - 2)
        u = t - i
        c = (-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0,
             -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2)
        for m, w in zip((i - 1, i, i + 1, i + 2), c):
            W[a, min(max(m, 0), n - 1)] += w
    return W


def grid_faces(m: int) -> np.ndarray:
    f = []
    for a in range(m - 1):
        for b in range(m - 1):
            v00, v01, v10, v11 = a * m + b, a * m + b + 1, (a + 1) * m + b, (a + 1) * m + b + 1
            f += [(v00, v10, v11), (v00, v11, v01)]
    return np.asarray(f, np.int32)


def grid_faces_split(m: int, crease: int) -> tuple[np.ndarray, np.ndarray]:
    """The (m x m) grid's triangles on either side of grid row ``crease`` (the plates' hinge)."""
    a_side, b_side = [], []
    for a in range(m - 1):
        for b in range(m - 1):
            v00, v01, v10, v11 = a * m + b, a * m + b + 1, (a + 1) * m + b, (a + 1) * m + b + 1
            (a_side if a < crease else b_side).extend([(v00, v10, v11), (v00, v11, v01)])
    return np.asarray(a_side, np.int32), np.asarray(b_side, np.int32)


def box_mesh(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """A box with its own vertices per face (flat shading)."""
    v, f = [], []
    for axis in range(3):
        for s in (-1.0, 1.0):
            u, w = [a for a in range(3) if a != axis]
            h = np.array([hx, hy, hz])
            quad = []
            for du, dw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u], p[w] = s * h[axis], du * h[u], dw * h[w]
                quad.append(p)
            n0 = len(v)
            v += quad
            tri = [(n0, n0 + 1, n0 + 2), (n0, n0 + 2, n0 + 3)]
            # outward winding: flip when the in-plane axes and the normal are left-handed
            normal = np.zeros(3)
            normal[axis] = s
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), normal) < 0:
                tri = [(a, c, b) for a, b, c in tri]
            f += tri
    return np.asarray(v, np.float32), np.asarray(f, np.int32)


def load_visuals(path: Path):
    d = np.load(path, allow_pickle=True)
    return [(str(k).split("|")[0], d[f"v{i}"].astype(np.float32), d[f"f{i}"].astype(np.int32),
             tuple(float(c) / 255.0 for c in d["rgb"][i])) for i, k in enumerate(d["keys"])]


class Renderer:
    def __init__(self, width: int, height: int, visual: Path):
        from newton.viewer import ViewerGL

        self.v = ViewerGL(width=width, height=height, headless=True)
        self.visuals = load_visuals(visual)
        self.meshes_logged = False
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except OSError:
            self.font = ImageFont.load_default()

    def _mesh(self, name, verts, faces, color, cull=True):
        self.v.log_mesh(name, wp.array(np.asarray(verts, np.float32), dtype=wp.vec3),
                        wp.array(np.asarray(faces, np.int32).ravel(), dtype=wp.int32), color=color,
                        backface_culling=cull)

    def _place(self, name, mesh, q7, color):
        xf = wp.array([wp.transform(wp.vec3(*map(float, q7[:3])), wp.quat(*map(float, q7[3:7])))], dtype=wp.transform)
        self.v.log_instances(name, mesh, xf, wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3),
                             wp.array([wp.vec3(*color)], dtype=wp.vec3), None)

    def render(self, rec: Path, out: Path, label: str | None) -> None:
        ep = np.load(rec, allow_pickle=True)
        sim = str(ep["sim"]) if "sim" in ep.files else "vbd"
        names = [str(x).rsplit("/", 1)[-1] for x in ep["body_label"]]
        body_q, cloth = ep["body_q"], ep["cloth"]
        T = len(cloth)
        res = int(round(len(cloth[0]) ** 0.5))
        W = catmull_rom_weights(res, SMOOTH)
        cf = grid_faces(W.shape[0])
        halves = grid_faces_split(W.shape[0], (res // 2) * SMOOTH) if sim == "physx" else None
        haus = ep["haus_mm"] if "haus_mm" in ep.files else None
        outcome = str(ep["outcome"]).replace("_", " ")
        fps = int(round(1.0 / float(ep["policy_dt"])))
        label = label or str(ep["label"]) if ("label" in ep.files or label) else rec.parent.name

        if not self.meshes_logged:
            for k, (body, v, f, rgb) in enumerate(self.visuals):
                self._mesh(f"/vis/m{k}", v, f, rgb)
            self._mesh("/table/m", *box_mesh(*(0.5 * s for s in TABLE_SIZE)), TABLE_RGB)
            self._mesh("/floor/m", *box_mesh(2.0, 2.0, 0.005), FLOOR_RGB)
            self.meshes_logged = True
        table_i = names.index("box") if "box" in names else None
        table_q = body_q[0][table_i] if table_i is not None else np.asarray(ep["table_q"], np.float32)
        floor_z = float(table_q[2]) - 0.5 * TABLE_SIZE[2]
        view_dir = np.array([0.30, -0.75, 0.60]) / np.linalg.norm([0.30, -0.75, 0.60])
        wrist = names.index("link7")
        cam = None
        frames = []
        best = float("inf")
        for t in range(T):
            q = body_q[t]
            pts = np.einsum("ai,bj,ijc->abc", W, W, np.asarray(cloth[t], np.float64).reshape(res, res, 3)).reshape(-1, 3)
            centre = np.asarray(cloth[t]).mean(0)
            t_new = 0.5 * (centre + q[wrist, :3])
            d_new = max(0.55, 1.4 * float(np.linalg.norm(q[wrist, :3] - centre)) + 0.35)
            cam = (t_new, d_new) if cam is None else (0.85 * cam[0] + 0.15 * t_new, 0.85 * cam[1] + 0.15 * d_new)
            eye = cam[0] + cam[1] * view_dir
            dx, dy, dz = (cam[0] - eye).tolist()
            self.v.set_camera(pos=wp.vec3(*map(float, eye)), yaw=math.degrees(math.atan2(dy, dx)),
                              pitch=math.degrees(math.atan2(dz, math.hypot(dx, dy))))
            self.v.begin_frame(0.0)
            for k, (body, _v, _f, rgb) in enumerate(self.visuals):
                if body in names:
                    self._place(f"/vis/i{k}", f"/vis/m{k}", q[names.index(body)], rgb)
            self._place("/table/i", "/table/m", table_q, TABLE_RGB)
            self._place("/floor/i", "/floor/m", np.array([0, 0, floor_z - 0.005, 0, 0, 0, 1], np.float32), FLOOR_RGB)
            if halves is None:
                self._mesh("/cloth", pts, cf, CLOTH_RGB.get(sim, CLOTH_RGB["vbd"]), cull=False)
            else:  # rigid plates: one colour per plate
                self._mesh("/cloth", pts, halves[0], CLOTH_RGB["physx"], cull=False)
                self._mesh("/cloth_b", pts, halves[1], PLATE2_RGB, cull=False)
            self.v.end_frame()
            img = self.v.get_frame().numpy()[..., :3].astype(np.uint8)
            h = float(haus[t]) if haus is not None else float("nan")
            if np.isfinite(h):
                best = min(best, h)
            frames.append(self._label(img, label, t + 1, h, best, None))
        frames += [self._label(img, label, T, h, best, outcome)] * fps
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, frames, fps=fps, quality=8, macro_block_size=8)
        print(f"[render_recording] wrote {out} ({T} steps, {outcome})", flush=True)

    def _label(self, img, label, step, h, best, outcome):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, im.width, 66), fill=(0, 0, 0))
        d.text((10, 5), label, fill=(255, 255, 255), font=self.font)
        line = f"step {step}" + (f" | Hausdorff {h:.0f} mm (best {best:.0f})" if np.isfinite(h) else "") \
            + (f" | {outcome}" if outcome else "")
        d.text((10, 36), line, fill=(120, 230, 120) if best < 10.0 else (220, 220, 220), font=self.font)
        return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="recordings (.npz) or directories of them")
    ap.add_argument("--out", default=None, help="output directory (default: next to each recording)")
    ap.add_argument("--label", default=None, help="the first label line (default: the recording's, else its folder)")
    ap.add_argument("--visual", default=str(VISUAL))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    recs = []
    for p in map(Path, a.paths):
        recs += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    wp.init()
    r = Renderer(a.width, a.height, Path(a.visual))
    for rec in recs:
        out = (Path(a.out) if a.out else rec.parent) / f"{rec.stem}.mp4"
        r.render(rec, out, a.label)


if __name__ == "__main__":
    main()
