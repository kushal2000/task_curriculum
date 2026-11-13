"""Visual-only fabric looks for the viser replay viewers.

Each fabric gets a look from LOOKS: a photo of a real weave (CC0 Poly Haven scans, cropped to the
sheet's true 10 cm, in assets/fabric_textures/), a plain colour, or a sheen material for silk. Photos
are recoloured to the tint the viewer passes (the swatch colour, or the parameter variant in the
cantilever view), keeping only their light/dark weave detail. Nothing here touches the simulation:
the viewers replay saved vertex frames and this only changes how one frame is drawn.

UVs come from the particle grid (row-major, vertex i = (i % res, i // res), as every probe saves
it), so the texture is pinned to the material and follows it as it deforms.

The mesh goes to viser as a hand-built GLB. The image for each (fabric, tint) is encoded once and
cached, so a redraw only re-sends positions, normals and indices plus a few KB of image.

    from fabric_textures import add_fabric_mesh
    add_fabric_mesh(server, name, verts, faces, "denim_measured", (50, 85, 165))
"""

from __future__ import annotations

import functools
import io
import json
import struct
from pathlib import Path

import numpy as np

PHOTO_DIR = Path(__file__).resolve().parents[2] / "assets/fabric_textures"

# fabric -> (photo in PHOTO_DIR or None for a plain colour, weave contrast, material).
# Materials are glTF pbrMetallicRoughness values; "sheen" adds KHR_materials_sheen, which three.js
# draws as the soft grazing-angle glow of silk/satin. A loader without the extension still gets the
# lower roughness, i.e. a glossy highlight.
MATTE = dict(roughness=0.9)
LOOKS = {
    "chiffon": ("terlenka.jpg", 0.30, MATTE),
    "silk": (None, 0.0, dict(roughness=0.35, sheen=(1.0, 1.0, 1.0), sheen_roughness=0.3)),
    "cotton": ("stretch_poplin.jpg", 0.30, MATTE),
    "napkin": (None, 0.0, MATTE),
    "big_napkin": ("terlenka.jpg", 0.30, MATTE),
    "towel": ("terry_cloth.jpg", 0.30, dict(roughness=1.0)),
    "denim": ("denim_fabric_03.jpg", 0.25, MATTE),
}
DEFAULT_LOOK = (None, 0.0, MATTE)


def base_name(fab: str) -> str:
    """chiffon_measured_r25 -> chiffon, 'default VBD' -> default"""
    fab = fab.removesuffix("_r25").removesuffix("_measured")
    return "default" if fab.startswith("default") or fab == "base" else fab


def look(fab: str):
    return LOOKS.get(base_name(fab), DEFAULT_LOOK)


@functools.lru_cache(maxsize=256)
def texture_image(fab: str, tint: tuple[int, int, int]) -> tuple[bytes, str]:
    """(image bytes, mime type): the fabric's photo recoloured to `tint`, or a plain `tint`."""
    from PIL import Image

    photo, contrast, _ = look(fab)
    t = np.asarray(tint, dtype=np.float32) / 255
    buf = io.BytesIO()
    if photo is None or not (PHOTO_DIR / photo).exists():
        Image.new("RGB", (4, 4), tuple(int(c) for c in tint)).save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    a = np.asarray(Image.open(PHOTO_DIR / photo).convert("RGB"), dtype=np.float32) / 255
    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    detail = (lum - lum.mean()) / (lum.std() + 1e-6)  # weave detail only, the photo's own colour dropped
    rgb = np.clip(t * (1 + contrast * detail[..., None]), 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8), "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


@functools.lru_cache(maxsize=16)
def grid_uv(n: int) -> np.ndarray:
    res = int(round(np.sqrt(n)))
    assert res * res == n, f"{n} vertices is not a square particle grid"
    i = np.arange(n)
    return (np.stack([i % res, i // res], axis=1) / (res - 1)).astype(np.float32)


def _normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _material(m: dict) -> tuple[dict, list[str]]:
    mat = {"doubleSided": True, "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": m.get("roughness", 0.9)}}
    if "sheen" not in m:
        return mat, []
    mat["extensions"] = {"KHR_materials_sheen": {
        "sheenColorFactor": list(m["sheen"]), "sheenRoughnessFactor": m.get("sheen_roughness", 0.3)}}
    return mat, ["KHR_materials_sheen"]


def fabric_glb(verts: np.ndarray, faces: np.ndarray, image: bytes, mime: str, uv: np.ndarray,
               material: dict = MATTE) -> bytes:
    """A one-mesh, one-texture, double-sided glTF binary."""
    v = np.ascontiguousarray(verts, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    parts = [v.tobytes(), _normals(v, f).tobytes(), np.ascontiguousarray(uv, np.float32).tobytes(), f.tobytes(), image]
    views, blob = [], b""
    for p in parts:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(p)})
        blob += p + b"\0" * (-len(p) % 4)
    for i, target in enumerate((34962, 34962, 34962, 34963)):
        views[i]["target"] = target
    n = len(v)
    mat, ext = _material(material)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [mat],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 4, "mimeType": mime}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": v.min(0).tolist(), "max": v.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": f.size, "type": "SCALAR"},
        ],
    }
    if ext:
        gltf["extensionsUsed"] = ext
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def add_fabric_mesh(server, name: str, verts, faces, fab: str, tint, textured: bool = True, **kw):
    """Drop-in for `add_mesh_simple(name, verts, faces, color=tint, side="double")` that draws the
    fabric's look. `textured=False` falls back to exactly that flat mesh."""
    tint = tuple(int(c) for c in tint)
    if not textured:
        return server.scene.add_mesh_simple(name, vertices=np.asarray(verts, np.float32), faces=faces,
                                            color=tint, side="double", flat_shading=False, **kw)
    image, mime = texture_image(fab, tint)
    glb = fabric_glb(verts, faces, image, mime, grid_uv(len(verts)), look(fab)[2])
    return server.scene.add_glb(name, glb, **kw)
