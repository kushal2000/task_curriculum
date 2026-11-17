"""Presentation videos in one Isaac Sim RTX studio scene: a live PhysX policy, or a replayed Newton episode.

    sbatch experiments/nov16_vbd_cloth/render_demo.sh rigid      # live: rigid fold pool on PhysX
    sbatch experiments/nov16_vbd_cloth/render_demo.sh denim      # record on Newton, replay here

Both modes share the scene, so the two clips match: the same robot, an oak table, a concrete floor,
a soft key light and a bright dome, the same camera and captions.

* **live** (default): one env per pool shape (``fold_pool.one_shape_per_env``: env k carries shape
  k). The camera visits the envs in turn and films one episode of each, reset to termination; the
  other envs are hidden. Plates are matte printed PLA, UV'd in the sheet frame (every plate's frame
  is the sheet frame at rest), so the print lines run straight across hinges and fold with them.
* **replay** (``--replay traj.npz`` from ``record_newton.py``): the Newton VBD cloth episode, drawn as
  a textured mesh (denim), with the robot set to the recorded joint positions every frame. No
  physics step runs here: the motion is Newton's, only the picture is Isaac Sim's.

Visual-only edits throughout -- the collision geometry PhysX cooked is untouched (the task's table,
ground, goal marker and, in replay, the plates are made invisible, not removed).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # textures.py

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysx-Direct-v0")
parser.add_argument("--checkpoint", default=None, help="live mode: the policy")
parser.add_argument("--policy_config", default="checkpoints/config.yaml")
parser.add_argument("--replay", type=Path, default=None, help="replay a record_newton.py trajectory")
parser.add_argument("--sapg_expl_coef", type=float, default=0.0)
parser.add_argument("--success_tolerance", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=60, help="policy runs at 60 Hz: 60 = real time")
parser.add_argument("--max_episode_steps", type=int, default=400, help="live: cut an episode's clip here")
parser.add_argument("--hold_s", type=float, default=0.8, help="hold the last frame of each episode")
parser.add_argument("--lead_s", type=float, default=0.4, help="hold the first frame of each episode")
parser.add_argument("--skip_drop_steps", type=int, default=18,
                    help="live: run (unfilmed) this many steps of each episode while the sheet drops onto the "
                         "table from its spawn height (0.2 m, ~0.2 s); replay finds the landing itself")
parser.add_argument("--cam_offset", type=float, nargs=3, default=(0.30, -0.46, 0.90))
parser.add_argument("--look_at", type=float, nargs=3, default=(0.03, 0.02, 0.54))
parser.add_argument("--plate_texture", default="pla")
parser.add_argument("--cloth_texture", default="denim")
parser.add_argument("--subdiv", type=int, default=2, help="replay: Loop subdivision levels (display only)")
parser.add_argument("--cloth_tile_m", type=float, default=0.12, help="replay: metres of cloth per texture repeat")
parser.add_argument("--cloth_lift_m", type=float, default=8e-4,
                    help="replay: draw the zero-thickness sheet this far up (about half a denim thickness)")
parser.add_argument("--floor_texture", default="floor")
parser.add_argument("--caption", default="")
parser.add_argument("--subtitle", default="")
parser.add_argument("--no_shape_tag", action="store_true")
parser.add_argument("--warmup_frames", type=int, default=24, help="RTX accumulation before each clip")
parser.add_argument("--out", type=Path, default=Path("videos/demo/rigid.mp4"))
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402
from textures import texture_path  # noqa: E402

SPACING = 8.0  # m between envs


# ---------------------------------------------------------------------------------------------
# USD dressing
# ---------------------------------------------------------------------------------------------
def _stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def _tex_material(path: str, tex_file, roughness: float, specular: float = 0.25, tint=(1.0, 1.0, 1.0)):
    from pxr import Sdf, UsdShade

    stage = _stage()
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((specular,) * 3)
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_file))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((*map(float, tint), 1.0))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _flat_material(path: str, color, roughness: float, specular: float = 0.2):
    from pxr import Sdf, UsdShade

    stage = _stage()
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(map(float, color)))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((specular,) * 3)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def _set_st(mesh, st):
    from pxr import Sdf, UsdGeom, Vt

    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    pv.Set(Vt.Vec2fArray([tuple(map(float, x)) for x in st]))


def _box(path: str, lo, hi, tile_m: float, mat):
    """A textured box mesh spanning [lo, hi] (world), 4 verts per face so faces own their UVs."""
    from pxr import Gf, UsdGeom, Vt

    (x0, y0, z0), (x1, y1, z1) = lo, hi
    faces = [
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
    ]
    axes = [(0, 1), (0, 1), (1, 2), (1, 2), (0, 2), (0, 2)]
    pts, st = [], []
    for f, (a, b) in zip(faces, axes):
        for c in f:
            pts.append(Gf.Vec3f(*map(float, c)))
            st.append(((c[a] - lo[a]) / tile_m, (c[b] - lo[b]) / tile_m))
    mesh = UsdGeom.Mesh.Define(_stage(), path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(24))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    _set_st(mesh, st)
    _bind(mesh.GetPrim(), mat)
    return mesh


def _visible(path: str, on: bool):
    from pxr import UsdGeom

    prim = _stage().GetPrimAtPath(path)
    if prim.IsValid():
        im = UsdGeom.Imageable(prim)
        im.MakeVisible() if on else im.MakeInvisible()


class Studio:
    """The shared look: materials, floor, lights, hidden task visuals, per-env textured tables."""

    def __init__(self, num_envs: int):
        from pxr import Gf, Usd, UsdGeom, UsdLux

        stage = _stage()
        self.num_envs = num_envs
        # Plain matte off-white table: the manipuland is the only textured thing in frame.
        self.table_mat = _flat_material("/World/Looks/Table", (0.52, 0.52, 0.51), 0.6, 0.2)
        floor_mat = _tex_material("/World/Looks/Floor", texture_path(args.floor_texture), 0.7, 0.2)
        for i in range(num_envs):
            _visible(f"/World/envs/env_{i}/GoalViz", False)  # the goal marker plate
            _visible(f"/World/envs/env_{i}/Table", False)  # replaced by a textured box (place_table)
        _visible("/World/ground", False)
        R = SPACING * (num_envs + 3)
        _box("/World/Pretty/floor", (-R, -R, -0.02), (R, R, -1e-4), 1.5, floor_mat)

        # Lights: a bright, slightly cool dome (also the backdrop) and a warm key with a soft penumbra.
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                UsdGeom.Imageable(prim).MakeInvisible()
        dome = UsdLux.DomeLight.Define(stage, "/World/Pretty/Dome")
        dome.CreateIntensityAttr(900.0)
        dome.CreateColorAttr(Gf.Vec3f(0.93, 0.95, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/Pretty/Key")
        key.CreateIntensityAttr(2600.0)
        key.CreateAngleAttr(3.0)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(38.0, 0.0, -35.0))

    def place_table(self, i: int, center, half):
        """Textured box for env i's table, from the live/recorded pose (USD xforms go stale under PhysX)."""
        e = 5e-4  # sides proud of the collision box; the TOP is exactly the collision top, where
        # the sheet rests (a VBD sheet sits ~0.2 mm above it -- a proud top would swallow it)
        c, h = np.asarray(center, float), np.asarray(half, float)
        lo = (c[0] - h[0] - e, c[1] - h[1] - e, c[2] - h[2])
        hi = (c[0] + h[0] + e, c[1] + h[1] + e, c[2] + h[2])
        _box(f"/World/Pretty/table_{i}", lo, hi, 0.6, self.table_mat)

    def solo(self, k: int):
        """Show env k only."""
        for i in range(self.num_envs):
            _visible(f"/World/envs/env_{i}", i == k)
            _visible(f"/World/Pretty/table_{i}", i == k)


def table_half_extents(urdf: str):
    """Half extents of the table box from its URDF's collision geometry (the USD bound drops the
    prim's own scale op and reads a box several times too small)."""
    import xml.etree.ElementTree as ET

    box = ET.parse(urdf).getroot().find(".//collision/geometry/box")
    return 0.5 * np.array([float(x) for x in box.get("size").split()])


# ---------------------------------------------------------------------------------------------
def _font(bold: bool, size: int):
    from PIL import ImageFont

    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for p in (Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name,
              Path("/usr/share/fonts/truetype/dejavu") / name):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def stamp_caption(image, title: str, subtitle: str = "", tag: str = ""):
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(image))
    draw = ImageDraw.Draw(img, "RGBA")
    s = img.height / 1080.0
    f1, f2 = _font(True, int(36 * s)), _font(False, int(25 * s))
    if title:
        x, y = int(44 * s), img.height - int((116 if subtitle else 80) * s)
        w = max(draw.textlength(title, font=f1), draw.textlength(subtitle, font=f2) if subtitle else 0)
        h = int((92 if subtitle else 54) * s)
        draw.rounded_rectangle([x - 20 * s, y - 14 * s, x + w + 20 * s, y + h], radius=int(14 * s),
                               fill=(22, 24, 30, 160))
        draw.text((x, y), title, font=f1, fill=(255, 255, 255, 255))
        if subtitle:
            draw.text((x, y + int(46 * s)), subtitle, font=f2, fill=(222, 226, 234, 255))
    if tag:
        x, y = int(44 * s), int(38 * s)
        w = draw.textlength(tag, font=f2)
        draw.rounded_rectangle([x - 16 * s, y - 10 * s, x + w + 16 * s, y + 40 * s], radius=int(12 * s),
                               fill=(22, 24, 30, 140))
        draw.text((x, y), tag, font=f2, fill=(255, 255, 255, 255))
    return np.asarray(img)


def loop_operator(tri, n: int, levels: int):
    """Loop subdivision as one dense linear map: ``fine_verts = S @ coarse_verts`` (and the fine tris).

    Display-only: the recorded 7x7 sheet is what the solver simulated; this only smooths how it is
    drawn. Boundary edges use the crease rules (so the hem stays a curve, not a zig-zag) and corners
    (boundary vertices on exactly two triangles' worth of fan, i.e. valence 2) are pinned, so a
    square sheet keeps square corners.
    """
    S = np.eye(n)
    tri = np.asarray(tri, np.int64)
    for _ in range(levels):
        edges = {}
        for t in tri:
            for a, b, c in ((t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])):
                edges.setdefault((min(a, b), max(a, b)), []).append(c)
        nbr = [set() for _ in range(n)]
        bnbr = [set() for _ in range(n)]
        for (a, b), opp in edges.items():
            nbr[a].add(b); nbr[b].add(a)
            if len(opp) == 1:
                bnbr[a].add(b); bnbr[b].add(a)
        eid = {e: n + i for i, e in enumerate(edges)}
        m = n + len(edges)
        L = np.zeros((m, n))
        for v in range(n):
            if bnbr[v]:
                if len(nbr[v]) <= 2:  # corner
                    L[v, v] = 1.0
                else:
                    L[v, v] = 0.75
                    for u in bnbr[v]:
                        L[v, u] += 0.125
            else:
                k = len(nbr[v])
                beta = 3.0 / (8.0 * k) if k > 3 else 3.0 / 16.0
                L[v, v] = 1.0 - k * beta
                for u in nbr[v]:
                    L[v, u] += beta
        for (a, b), opp in edges.items():
            r = eid[(a, b)]
            if len(opp) == 2:
                L[r, a] += 0.375; L[r, b] += 0.375
                L[r, opp[0]] += 0.125; L[r, opp[1]] += 0.125
            else:
                L[r, a] += 0.5; L[r, b] += 0.5
        new = []
        for a, b, c in tri:
            ab, bc, ca = eid[(min(a, b), max(a, b))], eid[(min(b, c), max(b, c))], eid[(min(c, a), max(c, a))]
            new += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        tri, n, S = np.asarray(new, np.int64), m, L @ S
    return S.astype(np.float32), tri


def _vertex_normals(v, tri):
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    n = np.zeros_like(v)
    for j in range(3):
        np.add.at(n, tri[:, j], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------------------------
def configure(env_cfg, num_envs: int):
    import isaaclab.sim as sim_utils

    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = SPACING
    env_cfg.assets.num_assets_per_type = 1
    env_cfg.seed = args.seed
    env_cfg.termination.eval_success_tolerance = args.success_tolerance
    disable_randomization(env_cfg)
    use_single_object_variant()
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args.cam_offset)
    env_cfg.viewer.lookat = tuple(args.look_at)
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.render = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="DLAA", enable_shadows=True,
        enable_ambient_occlusion=True, enable_reflections=True, enable_global_illumination=True,
    )


def live(env_cfg, writer):
    from isaacsimenvs.eval.player import PretrainedPlayer

    n_shapes = len(env_cfg.fold_pool.shapes)
    configure(env_cfg, n_shapes)
    env_cfg.fold_pool.one_shape_per_env = True
    env_cfg.fold_pool.weights = None
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    inner = env.unwrapped
    studio = Studio(inner.num_envs)
    shape_names = [sh.name for sh in inner._pool]
    env_shape = inner._env_shape_np

    for i in range(inner.num_envs):  # plates: printed PLA, metric UVs in the sheet frame
        from pxr import Usd, UsdGeom

        # One print colour per hinged segment (cycled), so each shape's hinge layout -- what tells
        # box / cross / star apart -- reads at a glance. Baked into the textures (textures.py
        # PLATE_PALETTE): RTX ignores UsdUVTexture's `scale`, so a shader-side tint does nothing.
        from textures import PLATE_PALETTE

        plate_mats = [_tex_material(f"/World/Looks/PLA{j}", texture_path(f"{args.plate_texture}_{j}"), 0.42, 0.45)
                      for j in range(len(PLATE_PALETTE))]
        for child in _stage().GetPrimAtPath(f"/World/envs/env_{i}").GetChildren():
            if child.GetName().startswith("FoldPool"):
                j = 0
                for p in Usd.PrimRange(child):
                    if p.IsA(UsdGeom.Mesh):
                        m = UsdGeom.Mesh(p)
                        _set_st(m, np.asarray(m.GetPointsAttr().Get())[:, :2] / 0.1)
                        _bind(p, plate_mats[j % len(plate_mats)])
                        j += 1

    player = PretrainedPlayer(
        config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=inner.num_envs,
        device=inner.device, num_observations=int(inner.cfg.observation_space),
        num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
    )
    obs, _ = env.reset()
    obs, *_ = env.step(torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device))
    # The table's pose from the reset config (pinned: no xy/z noise), not from the USD or the
    # articulation view -- both read a stale spawn pose here and put the box below the sheet.
    rc = inner.cfg.reset
    for i in range(inner.num_envs):
        half = table_half_extents(inner.cfg.assets.table_urdf)
        o = inner.scene.env_origins[i].cpu().numpy()
        c = o + np.array([rc.table_reset_xy[0], rc.table_reset_xy[1], rc.table_reset_z])
        print(f"[render_physx] env {i} table centre {np.round(c - o, 3)} half {np.round(half, 3)}", flush=True)
        studio.place_table(i, c, half)

    cam = inner.viewport_camera_controller
    pretty = lambda n: {"box2": "box", "box2diag4": "diagonal", "box2x8": "box 2x8", "cross4": "cross",
                        "corners8": "corners", "box4": "box 4", "star8": "star"}.get(n.replace("-mid", ""), n)

    def step():
        nonlocal obs
        act = player.get_action(obs["policy"], deterministic=True)
        obs, _r, term, trunc, _ = env.step(act)
        done = term.bool() | trunc.bool()
        if bool(done.any()):
            player.reset_rnn(done.nonzero(as_tuple=True)[0])
        return term.bool(), trunc.bool()

    results = []
    for k in range(inner.num_envs):
        studio.solo(k)
        cam.set_view_env_index(k)
        tag = "" if args.no_shape_tag else f"shape: {pretty(shape_names[env_shape[k]])}"
        for _ in range(2000):  # film a fresh episode: run until env k's current one ends
            t, tr = step()
            if bool(t[k] | tr[k]):
                break
        skipped = 0
        for skipped in range(args.skip_drop_steps):  # the reset's drop: real, but not the policy's doing
            t, tr = step()
            if bool(t[k] | tr[k]):
                break
        for _ in range(args.warmup_frames):
            env.render()
        frame = lambda: stamp_caption(env.render(), args.caption, args.subtitle, tag)
        last = frame()
        for _ in range(int(args.lead_s * args.fps)):
            writer.append_data(last)
        outcome = "cut"
        for n in range(args.max_episode_steps):
            folded = bool(inner.is_folded()[k]) if hasattr(inner, "is_folded") else False
            t, tr = step()
            if bool(t[k] | tr[k]):
                reasons = getattr(inner, "_termination_reasons", {})
                fired = [r for r, v in reasons.items() if bool(v[k])]
                outcome = "fold" if (folded and bool(t[k])) else (",".join(fired) or "timeout")
                break
            last = frame()
            writer.append_data(last)
        for _ in range(int(args.hold_s * args.fps)):
            writer.append_data(last)
        steps = n + 1 + args.skip_drop_steps
        results.append((shape_names[env_shape[k]], outcome, steps))
        print(f"[render_physx] env {k} {shape_names[env_shape[k]]}: {outcome} after {steps} steps", flush=True)
    return env, results


def replay(env_cfg, writer):
    from pxr import Sdf, UsdGeom, Vt

    tr = np.load(args.replay)
    configure(env_cfg, 1)
    env_cfg.fold_pool.shapes = [env_cfg.fold_pool.shapes[0]]
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    inner = env.unwrapped
    studio = Studio(1)
    for child in _stage().GetPrimAtPath("/World/envs/env_0").GetChildren():
        if child.GetName().startswith("FoldPool"):
            UsdGeom.Imageable(child).MakeInvisible()
    env.reset()
    origin = inner.scene.env_origins[0].cpu().numpy()

    # The recorded table (the Cloth task's own, narrower than the pool's).
    studio.place_table(0, tr["table_center"][0] + origin, tr["table_half"])

    # Cloth mesh: denim, UVs from the rest pose, smooth normals recomputed every frame.
    S, tri = loop_operator(tr["tris"], len(tr["rest"]), args.subdiv)
    print(f"[render_physx] cloth: {len(tr['rest'])} -> {len(S)} verts, {len(tri)} tris (Loop x{args.subdiv})", flush=True)
    mesh = UsdGeom.Mesh.Define(_stage(), "/World/Pretty/cloth")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tri)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(tri.flatten().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _set_st(mesh, (S @ tr["rest"])[:, :2] / args.cloth_tile_m)
    _bind(mesh.GetPrim(), _tex_material("/World/Looks/Denim", texture_path(args.cloth_texture), 0.9, 0.12))
    pts_attr, nrm_attr = mesh.CreatePointsAttr(), mesh.CreateNormalsAttr()

    robot = inner.robot
    names = [str(n) for n in tr["joint_names"]]
    idx = [names.index(n) if n in names else -1 for n in robot.joint_names]
    missing = [n for n, j in zip(robot.joint_names, idx) if j < 0]
    if missing:
        print(f"[render_physx] WARNING: joints not in the recording, left at default: {missing}", flush=True)
    q_default = robot.data.default_joint_pos[:1].clone()
    root = torch.as_tensor(np.concatenate([tr["root_pos"] + origin, tr["root_quat"]]), dtype=torch.float32,
                           device=inner.device)[None]

    def show(t):
        q = q_default.clone()
        for a, j in enumerate(idx):
            if j >= 0:
                q[0, a] = float(tr["joint_pos"][t, j])
        robot.write_root_pose_to_sim(root)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        inner.scene.write_data_to_sim()
        inner.sim.forward()
        v = S @ tr["verts"][t] + origin + np.array([0.0, 0.0, args.cloth_lift_m])
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        nrm_attr.Set(Vt.Vec3fArray.FromNumpy(_vertex_normals(v, tri).astype(np.float32)))

    T = len(tr["joint_pos"])
    # Start once the sheet has landed: the reset drops it from its spawn height (0.3 m above the
    # table), which is the task's reset, not the policy -- and reads as a floating sheet on film.
    top = float(tr["table_center"][0][2] + tr["table_half"][2])
    landed = np.nonzero(tr["verts"][:, :, 2].max(1) < top + 0.02)[0]
    t0 = int(landed[0]) if len(landed) else 0
    print(f"[render_physx] replay: sheet lands at frame {t0} of {T}; filming from there", flush=True)
    show(t0)
    for _ in range(args.warmup_frames):
        env.render()
    frame = lambda: stamp_caption(env.render(), args.caption, args.subtitle)
    last = frame()
    for _ in range(int(args.lead_s * args.fps)):
        writer.append_data(last)
    for t in range(t0 + 1, T):
        show(t)
        last = frame()
        writer.append_data(last)
    for _ in range(int(args.hold_s * args.fps)):
        writer.append_data(last)
    return env, [("replay", str(tr["outcome"]), T)]


@hydra_task_config_with_yaml(args.task, "")
def run(env_cfg, agent_cfg) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None, quality=9)
    env, results = replay(env_cfg, writer) if args.replay else live(env_cfg, writer)
    writer.close()
    print(f"[render_physx] wrote {args.out}: {results}", flush=True)
    env.close()


run()
sys.stdout.flush()
os._exit(0)
