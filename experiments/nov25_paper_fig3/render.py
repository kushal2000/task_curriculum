"""Isaac Sim (RTX path-traced) renders of Fig. 3's six sim images: posed plates and VBD frames on a table, no robot.

    .venv/isaacsim/bin/python experiments/nov25_paper_fig3/render.py <state_dir> <out_dir> <vbd frame> [--sweep]

--sweep also renders several VBD frames, to choose from.
"""
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

state, out, frame = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); sweep = "--sweep" in sys.argv
out.mkdir(parents=True, exist_ok=True)
app = AppLauncher(headless=True, enable_cameras=True).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from proxies import PROXIES  # noqa: E402

S = 10.0  # metres -> stage units, so nothing meets the camera's near plane
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
cs = carb.settings.get_settings()
cs.set("/rtx/rendermode", "PathTracing"); cs.set("/rtx/pathtracing/spp", 4); cs.set("/rtx/pathtracing/totalSpp", 128)
cs.set("/rtx/pathtracing/optixDenoiser/enabled", True)


def mat(name, rgb, rough=0.7):
    m = UsdShade.Material.Define(stage, f"/World/mat/{name}"); sh = UsdShade.Shader.Define(stage, f"/World/mat/{name}/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb)); sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return m


MAT = {"table": mat("table", (.30, .31, .33)), "blue": mat("blue", (.10, .25, .55)), "light": mat("light", (.35, .55, .82)),
       "orange": mat("orange", (.85, .33, .04)), "dark": mat("dark", (.03, .06, .15)), "cloth": mat("cloth", (.83, .70, .50), .95)}


def mesh(path, pts, faces, counts, m, smooth=False):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([Gf.Vec3f(*map(float, p * S)) for p in pts]); g.CreateFaceVertexIndicesAttr([int(i) for i in faces])
    g.CreateFaceVertexCountsAttr(list(counts)); g.CreateDoubleSidedAttr(True)
    g.CreateSubdivisionSchemeAttr("catmullClark" if smooth else "none")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)


def slab(path, poly, r, t, m, half=0.001, inset=0.0005):
    c = poly.mean(0); poly = c + (poly - c) * (1 - inset / np.abs(poly - c).max()); n = len(poly)
    p = np.r_[np.c_[poly, np.full(n, -half)], np.c_[poly, np.full(n, half)]] @ r.T + t
    f = [*range(n - 1, -1, -1), *range(n, 2 * n)]; cnt = [n, n]
    for i in range(n):
        f += [i, (i + 1) % n, n + (i + 1) % n, n + i]; cnt.append(4)
    mesh(path, p, f, cnt, m)


def proxy(name, ang):
    s = PROXIES[name]; tf = s.transforms(ang); lift = np.array([0, 0, 0.002])
    for i, (k, poly) in enumerate(s.plates.items()):
        slab(f"/World/obj/p{i}", poly, tf[k][0], tf[k][1] + lift, MAT["orange" if k in s.new else ("blue" if k not in s.hinges else "light")])
    for i, (c, (par, a, b)) in enumerate(s.hinges.items()):  # the hinge line, as a rod
        r, t = tf[par]; pa, pb = [r @ np.array([*q, 0.0]) + t + lift for q in (a, b)]
        d = (pb - pa) / np.linalg.norm(pb - pa); u = np.cross(d, r[:, 2]); v = np.cross(d, u); th = np.linspace(0, 2 * np.pi, 13)[:-1]
        ring = np.array([np.cos(x) * u + np.sin(x) * v for x in th]) * 0.0015
        f, cnt = [], []
        for j in range(12):
            f += [j, (j + 1) % 12, 12 + (j + 1) % 12, 12 + j]; cnt.append(4)
        mesh(f"/World/obj/h{i}", np.r_[pa + ring, pb + ring], f, cnt, MAT["orange" if c in s.new else "dark"])


def cloth(name, f):
    z = np.load(state / f"{name}.npz"); mesh("/World/obj/cloth", z["frames"][f] , z["faces"].ravel(), [3] * len(z["faces"]), MAT["cloth"])


mesh("/World/table", np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], float) / S * 3, [0, 1, 2, 3], [4], MAT["table"])
dome = UsdLux.DomeLight.Define(stage, "/World/dome"); dome.CreateIntensityAttr(220.0)
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(700.0); sun.CreateAngleAttr(4.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(38, 22, 0))
cam = rep.create.camera(position=(0.09 * S, -0.235 * S, 0.21 * S), look_at=(-0.01 * S, 0, 0.018 * S), focal_length=42, clipping_range=(0.01, 100))
rp = rep.create.render_product(cam, (1125, 1000)); an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])


def shot(fname, build):
    stage.RemovePrim("/World/obj"); UsdGeom.Xform.Define(stage, "/World/obj"); build()
    for _ in range(45):  # app.update, not orchestrator.step: that never returns under AppLauncher
        app.update()
    im = np.asarray(an.get_data())
    assert im.ndim == 3 and im.size, f"no image: {im.shape}"
    im = im[..., :3]
    Image.fromarray(im.astype(np.uint8)).save(out / f"{fname}.jpg", quality=92); print("[render]", fname, im.mean(), flush=True)


fit = np.load(state / "fit.npz")
ang = {n: {k.split("/")[1]: float(fit[k]) for k in fit.files if k.startswith(n + "/")} for n in ("strip", "corner")}
shot("train_1", lambda: proxy("box2", {"E": np.deg2rad(105)}))
shot("train_2", lambda: proxy("diag", {"B": np.deg2rad(110)}))
for i, n in enumerate(("strip", "corner"), 1):
    shot(f"eval_{i}", lambda: cloth(n, frame)); shot(f"reproduce_{i}", lambda: proxy(f"{n}_flap", ang[n]))
    if sweep:
        for f in (13, 17, 21, 25, 29, 33):
            shot(f"sweep_{n}_{f:02d}", lambda: cloth(n, f))
app.close()
