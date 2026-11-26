"""Run a fold policy LIVE in a browser (viser), with its knobs on sliders.

    experiments/newton_py -m isaacsimenvs.scripts.visualization.policies.live_viser --task Isaacsimenvs-Cloth-Direct-v0
    experiments/viewer.sh --task Isaacsimenvs-Cloth-Direct-v0   # on a GPU machine; prints the host to tunnel to

The env runs on the GPU exactly as `episodes.py` runs it (same reset, same zero-action tick, same
player), a handful of envs at once; one of them is drawn. What can change WITHOUT a restart:

* policy (any checkpoint in `POLICIES`), SAPG block coefficient, deterministic / sampled actions
* which env is drawn; for the fold pool, which shape each env gets at its next reset
* start pose: pinned (`disable_randomization`'s reset) or the training reset distribution
* fold tolerance (`cloth.keypoint_tolerance`, read by `is_folded` every step)
* rigid manipulands only: slat / fingertip friction, hinge damping and stiffness, contact margin,
  mass -- written through the rigid backend's `physics_dr.draw`, the same in-place path the training
  DR uses, so they take effect on the very next step

What needs a RESTART (the "Restart" button re-execs this process with the new task, ~2-3 min):
the task itself (pool / one chain / VBD cloth), the number of envs, and the sheet sizes (Newton
bakes collision meshes at build time). VBD cloth material parameters are not exposed.

Speed: the fold pool runs ~40-50 policy steps/s (real time is 60). The VBD cloth is captured as a
CUDA graph here (`cloth.use_cuda_graph`), ~20 steps/s -- a third of real time; without the graph it
is ~4.

Geometry comes from Newton's own viewer pipeline (`ViewerBase.set_model` -> `log_geo`), captured
by a null viewer, so what is drawn is exactly the collision geometry the solver sees.

**PhysX tasks** (`RigidCloth` / `*Physx*`, run in `.venv_isaacsim`; `experiments/viewer.sh --task Isaacsimenvs-RigidCloth-Direct-v0`): the
same viewer, with the geometry read from the USD stage (each rigid body's collision meshes and
primitives, in its own frame) and the body poses from Isaac Lab's PhysX views every step. The friction
sliders are hidden there: PhysX refuses friction DR (`backends/rigid/physx.py`). viser needs websockets >= 13,
newer than Isaac Sim's 12, so the launcher puts a copy first on PYTHONPATH (nothing Kit loads headless
imports websockets).

**Point-cloud students** (`distill/distill_pc.py`): a `POLICIES` entry whose checkpoint file is named
`student_*.pth` runs through `distill/student_player.py`. The "Point cloud" folder then draws what the
network reads -- the cloud by label or by source camera, the two cameras, the auxiliary head's keypoint
and goal estimates against the truth -- over the true sheet (made translucent), and the "Point-cloud
noise" folder switches the camera errors of the pointcloud DR block (new per-episode draws, so it resets the envs).

**Goal tasks** (`fold_pool.goal_task`, e.g. `LIVE_VISER_EXPERIMENT=rigidcloth_pc_any_fold`): a green
ghost shows the goal state the fold criterion matches best now (the task's own goal, or a look-alike
with `accept_lookalike`), for any policy; the HUD names the task and both errors.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaacsimenvs.scripts.evaluation.protocol import disable_randomization

ROOT = Path(__file__).resolve().parents[4]

#: Display name -> (checkpoint, default SAPG coef). Paths are relative to the repo root; more come
#: from $LIVE_VISER_POLICIES (below).
POLICIES = {
    "Play prior nov8 (right Wuji; untrained on folds)": (
        "pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth", 0.0),
    "rigid_cloth_right_wuji s1 ep1500 (flat7RW_base)": (
        "outputs/2026-11-11/02-51-52/0_pxpool_flat7RW_base_s1_277681/nn/"
        "snap_0_pxpool_flat7RW_base_s1_277681_ep_1500.pth", 0.0),
}

#: Extra menu entries from a JSON file {display name: [checkpoint, coef(, [task menu names])]} named by
#: $LIVE_VISER_POLICIES (survives the browser's restart, which re-execs with the same environment). With the
#: optional third element the policy is listed only while one of those $LIVE_VISER_TASKS entries is loaded.
POLICY_TASKS: dict[str, set[str]] = {}
if os.environ.get("LIVE_VISER_POLICIES"):
    import json

    with open(os.environ["LIVE_VISER_POLICIES"]) as _f:
        _extra = json.load(_f)
    # $LIVE_VISER_POLICIES_ONLY=1: the menu is the JSON's entries alone (no built-ins).
    if os.environ.get("LIVE_VISER_POLICIES_ONLY") == "1":
        POLICIES.clear()
    for _k, _v in _extra.items():
        POLICIES[_k] = (_v[0], float(_v[1]))
        if len(_v) > 2:
            POLICY_TASKS[_k] = set(_v[2])


def policies_for(task_name: str) -> list[str]:
    """Menu entries usable with the task menu entry `task_name` (all of them if none is task-bound)."""
    return [k for k in POLICIES if k not in POLICY_TASKS or task_name in POLICY_TASKS[k]] or list(POLICIES)

#: Display name -> (gym task, extra hydra overrides). Newton; PHYSX_TASKS for the PhysX viewer.
TASKS = {
    "VBD cloth, Flexiv + right Wuji (real setup)": ("Isaacsimenvs-Cloth-Direct-v0", []),
    "VBD cloth, Flexiv + right Wuji identified hand (real setup)": ("Isaacsimenvs-Cloth-Direct-v0", ["env.hand_sysid=true"]),
}
_POOL7 = ["env.fold_pool.shapes=[box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid]",
          "env.fold_pool.one_shape_per_env=true", "env.fold_pool.init_fold_prob=0.0"]
PHYSX_TASKS = {
    "fold pool 7 shapes flat, Flexiv + right Wuji (rigid_cloth_right_wuji)": (
        "Isaacsimenvs-RigidCloth-Direct-v0", ["env.cloth.start_height=0.20", *_POOL7],
    ),
    "fold pool (5 shapes, 10-20 cm), Flexiv + right Wuji": (
        "Isaacsimenvs-RigidCloth-Direct-v0", ["env.cloth.start_height=0.20"],
    ),
}
#: VBD cloth size -> hydra overrides (the large-cloth evals' settings):
#: grid spacing closest to the calibrated 16.67 mm (and >= the 16 mm particle thickness), reset centre
#: moved toward the robot by (size - 0.10) * sqrt(2) / 2 like the pool's reach_shift.
CLOTH_SIZES = {
    "10 cm (standard)": [],
    "12.5 cm": ["env.cloth.size=0.125", "env.cloth.resolution=8", "env.cloth.reset_offset_xy=[0.0,0.0177]"],
    "15 cm": ["env.cloth.size=0.15", "env.cloth.resolution=10", "env.cloth.reset_offset_xy=[0.0,0.0354]"],
    "17.5 cm": ["env.cloth.size=0.175", "env.cloth.resolution=11", "env.cloth.reset_offset_xy=[0.0,0.0530]"],
    "20 cm": ["env.cloth.size=0.2", "env.cloth.resolution=13", "env.cloth.reset_offset_xy=[0.0,0.0707]"],
}
_TASK_KEYS = ("env.rigid_cloth.variant=", "env.newton.per_env_triangle_pairs=",
              "env.fold_pool.init_fold_prob=", "env.rigid_cloth.joint_damping=",
              "env.cloth.size=", "env.cloth.resolution=", "env.cloth.reset_offset_xy=")
#: PhysX has no friction DR (a tensor material write resets the combine modes; backends/rigid/physx.py).
_PHYSX_NO = ("slat_friction_scale", "robot_friction_scale")

#: Physics sliders for rigid manipulands: (label, domain_randomization.cloth field, lo, hi, step, nominal).
#: Each is written as a degenerate range [v, v], so the env's own DR draw returns exactly v.
PHYSICS = (
    ("slat friction x", "slat_friction_scale", 0.2, 3.0, 0.05, 1.0),
    ("robot friction x", "robot_friction_scale", 0.2, 3.0, 0.05, 1.0),
    ("hinge damping x", "hinge_damping_scale", 0.5, 10.0, 0.1, 1.0),
    ("hinge stiffness [1e-4 N m/rad]", "hinge_stiffness", 0.0, 10.0, 0.1, 0.0),
    ("contact margin offset [mm]", "contact_margin_offset", -5.0, 5.0, 0.25, 0.0),
    ("mass x", "mass_scale", 0.3, 3.0, 0.05, 1.0),
)
_UNIT = {"hinge_stiffness": 1e-4, "contact_margin_offset": 1e-3}

OUTCOMES = ("fold_held", "fold", "fall", "hand_far", "nonfinite", "timeout")

ROBOT_RGB = (205, 208, 214)
TABLE_RGB = (96, 112, 138)
GOAL_RGB = (40, 200, 90)
KP_RGB = (255, 165, 25)
MOVING_RGB = (242, 77, 26)
STATIONARY_RGB = (250, 199, 38)
#: Student point cloud, by label (``PointCloudSetup.labels``): live points, first-frame reference
#: moving / stationary (goal label), the goal sheet (goal_fixed / goal_any), forward kinematics.
PC_LABEL_RGB = {"live": (245, 245, 245), "ref_moving": (255, 150, 200), "ref_stationary": (140, 195, 255),
                "goal": (255, 150, 200), "fk": (40, 40, 40)}
PC_CAM_RGB = np.array([(0, 190, 215), (235, 60, 200)], np.uint8)   # camera A, B
PC_NO_CAM_RGB = np.array((120, 120, 120), np.uint8)
PC_AUX_KP_RGB, PC_AUX_GOAL_RGB = (255, 0, 200), (0, 120, 110)
PC_NOISE_LABELS = {
    "cams": "camera position +-5 cm",
    "calib": "calibration: each camera +-1.5 cm, +-2 deg",
    "depth": "depth noise 1-4 mm",
    "drop": "random drop 0-30%",
}


def _parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
    p.add_argument("--policy", default=next(iter(POLICIES)), choices=list(POLICIES))
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--even_sizes", type=int, default=1,
                   help="fold pool: space the per-env sheet sizes evenly over size_range instead of "
                        "drawing them, so env k is always the k-th size")
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--seed", type=int, default=0)
    return p


# ---------------------------------------------------------------------------------------------
# geometry: Newton's viewer pipeline, captured


def _capture_geometry(model):
    """Every drawable shape as (body, world, mesh vertices in the BODY frame, faces)."""
    import newton
    from newton.viewer import ViewerNull

    class Capture(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.meshes = {}

        def log_mesh(self, name, points, indices, *args, **kwargs):
            pts = points.numpy() if hasattr(points, "numpy") else np.asarray(points)
            idx = indices.numpy() if hasattr(indices, "numpy") else np.asarray(indices)
            self.meshes[name] = (np.asarray(pts, np.float32).reshape(-1, 3),
                                 np.asarray(idx, np.int32).reshape(-1, 3))

    v = Capture()
    v.show_collision = True  # the robot's shapes are collision-only (see render_newton.py)
    v.show_static = True
    v.set_model(model)

    labels = [str(x) for x in model.body_label]
    out = []
    for batch in v._shape_instances.values():
        if not v._should_show_shape(batch.flags, batch.static, batch.geo_type):
            continue
        if int(batch.geo_type) == int(newton.GeoType.PLANE) or batch.mesh not in v.meshes:
            continue
        verts, faces = v.meshes[batch.mesh]
        parents = batch.parents.numpy()
        xforms = batch.xforms.numpy()
        scales = batch.scales.numpy()
        worlds = batch.worlds.numpy()
        for parent, xf, sc, world in zip(parents, xforms, scales, worlds):
            label = labels[parent] if parent >= 0 else ""
            if "GoalViz" in label:
                continue
            out.append(SimpleNamespace(
                body=int(parent), world=int(world), label=label,
                verts=verts * sc[None, :], faces=faces,
                pos=xf[:3].astype(np.float64), wxyz=np.array([xf[6], xf[3], xf[4], xf[5]]),
            ))
    return out


# ---------------------------------------------------------------------------------------------
# geometry: the USD stage, for PhysX


def _prim_tris(prim):
    """(vertices, triangles) of a UsdGeom mesh or implicit shape, in the prim's own frame; None if
    the prim is not drawable geometry."""
    import trimesh
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if pts is None or counts is None or idx is None or len(pts) == 0:
            return None
        idx = np.asarray(idx, np.int64)
        tris, o = [], 0
        for c in counts:
            for k in range(1, int(c) - 1):
                tris.append((idx[o], idx[o + k], idx[o + k + 1]))
            o += int(c)
        return np.asarray(pts, np.float64), np.asarray(tris, np.int32).reshape(-1, 3)
    tm = None
    if prim.IsA(UsdGeom.Cube):
        e = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        tm = trimesh.creation.box(extents=(e, e, e))
    elif prim.IsA(UsdGeom.Sphere):
        tm = trimesh.creation.icosphere(subdivisions=2, radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
    elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
        g = UsdGeom.Capsule(prim) if prim.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(prim)
        r, h = float(g.GetRadiusAttr().Get()), float(g.GetHeightAttr().Get())
        tm = (trimesh.creation.capsule(height=h, radius=r) if prim.IsA(UsdGeom.Capsule)
              else trimesh.creation.cylinder(radius=r, height=h))
        axis = str(g.GetAxisAttr().Get() or "Z")
        if axis == "X":
            tm.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
        elif axis == "Y":
            tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    if tm is None:
        return None
    return np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int32)


class _PhysxScene:
    """Drawable geometry and body poses of a PhysX env, per Isaac Lab asset.

    Shapes come from the USD stage: under each asset's prim in env ``w``, every collision geometry
    prim, expressed in the frame of its nearest rigid-body ancestor (so the per-step pose is that
    body's). Each env maps to the asset's instance through the PhysX view's own prim paths -- an
    articulation that only some envs carry (``fold_pool.one_shape_per_env``) indexes them 0..n-1.
    """

    SKIP = ("object", "goal_viz")

    def __init__(self, inner):
        import re

        self.assets = []  # (name, asset, root prim name, {env: instance}, body names)
        reg = {**dict(inner.scene.articulations), **dict(inner.scene.rigid_objects)}
        for name, asset in reg.items():
            if name in self.SKIP:
                continue
            root = asset.cfg.prim_path.rsplit("/", 1)[-1]
            inst = {}
            for i, path in enumerate(asset.root_physx_view.prim_paths):
                m = re.search(r"/env_(\d+)/", path)
                if m:
                    inst[int(m.group(1))] = i
            self.assets.append((name, asset, root, inst, list(asset.body_names)))
        self._cache = {}

    def shapes(self, w: int):
        if w in self._cache:
            return self._cache[w]
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        out, missing = [], set()
        for k, (name, _asset, root, inst, bodies) in enumerate(self.assets):
            if w not in inst:
                continue
            top = stage.GetPrimAtPath(f"/World/envs/env_{w}/{root}")
            if not top.IsValid():
                continue
            found = {"col": [], "vis": []}
            for prim in Usd.PrimRange(top, Usd.TraverseInstanceProxies()):
                if not (prim.IsA(UsdGeom.Gprim)):
                    continue
                body, collider, p = None, False, prim
                while p.IsValid() and p != top.GetParent():
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        en = p.GetAttribute("physics:collisionEnabled")
                        collider = collider or not (en and en.HasAuthoredValue() and not en.Get())
                    if p.HasAPI(UsdPhysics.RigidBodyAPI):
                        body = p
                        break
                    p = p.GetParent()
                if body is None:
                    continue
                if body.GetName() not in bodies:
                    missing.add(body.GetName())
                    continue
                geo = _prim_tris(prim)
                if geo is None:
                    continue
                v, f = geo
                rel = np.array(xc.GetLocalToWorldTransform(prim) * xc.GetLocalToWorldTransform(body).GetInverse())
                v = (np.c_[v, np.ones(len(v))] @ rel)[:, :3]
                found["col" if collider else "vis"].append(SimpleNamespace(
                    key=(k, bodies.index(body.GetName())), label=f"{root}/{body.GetName()}", world=w,
                    verts=v.astype(np.float32), faces=f, pos=np.zeros(3), wxyz=np.array([1.0, 0, 0, 0]),
                ))
            # Collision geometry is what the solver sees; an asset without any (none here) falls back
            # to its visuals so it is not silently missing.
            out += found["col"] or found["vis"]
        if missing:
            print(f"[live] PhysX bodies with no tensor index (not drawn): {sorted(missing)[:8]}", flush=True)
        self._cache[w] = out
        return out

    def poses(self, w: int, keys):
        """``{key: (position, wxyz)}`` in the world frame for ``keys`` of env ``w``."""
        out = {}
        by_asset = {}
        for k, b in keys:
            by_asset.setdefault(k, []).append(b)
        for k, links in by_asset.items():
            _name, asset, _root, inst, _b = self.assets[k]
            i = inst[w]
            pos = asset.data.body_pos_w[i, links].detach().cpu().numpy()
            quat = asset.data.body_quat_w[i, links].detach().cpu().numpy()  # wxyz (Isaac Lab 2.3)
            for b, p, q in zip(links, pos, quat):
                out[(k, b)] = (p, q)
        return out


# ---------------------------------------------------------------------------------------------


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = _parse()
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    os.chdir(ROOT)
    # PhysX = the rigid backend (`RigidCloth`). Read from the id's own spelling: the gym registry is not imported until after the app starts.
    physx = args.task.startswith("Isaacsimenvs-RigidCloth")
    tasks = PHYSX_TASKS if physx else TASKS
    task_keys = _TASK_KEYS
    # $LIVE_VISER_TASKS: a JSON {display name: [gym task, [hydra overrides]]} that replaces the task menu.
    # Every key its entries set counts as a task choice, so a restart drops the old task's value.
    if os.environ.get("LIVE_VISER_TASKS"):
        import json

        with open(os.environ["LIVE_VISER_TASKS"]) as f:
            tasks = {k: (t, list(ov)) for k, (t, ov) in json.load(f).items()}
        task_keys = _TASK_KEYS + tuple(sorted({o.split("=")[0] + "=" for _t, ov in tasks.values() for o in ov}))
    if physx:
        args.headless = True
    if not str(getattr(args, "device", "") or "").startswith("cuda"):
        args.device = "cuda:0"  # AppLauncher owns --device
    try:
        AppLauncher(args)
    except ImportError as exc:  # kit-less Newton backend, exactly as in episodes.py
        print(f"[live] running kit-less: {exc}", flush=True)

    import gymnasium as gym
    import torch
    import viser

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    status = server.gui.add_markdown(f"**Building `{args.task}` with {args.num_envs} envs...** (~2-3 min)")

    # $LIVE_VISER_EXPERIMENT: build the env from a run recipe (as train.py --experiment does).
    experiment = os.environ.get("LIVE_VISER_EXPERIMENT") or None
    if experiment:
        print(f"[live] experiment recipe {experiment}", flush=True)

    # A recipe carries an agent block too, so it needs the agent cfg loaded (unused here).
    @hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point" if experiment else "", experiment=experiment)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        env_cfg.termination.success_steps = 10  # a fold must be HELD, as in every eval
        env_cfg.cloth.nan_policy = "reset"
        rigid = hasattr(env_cfg, "rigid_cloth")
        if not rigid and not any(a.startswith("env.cloth.use_cuda_graph=") for a in hydra_args):
            # VBD is kernel-launch bound at a handful of envs: 4.3 -> 20.6 policy steps/s with the
            # coupled step captured as a CUDA graph, and the same folds (reference 64/64 with it,
            # 32/32 without; scripts/cluster/bos14_cloth_cuda_graph.sh). Pass the override to opt out.
            env_cfg.cloth.use_cuda_graph = True
        if rigid:
            # Build hinge actuators with a position gain (1e-9 N m/rad) so the stiffness slider can
            # reach them: Newton fixes the actuator TYPE at build time (rigid_cloth_env.py).
            rd = env_cfg.domain_randomization.cloth
            if float(rd.hinge_stiffness[1]) <= 0.0:
                rd.hinge_stiffness = [0.0, 1e-3]
        # Keep the training reset distribution's numbers, then pin; the GUI swaps between them.
        rs = env_cfg.reset
        reset_rand = {k: getattr(rs, k) for k in (
            "reset_position_noise_x", "reset_position_noise_y", "reset_position_noise_z",
            "reset_dof_pos_random_interval_arm", "reset_dof_pos_random_interval_fingers",
            "reset_dof_vel_random_interval", "table_reset_z_range", "fixed_start_pose")}
        # $LIVE_VISER_KEEP_DR=1: keep the task's domain randomisation (delays, noise, gains); the start
        # pose is still pinned / training via the GUI toggle below.
        keep_dr = os.environ.get("LIVE_VISER_KEEP_DR") == "1"
        dr_keep = copy.deepcopy(env_cfg.domain_randomization) if keep_dr else None
        disable_randomization(env_cfg)
        if keep_dr:
            env_cfg.domain_randomization = dr_keep
            print("[live] LIVE_VISER_KEEP_DR=1: domain randomisation left ON", flush=True)
        reset_pin = {k: getattr(rs, k) for k in reset_rand}

        if args.even_sizes and hasattr(env_cfg, "fold_pool"):
            # The per-env size draw is backends/rigid/scene.draw_env_sizes_raw.
            from isaacsimenvs.tasks.cloth.backends.rigid import scene as fpe

            class _Even:
                def uniform(self, lo, hi, n):
                    return np.linspace(lo, hi, n) if n > 1 else np.full(n, 0.5 * (lo + hi))

            real_np = fpe.np
            fpe.np = SimpleNamespace(**{k: getattr(real_np, k) for k in dir(real_np) if not k.startswith("__")})
            fpe.np.random = SimpleNamespace(default_rng=lambda seed: _Even())
        try:
            env = gym.make(args.task, cfg=env_cfg)
        finally:
            if args.even_sizes and hasattr(env_cfg, "fold_pool"):
                fpe.np = real_np
        inner = env.unwrapped
        n_env, device = inner.num_envs, inner.device

        origins = inner.scene.env_origins.detach().cpu().numpy()
        # The rigid backend's fold-shape pool (None on VBD).
        rb = inner.backend if hasattr(inner.backend, "pool") else None
        pool = [sh.name for sh in rb.shapes] if rb is not None else []
        # One shape per env (PhysX ``fold_pool.one_shape_per_env``): env w can only run the shapes
        # with carried[shape, w]. None when every env carries every shape.
        carried = rb.pool.carried if rb is not None else None
        carried = None if carried is None or bool(carried.all()) else carried.cpu().numpy()
        sizes = getattr(inner.backend, "env_size_np", None)

        if physx:
            px = _PhysxScene(inner)
            model = None

            def shapes_of(w):
                return px.shapes(w)

            # Manipuland bodies (neither robot nor table), coloured by a hue sweep over env 0's.
            manip = sorted({s.label.split("/", 1)[-1] for s in px.shapes(0)
                            if not s.label.startswith(("Robot", "Table"))})
            manip_rank = {m: i / max(len(manip), 1) for i, m in enumerate(manip)}

            def tail(label):
                return label.split("/", 1)[-1]
        else:
            import warp as wp
            from isaaclab_newton.physics.newton_manager import NewtonManager

            model = NewtonManager.get_model()
            shapes = _capture_geometry(model)
            for s in shapes:
                s.key = s.body

            def shapes_of(w):
                return [s for s in shapes if s.world in (-1, w)]

            # Manipuland bodies: anything that is neither robot nor table, coloured by a hue sweep.
            manip = sorted({s.body for s in shapes if s.body >= 0 and s.world == 0
                            and "Robot" not in s.label and "Table" not in s.label})
            hue = {b: i / max(len(manip), 1) for i, b in enumerate(manip)}
            manip_rank = {}  # body label tail -> hue, shared by every world
            for b in manip:
                manip_rank[model.body_label[b].split("/", 4)[-1]] = hue[b]

            def tail(label):
                return label.split("/", 4)[-1]

        def colour(s):
            if "Robot" in s.label:
                return ROBOT_RGB
            if "Table" in s.label or s.key is None or (not physx and s.key < 0):
                return TABLE_RGB
            h = manip_rank.get(tail(s.label), 0.0)
            r, g, b = colorsys.hsv_to_rgb(0.02 + 0.75 * h, 0.8, 0.95)
            return (int(r * 255), int(g * 255), int(b * 255))

        # Cloth: per-world particles and triangles, split into the two halves.
        tri_by_world = {}
        if model is not None and getattr(model, "tri_count", 0) and getattr(model, "particle_world", None) is not None:
            owner = model.particle_world.numpy()
            tri = model.tri_indices.numpy()
            mv = set(inner.task.geometry.moving_idx.tolist())
            for w in range(n_env):
                pidx = np.nonzero(owner == w)[0]
                base = int(pidx[0])
                t = tri[(owner[tri] == w).all(axis=1)] - base
                in_mv = np.isin(t, list(mv)).sum(axis=1) >= 2
                tri_by_world[w] = (pidx, t[in_mv], t[~in_mv])

        # Fingers through the cloth (VBD only: particles are its sole contact with the hand).
        pierce = None
        if tri_by_world:
            from isaacsimenvs.tasks.cloth.backends.vbd.pierce import PierceDetector

            pierce = PierceDetector(model, n_env, device, robot_spec=getattr(inner, "robot_spec", None))
            pierce_run = torch.zeros(n_env, len(pierce.names), dtype=torch.long, device=device)
        last_pierce: list = []

        # ------------------------------------------------------------------ shared state
        S = SimpleNamespace(
            playing=True, step_once=False, speed=1.0, world=0, deterministic=True,
            policy_name=args.policy, player=None, coef=POLICIES[args.policy][1],
            reset_all=False, reset_one=False, shape_choice=["random"] * n_env,
            stats=[{"episodes": 0, "folds": 0, "last": "-"} for _ in range(n_env)],
            lock=threading.Lock(), rebuild=True, restart=None, physics_dirty=False,
            fps=0.0, fold_choice=["random"] * n_env, angle_choice=[None] * n_env, pause_at=0,
            noise_sync=False,
        )
        # Partly folded starts (fold pool with fold_pool.init_fold_prob > 0): per shape, its start
        # folds in the order of the env's (sign-calibrated) direction table.
        start_tabs = rb.start_tables if rb is not None else None
        fold_names = list(rb.start_names) if start_tabs is not None else []
        # One FIXED option list ("<shape>: <fold>"): changing a dropdown's options while switching envs
        # let the browser echo a stale value back over the env's choice.
        fold_label = {f"{pool[s]}: {f}": (s, k) for s, names in enumerate(fold_names) for k, f in enumerate(names)}
        players: dict[str, PretrainedPlayer] = {}

        def get_player(name):
            # point-cloud students: distilled (student_*.pth) or RL-trained point-cloud actors (*_pcrl_*)
            if name not in players and (Path(POLICIES[name][0]).name.startswith("student_")
                                        or "_pcrl_" in Path(POLICIES[name][0]).name):
                from isaacsimenvs.scripts.distillation.student_player import StudentPlayer

                players[name] = StudentPlayer(inner, POLICIES[name][0], device)
            if name not in players:
                ckpt, _ = POLICIES[name]
                players[name] = PretrainedPlayer(
                    config_path=args.policy_config, checkpoint_path=ckpt, num_envs=n_env,
                    device=args.device, num_observations=int(inner.cfg.observation_space),
                    num_actions=int(inner.cfg.action_space), expl_coef=S.coef,
                )
            return players[name]

        def stu():
            """The active player if it is a point-cloud student, else None."""
            return S.player if getattr(S.player, "is_student", False) else None

        # Fold pool: honour the GUI's per-env shape at each reset.
        if pool:
            orig_draw = rb.draw_episode

            def draw(env_ids):
                orig_draw(env_ids)  # random draw first, then override the pinned envs
                forced = [(int(e), pool.index(S.shape_choice[int(e)])) for e in env_ids.tolist()
                          if S.shape_choice[int(e)] != "random"]
                if carried is not None:  # "apply to ALL" / layout may name shapes an env lacks
                    forced = [(e, k) for e, k in forced if carried[k, e]]
                if forced:
                    ids = torch.tensor([e for e, _ in forced], device=device)
                    shp = torch.tensor([k for _, k in forced], device=device)
                    rb.pool.select(ids, shp)
                    if start_tabs is not None:  # the random draw drew start folds for other shapes
                        rb.draw_start_folds(ids, shp)
                if start_tabs is not None:
                    for e in env_ids.tolist():
                        fc, ang = S.fold_choice[e], S.angle_choice[e]
                        if fc == "random" and ang is None:
                            continue
                        s = int(rb.pool.active[e])
                        tab = start_tabs[s]
                        if fc == "flat":
                            q, folded = torch.zeros(tab.shape[1], device=device), False
                        else:
                            if fc in fold_label:
                                fs_, k = fold_label[fc]
                                if fs_ != s:
                                    continue  # (the shape choice pins it; this cannot normally happen)
                            elif bool(rb.start_folded[e]):
                                k = start_info(e)[1]  # keep the drawn fold, change only its angle
                            else:
                                continue  # a flat draw and no fold named: stays flat
                            if ang is not None:
                                a = torch.full((rb.start_num_parts,), math.radians(ang), device=device)
                            else:  # each part of the start its own angle, as the training draw
                                a = torch.deg2rad(torch.tensor(np.random.uniform(
                                    *inner.cfg.fold_pool.init_fold_deg, size=rb.start_num_parts),
                                    device=device, dtype=torch.float32))
                            q, folded = tab[k] * a[rb.start_parts[s][k]], True
                        rb.pool.set_start_joints(s, torch.tensor([e], device=device), q.unsqueeze(0))
                        rb.start_folded[e] = folded

            rb.draw_episode = draw

        def start_info(e):
            """(shape index, start-fold index or None, angle [deg]) env ``e`` started its episode at."""
            s = int(rb.pool.active[e])
            if start_tabs is None or not bool(rb.start_folded[e]) or s not in rb.pool._start_q:
                return s, None, 0.0
            q = rb.pool._start_q[s][e]
            tab = start_tabs[s]
            cos = (tab @ q) / (tab.norm(dim=1) * q.norm()).clamp(min=1e-9)
            return s, int(cos.argmax()), math.degrees(float(q.abs().max()))

        rd_cfg = inner.cfg.domain_randomization.cloth if rigid else None
        rd_train = {f: list(getattr(rd_cfg, f)) for _, f, *_ in PHYSICS} if rigid else {}

        # ------------------------------------------------------------------ GUI
        status.content = "**Ready.**"
        task_names = list(tasks)
        matches = [k for k, (t, ov) in tasks.items() if t == args.task and all(o in hydra_args for o in ov)]
        cur_task = max(matches, key=lambda k: len(tasks[k][1])) if matches else task_names[0]

        with server.gui.add_folder("Scene"):
            g_task = server.gui.add_dropdown("task", task_names, initial_value=cur_task)
            g_csize = None
            if not physx:
                size_now = next((k for k, ov in CLOTH_SIZES.items() if ov and all(o in hydra_args for o in ov)),
                                "10 cm (standard)")
                g_csize = server.gui.add_dropdown("VBD cloth size (on restart)", list(CLOTH_SIZES),
                                                  initial_value=size_now)
            g_nenv = server.gui.add_number("num envs", args.num_envs, min=1, max=64, step=1)
            g_restart = server.gui.add_button("Restart with this task (~2-3 min)")

            def env_label(w):
                s = f"env {w}"
                if sizes is not None:
                    s += f" ({sizes[w] * 100:.1f} cm)"
                if carried is not None:
                    s += " (" + "/".join(pool[k] for k in np.flatnonzero(carried[:, w])) + ")"
                return s

            g_env = server.gui.add_dropdown("show env", [env_label(w) for w in range(n_env)])
            g_shape = server.gui.add_dropdown("shape at next reset", ["random", *pool]) if pool else None
            g_shape_all = server.gui.add_button("apply shape to ALL envs") if pool else None
            g_start = server.gui.add_dropdown("start pose", ["pinned", "randomised (training)"])
            g_tol = server.gui.add_slider("fold tolerance [mm]", 5.0, 60.0, 0.5,
                                          float(inner.cfg.cloth.keypoint_tolerance) * 1e3)
            g_reset = server.gui.add_button("reset this env")
            g_reset_all = server.gui.add_button("reset all envs")
            g_kp = server.gui.add_checkbox("show keypoints / targets", True)

        with server.gui.add_folder("Policy"):
            g_pol = server.gui.add_dropdown("policy", policies_for(cur_task), initial_value=args.policy)
            g_coef = server.gui.add_slider("SAPG coef (0 = block 5, 50 = block 0)", 0.0, 50.0, 10.0, S.coef)
            g_det = server.gui.add_checkbox("deterministic actions", True)
            g_play = server.gui.add_checkbox("playing", True)
            g_step = server.gui.add_button("single step")
            g_speed = server.gui.add_slider("speed (x real time; 4 = as fast as possible)", 0.1, 4.0, 0.1, 1.0)

        with server.gui.add_folder("Point cloud (student policies)"):
            g_pc_col = server.gui.add_dropdown("colour points by", ["label", "camera"])
            g_pc_show = {
                "live": server.gui.add_checkbox("live points (what the cameras see: cloth, hand, arm)", True),
                "fk": server.gui.add_checkbox("forward-kinematics points (forearm, hand)", True),
                "ref": server.gui.add_checkbox("goal cloud / first-frame reference cloud", True),
            }
            g_pc_ghost = server.gui.add_checkbox("goal tasks: matched goal state (green ghost)", True)
            g_pc_aux = server.gui.add_checkbox("aux head: estimated keypoints + goal", False)
            g_pc_cams = server.gui.add_checkbox("cameras", True)
            g_pc_true = server.gui.add_checkbox("true sheet particles (grey)", False)
            g_opac = server.gui.add_slider("true sheet opacity", 0.05, 1.0, 0.05, 0.4)
            g_pc_size = server.gui.add_slider("point size [mm]", 1.0, 10.0, 0.5, 4.0)
            server.gui.add_markdown(
                "by label: white = live points (cloth + hand + arm), black = forward kinematics, light pink = "
                "goal cloud as the student gets it (goal_fixed: folded sheet in place; goal_any: bare shape "
                "about the box centre, random yaw; label: light pink / light blue = moving / stationary); "
                "green ghost = the goal state the fold criterion matches best now.  \n"
                "by camera: cyan = camera A, magenta = camera B, grey = no camera.  \n"
                "aux head: magenta = estimated keypoints, teal = estimated goal, lines go to the truth "
                "(orange / green, 'show keypoints / targets').")
        with server.gui.add_folder("Point-cloud noise (student; a change resets all envs)"):
            g_noise = {k: server.gui.add_checkbox(label, False) for k, label in PC_NOISE_LABELS.items()}
            g_noise_trained = server.gui.add_button("as the student was trained")
            g_noise_clean = server.gui.add_button("clean (all off)")
            g_noise_c = server.gui.add_button("version C (all on)")

        g_fold = g_ang = g_ang_rand = g_prob = g_layout = g_pause = None
        if start_tabs is not None:
            with server.gui.add_folder("Folded starts", expand_by_default=True):
                g_prob = server.gui.add_slider("folded-start probability (random draws)", 0.0, 1.0, 0.05,
                                               float(inner.cfg.fold_pool.init_fold_prob))
                g_fold = server.gui.add_dropdown("start fold at next reset (this env; pins the shape)",
                                                 ["random", "flat", *fold_label])
                lo_deg, hi_deg = (float(v) for v in inner.cfg.fold_pool.init_fold_deg)
                g_ang_rand = server.gui.add_checkbox(f"random angle ({lo_deg:g}-{hi_deg:g} deg)", True)
                g_ang = server.gui.add_slider("start angle [deg]", 0.0, 180.0, 1.0, 90.0)
                g_layout = server.gui.add_button(
                    f"lay out ALL {sum(len(f) for f in fold_names)} start folds, one per env, at the start angle (pauses)")
                g_pause = server.gui.add_slider("pause at episode step (0 = never; ~12 = landed)", 0, 60, 1, 0)

        g_phys = {}
        if rigid:
            with server.gui.add_folder("Physics (rigid manipuland)"):
                g_dr = server.gui.add_checkbox("training DR ranges (ignore sliders)", False)
                for label, field, lo, hi, step, nom in PHYSICS:
                    if physx and field in _PHYSX_NO:
                        continue
                    g_phys[field] = server.gui.add_slider(label, lo, hi, step, nom)
                g_apply = server.gui.add_button("apply now (all envs)")
                g_nominal = server.gui.add_button("back to nominal")

        with server.gui.add_folder("Status", expand_by_default=True):
            hud = server.gui.add_markdown("")

        # ------------------------------------------------------------------ callbacks
        @g_restart.on_click
        def _(_e):
            task, ov = tasks[g_task.value]
            if g_csize is not None and task.startswith("Isaacsimenvs-Cloth"):
                ov = [*ov, *CLOTH_SIZES[g_csize.value]]
            ok = policies_for(g_task.value)  # the next task's menu may not list the current policy
            S.restart = (task, ov, int(g_nenv.value), S.policy_name if S.policy_name in ok else ok[0])

        @g_env.on_update
        def _(_e):
            S.world = [env_label(w) for w in range(n_env)].index(g_env.value)
            if g_shape is not None:
                g_shape.value = S.shape_choice[S.world]
            sync_fold_widgets()
            S.rebuild = True

        def sync_fold_widgets():
            if g_fold is None:
                return
            g_fold.value = S.fold_choice[S.world]
            a = S.angle_choice[S.world]
            g_ang_rand.value = a is None
            if a is not None:
                g_ang.value = a

        if g_shape is not None:
            @g_shape.on_update
            def _(_e):
                v = g_shape.value
                if carried is not None and v != "random" and not carried[pool.index(v), S.world]:
                    # This env cannot run that shape: show the first env that carries it instead.
                    w2 = int(np.flatnonzero(carried[pool.index(v)])[0])
                    S.shape_choice[w2] = v
                    g_env.value = env_label(w2)
                    return
                S.shape_choice[S.world] = v
                fc = S.fold_choice[S.world]
                if fc in fold_label and pool[fold_label[fc][0]] != g_shape.value:
                    S.fold_choice[S.world] = "random"  # a fold of another shape no longer applies
                    sync_fold_widgets()

            @g_shape_all.on_click
            def _(_e):
                S.shape_choice = [g_shape.value] * n_env

        if g_fold is not None:
            @g_prob.on_update
            def _(_e):
                inner.cfg.fold_pool.init_fold_prob = float(g_prob.value)

            @g_fold.on_update
            def _(_e):
                S.fold_choice[S.world] = g_fold.value
                if g_fold.value in fold_label:
                    shp = pool[fold_label[g_fold.value][0]]
                    S.shape_choice[S.world] = shp
                    if g_shape.value != shp:
                        g_shape.value = shp

            @g_ang_rand.on_update
            def _(_e):
                S.angle_choice[S.world] = None if g_ang_rand.value else float(g_ang.value)

            @g_ang.on_update
            def _(_e):
                if not g_ang_rand.value:
                    S.angle_choice[S.world] = float(g_ang.value)

            @g_layout.on_click
            def _(_e):
                pairs = [(pool[s], f"{pool[s]}: {f}") for s in range(len(pool)) for f in fold_names[s]]
                ang = float(g_ang.value)  # every layout env at the "start angle" slider's value
                g_ang_rand.value = False
                for w in range(n_env):
                    shp, f = pairs[w % len(pairs)]
                    S.shape_choice[w], S.fold_choice[w], S.angle_choice[w] = shp, f, ang
                if n_env < len(pairs):
                    status.content = (f"**{n_env} envs show the first {n_env} of {len(pairs)} start folds** "
                                      f"-- restart with num envs >= {len(pairs)} to see all")
                if g_pause.value == 0:
                    g_pause.value = 12
                S.pause_at = int(g_pause.value)
                sync_fold_widgets()
                S.reset_all = True
                S.playing = True
                g_play.value = True

            @g_pause.on_update
            def _(_e):
                S.pause_at = int(g_pause.value)

        @g_start.on_update
        def _(_e):
            src = reset_pin if g_start.value == "pinned" else reset_rand
            for k, v in src.items():
                setattr(inner.cfg.reset, k, v)

        @g_tol.on_update
        def _(_e):
            inner.cfg.cloth.keypoint_tolerance = float(g_tol.value) * 1e-3

        @g_reset.on_click
        def _(_e):
            S.reset_one = True

        @g_reset_all.on_click
        def _(_e):
            S.reset_all = True

        @g_pol.on_update
        def _(_e):
            S.policy_name = g_pol.value
            g_coef.value = POLICIES[g_pol.value][1]
            S.reset_all = True

        def apply_noise():
            """Checkboxes -> the active student's point-cloud cfg (per-episode draws: needs a reset)."""
            pl = stu()
            if pl is None or S.noise_sync:
                return
            if pl.set_noise({k: bool(g.value) for k, g in g_noise.items()}):
                S.reset_all = True

        def set_noise_gui(values):
            S.noise_sync = True
            for k, v in values.items():
                g_noise[k].value = bool(v)
            S.noise_sync = False
            apply_noise()

        def sync_noise_gui():
            """Show the active student's current noise switches (no change to its cfg)."""
            pl = stu()
            if pl is not None:
                S.noise_sync = True
                for k, v in pl.noise_state().items():
                    g_noise[k].value = bool(v)
                S.noise_sync = False

        for _g in g_noise.values():
            _g.on_update(lambda _e: apply_noise())

        @g_noise_trained.on_click
        def _(_e):
            pl = stu()
            if pl is not None:
                set_noise_gui(pl.trained_noise)

        @g_noise_clean.on_click
        def _(_e):
            set_noise_gui({k: False for k in g_noise})

        @g_noise_c.on_click
        def _(_e):
            set_noise_gui({k: True for k in g_noise})

        @g_opac.on_update
        def _(_e):
            S.rebuild = True  # rigid plates carry their opacity from the build

        @g_coef.on_update
        def _(_e):
            S.coef = float(g_coef.value)

        @g_det.on_update
        def _(_e):
            S.deterministic = bool(g_det.value)

        @g_play.on_update
        def _(_e):
            S.playing = bool(g_play.value)

        @g_step.on_click
        def _(_e):
            S.step_once = True

        @g_speed.on_update
        def _(_e):
            S.speed = float(g_speed.value)

        if rigid:
            def write_ranges():
                for _, field, *_ in PHYSICS:
                    if field not in g_phys:
                        continue
                    if g_dr.value:
                        setattr(rd_cfg, field, list(rd_train[field]))
                    else:
                        v = float(g_phys[field].value) * _UNIT.get(field, 1.0)
                        setattr(rd_cfg, field, [v, v])
                S.physics_dirty = True

            @g_apply.on_click
            def _(_e):
                write_ranges()

            @g_dr.on_update
            def _(_e):
                write_ranges()

            @g_nominal.on_click
            def _(_e):
                for _, field, _lo, _hi, _st, nom in PHYSICS:
                    if field in g_phys:
                        g_phys[field].value = nom
                g_dr.value = False
                write_ranges()

            write_ranges()  # start from the nominal (matched) physics, not a training draw

        # Preset views (camera position, look-at), env frame. The mouse still orbits / pans / zooms
        # freely; these only jump. The sheet sits at the table centre, ~0.53 m up; robot at +y.
        views = {
            "default": ((0.35, -0.45, 0.85), (0.0, 0.05, 0.53)),
            "top-down": ((0.0, -0.01, 1.25), (0.0, 0.0, 0.53)),
            "side (x)": ((0.65, 0.05, 0.62), (0.0, 0.05, 0.53)),
            "behind robot": ((-0.15, 1.15, 1.15), (0.0, 0.0, 0.53)),
            "wide": ((1.1, -1.1, 1.05), (0.0, 0.2, 0.55)),
        }

        def set_view(client, name):
            pos, at = views[name]
            client.camera.position = np.asarray(pos, dtype=float)
            client.camera.look_at = np.asarray(at, dtype=float)
            client.camera.up_direction = (0.0, 0.0, 1.0)

        with server.gui.add_folder("Camera (mouse: drag = orbit, right-drag = pan, scroll = zoom)"):
            for vname in views:
                b = server.gui.add_button(vname)

                @b.on_click
                def _(e, vname=vname):
                    for c in server.get_clients().values():
                        set_view(c, vname)

        @server.on_client_connect
        def _(client):
            set_view(client, "default")

        server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1, position=(0, 0, 0.0))

        # ------------------------------------------------------------------ drawing
        frames: dict = {}
        clouds: dict[str, object] = {}
        root: dict[str, object] = {}
        world_keys: list = []
        # The env the scene was last built for. The env dropdown sets S.world from the GUI thread,
        # so drawing must not read S.world: between that write and the next rebuild, world_keys
        # still belong to the previous env (on a one-shape-per-env pool, a shape that env lacks).
        scene_w = [0]
        pc_nodes: dict[str, object] = {}   # the student's point-cloud layers
        cam_key = [None]

        def sheet_opacity():
            """Under a student the true sheet is translucent, so the cloud on and under it shows."""
            return float(g_opac.value) if stu() is not None and float(g_opac.value) < 0.995 else None

        def is_body(s):
            return s.key is not None and (physx or s.key >= 0)

        def key_name(k):
            return f"b{k[0]}_{k[1]}" if physx else f"b{k}"

        def rebuild_scene():
            nonlocal world_keys
            # Everything drawn lives under /w, so dropping it clears the previous env completely.
            if "root" in root:
                root["root"].remove()
            root["root"] = server.scene.add_frame("/w", show_axes=False)
            frames.clear()
            clouds.clear()
            pc_nodes.clear()
            ghost_nodes.clear()
            cam_key[0] = None
            opac = sheet_opacity()
            w = scene_w[0] = S.world
            o = origins[w]
            mine = shapes_of(w)
            world_keys = sorted({s.key for s in mine if is_body(s)})
            for k in world_keys:
                frames[k] = server.scene.add_frame(f"/w/{key_name(k)}", show_axes=False)
            for i, s in enumerate(mine):
                parent = f"/w/{key_name(s.key)}" if is_body(s) else "/w/static"
                pos = s.pos - (o if not is_body(s) else 0.0)
                server.scene.add_mesh_simple(
                    f"{parent}/s{i}", s.verts, s.faces, color=colour(s),
                    position=pos, wxyz=s.wxyz, side="double" if is_body(s) and "Robot" not in s.label else "front",
                    opacity=opac if is_body(s) and "Robot" not in s.label and "Table" not in s.label else None,
                )
            S.rebuild = False

        def body_poses(w):
            if physx:
                return px.poses(w, world_keys)
            q = wp.to_torch(NewtonManager._state_0.body_q)[
                torch.as_tensor(world_keys, device=device, dtype=torch.long)].cpu().numpy()
            return {k: (x[:3], (x[6], x[3], x[4], x[5])) for k, x in zip(world_keys, q)}

        def draw():
            w = scene_w[0]
            o = origins[w]
            if world_keys:
                poses = body_poses(w)
                with server.atomic():
                    for k, (pw, q) in poses.items():
                        h = frames[k]
                        p = np.asarray(pw, dtype=np.float64) - o
                        far = abs(p[0]) > 1.5 or abs(p[1]) > 1.5 or p[2] > 1.6  # parked pool shapes
                        h.visible = not far
                        h.position = p
                        h.wxyz = tuple(float(v) for v in q)
            if w in tri_by_world:
                state = NewtonManager._state_0
                pidx, t_mv, t_st = tri_by_world[w]
                pts = wp.to_torch(state.particle_q)[torch.as_tensor(pidx, device=device)].cpu().numpy() - o
                for name, t, rgb in (("moving", t_mv, MOVING_RGB), ("stationary", t_st, STATIONARY_RGB)):
                    if len(t):
                        server.scene.add_mesh_simple(f"/w/cloth_{name}", pts.astype(np.float32), t,
                                                     color=rgb, side="double", opacity=sheet_opacity())
            if pierce is not None and last_pierce:
                hit, a, b = last_pierce[0]
                hit_w = hit[w].cpu().numpy()
                if hit_w.any():
                    segs = torch.stack((a[w], b[w]), dim=1)[torch.as_tensor(hit_w, device=device)]
                    segs = segs.cpu().numpy() - o
                    clouds["pierce"] = server.scene.add_line_segments(
                        "/w/pierce", segs.astype(np.float32),
                        np.tile([255, 0, 0], (len(segs), 2, 1)).astype(np.uint8), thickness=0.004)
                elif "pierce" in clouds:
                    clouds["pierce"].visible = False
            if g_kp.value and hasattr(inner, "fold_targets_w"):
                goal = inner.fold_targets_w()[w].detach().cpu().numpy() - o
                kp = inner.cloth_keypoints_w()[w].detach().cpu().numpy() - o
                clouds["goal"] = server.scene.add_point_cloud(
                    "/w/goal_kp", goal.astype(np.float32),
                    np.tile(GOAL_RGB, (len(goal), 1)).astype(np.uint8),
                    point_size=0.007, point_shape="circle", precision="float32")
                clouds["kp"] = server.scene.add_point_cloud(
                    "/w/kp", kp.astype(np.float32), np.tile(KP_RGB, (len(kp), 1)).astype(np.uint8),
                    point_size=0.005, point_shape="circle", precision="float32")
            else:
                for h in clouds.values():
                    h.visible = False
            draw_ghost()
            draw_cloud()

        def put_points(name, pts, rgb, size):
            if len(pts) == 0:
                if name in pc_nodes:
                    pc_nodes[name].visible = False
                return
            pc_nodes[name] = server.scene.add_point_cloud(
                f"/w/pc_{name}", np.asarray(pts, np.float32), np.asarray(rgb, np.uint8),
                point_size=size, point_shape="circle", precision="float32")

        def look_at(pos, target):
            """wxyz of a camera at ``pos`` (OpenCV axes: +z forward, +y down) looking at ``target``."""
            import viser.transforms as vtf

            z = (target - pos) / max(float(np.linalg.norm(target - pos)), 1e-9)
            x = np.cross(z, (0.0, 0.0, 1.0))
            x = x / max(float(np.linalg.norm(x)), 1e-9)
            return vtf.SO3.from_matrix(np.stack((x, np.cross(z, x), z), axis=1)).wxyz

        ghost_faces = [None]
        ghost_nodes: dict[str, object] = {}

        def goal_task_info(w):
            """``fold_pool.goal_task``: env ``w``'s task name, designated / look-alike errors and the
            matched goal state's grid (world). None without goal tasks."""
            geo = inner.task.geometry
            gt = geo.goal_tasks
            if gt is None or gt.task is None:
                return None
            view = inner.fold_view()
            grid, lookalike = geo.matched_goal_sheet(view)
            return {"name": gt.names[int(gt.task[w])], "des": float(view.designated_error[w]),
                    "alt": float(view.lookalike_error[w]),
                    "matched": "look-alike" if bool(lookalike[w]) else "designated",
                    "grid": grid[w].detach().cpu().numpy()}

        def draw_ghost():
            """Goal tasks: the goal state the fold criterion matches best now (any policy)."""
            w = scene_w[0]
            info = goal_task_info(w) if g_pc_ghost is not None and g_pc_ghost.value else None
            if info is None:
                for nm in ("ghost", "ghost_pts"):
                    if nm in ghost_nodes:
                        ghost_nodes[nm].visible = False
                return
            grid = info["grid"] - origins[w]
            if ghost_faces[0] is None or ghost_faces[0][0] != len(grid):
                r = int(round(len(grid) ** 0.5))
                ij = np.arange(r * r).reshape(r, r)
                a, b, c, d = ij[:-1, :-1].ravel(), ij[:-1, 1:].ravel(), ij[1:, :-1].ravel(), ij[1:, 1:].ravel()
                ghost_faces[0] = (len(grid), np.concatenate((np.stack((a, b, d), 1), np.stack((a, d, c), 1))).astype(np.int32))
            ghost_nodes["ghost"] = server.scene.add_mesh_simple(
                "/w/pc_ghost", grid.astype(np.float32), ghost_faces[0][1], color=(40, 200, 90),
                side="double", opacity=0.35)
            ghost_nodes["ghost_pts"] = server.scene.add_point_cloud(
                "/w/pc_ghost_pts", grid.astype(np.float32), np.tile((20, 150, 60), (len(grid), 1)).astype(np.uint8),
                point_size=0.004, point_shape="circle", precision="float32")

        def draw_cloud():
            """What the active student read at its latest step, in the drawn env."""
            pl, w = stu(), scene_w[0]
            if pl is None or pl.last is None or not hasattr(pl.cloud, "last_xyz"):
                for h in pc_nodes.values():
                    h.visible = False
                return
            pc, o = pl.cloud, origins[w]
            xyz = pc.last_xyz[w].detach().float().cpu().numpy()          # env frame
            names = np.array(pc.labels or ("live",))
            lab, cam = names[pc.last_label[w].cpu().numpy()], pc.last_cam[w].cpu().numpy()
            ok = pc.last_mask[w].cpu().numpy() & np.isfinite(xyz).all(axis=1)
            if g_pc_col.value == "camera":
                rgb = np.where((cam >= 0)[:, None], PC_CAM_RGB[np.clip(cam, 0, 1)], PC_NO_CAM_RGB)
            else:
                rgb = np.array([PC_LABEL_RGB[k] for k in lab], np.uint8).reshape(-1, 3)
            size = float(g_pc_size.value) * 1e-3
            groups = {"live": lab == "live", "fk": lab == "fk",
                      "ref": (lab == "ref_moving") | (lab == "ref_stationary") | (lab == "goal")}
            for name, sel in groups.items():
                sel = sel & ok & bool(g_pc_show[name].value)
                put_points(name, xyz[sel], rgb[sel], size)

            true = inner.cloth_points_w()[w].detach().cpu().numpy() - o if g_pc_true.value else np.zeros((0, 3))
            put_points("true", true, np.tile((90, 90, 90), (len(true), 1)), 0.003)

            if g_pc_cams.value:
                pos = pc.cams[w].cpu().numpy()
                ctr = (pc.centre if pc.centre.dim() == 1 else pc.centre[w]).cpu().numpy()
                key = (w, pos.round(4).tobytes(), ctr.round(3).tobytes())
                if cam_key[0] != key:   # the per-episode camera position draw
                    cam_key[0] = key
                    for c in range(2):
                        pc_nodes[f"cam{c}"] = server.scene.add_camera_frustum(
                            f"/w/pc_cam{c}", fov=0.9, aspect=1.33, scale=0.08,
                            color=tuple(int(v) for v in PC_CAM_RGB[c]), wxyz=look_at(pos[c], ctr), position=pos[c])
                        pc_nodes[f"camname{c}"] = server.scene.add_label(
                            f"/w/pc_cam{c}_name", f"camera {'AB'[c]}", position=pos[c])
                for c in range(2):
                    pc_nodes[f"cam{c}"].visible = pc_nodes[f"camname{c}"].visible = True
            else:
                for c in range(2):
                    for nm in (f"cam{c}", f"camname{c}"):
                        if nm in pc_nodes:
                            pc_nodes[nm].visible = False

            if g_pc_aux.value and hasattr(inner, "fold_targets_w"):
                # aux = [keypoints - palm, keypoints - goal] * scale; its error is in the env frame, so
                # the estimates are the truth moved by it (drawn on the current sheet).
                d = ((pl.last["aux_pred"][w] - pl.last["aux_true"][w]) / pl.scale).view(2, -1, 3).cpu().numpy()
                kp = inner.cloth_keypoints_w()[w].detach().cpu().numpy() - o
                goal = inner.fold_targets_w()[w].detach().cpu().numpy() - o
                kp_est, goal_est = kp + d[0], goal + d[0] - d[1]
                put_points("aux_kp", kp_est, np.tile(PC_AUX_KP_RGB, (len(kp), 1)), 0.006)
                put_points("aux_goal", goal_est, np.tile(PC_AUX_GOAL_RGB, (len(goal), 1)), 0.006)
                segs = np.concatenate((np.stack((kp, kp_est), axis=1), np.stack((goal, goal_est), axis=1)))
                cols = np.concatenate((np.tile(PC_AUX_KP_RGB, (len(kp), 2, 1)), np.tile(PC_AUX_GOAL_RGB, (len(goal), 2, 1))))
                pc_nodes["aux_lines"] = server.scene.add_line_segments(
                    "/w/pc_aux_lines", segs.astype(np.float32), cols.astype(np.uint8), thickness=0.002)
            else:
                for nm in ("aux_kp", "aux_goal", "aux_lines"):
                    if nm in pc_nodes:
                        pc_nodes[nm].visible = False

        def hud_text():
            w = S.world
            st = S.stats[w]
            tol = float(inner.cfg.cloth.keypoint_tolerance)
            err = float(inner.fold_error()[w]) if hasattr(inner, "fold_error") else float("nan")
            lines = [
                f"**{g_task.value if g_task else args.task}** -- {S.policy_name}, coef {S.coef:g}",
                f"env {w}" + (f", {sizes[w] * 100:.1f} cm" if sizes is not None
                              else f", sheet {float(inner.cfg.cloth.size) * 100:g} cm")
                + (f", shape **{pool[int(rb.pool.active[w])]}**" if pool else ""),
                f"step {int(inner.episode_length_buf[w])} / {int(inner.max_episode_length)}",
                *([(lambda si: f"start: **{fold_names[si[0]][si[1]]}** at {si[2]:.0f} deg"
                    if si[1] is not None else "start: **flat**")(start_info(w))] if start_tabs is not None else []),
                f"fold error **{err * 1e3:.1f} mm** (tolerance {tol * 1e3:.1f} mm) "
                + ("FOLDED" if err < tol else ""),
                f"this env: {st['folds']} folds / {st['episodes']} episodes, last: {st['last']}",
                f"all envs: {sum(s['folds'] for s in S.stats)} folds / "
                f"{sum(s['episodes'] for s in S.stats)} episodes",
                f"sim {S.fps:.0f} policy steps/s (real time = {1.0 / float(inner.step_dt):.0f})",
            ]
            gi = goal_task_info(w)
            if gi is not None:
                alt = f"{gi['alt'] * 1e3:.1f} mm" if np.isfinite(gi["alt"]) else "none"
                lines.append(f"goal task **{gi['name']}**: designated error {gi['des'] * 1e3:.1f} mm, "
                             f"look-alike {alt} -> matched **{gi['matched']}**")
            if pierce is not None:
                run = pierce_run[w].cpu().numpy()
                through = [f"{nm.replace('left_', '')} ({k} steps)" for nm, k in zip(pierce.names, run) if k > 0]
                lines.append("finger THROUGH cloth: **" + ", ".join(through) + "** (red)" if through
                             else "finger through cloth: none")
            pl = stu()
            if pl is not None and pl.last is not None and hasattr(pl.cloud, "last_xyz"):
                pc = pl.cloud
                e = ((pl.last["aux_pred"][w] - pl.last["aux_true"][w]) / pl.scale).view(2, -1, 3)
                e = e.norm(dim=-1).mean(dim=-1) * 100.0
                shift = pc.calib_t[w].norm(dim=-1) * 1e3
                lines += [
                    f"point cloud (goal input {pc.cfg.goal_mode}, fk points {pc.cfg.fk_points}): "
                    f"**{int(pc.last_count[w])} / {pc.cfg.n_points}** live points, "
                    f"cloth share {float(pc.last_cloth_share[w]) * 100:.0f}%, "
                    f"{float(pc.last_visible_frac[w]) * 100:.0f}% of the sheet seen",
                    f"aux head error: keypoints **{float(e[0]):.1f} cm**, keypoints - goal **{float(e[1]):.1f} cm**",
                    f"this episode: calibration shift A {float(shift[0]):.0f} mm / B {float(shift[1]):.0f} mm, "
                    f"depth std {float(pc.depth_std[w]) * 1e3:.1f} mm, drop {float(pc.drop_p[w]) * 100:.0f}%, "
                    f"cloud {int(pc.lag[w])} steps old, proprio {int(pl.p_lag[w])} steps old",
                ]
            if rb is not None and rb.physics_dr is not None:
                d = {k: float(v[w]) for k, v in rb.physics_dr.sample.items()}
                lines.append(
                    f"physics: slat mu {d['slat_friction']:.3f}, robot mu {d['robot_friction']:.3f}, "
                    f"damping {d['hinge_damping']:.3g}, stiffness {d['hinge_stiffness']:.2g}, "
                    f"margin {d['contact_margin'] * 1e3:.1f} mm, mass x{d['mass']:.2f}"
                )
            return "  \n".join(lines)

        # ------------------------------------------------------------------ loop
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n_env, inner.cfg.action_space), device=device))
        S.player = get_player(S.policy_name)
        active_name = S.policy_name
        sync_noise_gui()
        t_last, n_steps, t_fps = time.perf_counter(), 0, time.perf_counter()
        t_log = t_sim = t_draw = 0.0
        dt = float(inner.step_dt)
        while True:
            if S.restart is not None:
                task, ov, nenv, pol = S.restart
                status.content = f"**Restarting with {task}...** the page reconnects by itself in ~2-3 min."
                time.sleep(0.5)
                # Keep every override that is not a task choice (e.g. the contact buffer size).
                keep = [a for a in hydra_args if not a.startswith(task_keys)]
                argv = [sys.executable, "-m", "isaacsimenvs.scripts.visualization.policies.live_viser", "--task", task,
                        "--policy", pol, "--num_envs", str(nenv), "--port", str(args.port),
                        "--even_sizes", str(args.even_sizes), "--policy_config", str(args.policy_config),
                        "--seed", str(args.seed), *(["--headless"] if physx else []),
                        *keep, *ov]
                sys.stdout.flush()
                os.execv(sys.executable, argv)
            if S.rebuild:
                rebuild_scene()
                draw()  # poses now, not at the next step: the view may be paused
            if S.policy_name != active_name:
                S.player = get_player(S.policy_name)
                active_name = S.policy_name
                sync_noise_gui()
                S.rebuild = True  # the sheet is translucent under a student only
            if rigid and S.physics_dirty:
                rb.physics_dr.draw(torch.arange(n_env, device=device))
                S.physics_dirty = False
            if S.reset_all:
                obs, _ = env.reset()
                S.player.reset_rnn()
                S.reset_all = False
                draw()
            if S.reset_one:
                inner.episode_length_buf[S.world] = inner.max_episode_length  # truncates next step
                S.reset_one = False

            if S.playing or S.step_once:
                S.step_once = False
                S.player.expl_coef = S.coef
                action = S.player.get_action(obs["policy"], deterministic=S.deterministic)
                t0 = time.perf_counter()
                obs, _r, terminated, truncated, _x = env.step(action.to(device))
                torch.cuda.synchronize()
                t_sim += time.perf_counter() - t0
                ended = terminated.bool() | truncated.bool()
                if bool(ended.any()):
                    reasons = getattr(inner, "_termination_reasons", {})
                    for e in ended.nonzero(as_tuple=True)[0].tolist():
                        fired = [k for k in OUTCOMES if k in reasons and bool(reasons[k][e])]
                        out = fired[0] if fired else ("timeout" if bool(truncated[e]) else "?")
                        S.stats[e]["episodes"] += 1
                        S.stats[e]["folds"] += int(out == "fold_held")
                        S.stats[e]["last"] = out
                    S.player.reset_rnn(ended.nonzero(as_tuple=True)[0])
                n_steps += 1
                if S.pause_at and int(inner.episode_length_buf[S.world]) == S.pause_at:
                    S.playing = False
                    g_play.value = False
                if pierce is not None:
                    hit, pa, pb = pierce(NewtonManager._state_0)
                    last_pierce[:] = [(hit & ~ended[:, None], pa, pb)]
                    pierce_run = torch.where(hit & ~ended[:, None], pierce_run + 1, torch.zeros_like(pierce_run))
                t0 = time.perf_counter()
                draw()
                t_draw += time.perf_counter() - t0
            else:
                time.sleep(0.03)

            now = time.perf_counter()
            if now - t_fps > 0.5:
                S.fps = n_steps / (now - t_fps)
                n_steps, t_fps = 0, now
                hud.content = hud_text()
                if now - t_log <= 15.0:
                    t_sim = t_draw = 0.0
                if now - t_log > 15.0:
                    t_log = now
                    print(f"[live] {S.fps:.0f} steps/s, env {S.world}, "
                          f"{sum(x['folds'] for x in S.stats)} folds / "
                          f"{sum(x['episodes'] for x in S.stats)} episodes, "
                          f"{len(server.get_clients())} client(s); last 0.5 s: sim {t_sim * 1e3:.0f} ms, "
                          f"draw {t_draw * 1e3:.0f} ms", flush=True)
                t_sim = t_draw = 0.0
            if S.speed < 3.95:
                wait = dt / S.speed - (time.perf_counter() - t_last)
                if wait > 0:
                    time.sleep(wait)
            t_last = time.perf_counter()

    run()


if __name__ == "__main__":
    main()
