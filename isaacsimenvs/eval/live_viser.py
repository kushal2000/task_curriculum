"""Run a fold policy LIVE in a browser (viser), with its knobs on sliders.

    scripts/newton_py -m isaacsimenvs.eval.live_viser --task Isaacsimenvs-FoldPool-Direct-v0
    sbatch scripts/cluster/bos14_live_viser.sh          # on a GPU node; the log prints the tunnel

The env runs on the GPU exactly as `episodes.py` runs it (same reset, same zero-action tick, same
player), a handful of envs at once; one of them is drawn. What can change WITHOUT a restart:

* policy (any checkpoint in `POLICIES`), SAPG block coefficient, deterministic / sampled actions
* which env is drawn; for the fold pool, which shape each env gets at its next reset
* start pose: pinned (`disable_randomization`'s reset) or the training reset distribution
* fold tolerance (`cloth.keypoint_tolerance`, read by `is_folded` every step)
* rigid manipulands only: slat / fingertip friction, hinge damping and stiffness, contact margin,
  mass -- written through `RigidClothEnv._randomize_physics`, the same in-place path the training
  DR uses, so they take effect on the very next step

What needs a RESTART (the "Restart" button re-execs this process with the new task, ~2-3 min):
the task itself (pool / one chain / VBD cloth), the number of envs, and the sheet sizes (Newton
bakes collision meshes at build time). VBD cloth material parameters are not exposed.

Speed: the fold pool runs ~40-50 policy steps/s (real time is 60). The VBD cloth is captured as a
CUDA graph here (`cloth.use_cuda_graph`), ~20 steps/s -- a third of real time; without the graph it
is ~4.

Geometry comes from Newton's own viewer pipeline (`ViewerBase.set_model` -> `log_geo`), captured
by a null viewer, so what is drawn is exactly the collision geometry the solver sees.

**PhysX tasks** (`*Physx*`, run in `.venv_isaacsim`; `scripts/cluster/bos14_live_viser_physx.sh`): the
same viewer, with the geometry read from the USD stage (each rigid body's collision meshes and
primitives, in its own frame) and the body poses from Isaac Lab's PhysX views every step. The friction
sliders are hidden there: PhysX refuses friction DR (`physx_envs`). viser needs websockets >= 13,
newer than Isaac Sim's 12, so the launcher puts a copy first on PYTHONPATH (nothing Kit loads headless
imports websockets).
"""

from __future__ import annotations

import argparse
import colorsys
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant

ROOT = Path(__file__).resolve().parents[2]

#: Display name -> (checkpoint, default SAPG coef). Paths are relative to the repo root.
#: The best seed of each 2026-10-06 PhysX run (generic prior, 24576 envs, 7.47e8 steps), picked by
#: held folds on the VBD cloth summed over 4/1.5 cm x pinned/rand (docs/results/physx_generic_prior_runs.md);
#: label = its cloth score at 4 cm, pinned / randomised.
POLICIES = {
    "run 2: flat pool (s3; cloth 78/66%)": ("outputs/2026-10-06/14-53-54/0_pxpool_one_flat_s3_158376/nn/0_pxpool_one_flat_s3_158376.pth", 0.0),
    "run 1: pool + folded starts (s1; cloth 56/50%)": ("outputs/2026-10-06/14-33-35/0_pxpool_one_initfold_s1_158375/nn/0_pxpool_one_initfold_s1_158375.pth", 0.0),
    "run 3: box3-mid (s3; cloth 61/64%)": ("outputs/2026-10-06/13-32-04/0_pxchain_box3_s3_158324/nn/0_pxchain_box3_s3_158324.pth", 0.0),
    "previous best: PhysX stage 1 -> 2 (157622 s1; cloth 88/63%)": ("outputs/2026-10-05/23-51-55/0_pxpool_pxs1_arm1e3_e6144_s1_157622/nn/0_pxpool_pxs1_arm1e3_e6144_s1_157622.pth", 0.0),
}

#: Not in the menu; add one back to POLICIES to show it.
ARCHIVED_POLICIES = {
    "stage-2 pool, box3-mid prior s3 (Newton)": ("outputs/2026-10-04/20-17-57/0_fp_box3_s3_156746/nn/0_fp_box3_s3_156746.pth", 0.0),
    "stage-1 box3-mid chain (Newton rc2 fixdr s2)": ("outputs/2026-09-30/17-12-24/0_rc2_box3-mid_fixdr_s2_155349/nn/0_rc2_box3-mid_fixdr_s2_155349.pth", 0.0),
    "cloth-trained reference (VBD)": ("outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth", 0.0),
    "generic tool-use prior (untrained on folds)": ("pretrained_policy/model.pth", 50.0),
}

#: Rigid manipulands get the evaluation-sized contact buffer: a policy that folds a 16-17 slat chain
#: needs ~548k pairs/env, over the yaml's 524288 (see the rigid-cloth notes). The VBD cloth keeps its
#: yaml default -- the big buffer is pure cost there (it made the cloth step ~2x slower).
_RIGID_BUF = "env.newton.per_env_triangle_pairs=2097152"

#: Display name -> (gym task, extra hydra overrides). Newton; PHYSX_TASKS for the PhysX viewer.
TASKS = {
    "fold pool (5 shapes, 10-20 cm)": ("Isaacsimenvs-FoldPool-Direct-v0", [_RIGID_BUF]),
    "VBD cloth": ("Isaacsimenvs-Cloth-Direct-v0", []),
    **{
        f"chain {v}": ("Isaacsimenvs-RigidCloth-Direct-v0", [f"env.rigid_cloth.variant={v}", _RIGID_BUF])
        for v in ("box3-mid", "box2-surface", "cyl-mid-odd", "box-mid-odd", "box-surface")
    },
}
PHYSX_TASKS = {
    "fold pool (5 shapes, 10-20 cm)": ("Isaacsimenvs-FoldPoolPhysx-Direct-v0", []),
    # Run 1's training setup: 80% of episodes start partly folded, hinge damping 0.02.
    "fold pool + folded starts (run 1 setup)": (
        "Isaacsimenvs-FoldPoolPhysx-Direct-v0",
        ["env.fold_pool.init_fold_prob=0.8", "env.rigid_cloth.joint_damping=0.02"],
    ),
    "chain box3-mid": ("Isaacsimenvs-RigidClothPhysx-Direct-v0", ["env.rigid_cloth.variant=box3-mid"]),
}
#: VBD cloth size -> hydra overrides (the large-cloth evals' settings, scripts/cluster/bos14_cloth_size_sweep.sh):
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
#: PhysX has no friction DR (a tensor material write resets the combine modes; physx_envs).
_PHYSX_NO = ("slat_friction_scale", "fingertip_friction_scale")

#: Physics sliders for rigid manipulands: (label, randomization field, lo, hi, step, nominal).
#: Each is written as a degenerate range [v, v], so the env's own DR draw returns exactly v.
PHYSICS = (
    ("slat friction x", "slat_friction_scale", 0.2, 3.0, 0.05, 1.0),
    ("fingertip friction x", "fingertip_friction_scale", 0.2, 3.0, 0.05, 1.0),
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


def _parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="Isaacsimenvs-FoldPool-Direct-v0")
    p.add_argument("--policy", default="run 2: flat pool (s3; cloth 78/66%)", choices=list(POLICIES))
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
    physx = "Physx" in args.task
    tasks = PHYSX_TASKS if physx else TASKS
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
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    status = server.gui.add_markdown(f"**Building `{args.task}` with {args.num_envs} envs...** (~2-3 min)")

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = 0.01
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
            rd = env_cfg.rigid_cloth.randomization
            rd.enabled = True
            if float(rd.hinge_stiffness[1]) <= 0.0:
                rd.hinge_stiffness = [0.0, 1e-3]
        # Keep the training reset distribution's numbers, then pin; the GUI swaps between them.
        rs = env_cfg.reset
        reset_rand = {k: getattr(rs, k) for k in (
            "reset_position_noise_x", "reset_position_noise_y", "reset_position_noise_z",
            "reset_dof_pos_random_interval_arm", "reset_dof_pos_random_interval_fingers",
            "reset_dof_vel_random_interval", "table_reset_z_range", "fixed_start_pose")}
        disable_randomization(env_cfg)
        reset_pin = {k: getattr(rs, k) for k in reset_rand}
        use_single_object_variant()

        if args.even_sizes and hasattr(env_cfg, "fold_pool"):
            # The per-env size draw is FoldPoolTaskMixin._draw_env_sizes (both engines).
            from isaacsimenvs.tasks.cloth import fold_pool_task as fpe

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
        pool = [sh.name for sh in getattr(inner, "_pool", [])]
        sizes = getattr(inner, "_env_size_np", None)

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
            mv = set(inner._moving_idx.tolist()) if hasattr(inner, "_moving_idx") else set()
            for w in range(n_env):
                pidx = np.nonzero(owner == w)[0]
                base = int(pidx[0])
                t = tri[(owner[tri] == w).all(axis=1)] - base
                in_mv = np.isin(t, list(mv)).sum(axis=1) >= 2
                tri_by_world[w] = (pidx, t[in_mv], t[~in_mv])

        # Fingers through the cloth (VBD only: particles are its sole contact with the hand).
        pierce = None
        if tri_by_world:
            from isaacsimenvs.tasks.cloth.utils.pierce import PierceDetector

            pierce = PierceDetector(model, n_env, device)
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
        )
        # Partly folded starts (fold pool with fold_pool.init_fold_prob > 0): per shape, its start
        # folds in the order of the env's (sign-calibrated) direction table.
        start_tabs = getattr(inner, "_start_tables", None)
        fold_names = list(inner._start_names) if start_tabs is not None else []
        # One FIXED option list ("<shape>: <fold>"): changing a dropdown's options while switching envs
        # let the browser echo a stale value back over the env's choice.
        fold_label = {f"{pool[s]}: {f}": (s, k) for s, names in enumerate(fold_names) for k, f in enumerate(names)}
        players: dict[str, PretrainedPlayer] = {}

        def get_player(name):
            if name not in players:
                ckpt, _ = POLICIES[name]
                players[name] = PretrainedPlayer(
                    config_path=args.policy_config, checkpoint_path=ckpt, num_envs=n_env,
                    device=args.device, num_observations=int(inner.cfg.observation_space),
                    num_actions=int(inner.cfg.action_space), expl_coef=S.coef,
                )
            return players[name]

        # Fold pool: honour the GUI's per-env shape at each reset.
        if pool:
            orig_draw = inner._draw

            def draw(env_ids):
                orig_draw(env_ids)  # random draw first, then override the pinned envs
                forced = [(int(e), pool.index(S.shape_choice[int(e)])) for e in env_ids.tolist()
                          if S.shape_choice[int(e)] != "random"]
                if forced:
                    ids = torch.tensor([e for e, _ in forced], device=device)
                    shp = torch.tensor([k for _, k in forced], device=device)
                    inner.object.select(ids, shp)
                    if start_tabs is not None:  # the random draw drew start folds for other shapes
                        inner._draw_start_folds(ids, shp)
                if start_tabs is not None:
                    for e in env_ids.tolist():
                        fc, ang = S.fold_choice[e], S.angle_choice[e]
                        if fc == "random" and ang is None:
                            continue
                        s = int(inner.object.active[e])
                        tab = start_tabs[s]
                        if fc == "flat":
                            q, folded = torch.zeros(tab.shape[1], device=device), False
                        else:
                            if fc in fold_label:
                                fs_, k = fold_label[fc]
                                if fs_ != s:
                                    continue  # (the shape choice pins it; this cannot normally happen)
                            elif bool(inner._start_folded[e]):
                                k = start_info(e)[1]  # keep the drawn fold, change only its angle
                            else:
                                continue  # a flat draw and no fold named: stays flat
                            if ang is not None:
                                a = torch.full((inner._start_num_parts,), math.radians(ang), device=device)
                            else:  # each part of the start its own angle, as the training draw
                                a = torch.deg2rad(torch.tensor(np.random.uniform(
                                    *inner.cfg.fold_pool.init_fold_deg, size=inner._start_num_parts),
                                    device=device, dtype=torch.float32))
                            q, folded = tab[k] * a[inner._start_parts[s][k]], True
                        inner.object.set_start_joints(s, torch.tensor([e], device=device), q.unsqueeze(0))
                        inner._start_folded[e] = folded

            inner._draw = draw

        def start_info(e):
            """(shape index, start-fold index or None, angle [deg]) env ``e`` started its episode at."""
            s = int(inner.object.active[e])
            if start_tabs is None or not bool(inner._start_folded[e]) or s not in inner.object._start_q:
                return s, None, 0.0
            q = inner.object._start_q[s][e]
            tab = start_tabs[s]
            cos = (tab @ q) / (tab.norm(dim=1) * q.norm()).clamp(min=1e-9)
            return s, int(cos.argmax()), math.degrees(float(q.abs().max()))

        rd_cfg = inner.cfg.rigid_cloth.randomization if rigid else None
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
            g_pol = server.gui.add_dropdown("policy", list(POLICIES), initial_value=args.policy)
            g_coef = server.gui.add_slider("SAPG coef (0 = block 5, 50 = block 0)", 0.0, 50.0, 10.0, S.coef)
            g_det = server.gui.add_checkbox("deterministic actions", True)
            g_play = server.gui.add_checkbox("playing", True)
            g_step = server.gui.add_button("single step")
            g_speed = server.gui.add_slider("speed (x real time; 4 = as fast as possible)", 0.1, 4.0, 0.1, 1.0)

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
            if g_csize is not None and task == "Isaacsimenvs-Cloth-Direct-v0":
                ov = [*ov, *CLOTH_SIZES[g_csize.value]]
            S.restart = (task, ov, int(g_nenv.value))

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
                S.shape_choice[S.world] = g_shape.value
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
            w = S.world
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
                )
            S.rebuild = False

        def body_poses(w):
            if physx:
                return px.poses(w, world_keys)
            q = wp.to_torch(NewtonManager._state_0.body_q)[
                torch.as_tensor(world_keys, device=device, dtype=torch.long)].cpu().numpy()
            return {k: (x[:3], (x[6], x[3], x[4], x[5])) for k, x in zip(world_keys, q)}

        def draw():
            w = S.world
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
                                                     color=rgb, side="double")
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

        def hud_text():
            w = S.world
            st = S.stats[w]
            tol = float(inner.cfg.cloth.keypoint_tolerance)
            err = float(inner.fold_error()[w]) if hasattr(inner, "fold_error") else float("nan")
            lines = [
                f"**{g_task.value if g_task else args.task}** -- {S.policy_name}, coef {S.coef:g}",
                f"env {w}" + (f", {sizes[w] * 100:.1f} cm" if sizes is not None
                              else f", sheet {float(inner.cfg.cloth.size) * 100:g} cm")
                + (f", shape **{pool[int(inner.object.active[w])]}**" if pool else ""),
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
            if pierce is not None:
                run = pierce_run[w].cpu().numpy()
                through = [f"{nm.replace('left_', '')} ({k} steps)" for nm, k in zip(pierce.names, run) if k > 0]
                lines.append("finger THROUGH cloth: **" + ", ".join(through) + "** (red)" if through
                             else "finger through cloth: none")
            if rigid and hasattr(inner, "dr_sample"):
                d = {k: float(v[w]) for k, v in inner.dr_sample.items()}
                lines.append(
                    f"physics: slat mu {d['slat_friction']:.3f}, tip mu {d['fingertip_friction']:.3f}, "
                    f"damping {d['hinge_damping']:.3g}, stiffness {d['hinge_stiffness']:.2g}, "
                    f"margin {d['contact_margin'] * 1e3:.1f} mm, mass x{d['mass']:.2f}"
                )
            return "  \n".join(lines)

        # ------------------------------------------------------------------ loop
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n_env, inner.cfg.action_space), device=device))
        S.player = get_player(S.policy_name)
        active_name = S.policy_name
        t_last, n_steps, t_fps = time.perf_counter(), 0, time.perf_counter()
        t_log = t_sim = t_draw = 0.0
        dt = float(inner.step_dt)
        while True:
            if S.restart is not None:
                task, ov, nenv = S.restart
                status.content = f"**Restarting with {task}...** the page reconnects by itself in ~2-3 min."
                time.sleep(0.5)
                # Keep every override that is not a task choice (e.g. the contact buffer size).
                keep = [a for a in hydra_args if not a.startswith(_TASK_KEYS)]
                argv = [sys.executable, "-m", "isaacsimenvs.eval.live_viser", "--task", task,
                        "--policy", S.policy_name, "--num_envs", str(nenv), "--port", str(args.port),
                        "--even_sizes", str(args.even_sizes), *(["--headless"] if physx else []),
                        *keep, *ov]
                sys.stdout.flush()
                os.execv(sys.executable, argv)
            if S.rebuild:
                rebuild_scene()
                draw()  # poses now, not at the next step: the view may be paused
            if S.policy_name != active_name:
                S.player = get_player(S.policy_name)
                active_name = S.policy_name
            if rigid and S.physics_dirty:
                inner._randomize_physics(torch.arange(n_env, device=device))
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
