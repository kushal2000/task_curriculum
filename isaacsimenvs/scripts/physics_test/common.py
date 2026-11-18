"""Shared setup for the physics tests: config + materials, a robot that touches nothing, the camera."""
from __future__ import annotations

import json

import numpy as np

from isaacsimenvs.scripts.evaluation.eval_cloth import load_env_cfg
from isaacsimenvs.scripts.physics_test import dr_samples


def add_material_args(p, size_levels: int = 2) -> None:
    """Arguments both tests share: config, design, thickness, and the material/size selection."""
    from pathlib import Path

    p.add_argument("--config", type=Path, required=True, help="training run's .hydra/config.yaml")
    p.add_argument("--design", default="strip_4", help="design YAML name, without .yaml")
    p.add_argument("--thickness", type=float, help="panel thickness [m]; default: the config's")
    p.add_argument("--size", type=float, nargs=2, default=(0.2, 0.2), metavar=("LENGTH", "WIDTH"),
                   help="sheet size without --dr_random")
    p.add_argument("--density", type=float, help="areal density [kg/m^2] without --dr_random")
    p.add_argument("--material", type=json.loads, default={},
                   help=f"JSON material overrides without --dr_random; keys: {', '.join(dr_samples.PARAMS)}")
    p.add_argument("--dr_random", type=int, help="sweep the run's randomization: all corners + N random draws")
    p.add_argument("--dr_seed", type=int, default=0)
    p.add_argument("--size_levels", type=int, default=size_levels,
                   help="with --dr_random: sizes per side over the config's length/width ranges (2 = corners)")


def add_video_args(p, slowmo: float) -> None:
    from pathlib import Path

    p.add_argument("--video", type=Path, help="record --video_env to this .mp4")
    p.add_argument("--video_env", type=int, default=0)
    p.add_argument("--slowmo", type=float, default=slowmo, help="video slow-down factor")
    p.add_argument("--video_size", type=int, nargs=2, default=(960, 720))


def set_camera(cfg, args, eye_offset) -> None:
    """Viewer camera on --video_env, looking at the table centre, in env-local coordinates."""
    cx, cy = cfg.reset.table_reset_xy
    cz = cfg.reset.table_reset_z + 0.05
    cfg.viewer.origin_type, cfg.viewer.env_index = "env", args.video_env
    cfg.viewer.lookat = (cx, cy, cz)
    cfg.viewer.eye = (cx + eye_offset[0], cy + eye_offset[1], cz + eye_offset[2])
    cfg.viewer.resolution = tuple(args.video_size)


def disable_robot(stage, env_prim_paths, hide: bool) -> int:
    """Turn off every robot collider (and hide the robot when recording). Returns the count."""
    from pxr import Usd, UsdGeom, UsdPhysics

    disabled = 0
    for path in env_prim_paths:
        root = stage.GetPrimAtPath(f"{path}/Robot")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if prim.IsInstanceProxy():
                    raise RuntimeError(f"cannot disable instanced robot collider {prim.GetPath()}")
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                disabled += 1
        if hide:
            UsdGeom.Imageable(root).MakeInvisible()
    if disabled == 0:
        raise RuntimeError("found no robot colliders to disable")
    return disabled


def configure(args):
    """The env config for this test, and the per-material values (M, len(PARAMS))."""
    from isaacsimenvs.scripts.evaluation.utils import disable_randomization

    loaded = load_env_cfg(args.config, args.overrides, num_envs=1)
    cfg = loaded.cfg
    if getattr(cfg, "backend", None) != "rigid":
        raise ValueError("physics tests need the rigid cloth backend")
    rc = cfg.rigid_cloth
    if args.thickness is not None:
        rc.thickness = args.thickness
    rd = rc.randomization
    ranges = {p: list(dr_samples._range(rd, p)) for p in dr_samples.PARAMS}
    disable_randomization(cfg)  # no reset noise, no random yaw (and DR off; re-enabled below)
    rd.enabled, rd.profile = True, "material"  # materials are written per env after reset
    cfg.task.release.enabled = False  # its robot-clearance check needs robot colliders
    if args.dr_random is not None:
        values, labels = dr_samples.material_samples(rd, args.dr_random, args.dr_seed)
        # Sizes: a size_levels x size_levels grid over the config's length/width ranges (2 = the
        # four corners), set explicitly so the extremes are tested exactly rather than sampled.
        lengths = np.linspace(*rc.length_range, args.size_levels)
        widths = np.linspace(*rc.width_range, args.size_levels)
        rc.geometries = [[float(l), float(w)] for l in lengths for w in widths]
    else:
        nominal = {"bending_rigidity": rc.bending_rigidity, "bending_viscosity": rc.bending_viscosity,
                   "density": rc.density, "table_static_friction": rc.table_static_friction,
                   "hand_static_friction": rc.hand_static_friction,
                   "dynamic_friction_ratio": rc.dynamic_friction_ratio, "armature_scale": 0.0}
        if args.density is not None:
            nominal["density"] = args.density
        unknown = set(args.material) - set(nominal)
        if unknown:
            raise ValueError(f"unknown --material keys {sorted(unknown)}")
        nominal.update(args.material)
        values, labels = np.array([[float(nominal[p]) for p in dr_samples.PARAMS]]), ["fixed"]
        rc.geometries = [[float(args.size[0]), float(args.size[1])]]
    return cfg, values, labels, ranges


