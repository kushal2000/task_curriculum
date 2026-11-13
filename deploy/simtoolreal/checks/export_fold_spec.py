"""Export what the real-robot FOLD loop needs from a fold-pool training run, as JSON, with no Isaac Sim.

The fold policies in deploy/policy/ (flat7RW_*, box2RW_allDR) were trained on branch yiboc_wuji with
`train.py --experiment <name>`: a FoldPoolPhysx task on the Flexiv Rizon 4s + RIGHT Wuji Hand 2 in the
real setup's geometry. This script rebuilds that run's env config the way hydra_utils does -- the task
YAML, then the robot overlay (cfg/robot/), then the experiment's `base:` chain and the experiment --
and writes the spec: every field export_spec.py writes for Play (robot, action pipeline, observation,
table, ...), plus a `fold` block (the sheet's grid, the scored keypoints, the fold target, the
success test). The geometry comes from the branch's own fold_shapes.py and cloth_geometry.py (pure
Python), run from git.

Frames. The spec's env frame is the frame the policy OBSERVES in: the robot base at Play's
(0, 0.8, 0) (obs_utils.PLAY_ROBOT_ROOT_POS). The fold env itself puts the base at robot_root_pos
(0, 0.51, 0.20) and the table at the env origin; only palm_pos is absolute in the observation, and it
is shifted back to Play's frame, so the table and the sheet are exported shifted the same way.

    .venv_deploy/bin/python deploy/simtoolreal/checks/export_fold_spec.py               # flat7RW_base
    .venv_deploy/bin/python deploy/simtoolreal/checks/export_fold_spec.py --experiment box2RW_allDR --out /tmp/x.json

The five deploy/policy/ runs share every field the deploy code reads (`--compare` checks this).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import types
from pathlib import Path

import yaml

from export_spec import (HERE, ROBOT_ROOT_POS, ROBOTS_PY, cfg_default, git_show, rev_parse, section_get,
                         spec_fields, table_box)

DEFAULT_REF = "origin/yiboc_wuji"
DEFAULT_EXPERIMENT = "flat7RW_base"
POLICY_EXPERIMENTS = ["flat7RW_base", "flat7RW_fric", "flat7RW_sysid", "flat7RW_allDR", "box2RW_allDR"]
CFG = "isaacsimenvs/cfg"
CLOTH_INIT_PY = "isaacsimenvs/tasks/cloth/__init__.py"
CLOTH_CFG_PY = "isaacsimenvs/tasks/cloth/cloth_cfg.py"
FOLD_POOL_CFG_PY = "isaacsimenvs/tasks/cloth/fold_pool_cfg.py"
FOLD_SHAPES_PY = "isaacsimenvs/tasks/cloth/utils/fold_shapes.py"
CLOTH_GEOMETRY_PY = "isaacsimenvs/tasks/cloth/utils/cloth_geometry.py"
DEPLOY_SHAPE = "box2-mid"   # the real object: two plates taped along one edge


def deep_merge(base: dict, over: dict) -> dict:
    """hydra_utils._deep_merge: `over` onto `base`, recursing into dicts; anything else replaces."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment(ref: str, name: str) -> dict:
    """hydra_utils.load_experiment: cfg/experiment/<name>.yaml over its `base:` chain."""
    exp = yaml.safe_load(git_show(ref, f"{CFG}/experiment/{name}.yaml")) or {}
    base = exp.pop("base", None)
    return deep_merge(load_experiment(ref, base), exp) if base else exp


def run_module(ref: str, path: str, name: str) -> types.ModuleType:
    """Run a pure-Python file of the branch as a module (no package imports)."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod   # dataclasses look their module up there
    exec(compile(git_show(ref, path), f"{ref}:{path}", "exec"), mod.__dict__)
    return mod


def robot_variants(ref: str) -> dict[str, str]:
    """cloth/__init__.py's _ROBOT_VARIANTS: task-id suffix -> cfg/robot/<name>.yaml."""
    src = git_show(ref, CLOTH_INIT_PY)
    m = re.search(r"_ROBOT_VARIANTS\s*=\s*(\{.*?\})", src, re.S)
    return ast.literal_eval(m.group(1))


def env_config(ref: str, experiment: str) -> tuple[str, str, dict]:
    """(task id, robot overlay name, env config) as train.py --experiment <experiment> built it."""
    exp = load_experiment(ref, experiment)
    task = exp["task"]
    m = re.fullmatch(r"Isaacsimenvs-(FoldPoolPhysx)(\w*)-Direct-v0", task)
    assert m, f"{experiment}: {task} is not a PhysX fold-pool task"
    variant = robot_variants(ref)[m.group(2)]
    cfg = yaml.safe_load(git_show(ref, f"{CFG}/task/{m.group(1)}.yaml"))
    cfg = deep_merge(cfg, yaml.safe_load(git_show(ref, f"{CFG}/robot/{variant}.yaml")))
    cfg = deep_merge(cfg, exp.get("env") or {})
    return task, variant, cfg


def fold_fields(ref: str, cfg: dict, shift: list[float]) -> dict:
    """The `fold` block: sheet grid, keypoints, fold target and success test for the deploy shape.

    `shift` takes a fold-env position to the spec's env frame (Play's)."""
    def getter(section: dict, path: str):
        """section[key], or the configclass default in `path` when the YAMLs leave it out."""
        def get(key: str):
            value = section[key] if key in section else cfg_default(ref, path, key)
            assert value is not None, f"{key}: in no YAML and no default in {path}"
            return value
        return get

    c = getter(cfg["cloth"], CLOTH_CFG_PY)
    p = getter(cfg["fold_pool"], FOLD_POOL_CFG_PY)
    fs = run_module(ref, FOLD_SHAPES_PY, "branch_fold_shapes")
    geo = run_module(ref, CLOTH_GEOMETRY_PY, "branch_cloth_geometry")
    size, res, axis = float(c("size")), int(c("resolution")), c("fold_axis")
    assert axis == "x", axis

    pool = [fs.shape(n) for n in p("shapes")]
    shape = fs.shape(DEPLOY_SHAPE)
    assert DEPLOY_SHAPE in p("shapes"), f"{DEPLOY_SHAPE} not in the training pool {p['shapes']}"
    # Pool built at size cloth.size; the fold target's lift is the pool's ply gap (fold_lift_from_ply_gap).
    assert abs(shape.size - size) < 1e-12, (shape.size, size)
    gaps = {sh.ply_gap for sh in pool}
    assert len(gaps) == 1 and p("fold_lift_from_ply_gap"), gaps
    slab0 = {sh.thickness for sh in pool}
    assert len(slab0) == 1, slab0
    slab0 = slab0.pop()
    lo, hi = (float(v) for v in p("thickness_range"))
    assert lo == hi, f"slab thickness drawn per env ({lo}-{hi}): object_scales z is not one number"
    # fold_pool_task._write_object_scale: felt = cloth.thickness + slab - the pool's nominal slab.
    felt = float(c("thickness")) + lo - slab0

    grid = fs.grid_plate_map(shape, res)   # [(plate, (x, y, 0))], i over x then j over y
    plates = [g[0] for g in grid]
    assert set(plates) == {"left", "right"}, set(plates)
    reach_shift, reset_offset = p("reach_shift"), c("reset_offset_xy")
    table_z = float(cfg["reset"]["table_reset_z"])
    return {
        "shape": DEPLOY_SHAPE,
        "training_shapes": list(p("shapes")),
        "sheet_size_nominal": size,                     # the grid below is at this size; scale x, y by size / this
        "size_range": [float(v) for v in p("size_range")],
        "resolution": res,
        "fold_axis": axis,
        # grid_plate_map: the cloth grid carried onto the plates, in the SHEET frame (x across the hinge,
        # from the stationary plate "left" to the moving one "right"; y along the hinge; z up; the origin
        # on the hinge line, mid-plane). The hinge row (x = 0) is carried by "left".
        "grid_rest": [list(g[1]) for g in grid],
        "grid_plate": plates,
        "moving_plate": "right",
        "stationary_plate": "left",
        "moving_idx": geo.half_indices(res, axis, positive=True),
        "stationary_idx": geo.half_indices(res, axis, positive=False),
        # the 4 scored / observed keypoints: the moving half's corners, crease-near then crease-far
        "corner_idx": geo.corner_indices(res, axis),
        "goal_lift": float(next(iter(gaps))),           # mid-plane hinges: 0
        # Training writes the goal marker's quaternion as quat_from_matrix(R)[..., [1, 2, 3, 0]] (PhysX,
        # physx_envs._goal_quat_from_matrix, to match Newton): the observed goal keypoints use that
        # relabelled quaternion, read as wxyz.
        "goal_quat_relabel": [1, 2, 3, 0],
        "object_scale_z_m": felt,                       # object_scales = (size, size, this) / object_base_size
        "keypoint_tolerance": float(c("keypoint_tolerance")),
        "max_folded_footprint": float(c("max_folded_footprint")),
        "slab_m": lo,
        "contact_margin_m": float(cfg["rigid_cloth"]["contact_margin"]),
        "start_height": float(c("start_height")),
        # where a sheet of size_range[0] starts: centred here (+- reset_position_noise_x / _y), random yaw
        "sheet_centre_env": [float(reset_offset[0]) + shift[0], float(reset_offset[1]) + shift[1],
                             table_z + 0.5 * table_box(ref, cfg)[2] + shift[2]],
        "sheet_centre_noise_xy": [float(cfg["reset"]["reset_position_noise_x"]),
                                  float(cfg["reset"]["reset_position_noise_y"])],
        "reach_shift": [float(v) for v in reach_shift],
    }


def build(ref: str, experiment: str) -> tuple[dict, str]:
    """(spec dict, robot URDF path in the branch) for one experiment."""
    task, variant, cfg = env_config(ref, experiment)
    robots = run_module(ref, ROBOTS_PY, "branch_robots")
    robot_name = cfg["robot"]
    spec = robots.ROBOT_SPECS[robot_name]
    root_fold = [float(v) for v in cfg["robot_root_pos"]]
    # fold env -> spec env (Play's frame, base at ROBOT_ROOT_POS): shift by the base's offset
    shift = [a - b for a, b in zip(ROBOT_ROOT_POS, root_fold)]
    table = table_box(ref, cfg)
    table_z = float(cfg["reset"]["table_reset_z"])
    table_pos = [shift[0], shift[1], table_z + shift[2]]      # the fold table sits on the env origin
    out = spec_fields(ref, rev_parse(ref), task, robot_name, spec, cfg, robot_root_pos=ROBOT_ROOT_POS,
                      table_pos_env=table_pos, table_top_z_env=table_pos[2] + table[2] / 2)
    out["experiment"] = experiment
    out["robot_overlay"] = variant
    out["robot_root_pos_train_env"] = root_fold
    # PhysX resolves fingertip_body_regex in body order; robots.py lists fingertip_link_names in that
    # order (commit 224357d), so the observation's fingertips are fingertip_links, in order.
    out["fingertip_obs_bodies"] = out["fingertip_links"]
    out["fingertip_obs_offsets"] = out["fingertip_offsets"]
    out["fingertip_obs_source"] = "robots.py fingertip_link_names (PhysX body order)"
    dr = cfg["domain_randomization"]
    out["dr"] = {k: section_get(ref, dr, k) for k in (
        "use_obs_delay", "obs_delay_max", "use_action_delay", "action_delay_max", "point_dr_mode",
        "point_base_delay_range", "point_hold_choices", "point_noise_std", "point_bias_xyz",
        "point_common_bias_xyz", "joint_pos_obs_noise_std", "joint_velocity_obs_noise_std",
        "robot_stiffness_scale_range", "robot_damping_scale_range", "hand_friction_scale_range",
        "hand_effort_limit_scale_range", "action_noise_std")}
    out["fold"] = fold_fields(ref, cfg, shift)
    return out, cfg["assets"]["robot_urdf"]


# Fields that differ between the five runs without touching the deploy code (what each was trained on).
INFORMATIONAL = {"task", "experiment", "robot", "robot_overlay", "effort_limit", "hand_armature",
                 "hand_friction", "dr", "table_reset_z_range",
                 "fold.training_shapes", "fold.contact_margin_m"}


def flat(spec: dict) -> dict:
    """The spec with the fold block's fields as fold.<key>."""
    out = {k: v for k, v in spec.items() if k != "fold"}
    out.update({f"fold.{k}": v for k, v in spec.get("fold", {}).items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default=DEFAULT_REF, help="git ref of the training branch")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="cfg/experiment/<name>.yaml of the run")
    p.add_argument("--out", type=Path, default=HERE.parent / "spec" / "spec_flexiv_wuji_right_fold.json")
    p.add_argument("--compare", action="store_true",
                   help=f"build all of {POLICY_EXPERIMENTS} and print every field that differs")
    args = p.parse_args()

    if args.compare:
        specs = {e: flat(build(args.ref, e)[0]) for e in POLICY_EXPERIMENTS}
        ref = specs[POLICY_EXPERIMENTS[0]]
        for e, s in specs.items():
            for k in sorted(set(ref) | set(s)):
                if ref.get(k) != s.get(k):
                    tag = "info" if k in INFORMATIONAL else "DEPLOY"
                    print(f"[{tag}] {e:14s} {k}: {s.get(k)}")
        fold = [k for e, s in specs.items() for k in set(ref) | set(s)
                if ref.get(k) != s.get(k) and k not in INFORMATIONAL]
        print("deploy-relevant fields identical across the five runs" if not fold else f"DIFFER: {sorted(set(fold))}")
        return

    out, urdf_rel = build(args.ref, args.experiment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    (args.out.parent / Path(urdf_rel).name).write_text(git_show(args.ref, urdf_rel))
    print(f"wrote {args.out} and {Path(urdf_rel).name} from {args.ref} @ {out['source_commit'][:7]} "
          f"({args.experiment}, {out['task']})")


if __name__ == "__main__":
    main()
