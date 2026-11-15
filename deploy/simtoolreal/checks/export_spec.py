"""Export what the real-robot Play loop needs from the training env, as JSON, with no Isaac Sim.

The Play policy in pretrained_policy/nov6_dr.pth was trained on Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0 on
branch kushal/2026-10-30_pretraining_real (Flexiv Rizon 4s + RIGHT Wuji Hand 2, human-like limits,
the real setup's table and home pose). This script reads that branch's files straight from git --
robots.py (joint order, gains, palm / fingertip bodies and offsets, home pose), the task YAML (action
pipeline, obs layout, keypoints, table, goal sampling, success test, DR) and the robot URDF (joint
limits, arm effort limits) -- and writes them in one file, in the policy's joint order
(deploy/utils/spec_export.py builds the fields). The deploy
code then holds no hand-copied numbers. Writes spec/ (the spec json, and the URDF for the FK).

Static on purpose: the joint limits here are the URDF's, which PhysX normally keeps as they are;
check_rec1.py (a sim rollout run through the deploy code) checks that. --rollout_meta adds the
fingertip order the observation really uses, which only the sim rollout shows. The spec in spec/
was made with:

    .venv_deploy/bin/python deploy/simtoolreal/checks/export_spec.py \
        --rollout_meta docs/results/simtoolreal/rec1/rollout_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent     # deploy/simtoolreal/checks
sys.path.insert(0, str(HERE.parents[2]))   # repo root, for deploy.*
from deploy.utils.spec_export import ROBOT_ROOT_POS, ROBOTS_PY, git_show, rev_parse, spec_fields, table_box   # noqa: E402

DEFAULT_REF = "origin/kushal/2026-10-30_pretraining_real"
TASK = "Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0"
TASK_YAML = "isaacsimenvs/cfg/task/PlayFlexivWujiRightHumanLike.yaml"
ROBOT_NAME = "flexiv_wuji_right_human_like"


def load_robot_spec(ref: str):
    """robots.py is plain Python (dataclasses + math), so it runs outside its package."""
    mod = types.ModuleType("branch_robots")
    sys.modules[mod.__name__] = mod  # dataclasses look their module up there
    exec(compile(git_show(ref, ROBOTS_PY), f"{ref}:{ROBOTS_PY}", "exec"), mod.__dict__)
    return mod.ROBOT_SPECS[ROBOT_NAME]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default=DEFAULT_REF, help="git ref of the training branch")
    p.add_argument("--out", type=Path, default=HERE.parent / "spec" / "spec_flexiv_wuji_right_hl.json")
    p.add_argument("--rollout_meta", type=Path, default=None,
                   help="rollout_meta.json from record_sim_rollout.py: adds the fingertip order the "
                        "observation really uses (PhysX body order) and the offset applied at each position")
    args = p.parse_args()

    commit = rev_parse(args.ref)
    spec = load_robot_spec(args.ref)
    cfg = yaml.safe_load(git_show(args.ref, TASK_YAML))
    assert cfg["robot"] == ROBOT_NAME, cfg["robot"]
    reset = cfg["reset"]
    out = spec_fields(args.ref, commit, TASK, ROBOT_NAME, spec, cfg, robot_root_pos=ROBOT_ROOT_POS,
                      table_pos_env=[*reset["table_reset_xy"], float(reset["table_reset_z"])],
                      table_top_z_env=float(reset["table_reset_z"]) + table_box(args.ref, cfg)[2] / 2)
    urdf_rel = cfg["assets"]["robot_urdf"]

    if args.rollout_meta is not None:
        meta = json.loads(args.rollout_meta.read_text())
        assert meta["task"] == TASK and meta["palm_body"] == spec.palm_body_name, meta["task"]
        out["fingertip_obs_bodies"] = meta["fingertip_bodies_obs_order"]
        out["fingertip_obs_offsets"] = meta["fingertip_offsets_by_position"]
        out["fingertip_obs_source"] = str(args.rollout_meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    (args.out.parent / Path(urdf_rel).name).write_text(git_show(args.ref, urdf_rel))
    print(f"wrote {args.out} and {Path(urdf_rel).name} from {args.ref} @ {commit[:7]}")


if __name__ == "__main__":
    main()
