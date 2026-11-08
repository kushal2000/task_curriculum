"""Export what the real-robot Play loop needs from the training env, as JSON, with no Isaac Sim.

The Play policy in pretrained_policy/nov6_dr.pth was trained on Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0 on
branch kushal/2026-10-30_pretraining_real (Flexiv Rizon 4s + RIGHT Wuji Hand 2, human-like limits,
the real setup's table and home pose). This script reads that branch's files straight from git --
robots.py (joint order, gains, palm / fingertip bodies and offsets, home pose), the task YAML (action
pipeline, obs layout, keypoints, table, goal sampling, success test, DR) and the robot URDF (joint
limits, arm effort limits) -- and writes them in one file, in the policy's joint order. The deploy
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
import math
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent     # deploy/simtoolreal/checks
REPO = HERE.parents[2]
DEFAULT_REF = "origin/kushal/2026-10-30_pretraining_real"
TASK = "Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0"
TASK_YAML = "isaacsimenvs/cfg/task/PlayFlexivWujiRightHumanLike.yaml"
ROBOTS_PY = "isaacsimenvs/tasks/play/robots.py"
OBS_UTILS_PY = "isaacsimenvs/tasks/play/utils/obs_utils.py"
ROBOT_NAME = "flexiv_wuji_right_human_like"
# Robot root in the env frame (scene_utils.build_robot_articulation_usd_cfg), the same for every robot.
ROBOT_ROOT_POS = (0.0, 0.8, 0.0)
ROBOT_ROOT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def git_show(ref: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def load_robot_spec(ref: str):
    """robots.py is plain Python (dataclasses + math), so it runs outside its package."""
    mod = types.ModuleType("branch_robots")
    sys.modules[mod.__name__] = mod  # dataclasses look their module up there
    exec(compile(git_show(ref, ROBOTS_PY), f"{ref}:{ROBOTS_PY}", "exec"), mod.__dict__)
    return mod.ROBOT_SPECS[ROBOT_NAME]


def keypoint_corners(ref: str) -> list[list[int]]:
    """obs_utils.KEYPOINT_CORNERS, read as a literal (obs_utils itself imports torch / Isaac Lab)."""
    src = git_show(ref, OBS_UTILS_PY)
    start = src.index("KEYPOINT_CORNERS:")
    body = src[src.index("(", src.index("=", start)):]
    depth = 0
    for i, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            return [list(c) for c in eval(body[: i + 1], {})]  # a tuple-of-int literal
    raise ValueError("KEYPOINT_CORNERS literal not closed")


def urdf_joints(urdf_text: str) -> dict[str, dict]:
    out = {}
    for j in ET.fromstring(urdf_text).iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "prismatic") and lim is not None:
            out[j.get("name")] = {k: float(lim.get(k)) for k in ("lower", "upper", "effort", "velocity")
                                  if lim.get(k) is not None}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default=DEFAULT_REF, help="git ref of the training branch")
    p.add_argument("--out", type=Path, default=HERE.parent / "spec" / "spec_flexiv_wuji_right_hl.json")
    p.add_argument("--rollout_meta", type=Path, default=None,
                   help="rollout_meta.json from record_sim_rollout.py: adds the fingertip order the "
                        "observation really uses (PhysX body order) and the offset applied at each position")
    args = p.parse_args()

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", args.ref],
                            check=True, capture_output=True, text=True).stdout.strip()
    spec = load_robot_spec(args.ref)
    cfg = yaml.safe_load(git_show(args.ref, TASK_YAML))
    assert cfg["robot"] == ROBOT_NAME, cfg["robot"]
    urdf_rel = cfg["assets"]["robot_urdf"]
    urdf_text = git_show(args.ref, urdf_rel)
    limits = urdf_joints(urdf_text)

    names = list(spec.joint_names_canonical)
    arm, hand = list(spec.arm_joint_names), list(spec.hand_joint_names)
    assert names == arm + hand and len(arm) == 7 and len(hand) == 20, names
    missing = [n for n in names if n not in limits]
    assert not missing, f"joints not in {urdf_rel}: {missing}"

    lower = [limits[n]["lower"] for n in names]
    upper = [limits[n]["upper"] for n in names]
    # Reset pose: the arm's home pose, every hand joint at 0 (scene_utils init_state).
    default_q = [spec.arm_default_joint_pos[n] for n in arm] + [0.0] * len(hand)
    for n, q, lo, hi in zip(names, default_q, lower, upper):
        assert lo <= q <= hi, f"default {n}={q} outside [{lo}, {hi}]"
    # Arm effort: the actuator sets none, so PhysX uses the URDF's. Hand: robots.py's.
    effort = [limits[n]["effort"] for n in arm] + [spec.hand_effort_limit[n] for n in hand]

    sim, act, obs, rew = cfg["sim"], cfg["action"], cfg["obs"], cfg["reward"]
    reset, term, dr = cfg["reset"], cfg["termination"], cfg["domain_randomization"]
    table_size = [float(v) for v in ET.fromstring(git_show(args.ref, cfg["assets"]["table_urdf"]))
                  .find(".//collision/geometry/box").get("size").split()]

    out = {
        "task": TASK,
        "source_ref": args.ref,
        "source_commit": commit,
        "robot": ROBOT_NAME,
        "robot_urdf": Path(urdf_rel).name,
        "robot_urdf_in_source": urdf_rel,
        # --- joints, policy (canonical) order: Rizon joint1-7, then r_ Wuji joints thumb..pinky ---
        "joint_names": names,
        "num_arm_joints": len(arm),
        "joint_lower": lower,
        "joint_upper": upper,
        "default_q": default_q,
        "default_q_deg": [round(math.degrees(q), 3) for q in default_q],
        "stiffness": [spec.arm_stiffness[n] for n in arm] + [spec.hand_stiffness[n] for n in hand],
        "damping": [spec.arm_damping[n] for n in arm] + [spec.hand_damping[n] for n in hand],
        "effort_limit": effort,
        "hand_armature": [spec.hand_armature[n] for n in hand],
        "hand_velocity_limit": spec.hand_velocity_limit,
        "hand_friction": spec.hand_friction,
        "robot_gravity": "off in sim (the real arm's controller compensates it)",
        # --- frames ---
        "robot_root_pos_env": list(ROBOT_ROOT_POS),
        "robot_root_quat_wxyz": list(ROBOT_ROOT_QUAT_WXYZ),
        "palm_body": spec.palm_body_name,
        "palm_center_offset": list(spec.palm_center_offset),
        "fingertip_links": list(spec.fingertip_link_names),
        "fingertip_offsets": [list(o) for o in spec.fingertip_offsets],
        # --- action pipeline ---
        "policy_dt": float(sim["dt"]) * int(cfg["decimation"]),
        "sim_dt": float(sim["dt"]),
        "decimation": int(cfg["decimation"]),
        "dof_speed_scale": float(act["dof_speed_scale"]),
        "arm_moving_average": float(act["arm_moving_average"]),
        "hand_moving_average": float(act["hand_moving_average"]),
        # --- observation ---
        "obs_list": list(obs["obs_list"]),
        "obs_clip": float(obs["clamp_abs_observations"]),
        "quat_convention_obs": "xyzw",
        "keypoint_corners": keypoint_corners(args.ref),
        "object_base_size": float(rew["object_base_size"]),
        "keypoint_scale": float(rew["keypoint_scale"]),
        "keypoint_offsets_formula": "corner * object_scales * object_base_size * keypoint_scale / 2, in the object frame",
        # --- scene ---
        "table_size": table_size,
        "table_pos_env": [*reset["table_reset_xy"], float(reset["table_reset_z"])],
        "table_top_z_env": float(reset["table_reset_z"]) + table_size[2] / 2,
        "table_reset_z_range": float(reset["table_reset_z_range"]),
        # --- goals and success (training) ---
        "goal_sampling_type": reset["goal_sampling_type"],
        "delta_goal_distance": float(reset["delta_goal_distance"]),
        "delta_rotation_degrees": float(reset["delta_rotation_degrees"]),
        "target_volume_mins": list(reset["target_volume_mins"]),
        "target_volume_maxs": list(reset["target_volume_maxs"]),
        "target_volume_region_scale": float(reset["target_volume_region_scale"]),
        "use_goal_table_clearance": bool(reset["use_goal_table_clearance"]),
        "goal_table_clearance_m": float(reset["goal_table_clearance_m"]),
        "success_tolerance_start": float(term["success_tolerance"]),
        "success_tolerance_final": float(term["target_success_tolerance"]),
        "success_steps": int(term["success_steps"]),
        "episode_length_steps": int(term["episode_length"]),
        "fixed_size": list(rew["fixed_size"]),
        "fixed_size_keypoint_reward": bool(rew["fixed_size_keypoint_reward"]),
        # --- domain randomization the policy was trained under (what the real loop must stay inside) ---
        "dr": {k: dr[k] for k in ("obs_delay_max", "action_delay_max", "object_state_delay_max",
                                  "object_state_xyz_noise_std", "object_state_rotation_noise_degrees",
                                  "joint_velocity_obs_noise_std")},
    }

    if args.rollout_meta is not None:
        meta = json.loads(args.rollout_meta.read_text())
        assert meta["task"] == TASK and meta["palm_body"] == spec.palm_body_name, meta["task"]
        out["fingertip_obs_bodies"] = meta["fingertip_bodies_obs_order"]
        out["fingertip_obs_offsets"] = meta["fingertip_offsets_by_position"]
        out["fingertip_obs_source"] = str(args.rollout_meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    (args.out.parent / Path(urdf_rel).name).write_text(urdf_text)
    print(f"wrote {args.out} and {Path(urdf_rel).name} from {args.ref} @ {commit[:7]}")


if __name__ == "__main__":
    main()
