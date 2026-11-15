"""Build a spec json (the training env's facts the deploy code reads) from a training branch, read
straight from git with no Isaac Sim. The tasks' exporters (simtoolreal/checks/export_spec.py,
rigid2deform/checks/export_fold_spec.py) choose the branch, the task and the frames, and call
spec_fields; this has what they share.

    git_show(ref, path)          a file of the branch, as text
    spec_fields(ref, commit, task, robot_name, robot_spec, env_cfg, robot_root_pos=..., table_pos_env=...,
                table_top_z_env=...)      robot, action pipeline, observation, table, goals, success, DR
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import REPO

ROBOTS_PY = "isaacsimenvs/tasks/play/robots.py"
OBS_UTILS_PY = "isaacsimenvs/tasks/play/utils/obs_utils.py"
PLAY_ENV_CFG_PY = "isaacsimenvs/tasks/play/play_env_cfg.py"
# Robot root in the env frame (scene_utils.build_robot_articulation_usd_cfg), the same for every robot.
ROBOT_ROOT_POS = (0.0, 0.8, 0.0)
ROBOT_ROOT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def git_show(ref: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],
                          check=True, capture_output=True, text=True).stdout


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


PLAY_ENV_CFG_PY = "isaacsimenvs/tasks/play/play_env_cfg.py"


def cfg_default(ref: str, path: str, field: str):
    """A configclass field's default, read as a literal (the file imports Isaac Lab). The field name
    must not repeat in the file. None when the file has no such field."""
    found = re.findall(rf"^\s+{field}:\s*[^=\n]+=\s*(.+?)\s*$", git_show(ref, path), re.M)
    assert len(found) <= 1, f"{path}: {len(found)} defaults for {field}"
    return ast.literal_eval(found[0]) if found else None


def section_get(ref: str, section: dict, key: str):
    """section[key], or play_env_cfg.py's default when the task YAML leaves it out (None: neither
    has it -- a Play goal-sampling field the fold task does not use)."""
    return section[key] if key in section else cfg_default(ref, PLAY_ENV_CFG_PY, key)


def optional(cast, value):
    return None if value is None else cast(value)


def urdf_joints(urdf_text: str) -> dict[str, dict]:
    out = {}
    for j in ET.fromstring(urdf_text).iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "prismatic") and lim is not None:
            out[j.get("name")] = {k: float(lim.get(k)) for k in ("lower", "upper", "effort", "velocity")
                                  if lim.get(k) is not None}
    return out


def rev_parse(ref: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", ref],
                          check=True, capture_output=True, text=True).stdout.strip()


def table_box(ref: str, cfg: dict) -> list[float]:
    """The table URDF's collision box size [x, y, z], m."""
    return [float(v) for v in ET.fromstring(git_show(ref, cfg["assets"]["table_urdf"]))
            .find(".//collision/geometry/box").get("size").split()]


def spec_fields(ref: str, commit: str, task: str, robot_name: str, spec, cfg: dict, *,
                robot_root_pos: tuple, table_pos_env: list, table_top_z_env: float) -> dict:
    """The spec dict for one training env: robot (from robots.py + the URDF), action pipeline,
    observation, table, goal sampling, success test, DR. `cfg` is the env config as the task ran it
    (task YAML, plus any overlays); the table and the robot root are passed in, in the env frame the
    deploy code uses (the robot base at robot_root_pos)."""
    urdf_rel = cfg["assets"]["robot_urdf"]
    urdf_text = git_show(ref, urdf_rel)
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
    # Newer robots.py: fingertip_offset is one 3-vector or five (one per fingertip link); older: a property.
    if hasattr(spec, "fingertip_offsets"):
        tip_offsets = [list(o) for o in spec.fingertip_offsets]
    else:
        off = spec.fingertip_offset
        tip_offsets = [list(o) for o in off] if len(off) == 5 else [list(off)] * 5

    return {
        "task": task,
        "source_ref": ref,
        "source_commit": commit,
        "robot": robot_name,
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
        "robot_root_pos_env": list(robot_root_pos),
        "robot_root_quat_wxyz": list(ROBOT_ROOT_QUAT_WXYZ),
        "palm_body": spec.palm_body_name,
        "palm_center_offset": list(spec.palm_center_offset),
        "fingertip_links": list(spec.fingertip_link_names),
        "fingertip_offsets": tip_offsets,
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
        "keypoint_corners": keypoint_corners(ref),
        "object_base_size": float(rew["object_base_size"]),
        "keypoint_scale": float(rew["keypoint_scale"]),
        "keypoint_offsets_formula": "corner * object_scales * object_base_size * keypoint_scale / 2, in the object frame",
        # --- scene ---
        "table_size": table_box(ref, cfg),
        "table_pos_env": list(table_pos_env),
        "table_top_z_env": float(table_top_z_env),
        "table_reset_z_range": float(reset["table_reset_z_range"]),
        # --- goals and success (training) ---
        "goal_sampling_type": reset["goal_sampling_type"],
        "delta_goal_distance": float(reset["delta_goal_distance"]),
        "delta_rotation_degrees": float(reset["delta_rotation_degrees"]),
        "target_volume_mins": list(reset["target_volume_mins"]),
        "target_volume_maxs": list(reset["target_volume_maxs"]),
        "target_volume_region_scale": float(reset["target_volume_region_scale"]),
        "use_goal_table_clearance": optional(bool, section_get(ref, reset, "use_goal_table_clearance")),
        "goal_table_clearance_m": optional(float, section_get(ref, reset, "goal_table_clearance_m")),
        "success_tolerance_start": float(term["success_tolerance"]),
        "success_tolerance_final": float(term["target_success_tolerance"]),
        "success_steps": int(term["success_steps"]),
        "episode_length_steps": int(term["episode_length"]),
        "fixed_size": list(rew["fixed_size"]),
        "fixed_size_keypoint_reward": bool(rew["fixed_size_keypoint_reward"]),
        # --- domain randomization the policy was trained under (what the real loop must stay inside) ---
        "dr": {k: section_get(ref, dr, k) for k in ("obs_delay_max", "action_delay_max", "object_state_delay_max",
                                  "object_state_xyz_noise_std", "object_state_rotation_noise_degrees",
                                  "joint_velocity_obs_noise_std")},
    }
