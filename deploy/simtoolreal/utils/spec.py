"""The spec: facts about the training env (joint limits, observation layout, action pipeline, goal
sampling, ...), exported from the training branch by checks/export_spec.py. Never edit
the json by hand.

    spec = load_spec(repo_path(cfg.spec_path))
    spec["joint_lower"]            # (27,) numpy array
    spec["urdf_path"]              # Path of the robot URDF (next to the json)

Frames. "env" is the training env's frame: the robot base sits at robot_root_pos_env (0, 0.8, 0),
the table top at z = 0.28. Everything on the bus is in the robot base frame; the policy works in
the env frame, so poses go through base_to_env on the way in and env_to_base on the way out.
"""

import json
from pathlib import Path

import numpy as np

from .quat import quat_apply, quat_conj, quat_mul

# json lists that are used as numbers
ARRAY_KEYS = [
    "joint_lower", "joint_upper", "default_q", "stiffness", "damping", "effort_limit",
    "robot_root_pos_env", "robot_root_quat_wxyz", "palm_center_offset", "fingertip_obs_offsets",
    "keypoint_corners", "fixed_size", "table_pos_env", "table_size",
    "target_volume_mins", "target_volume_maxs",
]


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for k in ARRAY_KEYS:
        spec[k] = np.asarray(spec[k], dtype=np.float64)
    spec["urdf_path"] = Path(path).parent / spec["robot_urdf"]
    return spec


def base_to_env(spec: dict, pos: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A pose in the robot base frame -> the same pose in the env frame."""
    root_p, root_q = spec["robot_root_pos_env"], spec["robot_root_quat_wxyz"]
    return root_p + quat_apply(root_q, pos), quat_mul(root_q, quat_wxyz)


def env_to_base(spec: dict, pos: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A pose in the env frame -> the same pose in the robot base frame."""
    root_p, inv_q = spec["robot_root_pos_env"], quat_conj(spec["robot_root_quat_wxyz"])
    return quat_apply(inv_q, pos - root_p), quat_mul(inv_q, quat_wxyz)
