"""The policy's 134-number observation, as the training env builds it (obs_utils.build_observations,
every DR term off). All poses in the env frame, quaternions wxyz.

    obs = build_obs(spec, model, q, dq, prev_targets, object_pos, object_quat,
                    goal_pos, goal_quat, object_scales)

Play: the 4 keypoints are grasp-box corners placed at the object and goal poses. Fold mode passes
keypoints=(object_kp, goal_kp) instead (utils/fold.py: the moving half's corners, and the same
corners at the observed goal); goal_pos / goal_quat are then unused.

| field                  | size | what                                                        |
|------------------------|------|-------------------------------------------------------------|
| joint_pos              | 27   | joint angles scaled to [-1, 1] across the joint limits      |
| joint_vel              | 27   | joint velocities, rad/s                                     |
| prev_action_targets    | 27   | the joint targets sent last step                            |
| palm_pos               | 3    | palm centre                                                 |
| palm_rot               | 4    | palm quat, xyzw                                             |
| object_rot             | 4    | object quat, xyzw                                           |
| fingertip_pos_rel_palm | 15   | 5 fingertips - palm centre                                  |
| keypoints_rel_palm     | 12   | 4 object keypoints - palm centre                            |
| keypoints_rel_goal     | 12   | 4 object keypoints - the same 4 keypoints at the goal pose  |
| object_scales          | 3    | grasp-box scale (DexToolBench Object.scale), not metres     |

The order comes from spec["obs_list"]; every number is clipped to +-spec["obs_clip"].
"""

import numpy as np

from .quat import quat_apply, wxyz_to_xyzw
from .robot_model import fingertips, palm_center

NUM_OBS = 134


def keypoint_offsets(spec: dict, object_scales: np.ndarray) -> np.ndarray:
    """(4, 3) observation keypoints in the object frame: 4 corners of the grasp box x keypoint_scale."""
    half_size = np.asarray(object_scales) * spec["object_base_size"] * spec["keypoint_scale"] * 0.5
    return spec["keypoint_corners"] * half_size


def keypoints_world(pos: np.ndarray, quat_wxyz: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """(4, 3) keypoints placed at an object pose."""
    return pos + quat_apply(np.broadcast_to(quat_wxyz, offsets.shape[:-1] + (4,)), offsets)


def obs_fields(spec, model, q, dq, prev_targets, object_pos, object_quat, goal_pos, goal_quat,
               object_scales, keypoints=None) -> dict:
    """Each observation field by name (see the table above). keypoints: (object_kp, goal_kp), each
    (4, 3), in place of the grasp-box corners (fold mode)."""
    lower, upper = spec["joint_lower"], spec["joint_upper"]
    palm_pos, palm_quat = palm_center(spec, model, q)
    if keypoints is None:
        kp = keypoint_offsets(spec, object_scales)
        object_kp = keypoints_world(object_pos, object_quat, kp)
        goal_kp = keypoints_world(goal_pos, goal_quat, kp)
    else:
        object_kp, goal_kp = (np.asarray(k, dtype=np.float64) for k in keypoints)
    return {
        "joint_pos": 2.0 * (q - lower) / (upper - lower) - 1.0,
        "joint_vel": dq,
        "prev_action_targets": prev_targets,
        "palm_pos": palm_pos,
        "palm_rot": wxyz_to_xyzw(palm_quat),
        "object_rot": wxyz_to_xyzw(object_quat),
        "fingertip_pos_rel_palm": (fingertips(spec, model, q) - palm_pos).reshape(-1),
        "keypoints_rel_palm": (object_kp - palm_pos).reshape(-1),
        "keypoints_rel_goal": (object_kp - goal_kp).reshape(-1),
        "object_scales": np.asarray(object_scales),
    }


def build_obs(spec, model, q, dq, prev_targets, object_pos, object_quat, goal_pos, goal_quat,
              object_scales, keypoints=None) -> np.ndarray:
    """(134,) float32, the fields in spec["obs_list"] order, clipped."""
    fields = obs_fields(spec, model, q, dq, prev_targets, object_pos, object_quat, goal_pos, goal_quat,
                        object_scales, keypoints)
    obs = np.concatenate([np.asarray(fields[name], dtype=np.float64) for name in spec["obs_list"]])
    assert obs.shape == (NUM_OBS,), obs.shape
    return np.clip(obs, -spec["obs_clip"], spec["obs_clip"]).astype(np.float32)
