"""Goals: where the policy should move the object next, and whether it got there. Env frame, wxyz.

Sampling (training's reset_utils._reset_goal_pose):
    pos, quat = first_goal(spec, rng, corners)                  # uniform in the target volume, random rotation
    pos, quat = next_goal(spec, rng, corners, pos, quat)        # the last goal moved a little
    clearance=<m>: the object's lowest corner at least this far above the table top (default: training's)

Success (training's compute_intermediate_values + termination_utils):
    near = is_near(spec, obj_pos, obj_quat, goal_pos, goal_quat, tolerance)
    near_steps += near                  # in total since this goal was set, not necessarily in a row
    reached = near_steps >= spec["success_steps"]   # 10

corners: (8, 3) the object's bounding box in its own frame (object_corners), used to keep goals
above the table.
"""

import numpy as np

from .observation import keypoints_world
from .quat import quat_apply, quat_from_angle_axis, quat_mul, random_quat


def object_corners(mesh_path) -> np.ndarray:
    """(8, 3) bounding-box corners of an .obj mesh (in metres) in its own frame."""
    with open(mesh_path) as f:
        v = np.array([line.split()[1:4] for line in f if line.startswith("v ")], dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def first_goal(spec: dict, rng: np.random.Generator, corners: np.ndarray,
               clearance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Uniform in the target volume, uniformly random rotation."""
    lo, hi = _volume(spec)
    pos = lo + (hi - lo) * rng.random(3)
    quat = random_quat(rng)
    return _above_table(spec, corners, pos, quat, clearance), quat


def next_goal(spec: dict, rng: np.random.Generator, corners: np.ndarray,
              pos: np.ndarray, quat: np.ndarray, clearance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The last goal moved by up to delta_goal_distance per axis (kept in the volume) and turned by up
    to delta_rotation_degrees about a random axis."""
    lo, hi = _volume(spec)
    d = spec["delta_goal_distance"]
    new_pos = np.clip(pos + (rng.random(3) * 2 - 1) * d, lo, hi)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = (rng.random() * 2 - 1) * np.radians(spec["delta_rotation_degrees"])
    new_quat = quat_mul(quat_from_angle_axis(angle, axis), quat)
    return _above_table(spec, corners, new_pos, new_quat, clearance), new_quat


def keypoint_distance(spec: dict, obj_pos, obj_quat, goal_pos, goal_quat) -> float:
    """The largest distance (m) between the object's 4 success keypoints and the goal's.
    Success keypoints have one fixed size for every object (fixed_size), unlike the observation's."""
    kp = spec["keypoint_corners"] * (0.5 * spec["keypoint_scale"] * spec["fixed_size"])
    a = keypoints_world(np.asarray(obj_pos), np.asarray(obj_quat), kp)
    b = keypoints_world(np.asarray(goal_pos), np.asarray(goal_quat), kp)
    return float(np.linalg.norm(a - b, axis=-1).max())


def is_near(spec: dict, obj_pos, obj_quat, goal_pos, goal_quat, tolerance: float) -> bool:
    """Within tolerance (m, before keypoint_scale) of the goal."""
    return keypoint_distance(spec, obj_pos, obj_quat, goal_pos, goal_quat) <= tolerance * spec["keypoint_scale"]


def _volume(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """The target volume (min, max corners), shrunk about its centre by target_volume_region_scale."""
    mins, maxs = spec["target_volume_mins"], spec["target_volume_maxs"]
    centre, half = 0.5 * (mins + maxs), 0.5 * (maxs - mins) * spec["target_volume_region_scale"]
    return centre - half, centre + half


def _above_table(spec: dict, corners: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                 clearance: float | None = None) -> np.ndarray:
    """Raise pos until the object's lowest corner is clearance (None: the spec's goal_table_clearance_m)
    above the table top. An explicit clearance applies even if training turned the clearance off."""
    if clearance is None:
        if not spec["use_goal_table_clearance"]:
            return pos
        clearance = spec["goal_table_clearance_m"]
    lowest = quat_apply(np.broadcast_to(quat, corners.shape[:-1] + (4,)), corners)[:, 2].min()
    min_z = spec["table_top_z_env"] + clearance - lowest
    return np.array([pos[0], pos[1], max(pos[2], min_z)])


def rollout_goals(path) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """The goals of a sim rollout (checks/record_lift.py --goals train: openloop.npz), in the order the
    sim had them, each once; and the object's start pose. Env frame, the mesh's own frame (the one
    perception reports), wxyz."""
    d = np.load(path)
    if "goals_mesh_pos" not in d.files:
        raise SystemExit(f"[policy] {path}: no goals_mesh_pos (recorded without --goals train)")
    pos, quat = d["goals_mesh_pos"].astype(np.float64), d["goals_quat"].astype(np.float64)
    change = np.r_[True, np.any(np.abs(np.diff(np.c_[pos, quat], axis=0)) > 1e-9, axis=1)]
    goals = [(pos[i], quat[i]) for i in np.flatnonzero(change)]
    return goals, (d["start_mesh_pos"].astype(np.float64), d["start_quat"].astype(np.float64))
