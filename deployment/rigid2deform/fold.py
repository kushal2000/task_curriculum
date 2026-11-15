"""task=fold: the sheet's grid points (particles) -> what the fold-pool policy observes, and the success test.

The fold-pool task (branch yiboc_wuji, cloth_task.py / fold_pool_task.py / cloth_adapter.py) scores
a SHEET: a resolution x resolution grid of points (spec["fold"]["grid_rest"], in the sheet frame: x
across the hinge, from the stationary half to the moving one; y along the hinge; z up; origin on the
hinge line at the sheet's mid-plane). From the grid's current positions it builds:

  object  the moving half's 4 corners (corner_idx): position = their centroid, orientation = the
          rotation fitted from their rest offsets (cloth_adapter.fit_rotation_wxyz: Kabsch), as wxyz
          by cloth_adapter._mat_to_wxyz (its sign convention reaches the observation: object_rot).
  goal    the same 4 corners reflected across the hinge (x -> -x, + goal_lift), carried by the
          stationary half's rigid frame (fold_pool_task._stationary_frame: Kabsch of its points);
          position = their centroid, orientation = R_stationary x (180 deg about the hinge). The
          OBSERVED goal quaternion is that one's components relabelled ([1, 2, 3, 0], read as wxyz;
          physx_envs._goal_quat_from_matrix), so the goal keypoints the policy saw were built with
          it. Only keypoints_rel_goal carries the goal.
  success the corners within keypoint_tolerance of the fold target AND the sheet's extent along
          the stationary frame's x at most max_folded_footprint x size (is_folded).

Where the particles come from is the sheet source's business (sheets/: two FoundationPose-tracked
plates, or tracked cloth points). A source may also lift the success target off training's (the
real plates' hinge is above their mid-plane: target_lift).

    geom = fold_geometry(spec, sheet_size)          # the rest geometry (a dict)
    o = observe(geom, particles)                    # (P, 3) env frame -> object / goal / keypoints
    sc = score(geom, o, target_lift)                # fold_error(_real), footprint, folded

Every pose here is in the env frame (spec["robot_root_pos_env"]), quaternions wxyz.
"""

from __future__ import annotations

import numpy as np

from deploy.utils.quat import quat_apply

FOLD_FLIP = np.diag([-1.0, 1.0, -1.0])    # 180 deg about the sheet's y (the hinge), fold_axis x


def mat_to_wxyz(r: np.ndarray) -> np.ndarray:
    """cloth_adapter._mat_to_wxyz, for one matrix: w > 0 when the trace is, else the component of
    the largest diagonal entry > 0. Kept branch for branch: the sign is part of object_rot."""
    q = np.zeros(4)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        sq = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * sq
        q[1] = (r[2, 1] - r[1, 2]) / sq
        q[2] = (r[0, 2] - r[2, 0]) / sq
        q[3] = (r[1, 0] - r[0, 1]) / sq
    else:
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            if r[i, i] >= r[j, j] and r[i, i] >= r[k, k]:
                sq = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
                q[0] = (r[k, j] - r[j, k]) / sq
                q[1 + i] = 0.25 * sq
                q[1 + j] = (r[j, i] + r[i, j]) / sq
                q[1 + k] = (r[k, i] + r[i, k]) / sq
                break
    return q / max(np.linalg.norm(q), 1e-12)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz -> rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """rad between two rotation matrices."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


def kabsch(rest_centred: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, centroid): the rotation taking rest_centred (K, 3, centred) onto cur (K, 3), and cur's
    centroid. As training's fit_rotation_wxyz / _stationary_frame: H = P^T Q + 1e-8 I,
    R = V diag(1, 1, det(V U^T)) U^T (no reflection); non-finite -> identity."""
    centroid = cur.mean(axis=0)
    h = rest_centred.T @ (cur - centroid) + 1e-8 * np.eye(3)
    u, _, vh = np.linalg.svd(h)
    v = vh.T
    r = v @ np.diag([1.0, 1.0, np.linalg.det(v @ u.T)]) @ u.T
    return (r if np.all(np.isfinite(r)) else np.eye(3)), centroid


def fold_geometry(spec: dict, sheet_size: float) -> dict:
    """The training task's sheet, scaled to the real sheet (sheet_size: its side, m): the rest geometry
    observe / score / sheet_points read.

    grid (P, 3) sheet frame; on_moving (P,) bool; corner_idx, stationary_idx; corner_offsets (4, 3);
    st_centroid, st_centred; kp_folded_rest (4, 3); relabel; tolerance; max_footprint; object_scales;
    size."""
    f = spec["fold"]
    lo, hi = f["size_range"]
    if not lo - 1e-9 <= sheet_size <= hi + 1e-9:
        raise SystemExit(f"[fold] sheet_size={sheet_size} m outside the training range {f['size_range']}")
    size = float(sheet_size)
    grid = np.asarray(f["grid_rest"], dtype=np.float64)
    grid[:, :2] *= size / f["sheet_size_nominal"]               # fold_pool_task: rest geometry x _scale3
    corner_idx = np.asarray(f["corner_idx"])
    stationary_idx = np.asarray(f["stationary_idx"])
    corners = grid[corner_idx]
    st = grid[stationary_idx]
    folded = corners.copy()                                     # cloth_task._folded_rest
    folded[:, 0] = -folded[:, 0]                                # reflect across the crease (x = 0)
    folded[:, 2] += f["goal_lift"]
    return {
        "size": size,
        "grid": grid,
        "on_moving": np.array([p == f["moving_plate"] for p in f["grid_plate"]]),
        "corner_idx": corner_idx,
        "stationary_idx": stationary_idx,
        "corner_offsets": corners - corners.mean(axis=0),       # _corner_rest_offsets
        "st_centroid": st.mean(axis=0),                         # _st_rest_centroid
        "st_centred": st - st.mean(axis=0),                     # _st_rest_centred
        "kp_folded_rest": folded,
        "relabel": list(f["goal_quat_relabel"]),
        "tolerance": float(f["keypoint_tolerance"]),
        "max_footprint": float(f["max_folded_footprint"]),
        # object_scales: fold_pool_task._write_object_scale
        "object_scales": np.array([size, size, f["object_scale_z_m"]]) / spec["object_base_size"],
    }


def sheet_points(geom: dict, origin: np.ndarray, r_sheet: np.ndarray) -> np.ndarray:
    """(P, 3) the flat grid at a sheet pose (origin = hinge line middle, mid-plane)."""
    return origin + geom["grid"] @ r_sheet.T


# --- what the policy observes ---------------------------------------------------------------
def observe(geom: dict, parts: np.ndarray) -> dict:
    """object pose, its 4 keypoints and the goal's, as the training env computes them from the
    particles (P, 3), env frame."""
    parts = np.asarray(parts, dtype=np.float64)
    obj_kp = parts[geom["corner_idx"]]
    r_m, obj_pos = kabsch(geom["corner_offsets"], obj_kp)
    obj_quat = mat_to_wxyz(r_m)                         # fit_rotation_wxyz
    # fold target: the stationary half's frame (_stationary_frame), _folded_w
    r_s, st_c = kabsch(geom["st_centred"], parts[geom["stationary_idx"]])
    targets = (geom["kp_folded_rest"] - geom["st_centroid"]) @ r_s.T + st_c
    goal_pos = targets.mean(axis=0)
    q_true = mat_to_wxyz(r_s @ FOLD_FLIP)
    q_obs = q_true[geom["relabel"]]
    q_obs = q_obs / np.linalg.norm(q_obs)
    goal_kp = goal_pos + quat_apply(np.broadcast_to(q_obs, (4, 4)), geom["corner_offsets"])
    return {"object_pos": obj_pos, "object_quat": obj_quat, "object_kp": obj_kp,
            "goal_pos": goal_pos, "goal_quat_true": q_true, "goal_quat_obs": q_obs, "goal_kp": goal_kp,
            "targets": targets, "particles": parts, "r_stationary": r_s}


# --- success --------------------------------------------------------------------------------
def score(geom: dict, obs: dict, target_lift: float = 0.0) -> dict:
    """fold_error (training's target), fold_error_real (the target target_lift higher along the
    stationary half's normal, where the real sheet lands), footprint (extent along the stationary
    x / size), folded (the real test)."""
    r_s = obs["r_stationary"]
    kp, targets = obs["object_kp"], obs["targets"]
    real_targets = targets + target_lift * r_s[:, 2]
    err = float(np.linalg.norm(kp - targets, axis=1).max())
    err_real = float(np.linalg.norm(kp - real_targets, axis=1).max())
    coord = obs["particles"] @ r_s[:, 0]
    footprint = float((coord.max() - coord.min()) / geom["size"])
    return {"fold_error": err, "fold_error_real": err_real, "footprint": footprint,
            "folded": err_real < geom["tolerance"] and footprint <= geom["max_footprint"]}
