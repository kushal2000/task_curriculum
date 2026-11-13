"""Fold mode: two tracked plates taped along one edge -> what the fold-pool policy observes.

The fold-pool task (branch yiboc_wuji, cloth_task.py / fold_pool_task.py / cloth_adapter.py) scores
a SHEET: a resolution x resolution grid of points carried by the plates (spec["fold"]["grid_rest"],
in the sheet frame: x across the hinge, from the stationary plate to the moving one; y along the
hinge; z up; origin on the hinge line at the plates' mid-plane). From that grid it builds:

  object  the moving half's 4 corners (corner_idx): position = their centroid, orientation = the
          rotation fitted from their rest offsets (= the moving plate's rotation), as wxyz by
          cloth_adapter._mat_to_wxyz (its sign convention reaches the observation: object_rot).
  goal    the same 4 corners reflected across the hinge (x -> -x, + goal_lift), carried by the
          stationary half's rigid frame; position = their centroid, orientation = R_stationary x
          (180 deg about the hinge). The OBSERVED goal quaternion is that one's components
          relabelled ([1, 2, 3, 0], read as wxyz; physx_envs._goal_quat_from_matrix), so the goal
          keypoints the policy saw were built with it. Only keypoints_rel_goal carries the goal.
  success the corners within keypoint_tolerance of the fold target AND the sheet's extent along
          the stationary frame's x at most max_folded_footprint x size (is_folded).

The real plates: each tracked by FoundationPose as its own mesh (a box centred on its origin). A
plate's pose gives its sheet frame up to the box's symmetries (identity, 180 deg about each mesh
axis); PlateFrames picks the right one -- at the start from the geometry (both lying flat, hinge
edges together), then by continuity.

Every pose here is in the env frame (spec["robot_root_pos_env"]), quaternions wxyz.
"""

from __future__ import annotations

import numpy as np

from .quat import quat_apply

# Rotation symmetries of a box centred on its origin: identity and 180 deg about each axis.
BOX_SYMMETRIES = (np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]))
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


def mesh_to_sheet(hinge_axis_mesh: str) -> np.ndarray:
    """M with R_sheet = R_mesh @ M: the sheet axes in the plate mesh's frame. hinge_axis_mesh is the
    mesh axis that runs along the hinge (plate5: x, its 10 cm side); the sheet's x is the mesh's other
    in-plane axis, z the mesh's z."""
    if hinge_axis_mesh == "x":
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # sheet x = mesh y, y = -mesh x
    if hinge_axis_mesh == "y":
        return np.eye(3)
    raise ValueError(f"fold.hinge_axis_mesh={hinge_axis_mesh}: x | y")


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """rad between two rotation matrices."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


class FoldGeometry:
    """The training task's sheet, scaled to the real sheet, and everything computed from it.

    sheet_size: the real sheet's side, m (two plates of sheet_size x sheet_size / 2).
    hinge_z: the real hinge line above the plates' mid-plane, m (tape on the top face of 5 mm plates:
        0.0025). Training's hinges are on the mid-plane (0): the OBSERVED goal stays training's, the
        success test uses the real folded position, 2 x hinge_z higher.
    """

    def __init__(self, spec: dict, sheet_size: float, hinge_z: float = 0.0):
        f = spec["fold"]
        lo, hi = f["size_range"]
        if not lo - 1e-9 <= sheet_size <= hi + 1e-9:
            raise SystemExit(f"[fold] fold.sheet_size={sheet_size} m outside the training range {f['size_range']}")
        self.size = float(sheet_size)
        self.hinge_z = float(hinge_z)
        k = self.size / f["sheet_size_nominal"]
        grid = np.asarray(f["grid_rest"], dtype=np.float64)
        grid[:, :2] *= k                                   # fold_pool_task: rest geometry x _scale3
        self.grid = grid                                   # (P, 3) sheet frame
        self.on_moving = np.array([p == f["moving_plate"] for p in f["grid_plate"]])
        self.corner_idx = np.asarray(f["corner_idx"])
        self.stationary_idx = np.asarray(f["stationary_idx"])
        corners = grid[self.corner_idx]
        self.corner_offsets = corners - corners.mean(axis=0)          # _corner_rest_offsets
        self.st_centroid = grid[self.stationary_idx].mean(axis=0)      # _st_rest_centroid
        folded = corners.copy()                                        # cloth_task._folded_rest
        folded[:, 0] = -folded[:, 0]                                   # reflect across the crease (x = 0)
        folded[:, 2] += f["goal_lift"]
        self.kp_folded_rest = folded
        self.relabel = list(f["goal_quat_relabel"])
        self.tolerance = float(f["keypoint_tolerance"])
        self.max_footprint = float(f["max_folded_footprint"])
        # object_scales: fold_pool_task._write_object_scale
        self.object_scales = np.array([self.size, self.size, f["object_scale_z_m"]]) / spec["object_base_size"]
        half = 0.25 * self.size                                         # plate centre, |x| in the sheet frame
        self.plate_centre = {"moving": np.array([half, 0.0, 0.0]), "stationary": np.array([-half, 0.0, 0.0])}

    # --- the sheet ---------------------------------------------------------------------------
    def particles(self, moving: tuple, stationary: tuple) -> np.ndarray:
        """(P, 3) the grid in the env frame. moving / stationary: (centre, R_sheet) of each plate."""
        out = np.empty_like(self.grid)
        for (c, r), which, sel in ((moving, "moving", self.on_moving), (stationary, "stationary", ~self.on_moving)):
            out[sel] = c + (self.grid[sel] - self.plate_centre[which]) @ r.T
        return out

    def hinge_point(self, which: str, centre: np.ndarray, r_sheet: np.ndarray) -> np.ndarray:
        """Where this plate puts the hinge line's middle (sheet origin), env frame."""
        return centre - r_sheet @ self.plate_centre[which]

    # --- what the policy observes ---------------------------------------------------------
    def observe(self, moving: tuple, stationary: tuple) -> dict:
        """object pose, its 4 keypoints and the goal's, as the training env computes them."""
        parts = self.particles(moving, stationary)
        r_m, r_s = moving[1], stationary[1]
        obj_kp = parts[self.corner_idx]
        obj_pos = obj_kp.mean(axis=0)
        obj_quat = mat_to_wxyz(r_m)                         # = the Kabsch fit of rigid corners
        # fold target: the stationary half's frame (rotation r_s, its particles' centroid)
        st_c = parts[self.stationary_idx].mean(axis=0)
        targets = (self.kp_folded_rest - self.st_centroid) @ r_s.T + st_c
        goal_pos = targets.mean(axis=0)
        q_true = mat_to_wxyz(r_s @ FOLD_FLIP)
        q_obs = q_true[self.relabel]
        q_obs = q_obs / np.linalg.norm(q_obs)
        goal_kp = goal_pos + quat_apply(np.broadcast_to(q_obs, (4, 4)), self.corner_offsets)
        return {"object_pos": obj_pos, "object_quat": obj_quat, "object_kp": obj_kp,
                "goal_pos": goal_pos, "goal_quat_true": q_true, "goal_quat_obs": q_obs, "goal_kp": goal_kp,
                "targets": targets, "particles": parts}

    # --- success ----------------------------------------------------------------------------
    def score(self, obs: dict, stationary: tuple) -> dict:
        """fold_error (training's target), fold_error_real (the target 2 x hinge_z higher, where the
        real plates land), footprint (extent along the stationary x / size), folded (the real test)."""
        r_s = stationary[1]
        kp, targets = obs["object_kp"], obs["targets"]
        real_targets = targets + 2.0 * self.hinge_z * r_s[:, 2]
        err = float(np.linalg.norm(kp - targets, axis=1).max())
        err_real = float(np.linalg.norm(kp - real_targets, axis=1).max())
        coord = obs["particles"] @ r_s[:, 0]
        footprint = float((coord.max() - coord.min()) / self.size)
        return {"fold_error": err, "fold_error_real": err_real, "footprint": footprint,
                "folded": err_real < self.tolerance and footprint <= self.max_footprint}

    def goal_plate_pose(self, stationary: tuple, m2s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where the moving plate's MESH sits when folded (the real fold, hinge_z): for the viewer."""
        c_s, r_s = stationary
        # turned 180 deg about the hinge line (sheet z = hinge_z): (x, z) -> (-x, 2 hinge_z - z), so the
        # moving centre (+half, 0, 0) lands on the stationary centre, 2 hinge_z higher
        centre = c_s + 2.0 * self.hinge_z * r_s[:, 2]
        return centre, mat_to_wxyz(r_s @ FOLD_FLIP @ m2s.T)

    def plate_poses(self, origin: np.ndarray, r_sheet: np.ndarray, fold_rad: float, m2s: np.ndarray):
        """Mesh poses of both plates for a sheet at origin / r_sheet with the moving plate turned up by
        fold_rad about the (real) hinge line. For the fake perception source and the checks.
        Returns {"moving": (centre, quat_wxyz), "stationary": (...)}."""
        out = {}
        c_s = origin + r_sheet @ self.plate_centre["stationary"]
        out["stationary"] = (c_s, mat_to_wxyz(r_sheet @ m2s.T))
        # up = about the sheet's -y, through the hinge line at height hinge_z
        a = -fold_rad
        rot = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
        pivot = np.array([0.0, 0.0, self.hinge_z])
        c_local = pivot + rot @ (self.plate_centre["moving"] - pivot)
        r_m = r_sheet @ rot
        out["moving"] = (origin + r_sheet @ c_local, mat_to_wxyz(r_m @ m2s.T))
        return out


class PlateFrames:
    """The sheet frame of each plate from its tracked mesh pose, through the box's symmetries.

    start(): both plates lying flat (sheet z within max_tilt of up), hinge edges together: the one
    symmetry pair that puts both plates' hinge points within max_gap and z up. update(): per plate,
    the symmetry closest to the last frame (FoundationPose may jump between symmetric poses).
    """

    def __init__(self, geom: FoldGeometry, m2s: np.ndarray, max_gap: float, max_tilt_deg: float = 30.0):
        self.geom, self.m2s = geom, m2s
        self.max_gap, self.cos_tilt = max_gap, np.cos(np.radians(max_tilt_deg))
        self.r = None   # {"moving": R_sheet, "stationary": R_sheet}

    def _candidates(self, quat_wxyz) -> list[np.ndarray]:
        r_mesh = quat_to_mat(quat_wxyz)
        return [r_mesh @ s @ self.m2s for s in BOX_SYMMETRIES]

    def start(self, moving: tuple, stationary: tuple) -> dict:
        """moving / stationary: (centre, mesh quat wxyz). Raises SystemExit if they are not lying flat
        side by side, hinge edges together."""
        best = None
        for r_m in self._candidates(moving[1]):
            for r_s in self._candidates(stationary[1]):
                if r_m[2, 2] < self.cos_tilt or r_s[2, 2] < self.cos_tilt:
                    continue
                gap = np.linalg.norm(self.geom.hinge_point("moving", moving[0], r_m)
                                     - self.geom.hinge_point("stationary", stationary[0], r_s))
                if best is None or gap < best[0]:
                    best = (gap, r_m, r_s)
        if best is None:
            raise SystemExit("[fold] the plates are not lying flat (each plate's face more than "
                             f"{np.degrees(np.arccos(self.cos_tilt)):.0f} deg from level)")
        if best[0] > self.max_gap:
            raise SystemExit(f"[fold] the plates' hinge edges are {100 * best[0]:.1f} cm apart (limit "
                             f"{100 * self.max_gap:.1f} cm): not side by side, or the wrong plate size / hinge axis")
        self.r = {"moving": best[1], "stationary": best[2]}
        return self.frames(moving, stationary)

    def update(self, moving: tuple, stationary: tuple) -> dict:
        for which, (_, quat) in (("moving", moving), ("stationary", stationary)):
            self.r[which] = min(self._candidates(quat), key=lambda c: rotation_angle(c, self.r[which]))
        return self.frames(moving, stationary)

    def frames(self, moving: tuple, stationary: tuple) -> dict:
        """{"moving": (centre, R_sheet), "stationary": (...), "hinge_gap": m}."""
        m = (np.asarray(moving[0], dtype=np.float64), self.r["moving"])
        s = (np.asarray(stationary[0], dtype=np.float64), self.r["stationary"])
        gap = float(np.linalg.norm(self.geom.hinge_point("moving", *m) - self.geom.hinge_point("stationary", *s)))
        return {"moving": m, "stationary": s, "hinge_gap": gap}
