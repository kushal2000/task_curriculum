"""Interactive HTML capture for cloth-folding training, logged to wandb.

Extends the Play pose viewer with the one thing a rigid-pose viewer cannot show: the sheet. A cloth
has no root transform -- its vertex positions *are* its state -- so it travels as a deformable
channel (static triangle indices plus a per-frame vertex trajectory) rather than as a (T, 7) pose.

**Nothing is uploaded to wandb as an asset.** The sheet is a procedural grid, so its geometry is
plain numbers in the payload: 81 vertices x 3 floats per frame, order 1 KB. The robot's ~41 MB of
STLs are fetched by the browser from the public repo via `--capture_viewer_github_raw_base`, exactly
as the Play viewer already does.

Three things are drawn that the Play viewer has no notion of:

* the moving half, in the colour the renders use for it
* the stationary half, likewise
* the fold target as a translucent ghost -- the same `folded_half_w()` the reward and the success
  criterion are computed from, so what is on screen is what is being optimised
"""

from __future__ import annotations

from typing import Any

from isaacsimenvs.tasks.play.pose_viewer import (
    PlayPoseViewerWrapper,
    build_pose_viewer_html,
    capture_pose_viewer_frame,
)

__all__ = ["ClothPoseViewerWrapper", "capture_cloth_viewer_frame"]

#: Matches the offscreen renderer (`eval/render_newton.py`) so a wandb clip and an mp4 of the same
#: run are directly comparable. Nothing here may be blue -- the table, floor and sky all are.
MOVING_COLOR = (0.95, 0.30, 0.10)      # vermilion: the half that must travel
STATIONARY_COLOR = (0.98, 0.78, 0.15)  # gold: the half that must not
GOAL_COLOR = (0.15, 0.85, 0.35)        # green: where the moving half must end up
GOAL_OPACITY = 0.45


def capture_cloth_viewer_frame(env, env_id: int) -> dict[str, Any]:
    """One frame: everything the Play viewer captures, plus the sheet and its fold target."""
    frame = capture_pose_viewer_frame(env, env_id)
    origin = env.scene.env_origins[env_id]

    parts = env._particles_w()[env_id] - origin
    frame["cloth_vertices"] = parts.detach().cpu().numpy().tolist()
    # The ghost is the goal, over every particle of the moving half rather than the four scored
    # keypoints -- `folded_half_w` shares `_folded_w` with `fold_targets_w`, so the drawing cannot
    # drift from the criterion.
    ghost = env.folded_half_w()[env_id] - origin
    frame["goal_vertices"] = ghost.detach().cpu().numpy().tolist()
    # Reported so the HTML caption can carry the same numbers as the mp4 HUD.
    frame["fold_err"] = float(env.fold_error()[env_id])
    frame["footprint"] = float(env.footprint_ratio()[env_id])
    return frame


class ClothPoseViewerWrapper(PlayPoseViewerWrapper):
    """Play viewer plus the cloth sheet and its fold target."""

    def _capture_frame(self):  # pragma: no cover - exercised in training
        return capture_cloth_viewer_frame(self.env.unwrapped, self.env_id)

    def _build_html(self, frames: list[dict[str, Any]]) -> str:
        import numpy as np

        from isaacsimenvs.tasks.cloth.utils.cloth_geometry import grid_mesh, half_indices, half_mesh

        inner = self.env.unwrapped
        c = inner.cfg.cloth
        _, sheet_tris = grid_mesh(float(c.size), int(c.resolution))
        _, ghost_tris = half_mesh(float(c.size), int(c.resolution), c.fold_axis)

        sheet = np.asarray([f["cloth_vertices"] for f in frames], dtype=float)
        ghost = np.asarray([f["goal_vertices"] for f in frames], dtype=float)

        # Split the sheet by which half each triangle belongs to, by MAJORITY vertex. An
        # all-three test leaves every triangle touching the excluded hinge row unclassified --
        # a quarter of the sheet -- which reads as a third stripe rather than two halves.
        moving = set(half_indices(int(c.resolution), c.fold_axis, positive=True))
        tri = np.asarray(sheet_tris, dtype=int).reshape(-1, 3)
        in_moving = np.isin(tri, list(moving)).sum(axis=1) >= 2

        return build_pose_viewer_html(
            frames=frames,
            object_urdf_text=self._object_urdf_text,
            table_urdf_text=self._table_urdf_text,
            hole_urdf_text=self._hole_urdf_text,
            object_urdf_path=self._object_urdf_path,
            table_urdf_path=self._table_urdf_path,
            hole_urdf_path=self._hole_urdf_path,
            github_raw_base=self.github_raw_base,
            url_check=self.url_check,
            deformables={
                "cloth_moving": {
                    "indices": tri[in_moving].reshape(-1).tolist(),
                    "vertices": sheet,
                    "color": MOVING_COLOR,
                },
                "cloth_stationary": {
                    "indices": tri[~in_moving].reshape(-1).tolist(),
                    "vertices": sheet,
                    "color": STATIONARY_COLOR,
                },
                "fold_goal": {
                    "indices": ghost_tris,
                    "vertices": ghost,
                    "color": GOAL_COLOR,
                    "opacity": GOAL_OPACITY,
                },
            },
        )
