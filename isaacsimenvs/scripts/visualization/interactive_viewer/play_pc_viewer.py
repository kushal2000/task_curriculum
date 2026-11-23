"""Interactive HTML capture for the point-cloud Play task (``PlayPCEnv``), logged to wandb.

The Play pose viewer (robot, tool, green goal tool, table) plus the two clouds the actor reads,
as point channels:

* the live cloud -- what the cameras see after occlusion, noise, drops, delay and the crop box
  (``PointCloud.last_xyz``), in grey; subsampled to ``LIVE_SHOWN`` points to keep the page small;
* the goal cloud -- the tool's surface at the goal pose, in green.

Points are env-frame positions, as the poses. A slot with no valid point repeats a valid one (or
sits under the floor when the cloud is empty), so the channel keeps a fixed size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaacsimenvs.scripts.visualization.interactive_viewer.pose_viewer import (
    PlayPoseViewerWrapper,
    build_pose_viewer_html,
    capture_pose_viewer_frame,
)

__all__ = ["PlayPCViewerWrapper"]

LIVE_SHOWN = 512
GOAL_SHOWN = 256
LIVE_COLOR = (0.35, 0.35, 0.38)
GOAL_COLOR = (0.15, 0.80, 0.35)


def _fixed_size(pts: np.ndarray, ok: np.ndarray, k: int) -> np.ndarray:
    valid = pts[ok]
    if len(valid) == 0:
        return np.tile(np.array([[0.0, 0.0, -1.0]]), (k, 1))
    idx = np.linspace(0, len(valid) - 1, num=min(k, len(valid))).astype(int)
    out = valid[idx]
    if len(out) < k:
        out = np.concatenate([out, np.repeat(out[:1], k - len(out), axis=0)])
    return out


class PlayPCViewerWrapper(PlayPoseViewerWrapper):
    """Play viewer plus the actor's live and goal clouds."""

    def _capture_frame(self) -> dict[str, Any]:  # pragma: no cover - exercised in training
        inner, e = self.env.unwrapped, self.env_id
        frame = capture_pose_viewer_frame(inner, e)
        n_live = int(inner.layout["n_live"])
        xyz = inner.cloud.last_xyz[e, :n_live].detach().float().cpu().numpy()
        ok = inner.cloud.last_mask[e, :n_live].detach().cpu().numpy().astype(bool)
        frame["live_cloud"] = _fixed_size(xyz, ok, LIVE_SHOWN)
        goal = inner._last_goal_cloud[e].detach().float().cpu().numpy()
        frame["goal_cloud"] = _fixed_size(goal, np.ones(len(goal), dtype=bool), GOAL_SHOWN)
        return frame

    def _build_html(self, frames: list[dict[str, Any]]) -> str:
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
            robot_urdf_relative_path=self.env.unwrapped.cfg.assets.robot_urdf,
            deformables={
                "live_cloud": {"indices": [], "vertices": np.stack([f["live_cloud"] for f in frames]),
                               "color": LIVE_COLOR, "point_size": 0.006},
                "goal_cloud": {"indices": [], "vertices": np.stack([f["goal_cloud"] for f in frames]),
                               "color": GOAL_COLOR, "point_size": 0.005, "opacity": 0.8},
            },
        )
