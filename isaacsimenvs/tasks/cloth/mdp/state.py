"""What the parts of the cloth env hand each other: the simulator state, the sheet's layout, and
the fold as derived from one state.

    backend --SimState--> task.derive --FoldView--> task (reward, dones), observations
    backend --SheetLayout--> task (built once)

World frame throughout; joints in Isaac Lab's ("Lab") order; quaternions (w, x, y, z). Torch only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["FoldView", "SheetLayout", "SimState"]


@dataclass
class SimState:
    """One read of the simulator. Read again after anything writes to it (a reset)."""

    joint_pos: torch.Tensor  # (N, J) Lab order
    joint_vel: torch.Tensor  # (N, J)
    body_state_w: torch.Tensor  # (N, B, 13) robot bodies: position, wxyz, linear and angular velocity
    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    points_w: torch.Tensor  # (N, P, 3) the sheet's grid points (VBD particles, or the plates' grid)
    points_vel_w: torch.Tensor  # (N, P, 3)


@dataclass
class SheetLayout:
    """The sheet's grid as the task scores it, fixed for the run (the backend builds it).

    ``rest_local`` is the nominal ``size`` sheet; an env whose sheet was built at another size has
    its in-plane rest geometry scaled by ``xy_scale``. ``None`` means every env is nominal -- the
    task then keeps one shared rest geometry (and its own arithmetic for it).
    """

    rest_local: list  # (P, 3) nominal grid points in the sheet frame, z = 0
    keypoint_idx: list[int]  # the moving half's four corners, the scored keypoints
    resolution: int
    fold_axis: str
    size: float  # nominal side length [m]
    fold_lift: float  # height of the folded half above the stationary one [m]
    object_scale: torch.Tensor  # (N, 3) the observed sheet size, normalised (``object_scales``)
    xy_scale: torch.Tensor | None = None  # (N,) per-env in-plane size factor
    env_size: torch.Tensor | None = None  # (N,) per-env side length [m]


@dataclass
class FoldView:
    """The fold, derived from one ``SimState``: what reward, dones and observations all read."""

    points_w: torch.Tensor  # (N, P, 3)
    points_vel_w: torch.Tensor  # (N, P, 3)
    frame_r: torch.Tensor  # (N, 3, 3) the stationary half, rest -> current
    frame_c: torch.Tensor  # (N, 3) its centroid
    keypoints_w: torch.Tensor  # (N, K, 3) the moving half's corners
    targets_w: torch.Tensor  # (N, K, 3) where they belong when folded
    fold_error: torch.Tensor  # (N,) max corner-to-target distance
    footprint: torch.Tensor  # (N,) extent along the fold axis / sheet size
    object_pos_w: torch.Tensor  # (N, 3) the moving half as a rigid object: corner centroid
    object_quat_w: torch.Tensor  # (N, 4) its rotation from rest, Kabsch-fitted to the corners
    object_lin_vel_w: torch.Tensor  # (N, 3) the corners' mean velocity
    object_ang_vel_w: torch.Tensor  # (N, 3) zero

