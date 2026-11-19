"""``ClothBackend``: the simulator side of the cloth env -- the scene it builds (robot, table, goal
marker, sheet), what the task reads from it and what it writes to it.

``ClothEnv`` owns one and calls it in this order:

    __init__            before the simulator exists (the config is still editable)
    build_scene         inside DirectRLEnv.__init__ (``_setup_scene``): every asset
    finish_setup        the assets are live; before the task first reads the sheet
    start               the env is complete; last thing before the first reset

    begin_reset / write_sheet / end_reset   each reset
    pre_physics_step                        each policy step
    read_points                             whenever the task needs the sheet

The scene builder (``backends/scene.py``) and the Newton patches read the backend the way they read
an env: ``cfg``, ``device``, ``num_envs``, ``robot_spec``, ``scene``, ``sim``, and the assets
``robot``, ``table``, ``goal_viz``.
"""

from __future__ import annotations

import torch

from ..mdp.state import SheetLayout

__all__ = ["ClothBackend", "reset_yaw", "sheet_yaw"]


def sheet_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """``(n, 4) -> (n,)``: the heading (rotation about +z) of wxyz quaternions."""
    qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def reset_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """``(n, 4) -> (n,)``: the heading a sheet starts at for a reset pose's wxyz quaternion.

    **Not** :func:`sheet_yaw` of it: the quaternion's components are first reordered (x, y, z, w),
    as Isaac Lab 3.0's pose-write conversion reorders every root-pose write on Newton, and the
    heading is decoded from that. For the random draws of training this is harmless -- a uniform
    SO(3) draw relabelled is still uniform, so the heading is still uniform on [0, 2 pi) -- but a
    pure yaw arrives as (0, 0, sin, cos) and decodes to 180 deg, so the pinned evaluation start
    (identity) puts the sheet at 180 deg. Kept on both backends: every cloth result so far was
    produced under it, and fixing it would move the pinned start.
    """
    return sheet_yaw(torch.cat((quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]), dim=-1))


class ClothBackend:
    def __init__(self, cfg, robot_spec) -> None:
        self.cfg = cfg
        self.robot_spec = robot_spec

    # ------------------------------------------------------------------ construction

    def build_scene(self, scene, sim) -> None:
        """Spawn every asset; the scene builder ends with :meth:`install_sheet`."""
        self.scene, self.sim = scene, sim
        self.device, self.num_envs = sim.device, scene.num_envs
        from . import scene as scene_builder

        # Through the module: the Newton backend wraps it with the scene replication.
        scene_builder.setup_scene(self)

    def install_sheet(self) -> None:
        """Author the sheet and fill ``object_scale`` (N, 3)."""
        raise NotImplementedError

    def finish_setup(self) -> None:
        """The assets are live; runs before the task first reads the sheet."""

    def start(self, reset_table) -> None:
        """The env is complete. ``reset_table(env_ids)`` draws and places the table (the task's)."""

    # ------------------------------------------------------------------ the sheet, for the task

    def sheet_layout(self) -> SheetLayout:
        raise NotImplementedError

    def read_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(points_w, points_vel_w)``, each (N, P, 3): the sheet's grid points now."""
        raise NotImplementedError

    def observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        """Rotation matrices -> the goal quaternion the policy observes (see ``mdp/fold.py``)."""
        raise NotImplementedError

    def write_sheet(self, env_ids: torch.Tensor, pose_w: torch.Tensor) -> None:
        """Lay the sheet flat, at rest, at ``pose_w`` (n, 7, world, wxyz): at its xy (plus the
        backend's own offset), the spawn height, and the heading :func:`reset_yaw` decodes."""
        raise NotImplementedError

    # ------------------------------------------------------------------ per reset / per step

    def begin_reset(self, env_ids: torch.Tensor) -> None:
        """First thing in a reset, before any of the task's draws."""

    def end_reset(self, env_ids: torch.Tensor) -> None:
        """Last thing in a reset."""

    def pre_physics_step(self) -> None:
        """Once per policy step, after the action is processed."""

    # ------------------------------------------------------------------ logging

    def add_termination_reasons(self, reasons: dict[str, torch.Tensor]) -> None:
        """Extra per-episode ``done_<name>`` outcomes (reported, not acted on)."""

    def add_episode_final(self, final: dict, reasons: dict, episode_length_buf: torch.Tensor, tracker_stats) -> None:
        """Extra per-episode values for ``extras["episode_final"]``."""
