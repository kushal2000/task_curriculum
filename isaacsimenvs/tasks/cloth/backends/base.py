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
``robot``, ``table``, ``goal_viz``, ``object``.
"""

from __future__ import annotations

import torch

from ..mdp.state import SheetLayout
from . import scene as scene_builder

__all__ = ["ClothBackend"]


class ClothBackend:
    def __init__(self, cfg, robot_spec) -> None:
        self.cfg = cfg
        self.robot_spec = robot_spec

    # ------------------------------------------------------------------ construction

    def build_scene(self, scene, sim) -> None:
        """Spawn every asset; the scene builder ends with :meth:`install_sheet`."""
        self.scene, self.sim = scene, sim
        self.device, self.num_envs = sim.device, scene.num_envs
        # Through the module: the Newton backend wraps it with the scene replication.
        scene_builder.setup_scene(self)

    def install_sheet(self) -> None:
        """Author the sheet; set ``object`` and fill ``object_scale`` (N, 3)."""
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
        """Lay the sheet flat at ``pose_w`` (n, 7): its xy and its yaw; at rest."""
        self.object.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids)

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
