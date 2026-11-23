"""Thin DirectRLEnv wrapper for Play.

The env owns Isaac Lab hook wiring and state buffers. Task math lives in the
utility modules called from each hook.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from .play_env_cfg import PlayEnvCfg
from isaacsimenvs.robots import FLEXIV_WUJI_RIGHT
from .utils.action_utils import apply_action_pipeline, apply_wrench_dr, hand_spring_effort
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import (
    build_observations,
    build_student_observations,
    compute_intermediate_values,
    compute_obs_dim,
)
from .utils.reset_utils import allocate_state_buffers, reset_env_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import apply_physx_material_properties, setup_scene
from .utils.termination_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["PlayEnv", "PlayEnvCfg"]


class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(
        self, cfg: PlayEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        self.robot_spec = FLEXIV_WUJI_RIGHT
        # Override action/obs/state space from the robot and the configured field lists before
        # DirectRLEnv / rl_games observes the configclass.
        n = self.robot_spec.num_joints
        cfg.action_space = n
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list, n)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list, n)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        self._setup_direct_writes()

    def _setup_direct_writes(self) -> None:
        """``cfg.direct_sim_writes``: this env writes the robot's commands and the object's wrench to
        PhysX itself (``_apply_action``); Isaac Lab's per-asset writes are turned off."""
        from isaaclab.actuators import ImplicitActuator

        self._direct_writes = bool(getattr(self.cfg, "direct_sim_writes", False))
        if not self._direct_writes:
            return
        if not all(isinstance(a, ImplicitActuator) for a in self.robot.actuators.values()):
            raise ValueError("direct_sim_writes needs implicit actuators only (an explicit model computes "
                             "the commands in write_data_to_sim)")
        self._robot_idx = self.robot._ALL_INDICES
        self._object_idx = self.object._ALL_INDICES
        view = self.robot.root_physx_view
        view.set_dof_velocity_targets(torch.zeros_like(self.robot.data.joint_vel_target), self._robot_idx)
        self._zero_effort = torch.zeros_like(self.robot.data.joint_effort_target)
        view.set_dof_actuation_forces(self._zero_effort, self._robot_idx)
        self.robot.write_data_to_sim = lambda: None        # the commands go out in _apply_action
        self.object.write_data_to_sim = lambda: None       # its wrench too (it has nothing else to write)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)
        # The hand's passive spring, from this physics step's q (decimation re-reads it).
        spring = hand_spring_effort(self)
        if spring is not None:
            self.robot.set_joint_effort_target(spring)
        if getattr(self, "_direct_writes", False):
            # What Articulation / RigidObject.write_data_to_sim would send, without their per-group
            # loop and wrench composer (implicit actuators pass the targets through unchanged).
            view = self.robot.root_physx_view
            view.set_dof_position_targets(self._cur_targets, self._robot_idx)
            if spring is not None:
                view.set_dof_actuation_forces(spring, self._robot_idx)
            self.object.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._object_forces.view(-1, 3), torque_data=self._object_torques.view(-1, 3),
                position_data=None, indices=self._object_idx, is_global=True)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    def get_student_obs(self) -> dict[str, torch.Tensor]:
        """Return opt-in student observations for distillation code."""
        return build_student_observations(self)
