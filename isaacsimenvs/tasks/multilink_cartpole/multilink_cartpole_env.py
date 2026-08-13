"""Thin DirectRLEnv wrapper for the multi-link cartpole.

The env owns hook wiring and nothing else; task maths lives in the `utils/` modules,
following `tasks/play/play_env.py`.

Curriculum wiring, for orientation:
    __init__     allocate_curriculum_buffers → seeds every env's difficulty
    _reset_idx   record_episode_success → sample_difficulty → apply_difficulty
    _get_dones   update_curriculum (advances the sampling range)
    _get_rewards reward_scale (inside compute_rewards) → log_curriculum
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from isaacsimenvs.curriculum import allocate_curriculum_buffers, update_curriculum

from .multilink_cartpole_env_cfg import MultiLinkCartpoleEnvCfg
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import (
    build_observations,
    compute_intermediate_values,
    compute_obs_dim,
    compute_state_dim,
)
from .utils.reset_utils import allocate_state_buffers, apply_difficulty, reset_env_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import setup_scene
from .utils.termination_utils import compute_terminations

__all__ = ["MultiLinkCartpoleEnv", "MultiLinkCartpoleEnvCfg"]


class MultiLinkCartpoleEnv(DirectRLEnv):
    cfg: MultiLinkCartpoleEnvCfg

    def __init__(
        self, cfg: MultiLinkCartpoleEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Widths are derived from n_max before DirectRLEnv (and then rl_games) reads the
        # configclass, so `n_max` is the only place observation size is declared.
        cfg.observation_space = compute_obs_dim(cfg)
        cfg.state_space = compute_state_dim(cfg)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene

        allocate_state_buffers(self)
        allocate_curriculum_buffers(self)
        # Realise the seeded difficulties immediately so the articulation is in a valid
        # locked state even before the first reset.
        apply_difficulty(self, torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # `super()._reset_idx` zeroes `episode_length_buf`, and scoring the finished
        # episode needs its length, so snapshot it first.
        episode_lengths = self.episode_length_buf[env_ids].clone()
        super()._reset_idx(env_ids)
        reset_env_state(self, env_ids, episode_lengths)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        # Called `decimation` times per policy step; idempotent.
        self.cartpole.set_joint_effort_target(
            self.cfg.action.action_scale * self._actions, joint_ids=self._cart_dof_idx
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)
