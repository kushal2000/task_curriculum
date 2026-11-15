"""JointReachEnv: track a commanded joint configuration (see joint_reach_env_cfg.py).

Built on PlayEnv's scene, state buffers and action pipeline; replaces its reset, observation, reward
and termination hooks. Joint quantities are in the canonical (policy) order unless named ``_lab``.
Task math lives in the utility modules called from each hook (utils/__init__.py lists them); the
public methods below are what eval and replay scripts call.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from ..play.play_env import PlayEnv
from ..play.robots import get_robot_spec
from ..play.utils.action_utils import apply_action_pipeline
from ..play.utils.reset_utils import allocate_state_buffers
from ..play.utils.scene_utils import apply_physx_material_properties
from .joint_reach_env_cfg import JointReachEnvCfg
from .utils import goal_bank, goal_utils
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import STATE_EXTRA, build_observations
from .utils.reset_utils import allocate_reach_buffers, place_robot, reset_reach_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import size_gpu_buffers
from .utils.termination_utils import compute_terminations


class JointReachEnv(PlayEnv):
    cfg: JointReachEnvCfg

    def __init__(self, cfg: JointReachEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self.robot_spec = get_robot_spec(cfg.robot)
        n = self.robot_spec.num_joints
        cfg.action_space = n
        cfg.observation_space = 4 * n
        cfg.state_space = 4 * n + STATE_EXTRA
        size_gpu_buffers(cfg)
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)  # PlayEnv's spaces are not ours
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        allocate_reach_buffers(self)
        goal_bank.load_goals(self)

    # ------------------------------------------------------------------ public API (eval / replay)

    def build_goal_bank(self, size: int, seed: int) -> torch.Tensor:
        return goal_bank.build_goal_bank(self, size, seed)

    def build_goal_chains(self, starts: torch.Tensor, length: int, seed: int) -> torch.Tensor:
        return goal_bank.build_goal_chains(self, starts, length, seed)

    def default_goal(self) -> torch.Tensor:
        return goal_bank.default_goal(self)

    def chain_starts(self, bank: torch.Tensor, default_frac: float, seed: int) -> torch.Tensor:
        return goal_bank.chain_starts(self, bank, default_frac, seed)

    def hold_steps(self, goal_from: torch.Tensor, goal_to: torch.Tensor, settle_s: torch.Tensor) -> torch.Tensor:
        return goal_utils.hold_steps(self, goal_from, goal_to, settle_s)

    def place_robot(self, env_ids: torch.Tensor, q_canon: torch.Tensor) -> None:
        place_robot(self, env_ids, q_canon)

    # ------------------------------------------------------------------ hooks

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        DirectRLEnv._reset_idx(self, env_ids)
        reset_reach_state(self, env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        apply_action_pipeline(self, self._actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        goal_utils.update_goals(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)


__all__ = ["JointReachEnv"]
