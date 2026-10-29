"""Joint-reach terminations: timeouts only. Also refreshes the goal error the reward and logs read."""

from __future__ import annotations

import torch

__all__ = ["compute_terminations"]


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    q = env.robot.data.joint_pos[:, env._perm_lab_to_canon]
    env._err = env._goal - q
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    truncated = env.episode_length_buf >= env.max_episode_length - 1
    env._termination_reasons = {"timeout": truncated}
    return terminated, truncated
