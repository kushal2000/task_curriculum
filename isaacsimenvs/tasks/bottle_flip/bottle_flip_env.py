"""Bottle flipping on the Kuka + Sharpa hand — a `PlayEnv` subclass.

Everything about the robot is inherited: `_pre_physics_step`, `_apply_action`,
`_get_observations` and play's whole buffer/reset stack run unchanged. Only the object,
the goal semantics, the reward and the termination rules are replaced.

`_get_observations` in particular is deliberately *not* overridden. Play's observation is
already the right one — joint state, palm pose, fingertip positions, object rotation, and
keypoints relative to the goal, where the goal here is "upright, standing on the table".
Leaving it untouched keeps the observation layout byte-identical to play2perfect's, which
is what allows:

    train.py --task Isaacsimenvs-BottleFlip-Direct-v0 \\
             --checkpoint <play2perfect.pth> --checkpoint_load_mode weights
"""

from __future__ import annotations

import torch

from isaacsimenvs.curriculum import allocate_curriculum_buffers, update_curriculum
from isaacsimenvs.tasks.play.play_env import PlayEnv
from isaacsimenvs.tasks.play.utils.obs_utils import compute_intermediate_values

from .bottle_flip_env_cfg import BottleFlipEnvCfg
from .utils.flip_state import compute_flip_state
from .utils.logging_utils import log_step_metrics
from .utils.reset_utils import allocate_flip_buffers, apply_difficulty, reset_flip_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import setup_scene
from .utils.termination_utils import compute_terminations

__all__ = ["BottleFlipEnv", "BottleFlipEnvCfg"]


class BottleFlipEnv(PlayEnv):
    cfg: BottleFlipEnvCfg

    def __init__(
        self, cfg: BottleFlipEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)  # play allocates its own buffers

        allocate_flip_buffers(self)
        allocate_curriculum_buffers(self)
        apply_difficulty(self, torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)  # play: robot pose, table, bottle start pose, queues

        # Guard the very first reset, which `DirectRLEnv` triggers before `__init__` has
        # finished allocating the flip buffers.
        if hasattr(self, "_bf_released"):
            reset_flip_state(self, env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_curriculum(self)
        # Play's intermediates give fingertip distances, keypoints and the lifted flag,
        # all of which the flip state machine and reward consume.
        compute_intermediate_values(self)
        compute_flip_state(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _curriculum_success_threshold(self) -> float | None:
        """Play discovers this hook with `hasattr`; returning None keeps the configured
        threshold. Present as the documented extension point for a task-specific rule."""
        return None
