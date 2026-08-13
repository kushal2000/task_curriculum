"""Task-agnostic curriculum machinery shared by every env in this repo.

The contract is one buffer and one convention: each env carries
``env._difficulty`` of shape ``(num_envs, difficulty_dim)`` in ``[0, 1]``, where 0 is
the easiest instance of the task and 1 the hardest. This module owns *where that vector
is sampled from* and *when the range moves*; each task owns *what the numbers mean*.

Note that :mod:`.cfg` imports ``isaaclab.utils.configclass``, which only resolves after
``AppLauncher`` has booted Kit. :mod:`.core` and :mod:`.schedulers` are Kit-free and can
be imported directly by tests.
"""

from .cfg import CurriculumCfg
from .core import (
    allocate_curriculum_buffers,
    curriculum_progress,
    log_curriculum,
    record_episode_success,
    reward_scale,
    sample_difficulty,
    update_curriculum,
)

__all__ = [
    "CurriculumCfg",
    "allocate_curriculum_buffers",
    "curriculum_progress",
    "log_curriculum",
    "record_episode_success",
    "reward_scale",
    "sample_difficulty",
    "update_curriculum",
]
