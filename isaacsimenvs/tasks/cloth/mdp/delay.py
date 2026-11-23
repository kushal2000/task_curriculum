"""``DelayLine``: a per-env delay of a signal, for every DR delay (proprioception, keypoints, point
cloud, action). Torch only."""

from __future__ import annotations

import torch

__all__ = ["DelayLine"]


class DelayLine:
    """The last ``max_lag + 1`` values of a ``(N, *shape)`` signal per env (a ring buffer), read back at
    a per-env lag. ``max_lag = 0`` passes the signal through."""

    def __init__(self, num_envs: int, max_lag: int, shape, device, dtype=torch.float32) -> None:
        shape = (int(shape),) if isinstance(shape, int) else tuple(shape)
        self.size = int(max_lag) + 1
        self.buffer = torch.zeros(self.size, num_envs, *shape, device=device, dtype=dtype)
        self.ptr = 0
        self.env = torch.arange(num_envs, device=device)

    def clear(self, env_ids: torch.Tensor) -> None:
        self.buffer[:, env_ids] = 0.0

    def push_and_read(self, values: torch.Tensor, flush: torch.Tensor | None, lag: torch.Tensor) -> torch.Tensor:
        """Push ``values`` and read each env's entry ``lag`` steps back. ``flush`` envs (the first step
        of an episode) have their whole history set to ``values`` first, so an episode never reads
        the previous one."""
        values = values.to(self.buffer.dtype)
        if self.size == 1:
            return values
        self.ptr = (self.ptr + 1) % self.size
        self.buffer[self.ptr] = values
        if flush is not None:              # masked, not ``if flush.any()``: no host sync per step
            mask = flush.view(1, -1, *([1] * (values.dim() - 1)))
            self.buffer = torch.where(mask, values.unsqueeze(0), self.buffer)
        return self.buffer[(self.ptr - lag.clamp(0, self.size - 1)) % self.size, self.env]
