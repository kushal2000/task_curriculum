"""Point-level observation DR: lags, noise and biases on the tracked object points.

The observed object is a set of K tracked points -- Play's 4 keypoints, the cloth's 4 moving-half
corners -- plus the goal pose. On hardware these come from one camera stream (CoTracker for the
cloth), so every point and the goal share one frame and one latency, while each point carries its
own tracking error. This module models that:

* **one lag per env per step**, applied to the points and the goal together;
* **per-point** white noise (every step) and bias (drawn per episode), plus a **common** bias
  shared by the points and the goal (a calibration error, which cancels in point-minus-goal);
* the points are then rigidly fitted (Kabsch) and re-expanded, exactly as the clean observation
  builds its keypoints from a pose -- so the perception pipeline is unchanged, only its inputs are
  perturbed.

Two lag models (``domain_randomization.point_dr_mode``):

``simple``
    a fresh lag every step, uniform on ``point_delay_range`` (inclusive). The stream can step back
    in time.
``realistic``
    the camera delivers a frame every ``k`` policy steps (``k`` per episode from
    ``point_hold_choices``, random phase); each frame's latency is a per-episode base (uniform on
    ``point_base_delay_range``) plus a per-frame jitter (uniform on ``+-point_delay_jitter``); the
    policy always reads the newest frame that has arrived, so the frame it reads never gets older.
    Capped at ``point_delay_max``.

Torch only (no Isaac Lab), so it is unit-tested on CPU (tests/test_point_dr.py).
"""

from __future__ import annotations

import torch

MODES = ("off", "simple", "realistic")


# ----------------------------------------------------------------------------
# Rigid fit
# ----------------------------------------------------------------------------


def _mat_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """``(N, 3, 3)`` rotation matrices to ``(N, 4)`` wxyz quaternions with w >= 0."""
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    # Shepperd: the four candidate magnitudes; pick the largest for each row (numerically safe).
    cand = torch.stack(
        (1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22),
        dim=-1,
    ).clamp(min=0.0)
    i = cand.argmax(dim=-1)
    s = torch.sqrt(cand.gather(1, i.unsqueeze(1)).squeeze(1)) * 2.0  # 4 * |component i|
    q = torch.empty(r.shape[0], 4, device=r.device, dtype=r.dtype)
    # i == 0: w largest
    w0 = torch.stack((0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                      (r[:, 1, 0] - r[:, 0, 1]) / s), dim=-1)
    x0 = torch.stack(((r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                      (r[:, 0, 2] + r[:, 2, 0]) / s), dim=-1)
    y0 = torch.stack(((r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                      (r[:, 1, 2] + r[:, 2, 1]) / s), dim=-1)
    z0 = torch.stack(((r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                      (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s), dim=-1)
    q = torch.where((i == 0).unsqueeze(-1), w0, q)
    q = torch.where((i == 1).unsqueeze(-1), x0, q)
    q = torch.where((i == 2).unsqueeze(-1), y0, q)
    q = torch.where((i == 3).unsqueeze(-1), z0, q)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[:, :1] < 0.0, -q, q)


def kabsch(rest: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares rigid fit of ``rest`` onto ``pts``.

    ``rest``: ``(K, 3)`` or ``(N, K, 3)`` object-frame points; ``pts``: ``(N, K, 3)``.
    Returns ``(centroid (N, 3), R (N, 3, 3))`` with ``pts ~= centroid + R (rest - mean(rest))``.
    Reflections are excluded; a degenerate set falls back to identity rather than NaN.
    """
    if rest.dim() == 2:
        rest = rest.unsqueeze(0).expand(pts.shape[0], -1, -1)
    c = pts.mean(dim=1)
    p = rest - rest.mean(dim=1, keepdim=True)
    q = pts - c.unsqueeze(1)
    h = torch.einsum("nki,nkj->nij", p, q)
    h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    d = torch.ones(pts.shape[0], 3, device=pts.device, dtype=pts.dtype)
    d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
    r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
    bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
    if bool(bad.any()):
        r = r.clone()
        r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
    return c, r


def refit_points(rest: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigidly fit ``rest`` to ``pts`` and re-expand: ``(points (N, K, 3), quat wxyz (N, 4))``."""
    if rest.dim() == 2:
        rest = rest.unsqueeze(0).expand(pts.shape[0], -1, -1)
    c, r = kabsch(rest, pts)
    centred = rest - rest.mean(dim=1, keepdim=True)
    return c.unsqueeze(1) + torch.einsum("nij,nkj->nki", r, centred), _mat_to_wxyz(r)


# ----------------------------------------------------------------------------
# Delay queue
# ----------------------------------------------------------------------------


def push_and_read(
    queue: torch.Tensor, values: torch.Tensor, lags: torch.Tensor, flush: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push ``values`` (N, D) into ``queue`` (N, L, D) (index i = i steps ago) and read ``lags``.

    ``flush`` (N,) bool: those envs' queues are filled with ``values`` first, so an episode's first
    reads never see the previous episode.
    """
    if flush is not None and bool(flush.any()):
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)
    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    lags = lags.clamp(0, queue.shape[1] - 1)
    return queue, queue[torch.arange(queue.shape[0], device=queue.device), lags]


# ----------------------------------------------------------------------------
# Lag models
# ----------------------------------------------------------------------------


def simple_lags(n: int, lag_range: tuple[int, int], device) -> torch.Tensor:
    """A fresh lag per env, uniform on the inclusive ``lag_range``."""
    lo, hi = int(lag_range[0]), int(lag_range[1])
    return torch.randint(lo, hi + 1, (n,), device=device)


class RealisticLags:
    """Per-env camera-stream state for ``point_dr_mode: realistic``.

    Time ``t`` counts policy steps since the episode started (0 at the first observation). The
    camera captures frame ``m`` at ``c_m = phase + m k``; it reaches the policy at
    ``a_m = max(c_m + base + j_m, a_{m-1})`` -- a per-episode base latency plus a per-FRAME jitter
    ``j_m``, delivered in order (a pipeline does not reorder frames). At each step the policy reads
    the newest frame that has arrived, so it holds a frame until the next one lands and never steps
    back. ``lag = t - c`` of the frame read, capped at ``max_lag`` by reading a newer frame (only
    when the cap is tighter than ``base + jitter + k - 1``).

    The episode starts in steady state: the first read frame is the newest capture at or before
    ``t - base``. Frames before t = 0 read the flushed queue, i.e. the first observation.
    """

    def __init__(self, num_envs: int, device) -> None:
        z = lambda: torch.zeros(num_envs, dtype=torch.long, device=device)  # noqa: E731
        self.base, self.k, self.phase = z(), z() + 1, z()
        self.t, self.f = z(), z()
        self.next_c, self.next_a = z(), z()

    def draw_episode(
        self, env_ids: torch.Tensor, base_range: tuple[int, int], k_choices: tuple[int, ...]
    ) -> None:
        n = env_ids.numel()
        dev = self.base.device
        self.base[env_ids] = torch.randint(int(base_range[0]), int(base_range[1]) + 1, (n,), device=dev)
        ks = torch.as_tensor(tuple(int(v) for v in k_choices), dtype=torch.long, device=dev)
        k = ks[torch.randint(0, ks.numel(), (n,), device=dev)]
        self.k[env_ids] = k
        self.phase[env_ids] = (torch.rand(n, device=dev) * k).long().clamp(max=k - 1)

    def _jitter(self, jitter: int) -> torch.Tensor:
        return torch.randint(-int(jitter), int(jitter) + 1, self.t.shape, device=self.t.device)

    def step(self, start: torch.Tensor, jitter: int, max_lag: int) -> torch.Tensor:
        """Advance one policy step and return each env's lag. ``start``: envs at episode step 0."""
        self.t = torch.where(start, torch.zeros_like(self.t), self.t + 1)
        if bool(start.any()):
            # Steady state at t = 0: newest capture at or before -base is the frame being read.
            c0 = torch.div(-self.base - self.phase, self.k, rounding_mode="floor") * self.k + self.phase
            self.f = torch.where(start, c0, self.f)
            nc = c0 + self.k
            na = (nc + self.base + self._jitter(jitter)).clamp(min=nc)
            self.next_c = torch.where(start, nc, self.next_c)
            self.next_a = torch.where(start, na, self.next_a)
        # Deliver every frame that has arrived by now, in order (bounded: one per capture period).
        for _ in range(int(max_lag) + 2):
            arrived = self.next_a <= self.t
            if not bool(arrived.any()):
                break
            self.f = torch.where(arrived, self.next_c, self.f)
            nc = self.next_c + self.k
            na = torch.maximum((nc + self.base + self._jitter(jitter)).clamp(min=nc), self.next_a)
            self.next_c = torch.where(arrived, nc, self.next_c)
            self.next_a = torch.where(arrived, na, self.next_a)
        self.f = torch.maximum(self.f, self.t - int(max_lag))
        return self.t - self.f


__all__ = [
    "MODES",
    "RealisticLags",
    "kabsch",
    "push_and_read",
    "refit_points",
    "simple_lags",
]
