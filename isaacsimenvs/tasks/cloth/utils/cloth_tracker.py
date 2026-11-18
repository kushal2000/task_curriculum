"""The real cloth perception pipeline, replayed in sim as observation DR (``cloth.tracker``).

On the robot a stereo pair runs CoTracker on a 7x7 grid inset from the sheet's edges. A point is
*fresh* when both cameras see it (and it triangulates). Each half of the sheet is fitted rigidly
(Kabsch) to its fresh points; the object observation is the moving half's corners from that fit,
the goal the stationary half's fit carried through the fold (reflect across the crease). When either
half has fewer than ``n_min`` fresh points, or they are nearly collinear, the whole estimate --
object and goal -- holds its last value.

This module reproduces that from the sim's state instead of drawing freezes from a statistical
model, so freezes, their dependence on the hand, the goal's own error and the fit error of a few
clustered points all come out of the geometry:

* DR-0  the pipeline above (inset grid, fresh-only half fits, whole-sheet freeze);
* DR-1  per-episode camera positions; a point is covered when the ray from a camera passes within
        ``radius + margin`` of a hand / wrist / forearm capsule;
* DR-1b re-acquisition failure: an uncovered point returns only after a delay, sometimes a long one;
* DR-2  self-occlusion: the ray crosses the other half's surface (a folded flap over the
        stationary half);
* DR-3  per-point white noise per camera frame, and an offset redrawn at each re-acquisition;
* DR-4  a per-episode rigid calibration error on every observed point;
* DR-5  a frame every k steps, read L steps later;
* DR-6  the pipeline's assumed sheet size off by a per-episode factor.

Torch only (no Isaac Lab, no Kit), so it is unit-tested on CPU (tests/test_cloth_tracker.py). The
env supplies particles, rest geometry and capsules (``ClothTaskMixin._tracker_observe``).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "ClothTracker",
    "cell_triangles",
    "inset_grid_weights",
    "inset_halves",
    "minor_spread",
    "ray_hits_triangles",
    "segment_distance_sq",
    "small_rotation",
    "weighted_kabsch",
]


# ----------------------------------------------------------------------------
# Grid geometry
# ----------------------------------------------------------------------------


def inset_grid_weights(resolution: int, inset: float) -> torch.Tensor:
    """``(R*R, R*R)`` bilinear weights: tracked point ``q`` = ``sum_p W[q, p] * particle[p]``.

    The tracked grid has the particle grid's layout (row-major, x index first, as ``grid_mesh``),
    shrunk about the centre by ``1 - 2 * inset``.
    """
    r = int(resolution)
    f = 1.0 - 2.0 * float(inset)
    mid = (r - 1) / 2.0
    w1 = torch.zeros(r, r, dtype=torch.float64)  # 1-D weights: tracked index -> particle index
    for i in range(r):
        u = mid + f * (i - mid)
        lo = min(int(math.floor(u)), r - 2)
        a = u - lo
        w1[i, lo] += 1.0 - a
        w1[i, lo + 1] += a
    return torch.kron(w1, w1).float()  # index i * r + j on both sides


def inset_halves(resolution: int) -> tuple[list[int], list[int]]:
    """Tracked-point indices of the moving (+x) and stationary (-x) halves; the centre column
    (on the crease) belongs to neither, as ``half_indices`` does for the particles."""
    r = int(resolution)
    mid = (r - 1) / 2.0
    mv = [i * r + j for i in range(r) for j in range(r) if i > mid]
    st = [i * r + j for i in range(r) for j in range(r) if i < mid]
    return mv, st


def cell_triangles(resolution: int, i_lo: int, i_hi: int) -> torch.Tensor:
    """``(T, 3)`` particle indices of the triangles in grid cells with x index in ``[i_lo, i_hi)``."""
    r = int(resolution)
    tris = []
    for i in range(i_lo, i_hi):
        for j in range(r - 1):
            a = i * r + j
            b, c = a + 1, a + r
            tris += [(a, c, b), (b, c, c + 1)]
    return torch.tensor(tris, dtype=torch.long)


# ----------------------------------------------------------------------------
# Geometric primitives (broadcasting)
# ----------------------------------------------------------------------------


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def segment_distance_sq(p1, q1, p2, q2) -> torch.Tensor:
    """Squared distance between segments ``p1-q1`` and ``p2-q2`` (Ericson, RTCD 5.1.9).

    Inputs broadcast over leading dims, last dim 3. Both segments must have non-zero length.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = _dot(d1, d1).clamp(min=1e-12), _dot(d2, d2).clamp(min=1e-12)
    b, c, f = _dot(d1, d2), _dot(d1, r), _dot(d2, r)
    denom = a * e - b * b
    s = torch.where(denom > 1e-12, ((b * f - c * e) / denom.clamp(min=1e-12)).clamp(0.0, 1.0), torch.zeros_like(a))
    t = (b * s + f) / e
    s = torch.where(t < 0.0, (-c / a).clamp(0.0, 1.0), torch.where(t > 1.0, ((b - c) / a).clamp(0.0, 1.0), s))
    t = t.clamp(0.0, 1.0)
    diff = (p1 + d1 * s.unsqueeze(-1)) - (p2 + d2 * t.unsqueeze(-1))
    return _dot(diff, diff)


def ray_hits_triangles(origin, end, v0, v1, v2, gap: float) -> torch.Tensor:
    """Whether segment ``origin -> end`` crosses triangle ``(v0, v1, v2)`` at least ``gap`` [m]
    before ``end`` (Moller-Trumbore). Broadcasts; returns a bool tensor."""
    d = end - origin
    e1, e2 = v1 - v0, v2 - v0
    h = torch.cross(d.expand(torch.broadcast_shapes(d.shape, e2.shape)), e2.expand(torch.broadcast_shapes(d.shape, e2.shape)), dim=-1)
    a = _dot(e1, h)
    ok = a.abs() > 1e-12
    inv = 1.0 / torch.where(ok, a, torch.ones_like(a))
    s = origin - v0
    u = inv * _dot(s, h)
    q = torch.cross(s.expand(torch.broadcast_shapes(s.shape, e1.shape)), e1.expand(torch.broadcast_shapes(s.shape, e1.shape)), dim=-1)
    v = inv * _dot(d, q)
    t = inv * _dot(e2, q)
    t_max = 1.0 - gap / torch.linalg.norm(d, dim=-1).clamp(min=1e-9)
    return ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0) & (t < t_max)


def weighted_kabsch(rest: torch.Tensor, cur: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``cur ~= R rest + t`` over the points with weight ``w`` (``(n, M)``, 0/1). ``(R, t)``.

    A rank-deficient set falls back to identity rather than NaN (the caller only uses fits that
    passed ``n_min`` / ``min_spread``, so this guards the rejected ones).
    """
    ws = w.sum(dim=1, keepdim=True).clamp(min=1e-9)
    cr = (rest * w.unsqueeze(-1)).sum(dim=1) / ws
    cc = (cur * w.unsqueeze(-1)).sum(dim=1) / ws
    p = (rest - cr.unsqueeze(1)) * w.unsqueeze(-1)
    q = cur - cc.unsqueeze(1)
    h = torch.einsum("nki,nkj->nij", p, q)
    h = h + 1e-9 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    d = torch.ones(h.shape[0], 3, device=h.device, dtype=h.dtype)
    d[:, 2] = torch.sign(torch.linalg.det(v @ u.transpose(-2, -1)))
    d[:, 2] = torch.where(d[:, 2] == 0.0, torch.ones_like(d[:, 2]), d[:, 2])
    r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
    bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
    if bool(bad.any()):
        r = torch.where(bad.view(-1, 1, 1), torch.eye(3, device=r.device, dtype=r.dtype).expand_as(r), r)
    t = cc - torch.einsum("nij,nj->ni", r, cr)
    return r, t


def minor_spread(xy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Extent of the weighted points ``xy`` (``(n, M, 2)``) along their minor principal axis."""
    ws = w.sum(dim=1, keepdim=True).clamp(min=1e-9)
    mu = (xy * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / ws.unsqueeze(-1)
    c = xy - mu
    cov = torch.einsum("nki,nkj->nij", c * w.unsqueeze(-1), c) / ws.unsqueeze(-1)
    cov = cov + 1e-12 * torch.eye(2, device=xy.device, dtype=xy.dtype)
    _, vecs = torch.linalg.eigh(cov)
    minor = vecs[:, :, 0]  # ascending eigenvalues
    proj = _dot(c, minor.unsqueeze(1))
    on = w > 0
    hi = torch.where(on, proj, torch.full_like(proj, -1e9)).amax(dim=1)
    lo = torch.where(on, proj, torch.full_like(proj, 1e9)).amin(dim=1)
    return (hi - lo).clamp(min=0.0)


def small_rotation(angles: torch.Tensor) -> torch.Tensor:
    """``(n, 3, 3)`` = Rz Ry Rx for per-axis angles ``(n, 3)`` [rad]."""
    cx, cy, cz = torch.cos(angles).unbind(-1)
    sx, sy, sz = torch.sin(angles).unbind(-1)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    rx = torch.stack((one, zero, zero, zero, cx, -sx, zero, sx, cx), -1).view(-1, 3, 3)
    ry = torch.stack((cy, zero, sy, zero, one, zero, -sy, zero, cy), -1).view(-1, 3, 3)
    rz = torch.stack((cz, -sz, zero, sz, cz, zero, zero, zero, one), -1).view(-1, 3, 3)
    return rz @ ry @ rx


def _uniform(lo_hi, shape, device) -> torch.Tensor:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return lo + (hi - lo) * torch.rand(*shape, device=device)


def _choice(choices, n: int, device) -> torch.Tensor:
    c = torch.as_tensor([int(v) for v in choices], dtype=torch.long, device=device)
    return c[torch.randint(0, c.numel(), (n,), device=device)]


# ----------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------


#: Output row layout: object (moving-half) R and t, stationary-half R and t.
_OUT = 24


class ClothTracker:
    """Per-env pipeline state. Call :meth:`step` once per policy step, every env.

    ``step`` returns, per env, the estimate the policy reads now: the moving half's fit ``(R_mv,
    t_mv)`` and the stationary half's ``(R_st, t_st)``, with ``x_obs = R x_rest_assumed + t``, plus
    the per-env size factor the observed keypoint offsets must be scaled by.
    """

    def __init__(self, cfg, num_envs: int, resolution: int, cams_env: torch.Tensor, device) -> None:
        self.cfg = cfg
        self.n = int(num_envs)
        self.device = device
        r = int(resolution)
        self.res = r
        self.W = inset_grid_weights(r, cfg.inset).to(device)
        mv, st = inset_halves(r)
        self.mv = torch.tensor(mv, device=device)
        self.st = torch.tensor(st, device=device)
        self.half_pts = torch.cat((self.mv, self.st))  # tracked points that can be fresh
        self.m_mv = len(mv)
        mid = (r - 1) // 2
        # Self-occlusion: moving points against the stationary side's cells and vice versa, both
        # including the cells next to the crease column.
        self.tri_st = cell_triangles(r, 0, mid).to(device)
        self.tri_mv = cell_triangles(r, mid, r - 1).to(device)
        self.cams_nominal = torch.as_tensor(cams_env, dtype=torch.float32, device=device)  # (2, 3)

        n, m, dev = self.n, self.half_pts.numel(), device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        # Per-episode draws.
        self.k = z(n, dt=torch.long) + 1
        self.phase = z(n, dt=torch.long)
        self.lat = z(n, dt=torch.long)
        self.cams = self.cams_nominal.unsqueeze(0).repeat(n, 1, 1)
        self.margin = z(n)
        self.p_lost = z(n)
        self.sigma = z(n)
        self.bias_b = z(n)
        self.cal_r = torch.eye(3, device=dev).repeat(n, 1, 1)
        self.cal_t = z(n, 3)
        self.cal_c = z(n, 3)
        self.scale = torch.ones(n, device=dev)
        # Per-point state (over half_pts).
        self.covered = z(n, m, dt=torch.bool)
        self.back_at = z(n, m, dt=torch.long)
        self.fresh = torch.ones(n, m, dtype=torch.bool, device=dev)
        self.bias = z(n, m, 3)
        # Held estimate and its delay line.
        self.held = z(n, _OUT)
        self.queue_len = max(int(v) for v in cfg.latency_choices) + 1
        self.queue = z(n, self.queue_len, _OUT)
        # Per-episode counters (for logging).
        self.frames = z(n)
        self.frozen_frames = z(n)
        self.fresh_mv_sum = z(n)
        self.fresh_st_sum = z(n)
        self.cov_hand_sum = z(n)
        self.cov_cloth_sum = z(n)
        self.uncov_share_sum = z(n)

    # ------------------------------------------------------------------ episodes

    def _draw_episode(self, ids: torch.Tensor, particles: torch.Tensor) -> None:
        c, dev, n = self.cfg, self.device, int(ids.numel())
        k = _choice(c.frame_period_choices, n, dev)
        self.k[ids] = k
        self.phase[ids] = (torch.rand(n, device=dev) * k).long().clamp(max=k - 1)
        self.lat[ids] = _choice(c.latency_choices, n, dev)
        noise = torch.as_tensor(c.cam_pos_noise, dtype=torch.float32, device=dev)
        self.cams[ids] = self.cams_nominal.unsqueeze(0) + (torch.rand(n, 2, 3, device=dev) * 2 - 1) * noise
        self.margin[ids] = _uniform(c.margin_range, (n,), dev)
        self.p_lost[ids] = _uniform(c.lost_prob_range, (n,), dev)
        self.sigma[ids] = _uniform(c.noise_std_range, (n,), dev)
        self.bias_b[ids] = _uniform(c.reacquire_bias_range, (n,), dev)
        ang = (torch.rand(n, 3, device=dev) * 2 - 1) * math.radians(float(c.calib_rot_deg))
        self.cal_r[ids] = small_rotation(ang)
        xyz = torch.as_tensor(c.calib_xyz, dtype=torch.float32, device=dev)
        self.cal_t[ids] = (torch.rand(n, 3, device=dev) * 2 - 1) * xyz
        self.cal_c[ids] = particles[ids].mean(dim=1)
        self.scale[ids] = _uniform(c.size_scale_range, (n,), dev)
        self.covered[ids] = False
        self.back_at[ids] = 0
        self.fresh[ids] = True
        self.bias[ids] = 0.0
        for buf in (self.frames, self.frozen_frames, self.fresh_mv_sum, self.fresh_st_sum,
                    self.cov_hand_sum, self.cov_cloth_sum, self.uncov_share_sum):
            buf[ids] = 0.0

    # ------------------------------------------------------------------ one camera frame

    def _covered(self, ids, pts, particles, caps_a, caps_b, caps_r):
        """``(covered, by_hand, by_cloth)`` for the half points ``pts`` (``(n, M, 3)``), any camera."""
        c = self.cfg
        n, m = pts.shape[0], pts.shape[1]
        cams = self.cams[ids]  # (n, 2, 3)
        hand = torch.zeros(n, m, dtype=torch.bool, device=self.device)
        cloth = torch.zeros_like(hand)
        if c.hand_occlusion:
            # Ray from each camera to the point, stopped short of it.
            d = pts.unsqueeze(1) - cams.unsqueeze(2)                               # (n, 2, M, 3)
            dist = torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
            end = pts.unsqueeze(1) - d / dist * float(c.end_pullback)
            p1 = cams.view(n, 2, 1, 1, 3)
            q1 = end.unsqueeze(3)                                                  # (n, 2, M, 1, 3)
            a = caps_a[ids].view(n, 1, 1, -1, 3)
            b = caps_b[ids].view(n, 1, 1, -1, 3)
            reach = (caps_r.view(1, 1, 1, -1) + self.margin[ids].view(n, 1, 1, 1)) ** 2
            hit = segment_distance_sq(p1, q1, a, b) < reach                        # (n, 2, M, C)
            hand = hit.any(dim=-1).any(dim=1)
        if c.self_occlusion:
            parts = particles[ids]
            res = []
            for sel, tri in ((slice(0, self.m_mv), self.tri_st), (slice(self.m_mv, m), self.tri_mv)):
                p = pts[:, sel]                                                    # (n, Mh, 3)
                v = parts[:, tri]                                                  # (n, T, 3, 3)
                v0, v1, v2 = (v[:, :, i].view(n, 1, 1, -1, 3) for i in range(3))
                hits = ray_hits_triangles(
                    cams.view(n, 2, 1, 1, 3), p.view(n, 1, -1, 1, 3), v0, v1, v2, float(c.self_occlusion_gap)
                )                                                                  # (n, 2, Mh, T)
                res.append(hits.any(dim=-1).any(dim=1))
            cloth = torch.cat(res, dim=1)
        return hand | cloth, hand, cloth

    def _frame(self, ids, t, particles, rest, corner_rest, caps_a, caps_b, caps_r, force_all: bool):
        """One camera frame for envs ``ids``: visibility, noise, fit, and the hold rule."""
        c, dev = self.cfg, self.device
        n = int(ids.numel())
        tracked = torch.einsum("qp,npi->nqi", self.W, particles[ids])[:, self.half_pts]   # (n, M, 3)
        m = tracked.shape[1]
        if force_all:
            fresh = torch.ones(n, m, dtype=torch.bool, device=dev)
        else:
            covered, by_hand, by_cloth = self._covered(ids, tracked, particles, caps_a, caps_b, caps_r)
            self.cov_hand_sum[ids] += by_hand.float().mean(dim=1)
            self.cov_cloth_sum[ids] += by_cloth.float().mean(dim=1)
            if c.reacquire:
                tt = t[ids].unsqueeze(1)
                uncovered = self.covered[ids] & ~covered
                lo, hi = (round(float(v) * 60.0) for v in c.reacquire_delay_s)
                delay = torch.randint(lo, hi + 1, (n, m), device=dev)
                lo, hi = (round(float(v) * 60.0) for v in c.lost_duration_s)
                long = torch.randint(lo, hi + 1, (n, m), device=dev)
                lost = torch.rand(n, m, device=dev) < self.p_lost[ids].unsqueeze(1)
                back_at = torch.where(uncovered, tt + torch.where(lost, long, delay), self.back_at[ids])
                self.back_at[ids] = back_at
                fresh = ~covered & (tt >= back_at)
            else:
                fresh = ~covered
            self.covered[ids] = covered
            not_fresh = (~fresh).float().sum(dim=1)
            self.uncov_share_sum[ids] += torch.where(
                not_fresh > 0, (~fresh & ~covered).float().sum(dim=1) / not_fresh.clamp(min=1.0), torch.zeros_like(not_fresh)
            )
            # A point seen again is re-acquired at a slightly different spot on the texture.
            again = fresh & ~self.fresh[ids]
            b = self.bias_b[ids].view(n, 1, 1)
            new_bias = (torch.rand(n, m, 3, device=dev) * 2 - 1) * b
            self.bias[ids] = torch.where(again.unsqueeze(-1), new_bias, self.bias[ids])
        self.fresh[ids] = fresh

        obs = tracked + self.bias[ids] + torch.randn_like(tracked) * self.sigma[ids].view(n, 1, 1)
        cc = self.cal_c[ids].unsqueeze(1)
        obs = torch.einsum("nij,nmj->nmi", self.cal_r[ids], obs - cc) + cc + self.cal_t[ids].unsqueeze(1)

        # The pipeline's rest model: the true one at the assumed size.
        s = self.scale[ids].view(n, 1, 1)
        rest_ins = torch.einsum("qp,npi->nqi", self.W, rest[ids])[:, self.half_pts]
        rest_ins = torch.cat((rest_ins[..., :2] * s, rest_ins[..., 2:]), dim=-1)
        w = fresh.float()
        out = torch.empty(n, _OUT, device=dev)
        ok = torch.ones(n, dtype=torch.bool, device=dev)
        for h, sel in enumerate((slice(0, self.m_mv), slice(self.m_mv, m))):
            wh = w[:, sel]
            cnt = wh.sum(dim=1)
            (self.fresh_mv_sum if h == 0 else self.fresh_st_sum)[ids] += cnt
            spread = minor_spread(rest_ins[:, sel, :2], wh)
            ok &= (cnt >= int(c.n_min)) & (spread >= float(c.min_spread))
            r, tr = weighted_kabsch(rest_ins[:, sel], obs[:, sel], wh)
            out[:, 12 * h:12 * h + 9] = r.reshape(n, 9)
            out[:, 12 * h + 9:12 * h + 12] = tr
        if force_all:
            ok[:] = True
        self.frames[ids] += 1.0
        self.frozen_frames[ids] += (~ok).float()
        self.held[ids] = torch.where(ok.unsqueeze(1), out, self.held[ids])

    # ------------------------------------------------------------------ per step

    def step(self, t, start, particles, rest, corner_rest, caps_a=None, caps_b=None, caps_r=None):
        """Advance one policy step.

        ``t`` (N,) steps since the episode started; ``start`` (N,) envs at their first step;
        ``particles`` / ``rest`` (N, R*R, 3) env-frame particle positions and the TRUE rest grid;
        ``corner_rest`` (N, 4, 3) unused here (kept for the caller's symmetry); ``caps_*`` the
        occluder capsules (N, C, 3), (N, C, 3), (C,). Returns ``(R_mv, t_mv, R_st, t_st, scale)``.
        """
        del corner_rest
        if bool(start.any()):
            ids = start.nonzero(as_tuple=False).squeeze(1)
            self._draw_episode(ids, particles)
            self._frame(ids, t, particles, rest, None, caps_a, caps_b, caps_r, force_all=True)
            self.queue[ids] = self.held[ids].unsqueeze(1).expand(-1, self.queue_len, -1)
        cap = ~start & (torch.remainder(t - self.phase, self.k) == 0)
        if bool(cap.any()):
            ids = cap.nonzero(as_tuple=False).squeeze(1)
            self._frame(ids, t, particles, rest, None, caps_a, caps_b, caps_r, force_all=False)
        self.queue = torch.roll(self.queue, shifts=1, dims=1)
        self.queue[:, 0] = self.held
        out = self.queue[torch.arange(self.n, device=self.device), self.lat]
        return (
            out[:, 0:9].view(-1, 3, 3), out[:, 9:12],
            out[:, 12:21].view(-1, 3, 3), out[:, 21:24],
            self.scale,
        )

    def episode_stats(self) -> dict[str, torch.Tensor]:
        """Per-env means over this episode's camera frames (for ``episode_final`` logging)."""
        f = self.frames.clamp(min=1.0)
        return {
            "tracker_frozen": self.frozen_frames / f,
            "tracker_fresh_mv": self.fresh_mv_sum / f / float(self.m_mv),
            "tracker_fresh_st": self.fresh_st_sum / f / float(self.half_pts.numel() - self.m_mv),
            "tracker_cov_hand": self.cov_hand_sum / f,
            "tracker_cov_cloth": self.cov_cloth_sum / f,
            # Of the not-fresh points, the share not covered (re-acquisition delay / loss); the real
            # data has 3-6% (the user's m = 0 ball model explains 94-97% of the lost points).
            "tracker_notfresh_uncovered": self.uncov_share_sum / f,
        }
