"""An unlabelled point cloud of everything above the table, for a point-cloud policy.

What two depth cameras give on the real setup after the black table is filtered out and the cloud
is cropped to a box over the table: the cloth, the hand and the arm, as bare ``xyz`` -- no point
says what it belongs to. Built from the sim's state without rendering:

* candidates -- samples on the sheet (bilinear in the particle grid, redrawn every step) and a
  baked set of samples on the robot's link meshes (``robot_surface.py``), moved by the body poses;
* visibility -- each camera splats every candidate into a small depth image (z-buffer, 3x3
  dilation); a candidate is seen by a camera when it is inside the image (the D415 depth field of
  view) and within ``depth_tol`` of the nearest surface in its pixel. One model for every kind of
  occlusion: cloth by robot, robot by cloth, either by itself;
* noise -- per-camera calibration error, depth noise along the ray, random drops, a per-episode
  delay of the whole cloud (all off by default; ``domain_randomization.pointcloud``);
* crop and count -- the box, then ``n_points`` candidates drawn with probability proportional to
  the surface area each stands for (uniform density over what is visible, like a voxel filter).

Optional extra groups, each with its own label column (none of them when unused, so the plain
cloud is ``xyz`` only):
  ``goal_mode="label"``       the cloth points of the first frame with moving / stationary labels,
                              frozen for the episode (says WHICH half folds);
  ``goal_mode="goal_fixed"``  the FOLDED sheet where it belongs in the scene (samples on the goal
                              state carried on the sheet's pose at ``ref_step``), frozen, one
                              ``goal`` label; needs ``fold_pool.goal_task``;
  ``goal_mode="goal_any"``    the folded sheet's shape only: the same samples about the box centre
                              on the table plane, turned by a random yaw per episode (says nothing
                              about where the sheet is or which way to fold);
  ``fk_points > 0``      points on the forearm and hand from forward kinematics (no camera error).

Every point is ``(xyz - box centre) * scale`` (+ a one-hot label when there is more than one
group). Torch only, unit-tested on CPU (tests/test_pointcloud.py). Metres, env frame. The env's
observation of it is ``perception/pointcloud_obs.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine


def visible_warp(*args, **kwargs):
    """Lazy: Warp is imported only when ``use_warp`` is on."""
    from isaacsimenvs.tasks.cloth.perception.visibility_warp import visible_warp as fn

    return fn(*args, **kwargs)

__all__ = ["GOAL_MODES", "PointCloudSetup", "PointCloud", "look_at", "quat_rotate"]

#: The goal group of the cloud (module doc). Checkpoints from before 2026-11-19 name them first /
#: scene / shape (``scripts/distillation/student.legacy_env_overrides`` translates).
GOAL_MODES = ("none", "label", "goal_fixed", "goal_any")


@dataclass
class PointCloudSetup:
    """Everything the cloud is built from: what the run chose (``obs.pointcloud``), the camera errors
    (``domain_randomization.pointcloud``) and the fixed setup (box, cameras, image)."""

    n_points: int = 1024
    """Live points per step (the fixed output size; unused slots are masked)."""
    n_cloth_cand: int = 768
    """Cloth candidates per step (also the cloth's share of the z-buffer occluders)."""
    goal_mode: str = "none"
    """``none``, ``label`` (first-frame cloth cloud with moving / stationary labels), ``goal_fixed``
    or ``goal_any`` (a cloud of the folded goal sheet, in place / as a bare shape at any yaw)."""
    n_ref: int = 192
    ref_step: int = 0
    fk_points: int = 0
    """Forward-kinematics points on the forearm and hand (0: none)."""
    scale: float = 10.0

    # --- the box: the 22 x 15 inch table top, from the table edge next to the robot ---
    box_half_x: float = 0.2794
    box_y_near: float = 0.20
    """The table edge on the robot's side, env frame (the robot sits at +y)."""
    box_depth: float = 0.381
    box_below: float = 0.02
    """The box starts this far below the table top. No upper face: the cameras' view ends it."""

    # --- cameras (RealSense D415 depth: 65 x 40 deg) ---
    cam_a_base: tuple[float, float, float] = (0.6446, -0.9631, 0.8564)
    """Camera A's optical centre in the robot base frame (left_camera_nov7.json, ``T_base_cam``).
    Env frame = base + ``robot_root_pos``."""
    cam_b_base: tuple[float, float, float] = (-0.5352, -0.9046, 0.8636)
    """Camera B (right_camera_nov10.json)."""
    cam_a_rot: tuple[float, ...] | None = (0.45143, 0.56156, -0.69344, 0.89216, -0.26982, 0.36229, 0.01635, -0.78221, -0.62280)
    """Camera A's rotation in the base frame (``T_base_cam[:3, :3]``, row-major): its columns are
    the camera's x (right), y (down), z (forward) axes. ``None``: look at the box centre, no roll."""
    cam_b_rot: tuple[float, ...] | None = (0.37935, -0.59643, 0.70736, -0.92303, -0.19097, 0.33399, -0.06412, -0.77962, -0.62297)
    fov_h_deg: float = 65.0
    fov_v_deg: float = 40.0
    img_w: int = 208
    """Depth-image width for the visibility test (one pixel is ~6 mm at 1 m)."""
    depth_tol: float = 0.02
    """A candidate is hidden when a surface in its 3x3 pixel neighbourhood is nearer by more than
    this [m]. It has to exceed the depth change over two pixels of a surface seen at a slant (the
    table plane from these cameras: ~9 mm per pixel), so an occluder closer than this does not
    hide: the cloth right under a finger and the far side of a finger stay in the cloud."""
    min_depth: float = 0.2
    chunk: int = 1024
    """Envs per batch (memory)."""

    # --- camera errors (all off by default; domain_randomization.pointcloud) ---
    cam_pos_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    calib_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    calib_rot_deg: float = 0.0
    """Per-episode rigid error of EACH camera's cloud, about the box centre."""
    depth_noise_std_range: tuple[float, float] = (0.0, 0.0)
    drop_prob_range: tuple[float, float] = (0.0, 0.0)
    delay_max: int = 0
    """The cloud is this many policy steps old at most (per-episode lag, uniform in 0..max)."""

    def __post_init__(self) -> None:
        if self.goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, got {self.goal_mode!r}")

    @property
    def labels(self) -> tuple[str, ...]:
        """The label columns; empty when the cloud has one group only."""
        out = ["live"]
        if self.goal_mode == "label":
            out += ["ref_moving", "ref_stationary"]
        if self.goal_mode in ("goal_fixed", "goal_any"):
            out.append("goal")
        if self.fk_points > 0:
            out.append("fk")
        return tuple(out) if len(out) > 1 else ()

    @property
    def point_dim(self) -> int:
        return 3 + len(self.labels)

    @property
    def num_points(self) -> int:
        return int(self.n_points) + (int(self.n_ref) if self.goal_mode != "none" else 0) + int(self.fk_points)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (..., 3) by the unit quaternion ``q`` (..., 4), wxyz."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def look_at(cam: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` rotation whose columns are the camera's x (right), y (down), z (forward)
    axes in the world, for a camera at ``cam`` looking at ``target`` with no roll (world z is up)."""
    z = torch.nn.functional.normalize(target - cam, dim=-1)
    up = torch.zeros_like(z)
    up[..., 2] = 1.0
    x = torch.nn.functional.normalize(torch.cross(z, up, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _small_rotation(angles: torch.Tensor) -> torch.Tensor:
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
    return lo + (hi - lo) * torch.rand(shape, device=device)


# ----------------------------------------------------------------------------
# The cloud
# ----------------------------------------------------------------------------


class PointCloud:
    """See the module doc.

    ``pool_body`` ``(P,)`` body index of each robot surface sample, ``pool_off`` ``(P, 3)`` its
    position in that body's frame, ``pool_w`` ``(P,)`` the area [m^2] it stands for; ``fk_idx``
    the pool samples used as forward-kinematics points. ``cams_env`` ``(2, 3)`` camera positions,
    ``cam_rots`` ``(2, 3, 3)`` or ``None`` (look at the box centre); ``table_z`` the table top.
    """

    def __init__(self, cfg: PointCloudSetup, num_envs: int, resolution: int, cams_env, table_z: float,
                 pool_body, pool_off, pool_w, device, cam_rots=None, fk_idx=None) -> None:
        self.cfg, self.n, self.r, self.device = cfg, int(num_envs), int(resolution), device
        n, dev = self.n, device
        self.pool_body = torch.as_tensor(pool_body, dtype=torch.long, device=dev)
        self.pool_off = torch.as_tensor(pool_off, dtype=torch.float32, device=dev)
        self.pool_w = torch.as_tensor(pool_w, dtype=torch.float32, device=dev)
        self.n_robot, self.n_cloth = int(self.pool_body.numel()), int(cfg.n_cloth_cand)
        if cfg.fk_points > 0:
            if fk_idx is None or len(fk_idx) != cfg.fk_points:
                raise ValueError(f"fk_points {cfg.fk_points} needs fk_idx of that length")
            self.fk_idx = torch.as_tensor(fk_idx, dtype=torch.long, device=dev)
        self.labels = cfg.labels
        self.point_dim, self.num_points = cfg.point_dim, cfg.num_points
        self.mid = (self.r - 1) / 2.0

        self.box_lo = torch.tensor(
            [-cfg.box_half_x, cfg.box_y_near - cfg.box_depth, table_z - cfg.box_below], device=dev)
        self.box_hi = torch.tensor([cfg.box_half_x, cfg.box_y_near, float("inf")], device=dev)
        self.centre = torch.tensor([0.0, cfg.box_y_near - 0.5 * cfg.box_depth, table_z], device=dev)
        self.f = 0.5 * cfg.img_w / math.tan(math.radians(cfg.fov_h_deg) / 2.0)
        self.w = int(cfg.img_w)
        self.h = 2 * int(round(self.f * math.tan(math.radians(cfg.fov_v_deg) / 2.0)))

        self.cams0 = torch.as_tensor(cams_env, dtype=torch.float32, device=dev).view(2, 3)
        self.rots0 = None if cam_rots is None else torch.as_tensor(cam_rots, dtype=torch.float32, device=dev).view(2, 3, 3)
        self.cams = self.cams0.unsqueeze(0).repeat(n, 1, 1)
        self.rots = self._rotations(self.cams)
        self.calib_r = torch.eye(3, device=dev).repeat(n, 2, 1, 1)
        self.calib_t = torch.zeros(n, 2, 3, device=dev)
        self.depth_std = torch.zeros(n, device=dev)
        self.drop_p = torch.zeros(n, device=dev)
        self.lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.delay = DelayLine(n, cfg.delay_max, (cfg.n_points, 4), dev, torch.float16)
        k = int(cfg.n_ref)
        self.ref_pts = torch.zeros(n, k, 3, device=dev)
        self.ref_mask = torch.zeros(n, k, dtype=torch.bool, device=dev)
        self.ref_moving = torch.zeros(n, k, dtype=torch.bool, device=dev)
        self.ref_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_cloth_share = torch.zeros(n, device=dev)
        self.last_visible_frac = torch.ones(n, device=dev)
        self.last_count = torch.zeros(n, device=dev)
        self.manipuland_only = False
        self.use_warp = False
        """Run the visibility test as fused Warp kernels (``visibility_warp``): same result, ~8x faster
        (~20x with ``manipuland_only``, where the robot is only drawn, never built as points)."""
        """Keep only the sheet's (manipuland's) points in the live cloud, as after segmentation; the
        robot is still drawn into the depth images, so it still hides what is behind it."""

    def _rotations(self, cams: torch.Tensor) -> torch.Tensor:
        if self.rots0 is not None:
            return self.rots0.unsqueeze(0).expand(cams.shape[0], -1, -1, -1).clone()
        return look_at(cams, self.centre.view(1, 1, 3).expand_as(cams))

    # ------------------------------------------------------------------ per-episode draws

    def reset(self, ids: torch.Tensor) -> None:
        """New episode for envs ``ids``."""
        c, dev, m = self.cfg, self.device, int(ids.numel())
        if m == 0:
            return
        noise = torch.as_tensor(c.cam_pos_noise, dtype=torch.float32, device=dev)
        self.cams[ids] = self.cams0 + (2.0 * torch.rand(m, 2, 3, device=dev) - 1.0) * noise
        self.rots[ids] = self._rotations(self.cams[ids])
        ang = (2.0 * torch.rand(m, 2, 3, device=dev) - 1.0) * math.radians(float(c.calib_rot_deg))
        self.calib_r[ids] = _small_rotation(ang.view(-1, 3)).view(m, 2, 3, 3)
        xyz = torch.as_tensor(c.calib_xyz, dtype=torch.float32, device=dev)
        self.calib_t[ids] = (2.0 * torch.rand(m, 2, 3, device=dev) - 1.0) * xyz
        self.depth_std[ids] = _uniform(c.depth_noise_std_range, (m,), dev)
        self.drop_p[ids] = _uniform(c.drop_prob_range, (m,), dev)
        self.lag[ids] = torch.randint(0, int(c.delay_max) + 1, (m,), device=dev)
        self.fresh[ids] = True
        self.ref_valid[ids] = False
        self.ref_mask[ids] = False

    # ------------------------------------------------------------------ candidates

    def cloth_candidates(self, particles: torch.Tensor):
        """``(pts (n, K, 3), moving (n, K), weight (n, K))``: uniform samples on each sheet."""
        n, r, k = particles.shape[0], self.r, self.n_cloth
        uv = torch.rand(n, k, 2, device=particles.device) * (r - 1)
        i0 = uv.floor().clamp(max=r - 2).long()
        f = uv - i0
        base = i0[..., 0] * r + i0[..., 1]
        idx = torch.stack((base, base + r, base + 1, base + r + 1), dim=-1)
        wgt = torch.stack(
            ((1 - f[..., 0]) * (1 - f[..., 1]), f[..., 0] * (1 - f[..., 1]), (1 - f[..., 0]) * f[..., 1], f[..., 0] * f[..., 1]),
            dim=-1,
        )
        corners = torch.gather(particles, 1, idx.view(n, -1, 1).expand(-1, -1, 3)).view(n, k, 4, 3)
        grid = particles.view(n, r, r, 3)
        e1, e2 = grid[:, 1:, :-1] - grid[:, :-1, :-1], grid[:, :-1, 1:] - grid[:, :-1, :-1]
        area = torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1).sum(dim=(1, 2))           # (n,)
        return (corners * wgt.unsqueeze(-1)).sum(dim=2), uv[..., 0] > self.mid, (area / k).unsqueeze(1).expand(-1, k)

    def robot_points(self, body_pos: torch.Tensor, body_quat: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """``(n, P, 3)`` robot surface samples (``idx``: a subset of the pool) in the env frame."""
        body = self.pool_body if idx is None else self.pool_body[idx]
        off = self.pool_off if idx is None else self.pool_off[idx]
        return body_pos[:, body] + quat_rotate(body_quat[:, body], off.unsqueeze(0))

    # ------------------------------------------------------------------ visibility

    def visible(self, pts: torch.Tensor, cams: torch.Tensor, rots: torch.Tensor) -> torch.Tensor:
        """``(n, 2, P)``: camera c sees the point (inside its image, nearest surface in its pixel)."""
        if self.use_warp:
            return visible_warp(pts, cams, rots, self.f, self.w, self.h, float(self.cfg.min_depth), float(self.cfg.depth_tol))
        n, p = pts.shape[0], pts.shape[1]
        w, h, f = self.w, self.h, self.f
        local = torch.matmul(pts.unsqueeze(1) - cams.unsqueeze(2), rots)                        # (n, 2, P, 3): R^T d
        z = local[..., 2]
        inv = f / z.clamp(min=1e-3)
        u = torch.floor(local[..., 0] * inv + 0.5 * w)
        v = torch.floor(local[..., 1] * inv + 0.5 * h)
        ok = (z > float(self.cfg.min_depth)) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & torch.isfinite(z)
        pix = torch.where(ok, v * w + u, torch.zeros_like(u)).long().view(n * 2, p)
        depth = torch.where(ok, z, torch.full_like(z, float("inf"))).view(n * 2, p)
        zbuf = torch.full((n * 2, h * w), float("inf"), device=pts.device, dtype=depth.dtype)
        zbuf.scatter_reduce_(1, pix, depth, reduce="amin", include_self=True)
        near = -torch.nn.functional.max_pool2d(-zbuf.view(n * 2, 1, h, w), 3, 1, 1).view(n * 2, h * w)
        return (ok.view(n * 2, p) & (depth <= near.gather(1, pix) + float(self.cfg.depth_tol))).view(n, 2, p)

    def _live(self, sl: slice, particles, body_pos, body_quat):
        """One batch of envs: ``(xyz (n, n_points, 3), mask, camera, is_cloth)`` of the live cloud
        and the cloth candidates ``(pts (n, K, 3), ok (n, K), moving (n, K))`` after noise + crop."""
        c, dev = self.cfg, particles.device
        n, k = particles.shape[0], self.n_cloth
        cloth, moving, w_cloth = self.cloth_candidates(particles)
        cams = self.cams[sl]
        if self.manipuland_only and self.use_warp:
            # Only the sheet's points are kept: the robot is drawn into the depth images inside the
            # kernel (its surface never built as a tensor) and only the sheet's candidates are tested.
            pts, weight = cloth, w_cloth
            vis = visible_warp(cloth, cams, self.rots[sl], self.f, self.w, self.h, float(c.min_depth),
                               float(c.depth_tol), body_pos=body_pos, body_quat=body_quat,
                               pool_body=self.pool_body, pool_off=self.pool_off)
        else:
            pts = torch.cat((cloth, self.robot_points(body_pos, body_quat)), dim=1)            # (n, P, 3)
            weight = torch.cat((w_cloth, self.pool_w.unsqueeze(0).expand(n, -1)), dim=1)
            vis = self.visible(pts, cams, self.rots[sl])
        seen = vis[:, 0] | vis[:, 1]
        self.last_visible_frac[sl] = seen[:, :k].float().mean(dim=1)
        # A point both cameras see comes from either; then that camera's depth noise and calibration error.
        use_b = vis[:, 1] & (~vis[:, 0] | (torch.rand(seen.shape, device=dev) < 0.5))
        cam = use_b.long()
        pick = use_b.unsqueeze(-1)
        ray = torch.nn.functional.normalize(pts - torch.where(pick, cams[:, 1:2], cams[:, 0:1]), dim=-1)
        pts = pts + ray * (torch.randn(seen.shape, device=dev) * self.depth_std[sl].unsqueeze(1)).unsqueeze(-1)
        rel, cr, ct = pts - self.centre, self.calib_r[sl], self.calib_t[sl]
        pts = torch.where(pick, torch.matmul(rel, cr[:, 1].transpose(1, 2)) + ct[:, 1:2],
                          torch.matmul(rel, cr[:, 0].transpose(1, 2)) + ct[:, 0:1]) + self.centre
        ok = seen & (torch.rand(seen.shape, device=dev) >= self.drop_p[sl].unsqueeze(1))
        ok = ok & ((pts >= self.box_lo) & (pts <= self.box_hi)).all(dim=-1) & torch.isfinite(pts).all(dim=-1)
        if self.manipuland_only:            # the robot still occludes, but its own points are not kept
            ok[:, k:] = False
        # n_points candidates, probability proportional to area (Gumbel top-k = weighted, no repeats).
        gumbel = -torch.log(-torch.log(torch.rand(seen.shape, device=dev).clamp(1e-9, 1.0 - 1e-9)))
        key = torch.where(ok, torch.log(weight.clamp(min=1e-12)) + gumbel, torch.full_like(gumbel, float("-inf")))
        short = int(c.n_points) - key.shape[1]
        if short > 0:                       # fewer candidates than slots: the spare slots stay empty
            key = torch.cat((key, torch.full((n, short), float("-inf"), device=dev)), dim=1)
            pts = torch.cat((pts, torch.zeros(n, short, 3, device=dev)), dim=1)
            cam = torch.cat((cam, torch.zeros(n, short, dtype=torch.long, device=dev)), dim=1)
        top, idx = key.topk(int(c.n_points), dim=1)
        mask = torch.isfinite(top)
        xyz = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3)) * mask.unsqueeze(-1)
        return xyz, mask, torch.gather(cam, 1, idx), idx < k, (pts[:, :k], ok[:, :k], moving)

    # ------------------------------------------------------------------ the observation

    def _sample_sheet(self, grid: torch.Tensor, k: int) -> torch.Tensor:
        """``(n, k, 3)`` uniform samples on a sheet given as its particle grid ``(n, R*R, 3)``."""
        n, r = grid.shape[0], self.r
        uv = torch.rand(n, k, 2, device=grid.device) * (r - 1)
        i0 = uv.floor().clamp(max=r - 2).long()
        f = uv - i0
        base = i0[..., 0] * r + i0[..., 1]
        idx = torch.stack((base, base + r, base + 1, base + r + 1), dim=-1)
        wgt = torch.stack(
            ((1 - f[..., 0]) * (1 - f[..., 1]), f[..., 0] * (1 - f[..., 1]), (1 - f[..., 0]) * f[..., 1], f[..., 0] * f[..., 1]),
            dim=-1,
        )
        corners = torch.gather(grid, 1, idx.view(n, -1, 1).expand(-1, -1, 3)).view(n, k, 4, 3)
        return (corners * wgt.unsqueeze(-1)).sum(dim=2)

    def observe(self, step, particles, body_pos, body_quat, fk_pos=None, fk_quat=None, goal_now=None, goal_rest=None):
        """``(points (N, M, point_dim), mask (N, M))``.

        ``goal_now`` / ``goal_rest`` ``(N, R*R, 3)`` (``goal_mode`` goal_fixed / goal_any): the goal sheet's
        particle grid where it belongs now (env frame) and in its own frame (flat, z = 0).

        ``step`` ``(N,)`` episode step; ``particles`` ``(N, R*R, 3)``; ``body_pos`` ``(N, B, 3)`` and
        ``body_quat`` ``(N, B, 4)`` (wxyz) the robot's body poses, env frame. ``fk_pos`` /
        ``fk_quat``: the body poses the forward-kinematics points are built from (the delayed
        proprioception; default the current ones).
        """
        c, n, dev = self.cfg, particles.shape[0], particles.device
        parts = [
            self._live(slice(lo, min(lo + int(c.chunk), n)), particles[lo:lo + int(c.chunk)],
                       body_pos[lo:lo + int(c.chunk)], body_quat[lo:lo + int(c.chunk)])
            for lo in range(0, n, int(c.chunk))
        ]
        xyz, mask, cam, is_cloth = (torch.cat([p[i] for p in parts], dim=0) for i in range(4))
        cl_pts, cl_ok, cl_moving = (torch.cat([p[4][i] for p in parts], dim=0) for i in range(3))
        self.last_count = mask.float().sum(dim=1)
        self.last_cloth_share = (is_cloth & mask).float().sum(dim=1) / self.last_count.clamp(min=1.0)

        if c.goal_mode == "label":          # before the delay: the reference is the first frame as it was
            take = ~self.ref_valid & (step >= int(c.ref_step))
            if bool(take.any()):
                key = torch.where(cl_ok, torch.rand(cl_ok.shape, device=dev), torch.full(cl_ok.shape, -1.0, device=dev))
                top, idx = key.topk(int(c.n_ref), dim=1)
                self.ref_pts[take] = torch.gather(cl_pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))[take]
                self.ref_mask[take] = (top >= 0.0)[take]
                self.ref_moving[take] = torch.gather(cl_moving, 1, idx)[take]
                self.ref_valid[take] = True

        if c.goal_mode in ("goal_fixed", "goal_any"):
            # goal_fixed: taken once the sheet has landed (ref_step); goal_any: there from the first step.
            take = ~self.ref_valid & ((step >= int(c.ref_step)) | (c.goal_mode == "goal_any"))
            if bool(take.any()):
                if c.goal_mode == "goal_fixed":
                    pts = self._sample_sheet(goal_now, int(c.n_ref))
                    ok = ((pts >= self.box_lo) & (pts <= self.box_hi)).all(dim=-1) & torch.isfinite(pts).all(dim=-1)
                else:
                    pts = self._sample_sheet(goal_rest, int(c.n_ref))
                    pts = pts - goal_rest.mean(dim=1, keepdim=True)
                    yaw = 2.0 * math.pi * torch.rand(n, device=dev)
                    cy, sy = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
                    pts = torch.stack((cy * pts[..., 0] - sy * pts[..., 1], sy * pts[..., 0] + cy * pts[..., 1], pts[..., 2]), dim=-1)
                    pts = pts + self.centre
                    ok = torch.isfinite(pts).all(dim=-1)
                self.ref_pts[take], self.ref_mask[take] = pts[take], ok[take]
                self.ref_valid[take] = True

        if c.delay_max > 0:                 # stored relative to the box centre (float16)
            now = torch.cat((xyz - self.centre, mask.unsqueeze(-1).to(xyz.dtype)), dim=-1)
            old = self.delay.push_and_read(now, self.fresh, self.lag).float()
            xyz, mask = old[..., :3] + self.centre, old[..., 3] > 0.5
        self.fresh[:] = False

        # (xyz, mask, label) per group; the label is an index into ``self.labels``.
        groups = [(xyz, mask, torch.zeros(mask.shape, dtype=torch.long, device=dev))]
        if c.goal_mode == "label":
            groups.append((self.ref_pts, self.ref_mask, torch.where(self.ref_moving, 1, 2)))
        if c.goal_mode in ("goal_fixed", "goal_any"):
            groups.append((self.ref_pts, self.ref_mask, torch.ones(self.ref_mask.shape, dtype=torch.long, device=dev)))
        if c.fk_points > 0:
            fk = self.robot_points(body_pos if fk_pos is None else fk_pos, body_quat if fk_quat is None else fk_quat, self.fk_idx)
            groups.append((fk, torch.ones(fk.shape[:2], dtype=torch.bool, device=dev),
                           torch.full(fk.shape[:2], len(self.labels) - 1, dtype=torch.long, device=dev)))

        all_xyz = torch.cat([g[0] for g in groups], dim=1)
        all_mask = torch.cat([g[1] for g in groups], dim=1)
        label = torch.cat([g[2] for g in groups], dim=1)
        # For viewers: the cloud in the env frame, its labels and the camera each live point came from.
        self.last_xyz, self.last_mask, self.last_label = all_xyz, all_mask, label
        self.last_cam = torch.cat((cam, torch.full((n, all_xyz.shape[1] - cam.shape[1]), -1, dtype=torch.long, device=dev)), dim=1)
        feat = (all_xyz - self.centre) * float(c.scale)
        if self.labels:
            feat = torch.cat((feat, torch.nn.functional.one_hot(label, len(self.labels)).to(feat.dtype)), dim=-1)
        ok = all_mask & torch.isfinite(feat).all(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) * ok.unsqueeze(-1), ok
