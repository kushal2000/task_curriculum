"""A point-cloud observation of the cloth for a distilled student policy (no point identities).

What two depth cameras would give after RGB segmentation, built from the sim's state without
rendering:

* cloth points  -- random samples on the sheet surface (bilinear in the particle grid, new samples
  every step, so no slot carries a material point), kept only where at least one camera sees them:
  the ray from the camera must miss the hand capsules and the other half of the sheet;
* hand points   -- samples on the same capsules, i.e. forward kinematics on the robot ("imagined"
  points, DexPoint), never occluded;
* a goal input, one of
    g0  the four true fold-target corners (follow the stationary half; an oracle),
    g3  the same from an erroneous stationary-half pose (a registration estimate: per-episode bias
        + per-frame noise in the table plane),
    g1  the cloth cloud captured once at ``ref_step`` with moving / stationary labels, then frozen
        (says WHICH half folds; where the sheet is now must come from the live cloud),
    gl  no extra points; the live cloth points carry moving / stationary labels (a marked cloth).

Every point is ``(xyz - palm) * scale`` followed by a one-hot source label (``LABELS``). The output
has a fixed size; ``mask`` marks the slots that hold a point.

Torch only (no Isaac Lab, no Kit), unit-tested on CPU (tests/test_cloth_pointcloud.py). Lengths in
metres, everything in the env frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["LABELS", "GOAL_MODES", "PointCloudCfg", "ClothPointCloud", "noise_preset", "g3_noise_preset",
           "hand_capsules"]

LABELS = ("cloth", "cloth_moving", "cloth_stationary", "hand", "ref_moving", "ref_stationary", "goal")
GOAL_MODES = ("g0", "g3", "g1", "gl")


@dataclass
class PointCloudCfg:
    n_cloth: int = 192
    """Cloth sample slots per step (before visibility)."""
    hand_pts_per_capsule: int = 3
    hand_cam: bool = False
    """Also give the hand points AS THE CAMERAS SEE THEM (each through one camera's calibration
    error and depth noise, with the random drop), next to the forward-kinematics ones. The offset
    between the two is the calibration error, which is otherwise unobservable. g1 / gl only: the
    points reuse the ``goal`` label slot, which those modes leave empty, so the point width and
    older checkpoints stay as they are."""
    visibility: str = "cams"
    """``cams``: a point needs one camera with a free line of sight. ``none``: every sample is kept."""
    goal_mode: str = "g0"
    ref_step: int = 0
    """g1: the reference cloud is captured at the first observation with episode step >= this."""
    scale: float = 10.0
    """Points are in units of ``1 / scale`` metres (decimetres), so they are O(1) without a normaliser."""

    # --- the real setup: two cameras, and the capsules that stand for the arm and hand ---
    cam_a_base: tuple[float, float, float] = (0.645, -0.963, 0.856)
    """Camera A's optical centre in the robot base frame. Env frame = base + ``robot_root_pos``."""
    cam_b_base: tuple[float, float, float] = (-0.525, -0.936, 0.876)
    finger_radius: float = 0.009
    palm_radius: float = 0.02
    wrist_radius: float = 0.04
    arm_radius: float = 0.05

    # --- visibility geometry ---
    end_pullback: float = 0.005
    """The ray to a point stops this short of it for the hand test (a finger touching from below)."""
    self_gap: float = 0.001
    """A moving-half point is hidden by the stationary half only if hit at least this far before it."""
    overlap_tol: float = 0.002
    """A stationary-half point is hidden by the moving half also when the hit is up to this far
    BEHIND it: PhysX plates fold into the same plane (ply gap 0), and the moving half is on top."""
    chunk: int = 1024
    """Envs per visibility batch (memory)."""

    # --- noise (all off by default; ``noise_preset``) ---
    cam_pos_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    margin_range: tuple[float, float] = (0.0, 0.0)
    """Hand capsule inflation per episode (depth pixels next to an occluder's edge are dropped)."""
    calib_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    calib_rot_deg: float = 0.0
    """Per-episode rigid error of EACH camera's cloud (about the sheet centre at reset): the two
    clouds do not coincide."""
    depth_noise_std_range: tuple[float, float] = (0.0, 0.0)
    """Per-episode std of per-point noise along the camera ray [m]."""
    drop_prob_range: tuple[float, float] = (0.0, 0.0)
    """Per-episode probability that a visible point is dropped (holes, segmentation misses)."""
    tip_drop_prob: float = 0.0
    tip_drop_radius_range: tuple[float, float] = (0.01, 0.03)
    """With ``tip_drop_prob`` per episode, cloth points within this radius of a fingertip are
    dropped (the segmentation loses the cloth next to the fingers)."""

    # --- g3: error of the stationary-half pose estimate, in the table plane ---
    g3_bias_xy: float = 0.0
    g3_bias_yaw_deg: float = 0.0
    g3_noise_xy: float = 0.0
    g3_noise_yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.visibility not in ("cams", "none"):
            raise ValueError(f"visibility must be cams or none, got {self.visibility!r}")
        if self.goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, got {self.goal_mode!r}")


def noise_preset(cfg: PointCloudCfg) -> PointCloudCfg:
    """Version C: camera position +-5 cm, capsule margin 0-1.5 cm, per-camera calibration
    +-1.5/1.5/1 cm +-2 deg, depth noise 1-4 mm, 0-30% of the
    points dropped, and in half the episodes the cloth within 1-3 cm of a fingertip dropped."""
    cfg.cam_pos_noise = (0.05, 0.05, 0.05)
    cfg.margin_range = (0.0, 0.015)
    cfg.calib_xyz = (0.015, 0.015, 0.01)
    cfg.calib_rot_deg = 2.0
    cfg.depth_noise_std_range = (0.001, 0.004)
    cfg.drop_prob_range = (0.0, 0.3)
    cfg.tip_drop_prob = 0.5
    return cfg


def g3_noise_preset(cfg: PointCloudCfg) -> PointCloudCfg:
    """g3 estimate error: bias +-1 cm / +-5 deg per episode, +-2 mm / +-1 deg per frame."""
    cfg.g3_bias_xy, cfg.g3_bias_yaw_deg = 0.01, 5.0
    cfg.g3_noise_xy, cfg.g3_noise_yaw_deg = 0.002, 1.0
    return cfg


# ----------------------------------------------------------------------------
# Geometry (broadcasting)
# ----------------------------------------------------------------------------


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


def small_rotation(angles: torch.Tensor) -> torch.Tensor:
    """``(n, 3, 3)`` = Rz Ry Rx for per-axis angles ``(n, 3)`` [rad]."""
    cx, cy, cz = torch.cos(angles).unbind(-1)
    sx, sy, sz = torch.sin(angles).unbind(-1)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    rx = torch.stack((one, zero, zero, zero, cx, -sx, zero, sx, cx), -1).view(-1, 3, 3)
    ry = torch.stack((cy, zero, sy, zero, one, zero, -sy, zero, cy), -1).view(-1, 3, 3)
    rz = torch.stack((cz, -sz, zero, sz, cz, zero, zero, zero, one), -1).view(-1, 3, 3)
    return rz @ ry @ rx


# ----------------------------------------------------------------------------
# Hand capsules (Flexiv arm + Wuji hand)
# ----------------------------------------------------------------------------

#: Wrist point (link7 -> flange -> mount -> wrist) and fingertip ends, in the body frames after the
#: importer merges fixed joints (flange / mount / wrist into link7, the tip into the distal).
_WRIST_OFFSET = (0.003, 0.00025, 0.0955)
_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
_TIP = {"thumb": (0.0, 0.0, -0.02978)}
_TIP_DEFAULT = (0.0, 0.0, -0.02475)


def hand_capsules(cfg: PointCloudCfg, prefix: str = "") -> list:
    """``[(body_a, offset_a, body_b, offset_b, radius)]``: forearm, wrist, palm fan and finger
    segments of the Flexiv + Wuji robot. ``prefix`` is the hand's body-name prefix (``r_`` / ``l_``
    or empty)."""
    z, w = (0.0, 0.0, 0.0), _WRIST_OFFSET
    caps = [
        ("link5", z, "link6", z, cfg.arm_radius),
        ("link6", z, "link7", z, cfg.arm_radius),
        ("link7", z, "link7", w, cfg.wrist_radius),
    ]
    for f in _FINGERS:
        caps.append(("link7", w, f"{prefix}{f}_proximal", z, cfg.palm_radius))
        caps += [
            (f"{prefix}{f}_proximal_abd", z, f"{prefix}{f}_middle", z, cfg.finger_radius),
            (f"{prefix}{f}_middle", z, f"{prefix}{f}_distal", z, cfg.finger_radius),
            (f"{prefix}{f}_distal", z, f"{prefix}{f}_distal", _TIP.get(f, _TIP_DEFAULT), cfg.finger_radius),
        ]
    caps.append((f"{prefix}index_finger_proximal", z, f"{prefix}pinky_proximal", z, cfg.palm_radius))
    return caps


def _uniform(lo_hi, shape, device) -> torch.Tensor:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return lo + (hi - lo) * torch.rand(shape, device=device)


def _rot_z(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).view(*yaw.shape, 3, 3)


class ClothPointCloud:
    """See the module doc. ``cams_env`` ``(2, 3)`` camera positions in the env frame; ``cap_radii``
    ``(C,)`` the hand capsules' radii (``None`` or empty: no hand)."""

    def __init__(self, cfg: PointCloudCfg, num_envs: int, resolution: int, cams_env, cap_radii, device) -> None:
        self.cfg, self.n, self.r, self.device = cfg, int(num_envs), int(resolution), device
        r = self.r
        self.mid = (r - 1) / 2.0
        self.tri_st = cell_triangles(r, 0, (r - 1) // 2).to(device)
        self.tri_mv = cell_triangles(r, r // 2, r - 1).to(device)
        self.cams0 = torch.as_tensor(cams_env, dtype=torch.float32, device=device).view(2, 3)
        self.cap_r = (
            torch.zeros(0, device=device) if cap_radii is None
            else torch.as_tensor(cap_radii, dtype=torch.float32, device=device).view(-1)
        )
        n, k = self.n, int(cfg.n_cloth)
        self.n_hand = int(self.cap_r.numel()) * int(cfg.hand_pts_per_capsule)
        self.n_extra = {"g0": 4, "g3": 4, "g1": k, "gl": 0}[cfg.goal_mode]
        if cfg.hand_cam and cfg.goal_mode not in ("g1", "gl"):
            raise ValueError("hand_cam reuses the goal label slot: g1 / gl only")
        self.n_hand_cam = self.n_hand if cfg.hand_cam else 0
        self.num_points = k + self.n_hand + self.n_extra + self.n_hand_cam
        self.point_dim = 3 + len(LABELS)

        self.cams = self.cams0.unsqueeze(0).repeat(n, 1, 1)
        self.margin = torch.zeros(n, device=device)
        self.calib_r = torch.eye(3, device=device).repeat(n, 2, 1, 1)
        self.calib_t = torch.zeros(n, 2, 3, device=device)
        self.centre = torch.zeros(n, 3, device=device)
        self.depth_std = torch.zeros(n, device=device)
        self.drop_p = torch.zeros(n, device=device)
        self.tip_r = torch.zeros(n, device=device)
        self.g3_xy = torch.zeros(n, 2, device=device)
        self.g3_yaw = torch.zeros(n, device=device)
        self.ref_pts = torch.zeros(n, k, 3, device=device)
        self.ref_mask = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.ref_moving = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.ref_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_visible_frac = torch.ones(n, device=device)

    # ------------------------------------------------------------------ per-episode draws

    def reset(self, ids: torch.Tensor, centre: torch.Tensor) -> None:
        """New episode for envs ``ids``; ``centre`` ``(len(ids), 3)`` the sheet's centre now."""
        c, dev, m = self.cfg, self.device, int(ids.numel())
        if m == 0:
            return
        noise = torch.as_tensor(c.cam_pos_noise, dtype=torch.float32, device=dev)
        self.cams[ids] = self.cams0 + (2.0 * torch.rand(m, 2, 3, device=dev) - 1.0) * noise
        self.margin[ids] = _uniform(c.margin_range, (m,), dev)
        ang = (2.0 * torch.rand(m, 2, 3, device=dev) - 1.0) * math.radians(float(c.calib_rot_deg))
        self.calib_r[ids] = small_rotation(ang.view(-1, 3)).view(m, 2, 3, 3)
        xyz = torch.as_tensor(c.calib_xyz, dtype=torch.float32, device=dev)
        self.calib_t[ids] = (2.0 * torch.rand(m, 2, 3, device=dev) - 1.0) * xyz
        self.centre[ids] = centre
        self.depth_std[ids] = _uniform(c.depth_noise_std_range, (m,), dev)
        self.drop_p[ids] = _uniform(c.drop_prob_range, (m,), dev)
        on = torch.rand(m, device=dev) < float(c.tip_drop_prob)
        self.tip_r[ids] = torch.where(on, _uniform(c.tip_drop_radius_range, (m,), dev), torch.zeros(m, device=dev))
        self.g3_xy[ids] = (2.0 * torch.rand(m, 2, device=dev) - 1.0) * float(c.g3_bias_xy)
        self.g3_yaw[ids] = (2.0 * torch.rand(m, device=dev) - 1.0) * math.radians(float(c.g3_bias_yaw_deg))
        self.ref_valid[ids] = False
        self.ref_mask[ids] = False

    # ------------------------------------------------------------------ cloth samples

    def sample(self, particles: torch.Tensor):
        """``(pts (N, K, 3), moving (N, K))``: K uniform samples on each env's sheet."""
        n, r, k = particles.shape[0], self.r, int(self.cfg.n_cloth)
        uv = torch.rand(n, k, 2, device=particles.device) * (r - 1)
        i0 = uv.floor().clamp(max=r - 2).long()
        f = uv - i0
        base = i0[..., 0] * r + i0[..., 1]
        idx = torch.stack((base, base + r, base + 1, base + r + 1), dim=-1)                     # (N, K, 4)
        w = torch.stack(
            ((1 - f[..., 0]) * (1 - f[..., 1]), f[..., 0] * (1 - f[..., 1]), (1 - f[..., 0]) * f[..., 1], f[..., 0] * f[..., 1]),
            dim=-1,
        )
        corners = torch.gather(particles, 1, idx.view(n, -1, 1).expand(-1, -1, 3)).view(n, k, 4, 3)
        return (corners * w.unsqueeze(-1)).sum(dim=2), uv[..., 0] > self.mid

    def _visible(self, sl: slice, pts, moving, particles, caps_a, caps_b) -> torch.Tensor:
        """``(n, 2, K)``: camera c has a free line of sight to the point."""
        c = self.cfg
        n, k = pts.shape[0], pts.shape[1]
        cams = self.cams[sl]
        hidden = torch.zeros(n, 2, k, dtype=torch.bool, device=pts.device)
        if caps_a is not None and self.cap_r.numel():
            d = pts.unsqueeze(1) - cams.unsqueeze(2)                                           # (n, 2, K, 3)
            dist = torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
            end = pts.unsqueeze(1) - d / dist * float(c.end_pullback)
            reach = (self.cap_r.view(1, 1, 1, -1) + self.margin[sl].view(n, 1, 1, 1)) ** 2
            hit = segment_distance_sq(
                cams.view(n, 2, 1, 1, 3), end.unsqueeze(3), caps_a.view(n, 1, 1, -1, 3), caps_b.view(n, 1, 1, -1, 3)
            ) < reach
            hidden |= hit.any(dim=-1)
        origin, end = cams.view(n, 2, 1, 1, 3), pts.view(n, 1, k, 1, 3)

        def crossed(tri: torch.Tensor, gap: float) -> torch.Tensor:
            v = particles[:, tri]                                                              # (n, T, 3, 3)
            v0, v1, v2 = (v[:, :, i].view(n, 1, 1, -1, 3) for i in range(3))
            return ray_hits_triangles(origin, end, v0, v1, v2, gap).any(dim=-1)                # (n, 2, K)

        by_st = crossed(self.tri_st, float(c.self_gap))
        by_mv = crossed(self.tri_mv, -float(c.overlap_tol))
        hidden |= torch.where(moving.unsqueeze(1), by_st, by_mv)
        return ~hidden

    def cloth_cloud(self, particles, caps_a=None, caps_b=None, tips=None):
        """``(pts, mask, moving)``: the observed cloth points ``(N, K, 3)``, which slots hold a
        point, and each point's half. Applies visibility and every noise source."""
        c, n, dev = self.cfg, particles.shape[0], particles.device
        pts, moving = self.sample(particles)
        if c.visibility == "none":
            vis = torch.ones(n, 2, pts.shape[1], dtype=torch.bool, device=dev)
        else:
            out = []
            for lo in range(0, n, int(c.chunk)):
                sl = slice(lo, min(lo + int(c.chunk), n))
                out.append(self._visible(
                    sl, pts[sl], moving[sl], particles[sl],
                    None if caps_a is None else caps_a[sl], None if caps_b is None else caps_b[sl],
                ))
            vis = torch.cat(out, dim=0)
        mask = vis.any(dim=1)
        self.last_visible_frac = mask.float().mean(dim=1)
        # Which camera's cloud the point comes from: the one that sees it (a coin flip if both do).
        use_b = torch.where(vis[:, 0] & vis[:, 1], torch.rand(mask.shape, device=dev) < 0.5, vis[:, 1])
        cam = use_b.long()                                                                     # (N, K)
        self.last_cloth_cam = cam
        env = torch.arange(n, device=dev).unsqueeze(1).expand_as(cam)
        rot, shift, origin = self.calib_r[env, cam], self.calib_t[env, cam], self.cams[env, cam]
        ray = torch.nn.functional.normalize(pts - origin, dim=-1)
        pts = pts + ray * torch.randn(mask.shape, device=dev).unsqueeze(-1) * self.depth_std.view(n, 1, 1)
        centre = self.centre.unsqueeze(1)
        pts = torch.einsum("nkij,nkj->nki", rot, pts - centre) + centre + shift
        mask = mask & (torch.rand(mask.shape, device=dev) >= self.drop_p.unsqueeze(1))
        if tips is not None:
            near = torch.cdist(pts, tips).amin(dim=-1) < self.tip_r.unsqueeze(1)
            mask = mask & ~near
        return pts, mask, moving

    # ------------------------------------------------------------------ the observation

    def observe(self, step, particles, palm, goal_kp=None, st_centroid=None, caps_a=None, caps_b=None, tips=None):
        """``(points (N, M, 3 + L), mask (N, M))``.

        ``step`` ``(N,)`` episode step of this observation; ``particles`` ``(N, R*R, 3)``; ``palm``
        ``(N, 3)``; ``goal_kp`` ``(N, 4, 3)`` true fold targets and ``st_centroid`` ``(N, 3)`` the
        stationary half's centroid (g0 / g3); ``caps_a`` / ``caps_b`` ``(N, C, 3)`` capsule ends;
        ``tips`` ``(N, F, 3)`` fingertips.
        """
        c, n, dev = self.cfg, particles.shape[0], particles.device
        pts, mask, moving = self.cloth_cloud(particles, caps_a, caps_b, tips)
        lab = torch.zeros(n, pts.shape[1], dtype=torch.long, device=dev)
        if c.goal_mode == "gl":
            lab = torch.where(moving, torch.ones_like(lab), torch.full_like(lab, 2))
        groups = [(pts, mask, lab)]
        no_cam = lambda g: torch.full(g.shape[:2], -1, dtype=torch.long, device=dev)  # noqa: E731
        src = [self.last_cloth_cam]              # which camera each point came from (-1: none); viewers

        if self.n_hand:
            t = torch.linspace(0.0, 1.0, int(c.hand_pts_per_capsule), device=dev).view(1, 1, -1, 1)
            hand = (caps_a.unsqueeze(2) * (1.0 - t) + caps_b.unsqueeze(2) * t).reshape(n, -1, 3)
            groups.append((hand, torch.ones(n, hand.shape[1], dtype=torch.bool, device=dev),
                           torch.full((n, hand.shape[1]), 3, dtype=torch.long, device=dev)))
            src.append(no_cam(hand))

        if c.goal_mode in ("g0", "g3"):
            goal = goal_kp
            if c.goal_mode == "g3":
                yaw = self.g3_yaw + (2.0 * torch.rand(n, device=dev) - 1.0) * math.radians(float(c.g3_noise_yaw_deg))
                xy = self.g3_xy + (2.0 * torch.rand(n, 2, device=dev) - 1.0) * float(c.g3_noise_xy)
                pivot = st_centroid.unsqueeze(1)
                goal = torch.einsum("nij,nkj->nki", _rot_z(yaw), goal_kp - pivot) + pivot
                goal = goal + torch.cat((xy, torch.zeros(n, 1, device=dev)), dim=-1).unsqueeze(1)
            groups.append((goal, torch.ones(n, 4, dtype=torch.bool, device=dev),
                           torch.full((n, 4), 6, dtype=torch.long, device=dev)))
            src.append(no_cam(goal))
        elif c.goal_mode == "g1":
            take = ~self.ref_valid & (step >= int(c.ref_step))
            if bool(take.any()):
                self.ref_pts[take], self.ref_mask[take], self.ref_moving[take] = pts[take], mask[take], moving[take]
                self.ref_valid[take] = True
            ref_lab = torch.where(self.ref_moving, torch.full_like(lab, 4), torch.full_like(lab, 5))
            groups.append((self.ref_pts, self.ref_mask.clone(), ref_lab))
            src.append(no_cam(self.ref_pts))

        if self.n_hand_cam:
            cam = torch.randint(0, 2, hand.shape[:2], device=dev)
            env = torch.arange(n, device=dev).unsqueeze(1).expand_as(cam)
            ray = torch.nn.functional.normalize(hand - self.cams[env, cam], dim=-1)
            seen = hand + ray * torch.randn(cam.shape, device=dev).unsqueeze(-1) * self.depth_std.view(n, 1, 1)
            centre = self.centre.unsqueeze(1)
            seen = torch.einsum("nkij,nkj->nki", self.calib_r[env, cam], seen - centre) + centre + self.calib_t[env, cam]
            groups.append((seen, torch.rand(cam.shape, device=dev) >= self.drop_p.unsqueeze(1),
                           torch.full(cam.shape, 6, dtype=torch.long, device=dev)))
            src.append(cam)

        xyz = torch.cat([g[0] for g in groups], dim=1)
        mask = torch.cat([g[1] for g in groups], dim=1)
        label = torch.cat([g[2] for g in groups], dim=1)
        # For viewers (eval/live_viser.py): the cloud in the env frame, before the palm offset.
        self.last_xyz, self.last_mask, self.last_label, self.last_cam = xyz, mask, label, torch.cat(src, dim=1)
        xyz = (xyz - palm.unsqueeze(1)) * float(c.scale)
        onehot = torch.nn.functional.one_hot(label, len(LABELS)).to(xyz.dtype)
        points = torch.cat((xyz, onehot), dim=-1) * mask.unsqueeze(-1)
        return points, mask
