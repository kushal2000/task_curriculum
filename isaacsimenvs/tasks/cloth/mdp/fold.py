"""The fold's geometry: targets, the stationary half's frame, fold error, footprint, the goal pose,
and the moving half seen as a rigid object. Built once from the sheet's layout; every query reads
only the sheet's points.

**The targets are defined relative to the sheet, not to the world.** A fixed world target cannot
tell a fold from a slide: with targets pinned at the rest sheet's mirrored position, pushing the
whole sheet half its width puts the moving half exactly where the targets are (the pretrained tool
policy, which never saw cloth, scored 19 "folds" in 31 envs that way). So each keypoint's target is
its rest position reflected across the crease and lifted onto the stationary half -- in the sheet's
REST frame -- and carried onto the stationary half's CURRENT rigid frame, fitted from its points.
Translating or rotating the sheet moves the target with it; only folding the flap can reduce the
error.

**The lift is the ply gap, not the particle radius.** Two plies of VBD cloth settle one
``self_contact_radius`` apart (measured 2.5 mm at 2 mm); the 8 mm particle radius is the
cloth-to-RIGID offset and has no say in a fold. Using ``2 * particle_radius`` put ~14 mm of error on
every keypoint of a perfect fold. The rigid pool's lift is its plates' ply gap.

With per-env sheet sizes (the rigid pool) every rest quantity is scaled in-plane per env; with one
size (VBD) they are shared. The two keep their own arithmetic, so neither changes numerically.
"""

from __future__ import annotations

import torch

from .geometry.grid import half_indices, half_mesh
from .state import FoldView, SheetLayout

__all__ = ["FoldGeometry", "kabsch_quat_wxyz"]


def _mat_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """``(N, 3, 3)`` rotation matrices to ``(N, 4)`` wxyz quaternions with w >= 0 (Shepperd, a
    per-row branch; the goal marker pose). Unlike :func:`_rotation_to_wxyz` the sign is fixed."""
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


def _rotation_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` to wxyz quaternions, via the trace formulation.

    Uses the largest-diagonal branch rather than the naive trace form, which loses precision (and
    can divide by ~0) when the rotation approaches 180 degrees -- exactly the case a completed fold
    produces.
    """
    n = r.shape[0]
    q = torch.zeros((n, 4), device=r.device, dtype=r.dtype)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]

    big = trace > 0
    if bool(big.any()):
        sq = torch.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * sq
        q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / sq
        q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / sq
        q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / sq
    rest = ~big
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = rest & (r[:, i, i] >= r[:, j, j]) & (r[:, i, i] >= r[:, k, k])
        if not bool(sel.any()):
            continue
        sq = torch.sqrt(1.0 + r[sel, i, i] - r[sel, j, j] - r[sel, k, k]) * 2.0
        q[sel, 0] = (r[sel, k, j] - r[sel, j, k]) / sq
        q[sel, 1 + i] = 0.25 * sq
        q[sel, 1 + j] = (r[sel, j, i] + r[sel, i, j]) / sq
        q[sel, 1 + k] = (r[sel, k, i] + r[sel, i, k]) / sq
    # eps guards a zero-norm quaternion, which a degenerate fit can produce.
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)


def kabsch_quat_wxyz(rest: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Least-squares rotation taking ``rest`` (K, 3), or per env (N, K, 3), to ``cur`` (N, K, 3),
    both centred here, as wxyz.

    H = P^T Q, then R = V diag(1, 1, det(VU^T)) U^T. The determinant term forbids a reflection: a
    badly deformed patch could otherwise fit a mirror, and a mirrored frame is worse than a wrong
    one. A crumpled patch can make H rank deficient (the SVD then returns an arbitrary rotation and
    the quaternion can normalise to NaN), so the diagonal is ridged and a non-finite result falls
    back to identity: wrong but bounded, where a NaN would poison the whole observation.
    """
    q = cur - cur.mean(dim=1, keepdim=True)
    if rest.dim() == 2:
        p = rest - rest.mean(dim=0, keepdim=True)
        h = torch.einsum("ki,nkj->nij", p, q)
    else:
        p = rest - rest.mean(dim=1, keepdim=True)
        h = torch.einsum("nki,nkj->nij", p, q)
    h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    det = torch.linalg.det(v @ u.transpose(-2, -1))
    d = torch.ones((cur.shape[0], 3), device=cur.device)
    d[:, 2] = det
    r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
    quat = _rotation_to_wxyz(r)
    bad = ~torch.isfinite(quat).all(dim=-1)
    if bool(bad.any()):
        quat = quat.clone()
        quat[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype)
    return quat


class FoldGeometry:
    """Rest-frame fold quantities, and their live counterparts from the sheet's points."""

    def __init__(self, layout: SheetLayout, *, num_envs: int, device, observed_goal_quat) -> None:
        self.layout = layout
        self.num_envs, self.device = num_envs, device
        #: Rotation matrices -> the goal quaternion the policy observes (a backend convention).
        self._observed_goal_quat = observed_goal_quat
        self.axis = 0 if layout.fold_axis == "x" else 1
        self.per_env = layout.xy_scale is not None
        #: One-fold goal tasks (``goal_tasks.GoalTasks``), or ``None``. Active once the backend has
        #: drawn each env's task; until then the fold is the stage-1 one.
        self.goal_tasks = layout.goal_tasks

        rest = torch.tensor(layout.rest_local, device=device, dtype=torch.float32)
        self.rest_local = rest
        self.keypoint_idx = torch.tensor(layout.keypoint_idx, device=device, dtype=torch.long)
        corners = rest[self.keypoint_idx]
        #: The nominal corners, which the moving half's rotation is fitted against.
        self.corner_rest = corners
        corner_rest_offsets = corners - corners.mean(dim=0, keepdim=True)
        ax = self.axis
        crease_local = rest[:, ax].max() - layout.size / 2.0  # the crease at the sheet's midline (= 0)
        self.moving_idx = torch.tensor(
            sorted(set(half_indices(layout.resolution, layout.fold_axis, positive=True))), device=device, dtype=torch.long
        )
        self.stationary_idx = torch.tensor(
            half_indices(layout.resolution, layout.fold_axis, positive=False), device=device, dtype=torch.long
        )

        # Where each point belongs after the fold, in the sheet's REST frame: reflected across the
        # crease, lifted by the ply gap onto the stationary half.
        def folded_rest(sel: torch.Tensor) -> torch.Tensor:
            f = rest[sel].clone()
            f[:, ax] = 2.0 * crease_local - f[:, ax]
            f[:, 2] = f[:, 2] + layout.fold_lift
            return f

        mv_folded_rest = folded_rest(self.moving_idx)
        kp_folded_rest = folded_rest(self.keypoint_idx)
        st_rest = rest[self.stationary_idx]
        st_rest_centroid = st_rest.mean(dim=0)
        st_rest_centred = st_rest - st_rest_centroid

        if self.per_env:
            s = layout.xy_scale
            s3 = torch.stack((s, s, torch.ones_like(s)), dim=-1).unsqueeze(1)  # (N, 1, 3)
            self.scale3 = s3
            corner_rest_offsets = corner_rest_offsets.unsqueeze(0) * s3
            mv_folded_rest = mv_folded_rest.unsqueeze(0) * s3
            kp_folded_rest = kp_folded_rest.unsqueeze(0) * s3
            st_rest_centroid = st_rest_centroid.unsqueeze(0) * s3[:, 0]
            st_rest_centred = st_rest_centred.unsqueeze(0) * s3
        self.mv_folded_rest = mv_folded_rest
        self.kp_folded_rest = kp_folded_rest
        self.st_rest_centroid = st_rest_centroid
        self.st_rest_centred = st_rest_centred
        #: The observed keypoints' offsets from the moving half's centre: its REST corners. With the
        #: fitted rotation, ``centre + R * offsets`` tracks the real corners and ``goal_centre +
        #: R_goal * offsets`` is the folded configuration.
        self._keypoint_offsets = torch.zeros(num_envs, len(layout.keypoint_idx), 3, device=device)
        if self.per_env:
            self._keypoint_offsets[:] = corner_rest_offsets
        else:
            self._keypoint_offsets[:] = corner_rest_offsets.unsqueeze(0).expand(num_envs, -1, -1)

    @property
    def keypoint_offsets(self) -> torch.Tensor:
        """``(N, K, 3)`` the observed keypoints' offsets from their centre, at rest and each env's
        size. With ``any_fold`` goal tasks: the tracked points of each env's task in its start state."""
        g = self._active_goal_tasks()
        if g is None or not g.any_fold:
            return self._keypoint_offsets
        rest = g.keypoint_rest()
        return (rest - rest.mean(dim=1, keepdim=True)) * self._scale3()

    def _active_goal_tasks(self):
        g = self.goal_tasks
        return g if g is not None and g.task is not None else None

    def _scale3(self) -> torch.Tensor:
        """``(N, 1, 3)``: each env's in-plane size factor on x and y, 1 on z."""
        if self.per_env:
            return self.scale3
        return torch.ones((self.num_envs, 1, 3), device=self.device)

    # ------------------------------------------------------------------ the stationary half's frame

    def stationary_frame(self, points_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(R (N, 3, 3), centroid (N, 3))``: the stationary half from its rest pose to now.

        Ridged, determinant-corrected, identity on failure: a crumpled half can drive the fit rank
        deficient, and a NaN or mirrored frame would silently corrupt every target.
        """
        cur = points_w[:, self.stationary_idx, :]
        centroid = cur.mean(dim=1)
        q = cur - centroid.unsqueeze(1)
        if self.per_env:
            h = torch.einsum("nki,nkj->nij", self.st_rest_centred, q)
        else:
            h = torch.einsum("ki,nkj->nij", self.st_rest_centred, q)
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        d = torch.ones((cur.shape[0], 3), device=cur.device, dtype=cur.dtype)
        d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
        bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
        if bool(bad.any()):
            r = r.clone()
            r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
        return r, centroid

    def folded_w(self, frame_r: torch.Tensor, frame_c: torch.Tensor, folded_rest: torch.Tensor) -> torch.Tensor:
        """Rest-frame folded positions carried onto the sheet's live pose. ``(N, M, 3)``."""
        if self.per_env:
            local = folded_rest - self.st_rest_centroid.unsqueeze(1)
            return torch.einsum("nij,nmj->nmi", frame_r, local) + frame_c.unsqueeze(1)
        local = folded_rest - self.st_rest_centroid
        return torch.einsum("nij,mj->nmi", frame_r, local) + frame_c.unsqueeze(1)

    # ------------------------------------------------------------------ one state -> the fold

    def derive(self, points_w: torch.Tensor, points_vel_w: torch.Tensor) -> FoldView:
        g = self._active_goal_tasks()
        view = self._derive_any_fold(g, points_w, points_vel_w) if g is not None and g.any_fold else (
            self._derive_half_fold(points_w, points_vel_w)
        )
        if g is not None:
            view.designated_error = view.fold_error
            view.lookalike_error = (
                g.lookalike_error(self._scale3(), points_w) if g.accept_lookalike
                else torch.full_like(view.fold_error, float("inf"))
            )
            view.fold_error = torch.minimum(view.designated_error, view.lookalike_error)
        return view

    def _derive_half_fold(self, points_w: torch.Tensor, points_vel_w: torch.Tensor) -> FoldView:
        """The stage-1 fold: the moving half's corners onto the stationary half."""
        r, c = self.stationary_frame(points_w)
        keypoints = points_w[:, self.keypoint_idx, :]
        targets = self.folded_w(r, c, self.kp_folded_rest)
        # The max, not the mean: a fold with one corner still flat on the table is not a fold.
        fold_error = (keypoints - targets).norm(dim=-1).amax(dim=-1)
        return FoldView(
            points_w=points_w,
            points_vel_w=points_vel_w,
            frame_r=r,
            frame_c=c,
            keypoints_w=keypoints,
            targets_w=targets,
            fold_error=fold_error,
            footprint=self.footprint(points_w, r),
            object_pos_w=keypoints.mean(dim=1),
            object_quat_w=kabsch_quat_wxyz(self.corner_rest, keypoints),
            object_lin_vel_w=points_vel_w[:, self.keypoint_idx, :].mean(dim=1),
            object_ang_vel_w=torch.zeros((self.num_envs, 3), device=self.device),
        )

    def _derive_any_fold(self, g, points_w: torch.Tensor, points_vel_w: torch.Tensor) -> FoldView:
        """``any_fold`` goal tasks: the fold each env's task names. The frame is the part that fold
        leaves in place (fitted against the task's start state), the keypoints are on the piece it
        moves, the targets are those points in the goal state carried on that frame, and the error
        is the farthest any moved point is from its goal place."""
        task, s3 = g.task, self._scale3()
        r, c = g.frame(task, s3, points_w)
        idx = g.keypoint_idx().unsqueeze(-1).expand(-1, -1, 3)
        keypoints = torch.gather(points_w, 1, idx)
        goal_points, _ = g.goal_particles(task, s3, points_w)
        return FoldView(
            points_w=points_w,
            points_vel_w=points_vel_w,
            frame_r=r,
            frame_c=c,
            keypoints_w=keypoints,
            targets_w=torch.gather(goal_points, 1, idx),
            fold_error=g.error(task, s3, points_w),
            footprint=self.footprint(points_w, r),
            object_pos_w=keypoints.mean(dim=1),
            object_quat_w=kabsch_quat_wxyz(g.keypoint_rest(), keypoints),
            object_lin_vel_w=torch.gather(points_vel_w, 1, idx).mean(dim=1),
            object_ang_vel_w=torch.zeros((self.num_envs, 3), device=self.device),
        )

    def footprint(self, points_w: torch.Tensor, frame_r: torch.Tensor) -> torch.Tensor:
        """Sheet extent along its OWN fold axis, as a fraction of its unfolded size.

        A fold halves the footprint; a slide or a crumple does not necessarily -- the second,
        independent condition of the fold criterion. Along the sheet's axis (the stationary half's
        fitted rotation applied to the rest axis), not the world's: an unfolded sheet at 45 deg
        spans 1.41 of its size in world x.
        """
        coord = torch.einsum("npi,ni->np", points_w, frame_r[:, :, self.axis])
        size = self.layout.env_size if self.per_env else self.layout.size
        return (coord.amax(dim=1) - coord.amin(dim=1)) / size

    # ------------------------------------------------------------------ the goal

    def goal_rotation(self, frame_r: torch.Tensor) -> torch.Tensor:
        """``(N, 3, 3)``: the moving half's orientation after the fold -- 180 deg about the crease,
        composed with the stationary half's live rotation. With ``any_fold`` goal tasks: the task's
        own fold rotation, composed with the live rotation of the part it leaves in place."""
        g = self._active_goal_tasks()
        if g is not None and g.any_fold:
            return torch.bmm(frame_r, g.rot[g.task])
        flip = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        flip[:, self.axis, self.axis] = -1.0
        flip[:, 2, 2] = -1.0
        return torch.bmm(frame_r, flip)

    def goal_pose(self, view: FoldView) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(goal_pos_w, goal_quat, marker_pose)``: the targets' centre and the folded half's
        orientation. ``goal_quat`` is the policy's convention (the true quaternion's components
        relabelled; every policy was trained on it), ``marker_pose`` the true pose (w, x, y, z)."""
        rot = self.goal_rotation(view.frame_r)
        goal_pos = view.targets_w.mean(dim=1)
        return goal_pos, self._observed_goal_quat(rot), torch.cat([goal_pos, _mat_to_wxyz(rot)], dim=-1)

    def observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        return self._observed_goal_quat(rot)

    # ------------------------------------------------------------------ goal tasks: the goal sheet

    def goal_sheet(self, points_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For a goal point cloud: ``(goal sheet now (N, P, 3) world, goal sheet in its own frame
        (N, P, 3), flat at z = 0)`` of each env's designated goal, at each env's size. The first is
        the goal state carried on the live pose of the part the fold leaves in place."""
        g = self._active_goal_tasks()
        if g is None:
            raise ValueError("a goal point cloud needs fold_pool.goal_task.mode half_fold or any_fold")
        s3 = self._scale3()
        goal, _ = g.goal_particles(g.task, s3, points_w)
        return goal, g.p_goal[g.task] * s3

    def matched_goal_sheet(self, view: FoldView) -> tuple[torch.Tensor, torch.Tensor]:
        """For viewers: ``(grid (N, P, 3) world, lookalike (N,) bool)`` -- the goal state the fold
        criterion matches best now: the nearest look-alike where it is closer than the task's own
        goal (``accept_lookalike``), else the task's own goal."""
        g = self._active_goal_tasks()
        s3 = self._scale3()
        own, _ = g.goal_particles(g.task, s3, view.points_w)
        if not g.accept_lookalike:
            return own, torch.zeros(own.shape[0], dtype=torch.bool, device=own.device)
        member, _ = g.nearest_alternative(g.task, g.read_hinge_abs())
        other, _ = g.goal_particles(g.task, s3, view.points_w, member)
        use = torch.isfinite(view.lookalike_error) & (view.lookalike_error < view.designated_error)
        return torch.where(use.view(-1, 1, 1), other, own), use

    def goal_travel_direction(self, points_w: torch.Tensor) -> torch.Tensor:
        """``(N,)`` env-frame direction [rad] the designated fold carries its piece."""
        g = self._active_goal_tasks()
        r, _ = g.frame(g.task, self._scale3(), points_w)
        return torch.atan2(r[:, 1, 0], r[:, 0, 0]) + g.direction[g.task]

    # ------------------------------------------------------------------ the ghost of the folded half

    def folded_half_w(self, view: FoldView) -> torch.Tensor:
        """``(N, M, 3)``: where every point of the moving half belongs when folded -- the goal drawn
        as a copy of the sheet, from the same construction as the scored targets."""
        return self.folded_w(view.frame_r, view.frame_c, self.mv_folded_rest)

    def folded_half_topology(self) -> list[int]:
        """Triangle indices for :meth:`folded_half_w`. Winding reversed: a fold is a reflection, and
        the rest winding would leave every folded triangle facing down (rendered as a black back face)."""
        lay = self.layout
        _, idx = half_mesh(float(lay.size), int(lay.resolution), lay.fold_axis)
        return [v for t in range(0, len(idx), 3) for v in reversed(idx[t : t + 3])]
