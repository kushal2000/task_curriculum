"""ClothTaskMixin -- the fold itself, whichever backend simulates the sheet.

The fold targets (carried on the stationary half's live frame), the fold error, the fold criterion
(keypoints on target AND the footprint halved), and the goal marker the policy observes: ONE copy
for both backends (``backends/vbd`` and ``backends/rigid``), so a cross-simulator difference is the
physics and never the task. The reward and terminations built on it are ``rewards.py`` and
``terminations.py``.

Reads what a backend's ``_install_cloth`` sets: ``self._cloth`` (a mesh with ``data.nodal_pos_w``),
``self.object`` (a ``backends.adapter.ClothAsRigidObject``), ``self._cloth_rest_local`` and
``self._cloth_kp_idx``.

Kit-free at import (torch and ``isaaclab.utils.math`` only), so both stacks can import it.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.mdp.geometry.grid import (
    folded_targets,
    half_indices,
    half_mesh,
)

__all__ = ["ClothTaskMixin"]


class ClothTaskMixin:
    """The fold, independent of the simulator."""

    # ------------------------------------------------------------------ construction

    def _write_object_scale(self) -> None:
        """Tell the task the sheet's real dimensions.

        ``_object_scale_per_env`` is filled by ``setup_scene`` from the *procedural rigid tool*
        pool and nothing updates it when the manipuland is replaced. Left alone, the policy is
        handed a hammer's bounding box while facing a cloth, and the keypoint offsets that size the
        reward and the success test are derived from it. The cable env had exactly this bug.

        Convention: bounding box normalised by ``reward.object_base_size``.
        """
        c = self.cfg.cloth
        base = float(self.cfg.reward.object_base_size)
        bbox = torch.tensor(
            [c.size, c.size, c.thickness], device=self.device, dtype=torch.float32
        )
        self._object_scale_per_env[:] = bbox / base
        print(
            f"[cloth] object_scale = {[round(float(v), 2) for v in bbox / base]} "
            f"(bbox {[round(float(v), 3) for v in bbox]} m / base {base})",
            flush=True,
        )

    # ------------------------------------------------------------------ the fold

    def _init_fold_targets(self) -> None:
        """World-frame targets for the tracked keypoints, per env."""
        c = self.cfg.cloth
        # PLY SEPARATION IS SET BY `self_contact_radius`, NOT BY `particle_radius`.
        # These are two independent rest offsets that never meet: `particle_radius` is the
        # cloth<->RIGID offset (three body-contact kernels), `self_contact_radius` is the
        # cloth<->CLOTH one (`evaluate_self_contact_force_norm`). A fold is cloth on cloth, so the
        # rigid offset has no say in where the flap comes to rest.
        #
        # This used `2 * particle_radius` = 16 mm while the solver settles the plies at 2.5 mm,
        # so every keypoint on the moving half carried ~14 mm of error at a PERFECT fold. Measured
        # with `scripts/analysis/cloth_fold_probe.py`, teleport-and-release, 4 envs, stable 180
        # steps: settled fold_error 0.0160 -> 0.0020 from this line alone, with the 8 mm grip
        # radius untouched. See docs/reports/1_vbd_cloth/cloth_sim1_physics_and_fold_criterion.md.
        lift = c.self_contact_radius  # where cloth-on-cloth contact actually settles
        local = folded_targets(self._cloth_rest_local, self._cloth_kp_idx, c.fold_axis, lift)

        t = torch.tensor(local, device=self.device, dtype=torch.float32)  # (K, 3)
        # Rest geometry only. `fold_targets_w` combines these with the sheet's CURRENT crease, so
        # the criterion is translation-invariant.
        self._kp_idx_t = torch.tensor(self._cloth_kp_idx, device=self.device, dtype=torch.long)
        rest = torch.tensor(self._cloth_rest_local, device=self.device, dtype=torch.float32)
        corners = rest[self._kp_idx_t]
        self._corner_rest_offsets = corners - corners.mean(dim=0, keepdim=True)
        ax = 0 if c.fold_axis == "x" else 1
        kp_rest = rest[self._kp_idx_t]                       # (K, 3) local
        crease_local = rest[:, ax].max() - c.size / 2.0      # crease at the sheet's midline (=0)
        self._kp_rest_offset = kp_rest[:, ax] - crease_local  # distance past the crease
        self._kp_rest_lateral = kp_rest[:, 1 - ax]

        # Same two quantities for EVERY particle of the moving half, so the folded configuration
        # can be drawn as a ghost copy of the sheet rather than approximated by a rigid marker.
        moving = sorted(set(half_indices(c.resolution, c.fold_axis, positive=True)))
        self._moving_idx = torch.tensor(moving, device=self.device, dtype=torch.long)
        mv_rest = rest[self._moving_idx]
        self._mv_rest_offset = mv_rest[:, ax] - crease_local
        self._mv_rest_lateral = mv_rest[:, 1 - ax]

        # Where each particle belongs after the fold, expressed once in the sheet's REST frame:
        # reflect across the crease, then lift by one SELF-CONTACT RADIUS onto the stationary
        # half -- that is the distance cloth-on-cloth contact actually settles at. At
        # runtime this is carried onto the sheet's live pose by the stationary half's own rigid
        # frame, which is what makes the target follow the cloth when it is rotated as well as
        # when it is translated.
        def _folded_rest(sel: torch.Tensor) -> torch.Tensor:
            f = rest[sel].clone()
            f[:, ax] = 2.0 * crease_local - f[:, ax]
            f[:, 2] = f[:, 2] + c.self_contact_radius
            return f

        self._mv_folded_rest = _folded_rest(self._moving_idx)
        self._kp_folded_rest = _folded_rest(self._kp_idx_t)
        st_rest = rest[torch.tensor(
            half_indices(c.resolution, c.fold_axis, positive=False),
            device=self.device, dtype=torch.long,
        )]
        self._st_rest_centroid = st_rest.mean(dim=0)
        self._st_rest_centred = st_rest - self._st_rest_centroid
        self._fold_targets_xy = t.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)
        self._stationary_idx = torch.tensor(
            half_indices(c.resolution, c.fold_axis, positive=False),
            device=self.device, dtype=torch.long,
        )
        self._kp_idx_t = torch.tensor(self._cloth_kp_idx, device=self.device, dtype=torch.long)

    def fold_targets_w(self) -> torch.Tensor:
        """``(num_envs, K, 3)`` fold targets, defined RELATIVE TO THE SHEET, not to the world.

        **A fixed world target cannot tell a fold from a slide.** With targets pinned at the rest
        sheet's mirrored position, simply pushing the whole sheet half its width in ``-x`` puts the
        moving half exactly where the targets are, and a flat sheet's height matches too. Measured:
        the pretrained tool policy -- which has never seen cloth -- scored 19 "folds" in 31 envs
        that way, with zero falls. It was sliding the sheet, not folding it.

        So the fold is defined in the sheet's REST frame -- each keypoint reflected across the
        crease and lifted one self-contact radius onto the stationary half -- and then carried onto the
        stationary half's CURRENT rigid frame, fitted from its particles.

        That frame is the whole point. Translating or rotating the cloth moves the target with it,
        so neither can reduce the error; only actually folding the flap can. An earlier version got
        this half right -- the fold-axis coordinate tracked the live crease, but the lateral one
        stayed pinned to the spawn position, so a rotated sheet was scored against an axis-aligned
        target and a genuine fold could read as a failure.
        """

        # Carried onto the stationary half's CURRENT rigid frame, so the target follows the sheet
        # through rotation as well as translation. The rest-frame definition already encodes both
        # the reflection across the crease and the lift onto the stationary half. The lift is
        # `self_contact_radius`, MEASURED: two plies settle 2.5 mm apart at scr = 2 mm, while the
        # sheet floats 8.0 mm above the TABLE at particle_radius = 8 mm. Two different offsets for
        # two different contact pairs; only the cloth-cloth one governs a fold.
        return self._folded_w(self._kp_folded_rest)

    def _stationary_frame(self, parts: torch.Tensor):
        """Rigid transform taking the stationary half's REST pose to its CURRENT pose.

        Returns ``(R, centroid)`` with ``R`` of shape ``(N, 3, 3)``.

        This is what makes the fold target follow the sheet through a *rotation*, not just a
        translation. The previous construction set the fold-axis coordinate from the live crease but
        pinned the lateral coordinate to ``rest_lateral + env_origin`` -- the spawn position -- so
        once the hand turned the cloth the target stayed axis-aligned while the sheet rotated under
        it. On screen the goal appeared beside the stationary half and turned ninety degrees away
        from it; in the metric it meant a genuinely folded but rotated sheet scored as unfolded.
        """
        cur = parts[:, self._stationary_idx, :]
        centroid = cur.mean(dim=1)
        q = cur - centroid.unsqueeze(1)
        h = torch.einsum("ki,nkj->nij", self._st_rest_centred, q)
        # Ridge, determinant correction and an identity fallback, for the same reasons as the
        # corner fit in the adapter: a crumpled half can drive this rank deficient, and a NaN or a
        # mirrored frame here would silently corrupt every target.
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

    def _folded_w(self, folded_rest: torch.Tensor) -> torch.Tensor:
        """Carry rest-frame folded positions onto the sheet's live pose. ``(N, M, 3)``."""
        r, centroid = self._stationary_frame(self._particles_w())
        local = folded_rest - self._st_rest_centroid
        return torch.einsum("nij,mj->nmi", r, local) + centroid.unsqueeze(1)

    def _crease_w(self, parts: torch.Tensor, ax: int) -> torch.Tensor:
        """``(num_envs,)`` current position of the hinge line along the fold axis.

        **Midway between the two halves' inboard edges, not the stationary half's edge.**
        ``half_indices`` deliberately excludes the hinge row from both halves, so
        ``stationary[..., ax].amax()`` is one grid spacing *inboard* of the actual hinge. Using it
        as the crease reflected every target one spacing too far: measured at ``size=0.10``,
        ``resolution=9``, targets spanned x in [-0.0625, -0.025] where a true reflection about the
        hinge gives [-0.05, -0.0125] -- a systematic 12.5 mm error, a third of the 40 mm tolerance,
        placing the goal beyond the stationary half's outer edge instead of on top of it.

        The midpoint is used rather than the hinge row itself so this stays correct for an even
        ``resolution``, where there is no hinge row at all.
        """
        st = parts[:, self._stationary_idx, ax].amax(dim=1)
        mv = parts[:, self._moving_idx, ax].amin(dim=1)
        return 0.5 * (st + mv)

    def folded_half_w(self) -> torch.Tensor:
        """``(num_envs, M, 3)`` -- where EVERY particle of the moving half belongs when folded.

        The same construction as :meth:`fold_targets_w`, over the whole half instead of the four
        scored keypoints. This exists so the goal can be drawn as a ghost copy of the sheet: a
        folded cloth is a *surface*, and no rigid marker can depict one -- a hammer, a box and a
        flat mesh all read as some kind of slab, because ``goal_viz`` is a rigid body with a single
        pose and a single shape. The renderer can log an arbitrary mesh, so it does.

        Shares :meth:`_folded_w` with the scored targets, so the drawing cannot drift from the
        criterion -- the ghost is the goal, over every particle instead of over four keypoints.
        """
        out = self._folded_w(self._mv_folded_rest)
        return out

    def folded_half_topology(self) -> list[int]:
        """Triangle indices for :meth:`folded_half_w`, in that array's own vertex ordering.

        **Winding is reversed**, because a fold is a reflection and a reflection inverts triangle
        orientation. Keeping the rest sheet's winding leaves every folded triangle facing downward,
        so the goal replica renders as an unlit back face -- a black shape with a lit rim, which is
        exactly how it first appeared.
        """
        _, idx = half_mesh(
            float(self.cfg.cloth.size), int(self.cfg.cloth.resolution), self.cfg.cloth.fold_axis
        )
        return [v for t in range(0, len(idx), 3) for v in reversed(idx[t : t + 3])]

    def _particles_w(self) -> torch.Tensor:
        return self.object._particles()

    def cloth_keypoints_w(self) -> torch.Tensor:
        """Tracked keypoint positions, ``(num_envs, K, 3)`` in world coordinates."""
        nodal = self._cloth.data.nodal_pos_w
        nodal = nodal.torch if hasattr(nodal, "torch") else nodal
        return nodal.view(self.num_envs, -1, 3)[:, self._kp_idx_t, :]

    def _tracked_points_w(self) -> torch.Tensor:
        """The points ``point_dr_mode`` perturbs (``observations.py``): the moving-half corners themselves,
        in keypoint order -- what a point tracker sees -- rather than the rigid reconstruction."""
        return self.cloth_keypoints_w()

    def fold_error(self) -> torch.Tensor:
        """Per-env max keypoint distance from its fold target, ``(num_envs,)``.

        The *max* rather than the mean: a fold with one corner still flat on the table is not a
        fold, and averaging would call it two-thirds done.
        """
        return (self.cloth_keypoints_w() - self.fold_targets_w()).norm(dim=-1).amax(dim=-1)

    def _sync_observed_keypoints(self) -> None:
        """Pin the keypoint offsets to the moving half's REST corners.

        With a fitted rotation available, the task's own machinery does the right thing:
        ``obj_kp = centroid + R * rest_offsets`` tracks the real corners, and
        ``goal_kp = goal_centroid + R_goal * rest_offsets`` is the *folded* configuration.

        The previous version set the offsets to the sheet's live deviation, which made the object
        keypoints exact but silently destroyed the goal: with one shared offsets array,
        ``obj_kp - goal_kp`` collapses to ``obj_centroid - goal_centroid`` repeated four times, so
        the policy saw where the goal centroid was and nothing about its shape. Rest offsets plus a
        real rotation keep both halves of the comparison meaningful.
        """
        self._keypoint_offsets[:] = self._corner_rest_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)

    def _drive_goal_marker(self) -> None:
        """Point `goal_viz` at the fold target, so the OBSERVED goal is the fold.

        Without this the policy is shown the inherited tool-pose goal while success is measured as
        a fold: it would be optimising a different objective from the one being scored. The task
        reads the goal exclusively through `goal_viz`, so moving the marker moves the goal.
        """
        centre = self.fold_targets_w().mean(dim=1)  # (num_envs, 3)
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = centre
        # Folded ORIENTATION, not just position: a fold rotates the half 180 degrees about the
        # crease line. With the manipuland's orientation now fitted from its corners, the task's
        # keypoint machinery turns this into the folded corner positions -- so `keypoints_rel_goal`
        # finally describes a fold rather than a translation.
        #
        # The crease runs along the axis perpendicular to `fold_axis`, so a fold about an x-normal
        # crease is a 180-degree rotation about y (and vice versa) IN THE SHEET'S REST FRAME.
        #
        # Composed with the stationary half's live rotation rather than written as a fixed world
        # quaternion. The position above already tracks rotation (it comes from `fold_targets_w`),
        # but a constant orientation does not, so a yawed sheet was shown a goal whose position was
        # right and whose attitude was the rest one. That reaches the POLICY, not just the render:
        # the task turns `goal_viz`'s pose into `keypoints_rel_goal`. Required before init yaw
        # could be randomised.
        r, _ = self._stationary_frame(self._particles_w())        # (N, 3, 3) rest -> current
        fold_rest = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        # 180 degrees about rest y (fold_axis == "x") or about rest x.
        if self.cfg.cloth.fold_axis == "x":
            fold_rest[:, 0, 0] = -1.0
            fold_rest[:, 2, 2] = -1.0
        else:
            fold_rest[:, 1, 1] = -1.0
            fold_rest[:, 2, 2] = -1.0
        pose[:, 3:7] = self._goal_quat_from_matrix(torch.bmm(r, fold_rest))
        self.goal_viz.write_root_pose_to_sim(pose)

    def _goal_quat_from_matrix(self, rot: torch.Tensor) -> torch.Tensor:
        """The quaternion ``_drive_goal_marker`` writes for the fold rotation ``rot``.

        ``isaaclab.utils.math.quat_from_matrix`` as imported by this module, unpatched. Under Isaac
        Lab 3.0 that returns (x, y, z, w), which the pose-write patch then treats as (w, x, y, z), so
        the goal orientation the task reads back is the true one's components relabelled -- and every
        Newton policy was trained on that. A backend whose helper is wxyz overrides this to hand the
        task the same relabelled quaternion (see ``backends/rigid/env.py``).
        """
        return quat_from_matrix(rot)

    def footprint_ratio(self) -> torch.Tensor:
        """Sheet extent along the fold axis, as a fraction of its unfolded size.

        A fold halves the footprint; a slide or a crumple does not necessarily. Keypoint proximity
        alone cannot tell those apart -- a dragged, crumpled sheet put keypoints near their targets
        and scored, which is why this exists as a second, independent condition.

        **Measured along the sheet's own fold axis, not the world's.** This used to take the extent
        of `parts[:, :, ax]` -- a world axis -- which is only the fold axis while the sheet is
        axis-aligned. A yawed sheet then reports an inflated footprint for purely geometric
        reasons: an unfolded 0.10 m sheet at 45 degrees spans 0.141 m in world x, a ratio of 1.41
        with no deformation at all. That is the likely source of the 1.03-1.41 values seen in the
        render HUD, which were previously read as the sheet stretching under the hand.

        Since `footprint_ratio < max_folded_footprint` is half of `is_folded`, the world-axis
        version made a correctly folded but rotated sheet score as unfolded. It also had to be
        fixed before init yaw could be randomised at all.

        `r[:, :, ax]` is the image of the rest basis vector under the stationary half's fitted
        rotation, i.e. the fold axis carried onto the sheet's current pose. Reduces exactly to the
        old behaviour when the sheet is axis-aligned (r = I).
        """
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1
        parts = self._particles_w()
        r, _ = self._stationary_frame(parts)
        axis_dir = r[:, :, ax]                                   # (N, 3), unit
        coord = torch.einsum("npi,ni->np", parts, axis_dir)       # (N, P)
        return (coord.amax(dim=1) - coord.amin(dim=1)) / self.cfg.cloth.size

    def is_folded(self) -> torch.Tensor:
        """Both conditions: keypoints on target AND the footprint halved."""
        return (self.fold_error() < self.cfg.cloth.keypoint_tolerance) & (
            self.footprint_ratio() < self.cfg.cloth.max_folded_footprint
        )

    def fold_fraction(self) -> torch.Tensor:
        """Progress in ``[0, 1]``: 0 at the rest pose, 1 at the target.

        Reported rather than used for reward, so a run that moves the cloth without folding it is
        visibly distinct from one that does not touch it -- the distinction that took far too long
        to establish on the cable.
        """
        start = (
            torch.tensor(
                [self._cloth_rest_local[i] for i in self._cloth_kp_idx],
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            + torch.tensor(
                [0.0, 0.0, float(self.cfg.reset.table_reset_z) + self.cfg.cloth.start_height],
                device=self.device,
            )
            + self.scene.env_origins.unsqueeze(1)
        )
        span = (start - self.fold_targets_w()).norm(dim=-1).amax(dim=-1).clamp(min=1e-6)
        return (1.0 - self.fold_error() / span).clamp(0.0, 1.0)
