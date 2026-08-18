"""ClothEnv — fold half a cloth sheet onto the other half.

Subclasses :class:`PlayNewtonEnv`, so the robot, table, action space and coupled solver are the
ones already measured on the cable task. Only the manipuland changes.

The sheet is authored as a flat ``UsdGeom.Mesh`` grid before the environments are replicated --
`DeformableObject` infers ``surface`` (cloth) from a ``UsdGeom.Mesh`` and ``volume`` from a
``UsdGeom.TetMesh``, so the prim type *is* the switch.

State is the position of a few keypoints on the moving half; the goal is their mirror image across
the fold line. See ``utils/cloth_geometry.py`` for both, kept separate so the fold is testable
without a simulator.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.newton import patches
from isaacsimenvs.tasks.cloth.cloth_env_cfg import ClothEnvCfg
from isaacsimenvs.tasks.cloth.utils.cloth_adapter import ClothAsRigidObject
from isaacsimenvs.tasks.cloth.utils.cloth_geometry import (
    corner_indices,
    folded_targets,
    grid_mesh,
    half_indices,
    half_mesh,
    keypoint_indices,
)
from isaacsimenvs.tasks.play_newton.play_newton_env import PlayNewtonEnv

__all__ = ["ClothEnv"]


class ClothEnv(PlayNewtonEnv):
    """Play task with a VBD cloth sheet as the manipuland."""

    cfg: ClothEnvCfg

    @staticmethod
    def build_physics_cfg(cfg: "ClothEnvCfg"):
        """Coupled MJWarp (robot) + VBD (cloth) solve, sized for this env count."""
        return cfg.cloth.build_physics(cfg.newton, int(cfg.scene.num_envs))

    #: Runs at the end of `setup_scene`, before replication -- which is also when Newton imports
    #: the stage. A mesh authored after that point is invisible to the solver.
    pre_clone_scene_hook = staticmethod(lambda env: env._install_cloth())

    def __init__(self, cfg: ClothEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_fold_targets()
        self._fold_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Envs whose state went non-finite this step; consumed by `_get_dones` to force a reset.
        self._nonfinite_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._nonfinite_count = 0
        # Best fold error seen so far this episode, for the dense progress ratchet. Kept separate
        # from the inherited `_closest_keypoint_max_dist`, which `compute_intermediate_values`
        # resolves against the APPROXIMATE metric inside `super()._get_dones()` -- reusing it would
        # ratchet against a different quantity than the one being rewarded. -1 is the sentinel the
        # task uses for "not yet observed".
        self._closest_fold_err = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )
        # Set in `_get_dones`, consumed by `_get_rewards` in the same step: which envs earned the
        # sparse bonus this step, under the fold criterion rather than the inherited one.
        self._fold_bonus_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Fold error of a perfectly flat sheet -- the worst legitimate value. Used as the stand-in
        # when an env's fold error is non-finite, so the substitution can never look like progress.
        self._flat_fold_err = float(self.fold_error().nan_to_num(nan=0.0).max().item())
        self._drive_goal_marker()

    # ------------------------------------------------------------------ construction

    def _install_cloth(self) -> None:
        """Author the sheet and register it as a deformable, in place of the rigid tool."""
        # Re-validate here, not only in `ClothCfg.__post_init__`. Hydra constructs the dataclass
        # with its DEFAULTS -- which is when `__post_init__` runs and passes -- and only then
        # assigns the CLI overrides onto the instance. So `env.cloth.thickness=0.03` walks straight
        # past the thickness/spacing guard: it ran against thickness 0.012 and never saw 0.03.
        # That is how a 30 mm radius ended up on a 12.5 mm grid, every particle overlapping its
        # neighbours, which the solver expresses as instability rather than as a config error.
        self.cfg.cloth._check_thickness()
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Sdf, UsdGeom, UsdShade

        c = self.cfg.cloth
        verts, indices = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        # Corners of the moving half, not points along one edge: four points spanning an area
        # determine a rotation; four collinear points do not.
        self._cloth_kp_idx = corner_indices(c.resolution, c.fold_axis)

        # The inherited rigid tool is still spawned by `setup_scene` -- the asset pipeline, USD
        # cache and `object_scales` observation are all built around it. Make it inert, or it is a
        # live rigid body loose in the scene. (On the cable task this was visible for hours as a
        # hammer lying on the floor beside the table.)
        self._neutralise_rigid_object()
        self._reshape_goal_marker()

        prim_path = "/World/envs/env_.*/Cloth"
        spawn_z = float(self.cfg.reset.table_reset_z) + c.start_height

        # Author the mesh directly: `DeformableObject` reads vertices and indices off the prim, and
        # a `UsdGeom.Mesh` is what makes it a *surface* (cloth) rather than a volume.
        stage = None
        for env_prim in find_matching_prims("/World/envs/env_.*"):
            stage = env_prim.GetStage()
            mesh_path = env_prim.GetPath().AppendChild("Cloth")
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            # Height is baked into the POINTS, not applied as a transform and not left to
            # `init_state.pos`. Measured: init_state alone left the sheet on the ground (z=0.006),
            # and a translate op on top of it put the sheet at 1.15 m and then exploded to 70 m.
            # Baking it is the one placement that cannot compose with another.
            mesh.CreatePointsAttr([(v[0], v[1], v[2] + spawn_z) for v in verts])
            mesh.CreateFaceVertexIndicesAttr(list(indices))
            mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            mesh.CreateDisplayColorAttr([tuple(c.color)])
            # Doubled-sided: a cloth folded over itself is viewed from both sides, and a
            # single-sided sheet renders as half-invisible once flipped.
            mesh.CreateDoubleSidedAttr(True)
            # NO translate op here. `DeformableObject` reads these points and then applies
            # `init_state.pos` on top, so a transform on the prim composes with it and the sheet
            # spawns at twice the intended height -- which is exactly what it did.

            # `DeformableObject` reads its material off a *bound* physics material carrying
            # `newton:*` attributes, and refuses the prim outright without one. It identifies the
            # material by the presence of `newton:density`, so that attribute is what makes this a
            # Newton deformable material rather than an ordinary shading material.
            mat_path = env_prim.GetPath().AppendChild("ClothMaterial")
            mat = UsdShade.Material.Define(stage, mat_path)
            mp = mat.GetPrim()
            for name, value in (
                ("newton:density", c.density),
                ("newton:particleRadius", c.particle_radius),
                ("newton:triKe", c.tri_ke),
                ("newton:triKa", c.tri_ka),
                ("newton:triKd", c.tri_kd),
                ("newton:edgeKe", c.edge_ke),
                ("newton:edgeKd", c.edge_kd),
            ):
                mp.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))

            binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
            binding.GetDirectBindingRel("physics").SetTargets([mat_path])

        cloth_cfg = DeformableObjectCfg(
            prim_path=prim_path,
            spawn=None,  # the prim is authored above; nothing to spawn
            # Identity: the height is already in the points.
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        cloth = DeformableObject(cloth_cfg)
        self.scene.deformable_objects["cloth"] = cloth
        self._cloth = cloth

        # `PlayEnv` holds the manipuland as `env.object` and every downstream module reads it from
        # there. Without this the policy observes the *inert rigid tool* -- which this env has just
        # made kinematic and collision-free -- while the sheet sits in the scene as a bystander.
        area_density = c.density * c.thickness  # surface density -> sheet mass
        self.object = ClothAsRigidObject(
            cloth,
            num_envs=self.num_envs,
            # LOCAL coordinates (z=0). `write_root_pose_to_sim` adds a centre whose z is
            # `spawn_z`, so baking the height in here too would place the sheet at 2 x spawn_z --
            # which it did, at 1.06 m instead of 0.53 m.
            rest_local=verts,
            keypoint_idx=self._cloth_kp_idx,
            corner_idx=self._cloth_kp_idx,
            corner_rest=[verts[i] for i in self._cloth_kp_idx],
            spawn_z=spawn_z,
            mass=area_density * c.size * c.size,
            device=self.device,
        )
        self._write_object_scale()

        print(
            f"[cloth] {c.resolution}x{c.resolution} sheet, {c.size:.3f} m, "
            f"{c.num_particles} particles, {len(indices) // 3} triangles, "
            f"keypoints {self._cloth_kp_idx}",
            flush=True,
        )

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

    def _neutralise_rigid_object(self) -> None:
        """Disable the inherited rigid tool's colliders so it cannot touch the cloth.

        Walks the whole subtree: the colliders sit at ``/Object/<mesh>/collisions``, two levels
        down, and a one-level loop silently disables nothing.
        """
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Usd, UsdPhysics

        colliders = 0
        prims = 0
        for prim in find_matching_prims("/World/envs/env_.*/Object"):
            prims += 1
            for target in Usd.PrimRange(prim):
                if target.HasAPI(UsdPhysics.CollisionAPI):
                    api = UsdPhysics.CollisionAPI(target)
                    attr = api.GetCollisionEnabledAttr() or api.CreateCollisionEnabledAttr()
                    attr.Set(False)
                    colliders += 1
                if target.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = UsdPhysics.RigidBodyAPI(target)
                    attr = body.GetKinematicEnabledAttr() or body.CreateKinematicEnabledAttr()
                    attr.Set(True)
        print(f"[cloth] neutralised rigid tool ({prims} prims, {colliders} colliders off)", flush=True)
        if prims and not colliders:
            raise RuntimeError(
                "found the inherited rigid tool but disabled none of its colliders -- it would "
                "stay live and corrupt every measurement. Check the prim layout."
            )

    def _reshape_goal_marker(self) -> None:
        """Give the goal marker the folded half's shape instead of the tool's.

        ``goal_viz`` spawns from the same procedural handle+head USD as the manipuland, so the cloth
        task renders a hammer-shaped ghost for a goal that is a folded sheet. The marker *pose* is
        already correct and success is scored on keypoints, not on this mesh -- but the render is
        the only thing a person actually looks at, and a hammer standing in for a fold invites
        exactly the wrong reading. The same defect on the cable task cost hours.

        Swaps the tool geometry for a thin slab the size of the moving half. ``collisionEnabled``
        stays false so the marker can never touch the sheet;
        ``render_newton._reveal_goal_viz`` draws it by setting the VISIBLE flag, which does not
        require a collider.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from isaaclab.sim.utils import find_matching_prims

        c = self.cfg.cloth
        # The half's own grid, not a box: the marker stands in for a folded SHEET, and a solid slab
        # both reads as a brick and shows a thickness the cloth does not have (a VBD cloth is a
        # surface -- its thickness is a collision radius the renderer never draws, so the sheet
        # itself always renders flat).
        gv, gi = half_mesh(float(c.size), int(c.resolution), c.fold_axis)

        marked = 0
        for prim in find_matching_prims("/World/envs/env_.*/GoalViz"):
            stage = prim.GetStage()
            # The rigid body sits on a child (`.../GoalViz/object_root`), not the GoalViz root.
            # Deactivating that child would remove the body the physics view resolves, so keep the
            # body and replace only the geometry beneath it.
            body = next(
                (d for d in Usd.PrimRange(prim) if d.HasAPI(UsdPhysics.RigidBodyAPI)), prim
            )
            for child in body.GetChildren():
                child.SetActive(False)
            sheet = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("cloth_marker"))
            sheet.CreatePointsAttr([Gf.Vec3f(*v) for v in gv])
            sheet.CreateFaceVertexIndicesAttr(gi)
            sheet.CreateFaceVertexCountsAttr([3] * (len(gi) // 3))
            UsdPhysics.CollisionAPI.Apply(sheet.GetPrim()).CreateCollisionEnabledAttr(False)
            marked += 1

        print(
            f"[cloth] goal marker reshaped on {marked} prims "
            f"(half-sheet mesh, {len(gv)} verts / {len(gi) // 3} tris)",
            flush=True,
        )

    # ------------------------------------------------------------------ the fold

    def _init_fold_targets(self) -> None:
        """World-frame targets for the tracked keypoints, per env."""
        c = self.cfg.cloth
        lift = 2.0 * c.particle_radius  # the flap rests on top of the stationary half
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
        # reflect across the crease, then lift by one thickness onto the stationary half. At
        # runtime this is carried onto the sheet's live pose by the stationary half's own rigid
        # frame, which is what makes the target follow the cloth when it is rotated as well as
        # when it is translated.
        def _folded_rest(sel: torch.Tensor) -> torch.Tensor:
            f = rest[sel].clone()
            f[:, ax] = 2.0 * crease_local - f[:, ax]
            f[:, 2] = f[:, 2] + c.thickness
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
        crease and lifted one thickness onto the stationary half -- and then carried onto the
        stationary half's CURRENT rigid frame, fitted from its particles.

        That frame is the whole point. Translating or rotating the cloth moves the target with it,
        so neither can reduce the error; only actually folding the flap can. An earlier version got
        this half right -- the fold-axis coordinate tracked the live crease, but the lateral one
        stayed pinned to the spawn position, so a rotated sheet was scored against an axis-aligned
        target and a genuine fold could read as a failure.
        """

        # Carried onto the stationary half's CURRENT rigid frame, so the target follows the sheet
        # through rotation as well as translation. The rest-frame definition already encodes both
        # the reflection across the crease and the one-thickness lift onto the stationary half --
        # one thickness, not two, because particle centres are what both sides measure and a flap
        # resting on the half sits exactly one particle diameter above it.
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
        rest = self._corner_rest_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        self._keypoint_offsets[:] = rest
        self._keypoint_offsets_fixed[:] = rest

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
        pose[:, 3:7] = quat_from_matrix(torch.bmm(r, fold_rest))
        self.goal_viz.write_root_pose_to_sim(pose)

    def _get_rewards(self) -> torch.Tensor:
        """Re-point the inherited Play reward at the fold.

        Follows `PreciseAssemblyEnv._get_rewards` (play2perfect): keep `compute_rewards`, then
        subtract the terms that fight this task and add fold-based replacements. Written as
        `reward - term + replacement` so every inherited term stays visible in `_reward_terms` and
        in the wandb/tensorboard breakdown.

        Three substitutions:

        **Lift terms out.** `lifting_rew` (x20) and `lifting_bonus` (300) pay for raising the
        manipuland 0.15 m above its spawn height. A fold keeps the sheet on the table, so these
        reward the opposite of the task -- exactly why play2perfect masks them for insertion, where
        the goal pulls the peg *down*.

        **The `lifted` gate off the keypoint term.** `keypoint_reward` is multiplied by
        `lifted.float()` (`reward_utils.py:50`), and `_lifted_object` latches only on a 0.15 m rise.
        A folding sheet never rises that far, so the inherited keypoint reward pays **exactly zero
        for the entire episode**. Masking the lift terms without also removing this gate would leave
        the main shaping signal dead -- it already is.

        **The keypoint metric.** The inherited term ratchets on `_keypoints_max_dist`, which is
        `max |obj_kp - goal_kp|` built from the fitted rigid frame plus rest offsets -- an
        approximation of the fold. `fold_error()` is the exact quantity the success criterion and
        the eval both use. Ratcheting on that makes the dense signal *fold progress*.

        The bonus is likewise re-keyed: the inherited `reach_goal_bonus` fires on `_near_goal` /
        `_is_success`, which are computed from the approximate metric against
        `_current_success_tolerance`. `_fold_bonus_mask` is set in `_get_dones` from the same
        `is_folded()` the criterion uses, so the +1000 pays for a fold and nothing else.

        Kept unchanged: `fingertip_delta_rew` (approach shaping, helps the hand find the sheet) and
        both joint-velocity penalties, which are play2perfect's `r_smooth` and the regularisation
        the pretrained policy was trained under.
        """
        from isaacsimenvs.tasks.play.utils.logging_utils import log_step_metrics
        from isaacsimenvs.tasks.play.utils.reward_utils import compute_rewards

        reward = compute_rewards(self)
        terms = self._reward_terms
        rew_cfg = self.cfg.reward

        # --- out: lift progress, lift bonus, and the approximate keypoint term ---------------
        reward = reward - terms["lifting_rew"] - terms["lift_bonus_rew"] - terms["keypoint_rew"]
        zero = torch.zeros_like(terms["lifting_rew"])
        terms["lifting_rew"] = zero
        terms["lift_bonus_rew"] = zero

        # --- in: dense fold progress, ungated by `lifted` -------------------------------------
        # Same ratchet as `reward_utils.keypoint_reward`: pay only for improvement on the best
        # fold error seen so far this goal, clamped at zero so regression is free rather than
        # punished, and monotone so it cannot be farmed by oscillating.
        # A non-finite fold error would poison the ratchet permanently (torch.minimum with NaN
        # propagates) and then every subsequent reward. Envs whose state has diverged are being
        # reset this step anyway -- `_guard_finite` flagged them and `_get_dones` terminated them --
        # so substituting the flat-sheet value costs nothing and keeps the ratchet finite.
        fold_err = self.fold_error()
        bad = ~torch.isfinite(fold_err)
        if bool(bad.any()):
            fold_err = torch.where(bad, torch.full_like(fold_err, self._flat_fold_err), fold_err)
        sentinel = self._closest_fold_err < 0.0
        self._closest_fold_err = torch.where(sentinel, fold_err, self._closest_fold_err)
        delta = torch.clamp(self._closest_fold_err - fold_err, 0.0, 100.0)
        self._closest_fold_err = torch.minimum(self._closest_fold_err, fold_err)
        fold_kp_rew = delta * float(rew_cfg.keypoint_rew_scale)
        reward = reward + fold_kp_rew
        terms["keypoint_rew"] = fold_kp_rew

        # --- the sparse bonus, keyed on the fold criterion ------------------------------------
        reward = reward - terms["bonus_rew"]
        fold_bonus = self._fold_bonus_mask.float() * float(rew_cfg.reach_goal_bonus)
        reward = reward + fold_bonus
        terms["bonus_rew"] = fold_bonus

        # LAST LINE OF DEFENCE. `compute_rewards` reads robot state, fingertip distances and the
        # inherited keypoint metric, any of which can be non-finite when an articulation diverges --
        # and a NaN reward becomes a NaN advantage and then a NaN gradient, which destroys the
        # policy rather than just losing one env. Observed in a smoke run as
        # `shaped_rewards/iter -> nan` while `rewards/iter` still read finite.
        nonfinite_rew = ~torch.isfinite(reward)
        if bool(nonfinite_rew.any()):
            n = int(nonfinite_rew.sum())
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            self._nonfinite_envs |= nonfinite_rew
            print(
                f"[cloth] non-finite REWARD in {n}/{self.num_envs} envs -- zeroed and flagged "
                f"for reset",
                flush=True,
            )
        self.extras["nonfinite_reward"] = float(nonfinite_rew.float().mean())

        terms["total_reward"] = reward
        self.extras["fold_err_mean"] = float(fold_err.mean())
        self.extras["fold_rate"] = float(self._fold_bonus_mask.float().mean())
        log_step_metrics(self)
        return reward

    def _get_observations(self):
        # Offsets must be current *before* the task reads them, or the observation lags the sheet
        # by one step.
        self._sync_observed_keypoints()
        # The marker has to be re-driven EVERY step, not just at reset. `fold_targets_w` is defined
        # against the sheet's *current* crease and stationary height -- that is what makes the
        # criterion translation-invariant -- so a marker written once at reset drifts away from the
        # target actually being scored as soon as the sheet moves. It then shows the policy (and the
        # viewer) a goal that is not the goal.
        self._drive_goal_marker()
        obs = super()._get_observations()
        self._guard_finite(obs)
        return obs

    def _guard_finite(self, obs) -> None:
        """Keep non-finite values out of the policy, and flag the env for reset.

        A NaN anywhere in the 140-dim vector reaches the policy, propagates through the network and
        surfaces as ``normal expects all elements of std >= 0.0`` from the action sampler -- a
        message that names the sampler and says nothing about the cloth. Guarding the *rotation fit*
        was not enough, because that only covered one of several ways a blown-up sheet produces
        NaN. This reports which quantity went bad first, so the next fix is aimed rather than
        guessed.
        """
        # Ordered most-upstream first, so the guard names the ROOT quantity rather than the
        # observation it poisons.
        #
        # **The robot is checked, not just the cloth.** 80 of the 140 policy columns are robot
        # state (joint_pos 0-28, joint_vel 29-57, palm_pos 87-89, palm_rot 90-93,
        # fingertip_pos_rel_palm 98-112) against 20 that are cloth-derived. The first version of
        # this guard watched only cloth quantities and duly reported "obs bad, cloth fine" --
        # true, and useless, because it never looked where most of the vector comes from. In a
        # coupled solve the MJWarp articulation can diverge from a bad contact while the VBD sheet
        # stays perfectly well behaved.
        #
        # `reward_buf` is deliberately NOT here: the task does feed the reward into an observation
        # ("reward": reward_buf * 0.01, `obs_utils.py:326`) but only into `state_list`, the
        # critic's 162-dim state. It is absent from `obs_list`, so it cannot reach `obs["policy"]`.
        robot = self.robot.data
        offenders = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        parts = {
            "particles": self._particles_w(),
            "velocities": self.object._velocities(),
            "quat": self.object.data.root_quat_w,
            "fold_targets": self.fold_targets_w(),
            "robot_joint_pos": robot.joint_pos,
            "robot_joint_vel": robot.joint_vel,
            "robot_body_pos_w": robot.body_pos_w,
            "robot_body_quat_w": robot.body_quat_w,
            "obs": obs["policy"],
        }
        for name, tensor in parts.items():
            if tensor is None:
                continue
            flat = tensor.reshape(tensor.shape[0], -1)
            bad = ~torch.isfinite(flat).all(dim=-1)
            if not bool(bad.any()):
                continue
            offenders |= bad
            ids = torch.nonzero(bad).flatten().tolist()
            where = ""
            if name == "obs":
                # Which columns, so the term can be identified against `cfg.obs.obs_list`.
                cols = torch.nonzero(~torch.isfinite(flat[ids[0]])).flatten().tolist()
                where = f" cols {cols[:12]}{'...' if len(cols) > 12 else ''} of {flat.shape[1]}"
            # The OFFENDING envs' own progress, not `episode_length_buf.max()`. The max is taken
            # over all envs, so it is only equal to the offender's step while the envs are still
            # synchronised -- i.e. during the first episode. After that it saturates near
            # `episode_length` and every report reads "at step 599", which looks like the failure
            # clusters at the end of an episode when it does not. Per-env progress says when the
            # divergence actually happened.
            steps = self.episode_length_buf[bad].tolist()
            msg = (
                f"cloth: non-finite {name} in envs {ids[:8]}"
                f"{'...' if len(ids) > 8 else ''} ({len(ids)}/{self.num_envs}) "
                f"at env-step {steps[:8]}{'...' if len(steps) > 8 else ''}{where}"
            )
            if self.cfg.cloth.nan_policy == "raise":
                raise RuntimeError(msg)
            # `reset`: report once per occurrence and carry on. Printing every step would drown a
            # training log, so this is deliberately not rate-limited beyond being per-detection.
            print(f"[cloth] {msg} -- resetting those envs", flush=True)

        if not bool(offenders.any()):
            return

        # NEVER let a non-finite value reach the policy. The env is flagged for reset below, but the
        # reset only takes effect on the NEXT step, so this step's observation would still be handed
        # to the network -- and a single NaN propagates through the LSTM into the action sampler,
        # then into the gradient. Zeroing is not physically meaningful, but the env is being thrown
        # away anyway; what matters is that the number entering the optimizer is finite.
        for tensor in obs.values():
            if torch.is_tensor(tensor):
                torch.nan_to_num_(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Latched, and consumed by `_get_dones` on the same step so the env terminates and resets
        # through the normal path rather than being patched in place.
        self._nonfinite_envs |= offenders
        self._nonfinite_count += int(offenders.sum())
        self.extras["nonfinite_envs"] = float(offenders.float().mean())
        self.extras["nonfinite_total"] = float(self._nonfinite_count)

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

    #: Steps the fold criterion must hold to count as a *held* fold. Independent of
    #: `termination.success_steps`, which the finetune recipe drives to 1 so the reward bonus fires
    #: on first entry -- the strict criterion still has to be reported.
    HELD_FOLD_STEPS = 10

    def _get_dones(self):
        """Terminate on a completed fold, on the sheet leaving the table, or on timeout.

        The inherited test scores a *tool pose* against `goal_viz`, which is meaningless here: the
        manipuland is a sheet and the goal is a fold. This REPLACES the success half of it while
        keeping the task's fall/hand-far conditions.

        **The inherited increment is undone, not merely added to.** `super()._get_dones()` runs
        Play's own keypoint-success path in full, which increments `_successes`
        (`termination_utils.py:46`) whenever `_keypoints_max_dist <= _current_success_tolerance *
        keypoint_scale`. With `eval_success_tolerance` unset that fires at 0.075 * 1.5 = 11.25 cm,
        against the fold criterion's 4 cm -- so `_successes` had two independent writers and every
        "goals" number on this task was ambiguous between a fold and a loose tool-pose hit. Worse,
        a reward keyed on `_successes` would have been farmable at 11.25 cm without ever folding.
        Snapshot and restore so the fold is the only writer.
        """
        successes_before = self._successes.clone()
        terminated, truncated = super()._get_dones()
        self._successes.copy_(successes_before)

        folded = self.is_folded()

        # Two counters, deliberately. `entered` is what the reward bonus and termination key on --
        # `termination.success_steps` is 1 under the finetune recipe, matching Play2Perfect's
        # `success_steps=1` + `force_consecutive_near_goal_steps=True` lump sum. `held` is the
        # strict criterion every baseline number so far used, kept so training and evaluation can be
        # compared against the same history. The gap between them is a diagnostic: a flap that
        # enters the target and springs back scores `entered` but not `held`.
        self._fold_hold = torch.where(folded, self._fold_hold + 1, torch.zeros_like(self._fold_hold))
        entered = self._fold_hold >= int(self.cfg.termination.success_steps)
        held = self._fold_hold >= int(self.HELD_FOLD_STEPS)

        # `_get_rewards` runs after `_get_dones` in the same step, so this is the mask the sparse
        # bonus pays on. Keyed on the fold criterion, never on the inherited `_is_success`.
        self._fold_bonus_mask.copy_(entered)

        if bool(entered.any()):
            ids = entered.nonzero(as_tuple=True)[0]
            self._successes[ids] += 1

        # NOTE: `episode_length_buf` is deliberately NOT reset here. The inherited task chains
        # goals within an episode, but a folded sheet stays folded, so re-arming would let the
        # criterion pay repeatedly for one fold. The episode ends on the fold instead (below),
        # which is the analogue of Play2Perfect ending the episode on a successful retract.
        # Terminate any env whose state went non-finite, so it resets through the normal path.
        # `_guard_finite` has already zeroed the observation for these envs, so nothing non-finite
        # reaches the policy or the optimizer; this is what stops one diverged env from poisoning a
        # multi-hour run. Cleared after consumption -- the reset itself restores a valid state.
        nonfinite = self._nonfinite_envs.clone()
        self._nonfinite_envs.zero_()

        self._termination_reasons["fold"] = entered
        self._termination_reasons["fold_held"] = held
        self._termination_reasons["nonfinite"] = nonfinite
        return terminated | entered | nonfinite, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_fold_hold"):
            self._fold_hold[env_ids] = 0
        if hasattr(self, "_closest_fold_err"):
            # Back to the sentinel: the ratchet is per-episode, so a new episode must not inherit
            # the previous one's best fold error (which would make all progress unrewarded).
            self._closest_fold_err[env_ids] = -1.0
            self._fold_bonus_mask[env_ids] = False
        # The task re-samples `goal_viz` on reset; put it back on the fold target.
        if hasattr(self, "_fold_targets_xy"):
            self._drive_goal_marker()

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
