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

        So the fold is defined against the sheet's own current configuration:

        * the crease is the stationary half's current inboard edge, so it travels with the sheet
        * each target is its keypoint's rest offset reflected about that crease
        * the height is the stationary half's current mean plus two thicknesses

        Every term moves with the sheet, so a rigid translation leaves the error unchanged and only
        an actual fold reduces it.
        """
        parts = self._particles_w()
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1

        stationary = parts[:, self._stationary_idx, :]
        crease = self._crease_w(parts, ax)
        stationary_z = stationary[:, :, 2].mean(dim=1)                  # (N,)

        targets = self._particles_w()[:, self._kp_idx_t, :].clone()
        # Reflect the keypoint's REST offset from the crease -- rest, not current, or a keypoint
        # already at the target would define its own target and the error would read zero.
        targets[:, :, ax] = crease.unsqueeze(1) - self._kp_rest_offset.unsqueeze(0)
        targets[:, :, 1 - ax] = (
            self._kp_rest_lateral.unsqueeze(0) + self.scene.env_origins[:, 1 - ax].unsqueeze(1)
        )
        # ONE thickness, not two. Particle centres are what these targets and `stationary_z` both
        # measure, and a flap resting on the stationary half sits exactly one particle DIAMETER --
        # i.e. one `thickness` -- above it. `2 * thickness` put the target a full sheet-thickness
        # into the air: 30 mm of pure height error at `thickness=0.03`, against a 40 mm tolerance,
        # so a geometrically perfect fold could not score. It is also why the marker floats above
        # the sheet in the renders.
        targets[:, :, 2] = (stationary_z + self.cfg.cloth.thickness).unsqueeze(1)
        return targets

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
        """
        parts = self._particles_w()
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1
        stationary = parts[:, self._stationary_idx, :]
        crease = self._crease_w(parts, ax)
        stationary_z = stationary[:, :, 2].mean(dim=1)

        out = parts[:, self._moving_idx, :].clone()
        out[:, :, ax] = crease.unsqueeze(1) - self._mv_rest_offset.unsqueeze(0)
        out[:, :, 1 - ax] = (
            self._mv_rest_lateral.unsqueeze(0) + self.scene.env_origins[:, 1 - ax].unsqueeze(1)
        )
        out[:, :, 2] = (stationary_z + self.cfg.cloth.thickness).unsqueeze(1)
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
        # crease is a 180-degree rotation about y (and vice versa).
        pose[:, 3] = 0.0                                        # w = cos(pi/2) = 0
        pose[:, 4 + (1 if self.cfg.cloth.fold_axis == "x" else 0)] = 1.0   # sin(pi/2) about y or x
        self.goal_viz.write_root_pose_to_sim(pose)

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
        self._assert_finite(obs)
        return obs

    def _assert_finite(self, obs) -> None:
        """Fail at the source when any part of the observation goes non-finite.

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
            ids = torch.nonzero(bad).flatten().tolist()
            where = ""
            if name == "obs":
                # Which columns, so the term can be identified against `cfg.obs.obs_list`.
                cols = torch.nonzero(~torch.isfinite(flat[ids[0]])).flatten().tolist()
                where = f" cols {cols[:12]}{'...' if len(cols) > 12 else ''} of {flat.shape[1]}"
            raise RuntimeError(
                f"cloth: non-finite {name} in envs {ids[:8]}"
                f"{'...' if len(ids) > 8 else ''} ({len(ids)}/{self.num_envs}) "
                f"at step {int(self.episode_length_buf.max().item())}{where}"
            )

    def footprint_ratio(self) -> torch.Tensor:
        """Sheet extent along the fold axis, as a fraction of its unfolded size.

        A fold halves the footprint; a slide or a crumple does not necessarily. Keypoint proximity
        alone cannot tell those apart -- a dragged, crumpled sheet put keypoints near their targets
        and scored, which is why this exists as a second, independent condition.
        """
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1
        coord = self._particles_w()[:, :, ax]
        return (coord.amax(dim=1) - coord.amin(dim=1)) / self.cfg.cloth.size

    def is_folded(self) -> torch.Tensor:
        """Both conditions: keypoints on target AND the footprint halved."""
        return (self.fold_error() < self.cfg.cloth.keypoint_tolerance) & (
            self.footprint_ratio() < self.cfg.cloth.max_folded_footprint
        )

    def _get_dones(self):
        """Terminate on a completed fold, on the sheet leaving the table, or on timeout.

        The inherited test scores a *tool pose* against `goal_viz`, which is meaningless here: the
        manipuland is a sheet and the goal is a fold. This replaces the success half of it while
        keeping the task's fall/hand-far conditions.
        """
        terminated, truncated = super()._get_dones()

        folded = self.is_folded()
        # Held for `success_steps`, matching the task's own criterion: a sheet passing through the
        # target on its way elsewhere is not a fold.
        self._fold_hold = torch.where(folded, self._fold_hold + 1, torch.zeros_like(self._fold_hold))
        success = self._fold_hold >= int(self.cfg.termination.success_steps)

        if bool(success.any()):
            ids = success.nonzero(as_tuple=True)[0]
            self._successes[ids] += 1
            self._fold_hold[ids] = 0
            # Reset the deadline on success, as `termination_utils` does for goal hits.
            self.episode_length_buf[ids] = 0

        self._termination_reasons["fold"] = success
        return terminated | success, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_fold_hold"):
            self._fold_hold[env_ids] = 0
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
