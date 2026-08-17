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
    folded_targets,
    grid_mesh,
    half_indices,
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
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Sdf, UsdGeom, UsdShade

        c = self.cfg.cloth
        verts, indices = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        self._cloth_kp_idx = keypoint_indices(c.resolution, c.fold_axis, c.num_keypoints)

        # The inherited rigid tool is still spawned by `setup_scene` -- the asset pipeline, USD
        # cache and `object_scales` observation are all built around it. Make it inert, or it is a
        # live rigid body loose in the scene. (On the cable task this was visible for hours as a
        # hammer lying on the floor beside the table.)
        self._neutralise_rigid_object()

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

    # ------------------------------------------------------------------ the fold

    def _init_fold_targets(self) -> None:
        """World-frame targets for the tracked keypoints, per env."""
        c = self.cfg.cloth
        lift = 2.0 * c.particle_radius  # the flap rests on top of the stationary half
        local = folded_targets(self._cloth_rest_local, self._cloth_kp_idx, c.fold_axis, lift)

        t = torch.tensor(local, device=self.device, dtype=torch.float32)  # (K, 3)
        # x/y only: the height is read from the settled sheet in `fold_targets_w`, not assumed.
        self._fold_targets_xy = t.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)
        self._stationary_idx = torch.tensor(
            half_indices(c.resolution, c.fold_axis, positive=False),
            device=self.device, dtype=torch.long,
        )
        self._kp_idx_t = torch.tensor(self._cloth_kp_idx, device=self.device, dtype=torch.long)

    def fold_targets_w(self) -> torch.Tensor:
        """``(num_envs, K, 3)`` fold targets, with height read from the sheet itself.

        A fold means "mirrored in x/y, resting on top of the stationary half". The x/y half is
        fixed geometry, but the **height is not knowable in advance**: the sheet is dropped and
        settles wherever the table and its own thickness put it. Deriving the target z from the
        *spawn* height put the targets ~33 mm below anything the cloth could reach -- a teleported,
        physically perfect fold measured 0.0355 error against a 0.04 tolerance, i.e. passing only
        by luck.

        So the height comes from the stationary half's current position plus two sheet thicknesses
        (the stationary layer and the folded one). That is robust to the settle height, to table
        thickness, and to any future change in either.
        """
        stationary_z = self._particles_w()[:, self._stationary_idx, 2].mean(dim=1)  # (N,)
        targets = self._fold_targets_xy.clone()                                     # (N, K, 3)
        targets[:, :, 2] = (stationary_z + 2.0 * self.cfg.cloth.thickness).unsqueeze(1)
        return targets

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
        """Make the OBSERVED keypoints the sheet's real particle positions.

        The task builds its keypoints as ``centroid + _keypoint_offsets`` with the manipuland's
        orientation (identity here). Setting the offsets to the sheet's actual deviation from its
        centroid therefore makes the observed keypoints *exactly* the tracked particles -- so the
        policy sees the half's deformation, not a synthetic box around it. The 140-dim layout is
        untouched.

        The goal keypoints come out right too, because of how the keypoints were chosen: all four
        sit on the far edge and share one ``x``, so reflecting them across the midline is the same
        as translating in ``x``. ``goal_centroid + offsets`` therefore lands on the fold targets
        rather than merely near them. Pick keypoints off that edge and this stops holding.
        """
        kp = self.object.keypoints_w()                      # (N, K, 3), real particles
        offsets = kp - kp.mean(dim=1, keepdim=True)
        self._keypoint_offsets[:] = offsets
        self._keypoint_offsets_fixed[:] = offsets

    def _drive_goal_marker(self) -> None:
        """Point `goal_viz` at the fold target, so the OBSERVED goal is the fold.

        Without this the policy is shown the inherited tool-pose goal while success is measured as
        a fold: it would be optimising a different objective from the one being scored. The task
        reads the goal exclusively through `goal_viz`, so moving the marker moves the goal.
        """
        centre = self.fold_targets_w().mean(dim=1)  # (num_envs, 3)
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = centre
        pose[:, 3] = 1.0  # identity wxyz, matching the manipuland's orientation convention here
        self.goal_viz.write_root_pose_to_sim(pose)

    def _get_observations(self):
        # Offsets must be current *before* the task reads them, or the observation lags the sheet
        # by one step.
        self._sync_observed_keypoints()
        return super()._get_observations()

    def _get_dones(self):
        """Terminate on a completed fold, on the sheet leaving the table, or on timeout.

        The inherited test scores a *tool pose* against `goal_viz`, which is meaningless here: the
        manipuland is a sheet and the goal is a fold. This replaces the success half of it while
        keeping the task's fall/hand-far conditions.
        """
        terminated, truncated = super()._get_dones()

        folded = self.fold_error() < self.cfg.cloth.keypoint_tolerance
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
