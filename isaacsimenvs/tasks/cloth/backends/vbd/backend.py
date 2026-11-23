"""``VbdBackend``: the fold task's sheet as a VBD cloth, on Isaac Lab 3.0 + Newton.

The env is constructed on Newton in two windows (``newton_setup.py``): the patches that must be in
place before the scene is built -- the coupled MJWarp (robot) + VBD (cloth) solver among them --
installed by ``__init__``, and the ones that need live assets, by :meth:`finish_setup`.

The sheet is authored as a flat ``UsdGeom.Mesh`` grid before the environments are replicated --
``DeformableObject`` infers ``surface`` (cloth) from a ``UsdGeom.Mesh`` and ``volume`` from a
``UsdGeom.TetMesh``, so the prim type *is* the switch. Its grid points are the particles.
"""

from __future__ import annotations

import re
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import patches  # noqa: F401  (first: the Isaac Lab 3.0 shims)

import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend, reset_yaw
from isaacsimenvs.tasks.cloth.backends.vbd import friction, plasticity
from isaacsimenvs.tasks.cloth.backends.vbd.newton_setup import (
    install_post_construction_patches,
    install_pre_construction_patches,
)
from isaacsimenvs.tasks.cloth.backends.vbd.physics import build_physics
from isaacsimenvs.tasks.cloth.config.schema.env import ClothDRCfg
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.mdp.state import SheetLayout

__all__ = ["VbdBackend"]


def _to_torch(value):
    """Isaac Lab 3.0 asset data are warp-backed ProxyArrays."""
    return value.torch if hasattr(value, "torch") else value


class VbdBackend(ClothBackend):
    """A VBD cloth sheet, coupled to the robot on Newton."""

    def __init__(self, cfg, robot_spec) -> None:
        super().__init__(cfg, robot_spec)
        # The sheet-physics DR is the rigid plates'; the VBD sheet has nothing it could act on.
        neutral = vars(ClothDRCfg())
        moved = {
            name: list(v) for name, v in vars(cfg.domain_randomization.cloth).items()
            if tuple(float(x) for x in v) != tuple(float(x) for x in neutral[name])
        }
        if moved:
            raise NotImplementedError(f"domain_randomization.cloth is the rigid plates' DR; on VBD keep it neutral, got {moved}")
        self._plasticity: plasticity.BendingPlasticity | None = None
        self._edge_env: torch.Tensor | None = None  # (edges,) the env of each hinge, -1 on boundary edges
        # Coupled MJWarp (robot) + VBD (cloth), sized for this env count.
        physics_cfg = build_physics(cfg.cloth, cfg.newton, int(cfg.scene.num_envs), robot_spec)
        # The proxied body selectors, while they are still regexes (the coupler resolves them to ids).
        self._proxy_bodies = [str(b) for proxy in physics_cfg.solver_cfg.proxies for b in proxy.bodies]
        install_pre_construction_patches(cfg, physics_cfg=physics_cfg)

    @contextmanager
    def construction_context(self):
        from isaaclab_newton.physics import NewtonManager

        # The cloth solver's own copy of the shapes' mu, when a cloth.cloth_*_friction is set.
        with friction.construction_hooks(self.cfg.cloth), ExitStack() as stack:
            if self.cfg.cloth.full_surface_contact:
                stack.enter_context(self._proxy_mesh_sdfs())
            if self.cfg.cloth.plastic_yield_curvature > 0:
                # Plasticity wraps the VBD solver's step: installed right before Newton captures its
                # CUDA graph, so the graph holds it (re-capturing after construction fails).
                capture = NewtonManager.__dict__["_capture_or_defer_graph"]
                backend = self

                def install_then_capture(cls):
                    backend._install_plasticity(cls._model, cls._solver)
                    return capture.__func__(cls)

                stack.enter_context(patch.object(NewtonManager, "_capture_or_defer_graph", classmethod(install_then_capture)))
            yield

    @contextmanager
    def _proxy_mesh_sdfs(self):
        """Build an SDF at finalize for every mesh shape the cloth touches (``cloth.full_surface_contact``):
        Newton's edge / face contacts sample it, and raise for a participating mesh without one. The
        proxied bodies' meshes (the hand) and the static particle-colliding ones; analytic shapes (the
        table box) need none. As origin/cloth-physics ``configure_materials``."""
        import newton
        from isaaclab_newton.physics import NewtonManager

        patterns = [re.compile(body) for body in self._proxy_bodies]
        prepare = NewtonManager.__dict__["_prepare_builder_for_finalize"]

        def prepare_with_sdfs(cls, builder):
            prepare.__func__(cls, builder)
            meshes = (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
            for shape, (kind, body) in enumerate(zip(builder.shape_type, builder.shape_body)):
                if kind not in meshes:
                    continue
                if body >= 0:
                    touched = any(p.search(builder.body_label[body] or "") for p in patterns)
                else:
                    touched = bool(builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES)
                if touched:
                    builder.shape_force_sdf[shape] = True

        with patch.object(NewtonManager, "_prepare_builder_for_finalize", classmethod(prepare_with_sdfs)):
            yield

    def _install_plasticity(self, model, coupled_solver) -> None:
        if self._plasticity is not None or model is None or coupled_solver is None:
            return
        self._plasticity = plasticity.install(model, coupled_solver, self.cfg.cloth)
        if self._plasticity is not None:
            hinges = model.edge_indices.numpy()
            env = model.particle_world.numpy()[hinges[:, 2]]
            env[(hinges[:, 0] < 0) | (hinges[:, 1] < 0)] = -1
            self._edge_env = torch.as_tensor(env, dtype=torch.long, device=str(model.device))

    def finish_setup(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        c = self.cfg.cloth
        # Normally installed at graph capture (construction_context); here if nothing captured.
        self._install_plasticity(NewtonManager._model, NewtonManager._solver)
        # Before install_post_construction_patches: its notify_solver refreshes the model views,
        # which is when the cloth's copy is re-written from the final shape frictions.
        tips = self.robot_spec.fingertip_link_names
        friction.install(NewtonManager._solver, c, tips)
        install_post_construction_patches(self)
        print(f"[cloth] friction: {friction.report(NewtonManager._solver, tips)}", flush=True)

    def end_reset(self, env_ids: torch.Tensor) -> None:
        # The reset sheets are flat again: so are their hinges' rest angles (plasticity's set).
        if self._plasticity is None:
            return
        import warp as wp

        ids = torch.as_tensor(env_ids, device=self._edge_env.device, dtype=torch.long).reshape(-1)
        edges = torch.isin(self._edge_env, ids).nonzero().flatten().to(torch.int32)
        if edges.numel():
            self._plasticity.reset(wp.from_torch(edges))

    # ------------------------------------------------------------------ the sheet, for the task

    def sheet_layout(self) -> SheetLayout:
        c = self.cfg.cloth
        return SheetLayout(
            rest_local=self.rest_local,
            keypoint_idx=list(self.keypoint_idx),
            resolution=int(c.resolution),
            fold_axis=c.fold_axis,
            size=float(c.size),
            # Two plies settle one SELF-contact radius apart; the particle radius is the cloth-RIGID
            # offset and has no say in a fold (``mdp/fold.py``).
            fold_lift=c.self_contact_radius,
            object_scale=self.object_scale,
        )

    def read_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The particles, (N, P, 3) each."""
        data = self.cloth.data
        return _to_torch(data.nodal_pos_w).view(self.num_envs, -1, 3), _to_torch(data.nodal_vel_w).view(self.num_envs, -1, 3)

    def write_sheet(self, env_ids: torch.Tensor, pose_w: torch.Tensor) -> None:
        """The sheet flat at the pose's xy (plus ``cloth.reset_offset_xy``) and :func:`reset_yaw`
        heading, at the spawn height, every particle at rest. Roll and pitch are dropped: a cloth has
        no rigid pose to take, and the task is defined on a sheet lying flat."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        centre = pose_w[:, :3].clone()
        centre[:, 2] = self._spawn_z
        centre[:, :2] += self._reset_offset
        yaw = reset_yaw(pose_w[:, 3:7])
        cos_y = torch.cos(yaw).unsqueeze(1)  # (n, 1), broadcast over particles
        sin_y = torch.sin(yaw).unsqueeze(1)
        rest_x = self._rest_t[:, 0].unsqueeze(0)  # (1, P)
        rest_y = self._rest_t[:, 1].unsqueeze(0)
        rest_z = self._rest_t[:, 2].unsqueeze(0)
        rotated = torch.stack(
            (rest_x * cos_y - rest_y * sin_y, rest_x * sin_y + rest_y * cos_y, rest_z.expand(cos_y.shape[0], -1)),
            dim=-1,
        )  # (n, P, 3)
        target = rotated + centre.unsqueeze(1)
        self.cloth.write_nodal_pos_to_sim_index(target, env_ids)
        self.cloth.write_nodal_velocity_to_sim_index(torch.zeros_like(target), env_ids)

    def observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        """``isaaclab.utils.math.quat_from_matrix``, unpatched: under Isaac Lab 3.0 it returns
        (x, y, z, w), which the task has always read as (w, x, y, z) -- the true quaternion's
        components relabelled -- and every policy was trained on that."""
        return quat_from_matrix(rot)

    def _write_object_scale(self) -> None:
        """``object_scales``: the sheet's bounding box, normalised by ``reward.object_base_size``."""
        c = self.cfg.cloth
        base = float(self.cfg.reward.object_base_size)
        bbox = torch.tensor([c.size, c.size, c.thickness], device=self.device, dtype=torch.float32)
        self.object_scale[:] = bbox / base
        print(
            f"[cloth] object_scale = {[round(float(v), 2) for v in bbox / base]} "
            f"(bbox {[round(float(v), 3) for v in bbox]} m / base {base})",
            flush=True,
        )

    # ------------------------------------------------------------------ the scene

    def install_sheet(self) -> None:
        """Author the sheet and register it as a deformable.

        Runs at the end of the scene setup, before replication -- which is also when Newton imports
        the stage: a mesh authored after that point is invisible to the solver."""
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Sdf, UsdGeom, UsdShade

        c = self.cfg.cloth
        verts, indices = grid_mesh(c.size, c.resolution, c.mesh_diagonals)
        self.rest_local = verts
        # Corners of the moving half, not points along one edge: four points spanning an area
        # determine a rotation; four collinear points do not.
        self.keypoint_idx = corner_indices(c.resolution, c.fold_axis)

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
                ("newton:particleRadius", c.particle_radius()),
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
        self.cloth = cloth
        # The reset write (write_sheet): the rest grid in LOCAL coordinates (z = 0) -- the spawn
        # height is added there, so baking it in here too would place the sheet at 2 x spawn_z
        # (which it did, at 1.06 m instead of 0.53 m) -- and the configured centre shift.
        self._rest_t = torch.tensor(verts, device=self.device, dtype=torch.float32)
        self._spawn_z = spawn_z
        self._reset_offset = torch.tensor(
            [float(c.reset_offset_xy[0]), float(c.reset_offset_xy[1])], device=self.device, dtype=torch.float32
        )
        self._write_object_scale()

        print(
            f"[cloth] {c.resolution}x{c.resolution} sheet, {c.size:.3f} m, "
            f"{c.num_particles()} particles, {len(indices) // 3} triangles, "
            f"keypoints {self.keypoint_idx}",
            flush=True,
        )
