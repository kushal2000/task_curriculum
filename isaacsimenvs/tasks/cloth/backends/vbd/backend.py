"""``VbdBackend``: the fold task's sheet as a VBD cloth, on Isaac Lab 3.0 + Newton.

The env is constructed on Newton in two windows (``newton_setup.py``): the patches that must be in
place before the scene is built -- the coupled MJWarp (robot) + VBD (cloth) solver among them --
installed by ``__init__``, and the ones that need live assets, by :meth:`finish_setup`.

The sheet is authored as a flat ``UsdGeom.Mesh`` grid before the environments are replicated --
``DeformableObject`` infers ``surface`` (cloth) from a ``UsdGeom.Mesh`` and ``volume`` from a
``UsdGeom.TetMesh``, so the prim type *is* the switch. Its grid points are the particles.
"""

from __future__ import annotations

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import patches  # noqa: F401  (first: the Isaac Lab 3.0 shims)

import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend, reset_yaw
from isaacsimenvs.tasks.cloth.backends.vbd.materials import configure_friction, configure_materials, install_bending
from isaacsimenvs.tasks.cloth.backends.vbd.newton_setup import (
    install_post_construction_patches,
    install_pre_construction_patches,
)
from isaacsimenvs.tasks.cloth.backends.vbd.physics import build_physics
from isaacsimenvs.tasks.cloth.config.schema.env import ClothDRCfg
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
        self.mesh = cfg.cloth.sheet_mesh()
        self._plasticity = None
        self._material_solver = None
        install_pre_construction_patches(
            cfg,
            # Coupled MJWarp (robot) + VBD (cloth), sized for this env count.
            physics_cfg=build_physics(cfg.cloth, cfg.newton, int(cfg.scene.num_envs), robot_spec),
        )

    def construction_context(self):
        return configure_materials(self)

    def finish_setup(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        self._install_solver_materials(NewtonManager._model, NewtonManager._solver)
        install_post_construction_patches(self)

    def _install_solver_materials(self, model, solver) -> None:
        if self._material_solver is solver:
            return
        c = self.cfg.cloth
        entry = solver._entries["cloth"]
        if c.table_contact_mu > 0 or c.self_contact_mu > 0:
            configure_friction(solver, entry.view, c)
        if c.bending_rigidity > 0 or c.plastic_yield_curvature > 0:
            dt = self.cfg.sim.dt / self.cfg.newton.num_substeps / c.substeps
            self._plasticity = install_bending(model, entry, c, dt)
            hinges = model.edge_indices.numpy()
            world = model.particle_world.numpy()[hinges[:, 2]]
            world[(hinges[:, 0] < 0) | (hinges[:, 1] < 0)] = -1
            self._edge_world = torch.as_tensor(world, dtype=torch.long, device=str(model.device))
        self._material_solver = solver

    def end_reset(self, env_ids: torch.Tensor) -> None:
        if self._plasticity is not None:
            import warp as wp

            edges = torch.isin(self._edge_world, env_ids).nonzero().flatten().to(torch.int32)
            if edges.numel():
                self._plasticity.reset(wp.from_torch(edges))

    # ------------------------------------------------------------------ the sheet, for the task

    def goal_marker_mesh(self):
        return self.mesh.half_mesh()

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
            moving_idx=self.mesh.moving_idx,
            stationary_idx=self.mesh.stationary_idx,
            moving_triangles=self.mesh.half_mesh()[1],
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

    def spawn_ground(self) -> None:
        from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg

        # The VBD scene runs without Kit or Nucleus. A local static box has its top at z=0.
        ground = CuboidCfg(size=(100.0, 100.0, 0.1), collision_props=CollisionPropertiesCfg())
        ground.func("/World/ground", ground, translation=(0.0, 0.0, -0.05))

    def install_sheet(self) -> None:
        """Author the sheet and register it as a deformable.

        Runs at the end of the scene setup, before replication -- which is also when Newton imports
        the stage: a mesh authored after that point is invisible to the solver."""
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Sdf, UsdGeom, UsdShade

        c = self.cfg.cloth
        verts, indices = self.mesh.verts, self.mesh.indices
        self.rest_local = verts
        # Corners of the moving half, not points along one edge: four points spanning an area
        # determine a rotation; four collinear points do not.
        self.keypoint_idx = self.mesh.corner_idx

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
            f"[cloth] {c.mesh_kind} sheet, {c.size:.3f} m, "
            f"{c.num_particles()} particles, {len(indices) // 3} triangles, "
            f"keypoints {self.keypoint_idx}",
            flush=True,
        )
