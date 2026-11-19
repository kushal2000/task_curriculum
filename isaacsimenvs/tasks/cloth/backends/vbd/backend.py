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

from isaacsimenvs.tasks.cloth.backends.adapter import ClothAsRigidObject
from isaacsimenvs.tasks.cloth.backends.base import ClothBackend
from isaacsimenvs.tasks.cloth.backends.vbd.newton_setup import (
    install_post_construction_patches,
    install_pre_construction_patches,
)
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.mdp.state import SheetLayout

__all__ = ["VbdBackend"]


class VbdBackend(ClothBackend):
    """A VBD cloth sheet, coupled to the robot on Newton."""

    def __init__(self, cfg, robot_spec) -> None:
        super().__init__(cfg, robot_spec)
        install_pre_construction_patches(
            cfg,
            # Coupled MJWarp (robot) + VBD (cloth), sized for this env count.
            physics_cfg=cfg.cloth.build_physics(cfg.newton, int(cfg.scene.num_envs), robot_spec),
        )

    def finish_setup(self) -> None:
        install_post_construction_patches(self)

    # ------------------------------------------------------------------ the sheet, for the task

    def sheet_layout(self) -> SheetLayout:
        c = self.cfg.cloth
        return SheetLayout(
            rest_local=self._cloth_rest_local,
            keypoint_idx=list(self._cloth_kp_idx),
            resolution=int(c.resolution),
            fold_axis=c.fold_axis,
            size=float(c.size),
            # Two plies settle one SELF-contact radius apart; the particle radius is the cloth-RIGID
            # offset and has no say in a fold (``mdp/fold.py``).
            fold_lift=c.self_contact_radius,
            object_scale=self.object_scale,
        )

    def read_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.object._particles(), self.object._velocities()

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

        # The task reads the sheet as `env.object` (mdp/observations, rewards, terminations).
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
            device=self.device,
            reset_offset_xy=c.reset_offset_xy,
        )
        self._write_object_scale()

        print(
            f"[cloth] {c.resolution}x{c.resolution} sheet, {c.size:.3f} m, "
            f"{c.num_particles} particles, {len(indices) // 3} triangles, "
            f"keypoints {self._cloth_kp_idx}",
            flush=True,
        )
