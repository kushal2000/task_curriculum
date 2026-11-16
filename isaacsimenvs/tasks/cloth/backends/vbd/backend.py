"""Newton VBD cloth adapter. Isaac Lab owns the coupled physics stepping."""

from __future__ import annotations

import torch

from ...mdp.geometry import grid, rotations
from ..base import ClothBackend, as_torch
from ..robot import ROBOT
from . import contacts, solver
from . import scene as deformable_scene


class VbdBackend(ClothBackend):
    quaternion_order = "xyzw"

    def __init__(self, cfg, *, robot_spec=ROBOT):
        super().__init__(cfg, robot_spec=robot_spec)
        cfg.deformable_cloth.check_thickness()
        if cfg.task.mode == "mixed":
            raise NotImplementedError("task.mode=mixed: the VBD cloth runs fold or unfold")
        solver.configure(cfg, robot_spec=robot_spec)
        self._touch_cache = None

    def startup(self):
        return solver.startup(self, self.cfg, robot_spec=self.robot_spec)

    def finish_setup(self):
        solver.finish_setup(self, self.cfg, robot_spec=self.robot_spec)
        cloth = self.cfg.deformable_cloth
        print(f"[cloth] VBD {cloth.resolution}x{cloth.resolution}, {cloth.size:.3f} m", flush=True)

    def cloth_points(self):
        data = self._cloth.data
        return (
            as_torch(data.nodal_pos_w).view(self.num_envs, -1, 3),
            as_torch(data.nodal_vel_w).view(self.num_envs, -1, 3),
        )

    def _setup_scene(self) -> None:
        cloth = self.cfg.deformable_cloth
        dims = torch.tensor([[cloth.size, cloth.size, 2.0 * cloth.particle_radius]], device=self.device)
        self._dims = dims.expand(self.num_envs, 3).clone()
        rest = grid.grid_points(cloth.size, cloth.size, cloth.resolution)
        deformable_scene.build(self, self.robot_spec, rest, grid.grid_triangles(cloth.resolution))

    def _point_resolution(self) -> int:
        return int(self.cfg.deformable_cloth.resolution)

    def _ply_gap(self) -> torch.Tensor:
        """Two plies settle one SELF-contact radius apart (measured 2.5 mm at 2 mm); the particle
        radius is the cloth-RIGID offset and has no say in a fold."""
        return torch.full((self.num_envs,), float(self.cfg.deformable_cloth.self_contact_radius), device=self.device)

    def _write_cloth(self, ids, xy, yaw, angle) -> None:
        """The cloth flat (or the + half turned up by ``angle``), yawed, ``deformable_cloth.start_height`` above
        this episode's table centre -- it drops onto the table -- every particle at rest."""
        local = grid.folded_start(self.cloth_grid, angle, ids)
        world = rotations.apply(rotations.yaw(yaw).unsqueeze(1), local)
        centre = torch.zeros(ids.numel(), 3, device=self.device)
        centre[:, :2] = xy + torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        centre[:, 2] = self._table_z[ids] + float(self.cfg.deformable_cloth.start_height)
        target = world + (centre + self.scene.env_origins[ids]).unsqueeze(1)
        self._cloth.write_nodal_pos_to_sim_index(target, ids)
        self._cloth.write_nodal_velocity_to_sim_index(torch.zeros_like(target), ids)

    def _randomize(self, ids) -> None:
        """Per reset: the reset cloths' plastic set back to the build-time rest angles."""
        if getattr(self, "_plasticity", None) is None:
            return
        import warp as wp

        edges = torch.nonzero(torch.isin(self._edge_env, ids.to(self._edge_env.device))).reshape(-1)
        self._plasticity.reset(wp.from_torch(edges.to(torch.int32)))

    def hand_cloth_touching(self) -> torch.Tensor:
        """Whether each environment's hand touches its cloth, using the solver's contact gaps."""
        return contacts.hand_cloth_touching(self)
