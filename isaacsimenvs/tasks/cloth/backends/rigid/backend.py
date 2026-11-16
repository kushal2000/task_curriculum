"""Two rigid plates and a hinge, represented as common cloth surface samples."""

from __future__ import annotations

import numpy as np
import torch

from ...mdp.geometry import rotations
from ..base import ClothBackend, place_in_region
from ..robot import ROBOT
from . import designs, randomization
from . import scene as rigid_scene


class RigidBackend(ClothBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.task.release.touch == "contact":
            raise ValueError("task.release.touch=contact reads the VBD solver's contacts; use origin on PhysX")
        s = cfg.rigid_cloth
        grid = designs.design_grid(
            int(s.design_levels), tuple(s.length_range), tuple(s.width_range), tuple(s.thickness_range)
        )
        n = int(cfg.scene.num_envs)
        self._n_designs = len(grid)
        self._design_np = designs.design_of(np.arange(n), self._n_designs)
        self._dims_np = grid[self._design_np]  # (N, 3) length, width, thickness

    def finish_setup(self):
        s = self.cfg.rigid_cloth
        rigid_scene.set_base_materials(self, ROBOT)
        randomization.rigid_init(self)
        d = self._dims_np
        print(
            f"[rigid_body] {self._n_designs} designs: length {d[:, 0].min() * 100:.0f}-{d[:, 0].max() * 100:.0f} cm, "
            f"width {d[:, 1].min() * 100:.0f}-{d[:, 1].max() * 100:.0f} cm, thickness {d[:, 2].min() * 1e3:.0f}-"
            f"{d[:, 2].max() * 1e3:.0f} mm; placed inside table x {list(s.place_x)}, y {list(s.place_y)} m",
            flush=True,
        )

    def _setup_scene(self) -> None:
        self._dims = torch.tensor(self._dims_np, dtype=torch.float32, device=self.device)
        rigid_scene.build_robot_and_table(self, ROBOT)
        rigid_scene.author_rigid_cloths(self.num_envs, self._dims_np, self.cfg.rigid_cloth, self._spawn_z)
        self._rigid_cloth = rigid_scene.rigid_cloth_articulation(self.cfg.rigid_cloth, self._spawn_z(0))
        self.scene.articulations["cloth"] = self._rigid_cloth

    def _spawn_z(self, i: int) -> float:
        """Nominal env-frame height of env ``i``'s cloth root at rest on the table."""
        top = float(self.cfg.reset.table_reset_z) + self._table_half_height
        return top + 0.5 * float(self._dims_np[i, 2]) + float(self.cfg.rigid_cloth.spawn_clearance)

    def _point_resolution(self) -> int:
        return int(self.cfg.rigid_cloth.point_resolution)

    def _plate_index(self) -> torch.Tensor:
        """``(P,)`` body index carrying each tracked point: ``x > 0`` the + plate, the rest the -."""
        if getattr(self, "_point_body", None) is None:
            names = list(self._rigid_cloth.body_names)
            ids = torch.tensor([names.index(p) for p in rigid_scene.PLATES], device=self.device)
            on_plus = self.cloth_grid.rest[0, :, 0] > 1e-9
            self._point_body = ids[on_plus.long()]
        return self._point_body

    def _cloth_start_xy(self, ids, yaw) -> torch.Tensor:
        s, r = self.cfg.rigid_cloth, self.cfg.reset
        lo = torch.tensor([s.place_x[0], s.place_y[0]], device=self.device, dtype=torch.float32)
        hi = torch.tensor([s.place_x[1], s.place_y[1]], device=self.device, dtype=torch.float32)
        pinned = r.reset_position_noise_x == 0.0 and r.reset_position_noise_y == 0.0
        return place_in_region(self._dims[ids], yaw, lo, hi, pinned)

    def _write_cloth(self, ids, xy, yaw, angle) -> None:
        n = ids.numel()
        top = self._table_z[ids] + self._table_half_height
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, :2] = xy + torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        pos[:, 2] = top + 0.5 * self._dims[ids, 2] + float(self.cfg.rigid_cloth.spawn_clearance)
        pose = torch.cat([pos + self.scene.env_origins[ids], rotations.yaw(yaw)], dim=-1)
        self._rigid_cloth.write_root_pose_to_sim(pose, env_ids=ids)
        self._rigid_cloth.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=ids)
        q = angle.view(-1, 1)
        self._rigid_cloth.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self._rigid_cloth.set_joint_position_target(torch.zeros_like(q), env_ids=ids)

    def _randomize(self, ids) -> None:
        if hasattr(self, "_plate_mats"):  # after randomization.rigid_init
            randomization.rigid_randomize(self, ids)

    def viewer_plates(self, env_id: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """``(design, vertices (8, 3) world, triangles (4, 3), moving (4,))``: the two plates'
        mid-plane rectangles at their live poses."""
        length, width, _ = (float(v) for v in self._dims_np[env_id])
        sh = self._rigid_cloth.data
        names = list(self._rigid_cloth.body_names)
        verts, tris, moving = [], [], []
        for k, (name, x0, x1) in enumerate((("minus", -length / 2, 0.0), ("plus", 0.0, length / 2))):
            b = names.index(name)
            rect = torch.tensor(
                [(x0, -width / 2, 0.0), (x1, -width / 2, 0.0), (x1, width / 2, 0.0), (x0, width / 2, 0.0)],
                device=self.device,
            )
            verts.append(
                sh.body_link_pos_w[env_id, b] + rotations.apply(sh.body_link_quat_w[env_id, b].expand(4, 4), rect)
            )
            tris += [(4 * k, 4 * k + 1, 4 * k + 2), (4 * k, 4 * k + 2, 4 * k + 3)]
            moving += [name == "plus"] * 2
        v = torch.cat(verts).detach().cpu().numpy()
        return int(self._design_np[env_id]), v, np.asarray(tris, dtype=int), np.asarray(moving, dtype=bool)

    def cloth_points(self):
        data = self._rigid_cloth.data
        ids = self._plate_index()
        offset = rotations.apply(data.body_link_quat_w[:, ids], self.cloth_grid.rest)
        points = data.body_link_pos_w[:, ids] + offset
        velocity = data.body_link_lin_vel_w[:, ids] + torch.cross(data.body_link_ang_vel_w[:, ids], offset, dim=-1)
        return points, velocity
