"""The fold task on a rigid-body sheet: two plates and one hinge, one size design per env (PhysX).

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv/isaacsim``). The task -- hooks, obs, reward, success -- is
``utils/task_env.ClothTaskEnv``, shared with the VBD cloth env; this file is the PhysX side only:
the scene, reading a ``ClothState`` from it, and the reset writes.

    OMNI_KIT_ACCEPT_EULA=YES .venv/isaacsim/bin/python scripts/training/train.py --headless \\
        --task Isaacsimenvs-RigidBodyCloth-Direct-v0 --experiment rigid_body_cloth

**The sheet** (``utils/sheet_designs.single_hinge``): two boxes ``length/2 x width x thickness``
joined by a revolute hinge on the top surface at the crease, so folding the ``+x`` plate 180 deg
lays it one thickness above the ``-x`` plate: the fold target (reflect across the crease, lift by
the ply gap = the thickness). Env ``i`` carries design ``perm[i % n_designs]`` of the grid. The
task reads the sheet as a ``point_resolution`` grid of points carried by the two plates.

**Per-reset DR** (``utils/dr_utils.py``): hinge damping / stiffness and plate / robot / table friction.
"""

from __future__ import annotations

import numpy as np
import torch

from isaacsimenvs.tasks.cloth.rigid_body_env_cfg import RigidBodyClothEnvCfg
from isaacsimenvs.tasks.cloth.utils import dr_utils, quat, reset_utils, scene_physx, sheet_designs
from isaacsimenvs.tasks.cloth.utils.robot import ROBOT
from isaacsimenvs.tasks.cloth.utils.state import ClothState
from isaacsimenvs.tasks.cloth.utils.task_env import ClothTaskEnv

__all__ = ["RigidBodyClothEnv"]


class RigidBodyClothEnv(ClothTaskEnv):
    """Fold the ``+x`` plate of a rigid single-hinge sheet onto the ``-x`` plate."""

    cfg: RigidBodyClothEnvCfg

    def __init__(self, cfg: RigidBodyClothEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.fold.release.touch == "contact":
            raise ValueError("fold.release.touch=contact reads the VBD solver's contacts; use origin on PhysX")
        s = cfg.sheet
        grid = sheet_designs.design_grid(int(s.design_levels), tuple(s.length_range), tuple(s.width_range),
                                         tuple(s.thickness_range))
        n = int(cfg.scene.num_envs)
        self._n_designs = len(grid)
        self._design_np = sheet_designs.design_of(np.arange(n), self._n_designs)
        self._dims_np = grid[self._design_np]                       # (N, 3) length, width, thickness
        super().__init__(cfg, render_mode, **kwargs)
        scene_physx.set_base_materials(self, ROBOT)
        dr_utils.rigid_init(self)
        d = self._dims_np
        print(f"[rigid_body] {self._n_designs} designs: length {d[:, 0].min() * 100:.0f}-{d[:, 0].max() * 100:.0f} cm, "
              f"width {d[:, 1].min() * 100:.0f}-{d[:, 1].max() * 100:.0f} cm, thickness {d[:, 2].min() * 1e3:.0f}-"
              f"{d[:, 2].max() * 1e3:.0f} mm; placed inside table x {list(s.place_x)}, y {list(s.place_y)} m", flush=True)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self) -> None:
        from isaacsimenvs.utils.setup_timer import stamp

        stamp("scene: start")
        self._dims = torch.tensor(self._dims_np, dtype=torch.float32, device=self.device)
        scene_physx.build_robot_and_table(self, ROBOT)
        stamp("scene: robot + table")
        scene_physx.author_sheets(self.num_envs, self._dims_np, self.cfg.sheet, self._spawn_z)
        self._sheet = scene_physx.sheet_articulation(self.cfg.sheet, self._spawn_z(0))
        self.scene.articulations["sheet"] = self._sheet
        stamp(f"scene: {self.num_envs} sheets authored")

    def _spawn_z(self, i: int) -> float:
        """Nominal env-frame height of env ``i``'s sheet root at rest on the table."""
        top = float(self.cfg.reset.table_reset_z) + self._table_half_height
        return top + 0.5 * float(self._dims_np[i, 2]) + float(self.cfg.sheet.spawn_clearance)

    def _point_resolution(self) -> int:
        return int(self.cfg.sheet.point_resolution)

    # ------------------------------------------------------------------ reads

    def _plate_index(self) -> torch.Tensor:
        """``(P,)`` body index carrying each tracked point: ``x > 0`` the + plate, the rest the -."""
        if getattr(self, "_point_body", None) is None:
            names = list(self._sheet.body_names)
            ids = torch.tensor([names.index(p) for p in scene_physx.PLATES], device=self.device)
            on_plus = self._sheet_geom.rest[0, :, 0] > 1e-9
            self._point_body = ids[on_plus.long()]
        return self._point_body

    def _read_state(self) -> ClothState:
        r, sh = self.robot.data, self._sheet.data
        idx = self._plate_index()
        bq = sh.body_link_quat_w[:, idx]
        local = quat.apply(bq, self._sheet_geom.rest)
        points = sh.body_link_pos_w[:, idx] + local
        vel = sh.body_link_lin_vel_w[:, idx] + torch.cross(sh.body_link_ang_vel_w[:, idx], local, dim=-1)
        body_pos, body_quat = r.body_link_pos_w, r.body_link_quat_w
        return ClothState(
            joint_pos=r.joint_pos, joint_vel=r.joint_vel,
            palm_pos_w=body_pos[:, self._palm_id], palm_quat_w=body_quat[:, self._palm_id],
            palm_vel_w=torch.cat([r.body_link_lin_vel_w[:, self._palm_id], r.body_link_ang_vel_w[:, self._palm_id]], dim=-1),
            fingertip_pos_w=body_pos[:, self._fingertip_ids], fingertip_quat_w=body_quat[:, self._fingertip_ids],
            hand_pos_w=body_pos[:, self._hand_body_ids],
            points_w=points, points_vel_w=vel,
            table_top_z=self.scene.env_origins[:, 2] + self._table_z + self._table_half_height,
        )

    def _read_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    def _robot_defaults(self):
        d = self.robot.data
        return list(d.joint_names), d.default_joint_pos.clone(), d.joint_pos_limits.clone()

    def _body_ids(self, names: list[str]) -> list[int]:
        ids, found = self.robot.find_bodies(names, preserve_order=True)
        if list(found) != list(names):
            raise RuntimeError(f"robot bodies {names} resolved to {found}")
        return list(ids)

    def _body_ids_regex(self, regex: str) -> list[int]:
        ids, _ = self.robot.find_bodies(regex)
        if not ids:
            raise ValueError(f"fold.release.contact_bodies={regex!r} matched no robot body")
        return list(ids)

    # ------------------------------------------------------------------ writes

    def _set_robot_targets(self, pos_target, effort) -> None:
        self.robot.set_joint_position_target(pos_target)
        self.robot.set_joint_effort_target(effort)

    def _write_robot(self, ids, joint_pos, joint_vel) -> None:
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    def _write_table(self, ids, table_z) -> None:
        pose = torch.zeros(ids.numel(), 7, device=self.device)
        pose[:, :2] = torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        pose[:, 2] = table_z
        pose[:, :3] += self.scene.env_origins[ids]
        pose[:, 3] = 1.0
        self.table.write_root_pose_to_sim(pose, env_ids=ids)

    def _sheet_start_xy(self, ids, yaw) -> torch.Tensor:
        s, r = self.cfg.sheet, self.cfg.reset
        lo = torch.tensor([s.place_x[0], s.place_y[0]], device=self.device, dtype=torch.float32)
        hi = torch.tensor([s.place_x[1], s.place_y[1]], device=self.device, dtype=torch.float32)
        pinned = r.reset_position_noise_x == 0.0 and r.reset_position_noise_y == 0.0
        return reset_utils.place_in_region(self._dims[ids], yaw, lo, hi, pinned)

    def _write_sheet(self, ids, xy, yaw, angle) -> None:
        n = ids.numel()
        top = self._table_z[ids] + self._table_half_height
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, :2] = xy + torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        pos[:, 2] = top + 0.5 * self._dims[ids, 2] + float(self.cfg.sheet.spawn_clearance)
        pose = torch.cat([pos + self.scene.env_origins[ids], quat.yaw(yaw)], dim=-1)
        self._sheet.write_root_pose_to_sim(pose, env_ids=ids)
        self._sheet.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=ids)
        q = angle.view(-1, 1)
        self._sheet.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self._sheet.set_joint_position_target(torch.zeros_like(q), env_ids=ids)

    def _randomize(self, ids) -> None:
        if hasattr(self, "_sheet_mats"):   # after dr_utils.rigid_init
            dr_utils.rigid_randomize(self, ids)

    # ------------------------------------------------------------------ viewer

    def viewer_plates(self, env_id: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """``(design, vertices (8, 3) world, triangles (4, 3), moving (4,))``: the two plates'
        mid-plane rectangles at their live poses."""
        length, width, _ = (float(v) for v in self._dims_np[env_id])
        sh = self._sheet.data
        names = list(self._sheet.body_names)
        verts, tris, moving = [], [], []
        for k, (name, x0, x1) in enumerate((("minus", -length / 2, 0.0), ("plus", 0.0, length / 2))):
            b = names.index(name)
            rect = torch.tensor([(x0, -width / 2, 0.0), (x1, -width / 2, 0.0), (x1, width / 2, 0.0), (x0, width / 2, 0.0)],
                                device=self.device)
            verts.append(sh.body_link_pos_w[env_id, b] + quat.apply(sh.body_link_quat_w[env_id, b].expand(4, 4), rect))
            tris += [(4 * k, 4 * k + 1, 4 * k + 2), (4 * k, 4 * k + 2, 4 * k + 3)]
            moving += [name == "plus"] * 2
        v = torch.cat(verts).detach().cpu().numpy()
        return int(self._design_np[env_id]), v, np.asarray(tris, dtype=int), np.asarray(moving, dtype=bool)

