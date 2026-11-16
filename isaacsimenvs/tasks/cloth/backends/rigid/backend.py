"""Build a mixed-design PhysX cloth batch and expose one shared surface readout.

Creation follows the first four backend methods:
    __init__      load designs and assign topology and dimensions to environments
    _setup_scene  spawn robot/table, then one cloth articulation per design
    initialize    bind actual PhysX row order and the common surface samples
    finish_setup  set materials and draw the initial physics randomization

ClothEnv drives this lifecycle; DirectRLEnv advances physics. Scene adapters
translate global reset IDs into each articulation's local rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from ...mdp.geometry import rotations
from ...mdp.geometry.placement import placement_bounds, constrain_yaw
from ..base import ClothBackend, place_in_region
from ..robot import ROBOT
from . import designs, randomization
from . import scene as rigid_scene


@dataclass
class RigidGroup:
    """One topology's real articulation and its mapping into the RL batch."""

    index: int
    design: designs.ClothDesign
    env_ids: torch.Tensor
    articulation: object = None
    scene_asset: object = None
    body_ids: list[int] = field(default_factory=list)
    joint_ids: list[int] = field(default_factory=list)
    fold_joint_id: int = 0
    point_body_ids: torch.Tensor | None = None
    keypoint_body_ids: torch.Tensor | None = None
    local_from_global: torch.Tensor | None = None
    plate_mats: torch.Tensor | None = None

    def select(self, global_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions in a requested global-ID list and corresponding local asset rows."""
        local = self.local_from_global[global_ids]
        positions = torch.nonzero(local >= 0, as_tuple=False).flatten()
        return positions, local[positions]


class RigidBackend(ClothBackend):
    """Requested YAML designs are assigned round-robin; size sampling is independent."""

    def __init__(self, cfg, design_paths, *, robot_spec=ROBOT):
        super().__init__(cfg, robot_spec=robot_spec)
        if cfg.task.release.touch == "contact":
            raise ValueError("task.release.touch=contact reads the VBD solver's contacts; use origin on PhysX")
        if isinstance(design_paths, (str, bytes)) or not design_paths:
            raise ValueError("rigid_cloth.designs must be a nonempty list of design YAML paths")
        loaded = [designs.get_design(path) for path in design_paths]
        num_envs = int(cfg.scene.num_envs)
        # A repeated path weights its allocation, but equal definitions share one asset.
        unique = []
        requested = []
        for design in loaded:
            if design not in unique:
                unique.append(design)
            requested.append(unique.index(design))
        assignment = np.asarray(requested)[np.arange(num_envs) % len(requested)]
        self.groups = [
            RigidGroup(index, design, torch.as_tensor(np.flatnonzero(assignment == index), dtype=torch.long))
            for index, design in enumerate(unique) if np.any(assignment == index)
        ]

        cloth_cfg = cfg.rigid_cloth
        sizes = designs.design_grid(
            int(cloth_cfg.design_levels), tuple(cloth_cfg.length_range),
            tuple(cloth_cfg.width_range), tuple(cloth_cfg.thickness_range),
        )
        self._num_sizes = len(sizes)
        self._size_ids = designs.design_of(np.arange(num_envs), self._num_sizes)
        self._dimensions_cpu = sizes[self._size_ids]

    def _setup_scene(self) -> None:
        """Spawn the shared robot/table batch and a real articulation for each topology."""
        self._dims = torch.tensor(self._dimensions_cpu, dtype=torch.float32, device=self.device)
        rigid_scene.build_robot_and_table(self, self.robot_spec)
        for group in self.groups:
            assigned = group.env_ids.numpy()
            root_name = f"RigidCloth_{group.index}"
            rigid_scene.author_rigid_cloths(
                assigned, self._dimensions_cpu[assigned], self.cfg.rigid_cloth,
                self._spawn_z, group.design, root_name,
            )
            group.articulation = rigid_scene.rigid_cloth_articulation(
                self.cfg.rigid_cloth, self._spawn_z(int(assigned[0])), group.design, root_name,
            )
            group.env_ids = group.env_ids.to(self.device)
            group.scene_asset = rigid_scene.ArticulationGroup(group.articulation, group.env_ids, self.num_envs)
            self.scene.articulations[f"cloth_{group.index}"] = group.scene_asset

    def initialize(self) -> None:
        """Bind asset row order, named bodies/joints, and normalized surface ownership."""
        super().initialize()
        self._env_group = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        self._env_local = torch.empty_like(self._env_group)
        normalized = (self.cloth_grid.rest[0] / self._dims[0]).detach().cpu().numpy()
        keypoints = torch.tensor(designs.MOVING_KEYPOINTS, device=self.device, dtype=self._dims.dtype)
        self.cloth_grid.keypoint_rest = keypoints[None] * self._dims[:, None]
        for group_index, group in enumerate(self.groups):
            asset = group.articulation
            # Sparse PhysX views need not be sorted by global environment index.
            paths = asset.root_physx_view.prim_paths
            actual = []
            for path in paths:
                match = re.search(r"/env_(\d+)(?:/|$)", str(path))
                if match is None:
                    raise RuntimeError(f"cloth articulation path has no environment index: {path}")
                actual.append(int(match.group(1)))
            if len(actual) != len(set(actual)) or set(actual) != set(group.env_ids.tolist()):
                raise RuntimeError(f"cloth group {group.index} PhysX rows do not match its assigned environments")
            group.env_ids = torch.tensor(actual, dtype=torch.long, device=self.device)
            group.scene_asset.set_env_ids(group.env_ids)
            group.local_from_global = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            group.local_from_global[group.env_ids] = torch.arange(len(actual), device=self.device)
            self._env_group[group.env_ids] = group_index
            self._env_local[group.env_ids] = torch.arange(len(actual), device=self.device)
            group.body_ids = [asset.body_names.index(plate.name) for plate in group.design.plates]
            group.joint_ids = [asset.joint_names.index(name) for name in group.design.joint_names]
            group.fold_joint_id = group.joint_ids[group.design.fold_hinge]
            owners = torch.as_tensor(group.design.point_owners(normalized), device=self.device)
            bodies = torch.tensor(group.body_ids, device=self.device)
            group.point_body_ids = bodies[owners]
            moving = [i for i, plate in enumerate(group.design.plates) if plate.center[0] > 0.0]
            keypoint_owners = designs.point_owners(group.design, designs.MOVING_KEYPOINTS, plate_ids=moving)
            group.keypoint_body_ids = bodies[torch.as_tensor(keypoint_owners, device=self.device)]

    def finish_setup(self) -> None:
        """PhysX views are now ready for material and hinge-property writes."""
        randomization.rigid_init(self)
        summary = ", ".join(f"{group.design.name}: {len(group.env_ids)} envs" for group in self.groups)
        print(f"[rigid_body] {summary}; {self._num_sizes} independently assigned sizes", flush=True)

    def _spawn_z(self, env_id: int) -> float:
        table_top = float(self.cfg.reset.table_reset_z) + self._table_half_height
        return table_top + 0.5 * float(self._dimensions_cpu[env_id, 2]) + float(self.cfg.rigid_cloth.spawn_clearance)

    def _point_resolution(self) -> int:
        return int(self.cfg.rigid_cloth.point_resolution)

    def cloth_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter each topology's surface samples into global RL environment order."""
        points_w = torch.empty_like(self.cloth_grid.rest)
        velocities_w = torch.empty_like(points_w)
        for group in self.groups:
            data, bodies = group.articulation.data, group.point_body_ids
            # Every body frame uses the sheet's rest origin, including polygon plates.
            offset_w = rotations.apply(data.body_link_quat_w[:, bodies], self.cloth_grid.rest[group.env_ids])
            points_w[group.env_ids] = data.body_link_pos_w[:, bodies] + offset_w
            velocities_w[group.env_ids] = data.body_link_lin_vel_w[:, bodies] + torch.cross(
                data.body_link_ang_vel_w[:, bodies], offset_w, dim=-1
            )
        return points_w, velocities_w

    def cloth_keypoints(self) -> torch.Tensor:
        """Read the exact moving-half corners from their physical plate link frames."""
        rest = self.cloth_grid.keypoint_rest
        points_w = torch.empty_like(rest)
        for group in self.groups:
            data, bodies = group.articulation.data, group.keypoint_body_ids
            points_w[group.env_ids] = data.body_link_pos_w[:, bodies] + rotations.apply(
                data.body_link_quat_w[:, bodies], rest[group.env_ids]
            )
        return points_w

    def _cloth_start_xy(self, ids, yaw) -> torch.Tensor:
        cloth_cfg, reset_cfg = self.cfg.rigid_cloth, self.cfg.reset
        lo, hi = placement_bounds(cloth_cfg.place_x, cloth_cfg.place_y, self._table_size)
        lower, upper = self._dims.new_tensor(lo), self._dims.new_tensor(hi)
        yaw.copy_(constrain_yaw(self._dims[ids], yaw, lower, upper,
                               reset_cfg.yaw_range_deg, randomize=reset_cfg.random_yaw))
        pinned = reset_cfg.reset_position_noise_x == 0.0 and reset_cfg.reset_position_noise_y == 0.0
        return place_in_region(self._dims[ids], yaw, lower, upper, pinned)

    def _write_cloth(self, ids, xy, yaw, angle) -> None:
        """Reset selected global environments, folding only their designated center joint."""
        position = torch.zeros(ids.numel(), 3, device=self.device)
        position[:, :2] = xy + torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        position[:, 2] = (
            self._table_z[ids] + self._table_half_height + 0.5 * self._dims[ids, 2]
            + float(self.cfg.rigid_cloth.spawn_clearance)
        )
        pose_w = torch.cat([position + self.scene.env_origins[ids], rotations.yaw(yaw)], dim=-1)
        for group in self.groups:
            rows, local_ids = group.select(ids)
            if not rows.numel():
                continue
            asset = group.articulation
            asset.write_root_pose_to_sim(pose_w[rows], env_ids=local_ids)
            asset.write_root_velocity_to_sim(torch.zeros(rows.numel(), 6, device=self.device), env_ids=local_ids)
            q = torch.zeros(rows.numel(), asset.num_joints, device=self.device)
            q[:, group.fold_joint_id] = angle[rows]
            asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=local_ids)
            asset.set_joint_position_target(torch.zeros_like(q), env_ids=local_ids)
            asset.set_joint_velocity_target(torch.zeros_like(q), env_ids=local_ids)

    def _randomize(self, ids) -> None:
        if hasattr(self, "_dr_density"):
            randomization.rigid_randomize(self, ids)

    def viewer_plates(self, env_id: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Actual polygon surfaces, transformed by their bodies, for one global environment."""
        group = self.groups[int(self._env_group[env_id])]
        local = int(self._env_local[env_id])
        spec = group.design.resolve(*(float(value) for value in self._dimensions_cpu[env_id]))
        data = group.articulation.data
        vertices, triangles, moving = [], [], []
        offset = 0
        for plate, body in zip(spec.plates, group.body_ids):
            rest, faces = designs.surface_mesh(plate)
            rest = torch.as_tensor(rest, dtype=data.body_link_pos_w.dtype, device=self.device)
            vertices.append(
                data.body_link_pos_w[local, body]
                + rotations.apply(data.body_link_quat_w[local, body].expand(len(rest), 4), rest)
            )
            triangles.append(faces + offset)
            moving.extend([plate.center[0] > 0.0] * len(faces))
            offset += len(rest)
        shape_id = group.index * self._num_sizes + int(self._size_ids[env_id])
        return (
            shape_id, torch.cat(vertices).detach().cpu().numpy(),
            np.concatenate(triangles), np.asarray(moving, dtype=bool),
        )
