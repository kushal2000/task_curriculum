"""Common Isaac Lab scene access. Adapters never advance the simulation themselves."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from ..mdp.geometry import grid, rotations
from ..mdp.state import JointCommands, ResetState, SimState
from .robot import ROBOT


def as_torch(value):
    """Isaac Lab 3 ProxyArrays expose torch views; PhysX already supplies tensors."""
    return value.torch if hasattr(value, "torch") else value


def place_in_region(
    dims: torch.Tensor, yaw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, pinned: bool
) -> torch.Tensor:
    """``(n, 2)`` cloth centre: uniform over every position that keeps the yawed ``length x width``
    bounding box inside ``[lo, hi]``; centred where it does not fit, and everywhere when ``pinned``."""
    c, s = torch.cos(yaw).abs(), torch.sin(yaw).abs()
    half = torch.stack((0.5 * (dims[:, 0] * c + dims[:, 1] * s), 0.5 * (dims[:, 0] * s + dims[:, 1] * c)), dim=-1)
    room = hi - lo - 2.0 * half
    u = torch.full_like(half, 0.5) if pinned else torch.rand_like(half)
    return torch.where(room >= 0.0, lo + half + u * room.clamp(min=0.0), 0.5 * (lo + hi))


class ClothBackend:
    quaternion_order = "wxyz"

    def __init__(self, cfg):
        self.cfg = cfg

    def startup(self):
        return nullcontext()

    def build_scene(self, scene, sim):
        self.scene, self.sim = scene, sim
        self.device, self.num_envs = sim.device, scene.num_envs
        self._setup_scene()

    def initialize(self):
        self._palm_id = self._body_ids([ROBOT.palm_body_name])[0]
        self._fingertip_ids = self._body_ids(list(ROBOT.fingertip_link_names))
        pattern = self.cfg.task.release.contact_bodies
        if pattern:
            ids, _ = self.robot.find_bodies(pattern)
            if not ids:
                raise ValueError(f"task.release.contact_bodies={pattern!r} matched no robot body")
            self._hand_body_ids = list(ids)
        else:
            self._hand_body_ids = self._fingertip_ids + [self._palm_id]
        self.cloth_grid = grid.make_grid(self._dims, self._point_resolution(), self._ply_gap())
        self._table_z = torch.full((self.num_envs,), float(self.cfg.reset.table_reset_z), device=self.device)

    def _body_ids(self, names):
        ids, found = self.robot.find_bodies(names, preserve_order=True)
        if list(found) != list(names):
            raise RuntimeError(f"robot bodies {names} resolved to {found}")
        return list(ids)

    def robot_defaults(self):
        data = self.robot.data
        return list(data.joint_names), as_torch(data.default_joint_pos).clone(), as_torch(data.joint_pos_limits).clone()

    def read_joint_pos(self):
        return as_torch(self.robot.data.joint_pos)

    def read_state(self) -> SimState:
        data = self.robot.data
        bodies = as_torch(data.body_state_w)
        positions, quaternions = bodies[..., :3], bodies[..., 3:7]
        if self.quaternion_order == "xyzw":
            quaternions = rotations.to_wxyz(quaternions)
        points, velocities = self.cloth_points()
        return SimState(
            joint_pos=as_torch(data.joint_pos),
            joint_vel=as_torch(data.joint_vel),
            palm_pos_w=positions[:, self._palm_id],
            palm_quat_w=quaternions[:, self._palm_id],
            palm_vel_w=bodies[:, self._palm_id, 7:13],
            fingertip_pos_w=positions[:, self._fingertip_ids],
            fingertip_quat_w=quaternions[:, self._fingertip_ids],
            hand_pos_w=positions[:, self._hand_body_ids],
            points_w=points,
            points_vel_w=velocities,
            table_top_z=self.scene.env_origins[:, 2] + self._table_z + self._table_half_height,
        )

    def apply_joint_commands(self, commands: JointCommands):
        self.robot.set_joint_position_target(commands.position)
        self.robot.set_joint_effort_target(commands.effort)

    def reset(self, ids, initial: ResetState):
        self.robot.write_joint_state_to_sim(initial.joint_pos, initial.joint_vel, env_ids=ids)
        self._table_z[ids] = initial.table_z
        pose = torch.zeros(ids.numel(), 7, device=self.device)
        pose[:, :2] = torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        pose[:, 2] = initial.table_z
        pose[:, :3] += self.scene.env_origins[ids]
        pose[:, 6 if self.quaternion_order == "xyzw" else 3] = 1.0
        self.table.write_root_pose_to_sim(pose, env_ids=ids)
        self._write_cloth(ids, self._cloth_start_xy(ids, initial.yaw), initial.yaw, initial.fold_angle)
        self._randomize(ids)

    def _cloth_start_xy(self, ids, yaw):
        reset = self.cfg.reset
        noise = torch.tensor([reset.reset_position_noise_x, reset.reset_position_noise_y], device=self.device)
        return (2.0 * torch.rand(ids.numel(), 2, device=self.device) - 1.0) * noise

    def _ply_gap(self):
        return self._dims[:, 2]

    def _setup_scene(self):
        raise NotImplementedError

    def _point_resolution(self):
        raise NotImplementedError

    def cloth_points(self):
        raise NotImplementedError

    def _write_cloth(self, ids, xy, yaw, angle):
        raise NotImplementedError

    def _randomize(self, ids):
        pass

    def finish_setup(self):
        pass

    def hand_cloth_touching(self):
        raise ValueError("task.release.touch=contact requires the VBD backend")

    def viewer_plates(self, env_id):
        return None
