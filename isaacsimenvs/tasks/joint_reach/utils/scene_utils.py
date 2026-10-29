"""Scene geometry for joint reach: the table top, robot points, the workspace test and the parked object."""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.play.utils.obs_utils import NUM_FINGERTIPS, _apply_local_offset

__all__ = [
    "OBJECT_PARK_POS",
    "check_points",
    "park_object",
    "robot_points",
    "size_gpu_buffers",
    "table_top_and_halfsize",
]

#: Where the unused object waits: on the floor behind the table, out of the arm's reach [m, env frame].
OBJECT_PARK_POS = (0.0, -0.75, 0.1)


def size_gpu_buffers(cfg) -> None:
    """The cloth pool's per-env PhysX GPU buffers (tasks/cloth/physx_envs.py _size_gpu_buffers): Play's
    fixed defaults do not start 24576 Flexiv + Wuji envs with self-collision on."""
    px, n_env = cfg.sim.physx, int(cfg.scene.num_envs)
    px.gpu_max_rigid_patch_count = max(int(px.gpu_max_rigid_patch_count), 48 * n_env)
    px.gpu_max_rigid_contact_count = max(int(px.gpu_max_rigid_contact_count), 256 * n_env)
    px.gpu_found_lost_pairs_capacity = max(int(px.gpu_found_lost_pairs_capacity), 128 * n_env)
    px.gpu_total_aggregate_pairs_capacity = max(int(px.gpu_total_aggregate_pairs_capacity), 128 * n_env)


def table_top_and_halfsize(env) -> tuple[float, float, float]:
    """The highest table top the reset noise can place, and the table's half extents in x and y."""
    sx, sy, sz = env.cfg.goal.table_box_size
    top = env.cfg.reset.table_reset_z + env.cfg.reset.table_reset_z_range + 0.5 * sz
    return top, 0.5 * sx, 0.5 * sy


def check_points(env, palm_pos: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Workspace test per env: palm centre over the table at the allowed height, and no robot point
    over the table footprint closer to the top than the clearance. Positions in the env frame."""
    g = env.cfg.goal
    top, hx, hy = table_top_and_halfsize(env)
    m = g.palm_footprint_margin
    palm_ok = (
        (palm_pos[:, 0].abs() <= hx + m)
        & (palm_pos[:, 1].abs() <= hy + m)
        & (palm_pos[:, 2] >= top + g.palm_height_range[0])
        & (palm_pos[:, 2] <= top + g.palm_height_range[1])
    )
    over = (points[..., 0].abs() <= hx + g.body_table_clearance) & (
        points[..., 1].abs() <= hy + g.body_table_clearance
    )
    too_low = over & (points[..., 2] < top + g.body_table_clearance)
    return palm_ok & ~too_low.any(dim=-1)


def robot_points(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Palm centre (N, 3) and every body origin plus fingertip pad (N, B + 5, 3), env frame."""
    origins = env.scene.env_origins
    body_pos = env.robot.data.body_pos_w - origins.unsqueeze(1)
    palm_q = env.robot.data.body_quat_w[:, env._palm_body_id]
    palm = _apply_local_offset(
        env.robot.data.body_pos_w[:, env._palm_body_id], palm_q, env.robot_spec.palm_center_offset,
        (env.num_envs,),
    ) - origins
    ft_q = env.robot.data.body_quat_w[:, env._fingertip_body_ids]
    pads = _apply_local_offset(
        env.robot.data.body_pos_w[:, env._fingertip_body_ids], ft_q, env.robot_spec.fingertip_offset,
        (env.num_envs, NUM_FINGERTIPS),
    ) - origins.unsqueeze(1)
    return palm, torch.cat([body_pos, pads], dim=1)


def park_object(env, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :3] = torch.tensor(OBJECT_PARK_POS, device=env.device) + env.scene.env_origins[env_ids]
    pose[:, 3] = 1.0
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
