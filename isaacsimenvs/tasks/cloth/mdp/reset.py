"""Where an episode starts: the table, the sheet, and the task's per-episode trackers.
(The robot's start is ``robot.reset_robot``, the DR draws ``randomization.reset_dr``.)

* table: its height jittered by ``table_reset_z_range``, optionally shifted and yawed
  (``table_reset_xy_range_m``, ``table_reset_yaw_range_deg``);
* sheet: flat, at ``reset_position_noise_{x,y}`` around the env centre, at a uniformly random
  heading. The draw is a rigid-object pose (a uniform rotation); the sheet keeps only its yaw (see
  ``backends/adapter.task_yaw``). ``fixed_start_pose`` pins it (the evaluation protocol does).
"""

from __future__ import annotations

import torch

from .geometry import quat

__all__ = ["reset_sheet", "reset_table", "reset_task_trackers"]


def reset_table(env, env_ids: torch.Tensor) -> None:
    """Draw and write the table pose; ``env._table_z_per_env`` keeps its height."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    dz = torch.empty(n, device=env.device).uniform_(-cfg.table_reset_z_range, cfg.table_reset_z_range)
    table_z = cfg.table_reset_z + dz
    env._table_z_per_env[env_ids] = table_z

    pos = torch.zeros(n, 3, device=env.device)
    pos[:, 2] = table_z
    half_x, half_y = (float(v) for v in cfg.table_reset_xy_range_m)
    if half_x > 0.0 or half_y > 0.0:
        pos[:, 0] = torch.empty(n, device=env.device).uniform_(-half_x, half_x)
        pos[:, 1] = torch.empty(n, device=env.device).uniform_(-half_y, half_y)

    yaw_range_deg = float(cfg.table_reset_yaw_range_deg)
    if yaw_range_deg > 0.0:
        half_yaw = torch.empty(n, device=env.device).uniform_(-1.0, 1.0) * yaw_range_deg * (torch.pi / 180.0) * 0.5
        zero = torch.zeros_like(half_yaw)
        rot = torch.stack([torch.cos(half_yaw), zero, zero, torch.sin(half_yaw)], dim=-1)
    else:
        rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(n, -1)

    env.table.write_root_pose_to_sim(torch.cat([pos + env.scene.env_origins[env_ids], rot], dim=-1), env_ids=env_ids)


def reset_sheet(env, env_ids: torch.Tensor) -> None:
    """Draw and write the sheet's start pose; ``env._object_init_z`` keeps the drawn height (the
    reference of the lift latch in ``rewards``)."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    if cfg.fixed_start_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_start_pose, device=env.device, dtype=torch.float32)
        pos = fixed[:3].unsqueeze(0).expand(n, -1)
        rot = fixed[3:].unsqueeze(0).expand(n, -1)
    else:
        noise = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
        pos = torch.stack(
            (
                noise[:, 0] * cfg.reset_position_noise_x,
                noise[:, 1] * cfg.reset_position_noise_y,
                env._table_z_per_env[env_ids] + cfg.table_object_z_offset + noise[:, 2] * cfg.reset_position_noise_z,
            ),
            dim=-1,
        )
        rot = quat.random_orientation(n, device=env.device)
    env.object.write_root_pose_to_sim(torch.cat([pos + env.scene.env_origins[env_ids], rot], dim=-1), env_ids=env_ids)
    env.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    env._object_init_z[env_ids] = pos[:, 2]


def reset_task_trackers(env, env_ids: torch.Tensor) -> None:
    """Per-episode task state back to its start: successes (kept as the finished episode's count),
    the best-so-far trackers (-1 = not observed yet), the lift latch and the fold counters."""
    env._prev_episode_successes[env_ids] = env._successes[env_ids]
    env._successes[env_ids] = 0
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0
    env._lifted_object[env_ids] = False
    env._fold_hold[env_ids] = 0
    env._closest_fold_err[env_ids] = -1.0
    env._fold_bonus_mask[env_ids] = False
