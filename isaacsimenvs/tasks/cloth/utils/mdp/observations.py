"""Observations in Play's layout, computed from the cloth.

The layout -- fields, order, sizes, joint order, normalisation, quaternion convention (xyzw), the
env frame, the ``palm_pos`` base shift -- is Play's, so a Play-pretrained policy loads and its
robot inputs mean what they meant. The object fields are the cloth's:

* ``object_rot``: the stationary half's orientation (its fitted rigid frame);
* ``object_vel``: the cloth centre's linear velocity and the points' fitted angular velocity;
* ``keypoints_rel_palm`` / ``keypoints_rel_goal``: the four tracked moving-half corners, minus the
  palm centre / minus each corner's fold target;
* ``object_scales``: [length, width, thickness] / ``reward.object_base_size``.

Object-state DR (Play's): the cloth frame AND the corners expressed in it go through one delay
queue; the delayed frame gets position and rotation noise and the corners are rebuilt from it.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.utils.geometry import rotations

from .actions import delayed
from .snapshot import SimState, TaskView

NUM_FINGERTIPS = 5
NUM_KEYPOINTS = 4

FIELD_SIZES: dict[str, int] = {
    "palm_pos": 3, "palm_rot": 4, "palm_vel": 6, "object_rot": 4, "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS, "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS, "object_scales": 3, "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS, "lifted_object": 1, "progress": 1, "successes": 1,
    "reward": 1,
}
PER_JOINT_FIELDS = ("joint_pos", "joint_vel", "prev_action_targets")


def obs_dim(fields, num_joints: int) -> int:
    return sum(num_joints if f in PER_JOINT_FIELDS else FIELD_SIZES[f] for f in fields)


def stack(obs: dict[str, torch.Tensor], fields) -> torch.Tensor:
    return torch.cat([obs[f].reshape(obs[f].shape[0], -1) for f in fields], dim=-1)


def palm_center_w(env, s: SimState) -> torch.Tensor:
    return s.palm_pos_w + rotations.apply(s.palm_quat_w, env._palm_offset.expand_as(s.palm_pos_w))


def build(env, s: SimState, v: TaskView) -> dict[str, torch.Tensor]:
    """``{"policy": (N, obs dim), "critic": (N, state dim)}``, both clamped."""
    cfg, dr = env.cfg, env.cfg.domain_randomization
    n = env.num_envs
    origins = env.scene.env_origins
    perm = env._perm_lab_to_canon

    joint_pos = s.joint_pos[:, perm]
    joint_pos = 2.0 * (joint_pos - env._joint_lower_canon) / (env._joint_upper_canon - env._joint_lower_canon) - 1.0
    joint_vel = s.joint_vel[:, perm]

    palm = palm_center_w(env, s) - origins
    palm_obs = palm - env._palm_obs_shift  # as if the base were at Play's position
    pads = s.fingertip_pos_w + rotations.apply(s.fingertip_quat_w, env._fingertip_offsets.expand_as(s.fingertip_pos_w))
    fingertip_rel_palm = pads - origins.unsqueeze(1) - palm.unsqueeze(1)

    pos = v.pos_w - origins
    kp = v.corners_w - origins.unsqueeze(1)
    targets = v.targets_w - origins.unsqueeze(1)
    vel = torch.cat([v.lin_vel_w, v.ang_vel_w], dim=-1)

    if dr.use_object_state_delay_noise:
        kp_local = rotations.apply(_conj(v.quat_w).unsqueeze(1), kp - pos.unsqueeze(1))  # (N, 4, 3)
        packed = torch.cat([pos, v.quat_w, vel, kp_local.reshape(n, -1)], dim=-1)
        env._object_state_queue, d = delayed(env._object_state_queue, packed, env.episode_length_buf == 0)
        o_pos = d[:, 0:3] + torch.randn_like(d[:, 0:3]) * dr.object_state_xyz_noise_std
        o_rot = rotations.random_perturb(torch.nn.functional.normalize(d[:, 3:7], dim=-1),
                                    dr.object_state_rotation_noise_degrees)
        o_vel = d[:, 7:13]
        o_kp = o_pos.unsqueeze(1) + rotations.apply(o_rot.unsqueeze(1), d[:, 13:25].reshape(n, 4, 3))
    else:
        o_rot, o_vel, o_kp = v.quat_w, vel, kp

    clean = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": env._prev_targets[:, perm],
        "palm_pos": palm_obs,
        "palm_rot": rotations.to_xyzw(s.palm_quat_w),
        "palm_vel": s.palm_vel_w,
        "object_rot": rotations.to_xyzw(v.quat_w),
        "object_vel": vel,
        "fingertip_pos_rel_palm": fingertip_rel_palm,
        "keypoints_rel_palm": kp - palm.unsqueeze(1),
        "keypoints_rel_goal": kp - targets,
        "object_scales": env._object_scales,
        "closest_keypoint_max_dist": env._closest_fold_err.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        "lifted_object": torch.zeros(n, 1, device=env.device),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }
    noisy = dict(clean)
    noisy["object_rot"] = rotations.to_xyzw(o_rot)
    noisy["object_vel"] = o_vel
    noisy["keypoints_rel_palm"] = o_kp - palm.unsqueeze(1)
    noisy["keypoints_rel_goal"] = o_kp - targets
    if dr.joint_velocity_obs_noise_std > 0:
        noisy["joint_vel"] = joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
    if dr.joint_pos_obs_noise_std > 0:
        span = env._joint_upper_canon - env._joint_lower_canon
        noisy["joint_pos"] = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span

    policy = stack(noisy, cfg.obs.obs_list)
    critic = stack(clean, cfg.obs.state_list)
    if dr.use_obs_delay and dr.obs_delay_max > 0:
        env._obs_queue, policy = delayed(env._obs_queue, policy, env.episode_length_buf == 0)
    clip = cfg.obs.clamp_abs_observations
    return {"policy": policy.clamp(-clip, clip), "critic": critic.clamp(-clip, clip)}


def _conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def guard_finite(env, obs: dict[str, torch.Tensor], s: SimState) -> None:
    """Keep non-finite values out of the policy: name the most upstream bad quantity, zero the
    obs of those envs and flag them (``_get_dones`` terminates them next)."""
    parts = {"cloth points": s.points_w, "joint_pos": s.joint_pos, "joint_vel": s.joint_vel,
             "palm": s.palm_pos_w, "policy obs": obs["policy"], "critic obs": obs["critic"]}
    bad_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name, t in parts.items():
        bad = ~torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=-1)
        if not bool(bad.any()):
            continue
        ids = bad.nonzero(as_tuple=True)[0]
        msg = (f"cloth: non-finite {name} in envs {ids[:8].tolist()} ({ids.numel()}/{env.num_envs}) "
               f"at env-step {env.episode_length_buf[ids[:8]].tolist()}")
        if env.cfg.cloth.nan_policy == "raise":
            raise RuntimeError(msg)
        print(f"[cloth] {msg} -- resetting those envs", flush=True)
        bad_any |= bad
    if not bool(bad_any.any()):
        return
    for t in obs.values():
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    env._nonfinite_envs |= bad_any
    env._nonfinite_count += int(bad_any.sum())
