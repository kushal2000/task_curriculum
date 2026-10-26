"""JointReachEnv: track a commanded joint configuration (see joint_reach_env_cfg.py).

Built on PlayEnv's scene, state buffers and action pipeline; replaces its reset, observation, reward
and termination hooks. Joint quantities are in the canonical (policy) order unless named ``_lab``.

Policy observation (4 x 27 = 108), each block canonical order:
    joint_pos            normalised to [-1, 1] by the joint limits (obs_utils._canonical_joint_obs)
    joint_vel            rad/s
    prev_action_targets  rad, the action pipeline's last joint targets
    goal                 normalised like joint_pos
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from isaaclab.envs import DirectRLEnv

from ..play.play_env import PlayEnv
from ..play.robots import get_robot_spec
from ..play.utils.action_utils import apply_action_pipeline
from ..play.utils.obs_utils import NUM_FINGERTIPS, _apply_local_offset, _apply_obs_delay, _canonical_joint_obs
from ..play.utils.reset_utils import _draw_episode_dr, _reset_table_pose, allocate_state_buffers
from ..play.utils.scene_utils import apply_physx_material_properties
from .joint_reach_env_cfg import JointReachEnvCfg

#: Where the unused object waits: on the floor behind the table, out of the arm's reach [m, env frame].
_OBJECT_PARK_POS = (0.0, -0.75, 0.1)
#: Extra critic inputs: goal error (27) + log gain multipliers (4) + obs / action delay (2) + hold phase (1).
_STATE_EXTRA = 27 + 4 + 2 + 1


def _log_uniform(lo_hi, n: int, device) -> torch.Tensor:
    lo, hi = (float(v) for v in lo_hi)
    return torch.exp(torch.empty(n, device=device).uniform_(math.log(lo), math.log(hi)))


class JointReachEnv(PlayEnv):
    cfg: JointReachEnvCfg

    def __init__(self, cfg: JointReachEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self.robot_spec = get_robot_spec(cfg.robot)
        n = self.robot_spec.num_joints
        cfg.action_space = n
        cfg.observation_space = 4 * n
        cfg.state_space = 4 * n + _STATE_EXTRA
        # The cloth pool's per-env PhysX GPU buffers (tasks/cloth/physx_envs.py _size_gpu_buffers): Play's
        # fixed defaults do not start 24576 Flexiv + Wuji envs with self-collision on.
        px, n_env = cfg.sim.physx, int(cfg.scene.num_envs)
        px.gpu_max_rigid_patch_count = max(int(px.gpu_max_rigid_patch_count), 48 * n_env)
        px.gpu_max_rigid_contact_count = max(int(px.gpu_max_rigid_contact_count), 256 * n_env)
        px.gpu_found_lost_pairs_capacity = max(int(px.gpu_found_lost_pairs_capacity), 128 * n_env)
        px.gpu_total_aggregate_pairs_capacity = max(int(px.gpu_total_aggregate_pairs_capacity), 128 * n_env)
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)  # PlayEnv's spaces are not ours
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        self._allocate_reach_buffers()
        self._goal_bank = self._load_or_build_goal_bank()
        print(f"[joint_reach] goal bank: {self._goal_bank.shape[0]} configurations", flush=True)

    # ------------------------------------------------------------------ buffers

    def _allocate_reach_buffers(self) -> None:
        n_env, n = self.num_envs, self.robot_spec.num_joints
        dev = self.device
        self._goal = torch.zeros(n_env, n, device=dev)
        self._hold_left = torch.zeros(n_env, dtype=torch.long, device=dev)
        self._hold_len = torch.ones(n_env, dtype=torch.long, device=dev)
        self._actions = torch.zeros(n_env, n, device=dev)
        self._prev_actions = torch.zeros(n_env, n, device=dev)
        self._err = torch.zeros(n_env, n, device=dev)
        # When True the env never draws goals itself: the caller writes _goal (eval / replay).
        self.external_goals = False

        stiff, damp = self.robot.data.joint_stiffness, self.robot.data.joint_damping
        self._base_k_lab = (stiff.torch if hasattr(stiff, "torch") else stiff).clone()
        self._base_d_lab = (damp.torch if hasattr(damp, "torch") else damp).clone()
        self._arm_mask_lab = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arm_mask_lab[self._arm_joint_ids] = True
        # Columns: arm stiffness, hand stiffness, arm damping, hand damping.
        self._gain_scale = torch.ones(n_env, 4, device=dev)

        self._lo = self._joint_lower_canon
        self._hi = self._joint_upper_canon
        self._arm_tol = math.radians(self.cfg.reach_reward.arm_tol_deg)
        self._hand_tol = math.radians(self.cfg.reach_reward.hand_tol_deg)

        # Per-episode tallies over completed goal segments (logged at episode end).
        self._seg_done = torch.zeros(n_env, device=dev)
        self._seg_reached = torch.zeros(n_env, device=dev)
        self._seg_arm_err = torch.zeros(n_env, device=dev)
        self._seg_hand_err = torch.zeros(n_env, device=dev)
        self._seg_first_in_tol = torch.full((n_env,), -1, dtype=torch.long, device=dev)
        self._seg_settle_sum = torch.zeros(n_env, device=dev)
        self._steps_in_tol = torch.zeros(n_env, device=dev)
        self._near_table_steps = torch.zeros(n_env, device=dev)
        self._reward_terms: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ goal bank

    def _table_top_and_halfsize(self) -> tuple[float, float, float]:
        sx, sy, sz = self.cfg.goal.table_box_size
        top = self.cfg.reset.table_reset_z + self.cfg.reset.table_reset_z_range + 0.5 * sz
        return top, 0.5 * sx, 0.5 * sy

    def _check_points(self, palm_pos: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Workspace test per env: palm centre over the table at the allowed height, and no robot point
        over the table footprint closer to the top than the clearance. Positions in the env frame."""
        g = self.cfg.goal
        top, hx, hy = self._table_top_and_halfsize()
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

    def _robot_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Palm centre (N, 3) and every body origin plus fingertip pad (N, B + 5, 3), env frame."""
        origins = self.scene.env_origins
        body_pos = self.robot.data.body_pos_w - origins.unsqueeze(1)
        palm_q = self.robot.data.body_quat_w[:, self._palm_body_id]
        palm = _apply_local_offset(
            self.robot.data.body_pos_w[:, self._palm_body_id], palm_q, self.robot_spec.palm_center_offset,
            (self.num_envs,),
        ) - origins
        ft_q = self.robot.data.body_quat_w[:, self._fingertip_body_ids]
        pads = _apply_local_offset(
            self.robot.data.body_pos_w[:, self._fingertip_body_ids], ft_q, self.robot_spec.fingertip_offset,
            (self.num_envs, NUM_FINGERTIPS),
        ) - origins.unsqueeze(1)
        return palm, torch.cat([body_pos, pads], dim=1)

    def _park_object(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, :3] = torch.tensor(_OBJECT_PARK_POS, device=self.device) + self.scene.env_origins[env_ids]
        pose[:, 3] = 1.0
        self.object.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)

    def build_goal_bank(self, size: int, seed: int) -> torch.Tensor:
        """Sample collision-free goal configurations in the workspace, canonical order, (size, 27).

        Every env tests one candidate per round: the robot is written to it with zero joint gains and
        stepped; with gravity off only a contact can move it. Candidates that stay put and pass the
        workspace test are kept.
        """
        g = self.cfg.goal
        dev = self.device
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        all_ids = torch.arange(self.num_envs, device=dev)
        n_arm = len(self.robot_spec.arm_joint_names)

        # The highest table the reset noise can place, and the object out of the way.
        top, _, _ = self._table_top_and_halfsize()
        table_pose = torch.zeros(self.num_envs, 7, device=dev)
        table_pose[:, :3] = self.scene.env_origins
        table_pose[:, 2] += top - 0.5 * g.table_box_size[2]
        table_pose[:, 3] = 1.0
        self.table.write_root_pose_to_sim(table_pose)
        self._park_object(all_ids)

        lo, hi = self._lo.clone(), self._hi.clone()
        margin = math.radians(g.arm_limit_margin_deg)
        lo[:n_arm] += margin
        hi[:n_arm] -= margin

        zeros = torch.zeros(self.num_envs, self.robot_spec.num_joints, device=dev)
        self.robot.write_joint_stiffness_to_sim(zeros)
        self.robot.write_joint_damping_to_sim(zeros)
        kept, n_kept, n_tried, rounds = [], 0, 0, 0
        while n_kept < size:
            q = lo + (hi - lo) * torch.rand(self.num_envs, lo.numel(), device=dev, generator=gen)
            q_lab = q[:, self._perm_canon_to_lab]
            self.robot.write_joint_state_to_sim(q_lab, torch.zeros_like(q_lab))
            self.robot.set_joint_position_target(q_lab)
            for _ in range(g.collision_check_steps):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
            still = (self.robot.data.joint_pos - q_lab).abs().max(dim=-1).values < g.collision_tol_rad
            palm, points = self._robot_points()
            ok = still & self._check_points(palm, points)
            kept.append(q[ok])
            n_kept += int(ok.sum())
            n_tried += self.num_envs
            rounds += 1
            if rounds % 20 == 0 or n_kept >= size:
                print(
                    f"[joint_reach] goal bank round {rounds}: kept {n_kept}/{n_tried} "
                    f"({100.0 * n_kept / n_tried:.2f}%)", flush=True,
                )
            if rounds >= g.max_bank_rounds:
                raise RuntimeError(f"goal bank: {n_kept} of {size} after {rounds} rounds; loosen the workspace")
        self.robot.write_joint_stiffness_to_sim(self._base_k_lab)
        self.robot.write_joint_damping_to_sim(self._base_d_lab)
        return torch.cat(kept)[:size].contiguous()

    def _load_or_build_goal_bank(self) -> torch.Tensor:
        g = self.cfg.goal
        path = Path(g.bank_file)
        if path.is_file():
            payload = torch.load(path, map_location=self.device)
            bank = payload["goals"].to(self.device)
            if list(payload["joint_names"]) != list(self.robot_spec.joint_names_canonical):
                raise ValueError(f"{path}: joint order differs from {self.robot_spec.name}")
            return bank
        bank = self.build_goal_bank(g.bank_size, g.bank_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "goals": bank.cpu(),
                "joint_names": list(self.robot_spec.joint_names_canonical),
                "seed": g.bank_seed,
                "cfg": g.to_dict(),
            },
            tmp,
        )
        tmp.rename(path)
        print(f"[joint_reach] wrote goal bank {path}", flush=True)
        return bank

    # ------------------------------------------------------------------ goals

    def _draw_goals(self, env_ids: torch.Tensor, allow_near: bool) -> None:
        g = self.cfg.goal
        n = env_ids.numel()
        bank = self._goal_bank
        new = bank[torch.randint(0, bank.shape[0], (n,), device=self.device)]
        if allow_near and g.near_prob > 0.0:
            near = torch.rand(n, device=self.device) < g.near_prob
            if near.any():
                ids = env_ids[near]
                cand = bank[torch.randint(0, bank.shape[0], (ids.numel(), g.near_candidates), device=self.device)]
                dist = (cand - self._goal[ids].unsqueeze(1)).abs().sum(dim=-1)
                new[near] = cand[torch.arange(ids.numel(), device=self.device), dist.argmin(dim=-1)]
        self._goal[env_ids] = new
        lo_s, hi_s = g.hold_time_range_s
        steps = torch.empty(n, device=self.device).uniform_(lo_s, hi_s) / self.step_dt
        self._hold_len[env_ids] = steps.round().long().clamp(min=1)
        self._hold_left[env_ids] = self._hold_len[env_ids]
        self._seg_first_in_tol[env_ids] = -1

    def _close_segments(self, env_ids: torch.Tensor) -> None:
        """Tally the goal segments that end now (their hold time ran out)."""
        err = self._err[env_ids].abs()
        arm_max = err[:, :7].max(dim=-1).values
        hand_max = err[:, 7:].max(dim=-1).values
        reached = (arm_max < self._arm_tol) & (hand_max < self._hand_tol)
        self._seg_done[env_ids] += 1
        self._seg_reached[env_ids] += reached.float()
        self._seg_arm_err[env_ids] += torch.rad2deg(arm_max)
        self._seg_hand_err[env_ids] += torch.rad2deg(hand_max)
        first = self._seg_first_in_tol[env_ids]
        self._seg_settle_sum[env_ids] += torch.where(
            first >= 0, first.float(), self._hold_len[env_ids].float()
        ) * self.step_dt

    # ------------------------------------------------------------------ hooks

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        DirectRLEnv._reset_idx(self, env_ids)
        n = env_ids.numel()
        g, rc = self.cfg.goal, self.cfg.reset

        # Start: the default pose with Play's reset noise, or a bank configuration.
        default = self.robot.data.default_joint_pos[env_ids]
        lower = self.robot.data.joint_pos_limits[env_ids, :, 0]
        upper = self.robot.data.joint_pos_limits[env_ids, :, 1]
        scale = torch.zeros_like(default)
        scale[:, self._arm_joint_ids] = rc.reset_dof_pos_random_interval_arm
        scale[:, self._hand_joint_ids] = rc.reset_dof_pos_random_interval_fingers
        q = torch.lerp(default, lower + (upper - lower) * torch.rand_like(default), scale)
        from_bank = torch.rand(n, device=self.device) < g.start_from_bank_prob
        if from_bank.any():
            pick = self._goal_bank[torch.randint(0, self._goal_bank.shape[0], (int(from_bank.sum()),), device=self.device)]
            q[from_bank] = pick[:, self._perm_canon_to_lab]
        q = q.clamp(lower, upper)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self._prev_targets[env_ids] = q
        self._cur_targets[env_ids] = q

        _reset_table_pose(self, env_ids)
        self._park_object(env_ids)

        self._goal[env_ids] = q[:, self._perm_lab_to_canon]
        if not self.external_goals:
            self._draw_goals(env_ids, allow_near=False)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._action_queue[env_ids] = 0.0
        self._obs_queue[env_ids] = 0.0
        _draw_episode_dr(self, env_ids)  # delays; the gain ranges there are (1, 1)
        self._draw_gains(env_ids)

        for buf in (
            self._seg_done, self._seg_reached, self._seg_arm_err, self._seg_hand_err,
            self._seg_settle_sum, self._steps_in_tol, self._near_table_steps,
        ):
            buf[env_ids] = 0.0
        self._seg_first_in_tol[env_ids] = -1

    def _draw_gains(self, env_ids: torch.Tensor) -> None:
        gd = self.cfg.gain_dr
        if not gd.enabled:
            return
        n = env_ids.numel()
        scale = torch.stack(
            [
                _log_uniform(gd.arm_stiffness, n, self.device),
                _log_uniform(gd.hand_stiffness, n, self.device),
                _log_uniform(gd.arm_damping, n, self.device),
                _log_uniform(gd.hand_damping, n, self.device),
            ],
            dim=-1,
        )
        fixed = torch.tensor([float(v) for v in gd.fixed_scales], device=self.device)
        scale = torch.where(fixed > 0.0, fixed.expand(n, -1), scale)
        self._gain_scale[env_ids] = scale
        arm = self._arm_mask_lab.unsqueeze(0)
        k = self._base_k_lab[env_ids] * torch.where(arm, scale[:, 0:1], scale[:, 1:2])
        d = self._base_d_lab[env_ids] * torch.where(arm, scale[:, 2:3], scale[:, 3:4])
        self.robot.write_joint_stiffness_to_sim(k.contiguous(), env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(d.contiguous(), env_ids=env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        apply_action_pipeline(self, self._actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._perm_lab_to_canon]
        self._err = self._goal - q
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self._termination_reasons = {"timeout": truncated}
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        rc = self.cfg.reach_reward
        err = self._err.abs()
        arm_mean, hand_mean = err[:, :7].mean(dim=-1), err[:, 7:].mean(dim=-1)
        in_tol = (err[:, :7].max(dim=-1).values < self._arm_tol) & (err[:, 7:].max(dim=-1).values < self._hand_tol)

        top, hx, hy = self._table_top_and_halfsize()
        _, points = self._robot_points()
        over = (points[..., 0].abs() <= hx) & (points[..., 1].abs() <= hy)
        top_now = (self._table_z_per_env + 0.5 * self.cfg.goal.table_box_size[2]).unsqueeze(1)
        near_table = (over & (points[..., 2] < top_now + rc.table_proximity)).any(dim=-1)

        qd = self.robot.data.joint_vel
        terms = {
            "arm_error": -rc.arm_error_scale * arm_mean,
            "hand_error": -rc.hand_error_scale * hand_mean,
            "arm_exp": rc.arm_exp_weight * torch.exp(-arm_mean / rc.arm_exp_width),
            "hand_exp": rc.hand_exp_weight * torch.exp(-hand_mean / rc.hand_exp_width),
            "within_tol": rc.within_tol_bonus * in_tol.float(),
            "action_rate": -rc.action_rate_scale * (self._actions - self._prev_actions).pow(2).mean(dim=-1),
            "arm_action": -rc.arm_action_scale * self._actions[:, :7].pow(2).mean(dim=-1),
            "joint_vel": -rc.joint_vel_scale * qd.pow(2).mean(dim=-1),
            "near_table": -rc.table_proximity_penalty * near_table.float(),
        }
        reward = torch.stack(list(terms.values())).sum(dim=0)
        self._reward_terms = terms

        # Segment bookkeeping, then the next goals for segments whose hold time ran out.
        self._steps_in_tol += in_tol.float()
        self._near_table_steps += near_table.float()
        if not self.external_goals:
            elapsed = self._hold_len - self._hold_left
            first = in_tol & (self._seg_first_in_tol < 0)
            self._seg_first_in_tol = torch.where(first, elapsed, self._seg_first_in_tol)
            self._hold_left -= 1
            ended = (self._hold_left <= 0).nonzero(as_tuple=False).squeeze(-1)
            if ended.numel() > 0:
                self._close_segments(ended)
                self._draw_goals(ended, allow_near=True)
        self._log_metrics()
        return reward

    def _log_metrics(self) -> None:
        segs = self._seg_done.clamp(min=1.0)
        steps = self.episode_length_buf.float().clamp(min=1.0)
        self.extras["episode_cumulative"] = self._reward_terms
        self.extras["episode_final"] = {
            "segments": self._seg_done.clone(),
            "segment_reached": self._seg_reached / segs,
            "segment_arm_err_deg": self._seg_arm_err / segs,
            "segment_hand_err_deg": self._seg_hand_err / segs,
            "segment_settle_s": self._seg_settle_sum / segs,
            "frac_steps_in_tol": self._steps_in_tol / steps,
            "frac_steps_near_table": self._near_table_steps / steps,
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        dr = self.cfg.domain_randomization
        joint_pos, joint_vel, prev_targets = _canonical_joint_obs(self)
        span = self._hi - self._lo
        goal = 2.0 * (self._goal - self._lo) / span - 1.0

        pos_obs, vel_obs = joint_pos, joint_vel
        if dr.joint_pos_obs_noise_std > 0:
            pos_obs = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span
        if dr.joint_velocity_obs_noise_std > 0:
            vel_obs = joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        policy = torch.cat([pos_obs, vel_obs, prev_targets, goal], dim=-1)
        if dr.use_obs_delay:
            policy = _apply_obs_delay(self, policy)

        phase = (1.0 - self._hold_left.float() / self._hold_len.float()).unsqueeze(-1)
        state = torch.cat(
            [
                joint_pos, joint_vel, prev_targets, goal, self._err,
                torch.log(self._gain_scale),
                self._obs_delay_idx.float().unsqueeze(-1),
                self._action_delay_idx.float().unsqueeze(-1),
                phase,
            ],
            dim=-1,
        )
        clip = self.cfg.obs.clamp_abs_observations
        return {"policy": policy.clamp(-clip, clip), "critic": state.clamp(-clip, clip)}


__all__ = ["JointReachEnv"]
