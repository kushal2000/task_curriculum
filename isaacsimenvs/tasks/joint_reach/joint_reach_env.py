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
        self._load_goals()

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
        self._chain_id = torch.zeros(n_env, dtype=torch.long, device=dev)
        self._chain_idx = torch.zeros(n_env, dtype=torch.long, device=dev)
        self._chain_dir = torch.ones(n_env, dtype=torch.long, device=dev)

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

    def _arm_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampling limits, canonical order: arm limits shrunk by the margin, hand limits as they are."""
        lo, hi = self._lo.clone(), self._hi.clone()
        margin = math.radians(self.cfg.goal.arm_limit_margin_deg)
        lo[:7] += margin
        hi[:7] -= margin
        return lo, hi

    def _begin_validation(self) -> None:
        """Ready the scene for _valid: the highest table the reset noise can place, the object away,
        zero joint gains. _end_validation restores the gains."""
        g = self.cfg.goal
        top, _, _ = self._table_top_and_halfsize()
        table_pose = torch.zeros(self.num_envs, 7, device=self.device)
        table_pose[:, :3] = self.scene.env_origins
        table_pose[:, 2] += top - 0.5 * g.table_box_size[2]
        table_pose[:, 3] = 1.0
        self.table.write_root_pose_to_sim(table_pose)
        self._park_object(torch.arange(self.num_envs, device=self.device))
        zeros = torch.zeros(self.num_envs, self.robot_spec.num_joints, device=self.device)
        self.robot.write_joint_stiffness_to_sim(zeros)
        self.robot.write_joint_damping_to_sim(zeros)

    def _end_validation(self) -> None:
        self.robot.write_joint_stiffness_to_sim(self._base_k_lab)
        self.robot.write_joint_damping_to_sim(self._base_d_lab)

    def _valid(self, q: torch.Tensor) -> torch.Tensor:
        """Which of num_envs candidate configurations (canonical order) are goals: written with zero
        joint gains and stepped, a configuration in contact moves (gravity is off on the robot); one that
        stays put must also pass the workspace test."""
        g = self.cfg.goal
        q_lab = q[:, self._perm_canon_to_lab]
        self.robot.write_joint_state_to_sim(q_lab, torch.zeros_like(q_lab))
        self.robot.set_joint_position_target(q_lab)
        for _ in range(g.collision_check_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        still = (self.robot.data.joint_pos - q_lab).abs().max(dim=-1).values < g.collision_tol_rad
        palm, points = self._robot_points()
        return still & self._check_points(palm, points)

    def build_goal_bank(self, size: int, seed: int) -> torch.Tensor:
        """Collision-free configurations spread over the workspace, canonical order, (size, 27): every
        joint uniform in its sampling limits, one candidate per env per round, rejection by _valid."""
        g = self.cfg.goal
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        lo, hi = self._arm_limits()
        self._begin_validation()
        kept, n_kept, n_tried, rounds = [], 0, 0, 0
        while n_kept < size:
            q = lo + (hi - lo) * torch.rand(self.num_envs, lo.numel(), device=self.device, generator=gen)
            ok = self._valid(q)
            kept.append(q[ok])
            n_kept += int(ok.sum())
            n_tried += self.num_envs
            rounds += 1
            if rounds % 100 == 0 or n_kept >= size:
                print(
                    f"[joint_reach] goal bank round {rounds}: kept {n_kept}/{n_tried} "
                    f"({100.0 * n_kept / n_tried:.2f}%)", flush=True,
                )
            if rounds >= g.max_bank_rounds:
                raise RuntimeError(f"goal bank: {n_kept} of {size} after {rounds} rounds; loosen the workspace")
        self._end_validation()
        return torch.cat(kept)[:size].contiguous()

    def build_goal_chains(self, starts: torch.Tensor, length: int, seed: int) -> torch.Tensor:
        """Random walks of `length` goals from `starts` (num_envs, 27), canonical order, (num_envs,
        length, 27). Each step moves every arm joint by up to chain_arm_step_rad x a scale drawn per
        step and redraws the hand; a proposal _valid rejects is redrawn next round."""
        g = self.cfg.goal
        dev = self.device
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        lo, hi = self._arm_limits()
        n_env = self.num_envs
        rows = torch.arange(n_env, device=dev)
        chains = starts.unsqueeze(1).repeat(1, length, 1)
        filled = torch.ones(n_env, dtype=torch.long, device=dev)
        self._begin_validation()
        rounds, tried = 0, 0
        while bool((filled < length).any()) and rounds < g.max_chain_rounds:
            cur = chains[rows, filled - 1]
            s_lo, s_hi = g.chain_step_scale_range
            scale = s_lo + (s_hi - s_lo) * torch.rand(n_env, 1, device=dev, generator=gen)
            step = (2.0 * torch.rand(n_env, 7, device=dev, generator=gen) - 1.0) * g.chain_arm_step_rad * scale
            q = lo + (hi - lo) * torch.rand(n_env, lo.numel(), device=dev, generator=gen)
            q[:, :7] = torch.minimum(torch.maximum(cur[:, :7] + step, lo[:7]), hi[:7])
            open_ = filled < length
            ok = self._valid(q) & open_
            tried += int(open_.sum())
            ids = ok.nonzero(as_tuple=False).squeeze(-1)
            chains[ids, filled[ids]] = q[ids]
            filled[ids] += 1
            rounds += 1
            if rounds % 50 == 0:
                print(
                    f"[joint_reach] chains round {rounds}: {int(filled.sum())}/{n_env * length} goals, "
                    f"acceptance {100.0 * (int(filled.sum()) - n_env) / max(tried, 1):.1f}%", flush=True,
                )
        self._end_validation()
        # A chain that could not be extended repeats its last goal.
        short = filled < length
        if bool(short.any()):
            print(f"[joint_reach] {int(short.sum())} chains stopped short after {rounds} rounds", flush=True)
            idx = torch.arange(length, device=dev).unsqueeze(0).expand(n_env, -1)
            last = torch.minimum(idx, (filled - 1).unsqueeze(1))
            chains = chains[rows.unsqueeze(1), last]
        return chains

    def default_goal(self) -> torch.Tensor:
        """The default pose, canonical order (27,)."""
        return self.robot.data.default_joint_pos[0, self._perm_lab_to_canon].clone()

    def chain_starts(self, bank: torch.Tensor, default_frac: float, seed: int) -> torch.Tensor:
        """num_envs chain starts: the first default_frac at the default pose, the rest from `bank`."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        n_def = int(round(default_frac * self.num_envs))
        starts = bank[torch.randint(0, bank.shape[0], (self.num_envs,), device=self.device, generator=gen)]
        starts[:n_def] = self.default_goal()
        return starts

    def _load_or_build(self, path_str: str, build, what: str) -> torch.Tensor:
        path = Path(path_str)
        if path.is_file():
            payload = torch.load(path, map_location=self.device)
            if list(payload["joint_names"]) != list(self.robot_spec.joint_names_canonical):
                raise ValueError(f"{path}: joint order differs from {self.robot_spec.name}")
            return payload["goals"].to(self.device)
        goals = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {"goals": goals.cpu(), "joint_names": list(self.robot_spec.joint_names_canonical),
             "cfg": self.cfg.goal.to_dict()},
            tmp,
        )
        tmp.rename(path)
        print(f"[joint_reach] wrote {what} {path}", flush=True)
        return goals

    def _load_goals(self) -> None:
        g = self.cfg.goal
        self._goal_bank = self._load_or_build(
            g.bank_file, lambda: self.build_goal_bank(g.bank_size, g.bank_seed), "goal bank"
        )
        self._chains = self._load_or_build(
            g.chain_file,
            lambda: self.build_goal_chains(
                self.chain_starts(self._goal_bank, g.chain_default_frac, g.chain_seed), g.chain_length, g.chain_seed
            ),
            "goal chains",
        )
        self._n_default_chains = int(round(g.chain_default_frac * self._chains.shape[0]))
        print(
            f"[joint_reach] goal bank {tuple(self._goal_bank.shape)}, chains {tuple(self._chains.shape)} "
            f"({self._n_default_chains} from the default pose)", flush=True,
        )

    # ------------------------------------------------------------------ goals

    def hold_steps(self, goal_from: torch.Tensor, goal_to: torch.Tensor, settle_s: torch.Tensor) -> torch.Tensor:
        """Policy steps to hold `goal_to` after `goal_from`: the largest arm joint move at the action
        pipeline's arm speed limit, plus `settle_s`."""
        a = self.cfg.action
        v = a.dof_speed_scale * a.arm_moving_average  # rad/s the joint targets can move
        travel = (goal_to[:, :7] - goal_from[:, :7]).abs().max(dim=-1).values / v
        return ((travel + settle_s) / self.step_dt).round().long().clamp(min=1)

    def _set_goal(self, env_ids: torch.Tensor, goal_from: torch.Tensor) -> None:
        """Write the chain goal at _chain_idx as the goal of `env_ids` and start its hold timer."""
        lo_s, hi_s = self.cfg.goal.settle_time_range_s
        goal = self._chains[self._chain_id[env_ids], self._chain_idx[env_ids]]
        self._goal[env_ids] = goal
        settle = torch.empty(env_ids.numel(), device=self.device).uniform_(lo_s, hi_s)
        self._hold_len[env_ids] = self.hold_steps(goal_from, goal, settle)
        self._hold_left[env_ids] = self._hold_len[env_ids]
        self._seg_first_in_tol[env_ids] = -1

    def _advance_goals(self, env_ids: torch.Tensor) -> None:
        """Next goal along each env's chain, turning back at either end."""
        length = self._chains.shape[1]
        idx, direction = self._chain_idx[env_ids], self._chain_dir[env_ids]
        nxt = idx + direction
        turn = (nxt < 0) | (nxt >= length)
        direction = torch.where(turn, -direction, direction)
        self._chain_dir[env_ids] = direction
        self._chain_idx[env_ids] = torch.where(turn, idx + direction, nxt).clamp(0, length - 1)
        self._set_goal(env_ids, self._goal[env_ids].clone())

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

    def place_robot(self, env_ids: torch.Tensor, q_canon: torch.Tensor) -> None:
        """Write a joint configuration (canonical order) as the robot state and the pipeline's targets."""
        q = q_canon[:, self._perm_canon_to_lab]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self._prev_targets[env_ids] = q
        self._cur_targets[env_ids] = q

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        DirectRLEnv._reset_idx(self, env_ids)
        n = env_ids.numel()
        g, rc = self.cfg.goal, self.cfg.reset
        dev = self.device
        n_chain, length = self._chains.shape[:2]

        # Either the default pose (Play's reset noise) at the start of a default-seeded chain, or a
        # random goal of a random chain, exactly.
        from_default = torch.rand(n, device=dev) < g.default_start_prob
        chain_id = torch.where(
            from_default,
            torch.randint(0, max(self._n_default_chains, 1), (n,), device=dev),
            torch.randint(0, n_chain, (n,), device=dev),
        )
        idx = torch.where(from_default, torch.zeros_like(chain_id), torch.randint(0, length, (n,), device=dev))
        default = self.robot.data.default_joint_pos[env_ids]
        lower = self.robot.data.joint_pos_limits[env_ids, :, 0]
        upper = self.robot.data.joint_pos_limits[env_ids, :, 1]
        scale = torch.zeros_like(default)
        scale[:, self._arm_joint_ids] = rc.reset_dof_pos_random_interval_arm
        scale[:, self._hand_joint_ids] = rc.reset_dof_pos_random_interval_fingers
        q_lab = torch.lerp(default, lower + (upper - lower) * torch.rand_like(default), scale).clamp(lower, upper)
        q = q_lab[:, self._perm_lab_to_canon]
        q = torch.where(from_default.unsqueeze(1), q, self._chains[chain_id, idx])
        self.place_robot(env_ids, q)

        _reset_table_pose(self, env_ids)
        self._park_object(env_ids)

        self._chain_id[env_ids] = chain_id
        self._chain_idx[env_ids] = idx
        self._chain_dir[env_ids] = torch.where(torch.rand(n, device=dev) < 0.5, 1, -1)
        self._chain_dir[env_ids] = torch.where(from_default, torch.ones_like(chain_id), self._chain_dir[env_ids])
        self._goal[env_ids] = q
        if not self.external_goals:
            self._advance_goals(env_ids)
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
                self._advance_goals(ended)
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
