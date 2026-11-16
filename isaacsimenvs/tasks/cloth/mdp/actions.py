"""Policy joint actions -> Isaac Lab position targets and passive spring efforts."""

from __future__ import annotations

import torch

from ..backends.robot import ROBOT
from .state import JointCommands


def joint_maps(lab_names: list[str], spec) -> dict[str, torch.Tensor | list[int]]:
    """Permutations between Isaac Lab's joint order and the canonical one, and the arm / hand ids.

    Raises if the asset's joints are not exactly the spec's, or the arm is not first in Lab order
    (the action map indexes ``[:7]`` / ``[7:]`` after permuting)."""
    canon = list(spec.joint_names_canonical)
    if sorted(lab_names) != sorted(canon):
        raise ValueError(f"robot joints {sorted(lab_names)} are not the spec's {sorted(canon)}")
    canon_to_lab = [canon.index(n) for n in lab_names]  # lab_vec = canon_vec[:, canon_to_lab]
    lab_to_canon = [lab_names.index(n) for n in canon]  # canon_vec = lab_vec[:, lab_to_canon]
    arm = sorted(lab_names.index(n) for n in spec.arm_joint_names)
    hand = sorted(lab_names.index(n) for n in spec.hand_joint_names)
    if arm != list(range(len(arm))) or hand != list(range(len(arm), len(lab_names))):
        raise ValueError(f"expected the arm joints first in Lab order, got arm {arm}, hand {hand}")
    return {"canon_to_lab": canon_to_lab, "lab_to_canon": lab_to_canon, "arm": arm, "hand": hand}


def delayed(queue: torch.Tensor, values: torch.Tensor, flush: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Push ``values`` into a rolling ``(N, L, D)`` queue and read each env at a random lag in
    [0, L). ``flush`` envs (episode start) fill the whole queue with ``values`` first."""
    n = values.shape[0]
    if bool(flush.any()):
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)
    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0] = values
    lag = torch.randint(0, queue.shape[1], (n,), device=values.device)
    return queue, queue[torch.arange(n, device=values.device), lag]


def spring_constants(lab_names: list[str], spec, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ks, ref)`` per Lab joint for the hand's passive spring (0 / 0 on the arm)."""
    ks = torch.tensor([spec.hand_spring_stiffness.get(n, 0.0) for n in lab_names], device=device)
    ref = torch.tensor([spec.hand_spring_ref.get(n, 0.0) for n in lab_names], device=device)
    return ks, ref


class JointController:
    def __init__(self, cfg, names, default_pos, limits, step_dt):
        self.cfg = cfg
        self.device = default_pos.device
        self.num_envs = default_pos.shape[0]
        self.step_dt = step_dt
        self.default_pos, self.limits = default_pos, limits
        maps = joint_maps(names, ROBOT)
        self._perm_canon_to_lab = torch.tensor(maps["canon_to_lab"], device=self.device)
        self._perm_lab_to_canon = torch.tensor(maps["lab_to_canon"], device=self.device)
        self._arm_ids, self._hand_ids = maps["arm"], maps["hand"]
        self._joint_lower_canon = limits[0, :, 0][self._perm_lab_to_canon]
        self._joint_upper_canon = limits[0, :, 1][self._perm_lab_to_canon]
        self._arm_lower, self._arm_upper = limits[:, self._arm_ids, 0], limits[:, self._arm_ids, 1]
        self._hand_lower, self._hand_upper = limits[:, self._hand_ids, 0], limits[:, self._hand_ids, 1]
        self._spring_ks, self._spring_ref = spring_constants(names, ROBOT, self.device)
        self._cur_targets = torch.zeros_like(default_pos)
        self._prev_targets = torch.zeros_like(default_pos)
        self._action_queue = torch.zeros(
            self.num_envs,
            max(1, int(cfg.domain_randomization.action_delay_max)),
            len(names),
            device=self.device,
        )

    def commands(self, joint_pos) -> JointCommands:
        return JointCommands(self._cur_targets, self._spring_ks * (self._spring_ref - joint_pos))

    def sample_reset(self, ids):
        self._action_queue[ids] = 0.0
        return self._robot_start(ids, self.default_pos[ids], self.limits[ids, :, 0], self.limits[ids, :, 1])

    def process(self, actions: torch.Tensor, episode_length: torch.Tensor) -> None:
        """Policy actions (canonical order) -> ``self._cur_targets`` (Lab order); also ``_prev_targets``."""
        dr, act = self.cfg.domain_randomization, self.cfg.action
        a = actions.to(self.device).clone()[:, self._perm_canon_to_lab]  # canonical -> Lab order

        if dr.use_action_delay and dr.action_delay_max > 0:
            self._action_queue, a = delayed(self._action_queue, a, episode_length == 0)
        if dr.action_noise_std > 0:
            a = (a + torch.randn_like(a) * dr.action_noise_std).clamp(-1.0, 1.0)

        arm, hand, prev = self._arm_ids, self._hand_ids, self._prev_targets
        arm_raw = (prev[:, arm] + act.dof_speed_scale * self.step_dt * a[:, arm]).clamp(
            self._arm_lower, self._arm_upper
        )
        arm_t = act.arm_moving_average * arm_raw + (1.0 - act.arm_moving_average) * prev[:, arm]
        hand_raw = self._hand_lower + 0.5 * (a[:, hand] + 1.0) * (self._hand_upper - self._hand_lower)
        hand_t = act.hand_moving_average * hand_raw + (1.0 - act.hand_moving_average) * prev[:, hand]

        self._cur_targets[:, arm] = arm_t.clamp(self._arm_lower, self._arm_upper)
        self._cur_targets[:, hand] = hand_t.clamp(self._hand_lower, self._hand_upper)
        self._prev_targets = self._cur_targets.clone()

    def _robot_start(self, ids: torch.Tensor, default_pos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        """Play's reset: each joint ``lerp(default, U(lower, upper), interval)``, velocity U(+-v);
        the targets start at the reset pose. Returns ``(joint_pos, joint_vel)`` (Lab order)."""
        r = self.cfg.reset
        scale = torch.zeros_like(default_pos)
        scale[:, self._arm_ids] = r.reset_dof_pos_random_interval_arm
        scale[:, self._hand_ids] = r.reset_dof_pos_random_interval_fingers
        sampled = lower + (upper - lower) * torch.rand_like(default_pos)
        pos = torch.lerp(default_pos, sampled, scale).clamp(lower, upper)
        vel = torch.empty_like(default_pos).uniform_(-r.reset_dof_vel_random_interval, r.reset_dof_vel_random_interval)
        self._prev_targets[ids] = pos
        self._cur_targets[ids] = pos
        return pos, vel
