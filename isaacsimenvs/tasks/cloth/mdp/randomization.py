"""Observation, action and robot-dynamics domain randomization (``cfg.domain_randomization``).

:class:`DomainRandomization` holds every per-episode draw and re-draws them for the envs being
reset, in a FIXED order (it sets the random stream):

    observed sheet-size factor (``object_scale_noise_multiplier_range``)
    fixed delay lags           (``delay_mode: per_episode``: observation, action, object state)
    object pose bias           (``object_pose_bias_*``; the pose-level object DR)
    point-level object DR      (``point_dr_mode``; see ``point_dr.py``)
    robot PD gains             (``robot_{stiffness,damping}_scale_range``)
    hand joint dynamics        (``hand_{friction,effort_limit}_scale_range``; PhysX only)

The last two are written to the robot articulation here. The per-step terms (delays, noise) are
applied where they act: :class:`~.actions.JointController` and
:class:`~.observations.ObservationBuilder`, each with its own :class:`DelayLine`. The physics DR
of the rigid sheet is the rigid backend's.
"""

from __future__ import annotations

import math

import torch

from .geometry import quat
from .point_dr import MODES, RealisticLags
from .robot import RobotLayout, to_torch

__all__ = ["DelayLine", "DomainRandomization", "perturb_quat"]


def perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Rotate wxyz quaternions by a random axis and a uniform angle in ``+-max_deg``."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=q_wxyz.device), dim=-1)
    angle = torch.empty(n, device=q_wxyz.device).uniform_(-max_deg, max_deg) * (math.pi / 180.0)
    return quat.quat_mul(quat.quat_from_angle_axis(angle, axis), q_wxyz)


class DelayLine:
    """The last ``length`` values of a ``(N, width)`` signal per env, read back at a lag."""

    def __init__(self, num_envs: int, length: int, width: int, device) -> None:
        self.buffer = torch.zeros(num_envs, length, width, device=device)

    @property
    def length(self) -> int:
        return self.buffer.shape[1]

    def clear(self, env_ids: torch.Tensor) -> None:
        self.buffer[env_ids] = 0.0

    def push_and_read(self, values: torch.Tensor, flush: torch.Tensor | None, lag: torch.Tensor | None) -> torch.Tensor:
        """Push ``values`` and read each env's delayed entry: at ``lag`` (a fixed per-episode lag),
        or at a fresh uniform lag when ``lag`` is ``None``. ``flush`` rows are refilled with
        ``values`` first, so an episode never reads the previous one."""
        queue = self.buffer
        n = values.shape[0]
        if flush is not None and flush.any():
            queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)
        queue = torch.roll(queue, shifts=1, dims=1)
        queue[:, 0, :] = values
        self.buffer = queue
        if lag is None:
            lag = torch.randint(0, queue.shape[1], (n,), device=values.device)
        return queue[torch.arange(n, device=values.device), lag]


class DomainRandomization:
    """The per-episode DR draws. Allocating it draws the first sheet-size factors."""

    def __init__(self, cfg, layout: RobotLayout, robot, *, num_envs: int, num_keypoints: int, device) -> None:
        dr = cfg.domain_randomization
        self.cfg, self.layout, self.robot = cfg, layout, robot
        self.num_envs, self.device = num_envs, device
        n, dev = num_envs, device
        # Per-env factor on the observed sheet size (keypoint offsets and object_scales).
        lo, hi = dr.object_scale_noise_multiplier_range
        self.object_scale_multiplier = torch.empty(n, 3, device=dev).uniform_(lo, hi)

        # Delay-line lengths; the lines themselves belong to the code that reads them.
        self.obs_delay_len = max(1, dr.obs_delay_max)
        self.action_delay_len = max(1, dr.action_delay_max)
        self.object_delay_len = max(1, dr.object_state_delay_max)
        # Per-episode draws, neutral until the first reset.
        self.obs_delay_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.action_delay_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.object_delay_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.object_bias_pos = torch.zeros(n, 3, device=dev)
        self.object_bias_quat = torch.zeros(n, 4, device=dev)
        self.object_bias_quat[:, 0] = 1.0
        # Columns: arm stiffness, hand stiffness, arm damping, hand damping.
        self.robot_gain_scale = torch.ones(n, 4, device=dev)
        if tuple(dr.robot_stiffness_scale_range) != (1.0, 1.0) or tuple(dr.robot_damping_scale_range) != (1.0, 1.0):
            # The nominal gains the actuators wrote at startup; every draw multiplies these.
            self.robot_base_stiffness = to_torch(robot.data.joint_stiffness).clone()
            self.robot_base_damping = to_torch(robot.data.joint_damping).clone()
            self.arm_joint_mask = torch.zeros(self.robot_base_stiffness.shape[1], dtype=torch.bool, device=dev)
            self.arm_joint_mask[layout.arm_joint_ids] = True
        self._init_hand_dynamics()
        self._init_point_dr(num_keypoints)
        self._print_summary()

    # ------------------------------------------------------------------ construction

    def _init_hand_dynamics(self) -> None:
        """Nominal hand joint friction and torque limits for ``hand_{friction,effort_limit}_scale_range``."""
        dr = self.cfg.domain_randomization
        self.hand_friction_dr = tuple(float(v) for v in dr.hand_friction_scale_range) != (1.0, 1.0)
        self.hand_effort_dr = tuple(float(v) for v in dr.hand_effort_limit_scale_range) != (1.0, 1.0)
        if not (self.hand_friction_dr or self.hand_effort_dr):
            return
        if not hasattr(self.robot, "root_physx_view"):
            raise NotImplementedError(
                "hand_friction_scale_range / hand_effort_limit_scale_range are PhysX only (the Newton "
                "articulation has no dynamic joint friction); set them to [1.0, 1.0] here."
            )
        data = self.robot.data
        self.hand_friction_base = data.joint_friction_coeff.clone()
        self.hand_dyn_friction_base = data.joint_dynamic_friction_coeff.clone()
        self.hand_effort_base = data.joint_effort_limits.clone()
        hand = self.layout.hand_joint_ids
        print(
            f"[dr] hand dynamics: friction x{tuple(dr.hand_friction_scale_range)} per env and joint on "
            f"{float(self.hand_friction_base[0, hand].min()):.4f}-{float(self.hand_friction_base[0, hand].max()):.4f} N m, "
            f"torque limit x{tuple(dr.hand_effort_limit_scale_range)} per env on "
            f"{float(self.hand_effort_base[0, hand].min()):.3f}-{float(self.hand_effort_base[0, hand].max()):.3f} N m",
            flush=True,
        )

    def _init_point_dr(self, num_keypoints: int) -> None:
        """State for point-level object DR (``point_dr_mode``); validates the config."""
        dr = self.cfg.domain_randomization
        mode = str(dr.point_dr_mode)
        if mode not in MODES:
            raise ValueError(f"domain_randomization.point_dr_mode must be one of {MODES}, got {mode!r}")
        self.point_queue_len = 0
        self.point_lags = None
        if mode == "off":
            return
        pose_dr = (
            dr.use_object_state_delay_noise
            or any(float(v) != 0.0 for v in dr.object_pose_bias_xyz)
            or float(dr.object_pose_bias_rot_degrees) > 0.0
        )
        if pose_dr:
            raise ValueError(
                "point_dr_mode replaces the pose-level object DR: set use_object_state_delay_noise false "
                "and object_pose_bias_xyz / object_pose_bias_rot_degrees to zero."
            )
        if mode == "simple":
            lo, hi = (int(v) for v in dr.point_delay_range)
            if not 0 <= lo <= hi:
                raise ValueError(f"point_delay_range must be 0 <= lo <= hi, got {(lo, hi)}")
            self.point_queue_len = hi + 1
        else:
            lo, hi = (int(v) for v in dr.point_base_delay_range)
            cap = int(dr.point_delay_max)
            if not 0 <= lo <= hi <= cap or int(dr.point_delay_jitter) < 0 or min(dr.point_hold_choices) < 1:
                raise ValueError(
                    "realistic point DR needs 0 <= base lo <= hi <= point_delay_max, jitter >= 0, k >= 1; "
                    f"got base {(lo, hi)}, max {cap}, jitter {dr.point_delay_jitter}, k {dr.point_hold_choices}"
                )
            self.point_queue_len = cap + 1
            self.point_lags = RealisticLags(self.num_envs, self.device)
        self.point_bias = torch.zeros(self.num_envs, num_keypoints, 3, device=self.device)
        self.point_common_bias = torch.zeros(self.num_envs, 3, device=self.device)

    # ------------------------------------------------------------------ per step

    def lag(self, fixed: torch.Tensor) -> torch.Tensor | None:
        """The lag a delay line reads at: the per-episode one, or ``None`` (a fresh draw every step)."""
        return fixed if self.cfg.domain_randomization.delay_mode == "per_episode" else None

    # ------------------------------------------------------------------ per episode

    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-draw every per-episode term of ``env_ids`` (see the module doc for the order)."""
        dr = self.cfg.domain_randomization
        n = env_ids.numel()
        lo, hi = dr.object_scale_noise_multiplier_range
        self.object_scale_multiplier[env_ids] = torch.empty(n, 3, device=self.device).uniform_(lo, hi)
        self._draw_delay_lags(env_ids, dr)
        self._draw_object_pose_bias(env_ids, dr)
        self._draw_point_dr(env_ids, dr)
        self._draw_robot_gains(env_ids, dr)
        self._draw_hand_dynamics(env_ids, dr)

    def _symmetric_uniform(self, n: int, k: int | None, half_xyz) -> torch.Tensor:
        """Uniform in ``+-half_xyz`` per axis: ``(n, 3)``, or ``(n, k, 3)`` when ``k`` is given."""
        half = torch.as_tensor(tuple(float(v) for v in half_xyz), device=self.device, dtype=torch.float32)
        shape = (n, 3) if k is None else (n, k, 3)
        return (torch.rand(*shape, device=self.device) * 2.0 - 1.0) * half

    def _draw_delay_lags(self, env_ids, dr) -> None:
        """``delay_mode: per_episode``: one fixed lag per env for the obs, action and object lines."""
        if dr.delay_mode == "per_step":
            return
        if dr.delay_mode != "per_episode":
            raise ValueError(f"domain_randomization.delay_mode must be per_step or per_episode, got {dr.delay_mode!r}")
        n = env_ids.numel()
        for idx, length in (
            (self.obs_delay_idx, self.obs_delay_len),
            (self.action_delay_idx, self.action_delay_len),
            (self.object_delay_idx, self.object_delay_len),
        ):
            idx[env_ids] = torch.randint(0, length, (n,), device=self.device)

    def _draw_object_pose_bias(self, env_ids, dr) -> None:
        """Pose-level object DR: a per-episode position and rotation bias on the observed object."""
        n = env_ids.numel()
        if any(float(v) != 0.0 for v in dr.object_pose_bias_xyz):
            self.object_bias_pos[env_ids] = self._symmetric_uniform(n, None, dr.object_pose_bias_xyz)
        if float(dr.object_pose_bias_rot_degrees) > 0.0:
            ident = torch.zeros(n, 4, device=self.device)
            ident[:, 0] = 1.0
            self.object_bias_quat[env_ids] = perturb_quat(ident, float(dr.object_pose_bias_rot_degrees))

    def _draw_point_dr(self, env_ids, dr) -> None:
        """Point-level object DR: per-point and common biases, the camera stream."""
        if str(dr.point_dr_mode) == "off":
            return
        n = env_ids.numel()
        self.point_bias[env_ids] = self._symmetric_uniform(n, self.point_bias.shape[1], dr.point_bias_xyz)
        self.point_common_bias[env_ids] = self._symmetric_uniform(n, None, dr.point_common_bias_xyz)
        if dr.point_dr_mode == "realistic":
            self.point_lags.draw_episode(env_ids, dr.point_base_delay_range, dr.point_hold_choices)

    def _draw_robot_gains(self, env_ids, dr) -> None:
        """PD gain DR: one stiffness and one damping factor per env, arm and hand separately."""
        k_rng = tuple(float(v) for v in dr.robot_stiffness_scale_range)
        d_rng = tuple(float(v) for v in dr.robot_damping_scale_range)
        if k_rng == (1.0, 1.0) and d_rng == (1.0, 1.0):
            return
        scale = torch.empty(env_ids.numel(), 4, device=self.device)  # arm k, hand k, arm d, hand d
        scale[:, 0:2].uniform_(*k_rng)
        scale[:, 2:4].uniform_(*d_rng)
        self.robot_gain_scale[env_ids] = scale
        arm = self.arm_joint_mask.unsqueeze(0)
        k = self.robot_base_stiffness[env_ids] * torch.where(arm, scale[:, 0:1], scale[:, 1:2])
        d = self.robot_base_damping[env_ids] * torch.where(arm, scale[:, 2:3], scale[:, 3:4])
        # The plain (not `_index`) writers: the Isaac Lab in the PhysX venv predates the `_index` API.
        self.robot.write_joint_stiffness_to_sim(k.contiguous(), env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(d.contiguous(), env_ids=env_ids)

    def _draw_hand_dynamics(self, env_ids, dr) -> None:
        """Hand sysid DR (PhysX): Coulomb friction per env and joint, torque limit per env."""
        n, hand = env_ids.numel(), self.layout.hand_joint_ids
        if self.hand_friction_dr:
            f = torch.empty(n, len(hand), device=self.device).uniform_(*(float(v) for v in dr.hand_friction_scale_range))
            static = self.hand_friction_base[env_ids].clone()
            dynamic = self.hand_dyn_friction_base[env_ids].clone()
            static[:, hand] *= f
            dynamic[:, hand] *= f  # the same factor, so dynamic <= static still holds
            self.robot.write_joint_friction_coefficient_to_sim(static, joint_dynamic_friction_coeff=dynamic, env_ids=env_ids)
        if self.hand_effort_dr:
            e = torch.empty(n, 1, device=self.device).uniform_(*(float(v) for v in dr.hand_effort_limit_scale_range))
            limit = self.hand_effort_base[env_ids].clone()
            limit[:, hand] *= e
            self.robot.write_joint_effort_limit_to_sim(limit, env_ids=env_ids)

    def _print_summary(self) -> None:
        """One log line with every observation / action DR term, so a run's log states its regime."""
        dr = self.cfg.domain_randomization
        print(
            "[dr] obs delay "
            f"{'0-' + str(dr.obs_delay_max - 1) if dr.use_obs_delay else 'off'}, action delay "
            f"{'0-' + str(dr.action_delay_max - 1) if dr.use_action_delay else 'off'}, object state "
            + (
                f"delay 0-{dr.object_state_delay_max - 1} + noise {dr.object_state_xyz_noise_std} m / "
                f"{dr.object_state_rotation_noise_degrees} deg"
                if dr.use_object_state_delay_noise else "off"
            )
            + f" | delay_mode {dr.delay_mode} | joint vel noise {dr.joint_velocity_obs_noise_std}, "
            f"joint pos noise {dr.joint_pos_obs_noise_std} rad, action noise {dr.action_noise_std} | "
            f"object bias +-{tuple(dr.object_pose_bias_xyz)} m / {dr.object_pose_bias_rot_degrees} deg | "
            f"robot gains k x{tuple(dr.robot_stiffness_scale_range)} d x{tuple(dr.robot_damping_scale_range)}",
            flush=True,
        )
        if str(dr.point_dr_mode) != "off":
            lag = (
                f"lag {tuple(dr.point_delay_range)} per step"
                if dr.point_dr_mode == "simple"
                else f"base {tuple(dr.point_base_delay_range)} +-{dr.point_delay_jitter}, max "
                f"{dr.point_delay_max}, frame every k in {tuple(dr.point_hold_choices)}"
            )
            print(
                f"[dr] point DR {dr.point_dr_mode}: {lag} | noise {dr.point_noise_std} m, bias "
                f"+-{tuple(dr.point_bias_xyz)} m per point, common +-{tuple(dr.point_common_bias_xyz)} m "
                f"(points + goal) | obs delay on proprio only",
                flush=True,
            )
