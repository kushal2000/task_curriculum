"""Domain randomization of the observation, the action and the robot
(``cfg.domain_randomization.{observation,action,robot}``). The sheet's physics DR
(``domain_randomization.cloth``) is the rigid backend's (``backends/rigid/randomization.py``).

:class:`DomainRandomization` holds every per-episode draw and re-draws them for the envs being
reset, in a FIXED order (it sets the random stream):

    delay lags        observation, action, object state -- all three whenever any delay is on
    object pose bias  (``observation.object_pos_bias``, ``observation.object_rot_bias_deg``)
    robot PD gains    (``robot.stiffness_scale``, ``robot.damping_scale``)
    hand dynamics     (``robot.hand_friction_scale``, ``robot.hand_effort_limit_scale``; PhysX only)

The robot terms are written to the articulation here. The per-step terms (delays, noise) are
applied where they act: :class:`~.actions.JointController` and
:class:`~.observations.ObservationBuilder`, each with its own :class:`DelayLine`.
"""

from __future__ import annotations

import math

import torch

from .geometry import quat
from .robot import RobotLayout, to_torch

__all__ = ["DelayLine", "DomainRandomization", "perturb_quat"]


def perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Rotate wxyz quaternions by a random axis and a uniform angle in ``+-max_deg``."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=q_wxyz.device), dim=-1)
    angle = torch.empty(n, device=q_wxyz.device).uniform_(-max_deg, max_deg) * (math.pi / 180.0)
    return quat.quat_mul(quat.quat_from_angle_axis(angle, axis), q_wxyz)


def _neutral(scale) -> bool:
    return tuple(float(v) for v in scale) == (1.0, 1.0)


class DelayLine:
    """The last ``max_lag + 1`` values of a ``(N, width)`` signal per env, read back at a lag."""

    def __init__(self, num_envs: int, max_lag: int, width: int, device) -> None:
        self.buffer = torch.zeros(num_envs, max_lag + 1, width, device=device)

    def clear(self, env_ids: torch.Tensor) -> None:
        self.buffer[env_ids] = 0.0

    def push_and_read(self, values: torch.Tensor, flush: torch.Tensor | None, lag: torch.Tensor) -> torch.Tensor:
        """Push ``values`` and read each env's entry ``lag`` steps back. ``flush`` rows are refilled
        with ``values`` first, so an episode never reads the previous one."""
        queue = self.buffer
        n = values.shape[0]
        if flush is not None and flush.any():
            queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)
        queue = torch.roll(queue, shifts=1, dims=1)
        queue[:, 0, :] = values
        self.buffer = queue
        return queue[torch.arange(n, device=values.device), lag]


class DomainRandomization:
    """The per-episode DR draws of the observation, the action and the robot."""

    def __init__(self, cfg, layout: RobotLayout, robot, *, num_envs: int, device) -> None:
        dr = cfg.domain_randomization
        self.cfg, self.layout, self.robot = cfg, layout, robot
        self.num_envs, self.device = num_envs, device
        n, dev = num_envs, device
        _check(dr)
        obs, act = dr.observation, dr.action
        self.any_delay = obs.delay_max > 0 or act.delay_max > 0 or obs.object_delay_max > 0
        # Per-episode draws, neutral until the first reset.
        self.obs_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.action_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.object_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.object_bias_pos = torch.zeros(n, 3, device=dev)
        self.object_bias_quat = torch.zeros(n, 4, device=dev)
        self.object_bias_quat[:, 0] = 1.0
        # Columns: arm stiffness, hand stiffness, arm damping, hand damping.
        self.robot_gain_scale = torch.ones(n, 4, device=dev)
        rob = dr.robot
        self.gain_dr = not (_neutral(rob.stiffness_scale) and _neutral(rob.damping_scale))
        if self.gain_dr:
            # The nominal gains the actuators wrote at startup; every draw multiplies these.
            self.robot_base_stiffness = to_torch(robot.data.joint_stiffness).clone()
            self.robot_base_damping = to_torch(robot.data.joint_damping).clone()
            self.arm_joint_mask = torch.zeros(self.robot_base_stiffness.shape[1], dtype=torch.bool, device=dev)
            self.arm_joint_mask[layout.arm_joint_ids] = True
        self._init_hand_dynamics()
        self._print_summary()

    def _init_hand_dynamics(self) -> None:
        """Nominal hand joint friction and torque limits for ``robot.hand_*_scale``."""
        rob = self.cfg.domain_randomization.robot
        self.hand_friction_dr = not _neutral(rob.hand_friction_scale)
        self.hand_effort_dr = not _neutral(rob.hand_effort_limit_scale)
        if not (self.hand_friction_dr or self.hand_effort_dr):
            return
        if not hasattr(self.robot, "root_physx_view"):
            raise NotImplementedError(
                "domain_randomization.robot.hand_friction_scale / hand_effort_limit_scale are PhysX only "
                "(the Newton articulation has no dynamic joint friction); set them to [1.0, 1.0] here."
            )
        data = self.robot.data
        self.hand_friction_base = data.joint_friction_coeff.clone()
        self.hand_dyn_friction_base = data.joint_dynamic_friction_coeff.clone()
        self.hand_effort_base = data.joint_effort_limits.clone()
        hand = self.layout.hand_joint_ids
        if self.hand_friction_dr and float(self.hand_friction_base[:, hand].abs().max()) == 0.0:
            raise ValueError(
                "domain_randomization.robot.hand_friction_scale scales the hand's joint friction, and this "
                "hand has none: set hand_sysid: true (the identified hand) or the scale to [1.0, 1.0]."
            )
        print(
            f"[dr] hand dynamics: friction x{tuple(rob.hand_friction_scale)} per env and joint on "
            f"{float(self.hand_friction_base[0, hand].min()):.4f}-{float(self.hand_friction_base[0, hand].max()):.4f} N m, "
            f"torque limit x{tuple(rob.hand_effort_limit_scale)} per env on "
            f"{float(self.hand_effort_base[0, hand].min()):.3f}-{float(self.hand_effort_base[0, hand].max()):.3f} N m",
            flush=True,
        )

    # ------------------------------------------------------------------ per episode

    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-draw every per-episode term of ``env_ids`` (see the module doc for the order)."""
        dr = self.cfg.domain_randomization
        self._draw_delay_lags(env_ids, dr)
        self._draw_object_pose_bias(env_ids, dr.observation)
        self._draw_robot_gains(env_ids, dr.robot)
        self._draw_hand_dynamics(env_ids, dr.robot)

    def _draw_delay_lags(self, env_ids, dr) -> None:
        """One lag per env for each delay line. All three are drawn whenever any delay is on, a line
        that is off included (its lag is then 0): the random stream the trained runs used."""
        if not self.any_delay:
            return
        n = env_ids.numel()
        for lag, max_lag in (
            (self.obs_lag, dr.observation.delay_max),
            (self.action_lag, dr.action.delay_max),
            (self.object_lag, dr.observation.object_delay_max),
        ):
            lag[env_ids] = torch.randint(0, int(max_lag) + 1, (n,), device=self.device)

    def _draw_object_pose_bias(self, env_ids, obs) -> None:
        """A per-episode position and rotation bias on the observed sheet."""
        n = env_ids.numel()
        if any(float(v) != 0.0 for v in obs.object_pos_bias):
            half = torch.as_tensor(tuple(float(v) for v in obs.object_pos_bias), device=self.device)
            self.object_bias_pos[env_ids] = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * half
        if float(obs.object_rot_bias_deg) > 0.0:
            ident = torch.zeros(n, 4, device=self.device)
            ident[:, 0] = 1.0
            self.object_bias_quat[env_ids] = perturb_quat(ident, float(obs.object_rot_bias_deg))

    def _draw_robot_gains(self, env_ids, rob) -> None:
        """PD gain DR: one stiffness and one damping factor per env, arm and hand separately."""
        if not self.gain_dr:
            return
        scale = torch.empty(env_ids.numel(), 4, device=self.device)  # arm k, hand k, arm d, hand d
        scale[:, 0:2].uniform_(*(float(v) for v in rob.stiffness_scale))
        scale[:, 2:4].uniform_(*(float(v) for v in rob.damping_scale))
        self.robot_gain_scale[env_ids] = scale
        arm = self.arm_joint_mask.unsqueeze(0)
        k = self.robot_base_stiffness[env_ids] * torch.where(arm, scale[:, 0:1], scale[:, 1:2])
        d = self.robot_base_damping[env_ids] * torch.where(arm, scale[:, 2:3], scale[:, 3:4])
        # The plain (not `_index`) writers: the Isaac Lab in the PhysX venv predates the `_index` API.
        self.robot.write_joint_stiffness_to_sim(k.contiguous(), env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(d.contiguous(), env_ids=env_ids)

    def _draw_hand_dynamics(self, env_ids, rob) -> None:
        """Hand dynamics DR (PhysX): Coulomb friction per env and joint, torque limit per env."""
        n, hand = env_ids.numel(), self.layout.hand_joint_ids
        if self.hand_friction_dr:
            f = torch.empty(n, len(hand), device=self.device).uniform_(*(float(v) for v in rob.hand_friction_scale))
            static = self.hand_friction_base[env_ids].clone()
            dynamic = self.hand_dyn_friction_base[env_ids].clone()
            static[:, hand] *= f
            dynamic[:, hand] *= f  # the same factor, so dynamic <= static still holds
            self.robot.write_joint_friction_coefficient_to_sim(static, joint_dynamic_friction_coeff=dynamic, env_ids=env_ids)
        if self.hand_effort_dr:
            e = torch.empty(n, 1, device=self.device).uniform_(*(float(v) for v in rob.hand_effort_limit_scale))
            limit = self.hand_effort_base[env_ids].clone()
            limit[:, hand] *= e
            self.robot.write_joint_effort_limit_to_sim(limit, env_ids=env_ids)

    def _print_summary(self) -> None:
        """One log line per group, so a run's log states its regime."""
        dr = self.cfg.domain_randomization
        o, a, r = dr.observation, dr.action, dr.robot
        print(
            f"[dr] observation: delay 0-{o.delay_max}, joint pos noise {o.joint_pos_noise_std} rad, joint vel noise "
            f"{o.joint_vel_noise_std} rad/s, object delay 0-{o.object_delay_max} + noise {o.object_pos_noise_std} m / "
            f"{o.object_rot_noise_deg} deg, object bias +-{tuple(o.object_pos_bias)} m / {o.object_rot_bias_deg} deg\n"
            f"[dr] action: delay 0-{a.delay_max}, noise {a.noise_std}\n"
            f"[dr] robot: PD k x{tuple(r.stiffness_scale)} d x{tuple(r.damping_scale)}, hand friction "
            f"x{tuple(r.hand_friction_scale)}, hand torque limit x{tuple(r.hand_effort_limit_scale)}",
            flush=True,
        )


def _check(dr) -> None:
    """Refuse a malformed DR config before anything is allocated."""
    for name, v in (
        ("observation.delay_max", dr.observation.delay_max),
        ("observation.object_delay_max", dr.observation.object_delay_max),
        ("action.delay_max", dr.action.delay_max),
    ):
        if int(v) != v or v < 0:
            raise ValueError(f"domain_randomization.{name} must be an integer >= 0, got {v}")
    for group in ("robot", "cloth"):
        for name, v in vars(getattr(dr, group)).items():
            if isinstance(v, (tuple, list)) and len(v) == 2 and float(v[0]) > float(v[1]):
                raise ValueError(f"domain_randomization.{group}.{name} must be [lo, hi] with lo <= hi, got {list(v)}")
