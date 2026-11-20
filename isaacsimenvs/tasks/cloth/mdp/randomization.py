"""Domain randomization of the proprioception, the keypoint observation, the action and the robot
(``cfg.domain_randomization.{proprio,keypoints,action,robot}``). The point cloud's DR
(``domain_randomization.pointcloud``) is drawn by the cloud (``perception/pointcloud.py``), the
sheet's physics DR (``domain_randomization.cloth``) by the rigid backend
(``backends/rigid/randomization.py``).

:class:`DomainRandomization` holds every per-episode draw and re-draws them for the envs being
reset, in a FIXED order (it sets the random stream):

    delay lags        proprio, action, keypoints -- all three whenever any of them is on
    keypoint bias     (``keypoints.pos_bias``, ``keypoints.rot_bias_deg``)
    robot PD gains    (``robot.stiffness_scale``, ``robot.damping_scale``)
    hand dynamics     (``robot.hand_friction_scale``, ``robot.hand_effort_limit_scale``; PhysX only)

The robot terms are written to the articulation here. The per-step terms (delays, noise) are
applied where they act: :class:`~.actions.JointController` and the sensors in ``perception/``
(``proprio.py``, ``keypoints.py``), each with its own ``delay.DelayLine``.
"""

from __future__ import annotations

import math

import torch

from .geometry import quat
from .robot import RobotLayout, to_torch

__all__ = ["DomainRandomization", "perturb_quat"]


def perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Rotate wxyz quaternions by a random axis and a uniform angle in ``+-max_deg``."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=q_wxyz.device), dim=-1)
    angle = torch.empty(n, device=q_wxyz.device).uniform_(-max_deg, max_deg) * (math.pi / 180.0)
    return quat.quat_mul(quat.quat_from_angle_axis(angle, axis), q_wxyz)


def _as_tuple(v):
    return tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else float(v)


def _neutral(scale) -> bool:
    return tuple(float(v) for v in scale) == (1.0, 1.0)


class DomainRandomization:
    """The per-episode DR draws of the proprioception, the keypoints, the action and the robot."""

    def __init__(self, cfg, layout: RobotLayout, robot, *, num_envs: int, device) -> None:
        dr = cfg.domain_randomization
        self.cfg, self.layout, self.robot = cfg, layout, robot
        self.num_envs, self.device = num_envs, device
        n, dev = num_envs, device
        _check(cfg)
        self.any_delay = dr.proprio.delay_max > 0 or dr.action.delay_max > 0 or dr.keypoints.delay_max > 0
        # Per-episode draws, neutral until the first reset.
        self.proprio_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.action_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.keypoints_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.keypoints_bias_pos = torch.zeros(n, 3, device=dev)
        self.keypoints_bias_quat = torch.zeros(n, 4, device=dev)
        self.keypoints_bias_quat[:, 0] = 1.0
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
        self._draw_keypoints_bias(env_ids, dr.keypoints)
        self._draw_robot_gains(env_ids, dr.robot)
        self._draw_hand_dynamics(env_ids, dr.robot)

    def _draw_delay_lags(self, env_ids, dr) -> None:
        """One lag per env for each delay line. All three are drawn whenever any delay is on, a line
        that is off included (its lag is then 0): the random stream the trained runs used."""
        if not self.any_delay:
            return
        n = env_ids.numel()
        for lag, max_lag in (
            (self.proprio_lag, dr.proprio.delay_max),
            (self.action_lag, dr.action.delay_max),
            (self.keypoints_lag, dr.keypoints.delay_max),
        ):
            lag[env_ids] = torch.randint(0, int(max_lag) + 1, (n,), device=self.device)

    def _draw_keypoints_bias(self, env_ids, kp) -> None:
        """A per-episode position and rotation bias on the sheet pose the keypoints are built from."""
        n = env_ids.numel()
        if any(float(v) != 0.0 for v in kp.pos_bias):
            half = torch.as_tensor(tuple(float(v) for v in kp.pos_bias), device=self.device)
            self.keypoints_bias_pos[env_ids] = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * half
        if float(kp.rot_bias_deg) > 0.0:
            ident = torch.zeros(n, 4, device=self.device)
            ident[:, 0] = 1.0
            self.keypoints_bias_quat[env_ids] = perturb_quat(ident, float(kp.rot_bias_deg))

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
        p, k, c, a, r = dr.proprio, dr.keypoints, dr.pointcloud, dr.action, dr.robot
        sheet = (
            f"[dr] keypoints: delay 0-{k.delay_max}, noise {k.pos_noise_std} m / {k.rot_noise_deg} deg, bias "
            f"+-{tuple(k.pos_bias)} m / {k.rot_bias_deg} deg"
            if self.cfg.obs.object == "keypoints" else
            f"[dr] pointcloud: delay 0-{c.delay_max}, camera +-{tuple(c.cam_pos_noise)} m, calibration "
            f"+-{tuple(c.calib_xyz)} m / {c.calib_rot_deg} deg, depth noise {tuple(c.depth_noise_std_range)} m, "
            f"drop {tuple(c.drop_prob_range)}"
        )
        print(
            f"[dr] proprio: delay 0-{p.delay_max}, joint pos noise {p.joint_pos_noise_std} rad, joint vel noise "
            f"{p.joint_vel_noise_std} rad/s\n{sheet}\n"
            f"[dr] action: delay 0-{a.delay_max}, noise {a.noise_std}\n"
            f"[dr] robot: PD k x{tuple(r.stiffness_scale)} d x{tuple(r.damping_scale)}, hand friction "
            f"x{tuple(r.hand_friction_scale)}, hand torque limit x{tuple(r.hand_effort_limit_scale)}",
            flush=True,
        )


def _check(cfg) -> None:
    """Refuse a malformed DR config before anything is allocated, and DR for the way of seeing the
    sheet the policy does not use (``obs.object``)."""
    dr = cfg.domain_randomization
    if cfg.obs.object not in ("keypoints", "pointcloud"):
        raise ValueError(f"obs.object must be keypoints or pointcloud, got {cfg.obs.object!r}")
    unused = "pointcloud" if cfg.obs.object == "keypoints" else "keypoints"
    neutral = vars(type(getattr(dr, unused))())
    moved = {k: v for k, v in vars(getattr(dr, unused)).items() if _as_tuple(v) != _as_tuple(neutral[k])}
    if moved:
        raise ValueError(f"obs.object is {cfg.obs.object}: domain_randomization.{unused} must stay neutral, got {moved}")
    for group in ("proprio", "keypoints", "pointcloud", "action"):
        v = getattr(dr, group).delay_max
        if int(v) != v or v < 0:
            raise ValueError(f"domain_randomization.{group}.delay_max must be an integer >= 0, got {v}")
    for group in ("robot", "cloth"):
        for name, v in vars(getattr(dr, group)).items():
            if isinstance(v, (tuple, list)) and len(v) == 2 and float(v[0]) > float(v[1]):
                raise ValueError(f"domain_randomization.{group}.{name} must be [lo, hi] with lo <= hi, got {list(v)}")
