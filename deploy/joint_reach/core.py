"""Joint-reach policy on hardware: the sim's observation and action pipeline, without Isaac Sim.

Mirrors, step for step, JointReachEnv._get_observations and play/utils/action_utils.apply_action_pipeline
(with every DR term off), in the policy's canonical joint order: Rizon joint1-7, then the Wuji Hand 2
joints thumb to pinky, four each (the same order and signs as aiowuji.JOINT_NAMES). The numbers come
from spec_*.json, which isaacsimenvs/eval/joint_reach_export_spec.py reads from a live env.

check_offline.py checks this module against a rollout recorded in sim (real_test_sequence.pt).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = Path(__file__).with_name("spec_flexiv_wuji_hl.json")


@dataclass
class Spec:
    joint_names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    default_q: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray
    dt: float
    dof_speed_scale: float
    arm_ma: float
    hand_ma: float
    obs_clip: float
    arm_tol: float
    hand_tol: float
    n_arm: int = 7

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SPEC) -> "Spec":
        d = json.loads(Path(path).read_text())
        arr = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(
            joint_names=list(d["joint_names"]), lower=arr("joint_lower"), upper=arr("joint_upper"),
            default_q=arr("default_q"), stiffness=arr("stiffness"), damping=arr("damping"),
            effort_limit=arr("effort_limit"), dt=float(d["policy_dt"]),
            dof_speed_scale=float(d["dof_speed_scale"]), arm_ma=float(d["arm_moving_average"]),
            hand_ma=float(d["hand_moving_average"]), obs_clip=float(d["obs_clip"]),
            arm_tol=np.radians(d["arm_tol_deg"]), hand_tol=np.radians(d["hand_tol_deg"]),
            n_arm=int(d["num_arm_joints"]),
        )

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def normalize(self, q: np.ndarray) -> np.ndarray:
        return 2.0 * (q - self.lower) / (self.upper - self.lower) - 1.0


def build_obs(spec: Spec, q: np.ndarray, qd: np.ndarray, prev_target: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """The 4 x 27 policy observation: normalized joint pos, joint vel [rad/s], previous targets [rad],
    normalized goal; clipped like the env's."""
    obs = np.concatenate([spec.normalize(q), qd, prev_target, spec.normalize(goal)])
    return np.clip(obs, -spec.obs_clip, spec.obs_clip)


class ActionPipeline:
    """Policy action in [-1, 1] -> joint targets [rad].

    Arm: a velocity command, accumulated onto the previous target at dof_speed_scale rad/s per unit
    action, clamped to the limits, then a moving average with the previous target. Hand: an absolute
    position over the joint range, then a moving average. So the arm target moves at most
    dof_speed_scale x arm_ma = 0.15 rad/s.
    """

    def __init__(self, spec: Spec, start_target: np.ndarray) -> None:
        self.spec = spec
        self.prev = np.asarray(start_target, dtype=np.float64).copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        # float32, as the env: the arm target accumulates ~0.01 deg of round-off per minute otherwise.
        s, n, f = self.spec, self.spec.n_arm, np.float32
        a = np.clip(np.asarray(action, dtype=f), -1.0, 1.0)
        lo, hi, prev = s.lower.astype(f), s.upper.astype(f), self.prev.astype(f)
        arm_raw = np.clip(prev[:n] + f(s.dof_speed_scale * s.dt) * a[:n], lo[:n], hi[:n])
        arm = np.clip(f(s.arm_ma) * arm_raw + f(1.0 - s.arm_ma) * prev[:n], lo[:n], hi[:n])
        hand_raw = lo[n:] + f(0.5) * (a[n:] + f(1.0)) * (hi[n:] - lo[n:])
        hand = np.clip(f(s.hand_ma) * hand_raw + f(1.0 - s.hand_ma) * prev[n:], lo[n:], hi[n:])
        self.prev = np.concatenate([arm, hand]).astype(np.float64)
        return self.prev.copy()


def _load_player_module():
    """isaacsimenvs/eval/player.py by path: importing the isaacsimenvs package pulls in Isaac Lab."""
    path = REPO / "isaacsimenvs" / "eval" / "player.py"
    spec = importlib.util.spec_from_file_location("_jr_player", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Policy:
    """The rl_games SAPG LSTM checkpoint, one env, deterministic actions, SAPG block coefficient 0."""

    def __init__(self, checkpoint: str | Path, num_obs: int, num_actions: int,
                 policy_config: str | Path = REPO / "pretrained_policy" / "config.yaml",
                 device: str = "cpu", expl_coef: float = 0.0) -> None:
        import torch

        self._torch = torch
        self.device = device
        player = _load_player_module()
        self.player = player.PretrainedPlayer(
            config_path=policy_config, checkpoint_path=checkpoint, num_envs=1, device=device,
            num_observations=num_obs, num_actions=num_actions, expl_coef=expl_coef,
        )
        self.reset()

    def reset(self) -> None:
        self.player.reset_rnn()

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.player.get_action(x, deterministic=True)[0].cpu().numpy().astype(np.float64)


@dataclass
class GoalSequence:
    """Goals in canonical order and how many policy steps each is held."""

    start: np.ndarray
    goals: np.ndarray
    hold_steps: np.ndarray

    @classmethod
    def load(cls, path: str | Path, num_goals: int | None = None) -> "GoalSequence":
        import torch

        d = torch.load(path, map_location="cpu", weights_only=False)
        k = len(d["goals"]) if num_goals is None else min(num_goals, len(d["goals"]))
        return cls(d["start"].double().numpy(), d["goals"][:k].double().numpy(),
                   d["hold_steps"][:k].long().numpy())

    def goal_at(self) -> np.ndarray:
        """(sum(hold_steps), 27): the goal the policy sees at every step."""
        return np.repeat(self.goals, self.hold_steps, axis=0)

    def check(self, spec: Spec) -> None:
        for q in (self.start, *self.goals):
            if np.any(q < spec.lower - 1e-6) or np.any(q > spec.upper + 1e-6):
                raise ValueError("a goal is outside the joint limits")
