"""Smoke-test a Play-family task's robot in Isaac Sim: sizes, geometry, PD hold, stability.

Run it on both robots and compare; with randomization off, both start in their default pose, which
viser_compare.py matched hand-to-hand, so their palm centres should coincide.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/smoke_env.py \
        --task Isaacsimenvs-PlayFlexivWuji-Direct-v0 --headless [--out report.json]

Reports, for env 0 unless noted:
  * observation / action sizes, and whether robot self-collision is on
  * reset hold: the largest joint drift and speed while holding the reset pose. Robot links
    that overlap at reset and are not filtered get pushed apart here
  * palm centre and fingertip pad positions after reset (env frame), as the policy sees them
  * arm step response: constant-velocity command then stop (tracking lag, settling error); a
    plain hold would show nothing, since gravity is off on the robot
  * hand step response: fully open, then fully closed (steps to 90%, worst joint)
  * random actions in [-1, 1] for ``--random_steps``, all envs: non-finite values, peak joint
    speed, and how many envs terminated
"""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-PlayFlexivWuji-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--hold_steps", type=int, default=60, help="steps per step-response phase")
parser.add_argument("--random_steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None, help="JSON report path")
parser.add_argument("--no_filter", action="store_true",
                    help="negative control: self-collision with the robot's filter list emptied")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401  gym.register side effects
from isaacsimenvs.eval.protocol import disable_randomization  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402


def _r(t, nd=4):
    return [round(float(v), nd) for v in t.flatten()]


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.assets.num_assets_per_type = 1
    env_cfg.seed = args_cli.seed
    disable_randomization(env_cfg)
    if args_cli.no_filter:
        import dataclasses

        from isaacsimenvs.tasks.play import robots

        robots.ROBOT_SPECS[env_cfg.robot] = dataclasses.replace(
            robots.ROBOT_SPECS[env_cfg.robot], self_collision_filter={})
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    spec, robot, device = inner.robot_spec, inner.robot, inner.device
    n_act = int(inner.cfg.action_space)
    report = {
        "task": args_cli.task, "robot": spec.name,
        "obs_dim": int(inner.cfg.observation_space), "state_dim": int(inner.cfg.state_space),
        "action_dim": n_act, "num_joints": robot.num_joints,
        "joint_names_lab": list(robot.data.joint_names),
        "robot_self_collision": bool(inner.cfg.robot_self_collision),
        "self_collision_filter_pairs": sum(map(len, spec.self_collision_filter.values())) // 2,
    }

    env.reset()
    zeros = torch.zeros((inner.num_envs, n_act), device=device)
    # Hand actions are absolute in [-1, 1] over the joint range; hold the hand at its reset pose
    # by sending the action that maps back onto it.
    lo, hi = inner._hand_lower[0], inner._hand_upper[0]
    hand_hold = 2.0 * (robot.data.joint_pos[0, inner._hand_joint_ids] - lo) / (hi - lo) - 1.0
    hold = zeros.clone()
    # Actions arrive in canonical order; Lab column j reads canonical column _perm_canon_to_lab[j].
    hold[:, inner._perm_canon_to_lab[inner._hand_joint_ids]] = hand_hold
    q_reset = robot.data.joint_pos.clone()
    drift, speed = 0.0, 0.0
    for _ in range(args_cli.hold_steps):
        env.step(hold)
        drift = max(drift, (robot.data.joint_pos - q_reset).abs().max().item())
        speed = max(speed, robot.data.joint_vel.abs().max().item())
    worst = int((robot.data.joint_pos - q_reset).abs().max(dim=0).values.argmax())
    report["reset_hold"] = {
        "steps": args_cli.hold_steps, "envs": inner.num_envs,
        "max_joint_drift_rad": round(drift, 4), "max_joint_speed_rad_s": round(speed, 3),
        "worst_joint": robot.data.joint_names[worst],
    }

    origin = inner.scene.env_origins[0]

    def palm_centre():
        b = robot.data.body_state_w[0, inner._palm_body_id]
        off = torch.tensor(spec.palm_center_offset, device=device)
        return b[:3] + quat_apply(b[3:7], off) - origin

    def fingertips():
        s = robot.data.body_state_w[0, inner._fingertip_body_ids]
        off = torch.tensor(spec.fingertip_offset, device=device).expand(s.shape[0], 3)
        return s[:, :3] + quat_apply(s[:, 3:7], off) - origin

    p0 = palm_centre()
    report["palm_centre_after_reset"] = _r(p0)
    report["palm_body"] = robot.data.body_names[inner._palm_body_id]
    report["fingertip_bodies"] = [robot.data.body_names[i] for i in inner._fingertip_body_ids]
    report["fingertip_pads_after_reset"] = [_r(p) for p in fingertips()]
    report["table_top_z"] = round(float(inner._table_z_per_env[0]) + 0.15, 4)

    # Arm step response: every arm joint commanded at a constant velocity (action 0.5, i.e.
    # 0.5 * dof_speed_scale rad/s), then zero. Gravity is off on the robot, so a plain hold would
    # never move; this is what loads the PD gains.
    arm, hand = inner._arm_joint_ids, inner._hand_joint_ids
    canon_arm = inner._perm_canon_to_lab[arm]
    canon_hand = inner._perm_canon_to_lab[hand]

    def track_err(ids):
        return (robot.data.joint_pos[0, ids] - inner._cur_targets[0, ids]).abs().max().item()

    move = hold.clone()
    move[:, canon_arm] = 0.5
    moving = []
    for _ in range(args_cli.hold_steps):
        env.step(move)
        moving.append(track_err(arm))
    settle = []
    for _ in range(args_cli.hold_steps):
        env.step(hold)
        settle.append(track_err(arm))
    report["arm_step"] = {
        "steps_each": args_cli.hold_steps,
        "max_lag_while_moving_rad": round(max(moving), 4),
        "err_after_stop_rad": round(settle[-1], 5),
    }

    # Hand step response: fully open (-1) for a while, then fully closed (+1). The target itself
    # ramps through hand_moving_average, so report error against the *final* target as well.
    opened = hold.clone()
    opened[:, canon_hand] = -1.0
    for _ in range(args_cli.hold_steps):
        env.step(opened)
    closed = hold.clone()
    closed[:, canon_hand] = 1.0
    final_target = inner._hand_upper[0]
    span = (final_target - inner._hand_lower[0]).abs()
    reached = None
    for i in range(args_cli.hold_steps):
        env.step(closed)
        frac = 1.0 - (robot.data.joint_pos[0, hand] - final_target).abs() / span
        if reached is None and bool((frac >= 0.9).all()):
            reached = i + 1
    report["hand_step"] = {
        "steps": args_cli.hold_steps,
        "steps_to_90pct_all_joints": reached,
        "worst_joint_fraction_of_range_reached": round(float(frac.min()), 3),
        "worst_joint": robot.data.joint_names[hand[int(frac.argmin())]],
        "max_err_vs_current_target_rad": round(track_err(hand), 4),
    }

    nonfinite, peak_speed, terminated = 0, 0.0, 0
    torch.manual_seed(args_cli.seed)
    for _ in range(args_cli.random_steps):
        obs, _, term, trunc, _ = env.step(torch.rand((inner.num_envs, n_act), device=device) * 2 - 1)
        policy = obs["policy"] if isinstance(obs, dict) else obs
        nonfinite += int((~torch.isfinite(policy)).any(dim=-1).sum())
        peak_speed = max(peak_speed, robot.data.joint_vel.abs().max().item())
        terminated += int(term.sum())
    report["random"] = {
        "steps": args_cli.random_steps, "envs": inner.num_envs,
        "env_steps_with_nonfinite_obs": nonfinite,
        "peak_joint_speed_rad_s": round(peak_speed, 3),
        "terminations": terminated,
    }

    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write(text + "\n")
    env.close()


if __name__ == "__main__":
    run()
    app.close()
