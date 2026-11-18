"""Follow-up to probe_real_dynamics.py: per-joint static holding with the PD drive on, and the
drive-effort saturation, for all 20 hand joints of a Flexiv + right Wuji robot.

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_real_stick.py \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --headless [--out report.json]

Steps the SIM directly (no env actions), randomization off, robot has no gravity, hand in free space.
  stick: joint from lower + 20% range, position step of min(0.3, 60% range); after 0.5 s the stuck
     error e, kp e / f (Coulomb theory: <= 1); then a feed-forward effort ramped toward the target
     until the joint breaks away: holding capacity (kp e + ff) / f. Index PIP also with other
     fingers fully open (contact check), and with kd x0 / x4, kp x4, and armature x10.
  sat: joint from lower + 0.02 rad, target = upper; the first-step generalized force
     ((M + diag(armature)) qdd)_j + f_j against the drive max force (PhysX's M excludes armature).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.scripts.evaluation.protocol import disable_randomization  # noqa: E402
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml  # noqa: E402


def _f(x, nd=6):
    return round(float(x), nd)


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point")
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    disable_randomization(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    robot, device = inner.robot, inner.device
    view = robot.root_physx_view
    names = list(robot.data.joint_names)
    hand = [int(i) for i in inner._hand_joint_ids]
    env.reset()
    dt = float(inner.physics_dt)
    fr = view.get_dof_friction_properties()[0].to(device)
    arma0 = view.get_dof_armatures().clone()
    A = arma0[0].to(device)
    maxf = view.get_dof_max_forces()[0].to(device)
    kp0, kd0 = robot.data.joint_stiffness.clone(), robot.data.joint_damping.clone()
    lo, hi = robot.data.joint_pos_limits[0, :, 0], robot.data.joint_pos_limits[0, :, 1]
    home = robot.data.default_joint_pos.clone()
    q_mid = home.clone()
    q_mid[:, hand] = (0.5 * (lo + hi))[hand]
    q_open = home.clone()
    q_open[:, hand] = lo[hand] + 0.02
    report = {"task": args_cli.task, "robot": inner.robot_spec.name}

    def step(n=1, log=None):
        for _ in range(n):
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            robot.update(dt)
            if log is not None:
                log.append((robot.data.joint_pos[0].clone(), robot.data.joint_vel[0].clone()))

    def set_state(q):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.set_joint_velocity_target(torch.zeros_like(q))
        robot.set_joint_effort_target(torch.zeros_like(q))

    def gains(kp=None, kd=None):
        robot.write_joint_stiffness_to_sim(kp if kp is not None else kp0)
        robot.write_joint_damping_to_sim(kd if kd is not None else kd0)

    def stick(j, base, kp=None, kd=None):
        gains(kp, kd)
        k_j = float((kp if kp is not None else kp0)[0, j])
        f_j = float(fr[j, 0])
        rng = float(hi[j] - lo[j])
        q0 = base.clone()
        q0[:, j] = float(lo[j]) + 0.2 * rng
        set_state(q0)
        step(5)
        set_state(q0)
        delta = min(0.3, 0.6 * rng)
        tgt = q0.clone()
        tgt[:, j] += delta
        robot.set_joint_position_target(tgt)
        log = []
        step(60, log)
        e = float(tgt[0, j] - robot.data.joint_pos[0, j])
        q_stuck = float(robot.data.joint_pos[0, j])
        moved_frac = 1.0 - e / delta
        # Breakaway: feed-forward toward the target, ramped 0 -> 1.5 f over 90 steps (0.75 s).
        ff_break = None
        for i in range(90):
            ff = (i + 1) / 90 * 1.5 * max(f_j, 1e-4)
            eff = torch.zeros_like(q0)
            eff[:, j] = ff
            robot.set_joint_effort_target(eff)
            step(1)
            if abs(float(robot.data.joint_pos[0, j]) - q_stuck) > 2e-4:
                ff_break = ff
                break
        robot.set_joint_effort_target(torch.zeros_like(q0))
        gains()
        return {
            "kp": _f(k_j, 4), "f": _f(f_j, 5), "delta": _f(delta, 4), "moved_frac": _f(moved_frac, 3),
            "stuck_err": _f(e, 5), "pred_f_over_kp": _f(f_j / k_j if k_j else float("nan"), 5),
            "kp_e_over_f": _f(k_j * e / f_j, 3) if f_j > 0 else None,
            "ff_break": _f(ff_break, 5) if ff_break is not None else None,
            "holding_over_f": _f((k_j * e + ff_break) / f_j, 3) if (f_j > 0 and ff_break is not None) else None,
            "vel_end": _f(float(log[-1][1][j]), 5),
        }

    st = {names[j]: stick(j, q_mid) for j in hand}
    jp = names.index("r_index_finger_pip")
    st["index_pip_others_open"] = stick(jp, q_open)
    k = kp0.clone(); d = kd0.clone(); d[:, jp] = 0.0
    st["index_pip_kd0"] = stick(jp, q_mid, k, d)
    d = kd0.clone(); d[:, jp] *= 4
    st["index_pip_kdx4"] = stick(jp, q_mid, kp0, d)
    k = kp0.clone(); k[:, jp] *= 4
    st["index_pip_kpx4"] = stick(jp, q_mid, k, kd0)
    a = arma0.clone(); a[:, jp] *= 10
    view.set_dof_armatures(a, indices=torch.arange(a.shape[0], dtype=torch.int32))
    st["index_pip_armx10"] = stick(jp, q_mid)
    view.set_dof_armatures(arma0, indices=torch.arange(arma0.shape[0], dtype=torch.int32))
    report["stick"] = st
    print("[probe] stick", json.dumps(st, indent=1), flush=True)

    sat = {}
    for j in hand:
        gains()
        q0 = q_mid.clone()
        q0[:, j] = float(lo[j]) + 0.02
        set_state(q0)
        step(5)
        set_state(q0)
        M = view.get_generalized_mass_matrices()[0].to(device) + torch.diag(A)
        tgt = q0.clone()
        tgt[:, j] = float(hi[j])
        robot.set_joint_position_target(tgt)
        log = []
        step(1, log)
        qdd = log[0][1] / dt
        gen = float(M[j] @ qdd)
        f_j = float(fr[j, 0]) + float(fr[j, 2]) * float(log[0][1][j])
        sat[names[j]] = {
            "kp_x_err": _f(float(kp0[0, j]) * (float(hi[j]) - float(q0[0, j])), 4),
            "max_force": _f(float(maxf[j]), 5),
            "est_drive_effort": _f(gen + f_j, 5),
            "ratio": _f((gen + f_j) / float(maxf[j]), 4),
        }
    report["saturation"] = sat
    print("[probe] sat", json.dumps(sat, indent=1), flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as fh:
            fh.write(json.dumps(report, indent=2) + "\n")
    print("[probe] done", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    run()
