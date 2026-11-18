"""Check the hand's passive spring (a joint effort outside the PD) and the hand-dynamics DR in PhysX.

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_spring.py \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment flat7FWRHL_dA \
        --headless --num_envs 64 [--out report.json]

1. readback after a reset (every DR has drawn): hand drive stiffness == spec kp x one per-env factor
   inside robot_stiffness_scale_range (no spring in the drive); ks / ref == the spec, arm ks 0;
   hand static and dynamic friction == spec x one factor per env AND joint inside
   hand_friction_scale_range (static and dynamic the same factor), viscous untouched; hand torque
   limit == spec x one factor per env inside hand_effort_limit_scale_range.
2. equilibrium: joint friction zeroed, hand targets a = spring rest + 0.5 rad on the flexion joints;
   the env's own _apply_action runs, once with the spring and once without (ks None). With g the
   gravity torque read off the no-spring run, g = kp (q_off - a), the spring run must settle at
   (kp a + ks ref + g) / (kp + ks).
   Then: the worst (env, joint) |joint_vel| traced over 40 steps, and the settle repeated with the
   joint friction on. Finding (11-10): a still joint can READ 0.2-0.6 rad/s with the spring on
   while its position does not move (not sleep, not velocity iterations: --no_sleep / --vel_iters).
2b. velocity readback while moving: 1 Hz sine targets, friction on; reported joint_vel vs the
   backward difference of joint_pos, spring on and off (11-10: 0.39 vs 0.36 rad/s RMS).
3. not clipped: hand torque limit 1e-4 N m (the PD is dead), friction zero, start at rest + 0.4 rad
   on the flexion joints, target = start. With the spring the joints must get at least 90% of the
   way to the rest angle (PIP: the lower limit 0 is the rest angle); without it they must stay put.
   A spring inside the drive would be clipped to 1e-4 too.
"""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--experiment", default="flat7FWRHL_dA")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle_steps", type=int, default=600)
parser.add_argument("--out", default=None)
parser.add_argument("--no_sleep", type=int, default=0, help="robot articulation sleep threshold 0")
parser.add_argument("--vel_iters", type=int, default=-1, help=">=0: robot + scene velocity iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401

if args_cli.no_sleep or args_cli.vel_iters >= 0:
    import isaaclab.sim as sim_utils  # noqa: E402

    from isaacsimenvs.tasks.play.utils import scene_utils as _su  # noqa: E402

    _build = _su.build_robot_articulation_usd_cfg

    def _no_sleep_build(*a, **k):
        cfg = _build(*a, **k)
        kw = {}
        if args_cli.no_sleep:
            kw["sleep_threshold"] = 0.0
        if args_cli.vel_iters >= 0:
            kw["solver_velocity_iteration_count"] = args_cli.vel_iters
        cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(**kw)
        return cfg

    _su.build_robot_articulation_usd_cfg = _no_sleep_build
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


def _t(x):
    return x.torch if hasattr(x, "torch") else x


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment or None)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    if args_cli.vel_iters >= 0:
        env_cfg.sim.physx.min_velocity_iteration_count = args_cli.vel_iters
        env_cfg.sim.physx.max_velocity_iteration_count = max(args_cli.vel_iters, 1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    spec, robot, device = inner.robot_spec, inner.robot, inner.device
    names = list(robot.data.joint_names)
    hand = [int(i) for i in inner._hand_joint_ids]
    arm = [int(i) for i in inner._arm_joint_ids]
    env.reset()
    rep = {"task": args_cli.task, "experiment": args_cli.experiment, "num_envs": args_cli.num_envs}
    ok = True

    # ------------------------------------------------------------ 1. readback
    dr = env_cfg.domain_randomization
    ks, ref = inner._spring_ks, inner._spring_ref
    view = robot.root_physx_view
    spec_kp = torch.tensor([spec.hand_stiffness.get(n, 0.0) for n in names], device=device)
    spec_ks = torch.tensor([spec.hand_spring_stiffness.get(n, 0.0) for n in names], device=device)
    stiff = _t(robot.data.joint_stiffness).clone()
    view_k = view.get_dof_stiffnesses().to(device)
    factor = view_k[:, hand] / spec_kp[hand]
    kp = stiff  # the drives carry the PD only

    def _in(lo_hi, x):
        return float(lo_hi[0]) - 1e-5 <= float(x.min()) and float(x.max()) <= float(lo_hi[1]) + 1e-5

    fr = view.get_dof_friction_properties().to(device)  # (N, J, 3): static, dynamic, viscous
    spec_fs = torch.tensor([spec.hand_friction[n] for n in (names[i] for i in hand)], device=device)
    spec_fd = torch.tensor([spec.hand_dynamic_friction[n] for n in (names[i] for i in hand)], device=device)
    spec_fv = torch.tensor([spec.hand_viscous_friction[n] for n in (names[i] for i in hand)], device=device)
    f_s, f_d = fr[:, hand, 0] / spec_fs, fr[:, hand, 1] / spec_fd
    lim = view.get_dof_max_forces().to(device)[:, hand]
    spec_lim = torch.tensor([spec.hand_effort_limit[names[i]] for i in hand], device=device)
    f_e = lim / spec_lim
    rk = tuple(dr.robot_stiffness_scale_range)
    rep["readback"] = {
        "drive_vs_data_max_err": float((view_k[:, hand] - stiff[:, hand]).abs().max()),
        "kp_factor_spread_within_env": float((factor.max(1).values - factor.min(1).values).max()),
        "kp_factor_range": [float(factor.min()), float(factor.max())], "kp_range": list(rk),
        "spring_vs_spec_max_err": float((ks - spec_ks).abs().max()), "arm_ks_max": float(ks[arm].abs().max()),
        "friction_factor_range": [float(f_s.min()), float(f_s.max())],
        "friction_range": list(dr.hand_friction_scale_range),
        "friction_static_vs_dynamic_factor_max_diff": float((f_s - f_d).abs().max()),
        "friction_factor_std_within_env": float(f_s.std(1).mean()),
        "viscous_vs_spec_max_err": float((fr[:, hand, 2] - spec_fv).abs().max()),
        "effort_factor_range": [float(f_e.min()), float(f_e.max())],
        "effort_range": list(dr.hand_effort_limit_scale_range),
        "effort_factor_spread_within_env": float((f_e.max(1).values - f_e.min(1).values).max()),
        "effort_factor_std_across_envs": float(f_e[:, 0].std()),
    }
    r = rep["readback"]
    checks = {
        "drive == data": r["drive_vs_data_max_err"] < 1e-4,
        "kp one factor per env": r["kp_factor_spread_within_env"] < 1e-4,
        "kp factor in range": _in(rk, factor),
        "spring == spec, arm none": r["spring_vs_spec_max_err"] < 1e-6 and r["arm_ks_max"] == 0.0,
        "friction factor in range": _in(dr.hand_friction_scale_range, f_s),
        "static, dynamic same factor": r["friction_static_vs_dynamic_factor_max_diff"] < 1e-4,
        "friction varies per joint": (r["friction_factor_std_within_env"] > 1e-3)
        == (tuple(dr.hand_friction_scale_range) != (1.0, 1.0)),
        "viscous untouched": r["viscous_vs_spec_max_err"] < 1e-7,
        "effort factor in range": _in(dr.hand_effort_limit_scale_range, f_e),
        "effort one factor per env": r["effort_factor_spread_within_env"] < 1e-4,
    }
    r["checks"] = checks
    ok &= all(checks.values())
    print("[spring] readback", json.dumps(r), flush=True)

    # ------------------------------------------------------------ 2. equilibrium
    idx = torch.arange(inner.num_envs, dtype=torch.int32)
    fr = view.get_dof_friction_properties().clone()
    fr0 = fr.clone()
    fr[:, :, :] = 0.0
    view.set_dof_friction_properties(fr, indices=idx)
    lo, hi = (_t(robot.data.soft_joint_pos_limits)[0, :, k] for k in (0, 1))
    a = _t(robot.data.default_joint_pos).clone()
    flex = [i for i in hand if ("flex" in names[i] or "pip" in names[i] or "dip" in names[i]
                                 or names[i].endswith("_mcp") or names[i].endswith("_ip"))]
    a[:, flex] = (ref[flex] + 0.5).unsqueeze(0)
    a[:, hand] = torch.clamp(a[:, hand], lo[hand] + 0.05, hi[hand] - 0.05)
    dt = float(inner.cfg.sim.dt)

    def settle():
        robot.write_joint_state_to_sim(a, torch.zeros_like(a))
        inner._cur_targets[:] = a
        for _ in range(args_cli.settle_steps):
            inner._apply_action()
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            robot.update(dt)
        v = _t(robot.data.joint_vel)[:, hand].abs().max(0).values
        print("[spring]   end |v| per hand joint (rad/s):", json.dumps(
            {names[i]: round(float(v[k]), 4) for k, i in enumerate(hand) if float(v[k]) > 1e-3}), flush=True)
        return _t(robot.data.joint_pos).clone(), float(v.max())

    q_on, v_on = settle()
    # Jitter diagnostics: the worst (env, joint) over 40 more steps (period-2 = sign flip every step).
    v_now = _t(robot.data.joint_vel)[:, hand].abs()
    we, wj = divmod(int(v_now.argmax()), len(hand))
    trace_q, trace_v = [], []
    for _ in range(40):
        inner._apply_action()
        robot.write_data_to_sim()
        inner.sim.step(render=False)
        robot.update(dt)
        trace_q.append(round(1e3 * float(_t(robot.data.joint_pos)[we, hand[wj]] - q_on[we, hand[wj]]), 3))
        trace_v.append(round(float(_t(robot.data.joint_vel)[we, hand[wj]]), 4))
    frac = float((v_now > 0.01).float().mean())
    rep["jitter"] = {"env": we, "joint": names[hand[wj]], "dq_mrad": trace_q, "v": trace_v,
                     "frac_env_joints_above_0.01": frac}
    print("[spring] jitter", json.dumps(rep["jitter"]), flush=True)
    # Same settle with the joint friction back on (as in training).
    view.set_dof_friction_properties(fr0, indices=idx)
    _, v_fric = settle()
    rep["jitter"]["max_vel_end_with_friction"] = v_fric
    print(f"[spring] jitter with friction on: max |v| {v_fric:.4f}", flush=True)
    view.set_dof_friction_properties(fr, indices=idx)
    inner._spring_ks = None
    robot.set_joint_effort_target(torch.zeros_like(a))  # no stale spring effort
    q_off, v_off = settle()
    inner._spring_ks = ks

    g = kp * (q_off - a)
    pred = (kp * a + ks * ref + g) / (kp + ks)
    err = (q_on - pred)[:, hand].abs()
    shift = (q_on - q_off)[:, hand]
    per_joint = {}
    for k, i in enumerate(hand):
        per_joint[names[i]] = {
            "a": round(float(a[0, i]), 4), "q_off": round(float(q_off[:, i].mean()), 4),
            "q_on": round(float(q_on[:, i].mean()), 4), "pred_on": round(float(pred[:, i].mean()), 4),
            "shift_mrad": round(1e3 * float(shift[:, k].mean()), 2),
            "max_err_mrad": round(1e3 * float(err[:, k].max()), 3),
        }
    rep["equilibrium"] = {
        "settle_steps": args_cli.settle_steps, "max_vel_end": [v_on, v_off],
        "max_err_mrad": round(1e3 * float(err.max()), 3),
        "mean_abs_shift_mrad": round(1e3 * float(shift.abs().mean()), 2),
        "per_joint": per_joint,
    }
    ok &= float(err.max()) < 3e-3
    print("[spring] equilibrium", json.dumps({k: v for k, v in rep["equilibrium"].items() if k != "per_joint"}), flush=True)
    for n, row in per_joint.items():
        print(f"[spring]   {n:26s} {json.dumps(row)}", flush=True)

    # ------------------------------------------------------------ 2b. velocity readback while moving
    # 1 Hz sine targets on the flexion joints, joint friction ON (as in training); reported joint
    # velocity vs the backward difference of the joint position, spring on and off.
    import math as _m
    view.set_dof_friction_properties(fr0, indices=idx)

    def sine_run(steps):
        robot.write_joint_state_to_sim(a, torch.zeros_like(a))
        q_prev = _t(robot.data.joint_pos).clone()
        diffs, speeds = [], []
        for k in range(steps):
            tgt = a.clone()
            tgt[:, flex] = a[:, flex] + 0.3 * _m.sin(2 * _m.pi * (k + 1) * dt)
            inner._cur_targets[:] = tgt
            inner._apply_action()
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            robot.update(dt)
            q = _t(robot.data.joint_pos).clone()
            if k >= steps // 4:  # skip the start transient
                fd = (q - q_prev)[:, hand] / dt
                diffs.append(_t(robot.data.joint_vel)[:, hand] - fd)
                speeds.append(fd.abs())
            q_prev = q
        d = torch.stack(diffs)  # (T, N, H)
        return {"rms_v_minus_fd": float(d.pow(2).mean().sqrt()), "max_abs": float(d.abs().max()),
                "mean_bias": float(d.mean()), "rms_speed_fd": float(torch.stack(speeds).pow(2).mean().sqrt()),
                "per_joint_rms": {names[i]: round(float(d[..., k].pow(2).mean().sqrt()), 4)
                                  for k, i in enumerate(hand)}}

    n_sine = int(round(2.0 / dt))
    vr_on = sine_run(n_sine)
    inner._spring_ks = None
    robot.set_joint_effort_target(torch.zeros_like(a))
    vr_off = sine_run(n_sine)
    inner._spring_ks = ks
    rep["vel_readback_moving"] = {"spring_on": vr_on, "spring_off": vr_off}
    print("[spring] vel_readback_moving", json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_joint_rms"} for k, v in rep["vel_readback_moving"].items()}),
        flush=True)
    print("[spring]   per joint rms (on)", json.dumps(vr_on["per_joint_rms"]), flush=True)
    view.set_dof_friction_properties(fr, indices=idx)

    # ------------------------------------------------------------ 3. not clipped
    lim0 = view.get_dof_max_forces().clone()
    lim_t = lim0.clone()
    lim_t[:, hand] = 1e-4
    view.set_dof_max_forces(lim_t, indices=idx)
    start = _t(robot.data.default_joint_pos).clone()
    start[:, flex] = torch.clamp((ref[flex] + 0.4).unsqueeze(0), lo[flex] + 0.02, hi[flex] - 0.02)
    sign = torch.sign(start[:, flex] - ref[flex])

    def swing(steps):
        robot.write_joint_state_to_sim(start, torch.zeros_like(start))
        inner._cur_targets[:] = start
        through = torch.full_like(start[:, flex], float("inf"))
        for _ in range(steps):
            inner._apply_action()
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            robot.update(dt)
            through = torch.minimum(through, (_t(robot.data.joint_pos)[:, flex] - ref[flex]) * sign)
        return through  # min signed distance past the rest angle (< 0: swung through it)

    steps = int(round(1.0 / dt))
    thr_on = swing(steps)
    inner._spring_ks = None
    robot.set_joint_effort_target(torch.zeros_like(a))  # no stale spring effort
    thr_off = swing(steps)
    inner._spring_ks = ks
    view.set_dof_max_forces(lim0, indices=idx)
    view.set_dof_friction_properties(fr0, indices=idx)
    d0 = ((start[:, flex] - ref[flex]) * sign)
    moved_off = (d0 - thr_off).abs().max()
    rep["not_clipped"] = {
        "hand_torque_limit": 1e-4, "seconds": 1.0,
        "joints": [names[i] for i in flex],
        "start_minus_ref_rad": [round(float(v), 3) for v in d0[0]],
        "with_spring_min_dist_past_ref_rad": [round(float(v), 3) for v in thr_on.max(0).values],
        "without_spring_max_move_rad": float(moved_off),
    }
    clipped_ok = bool((thr_on < 0.1 * d0).all()) and float(moved_off) < 0.02  # PIP: limit 0 ~ ref
    rep["not_clipped"]["pass"] = clipped_ok
    ok &= clipped_ok
    print("[spring] not_clipped", json.dumps(rep["not_clipped"]), flush=True)

    rep["pass"] = bool(ok)
    print(f"[spring] PASS={bool(ok)}", flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(rep, f, indent=1)
    env.close()


if __name__ == "__main__":
    run()
    app.close()
