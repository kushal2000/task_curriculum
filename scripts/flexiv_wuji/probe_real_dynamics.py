"""Probe the hand / arm joint dynamics PhysX actually simulates for a Flexiv + right Wuji robot.

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_real_dynamics.py \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment flat7FlexivWujiRHL \
        --headless [--out report.json]

Everything after env creation steps the SIM directly (robot targets -> sim.step -> robot.update),
bypassing the env's action pipeline, so the commanded targets / efforts are exactly what PhysX sees.
Randomization is off. The robot has no gravity; the hand sits ~20 cm above the table (free space).

  1. readback: PhysX friction [static, dynamic, viscous], armature, max force, limits, gains per joint
     vs the RobotSpec and the URDF; env._joint_lower/upper_canon vs the URDF.
  2. friction: one joint with its drive gains zeroed, a constant feed-forward effort of
     {0, 0.5, 0.9, 1.1, 2} x its static friction (or fixed efforts when it has none); motion + the
     initial acceleration against (tau - f) / M_jj.
  3. step: drives restored, +0.3 rad / -0.3 rad position steps; rise time, overshoot, steady error,
     against f / kp.
  4. torque limit: a far target; the effort PhysX applied, from (M qdd)_j + f at the first step.
  5. limits: PIP commanded to 120 deg; each arm joint teleported 3 deg inside a limit and commanded
     5 deg beyond it (the table moved away first).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--experiment", default="", help="optional recipe; must match --task")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.eval.protocol import disable_randomization  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402

D = math.degrees


def _f(x, nd=6):
    return round(float(x), nd)


def _urdf_limits(path):
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "prismatic") and lim is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")), float(lim.get("effort", "nan")))
    return out


def _as_dict(value, names):
    if value is None:
        return None
    if isinstance(value, dict):
        return {n: float(value[n]) for n in names}
    return {n: float(value) for n in names}


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment or None)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    disable_randomization(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    spec, robot, device = inner.robot_spec, inner.robot, inner.device
    view = robot.root_physx_view
    names = list(robot.data.joint_names)
    J = len(names)
    hand = list(inner._hand_joint_ids.tolist() if hasattr(inner._hand_joint_ids, "tolist") else inner._hand_joint_ids)
    arm = list(inner._arm_joint_ids.tolist() if hasattr(inner._arm_joint_ids, "tolist") else inner._arm_joint_ids)
    hand_names = [names[i] for i in hand]
    urdf_path = env_cfg.assets.robot_urdf
    urdf = _urdf_limits(urdf_path)
    report = {"task": args_cli.task, "robot": spec.name, "urdf": urdf_path,
              "physx_view_getters": sorted(m for m in dir(view) if m.startswith("get_dof") or "mass" in m)}
    env.reset()

    # ---------------------------------------------------------------- 1. readback
    fr = view.get_dof_friction_properties()[0].cpu()  # (J, 3)
    arma = view.get_dof_armatures()[0].cpu()
    maxf = view.get_dof_max_forces()[0].cpu()
    lims = view.get_dof_limits()[0].cpu()
    kp = view.get_dof_stiffnesses()[0].cpu()
    kd = view.get_dof_dampings()[0].cpu()
    maxv = view.get_dof_max_velocities()[0].cpu()
    exp = {
        "static_friction": _as_dict(spec.hand_friction, hand_names),
        "dynamic_friction": _as_dict(spec.hand_dynamic_friction, hand_names),
        "viscous_friction": _as_dict(spec.hand_viscous_friction, hand_names),
        "armature": dict(spec.hand_armature),
        "max_force": dict(spec.hand_effort_limit) or {n: urdf[n][2] for n in hand_names},
        "stiffness": dict(spec.hand_stiffness),
        "damping": dict(spec.hand_damping),
    }
    got = {
        "static_friction": fr[:, 0], "dynamic_friction": fr[:, 1], "viscous_friction": fr[:, 2],
        "armature": arma, "max_force": maxf, "stiffness": kp, "damping": kd,
    }
    per_joint = {}
    for i, n in enumerate(names):
        per_joint[n] = {
            "static_f": _f(fr[i, 0]), "dynamic_f": _f(fr[i, 1]), "viscous_f": _f(fr[i, 2], 8),
            "armature": _f(arma[i], 8), "max_force": _f(maxf[i]), "kp": _f(kp[i]), "kd": _f(kd[i]),
            "max_vel": _f(maxv[i], 3),
            "limits_deg": [_f(D(lims[i, 0]), 3), _f(D(lims[i, 1]), 3)],
            "urdf_limits_deg": [_f(D(urdf[n][0]), 3), _f(D(urdf[n][1]), 3)] if n in urdf else None,
        }
    mism = {}
    for q, table in exp.items():
        if table is None:
            mism[q] = "spec None (asset value kept)"
            continue
        errs = [(abs(float(got[q][names.index(n)]) - v), n, v, float(got[q][names.index(n)])) for n, v in table.items()]
        e, n, v, g = max(errs)
        rel = max(abs(gg - vv) / max(abs(vv), 1e-12) for _, _, vv, gg in errs)
        mism[q] = {"max_abs": _f(e, 9), "max_rel": _f(rel, 6), "at": n, "spec": v, "physx": g,
                   "physx_min_over_hand": _f(min(float(got[q][names.index(n)]) for n in table), 8),
                   "physx_max_over_hand": _f(max(float(got[q][names.index(n)]) for n in table), 8)}
    lim_err = max(max(abs(float(lims[i, 0]) - urdf[n][0]), abs(float(lims[i, 1]) - urdf[n][1]))
                  for i, n in enumerate(names))
    mism["limits_vs_urdf_rad"] = _f(lim_err, 7)
    canon = list(spec.joint_names_canonical)
    lo_c, hi_c = inner._joint_lower_canon.cpu(), inner._joint_upper_canon.cpu()
    mism["obs_canon_limits_vs_urdf_rad"] = _f(max(
        max(abs(float(lo_c[k]) - urdf[n][0]), abs(float(hi_c[k]) - urdf[n][1])) for k, n in enumerate(canon)), 7)
    mism["arm_friction_armature"] = {n: [per_joint[n]["static_f"], per_joint[n]["dynamic_f"],
                                         per_joint[n]["viscous_f"], per_joint[n]["armature"]] for n in names[:7]}
    mism["lab_data_friction_equals_physx"] = bool(torch.allclose(robot.data.joint_friction_coeff[0].cpu(), fr[:, 0]))
    report["readback_mismatch"] = mism
    report["readback_per_joint"] = per_joint
    report["canon_limits_deg"] = {n: [_f(D(lo_c[k]), 3), _f(D(hi_c[k]), 3)] for k, n in enumerate(canon)}
    print("[probe] readback", json.dumps(mism, indent=1), flush=True)

    # ---------------------------------------------------------------- helpers
    dt = float(inner.physics_dt)
    origin = inner.scene.env_origins[0]
    kp0, kd0 = robot.data.joint_stiffness.clone(), robot.data.joint_damping.clone()
    home = robot.data.default_joint_pos.clone()

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

    def restore_gains():
        robot.write_joint_stiffness_to_sim(kp0)
        robot.write_joint_damping_to_sim(kd0)

    def mass_matrix():
        return view.get_generalized_mass_matrices()[0].to(device)

    def jid(name):
        return names.index(name)

    def fingertip_min_z_above_table():
        z = robot.data.body_pos_w[0, inner._fingertip_body_ids, 2]
        return _f(float(z.min()) - float(inner._table_z_per_env[0] + 0.15), 4)

    # Mid pose: fingers half-way through their range (index PIP 0.6, MCP 0.5 rad).
    q_mid = home.clone()
    lo, hi = robot.data.joint_pos_limits[..., 0], robot.data.joint_pos_limits[..., 1]
    q_mid[:, hand] = (0.5 * (lo + hi))[:, hand]
    for n_, v in (("r_index_finger_pip", 0.6), ("r_index_finger_mcp_flex", 0.5), ("r_index_finger_dip", 0.3)):
        q_mid[:, jid(n_)] = v
    set_state(q_mid)
    step(60)
    report["free_space"] = {"fingertip_min_z_above_table_m": fingertip_min_z_above_table(),
                            "mid_pose_hold_drift_rad": _f((robot.data.joint_pos - q_mid).abs().max(), 5)}
    mm = mass_matrix()
    report["mass_matrix_diag"] = {n: _f(mm[jid(n), jid(n)], 9) for n in hand_names}

    test_joints = ["r_index_finger_pip", "r_index_finger_mcp_flex"]

    # ---------------------------------------------------------------- 2. friction
    fric = {}
    for name in test_joints:
        j = jid(name)
        f_s, f_d = float(fr[j, 0]), float(fr[j, 1])
        base = f_s if f_s > 0 else 0.02
        res = {"static_friction": f_s, "dynamic_friction": f_d, "efforts": {}}
        for mult in (0.0, 0.5, 0.9, 1.1, 2.0, 4.0):
            for sign in (+1, -1):
                tau = sign * mult * base
                set_state(q_mid)
                restore_gains()
                step(5)
                set_state(q_mid)
                k = kp0.clone()
                d = kd0.clone()
                k[:, j] = 0.0
                d[:, j] = 0.0
                robot.write_joint_stiffness_to_sim(k)
                robot.write_joint_damping_to_sim(d)
                M = mass_matrix()
                eff = torch.zeros_like(q_mid)
                eff[:, j] = tau
                robot.set_joint_effort_target(eff)
                log = []
                step(60, log)  # 0.5 s
                q0 = float(q_mid[0, j])
                disp = [float(p[j]) - q0 for p, _ in log]
                v1, v2 = float(log[0][1][j]), float(log[1][1][j])
                # generalized force on row j from the first step: (M qdd)_j
                qdd = log[0][1] / dt
                gen = float((M[j] @ qdd))
                m_jj = float(M[j, j])
                pred_coulomb = (abs(tau) - f_d) / m_jj if abs(tau) > f_s else 0.0
                res["efforts"][f"{sign * mult:+.1f}x"] = {
                    "tau_Nm": _f(tau, 5),
                    "disp_after_1_step_rad": _f(disp[0], 6), "disp_after_6_steps_rad": _f(disp[5], 5),
                    "disp_after_0.5s_rad": _f(disp[-1], 5),
                    "moved": abs(disp[-1]) > 1e-3,
                    "qdd_first_step": _f(v1 / dt, 3),
                    "qdd_pred_coulomb": _f(sign * pred_coulomb, 3),
                    "qdd_pred_no_friction": _f(tau / m_jj, 3),
                    "gen_force_row_j_first_step_Nm": _f(gen, 6),
                    "net_minus_tau_Nm (friction seen)": _f(gen - tau, 6),
                }
        restore_gains()
        fric[name] = res
    report["friction_test"] = fric
    print("[probe] friction", json.dumps(fric, indent=1), flush=True)

    # ---------------------------------------------------------------- 3. step response
    def step_resp(name, delta, n_steps=120):
        j = jid(name)
        restore_gains()
        set_state(q_mid)
        step(5)
        set_state(q_mid)
        tgt = q_mid.clone()
        tgt[:, j] += delta
        robot.set_joint_position_target(tgt)
        log = []
        step(n_steps, log)
        q0 = float(q_mid[0, j])
        y = torch.tensor([(float(p[j]) - q0) / delta for p, _ in log])
        t = torch.arange(1, n_steps + 1) * dt

        def first(cond):
            idx = torch.nonzero(cond)
            return float(t[idx[0, 0]]) if len(idx) else None

        t10, t90 = first(y >= 0.1), first(y >= 0.9)
        k_, f_ = float(kp[j]), float(fr[j, 1])
        i05 = int(round(0.5 / dt)) - 1
        return {
            "delta_rad": delta, "kp": _f(k_, 4), "kd": _f(float(kd[j]), 5), "friction": f_,
            "rise_10_90_s": _f(t90 - t10, 4) if (t10 is not None and t90 is not None) else None,
            "t90_s": t90, "overshoot_pct": _f(max(0.0, float(y.max()) - 1.0) * 100, 2),
            # Against the FINAL displacement (where it stalls), when it never reaches 90% of the step.
            "rise_10_90_of_final_s": (lambda a, b: _f(b - a, 4) if (a is not None and b is not None) else None)(
                first(y >= 0.1 * float(y[-1])), first(y >= 0.9 * float(y[-1]))),
            "t_stall_s": first((y - float(y[-1])).abs() < 1e-4),
            "final_fraction": _f(float(y[-1]), 4),
            "ss_err_0.5s_rad": _f((1.0 - float(y[i05])) * delta, 5),
            "ss_err_1.0s_rad": _f((1.0 - float(y[-1])) * delta, 5),
            "pred_f_over_kp_rad": _f(f_ / k_, 5),
            "vel_at_0.5s": _f(float(log[i05][1][j]), 5),
            "isaaclab_applied_torque_est_at_end": _f(float(robot.data.applied_torque[0, j]), 5),
        }

    report["step_response"] = {f"{n}{s:+.1f}": step_resp(n, s) for n in test_joints for s in (0.3, -0.3)}
    print("[probe] step", json.dumps(report["step_response"], indent=1), flush=True)

    # ---------------------------------------------------------------- 4. torque limit
    sat = {}
    for name, tgt_val in (("r_index_finger_mcp_flex", 1.55), ("r_index_finger_dip", 1.55),
                          ("r_index_finger_pip", 1.7), ("r_thumb_cmc_flex", 1.25)):
        j = jid(name)
        restore_gains()
        q0 = q_mid.clone()
        q0[:, j] = float(lo[0, j]) + 0.02
        set_state(q0)
        step(5)
        set_state(q0)
        M = mass_matrix()
        tgt = q0.clone()
        tgt[:, j] = tgt_val
        robot.set_joint_position_target(tgt)
        log = []
        step(3, log)
        gen = float(M[j] @ (log[0][1] / dt))  # generalized force at row j, first step (v0 = 0)
        f_s = float(fr[j, 0])
        sat[name] = {
            "err_rad": _f(tgt_val - float(q0[0, j]), 4),
            "kp_x_err_Nm": _f(float(kp[j]) * (tgt_val - float(q0[0, j])), 4),
            "physx_max_force_Nm": _f(float(maxf[j]), 5),
            "spec_effort_limit_Nm": _f(spec.hand_effort_limit.get(name, float("nan")), 5),
            "drive_effort_est_Nm (Mqdd + f_static)": _f(gen + f_s, 5),
            "Mqdd_Nm": _f(gen, 5),
            "isaaclab_applied_torque_Nm": _f(float(robot.data.applied_torque[0, j]), 5),
        }
    if hasattr(view, "get_dof_projected_joint_forces"):
        try:
            report["projected_joint_forces_sample"] = [_f(v, 5) for v in view.get_dof_projected_joint_forces()[0].cpu()]
        except Exception as e:  # noqa: BLE001
            report["projected_joint_forces_sample"] = repr(e)
    report["torque_limit"] = sat
    print("[probe] torque", json.dumps(sat, indent=1), flush=True)

    # ---------------------------------------------------------------- 5. limits
    lim_res = {}
    restore_gains()
    for n_ in [n for n in hand_names if n.endswith("_pip")]:
        j = jid(n_)
        set_state(q_mid)
        tgt = q_mid.clone()
        tgt[:, j] = math.radians(120.0)
        robot.set_joint_position_target(tgt)
        step(120)
        lim_res[n_] = {"cmd_deg": 120.0, "stop_deg": _f(D(robot.data.joint_pos[0, j]), 3),
                       "urdf_upper_deg": _f(D(urdf[n_][1]), 3)}
    # Table away (kinematic): +5 m in z below, so arm moves cannot touch it.
    tp = inner.table.data.root_state_w[:, :7].clone()
    tp[:, 2] -= 5.0
    inner.table.write_root_pose_to_sim(tp)
    for n_ in names[:7]:
        j = jid(n_)
        for side in (0, 1):
            lim_v = urdf[n_][side]
            inside = lim_v + (math.radians(3.0) if side == 0 else -math.radians(3.0))
            beyond = lim_v + (-math.radians(5.0) if side == 0 else math.radians(5.0))
            q0 = home.clone()
            q0[:, hand] = q_mid[:, hand]
            q0[:, j] = inside
            set_state(q0)
            step(10)
            tgt = q0.clone()
            tgt[:, j] = beyond
            robot.set_joint_position_target(tgt)
            step(120)
            others = [k for k in arm if k != j]
            lim_res[f"{n_}_{'lower' if side == 0 else 'upper'}"] = {
                "limit_deg": _f(D(lim_v), 3), "cmd_deg": _f(D(beyond), 3),
                "stop_deg": _f(D(robot.data.joint_pos[0, j]), 3),
                "past_limit_deg": _f(D(robot.data.joint_pos[0, j] - lim_v) * (1 if side else -1), 4),
                "other_arm_max_dev_deg": _f(D((robot.data.joint_pos[0, others] - q0[0, others]).abs().max()), 4),
            }
    report["limits"] = lim_res
    print("[probe] limits", json.dumps(lim_res, indent=1), flush=True)

    text = json.dumps(report, indent=2)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write(text + "\n")
    print("[probe] done", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    run()
