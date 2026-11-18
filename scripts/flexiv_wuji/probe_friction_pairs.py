"""Do the PhysX ``friction_model: cloth_min`` pairs act as intended? Measured, per env.

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_friction_pairs.py \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment <recipe> \
        --headless --num_envs 256 [--shapes box2-mid] [--slat_friction 0.364] --out report.json

Steps the SIM directly after one env.reset() (no env.step, so no task resets). Robot links carry
PhysX contact-report APIs (spawn patched here) so one rigid-contact view per robot link, filtered
against the table (and, with a single pool shape, every plate of it), gives per-env normal and
friction forces.

  (a) cloth-table: the hand is raised 10 cm, the sheet settles flat (1.5 s); every pool
      articulation is given 0.5 m/s horizontal (joint velocities 0) and the mass-weighted sheet
      speed's deceleration is fitted: mu = -dv/dt / g. Against min(slat, table).
  (c) cloth-robot (single shape only): every plate x100 mass, the lowest fingertip pad pressed
      on the plate nearest the robot at ~2 N (force loop on the arm via DLS), then slid 3 cm in x;
      |sum friction| / |sum normal| over robot-plate contacts while the pad slides. Against
      min(slat, robot).
  (b) robot-table: the sheet parked away, the pad pressed on the bare table and slid the same
      way; the ratio over robot-table contacts. Against (robot + table) / 2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--experiment", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--shapes", default=None, help="comma list overriding fold_pool.shapes")
parser.add_argument("--slat_friction", type=float, default=None,
                    help="set rigid_cloth.slat_friction in python (the yaml route refuses a float)")
parser.add_argument("--v0", type=float, default=0.5)
parser.add_argument("--press_n", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.robots import usd as scene_utils  # noqa: E402
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

# Contact reports on every robot rigid body (only the probe's robot).
_orig_robot_cfg = scene_utils.build_robot_articulation_usd_cfg


def _robot_cfg_with_contacts(*a, **k):
    cfg = _orig_robot_cfg(*a, **k)
    cfg.spawn.activate_contact_sensors = True
    return cfg


scene_utils.build_robot_articulation_usd_cfg = _robot_cfg_with_contacts

TABLE_HALF = 0.15
G = 9.81


def _st(t):
    t = t.flatten().float()
    if t.numel() == 0:
        return None
    return {"n": int(t.numel()), "min": round(float(t.min()), 4), "max": round(float(t.max()), 4),
            "mean": round(float(t.mean()), 4), "median": round(float(t.median()), 4)}


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.slat_friction is not None:
        env_cfg.rigid_cloth.slat_friction = float(args_cli.slat_friction)
    if args_cli.shapes:
        env_cfg.fold_pool.shapes = args_cli.shapes.split(",")
    # Keep the sheet where the probe puts it: no cloth-driven terminations matter (no env.step).
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    robot, device, n = inner.robot, inner.device, inner.num_envs
    spec = inner.robot_spec
    dt = float(inner.physics_dt)
    origins = inner.scene.env_origins
    arts = list(inner._pool_arts)
    pc = inner.cfg.physx_chain
    report = {"task": args_cli.task, "experiment": args_cli.experiment, "num_envs": n,
              "friction_model": pc.friction_model, "shapes": list(inner.cfg.fold_pool.shapes),
              "slat_friction_cfg": inner.cfg.rigid_cloth.friction,
              "robot_friction_cfg": inner.cfg.assets.finger_tip_friction,
              "table_friction_cfg": inner.cfg.assets.table_friction,
              "dr_enabled": bool(inner.cfg.rigid_cloth.randomization.enabled),
              "scales": {k: list(getattr(inner.cfg.rigid_cloth.randomization, k)) for k in
                         ("slat_friction_scale", "fingertip_friction_scale", "table_friction_scale")}}

    env.reset()
    s_mu = inner.dr_sample["slat_friction"].float()
    r_mu = inner.dr_sample["fingertip_friction"].float()
    t_mu = inner.dr_sample["table_friction"].float()
    report["buckets"] = {
        "distinct_slat": int(torch.unique(s_mu).numel()),
        "distinct_robot": int(torch.unique(r_mu).numel()),
        "distinct_table": int(torch.unique(t_mu).numel()),
        "slat": _st(s_mu), "robot": _st(r_mu), "table": _st(t_mu),
        "frac_slat_gt_table": round(float((s_mu > t_mu).float().mean()), 4),
        "frac_slat_gt_robot": round(float((s_mu > r_mu).float().mean()), 4),
        "frac_slat_gt_robot_or_table": round(float(((s_mu > t_mu) | (s_mu > r_mu)).float().mean()), 4),
    }
    # Read the materials back off the views too (what PhysX holds per shape).
    rm = robot.root_physx_view.get_material_properties()[..., 0]
    tm = inner.table.root_physx_view.get_material_properties()[..., 0]
    report["readback"] = {
        "robot_shapes_per_env": int(rm.shape[1]),
        "robot_max_abs_vs_dr_sample": round(float((rm - r_mu.cpu().unsqueeze(1)).abs().max()), 6),
        "table_max_abs_vs_dr_sample": round(float((tm - t_mu.cpu().unsqueeze(1)).abs().max()), 6),
    }
    print("[probe] buckets", json.dumps(report["buckets"]), json.dumps(report["readback"]), flush=True)

    # ---------------------------------------------------------------- contact views
    from isaacsim.core.simulation_manager import SimulationManager

    sv = SimulationManager.get_physics_sim_view()
    link_paths = list(robot.root_physx_view.link_paths[0])
    table_path = inner.table.root_physx_view.prim_paths[0]

    def glob(p):
        return p.replace("/env_0/", "/env_*/")

    single = len(arts) == 1
    filters = [glob(table_path)]
    if single:
        filters += [glob(p) for p in arts[0].root_physx_view.link_paths[0]]
    n_f = len(filters)
    report["contact_filters"] = filters
    views = []
    for lp in link_paths:
        bv = sv.create_rigid_body_view(glob(lp))
        order = [int(p.split("/env_")[1].split("/")[0]) for p in bv.prim_paths]
        if order != list(range(n)):
            raise RuntimeError(f"contact view env order is not 0..N-1 for {lp}")
        views.append(sv.create_rigid_contact_view(glob(lp), filter_patterns=filters,
                                                  max_contact_data_count=64 * n * n_f))

    def contacts():
        """(normal (N, F, 3), friction (N, F, 3)) summed over every robot link."""
        fn = torch.zeros((n, n_f, 3), device=device)
        ft = torch.zeros((n, n_f, 3), device=device)
        for v in views:
            fn += v.get_contact_force_matrix(dt=dt).view(n, n_f, 3).to(device)
            fr, _pts, cnt, start = v.get_friction_data(dt=dt)
            cnt = cnt.view(-1).to(device).long()
            start = start.view(-1).to(device).long()
            tot = int(cnt.sum())
            if tot == 0:
                continue
            pair = torch.repeat_interleave(torch.arange(cnt.numel(), device=device), cnt)
            first = torch.cumsum(cnt, 0) - cnt
            idx = start[pair] + (torch.arange(tot, device=device) - first[pair])
            acc = torch.zeros((cnt.numel(), 3), device=device)
            acc.index_add_(0, pair, fr.to(device)[idx])
            ft += acc.view(n, n_f, 3)
        return fn, ft

    # ---------------------------------------------------------------- arm control (DLS on a pad)
    arm = torch.as_tensor(inner._arm_joint_ids, device=device, dtype=torch.long)
    tips = torch.as_tensor(inner._fingertip_body_ids, device=device, dtype=torch.long)
    off = spec.fingertip_offset
    off = torch.tensor(off if len(off) == 5 else [off] * 5, device=device, dtype=torch.float32)
    q_tgt = robot.data.joint_pos.clone()
    robot.set_joint_position_target(q_tgt)
    ar = torch.arange(n, device=device)

    def pads():
        st = robot.data.body_state_w[:, tips]
        return st[..., :3] + quat_apply(st[..., 3:7].reshape(-1, 4), off.repeat(n, 1)).view(n, 5, 3)

    finger = torch.zeros(n, dtype=torch.long, device=device)

    def pad():
        return pads()[ar, finger]

    def dls_move(dx):
        jac = robot.root_physx_view.get_jacobians().to(device)  # (N, B-1, 6, ndof), fixed base
        b = tips[finger]
        J = jac[ar, b - 1][:, :, arm]  # (N, 6, 7)
        r = pad() - robot.data.body_pos_w[ar, b]
        Jv, Jw = J[:, :3], J[:, 3:]
        rx = torch.zeros((n, 3, 3), device=device)
        rx[:, 0, 1], rx[:, 0, 2] = -r[:, 2], r[:, 1]
        rx[:, 1, 0], rx[:, 1, 2] = r[:, 2], -r[:, 0]
        rx[:, 2, 0], rx[:, 2, 1] = -r[:, 1], r[:, 0]
        Jp = Jv - rx @ Jw
        A = Jp @ Jp.transpose(1, 2) + (0.02 ** 2) * torch.eye(3, device=device)
        dq = (Jp.transpose(1, 2) @ torch.linalg.solve(A, dx.unsqueeze(-1))).squeeze(-1)
        q_tgt[:, arm] += dq.clamp(-0.01, 0.01)
        robot.set_joint_position_target(q_tgt)

    def step():
        robot.write_data_to_sim()
        inner.sim.step(render=False)
        inner.scene.update(dt)

    def top():
        return inner._table_z_per_env + TABLE_HALF  # env-frame z of the table top

    def pool_state():
        """Per env: mass-weighted sheet velocity (N,3), min/max plate COM z (env frame), COM (N,3)."""
        vel = torch.zeros((n, 3), device=device)
        com = torch.zeros((n, 3), device=device)
        zmin = torch.full((n,), 9.0, device=device)
        zmax = torch.full((n,), -9.0, device=device)
        for s, a in enumerate(arts):
            ids = inner._cloth.members(s)
            m = a.root_physx_view.get_masses().to(device)  # (n_s, B)
            w = m / m.sum(1, keepdim=True)
            vel[ids] = (a.data.body_com_lin_vel_w * w.unsqueeze(-1)).sum(1)
            p = a.data.body_com_pos_w - origins[ids].unsqueeze(1)
            com[ids] = (p * w.unsqueeze(-1)).sum(1)
            zmin[ids] = p[..., 2].amin(1)
            zmax[ids] = p[..., 2].amax(1)
        return vel, zmin, zmax, com

    def track(goal_env, steps, vmax=0.15):
        """Move the pad toward a fixed env-frame goal (feedback DLS)."""
        for _ in range(steps):
            err = goal_env + origins - pad()
            v = err * 4.0
            nv = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            v = torch.where(nv > vmax, v / nv * vmax, v)
            dls_move(v * dt)
            step()

    # ---------------------------------------------------------------- (a) cloth-table
    p0 = pad() - origins
    goal = p0.clone()
    goal[:, 2] += 0.10
    track(goal, 180)
    fn, ft = contacts()
    vel, zmin, zmax, com = pool_state()
    low_body = (robot.data.body_pos_w[..., 2] - origins[:, 2:3]).amin(1) - top()
    pad_lowest = (pads()[..., 2] - origins[:, 2:3]).amin(1) - top()
    flat = ((zmax - zmin) < 0.004) & ((zmin - top()) < 0.02) & ((zmin - top()) > -0.005)
    report["a_settle"] = {
        "sheets_flat_on_table": int(flat.sum()),
        "plate_com_z_minus_top_mm": _st((zmin - top()) * 1e3),
        "plate_com_z_spread_mm": _st((zmax - zmin) * 1e3),
        "lowest_pad_above_top_m": _st(pad_lowest),
        "robot_table_normal_N_max": round(float(fn[:, 0].norm(dim=-1).max()), 4),
        "robot_plate_normal_N_max": round(float(fn[:, 1:].norm(dim=-1).sum(-1).max()), 4) if single else None,
        "sheet_speed_before_push_mm_s": _st(vel[:, :2].norm(dim=-1) * 1e3),
    }
    print("[probe] a_settle", json.dumps(report["a_settle"]), flush=True)
    # Slide toward the table's centre line in x (more room), 0.5 m/s.
    dirx = torch.where(com[:, 0] > 0, -1.0, 1.0)
    v0 = float(args_cli.v0)
    for s, a in enumerate(arts):
        ids = inner._cloth.members(s)
        rv = torch.zeros((a.num_instances, 6), device=device)
        rv[:, 0] = dirx[ids] * v0
        a.write_joint_state_to_sim(a.data.joint_pos.clone(), torch.zeros_like(a.data.joint_vel))
        a.write_root_velocity_to_sim(rv)
    T = 90
    speeds = torch.zeros((T + 1, n), device=device)
    vel, *_ = pool_state()
    speeds[0] = vel[:, 0] * dirx
    max_robot_contact = torch.zeros(n, device=device)
    for t in range(T):
        step()
        vel, *_ = pool_state()
        speeds[t + 1] = vel[:, 0] * dirx
        fn, _ = contacts()
        if single:
            max_robot_contact = torch.maximum(max_robot_contact, fn[:, 1:].norm(dim=-1).sum(-1))
    vel, zmin2, zmax2, com2 = pool_state()
    on_table = ((zmin2 - top()) < 0.02) & ((zmin2 - top()) > -0.005) & ((zmax2 - zmin2) < 0.004)
    # Fit the deceleration on samples between 85% and 15% of v0.
    tt = torch.arange(T + 1, device=device, dtype=torch.float32).unsqueeze(1).expand(-1, n) * dt
    sel = (speeds < 0.85 * v0) & (speeds > 0.15 * v0)
    cnt = sel.sum(0).float()
    w = sel.float()
    mt = (tt * w).sum(0) / cnt.clamp(min=1)
    mv = (speeds * w).sum(0) / cnt.clamp(min=1)
    slope = (((tt - mt) * (speeds - mv)) * w).sum(0) / (((tt - mt) ** 2) * w).sum(0).clamp(min=1e-12)
    mu_fit = -slope / G
    # Second estimate from the stopping distance: v0^2 / (2 g d).
    dist = ((com2 - com)[:, 0] * dirx).clamp(min=1e-6)
    mu_dist = v0 ** 2 / (2 * G * dist)
    # Third estimate: median per-step deceleration over the same window.
    acc = -(speeds[1:] - speeds[:-1]) / dt
    win = sel[1:] & sel[:-1]
    mu_step = torch.nanmedian(torch.where(win, acc, torch.full_like(acc, float("nan"))), dim=0).values / G
    shape_of = torch.zeros(n, dtype=torch.long, device=device)
    for s in range(len(arts)):
        shape_of[inner._cloth.members(s)] = s
    want = torch.minimum(s_mu, t_mu)
    ok = flat & on_table & (cnt >= 3)
    report["a_series"] = {"dt": dt, "envs": list(range(12)),
                          "speed": [[round(float(v), 5) for v in speeds[:, i]] for i in range(12)],
                          "slat": [round(float(s_mu[i]), 4) for i in range(12)],
                          "table": [round(float(t_mu[i]), 4) for i in range(12)],
                          "shape": [int(shape_of[i]) for i in range(12)]}
    report["a_by_shape"] = {
        inner.cfg.fold_pool.shapes[s]: {
            "n": int((ok & (shape_of == s)).sum()),
            "fit_minus_min": _st((mu_fit - want)[ok & (shape_of == s)]),
            "step_minus_min": _st((mu_step - want)[ok & (shape_of == s)]),
            "dist_minus_min": _st((mu_dist - want)[ok & (shape_of == s)]),
        } for s in range(len(arts))}
    err = (mu_fit - want)[ok]
    capped = ok & (s_mu > t_mu)
    report["a_cloth_table"] = {
        "v0": v0, "envs_used": int(ok.sum()), "envs_flat_before": int(flat.sum()),
        "envs_left_table_or_unflat": int((flat & ~on_table).sum()),
        "fit_samples": _st(cnt[ok]),
        "mu_fit_minus_min_rule": _st(err), "max_abs_err": round(float(err.abs().max()), 4),
        "mean_abs_err": round(float(err.abs().mean()), 4),
        "mu_fit_minus_avg_rule_mean_abs": round(float((mu_fit - 0.5 * (s_mu + t_mu))[ok].abs().mean()), 4),
        "mu_dist_minus_min_rule": _st((mu_dist - want)[ok]),
        "mu_step_minus_min_rule": _st((mu_step - want)[ok]),
        "mu_step_minus_avg_rule": _st((mu_step - 0.5 * (s_mu + t_mu))[ok]),
        "envs_slat_gt_table": int(capped.sum()),
        "slat_gt_table_mu_fit_minus_table": _st((mu_fit - t_mu)[capped]),
        "slat_gt_table_mu_fit_minus_slat": _st((mu_fit - s_mu)[capped]),
        "slat_le_table_mu_fit_minus_slat": _st((mu_fit - s_mu)[ok & ~capped]),
        "max_robot_plate_normal_N_during_slide": round(float(max_robot_contact.max()), 4) if single else None,
        "per_env": [{"env": int(i), "slat": round(float(s_mu[i]), 4), "table": round(float(t_mu[i]), 4),
                     "want": round(float(want[i]), 4), "mu_fit": round(float(mu_fit[i]), 4),
                     "mu_dist": round(float(mu_dist[i]), 4), "mu_step": round(float(mu_step[i]), 4),
                     "shape": int(shape_of[i])} for i in torch.nonzero(ok).flatten().tolist()],
    }
    print("[probe] a_cloth_table", json.dumps({k: v for k, v in report["a_cloth_table"].items() if k != "per_env"}),
          flush=True)

    # ---------------------------------------------------------------- press-and-slide helper
    def press_and_slide(goal_xy, surface_z, filt, label, settle=420, press=360, slide=150, v_slide=0.02):
        """Pad over goal_xy (env frame), descend under a force loop on filters `filt` to
        `press_n`, then slide +x at v_slide; returns per-env median ratio while sliding."""
        nonlocal finger
        # pick per env the finger whose pad is lowest
        finger = (pads()[..., 2]).argmin(1)
        above = torch.cat([goal_xy, (surface_z + 0.02).unsqueeze(1)], 1)
        track(above, settle, vmax=0.25)
        gap = (pad() - origins - above).norm(dim=-1)
        ref_xy = goal_xy + origins[:, :2]
        F = float(args_cli.press_n)
        ratios = []
        valid = []
        fns = []
        for t in range(press + slide):
            fn, ft = contacts()
            fz = fn[:, filt].sum(1)
            fr = ft[:, filt].sum(1)
            fnorm = fz.norm(dim=-1)
            p_before = pad()
            vz = (0.004 * (fnorm - F)).clamp(-0.03, 0.02)
            vx = torch.full((n,), v_slide if t >= press else 0.0, device=device)
            ref_xy[:, 0] += vx * dt
            dx = torch.zeros((n, 3), device=device)
            dx[:, 0] = vx * dt
            dx[:, :2] += 0.1 * (ref_xy - p_before[:, :2])
            dx[:, 2] = vz * dt
            dls_move(dx)
            step()
            if t == press:
                start_pose = [a.data.body_com_pos_w.clone() for a in arts]
            if t >= press:
                spd = (pad() - p_before)[:, :2].norm(dim=-1) / dt
                frt = fr - (fr * fz).sum(-1, keepdim=True) / fnorm.clamp(min=1e-9).unsqueeze(-1) ** 2 * fz
                ratios.append(frt.norm(dim=-1) / fnorm.clamp(min=1e-9))
                valid.append((fnorm > 0.3) & (spd > 0.005))
                fns.append(fnorm)
        R = torch.stack(ratios)
        V = torch.stack(valid)
        FN = torch.stack(fns)
        Rm = torch.where(V, R, torch.full_like(R, float("nan")))
        med = torch.nanmedian(Rm, dim=0).values
        p90 = torch.nanquantile(Rm, 0.9, dim=0)
        nval = V.sum(0)
        moved = torch.zeros(n, device=device)
        for s, a in enumerate(arts):
            moved[inner._cloth.members(s)] = (a.data.body_com_pos_w - start_pose[s]).norm(dim=-1).amax(1)
        return med, p90, nval, FN, gap, moved

    # ---------------------------------------------------------------- (c) cloth-robot
    if single:
        a = arts[0]
        view = a.root_physx_view
        m0, i0 = view.get_masses().clone(), view.get_inertias().clone()
        all_idx = torch.arange(a.num_instances, dtype=torch.int32)
        view.set_masses(m0 * 100.0, all_idx)
        view.set_inertias(i0 * 100.0, all_idx)
        p = a.data.body_com_pos_w - origins.unsqueeze(1)  # (N, B, 3)
        near = p[..., 1].argmax(1)  # the plate nearest the robot (+y)
        goal_xy = p[ar, near, :2].clone()
        goal_xy[:, 0] -= 0.015  # slide +x across the plate's centre
        surf = p[ar, near, 2]
        med, p90, nval, FN, gap, moved = press_and_slide(goal_xy, surf, list(range(1, n_f)), "c")
        want_c = torch.minimum(s_mu, r_mu)
        okc = (nval >= 10) & (moved < 0.002)
        e = (med - want_c)[okc]
        report["c_cloth_robot"] = {
            "envs_used": int(okc.sum()), "envs_with_slip_samples": int((nval >= 10).sum()),
            "plates_moved_gt_2mm_during_slide": int((moved >= 0.002).sum()),
            "approach_gap_mm": _st(gap * 1e3),
            "normal_N_while_sliding": _st(FN[-50:].mean(0)),
            "ratio_median_minus_min_rule": _st(e),
            "max_abs_err": round(float(e.abs().max()), 4) if e.numel() else None,
            "mean_abs_err": round(float(e.abs().mean()), 4) if e.numel() else None,
            "p90_minus_min_rule": _st((p90 - want_c)[okc]),
            "ratio_minus_avg_rule_mean_abs": round(float((med - 0.5 * (s_mu + r_mu))[okc].abs().mean()), 4) if e.numel() else None,
            "envs_slat_gt_robot": int((okc & (s_mu > r_mu)).sum()),
            "slat_gt_robot_ratio_minus_robot": _st((med - r_mu)[okc & (s_mu > r_mu)]),
            "slat_le_robot_ratio_minus_slat": _st((med - s_mu)[okc & (s_mu <= r_mu)]),
            "per_env": [{"env": int(i), "slat": round(float(s_mu[i]), 4), "robot": round(float(r_mu[i]), 4),
                         "want": round(float(want_c[i]), 4), "ratio_med": round(float(med[i]), 4),
                         "ratio_p90": round(float(p90[i]), 4), "n": int(nval[i])}
                        for i in torch.nonzero(okc).flatten().tolist()],
        }
        print("[probe] c_cloth_robot", json.dumps({k: v for k, v in report["c_cloth_robot"].items() if k != "per_env"}),
              flush=True)
        view.set_masses(m0, all_idx)
        view.set_inertias(i0, all_idx)
        goal = pad() - origins
        goal[:, 2] += 0.08
        track(goal, 90)

    # ---------------------------------------------------------------- (b) robot-table
    pcfg = inner.cfg.fold_pool
    for s, a in enumerate(arts):
        ids = inner._cloth.members(s)
        pose = torch.zeros((a.num_instances, 7), device=device)
        pose[:, :3] = origins[ids] + torch.tensor([0.0, pcfg.park_distance + s * pcfg.park_spacing, 0.3], device=device)
        pose[:, 3] = 1.0
        a.write_root_pose_to_sim(pose)
        a.write_root_velocity_to_sim(torch.zeros((a.num_instances, 6), device=device))
    step()
    goal_xy = torch.tensor([-0.03, 0.08], device=device).expand(n, 2).clone()
    med, p90, nval, FN, gap, _m = press_and_slide(goal_xy, top(), [0], "b")
    want_b = 0.5 * (r_mu + t_mu)
    okb = nval >= 10
    e = (med - want_b)[okb]
    report["b_robot_table"] = {
        "envs_used": int(okb.sum()),
        "approach_gap_mm": _st(gap * 1e3),
        "normal_N_while_sliding": _st(FN[-50:].mean(0)),
        "ratio_median_minus_avg_rule": _st(e),
        "max_abs_err": round(float(e.abs().max()), 4) if e.numel() else None,
        "mean_abs_err": round(float(e.abs().mean()), 4) if e.numel() else None,
        "p90_minus_avg_rule": _st((p90 - want_b)[okb]),
        "ratio_minus_min_rule_mean_abs": round(float((med - torch.minimum(r_mu, t_mu))[okb].abs().mean()), 4) if e.numel() else None,
        "ratio_minus_max_rule_mean_abs": round(float((med - torch.maximum(r_mu, t_mu))[okb].abs().mean()), 4) if e.numel() else None,
        "corr_ratio_vs_avg_rule": round(float(torch.corrcoef(torch.stack([med[okb], want_b[okb]]))[0, 1]), 4) if int(okb.sum()) > 2 else None,
        "per_env": [{"env": int(i), "robot": round(float(r_mu[i]), 4), "table": round(float(t_mu[i]), 4),
                     "want": round(float(want_b[i]), 4), "ratio_med": round(float(med[i]), 4),
                     "ratio_p90": round(float(p90[i]), 4), "n": int(nval[i])}
                    for i in torch.nonzero(okb).flatten().tolist()],
    }
    print("[probe] b_robot_table", json.dumps({k: v for k, v in report["b_robot_table"].items() if k != "per_env"}),
          flush=True)

    text = json.dumps(report, indent=1)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write(text + "\n")
    print("[probe] done", flush=True)
    import os
    os._exit(0)  # Kit shutdown hangs on bos14


if __name__ == "__main__":
    run()
