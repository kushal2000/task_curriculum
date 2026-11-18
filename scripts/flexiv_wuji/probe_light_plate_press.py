"""Finger contact on light plates: press a fingertip pad on one plate of box2-mid, then drag it over
the crease onto the other plate. PhysX fold pool, per env (sheet sizes 10-20 cm from the pool's DR).

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_light_plate_press.py \
        --experiment flat7FlexivWujiRHL --headless --out report.json [env.rigid_cloth.density=2.0 ...]

Per press force (--forces): the pad is brought over the centre of plate 0, lowered under a force loop
and held (3 s), then dragged to the centre of plate 1 (3.3 s). Reported per env over press + drag:
  hinge_max_deg / hinge_speed_max  the hinge opening while a finger presses a flat sheet ("flick")
  other_rise_mm                    how far the plate that is NOT pressed lifted (COM z)
  sink_mm                          how far the lowest plate went below its resting height (into the table)
  pad_minus_plate_mm               pad height above the pressed plate's COM while held, minus that at
                                   first contact: negative = the finger sinks into the plate
  plate_speed_max, sheet_shift_mm, normal force, non-finite envs
"""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--experiment", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--forces", default="2,10")
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
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

# Contact reports on every robot rigid body (only the probe's robot).
_orig_robot_cfg = scene_utils.build_robot_articulation_usd_cfg


def _robot_cfg_with_contacts(*a, **k):
    cfg = _orig_robot_cfg(*a, **k)
    cfg.spawn.activate_contact_sensors = True
    return cfg


scene_utils.build_robot_articulation_usd_cfg = _robot_cfg_with_contacts

TABLE_HALF = 0.15


def _st(t):
    t = t.flatten().float()
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return None
    return {"median": round(float(t.median()), 3), "p90": round(float(torch.quantile(t, 0.9)), 3),
            "max": round(float(t.max()), 3), "min": round(float(t.min()), 3)}


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.fold_pool.shapes = ["box2-mid"]
    env_cfg.fold_pool.weights = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    robot, device, n = inner.robot, inner.device, inner.num_envs
    spec = inner.robot_spec
    dt = float(inner.physics_dt)
    origins = inner.scene.env_origins
    arts = list(inner._pool_arts)
    a = arts[0]
    r = inner.cfg.rigid_cloth
    report = {"density": r.density, "joint_damping": r.joint_damping, "joint_armature": r.joint_armature,
              "num_envs": n, "hydra": hydra_args}
    print("[press] cfg", json.dumps(report), flush=True)
    env.reset()

    # ---------------------------------------------------------------- contact views
    from isaacsim.core.simulation_manager import SimulationManager

    sv = SimulationManager.get_physics_sim_view()
    link_paths = list(robot.root_physx_view.link_paths[0])
    table_path = inner.table.root_physx_view.prim_paths[0]

    def glob(p):
        return p.replace("/env_0/", "/env_*/")

    single = True
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


    def plates():
        return a.data.body_com_pos_w - origins.unsqueeze(1)  # (N, 2, 3)

    # Hand up 10 cm, sheet settles.
    goal = pad() - origins
    goal[:, 2] += 0.10
    track(goal, 240)
    rest = plates().clone()
    q0 = a.data.joint_pos.clone()
    mass = a.root_physx_view.get_masses().to(device).sum(1)
    report["sheet_mass_g"] = _st(mass * 1e3)
    report["rest_plate_z_minus_top_mm"] = _st((rest[..., 2] - top().unsqueeze(1)) * 1e3)
    report["rest_hinge_deg"] = _st(torch.rad2deg(q0.abs()))
    print("[press] rest", json.dumps({k: report[k] for k in ("sheet_mass_g", "rest_plate_z_minus_top_mm", "rest_hinge_deg")}), flush=True)

    size = inner._env_size.to(device).float()
    on0 = (rest[..., 2].amin(1) - top()) > -0.02
    report["envs_on_table_at_rest"] = int(on0.sum())
    PRESS, DRAG, HOLD = 720, 400, 120

    def off_table():
        return (plates()[..., 2].amin(1) - top()) < -0.05

    for F in [float(v) for v in args_cli.forces.split(",")]:
        p = plates()
        ok0 = ~off_table()
        base = p.clone()
        finger = pads()[..., 2].argmin(1)
        above = p[:, 0].clone()
        above[:, 2] += 0.03
        track(above, 420, vmax=0.25)
        ok0 &= ~off_table() & ((pad() - origins - above).norm(dim=-1) < 0.01)
        p = plates()
        ref_xy = (p[:, 0, :2] + origins[:, :2]).clone()
        z = lambda: torch.zeros(n, device=device)  # noqa: E731
        m = {k: z() for k in ("hinge", "hspeed", "rise", "sink", "pspeed", "fmax", "shift0")}
        d = {k: z() for k in ("hinge", "rise", "pspeed", "fmax")}
        touch_h = torch.full((n,), float("nan"), device=device)
        hold_h, hold_f, hold_off = [], [], []
        finite = torch.ones(n, dtype=torch.bool, device=device)
        for t in range(PRESS + DRAG):
            fn, _ft = contacts()
            fnorm = fn[:, 1:].sum(1).norm(dim=-1)
            pb = pad()
            if t == PRESS:
                fell_press = off_table()
                pl = plates()
                v_drag = (pl[:, 1, :2] - pl[:, 0, :2]) / (DRAG * dt)
                ref_xy = pb[:, :2].clone()
            if t >= PRESS:
                ref_xy += v_drag * dt
            dx = torch.zeros((n, 3), device=device)
            dx[:, :2] = 0.1 * (ref_xy - pb[:, :2])
            dx[:, 2] = (0.01 * (fnorm - F)).clamp(-0.03, 0.02) * dt
            dls_move(dx)
            step()
            pl = plates()
            finite &= torch.isfinite(pl).all(-1).all(-1) & torch.isfinite(a.data.joint_pos).all(-1)
            q = torch.rad2deg(torch.nan_to_num(a.data.joint_pos - q0).abs().amax(1))
            dz = torch.nan_to_num(pl[..., 2] - base[..., 2])
            ps = torch.nan_to_num(a.data.body_com_lin_vel_w).norm(dim=-1).amax(1)
            acc = m if t < PRESS else d
            acc["hinge"] = torch.maximum(acc["hinge"], q)
            acc["rise"] = torch.maximum(acc["rise"], dz[:, 1])
            acc["pspeed"] = torch.maximum(acc["pspeed"], ps)
            acc["fmax"] = torch.maximum(acc["fmax"], fnorm)
            if t < PRESS:
                m["hspeed"] = torch.maximum(m["hspeed"], torch.nan_to_num(a.data.joint_vel).abs().amax(1))
                m["sink"] = torch.maximum(m["sink"], (-dz).amax(1))
                m["shift0"] = torch.maximum(m["shift0"], (pl[:, 0, :2] - base[:, 0, :2]).norm(dim=-1))
                h = (pad() - origins)[:, 2] - pl[:, 0, 2]
                touch_h = torch.where(torch.isnan(touch_h) & (fnorm > 0.02), h, touch_h)
                if t >= PRESS - HOLD:
                    hold_h.append(h)
                    hold_f.append(fnorm)
                    hold_off.append(((pad() - origins)[:, :2] - pl[:, 0, :2]).norm(dim=-1))
        fell_drag = off_table() & ~fell_press
        hh, hf, ho = torch.stack(hold_h).mean(0), torch.stack(hold_f).mean(0), torch.stack(hold_off).mean(0)
        touched = ok0 & ~torch.isnan(touch_h)
        ok = touched & ~fell_press
        under = ok & (ho < size / 4 - 0.005)  # the pad is still over plate 0 (short side L/2)
        g = lambda t, k=ok: _st(t[k])  # noqa: E731
        res = {
            "envs": int(ok0.sum()), "touched": int(touched.sum()), "nonfinite": int((~finite).sum()),
            "sheet_fell_off_during_press": int((ok0 & fell_press).sum()),
            "sheet_fell_off_during_drag": int((ok0 & fell_drag).sum()),
            "pad_still_over_plate_at_hold": int(under.sum()),
            "hold_normal_N": g(hf), "peak_normal_N": g(m["fmax"]),
            "press_hinge_max_deg": g(m["hinge"]), "press_hinge_speed_max_rad_s": g(m["hspeed"]),
            "press_other_plate_rise_mm": g(m["rise"] * 1e3),
            "press_sink_mm": g(m["sink"] * 1e3),
            "press_plate0_shift_mm": g(m["shift0"] * 1e3),
            "press_plate_speed_max_m_s": g(m["pspeed"]),
            "hold_pad_into_plate_mm": _st(((touch_h - hh) * 1e3)[under]),
            "drag_hinge_max_deg": g(d["hinge"], ok & ~fell_drag),
            "drag_other_plate_rise_mm": g(d["rise"] * 1e3, ok & ~fell_drag),
            "drag_plate_speed_max_m_s": g(d["pspeed"], ok & ~fell_drag),
            "drag_peak_normal_N": g(d["fmax"], ok & ~fell_drag),
        }
        report[f"press_{F:g}N"] = res
        print(f"[press] F={F:g}", json.dumps(res), flush=True)
        goal = pad() - origins
        goal[:, 2] += 0.06
        track(goal, 180)

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write(json.dumps(report, indent=1) + "\n")
    print("[press] done", flush=True)
    import os
    os._exit(0)  # Kit shutdown hangs on bos14


if __name__ == "__main__":
    run()
