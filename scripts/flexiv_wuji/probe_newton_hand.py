"""Check the Flexiv + right Wuji (RightReal) hand as the Newton / VBD eval builds it.

    scripts/newton_py scripts/flexiv_wuji/probe_newton_hand.py \
        --task Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0 --num_envs 4 \
        env.cloth.start_height=0.20 env.cloth.soft_contact_mu=0.265 env.assets.finger_tip_friction=0.5

1. readback (randomization off, as the eval runs): hand drive stiffness == spec kp (no spring in the
   drive), armature, Coulomb friction, viscous friction and torque limit == the spec; the spring
   tensors == the spec.
2. equilibrium: hand joint friction zeroed, flexion targets a = spring rest + 0.5 rad; the env's own
   _apply_action (spring as a joint effort) runs with and without the spring. With g from the
   no-spring run, g = kp (q_off - a), the spring run must settle at (kp a + ks ref + g) / (kp + ks).
3. the sheet: no non-finite particle after the settle steps.
"""

from __future__ import annotations

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=600)
parser.add_argument("--out", default=None)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

try:  # Newton runs kit-less (as isaacsimenvs.eval.episodes)
    app = AppLauncher(args_cli).app
except ImportError as exc:
    app = None
    print(f"[newton_hand] kit-less: {exc}", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


def _t(x):
    v = getattr(x, "torch", None)
    if v is None:
        return x
    return v() if callable(v) else v


def _ring_pinky_contacts(inner) -> dict:
    """World 0: every MJWarp rigid contact touching a ring / pinky link (other side, distance), the
    cloth coupling force on those links, and whether ring-pinky shape pairs are filtered."""
    import re

    import numpy as np
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonManager

    model = NewtonManager.get_model()
    solver = NewtonManager._solver
    labels = [lab or "" for lab in model.body_label]
    rp = {b for b, lab in enumerate(labels) if re.search(r"env_0/Robot/r_(ring|pinky)", lab)}
    out = {"ring_pinky_bodies": sorted(labels[b].rsplit("/", 1)[-1] for b in rp)}
    entries = getattr(solver, "_entries", None)
    rigid = entries["rigid"].solver if entries else solver
    mjd = rigid.mjw_data
    nacon = int(wp.to_torch(mjd.nacon)[0])
    geom = wp.to_torch(mjd.contact.geom).long()[:nacon].cpu().numpy()
    world = wp.to_torch(mjd.contact.worldid).long()[:nacon].cpu().numpy()
    dist = wp.to_torch(mjd.contact.dist)[:nacon].cpu().numpy()
    g2s = wp.to_torch(rigid.mjc_geom_to_newton_shape).long().cpu().numpy()
    sb = model.shape_body.numpy()
    rows = {}
    for k in range(nacon):
        if world[k] != 0:
            continue
        bodies = []
        for side in (0, 1):
            sh = g2s[world[k], geom[k, side]] if geom[k, side] >= 0 else -1
            b = int(sb[sh]) if sh >= 0 else -1
            bodies.append(b)
        if not (bodies[0] in rp or bodies[1] in rp):
            continue
        names_ = [labels[b].split("env_0/", 1)[-1] if b >= 0 else "static/ground" for b in bodies]
        key = " <-> ".join(sorted(names_))
        r = rows.setdefault(key, {"n": 0, "min_dist_mm": 1e9})
        r["n"] += 1
        r["min_dist_mm"] = round(min(r["min_dist_mm"], 1e3 * float(dist[k])), 3)
    out["rigid_contacts"] = rows
    filt = {tuple(sorted(map(int, p))) for p in (model.shape_collision_filter_pairs or [])}
    rp_shapes = [s for s in range(len(sb)) if sb[s] in rp]
    ring = [s for s in rp_shapes if "ring" in labels[sb[s]]]
    pinky = [s for s in rp_shapes if "pinky" in labels[sb[s]]]
    pairs = [(min(i, j), max(i, j)) for i in ring for j in pinky]
    out["ring_pinky_shape_pairs"] = len(pairs)
    out["ring_pinky_pairs_filtered"] = sum(p_ in filt for p_ in pairs)
    try:
        maps = [m for m in solver._proxy_mappings if m.src_name == "rigid" and m.dst_name == "cloth"]
        cf = wp.to_torch(maps[0].coupling_forces)[:, :3].norm(dim=-1).cpu().numpy()
        s2p = dict(zip(maps[0].src_ids.numpy().tolist(), maps[0].proxy_ids_global.numpy().tolist()))
        out["cloth_force_N"] = {labels[b].rsplit("/", 1)[-1]: round(float(cf[s2p[b]]), 4) for b in sorted(rp) if b in s2p}
    except Exception as e:  # noqa: BLE001
        out["cloth_force_N"] = f"unavailable: {e}"
    return out


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.assets.num_assets_per_type = 1
    disable_randomization(env_cfg)
    use_single_object_variant()  # MuJoCo needs homogeneous worlds (as the eval does)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    spec, robot, device = inner.robot_spec, inner.robot, inner.device
    names = list(robot.data.joint_names)
    hand = [int(i) for i in inner._hand_joint_ids]
    hn = [names[i] for i in hand]
    env.reset()
    rep = {"task": args_cli.task, "overrides": hydra_args, "robot": spec.name,
           "cloth_start_height": float(env_cfg.cloth.start_height),
           "soft_contact_mu": float(env_cfg.cloth.soft_contact_mu),
           "finger_tip_friction": float(env_cfg.assets.finger_tip_friction)}
    checks = {}

    # ------------------------------------------------------------ 1. readback
    def col(attr):
        return _t(getattr(robot.data, attr))[:, hand].float()

    def want(table):
        return torch.tensor([float(table[n]) for n in hn], device=device).unsqueeze(0)

    pairs = {
        "stiffness": ("joint_stiffness", spec.hand_stiffness),
        "armature": ("joint_armature", spec.hand_armature),
        "coulomb": ("joint_friction_coeff", spec.hand_friction),
        "viscous": ("joint_viscous_friction_coeff", spec.hand_viscous_friction),
        "effort_limit": ("joint_effort_limits", spec.hand_effort_limit),
    }
    rb = {}
    for key, (attr, table) in pairs.items():
        try:
            got, w = col(attr), want(table)
            err = float(((got - w).abs() / w.abs().clamp(min=1e-9)).max())
            rb[key] = {"max_rel_err": err, "env0_first3": [round(float(v), 6) for v in got[0, :3]],
                       "spec_first3": [round(float(v), 6) for v in w[0, :3]]}
            checks[f"{key} == spec"] = err < 1e-3
        except Exception as e:  # noqa: BLE001
            rb[key] = f"unreadable: {e}"
            checks[f"{key} == spec"] = False
    ks, ref = inner._spring_ks, inner._spring_ref
    checks["spring == spec"] = ks is not None and float((ks[hand] - want(spec.hand_spring_stiffness)[0]).abs().max()) < 1e-6
    rep["readback"] = rb
    print("[newton_hand] readback", json.dumps(rb), flush=True)

    # ------------------------------------------------------------ 2. equilibrium
    zero_fric = torch.zeros(inner.num_envs, len(hand), device=device)
    robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=zero_fric, joint_ids=hand)
    lo, hi = (_t(robot.data.soft_joint_pos_limits)[0, :, k] for k in (0, 1))
    a = _t(robot.data.default_joint_pos).clone()
    flex = [i for i in hand if ("flex" in names[i] or "pip" in names[i] or "dip" in names[i]
                                 or names[i].endswith("_mcp") or names[i].endswith("_ip"))]
    a[:, flex] = (ref[flex] + 0.5).unsqueeze(0)
    a[:, hand] = torch.clamp(a[:, hand], lo[hand] + 0.05, hi[hand] - 0.05)
    dt = float(inner.cfg.sim.dt)
    kp = _t(robot.data.joint_stiffness).clone()

    def settle():
        robot.write_joint_state_to_sim(a, torch.zeros_like(a))
        inner._cur_targets[:] = a
        for _ in range(args_cli.settle_steps):
            inner._apply_action()
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            inner.scene.update(dt)
        return _t(robot.data.joint_pos).clone()

    q_on = settle()
    inner._spring_ks = None
    robot.set_joint_effort_target(torch.zeros_like(a))
    q_off = settle()
    inner._spring_ks = ks
    try:
        rep["contacts_after_off_settle"] = _ring_pinky_contacts(inner)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        rep["contacts_after_off_settle"] = f"failed: {e}"
    print("[newton_hand] ring/pinky contacts", json.dumps(rep["contacts_after_off_settle"]), flush=True)
    g = kp * (q_off - a)
    pred = (kp * a + ks * ref + g) / (kp + ks)
    err = (q_on - pred)[:, flex].abs()
    shift = (q_on - q_off)[:, flex]
    rep["equilibrium"] = {
        "max_err_mrad": round(1e3 * float(err.max()), 3),
        "mean_abs_shift_mrad": round(1e3 * float(shift.abs().mean()), 2),
        "per_joint": {names[i]: {"shift_mrad": round(1e3 * float(shift[:, k].mean()), 2),
                                 "pred_shift_mrad": round(1e3 * float((pred - q_off)[:, i].mean()), 2),
                                 "err_mrad": round(1e3 * float(err[:, k].max()), 3)}
                      for k, i in enumerate(flex)},
    }
    rep["equilibrium"]["q_off_minus_a_mrad"] = {names[i]: round(1e3 * float((q_off - a)[:, i].abs().max()), 2)
                                                 for i in flex}
    # Which joint does an effort on ONE joint move? (spring off, friction off; 0.05 N m on each probe joint)
    inner._spring_ks = None
    moved = {}
    for probe_name in ("r_ring_finger_mcp_flex", "r_pinky_mcp_flex", "r_index_finger_mcp_flex"):
        j = names.index(probe_name)
        eff = torch.zeros_like(a)
        eff[:, j] = 0.05
        robot.write_joint_state_to_sim(a, torch.zeros_like(a))
        inner._cur_targets[:] = a
        for _ in range(240):
            robot.set_joint_position_target(a)
            robot.set_joint_effort_target(eff)
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            inner.scene.update(dt)
        dq = (_t(robot.data.joint_pos) - q_off)[0]
        moved[probe_name] = {names[i]: round(1e3 * float(dq[i]), 2) for i in hand if abs(float(dq[i])) > 1e-3}
    inner._spring_ks = ks
    robot.set_joint_effort_target(torch.zeros_like(a))
    rep["single_joint_effort_mrad"] = moved
    rep["joint_names_sim_order"] = names
    print("[newton_hand] q_off - a (mrad)", json.dumps(rep["equilibrium"]["q_off_minus_a_mrad"]), flush=True)
    print("[newton_hand] single-joint effort 0.05 N m moves (mrad)", json.dumps(moved), flush=True)

    checks["spring shifts the equilibrium as predicted"] = float(err.max()) < 3e-3 and float(shift.abs().mean()) > 5e-3
    print("[newton_hand] equilibrium", json.dumps(rep["equilibrium"]), flush=True)

    # ------------------------------------------------------------ 3. sheet finite
    try:
        parts = _t(inner._particles_w())
        checks["sheet finite"] = bool(torch.isfinite(parts).all())
        rep["sheet_z_range"] = [float(parts[..., 2].min()), float(parts[..., 2].max())]
    except Exception as e:  # noqa: BLE001
        rep["sheet"] = f"unreadable: {e}"

    rep["checks"] = checks
    rep["pass"] = all(checks.values())
    print("[newton_hand] checks", json.dumps(checks), flush=True)
    print(f"[newton_hand] PASS={rep['pass']}", flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(rep, f, indent=1)
    import os

    os._exit(0)


if __name__ == "__main__":
    run()
