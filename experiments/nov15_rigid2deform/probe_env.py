"""Startup proofs for Isaacsimenvs-RigidBodyCloth-Direct-v0 (fails loudly on any mismatch).

    OMNI_KIT_ACCEPT_EULA=YES .venv/isaacsim/bin/python experiments/nov15_rigid2deform/probe_env.py \\
        --headless --num_envs 256

1. authored dims: every env's two colliders have its design's size, position and thickness;
2. rest: after a reset and 60 zero-action steps every cloth lies flat on its table (plate centres at
   table top + t/2, hinge ~0) and is NOT folded;
   checks 2-4 skip envs where a fingertip came within 3 cm of the cloth (Play's random reset arm pose
   can rest the hand on it: the policy's business, not the cloth's physics); the count is printed;
3. hinge sign: +90 deg lifts the + plate's far edge;
4. a perfect fold: the hinge at 180 deg puts every tracked corner on its target (fold_error ~ 0) and
   passes `is_folded` for every design; still so after 30 physics steps;
5. per-reset DR: resetting envs 0-31 changes their hinge damping / stiffness and plate / robot /
   table frictions and leaves envs 32+ untouched (the untouched ones are the negative control);
6. the table's base friction is ``assets.table_friction`` (Play gave it the robot's);
7. with ``env.cloth.task=unfold`` (pass it on the command line): every cloth starts folded (hinge
   ~180 deg) and is NOT a success; laid flat, every one is.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--experiment", default="rigid_body_cloth")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402

TASK = "Isaacsimenvs-RigidBodyCloth-Direct-v0"
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"[probe] {'PASS' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


@hydra_task_config_with_yaml(TASK, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(TASK, cfg=env_cfg)
    e = env.unwrapped
    n, dev = e.num_envs, e.device
    dims = e._dims

    # 1. authored dims
    from pxr import Usd, UsdGeom  # noqa: F401
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    worst = 0.0
    for i in range(0, n, max(1, n // 32)):
        L, W, T = (float(v) for v in e._dims_np[i])
        for name, cx in (("minus", -L / 4), ("plus", L / 4)):
            p = stage.GetPrimAtPath(f"/World/envs/env_{i}/RigidCloth/{name}/collider")
            sc = p.GetAttribute("xformOp:scale").Get()
            tr = p.GetAttribute("xformOp:translate").Get()
            worst = max(worst, abs(sc[0] - L / 2), abs(sc[1] - W), abs(sc[2] - T), abs(tr[0] - cx))
    check(worst < 1e-9, f"authored collider dims match the designs (worst {worst:.2e} m)")

    # 2. rest
    zero = torch.zeros((n, e.cfg.action_space), device=dev)
    env.reset()
    near = torch.full((n,), 9.0, device=dev)

    def hand_gap() -> torch.Tensor:
        ft = e.robot.data.body_pos_w[:, e._fingertip_ids]
        return torch.cdist(ft, e._particles_w()).amin(dim=(1, 2))

    unfold = e.cfg.cloth.task == "unfold"
    if unfold:
        q0 = e._rigid_cloth.data.joint_pos[:, 0]
        check(float((q0 - math.pi).abs().max()) < math.radians(1), f"unfold: every cloth starts folded (hinge {math.degrees(float(q0.min())):.1f}-{math.degrees(float(q0.max())):.1f} deg)")

    for _ in range(60):
        env.step(zero)
        near = torch.minimum(near, hand_gap())
    free = near > 0.03
    print(f"[probe] {int((~free).sum())}/{n} envs had a fingertip within 3 cm of the cloth: skipped in 2-4", flush=True)
    pos = e._rigid_cloth.data.body_link_pos_w
    top = e.scene.env_origins[:, 2] + e._table_z + e._table_half_height
    # body frame = cloth frame (mid-plane); at rest the - plate sits t/2 above the table top
    dz = (pos[:, :1, 2] - (top + 0.5 * dims[:, 2]).unsqueeze(1)).abs() if unfold else \
        (pos[:, :, 2] - (top + 0.5 * dims[:, 2]).unsqueeze(1)).abs()
    q = e._rigid_cloth.data.joint_pos[:, 0]
    dz, q = dz[free], q[free]
    check(float(dz.max()) < 2e-3, f"cloths rest on their tables: |z - (top + t/2)| max {float(dz.max()) * 1e3:.2f} mm")
    if unfold:
        check(not bool(e.is_folded()[free].any()), "unfold: no folded start counts as unfolded")
    else:
        check(float(q.abs().max()) < math.radians(3), f"hinges flat at rest: max |q| {math.degrees(float(q.abs().max())):.2f} deg")
        check(not bool(e.is_folded()[free].any()), "no flat cloth counts as folded")
    fe_flat = e.fold_error()
    print(f"[probe] flat fold_error {float(fe_flat.min()):.3f}-{float(fe_flat.max()):.3f} m", flush=True)

    def set_hinge(angle: float) -> None:  # noqa: E306
        a = torch.full((n, 1), angle, device=dev)
        e._rigid_cloth.write_joint_state_to_sim(a, torch.zeros_like(a))
        e._rigid_cloth.set_joint_position_target(a)
        e.scene.write_data_to_sim()
        e.sim.step(render=False)
        e.scene.update(e.physics_dt)

    if unfold:
        set_hinge(0.0)
        for _ in range(10):
            e._rigid_cloth.set_joint_position_target(torch.zeros((n, 1), device=dev))
            e.scene.write_data_to_sim()
            e.sim.step(render=False)
            e.scene.update(e.physics_dt)
        ok = e.is_folded()[free]
        check(bool(ok.all()), f"unfold: laid flat, {int(ok.sum())}/{int(free.sum())} cloths pass the unfold test")
        print(f"[probe] {'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}", flush=True)
        env.close()
        return

    # 3. hinge sign: the + plate's far corners rise
    kp_before = e.cloth_keypoints_w()[:, 2:, 2].clone()   # far corners
    set_hinge(math.radians(90))
    rise = e.cloth_keypoints_w()[:, 2:, 2] - kp_before
    free = free & (hand_gap() > 0.03)
    palm = e.robot.data.body_pos_w[:, e._palm_id] - e.scene.env_origins
    for i in ((rise.amin(dim=1) < 0.25 * dims[:, 0]) & free).nonzero(as_tuple=True)[0][:6].tolist():
        root = e._rigid_cloth.data.root_pos_w[i] - e.scene.env_origins[i]
        print(f"[probe]   env {i}: dims {[round(float(v), 3) for v in dims[i]]} root {[round(float(v), 3) for v in root]} "
              f"yaw-q {[round(float(v), 2) for v in e._rigid_cloth.data.root_quat_w[i]]} q {math.degrees(float(e._rigid_cloth.data.joint_pos[i, 0])):.1f} deg "
              f"rise {[round(float(v), 3) for v in rise[i]]} palm {[round(float(v), 3) for v in palm[i]]}", flush=True)
    check(bool((rise > 0.25 * dims[:, :1])[free].all()), f"+90 deg lifts the far corners (min rise {float(rise[free].min()) * 100:.1f} cm)")

    # 4. a perfect fold
    set_hinge(math.pi)
    free = free & (hand_gap() > 0.03)
    fe = e.fold_error()[free]
    check(float(fe.max()) < 2e-3, f"hinge at 180 deg: fold_error max {float(fe.max()) * 1e3:.2f} mm")
    folded = e.is_folded()[free]
    check(bool(folded.all()), f"hinge at 180 deg: is_folded for {int(folded.sum())}/{int(free.sum())} envs")
    for _ in range(30):
        e._rigid_cloth.set_joint_position_target(torch.full((n, 1), math.pi, device=dev))
        e.scene.write_data_to_sim()
        e.sim.step(render=False)
        e.scene.update(e.physics_dt)
    free = free & (hand_gap() > 0.03)
    fe = torch.where(free, e.fold_error(), torch.zeros_like(e.fold_error()))
    bad = (fe > 5e-3).nonzero(as_tuple=True)[0]
    for i in bad[:6].tolist():
        root = e._rigid_cloth.data.root_pos_w[i] - e.scene.env_origins[i]
        print(f"[probe]   env {i}: dims {[round(float(v), 3) for v in dims[i]]} root {[round(float(v), 3) for v in root]} "
              f"q {math.degrees(float(e._rigid_cloth.data.joint_pos[i, 0])):.1f} deg err {float(fe[i]) * 1e3:.1f} mm", flush=True)
    check(float(fe.max()) < 5e-3, f"after 30 physics steps folded: fold_error max {float(fe.max()) * 1e3:.2f} mm "
          f"({int(bad.numel())}/{n} envs > 5 mm)")
    print(f"[probe] footprint at the fold {float(e.footprint_ratio().min()):.3f}-{float(e.footprint_ratio().max()):.3f}", flush=True)

    # 5. per-reset DR, negative control = the envs not reset
    view = e._rigid_cloth.root_physx_view
    snap = {
        "damping": view.get_dof_dampings().clone(),
        "stiffness": view.get_dof_stiffnesses().clone(),
        "cloth mu": view.get_material_properties().clone(),
        "robot mu": e.robot.root_physx_view.get_material_properties().clone(),
        "table mu": e.table.root_physx_view.get_material_properties().clone(),
    }
    ids = torch.arange(32, device=dev)
    e._reset_idx(ids)
    after = {
        "damping": view.get_dof_dampings(),
        "stiffness": view.get_dof_stiffnesses(),
        "cloth mu": view.get_material_properties(),
        "robot mu": e.robot.root_physx_view.get_material_properties(),
        "table mu": e.table.root_physx_view.get_material_properties(),
    }
    for k in snap:
        a, b = snap[k].reshape(n, -1), after[k].reshape(n, -1)
        changed = (a[:32] != b[:32]).any(dim=1).float().mean()
        untouched = bool((a[32:] == b[32:]).all())
        check(float(changed) > 0.5 and untouched,
              f"DR {k}: {float(changed) * 100:.0f}% of reset envs changed, others untouched: {untouched}")

    # 6. the table's base friction
    tf = float(e.cfg.assets.table_friction)
    base = e._table_mats_base[..., :2]
    check(bool(torch.allclose(base, torch.full_like(base, tf))), f"table base friction = assets.table_friction {tf}")

    print(f"[probe] {'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}", flush=True)
    env.close()


run()
sys.stdout.flush()
os._exit(0 if not FAILS else 1)
