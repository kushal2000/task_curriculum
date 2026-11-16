"""Per-reset physics DR of the sheet's material.

Rigid sheet (PhysX, ``sheet.randomization``), re-drawn per env at EVERY reset: hinge damping
(log-uniform multiplier) and stiffness (uniform) via the joint tensor writes, and the plate / robot /
table frictions (multipliers quantised to ``friction_buckets`` levels -- PhysX caps a scene at 64K
materials, and each distinct (static, dynamic, restitution) triple a tensor write creates is one).
"""

from __future__ import annotations

import math

import torch


def rigid_init(env) -> None:
    """Base frictions (plates ``sheet.friction``; robot and table as ``scene_physx.set_base_materials``
    left them), the first draw for every env, and a read-back of what it wrote."""
    s = env.cfg.sheet
    view = env._sheet.root_physx_view
    env._sheet_mats = view.get_material_properties().clone()
    env._sheet_mats[..., 0:2] = float(s.friction)
    env._sheet_mats[..., 2] = 0.0
    env._robot_mats_base = env.robot.root_physx_view.get_material_properties().clone()
    env._table_mats_base = env.table.root_physx_view.get_material_properties().clone()
    env._robot_mats = env._robot_mats_base.clone()
    env._table_mats = env._table_mats_base.clone()
    ids = torch.arange(env.num_envs, device=env.device)
    view.set_material_properties(env._sheet_mats, ids.cpu())
    rigid_randomize(env, ids)
    rigid_check(env, ids[: min(64, env.num_envs)])
    rd = s.randomization
    print(f"[rigid_body] DR every reset: {'on' if rd.enabled else 'OFF'}; hinge damping {s.hinge_damping:g} x "
          f"logU{list(rd.hinge_damping_scale)}, stiffness U{list(rd.hinge_stiffness)}; friction "
          f"x{list(rd.sheet_friction_scale)} plates ({s.friction:g}), x{list(rd.robot_friction_scale)} robot, "
          f"x{list(rd.table_friction_scale)} table ({rd.friction_buckets} levels)", flush=True)


def rigid_randomize(env, ids: torch.Tensor) -> None:
    s = env.cfg.sheet
    rd = s.randomization
    n = int(ids.numel())
    if n == 0 or not rd.enabled:
        return
    lo, hi = (math.log(float(v)) for v in rd.hinge_damping_scale)
    damping = float(s.hinge_damping) * torch.exp(torch.empty(n, device=env.device).uniform_(lo, hi))
    stiffness = torch.empty(n, device=env.device).uniform_(*(float(v) for v in rd.hinge_stiffness))
    env._sheet.write_joint_damping_to_sim(damping.view(-1, 1), env_ids=ids)
    env._sheet.write_joint_stiffness_to_sim(stiffness.view(-1, 1), env_ids=ids)

    cpu = ids.cpu()
    b = int(rd.friction_buckets)

    def bucket(scale) -> torch.Tensor:
        levels = torch.linspace(float(scale[0]), float(scale[1]), b)
        return levels[torch.randint(0, b, (n,))].view(-1, 1, 1)

    env._sheet_mats[cpu, :, 0:2] = float(s.friction) * bucket(rd.sheet_friction_scale)
    env._robot_mats[cpu, :, 0:2] = env._robot_mats_base[cpu, :, 0:2] * bucket(rd.robot_friction_scale)
    env._table_mats[cpu, :, 0:2] = env._table_mats_base[cpu, :, 0:2] * bucket(rd.table_friction_scale)
    env._sheet.root_physx_view.set_material_properties(env._sheet_mats, cpu)
    env.robot.root_physx_view.set_material_properties(env._robot_mats, cpu)
    env.table.root_physx_view.set_material_properties(env._table_mats, cpu)
    env._dr_damping, env._dr_stiffness = damping, stiffness


def rigid_check(env, ids: torch.Tensor) -> None:
    """Read back what the last ``rigid_randomize(ids)`` wrote; refuse to run on a mismatch."""
    if not env.cfg.sheet.randomization.enabled:
        return
    view = env._sheet.root_physx_view
    k, cpu = int(ids.numel()), ids.cpu()
    got_d = view.get_dof_dampings().reshape(env.num_envs, -1)[cpu, 0]
    got_s = view.get_dof_stiffnesses().reshape(env.num_envs, -1)[cpu, 0]
    for label, got, want in (("damping", got_d, env._dr_damping[:k].cpu()), ("stiffness", got_s, env._dr_stiffness[:k].cpu())):
        if not torch.allclose(got.float(), want.float(), rtol=1e-4, atol=1e-9):
            raise RuntimeError(f"hinge {label} read-back mismatch: wrote {want[:4].tolist()}, read {got[:4].tolist()}")
    for label, v, want in (("sheet", view, env._sheet_mats), ("robot", env.robot.root_physx_view, env._robot_mats),
                           ("table", env.table.root_physx_view, env._table_mats)):
        if not torch.allclose(v.get_material_properties()[cpu], want[cpu], atol=1e-6):
            raise RuntimeError(f"{label} friction read-back mismatch")
