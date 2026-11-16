"""Historical flat7RW_fric contacts, isolated from sheet-material randomization.

Friction is drawn once from seeded buckets per global environment. All cloth
colliders use min combine; robot/table use average. Static equals dynamic.
Only the cloth collision rest shell is redrawn on reset. This recreates that
contact profile on current designs; it does not restore the historical topology.
"""
from __future__ import annotations

import math

import torch


def _range(values, name):
    low, high = map(float, values)
    if not math.isfinite(low) or not math.isfinite(high) or low < 0 or low > high:
        raise ValueError(f"{name} must contain finite nonnegative ordered bounds")
    return low, high


def initialize(env):
    cfg, rd = env.cfg.rigid_cloth, env.cfg.rigid_cloth.randomization
    bounds = _range(rd.yibo_rest_offset, "yibo_rest_offset")
    if bounds[1] >= float(cfg.contact_offset):
        raise ValueError("yibo_rest_offset must stay below rigid_cloth.contact_offset")
    generator = torch.Generator(device="cpu").manual_seed(int(rd.yibo_friction_seed))
    buckets = int(rd.friction_buckets)
    if buckets < 1:
        raise ValueError("friction_buckets must be positive")
    for name, nominal in (("cloth", 1.), ("robot", cfg.hand_static_friction), ("table", cfg.table_static_friction)):
        low, high = _range(getattr(rd, f"yibo_{name}_friction"), f"yibo_{name}_friction")
        if not rd.enabled:
            friction = torch.full((env.num_envs,), float(nominal))
        elif low == high:
            friction = torch.full((env.num_envs,), low)
        else:
            levels = torch.linspace(low, high, buckets)
            friction = levels[torch.randint(buckets, (env.num_envs,), generator=generator)]
        setattr(env, f"_yibo_{name}_friction", friction)
    for group in env.groups:
        cpu = group.env_ids.cpu()
        mu = env._yibo_cloth_friction[cpu]
        group.plate_mats[:, :, 0] = mu[:, None]
        group.plate_mats[:, :, 1] = mu[:, None]
        group.plate_mats[:, :, 2] = 0.
        view = group.articulation.root_physx_view
        local = torch.arange(len(cpu), dtype=torch.int64)
        view.set_material_properties(group.plate_mats, local)
        contact = view.get_contact_offsets().cpu()
        contact.fill_(float(cfg.contact_offset))
        view.set_contact_offsets(contact, local)
    cpu = torch.arange(env.num_envs, dtype=torch.int64)
    for name, asset, buffer in (("robot", env.robot, env._robot_mats), ("table", env.table, env._table_mats)):
        mu = getattr(env, f"_yibo_{name}_friction")
        buffer[:, :, 0] = mu[:, None]
        buffer[:, :, 1] = mu[:, None]
        buffer[:, :, 2] = 0.
        asset.root_physx_view.set_material_properties(buffer, cpu)
    env._dr_rest_offset = torch.full((env.num_envs,), float(cfg.rest_offset), device=env.device)
    env._yibo_initialized = True
    ids = torch.arange(env.num_envs, device=env.device)
    reset(env, ids)
    check(env, ids[:min(64, env.num_envs)])
    print(f"[rigid_body] yibo contact DR {'on' if rd.enabled else 'OFF'}: startup {buckets} friction buckets "
          f"cloth/robot/table {list(rd.yibo_cloth_friction)}/{list(rd.yibo_robot_friction)}/"
          f"{list(rd.yibo_table_friction)}, combine min/average/average; "
          f"reset rest offset U{list(bounds)} m, contact offset {cfg.contact_offset:g} m", flush=True)


def reset(env, ids):
    """Write a common rest shell to all shapes of each selected cloth, preserving others."""
    rd = env.cfg.rigid_cloth.randomization
    if not len(ids):
        return
    if rd.enabled:
        env._dr_rest_offset[ids] = torch.empty(len(ids), device=env.device).uniform_(*rd.yibo_rest_offset)
    for group in env.groups:
        rows, local = group.select(ids)
        if not rows.numel():
            continue
        view, cpu = group.articulation.root_physx_view, local.cpu()
        rest = view.get_rest_offsets().cpu()
        rest[cpu] = env._dr_rest_offset[ids[rows]].cpu()[:, None]
        view.set_rest_offsets(rest, cpu)


def check(env, ids):
    """Check actual sparse PhysX shape buffers against the stored startup/reset draws."""
    for group in env.groups:
        rows, local = group.select(ids)
        if not rows.numel():
            continue
        selected, cpu = ids[rows].cpu(), local.cpu()
        view = group.articulation.root_physx_view
        mats = view.get_material_properties()[cpu]
        expected = env._yibo_cloth_friction[selected, None].expand_as(mats[:, :, 0])
        if not torch.allclose(mats[:, :, 0], expected) or not torch.allclose(mats[:, :, 1], expected):
            raise RuntimeError("yibo cloth startup friction read-back mismatch")
        rest = view.get_rest_offsets()[cpu]
        if not torch.allclose(rest, env._dr_rest_offset[ids[rows]].cpu()[:, None].expand_as(rest)):
            raise RuntimeError("yibo rest offset read-back mismatch")
        contact = view.get_contact_offsets()[cpu]
        if not torch.allclose(contact, torch.full_like(contact, float(env.cfg.rigid_cloth.contact_offset))):
            raise RuntimeError("yibo contact offset read-back mismatch")
    cpu = ids.cpu()
    for name, asset in (("robot", env.robot), ("table", env.table)):
        mats = asset.root_physx_view.get_material_properties()[cpu]
        expected = getattr(env, f"_yibo_{name}_friction")[cpu, None].expand_as(mats[:, :, 0])
        if not torch.allclose(mats[:, :, 0], expected) or not torch.allclose(mats[:, :, 1], expected):
            raise RuntimeError(f"yibo {name} startup friction read-back mismatch")
