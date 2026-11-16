"""Per-sheet material draws mapped to each topology's physical joints and contacts.

Bending rigidity/viscosity multiply each hinge's geometric edge/span factor.
Density scales mass and all inertia entries. Material draws follow global reset-ID
order, independent of sparse articulation grouping. Contact friction is an absolute
cloth/table or cloth/hand pair coefficient, using authored multiply-combine materials.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .bending import hinge_geometry_factors
from .inertia import hinge_inertias
from .scene import pair_materials


_MATERIAL_FIELDS = ("bending_rigidity", "bending_viscosity", "density",
                    "table_static_friction", "hand_static_friction", "dynamic_friction_ratio")


def rigid_init(env) -> None:
    """Cache physical references; apply nominal material settings before optional DR."""
    cfg = env.cfg.rigid_cloth
    for group in env.groups:
        view = group.articulation.root_physx_view
        group.plate_mats = view.get_material_properties().clone()
        group.plate_mats[:] = torch.tensor(pair_materials(0., 0., 1.)["cloth"])
        view.set_material_properties(group.plate_mats, torch.arange(len(group.env_ids)))
        group.mass_base = view.get_masses().cpu().clone()
        group.inertia_base = view.get_inertias().cpu().clone()
        references, inertias, factors = {}, [], []
        for global_id in group.env_ids.cpu().tolist():
            dimensions = tuple(float(value) for value in env._dimensions_cpu[global_id])
            if dimensions not in references:
                spec = group.design.resolve(*dimensions)
                references[dimensions] = hinge_inertias(spec, float(cfg.density)), hinge_geometry_factors(spec)
            inertia, factor = references[dimensions]
            inertias.append(inertia)
            factors.append(factor)
        shape = (len(group.env_ids), group.articulation.num_joints)
        group.hinge_inertia = torch.zeros(shape, dtype=torch.float32, device=env.device)
        group.hinge_geometry = torch.zeros_like(group.hinge_inertia)
        for destination, values in ((group.hinge_inertia, inertias), (group.hinge_geometry, factors)):
            destination[:, group.joint_ids] = torch.as_tensor(np.stack(values), dtype=torch.float32, device=env.device)
    env._robot_mats = env.robot.root_physx_view.get_material_properties().clone()
    env._table_mats = env.table.root_physx_view.get_material_properties().clone()
    for name in _MATERIAL_FIELDS:
        setattr(env, f"_dr_{name}", torch.full((env.num_envs,), float(getattr(cfg, name)), device=env.device))
    env._dr_armature_scale = torch.zeros(env.num_envs, device=env.device)
    ids = torch.arange(env.num_envs, device=env.device)
    _apply(env, ids)
    rigid_randomize(env, ids)
    rigid_check(env, ids[:min(64, env.num_envs)])
    rd = cfg.randomization
    print(f"[rigid_body] material DR {'on' if rd.enabled else 'OFF'}: "
          f"B logU{list(rd.bending_rigidity)} N m, C logU{list(rd.bending_viscosity)} N m s; "
          f"density logU{list(rd.density)} kg/m²; effective table/hand static friction "
          f"U{list(rd.table_static_friction)}/U{list(rd.hand_static_friction)}, "
          f"dynamic ratio U{list(rd.dynamic_friction_ratio)}; multiply combine", flush=True)


def rigid_randomize(env, ids: torch.Tensor) -> None:
    """Draw one set of material parameters per selected global environment."""
    rd, count = env.cfg.rigid_cloth.randomization, len(ids)
    if count == 0 or not rd.enabled:
        return
    for name in ("bending_rigidity", "bending_viscosity", "density"):
        low, high = (math.log(float(value)) for value in getattr(rd, name))
        getattr(env, f"_dr_{name}")[ids] = torch.empty(count, device=env.device).uniform_(low, high).exp_()
    env._dr_armature_scale[ids] = torch.empty(count, device=env.device).uniform_(*rd.hinge_armature_scale)
    for name in ("table_static_friction", "hand_static_friction", "dynamic_friction_ratio"):
        levels = torch.linspace(*getattr(rd, name), int(rd.friction_buckets))
        getattr(env, f"_dr_{name}")[ids] = levels[torch.randint(len(levels), (count,))].to(env.device)
    _apply(env, ids)


def _apply(env, ids):
    """Apply nominal or drawn material parameters without compounding previous resets."""
    for group in env.groups:
        rows, local_ids = group.select(ids)
        if not rows.numel():
            continue
        selected, cpu = ids[rows], local_ids.cpu()
        asset, factors = group.articulation, group.hinge_geometry[local_ids]
        asset.write_joint_stiffness_to_sim(env._dr_bending_rigidity[selected, None] * factors, env_ids=local_ids)
        asset.write_joint_damping_to_sim(env._dr_bending_viscosity[selected, None] * factors, env_ids=local_ids)
        ratio = env._dr_density[selected] / float(env.cfg.rigid_cloth.density)
        armature = float(env.cfg.rigid_cloth.hinge_armature) + group.hinge_inertia[local_ids] * (
            ratio * env._dr_armature_scale[selected]
        )[:, None]
        asset.write_joint_armature_to_sim(armature, env_ids=local_ids)
        masses, inertias = asset.root_physx_view.get_masses(), asset.root_physx_view.get_inertias()
        masses[cpu] = group.mass_base[cpu] * ratio.cpu()[:, None]
        inertias[cpu] = group.inertia_base[cpu] * ratio.cpu()[:, None, None]
        asset.root_physx_view.set_masses(masses, cpu)
        asset.root_physx_view.set_inertias(inertias, cpu)
    cpu = ids.cpu()
    materials = pair_materials(env._dr_table_static_friction[ids].cpu(),
                               env._dr_hand_static_friction[ids].cpu(),
                               env._dr_dynamic_friction_ratio[ids].cpu())
    for name, asset, buffer in (("robot", env.robot, env._robot_mats), ("table", env.table, env._table_mats)):
        static, dynamic, restitution = materials[name]
        buffer[cpu, :, 0], buffer[cpu, :, 1], buffer[cpu, :, 2] = static[:, None], dynamic[:, None], restitution
        asset.root_physx_view.set_material_properties(buffer, cpu)


def rigid_check(env, ids: torch.Tensor) -> None:
    """Read back geometry-scaled gains, masses, inertias and effective pair materials."""
    for group in env.groups:
        rows, local_ids = group.select(ids)
        if not rows.numel():
            continue
        view, selected, cpu = group.articulation.root_physx_view, ids[rows], local_ids.cpu()
        ratio = (env._dr_density[selected] / float(env.cfg.rigid_cloth.density)).cpu()
        factors = group.hinge_geometry[local_ids].cpu()
        armature = float(env.cfg.rigid_cloth.hinge_armature) + group.hinge_inertia[local_ids].cpu() * (
            ratio * env._dr_armature_scale[selected].cpu()
        )[:, None]
        for label, got, expected in (
            ("stiffness", view.get_dof_stiffnesses()[cpu], factors * env._dr_bending_rigidity[selected].cpu()[:, None]),
            ("damping", view.get_dof_dampings()[cpu], factors * env._dr_bending_viscosity[selected].cpu()[:, None]),
            ("mass", view.get_masses()[cpu], group.mass_base[cpu] * ratio[:, None]),
            ("inertia", view.get_inertias()[cpu], group.inertia_base[cpu] * ratio[:, None, None]),
            ("armature", view.get_dof_armatures()[cpu], armature),
        ):
            if not torch.allclose(got.float(), expected.float(), rtol=1e-4, atol=1e-12):
                raise RuntimeError(f"{group.design.name} {label} read-back mismatch")
        if not torch.allclose(view.get_material_properties()[cpu], group.plate_mats[cpu], atol=1e-6):
            raise RuntimeError(f"{group.design.name} cloth friction read-back mismatch")
    cpu = ids.cpu()
    for label, asset, expected in (("robot", env.robot, env._robot_mats), ("table", env.table, env._table_mats)):
        if not torch.allclose(asset.root_physx_view.get_material_properties()[cpu], expected[cpu], atol=1e-6):
            raise RuntimeError(f"{label} friction read-back mismatch")
