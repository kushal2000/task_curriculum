"""The rigid-chain (stage 1) and fold-pool (stage 2) fold tasks on PhysX.

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv_isaacsim``), on the Play task's own ``PlayEnv``. The task --
fold targets, reward, success, observations, the chain and the pool, their startup proofs -- is the
same code the Newton envs run (``cloth_task``, ``rigid_cloth_task``, ``fold_pool_task``); this module
only expresses the manipuland's physics in PhysX terms. Policies move between the two: train here
(PhysX steps this robot ~10x faster than MJWarp per GPU, see ``scripts/cluster/bos14_engine_bench.sh``),
evaluate on the Newton VBD cloth.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.train --headless \\
        --task Isaacsimenvs-RigidClothPhysx-Direct-v0 --agent rl_games_sapg_cfg_entry_point

**Contact matching** (what the Newton chain does with MJWarp-specific writes):

* *Friction* by USD-authored physics materials with combine modes -- slats ``min`` at
  ``rigid_cloth.friction``, fingertips ``max`` at ``rigid_cloth.fingertip_friction``, everything else
  PhysX's default 0.5 ``average``. That reproduces Newton's priority rule on all four pairs
  (slat-table/link 0.354, slat-fingertip 0.612, fingertip-table 0.612). It has to be USD: a
  tensor-API material write (``set_material_properties``) creates materials with the DEFAULT combine
  mode -- measured by ``scripts/analysis/physx_contact_probe.py``, a ``min`` slat rewritten to its own
  value went from 0.354 to the averaged 0.427 -- so the Play task's tensor material pass is switched
  off here, and friction DR is refused rather than silently averaged. The two materials are defined
  ONCE in the scene and bound per env (``_bind_contact_materials``): PhysX caps a scene at 64K
  materials, and a material inside the referenced asset is copied into every env.
* *Contact height* by a PhysX rest offset of ``rigid_cloth.contact_margin`` on every slat. Rest
  offsets add per contact exactly as Newton's margins do (probe: +7.000 mm resting height).
* *No slat-slat contact* by disabling the chain articulation's self-collision -- the same pairs
  Newton removes from its explicit pair list, and for the same reason (summed margins would hold the
  plies 14 mm apart).

**Conventions.** Two Newton quirks reach the observation, and every Newton policy was trained under
them, so they are reproduced here rather than fixed (``tests/test_physx_cloth_conventions.py``):

* the task's root-pose writes to ``env.object`` arrive reordered wxyz -> xyzw (Newton's
  ``patches.install_pose_write_conversion``), so the reset yaw decode (``cloth_adapter.task_yaw``)
  sees the same draw -- and the pinned start lands at the same 180 degrees;
* the fold goal's orientation is the true one's components relabelled (x, y, z, w) -> read as
  (w, x, y, z); see ``ClothTaskMixin._goal_quat_from_matrix``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.cloth_task import ClothTaskMixin
from isaacsimenvs.tasks.cloth.fold_pool_task import FoldPoolTaskMixin
from isaacsimenvs.tasks.cloth.physx_env_cfg import FoldPoolPhysxEnvCfg, RigidClothPhysxEnvCfg
from isaacsimenvs.tasks.cloth.rigid_cloth_task import RigidClothTaskMixin
from isaacsimenvs.tasks.cloth.utils import fold_shape_assets as fsa
from isaacsimenvs.tasks.cloth.utils import fold_shapes as fs
from isaacsimenvs.tasks.cloth.utils import rigid_cloth_assets as assets
from isaacsimenvs.tasks.cloth.utils.usd_bulk import bulk_edit_layer
from isaacsimenvs.tasks.cloth.utils.physx_adapters import (
    ChainAsClothMeshPhysx,
    ChainAsRigidObjectPhysx,
    PoolAsClothMeshPhysx,
    PoolAsRigidObjectPhysx,
)
from isaacsimenvs.tasks.play.play_env import PlayEnv

__all__ = [
    "FoldPoolPhysxEnv",
    "RigidClothPhysxEnv",
    "author_articulation_props",
    "bind_material_in_every_env",
    "prepare_collider_asset",
]

#: PhysX's scene-default material, which every collider without a bound material uses: the robot
#: links and the table. The Newton side sets both to ``assets.robot_friction`` / ``table_friction``.
_DEFAULT_MU = 0.5


def _apply_physx_api(prim, schema: str, attrs: dict) -> None:
    """Apply a PhysX API schema by NAME and author its attributes.

    By name rather than through ``pxr.PhysxSchema``: the kit-less stack's usd-core has no PhysX
    schema module (``newton/physx_schema_shim.py`` stubs only part of it), and the authored USD is
    the same either way -- an ``apiSchemas`` entry and namespaced attributes.
    """
    if schema not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(schema)
    for name, (vtype, value) in attrs.items():
        prim.CreateAttribute(name, vtype).Set(value)


def prepare_collider_asset(
    usd_path: str, *, rest_offset: float, contact_offset: float, max_depenetration_velocity: float = 0.0
) -> int:
    """De-instance a USD file's colliders and set their PhysX rest / contact offsets, in place.

    Returns the number of colliders. Offsets live in the asset, so every env that references it
    carries them; friction does NOT -- see ``_bind_contact_materials`` for why materials are bound
    per env to ONE scene-level material instead. ``max_depenetration_velocity > 0`` also caps how
    fast PhysX may push each rigid body of the asset out of a penetration [m/s].
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = next(iter(stage.GetPseudoRoot().GetChildren()))
    stage.SetEditTarget(stage.GetRootLayer())
    # Isaac Sim 5.1's URDF importer instances the collision geometry even with
    # `make_instanceable=False`, and an instance proxy cannot be edited (nor bound per env later).
    # De-instance first, as `scene_utils._bake_usd` does; repeat until no nested instance is left.
    while True:
        instances = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsInstance()]
        if not instances:
            break
        for p in instances:
            if not p.IsInstanceProxy():
                p.SetInstanceable(False)
    count = 0
    for prim in Usd.PrimRange(root):
        if max_depenetration_velocity > 0.0 and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            _apply_physx_api(prim, "PhysxRigidBodyAPI", {
                "physxRigidBody:maxDepenetrationVelocity": (
                    Sdf.ValueTypeNames.Float, float(max_depenetration_velocity)
                ),
            })
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        _apply_physx_api(prim, "PhysxCollisionAPI", {
            "physxCollision:contactOffset": (Sdf.ValueTypeNames.Float, float(contact_offset)),
            "physxCollision:restOffset": (Sdf.ValueTypeNames.Float, float(rest_offset)),
        })
        count += 1
    stage.GetRootLayer().Save()
    return count


def define_scene_material(
    stage, path: str, *, mu: float, combine: str, compliant_stiffness: float = 0.0,
    compliant_damping: float = 0.0, compliant_acceleration: bool = True,
):
    """A physics material with a friction combine mode, defined once in the scene.

    ``compliant_stiffness > 0`` makes every contact of a collider bound to it a PhysX compliant
    (implicit spring-damper) contact instead of a rigid one; ``compliant_acceleration`` reads both
    numbers per unit mass, so a light plate and the arm see the same spring.
    """
    from pxr import Sdf, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(mu))
    api.CreateDynamicFrictionAttr().Set(float(mu))
    api.CreateRestitutionAttr().Set(0.0)
    attrs = {"physxMaterial:frictionCombineMode": (Sdf.ValueTypeNames.Token, combine)}
    if compliant_stiffness > 0.0:
        attrs.update({
            "physxMaterial:compliantContactStiffness": (Sdf.ValueTypeNames.Float, float(compliant_stiffness)),
            "physxMaterial:compliantContactDamping": (Sdf.ValueTypeNames.Float, float(compliant_damping)),
            "physxMaterial:compliantContactAccelerationSpring": (
                Sdf.ValueTypeNames.Bool, bool(compliant_acceleration)
            ),
        })
    _apply_physx_api(material.GetPrim(), "PhysxMaterialAPI", attrs)
    return material


def collider_paths(stage, root_path: str, select=None) -> list[str]:
    """Collider prims under ``root_path`` (composed), relative to it, optionally filtered."""
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"[physx] no prim at {root_path}")
    out = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstanceProxy():
            raise RuntimeError(f"[physx] {prim.GetPath()} is an instance proxy and cannot be bound per env")
        if prim.HasAPI(UsdPhysics.CollisionAPI) and (select is None or select(prim)):
            out.append(str(prim.GetPath())[len(root_path) + 1:])
    return out


def bind_material_in_every_env(
    stage, num_envs: int, asset: str, rel_paths: list[str], material_path: str, env_ids=None
) -> int:
    """Bind ``material_path`` to ``env_<i>/<asset>/<rel>`` for every env (or every env in ``env_ids``:
    a pool shape that only some envs carry).

    Sdf-level, into ``usd_bulk.bulk_edit_layer``: at 24k envs this is ~10^5-10^6 specs, which the
    Usd API -- or Sdf edits made on Kit's root layer, O(N^2) -- take many minutes over.
    """
    from pxr import Sdf

    target = Sdf.Path(material_path)
    schemas = Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"])
    envs = list(range(num_envs)) if env_ids is None else [int(i) for i in env_ids]
    with bulk_edit_layer(stage, f"bind_{asset}") as layer:
        for i in envs:
            for rel in rel_paths:
                spec = Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{i}/{asset}/{rel}")
                spec.SetInfo("apiSchemas", schemas)
                rs = spec.relationships.get("material:binding:physics")
                if rs is None:
                    rs = Sdf.RelationshipSpec(spec, "material:binding:physics", custom=False)
                rs.targetPathList.explicitItems = [target]
    return len(envs) * len(rel_paths)


def author_articulation_props(
    usd_path: str, *, self_collisions: bool, position_iterations: int, velocity_iterations: int
) -> int:
    """Set PhysX articulation-root properties in a USD file, in place. Returns roots touched.

    For assets spawned without a spawner (the pool's per-env prims), where ``UsdFileCfg``'s
    ``articulation_props`` cannot apply them.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    stage.SetEditTarget(stage.GetRootLayer())
    roots = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        _apply_physx_api(prim, "PhysxArticulationAPI", {
            "physxArticulation:enabledSelfCollisions": (Sdf.ValueTypeNames.Bool, bool(self_collisions)),
            "physxArticulation:solverPositionIterationCount": (Sdf.ValueTypeNames.Int, int(position_iterations)),
            "physxArticulation:solverVelocityIterationCount": (Sdf.ValueTypeNames.Int, int(velocity_iterations)),
        })
        roots += 1
    stage.GetRootLayer().Save()
    return roots


def _reorder_object_pose_writes(obj) -> None:
    """Mirror Newton's ``patches.install_pose_write_conversion`` on ``env.object``: wxyz -> xyzw.

    The adapters decode the reset yaw as wxyz from what they RECEIVE, and on Newton they receive the
    task's quaternion reordered. Reproducing the reorder keeps the decoded heading -- for a random
    draw and for the pinned start -- identical across the two simulators.
    """
    for method in ("write_root_pose_to_sim", "write_root_state_to_sim", "write_root_link_pose_to_sim"):
        original = getattr(obj, method, None)
        if original is None:
            continue

        def converted(value, *args, _orig=original, **kwargs):
            value = value.clone()
            value[..., 3:7] = torch.cat((value[..., 4:7], value[..., 3:4]), dim=-1)
            return _orig(value, *args, **kwargs)

        setattr(obj, method, converted)


def _size_gpu_buffers(cfg) -> None:
    """Grow PhysX's fixed GPU contact buffers with the env count; never shrink them.

    The defaults suit the Play task's one tool per env. The pool carries 26 plates per env, and at
    12288 envs its startup check alone overflowed the patch buffer ("increase its size to at least
    190888"), which then took the CUDA context down with an illegal memory access.
    """
    px, n = cfg.sim.physx, int(cfg.scene.num_envs)
    per = cfg.physx_chain
    px.gpu_max_rigid_patch_count = max(int(px.gpu_max_rigid_patch_count), per.gpu_patches_per_env * n)
    px.gpu_max_rigid_contact_count = max(int(px.gpu_max_rigid_contact_count), per.gpu_contacts_per_env * n)
    px.gpu_found_lost_pairs_capacity = max(int(px.gpu_found_lost_pairs_capacity), per.gpu_pairs_per_env * n)
    px.gpu_total_aggregate_pairs_capacity = max(
        int(px.gpu_total_aggregate_pairs_capacity), per.gpu_pairs_per_env * n
    )


class _PhysxClothMixin:
    """PhysX construction and contact physics shared by the chain and the pool. First in the bases."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        a = cfg.assets
        # The Play task's material pass writes every robot/table/object shape through the tensor API,
        # which resets combine modes (see the module docstring). Contacts are authored in USD instead,
        # so it must not run; the colliders it would have set to 0.5 keep PhysX's default 0.5.
        for name in ("robot_friction", "table_friction"):
            if abs(float(getattr(a, name)) - _DEFAULT_MU) > 1e-9:
                raise ValueError(
                    f"assets.{name}={getattr(a, name)}: the PhysX fold tasks leave robot links and the "
                    f"table on PhysX's default material ({_DEFAULT_MU}); a different value needs a "
                    f"USD material, not the tensor write"
                )
        a.modify_asset_frictions = False
        # A PhysX numerical requirement, not a task parameter: see PhysxChainCfg.hinge_armature. Written
        # into the task block so the actuator (and the logs) carry the value the hinges really get.
        if float(cfg.physx_chain.hinge_armature) > 0.0:
            cfg.rigid_cloth.joint_armature = float(cfg.physx_chain.hinge_armature)
        print(f"[physx] hinge armature {cfg.rigid_cloth.joint_armature:g} kg m^2", flush=True)
        _size_gpu_buffers(cfg)
        rd = cfg.rigid_cloth.randomization
        if rd.enabled and (
            tuple(rd.slat_friction_scale) != (1.0, 1.0) or tuple(rd.fingertip_friction_scale) != (1.0, 1.0)
        ):
            raise NotImplementedError(
                "friction DR on PhysX: a tensor-API material write resets the combine modes that "
                "match the cloth's friction, so it would silently average instead. Set "
                "rigid_cloth.randomization.{slat,fingertip}_friction_scale to [1.0, 1.0]."
            )
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------ construction

    def _setup_scene(self) -> None:
        super()._setup_scene()
        # Newton runs this inside setup_scene too (`pre_clone_scene_hook`); here there is no cloning
        # step to beat -- every env prim already exists -- only the simulator start, which follows.
        self._install_cloth()
        self._bind_contact_materials()
        _reorder_object_pose_writes(self.object)

    def _bind_contact_materials(self) -> None:
        """Two scene-level materials -- slat and fingertip -- bound to those colliders in every env.

        ONE material each, not one per env: PhysX caps a scene at 64K materials, and a material
        authored inside the asset becomes a separate material in every env that references it. At
        12288 envs the pool's five shapes plus the fingertips crossed the cap and the slats silently
        kept the default friction (caught by `_setup_chain_physics`'s read-back).
        """
        from isaaclab.sim.utils import get_current_stage

        from isaacsimenvs.tasks.play.utils.scene_utils import FINGERTIP_LINK_NAMES

        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        stage = get_current_stage()
        slat = "/World/PhysicsMaterials/slat"
        tip = "/World/PhysicsMaterials/fingertip"
        define_scene_material(
            stage, slat, mu=r.friction, combine=pc.slat_friction_combine,
            compliant_stiffness=pc.slat_compliant_stiffness, compliant_damping=pc.slat_compliant_damping,
            compliant_acceleration=pc.slat_compliant_acceleration,
        )
        define_scene_material(stage, tip, mu=r.fingertip_friction, combine=pc.fingertip_friction_combine)
        n = self.num_envs
        slats = 0
        for k, art in enumerate(self._manipuland_arts()):
            asset = art.cfg.prim_path.rsplit("/", 1)[-1]
            envs = self._manipuland_envs(k)
            rels = collider_paths(stage, f"/World/envs/env_{envs[0]}/{asset}")
            if not rels:
                raise RuntimeError(f"[physx] no collider under env_{envs[0]}/{asset}")
            slats += bind_material_in_every_env(stage, n, asset, rels, slat, env_ids=envs)
        tips = tuple(FINGERTIP_LINK_NAMES)
        robot = self.robot.cfg.prim_path.rsplit("/", 1)[-1]
        rels = collider_paths(
            stage, f"/World/envs/env_0/{robot}",
            select=lambda prim: any(f"/{t}/" in str(prim.GetPath()) + "/" for t in tips),
        )
        if not rels:
            raise RuntimeError(f"[physx] no fingertip collider ({tips}) under env_0/{robot}")
        bind_material_in_every_env(stage, n, robot, rels, tip)
        self._fingertip_colliders = len(rels)
        print(
            f"[physx] bound {slats} slat colliders to {slat} and {n * len(rels)} fingertip colliders "
            f"to {tip} ({len(rels)} per env); slat contacts "
            + (
                f"COMPLIANT, stiffness {pc.slat_compliant_stiffness:g} damping {pc.slat_compliant_damping:g} "
                f"({'per unit mass' if pc.slat_compliant_acceleration else 'force'})"
                if pc.slat_compliant_stiffness > 0.0 else "rigid"
            ),
            flush=True,
        )

    def _goal_quat_from_matrix(self, rot: torch.Tensor) -> torch.Tensor:
        # Isaac Lab 2.3's helper is wxyz; hand the task the (x, y, z, w) relabelling Newton does.
        return quat_from_matrix(rot)[..., [1, 2, 3, 0]]

    # ------------------------------------------------------------------ the manipuland's physics

    def _manipuland_arts(self) -> list:
        raise NotImplementedError

    def _manipuland_envs(self, k: int) -> list[int]:
        """The envs manipuland articulation ``k`` exists in (before the simulator starts)."""
        return list(range(self.num_envs))

    def _manipuland_rows(self, k: int, ids: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``(keep, local)``: which of env ``ids`` articulation ``k`` exists in (``None`` = all) and
        its instance indices there."""
        return None, ids

    def _contact_margin_value(self) -> float:
        return float(self.cfg.rigid_cloth.resolved_contact_margin(self.cfg.cloth.thickness))

    def _contact_offset_value(self) -> float:
        """The slats' PhysX contact offset: above the LARGEST rest offset margin DR can draw.

        Fixed for the run. PhysX refuses a rest offset at or above the contact offset, so a contact
        offset that tracked each draw would have to be rewritten before or after the rest offset
        depending on the direction of the change. It only sets how early a contact is generated,
        not where the surface rests.
        """
        rd = self.cfg.rigid_cloth.randomization
        top = float(rd.contact_margin_offset[1]) if rd.enabled else 0.0
        return self._contact_margin_value() + max(top, 0.0) + float(self.cfg.physx_chain.contact_offset_margin)

    def _setup_chain_physics(self) -> None:
        """Contacts are already authored; read them back and refuse to run on a mismatch."""
        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        if r.contact_ke > 0.0 or r.contact_kd > 0.0:
            raise ValueError("[physx_chain] rigid_cloth.contact_ke / contact_kd are Newton (MJWarp) only")
        m = self._contact_margin_value()
        self._contact_margin = m
        for art in self._manipuland_arts():
            view = art.root_physx_view
            mats = view.get_material_properties()
            rest = view.get_rest_offsets()
            bad_mu = float((mats[..., 0] - r.friction).abs().max())
            bad_rest = float((rest - m).abs().max())
            if bad_mu > 1e-4 or bad_rest > 1e-6:
                raise RuntimeError(
                    f"[physx] {art.cfg.prim_path}: slat friction off by {bad_mu:.4f} or rest offset off "
                    f"by {bad_rest * 1e3:.3f} mm from what was authored"
                )
        tips = self.robot.root_physx_view.get_material_properties()[0, :, 0]
        n_tip = int(((tips - r.fingertip_friction).abs() < 1e-4).sum())
        n_def = int(((tips - _DEFAULT_MU).abs() < 1e-4).sum())
        if n_tip < self._fingertip_colliders or n_tip + n_def != tips.numel():
            raise RuntimeError(
                f"[physx] robot frictions: {n_tip} shapes at {r.fingertip_friction} (bound "
                f"{self._fingertip_colliders} fingertip colliders), {n_def} at {_DEFAULT_MU}, of "
                f"{tips.numel()}: {sorted(set(round(float(v), 3) for v in tips))}"
            )
        print(
            f"[rigid_cloth] friction: slat {r.friction:.3f} ({pc.slat_friction_combine}), fingertip "
            f"{r.fingertip_friction:.3f} ({pc.fingertip_friction_combine}, {n_tip} colliders), robot "
            f"links and table {_DEFAULT_MU} (average) -> slat-table/link {min(r.friction, _DEFAULT_MU):.3f}, "
            f"slat-fingertip {r.fingertip_friction:.3f}, fingertip-table "
            f"{max(r.fingertip_friction, _DEFAULT_MU):.3f}\n"
            f"[rigid_cloth] contact margin {m * 1e3:.2f} mm as PhysX rest offset on every slat "
            f"(contact offset {self._contact_offset_value() * 1e3:.1f} mm); slat-slat contact "
            f"{'off (articulation self-collision disabled)' if m > 0.0 else 'on'}",
            flush=True,
        )

    def _init_physics_randomization(self) -> None:
        n = self.num_envs
        arts = self._manipuland_arts()
        self._mass_base = [a.root_physx_view.get_masses().clone() for a in arts]
        self._inertia_base = [a.root_physx_view.get_inertias().clone() for a in arts]
        self.dr_sample = {
            k: torch.ones(n, device=self.device)
            for k in ("slat_friction", "fingertip_friction", "hinge_damping", "mass")
        }
        self.dr_sample["hinge_stiffness"] = torch.zeros(n, device=self.device)
        self.dr_sample["contact_margin"] = torch.full((n,), self._contact_margin, device=self.device)
        self._dr_ready = bool(self.cfg.rigid_cloth.randomization.enabled)
        if self._dr_ready:
            self._randomize_physics(torch.arange(n, device=self.device))
            rd = self.cfg.rigid_cloth.randomization
            print(
                f"[rigid_cloth] physics randomisation ON, per env at every reset: hinge damping x"
                f"{tuple(rd.hinge_damping_scale)} (log), hinge stiffness {tuple(rd.hinge_stiffness)} "
                f"N m/rad, contact margin {self._contact_margin * 1e3:.1f}"
                f"{rd.contact_margin_offset[0] * 1e3:+.1f}..{rd.contact_margin_offset[1] * 1e3:+.1f} mm, "
                f"mass x{tuple(rd.mass_scale)} (friction fixed: see physx_envs)",
                flush=True,
            )

    def _randomize_physics(self, env_ids) -> None:
        """Re-draw the manipuland's physics for ``env_ids``: only the ranges that are not a point."""
        rd = self.cfg.rigid_cloth.randomization
        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        n = int(ids.numel())
        if n == 0:
            return
        ids_cpu = ids.to("cpu", torch.int32)

        def uni(lo_hi) -> torch.Tensor:
            return torch.empty(n, device=self.device).uniform_(float(lo_hi[0]), float(lo_hi[1]))

        def varies(lo_hi) -> bool:
            return float(lo_hi[0]) != float(lo_hi[1])

        arts = self._manipuland_arts()
        # Per articulation: which of `ids` it exists in, and its own instance rows for them.
        rows = []
        for k in range(len(arts)):
            keep, local = self._manipuland_rows(k, ids)
            rows.append((keep, local, local.to("cpu", torch.int32)))

        def sel(v: torch.Tensor, keep) -> torch.Tensor:
            return v if keep is None else v[keep.to(v.device)]

        if varies(rd.contact_margin_offset) and self._contact_margin > 0.0:
            margin = (self._contact_margin + uni(rd.contact_margin_offset)).clamp(min=0.0)
            m_cpu = margin.cpu().unsqueeze(1)
            for art, (keep, local, l_cpu) in zip(arts, rows):
                if local.numel() == 0:
                    continue
                view = art.root_physx_view
                rest = view.get_rest_offsets()
                rest[l_cpu.long()] = sel(m_cpu, keep).expand(-1, rest.shape[1])
                # The contact offset is fixed above the whole range (`_contact_offset_value`).
                view.set_rest_offsets(rest, l_cpu)
            self.dr_sample["contact_margin"][ids] = margin
        if varies(rd.hinge_damping_scale):
            lo, hi = math.log(float(rd.hinge_damping_scale[0])), math.log(float(rd.hinge_damping_scale[1]))
            damp = float(r.joint_damping) * torch.exp(torch.empty(n, device=self.device).uniform_(lo, hi))
            for art, (keep, local, _l) in zip(arts, rows):
                if local.numel():
                    d = sel(damp, keep)
                    art.write_joint_damping_to_sim(d.unsqueeze(1).expand(-1, art.num_joints), env_ids=local)
            self.dr_sample["hinge_damping"][ids] = damp
        if varies(rd.hinge_stiffness):
            stiff = uni(rd.hinge_stiffness)
            for art, (keep, local, _l) in zip(arts, rows):
                if local.numel():
                    k_ = sel(stiff, keep)
                    art.write_joint_stiffness_to_sim(k_.unsqueeze(1).expand(-1, art.num_joints), env_ids=local)
            self.dr_sample["hinge_stiffness"][ids] = stiff
        if varies(rd.mass_scale):
            scale = uni(rd.mass_scale)
            s_cpu = scale.cpu()
            for art, m0, i0, (keep, local, l_cpu) in zip(arts, self._mass_base, self._inertia_base, rows):
                if local.numel() == 0:
                    continue
                view = art.root_physx_view
                masses = view.get_masses()
                inertias = view.get_inertias()
                sc = sel(s_cpu, keep)
                masses[l_cpu.long()] = m0[l_cpu.long()] * sc.unsqueeze(1)
                inertias[l_cpu.long()] = i0[l_cpu.long()] * sc.view(-1, 1, 1)
                view.set_masses(masses, l_cpu)
                view.set_inertias(inertias, l_cpu)
            self.dr_sample["mass"][ids] = scale


class RigidClothPhysxEnv(_PhysxClothMixin, RigidClothTaskMixin, ClothTaskMixin, PlayEnv):
    """Stage 1: one hinged chain (``rigid_cloth.variant``) on PhysX."""

    cfg: RigidClothPhysxEnvCfg
    _chain_mesh_cls = ChainAsClothMeshPhysx
    _chain_object_cls = ChainAsRigidObjectPhysx

    def _manipuland_arts(self) -> list:
        return [self._chain]

    def _chain_usd(self, urdf_path, work_dir) -> str:
        """URDF -> USD with Isaac Sim's converter, then the slat material and offsets authored on it."""
        from isaacsimenvs.tasks.play.utils import scene_utils

        m = self._contact_margin_value()
        kw = dict(assets.CONVERT_KWARGS)
        # Slat-slat contact goes when there is a margin (see the module docstring); the joint drive
        # is what lets the actuator's damping and armature land on a PhysX joint.
        kw["self_collision"] = m <= 0.0
        kw["joint_drive"] = scene_utils._robot_joint_drive_cfg()
        usd = scene_utils._convert_urdf_to_usd(str(urdf_path), Path(work_dir), **kw)
        self._author_slats(usd)
        return usd

    def _author_slats(self, usd: str) -> None:
        n = prepare_collider_asset(
            usd, rest_offset=self._contact_margin_value(), contact_offset=self._contact_offset_value(),
            max_depenetration_velocity=self.cfg.physx_chain.slat_max_depenetration_velocity,
        )
        if n == 0:
            raise RuntimeError(f"[physx] no collider in {usd} for the slat offsets")

    def _chain_spawn_cfg(self, usd_path: str):
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.from_files import UsdFileCfg

        pc = self.cfg.physx_chain
        return UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self._contact_margin_value() <= 0.0,
                solver_position_iteration_count=pc.solver_position_iterations,
                solver_velocity_iteration_count=pc.solver_velocity_iterations,
            ),
        )


class FoldPoolPhysxEnv(FoldPoolTaskMixin, RigidClothPhysxEnv):
    """Stage 2: the fold-shape pool on PhysX, each env's sheets at its own size."""

    cfg: FoldPoolPhysxEnvCfg
    _pool_mesh_cls = PoolAsClothMeshPhysx
    _pool_object_cls = PoolAsRigidObjectPhysx
    # PhysX parses every env on its own (`replicate_physics: false`), so an env can carry just one
    # shape: each shape's articulation then spans the envs that carry it.
    _supports_one_shape_per_env = True

    def _manipuland_arts(self) -> list:
        return list(self._pool_arts)

    def _pool_envs(self, s: int) -> list[int]:
        if self._env_shape_np is None:
            return list(range(self.num_envs))
        return np.nonzero(self._env_shape_np == s)[0].tolist()

    def _manipuland_envs(self, k: int) -> list[int]:
        return self._pool_envs(k)

    def _manipuland_rows(self, k: int, ids: torch.Tensor):
        return self._cloth.local_rows(k, ids)

    def _pool_articulation_cls(self):
        from isaaclab.assets import Articulation

        if self._env_shape_np is None:
            return Articulation

        class _SubsetArticulation(Articulation):
            """A pool shape's articulation when only some envs carry it.

            The scene resets assets with GLOBAL env ids, which do not index this articulation's
            instances. Its per-instance reset state is the actuators' (implicit, stateless) and the
            external wrench buffers (never used on the pool), so resetting every instance is exact.
            """

            def reset(self, env_ids=None):
                super().reset(None)

        return _SubsetArticulation

    def _draw_env_sizes(self, cfg, n: int):
        """The Newton draws, quantised: every env references the USD of its own (size, slab)."""
        size, slab = super()._draw_env_sizes(cfg, n)
        q = float(cfg.physx_chain.size_quantum)
        size = np.round(size / q) * q
        # Never the range top itself: a 0.200 m box2-mid half-plate (200 x 100 x 2 mm) fails PhysX's
        # GPU convex cooking ("failed to cook GPU-compatible mesh"; 0.199 m and 0.150 m cook) and
        # falls back to CPU collision inside the GPU pipeline. Rounding put ~1/200 of envs there.
        top = float(cfg.fold_pool.size_range[1])
        size = np.minimum(size, np.round((top - q) / q) * q) if top - q >= float(cfg.fold_pool.size_range[0]) else size
        return size, np.round(slab / 1.0e-4) * 1.0e-4

    def _pool_spawn(self, s: int, shape, usd_dir: Path, park_local):
        """Write one USD per distinct (size, slab) of shape ``s`` and point each env's prim at its own.

        Exact physics at every size: ``fold_shapes.make_pool(size, thickness)`` is the shape itself at
        that size, so mass, inertia, hinge and seam anchors all come out of the same construction as
        the nominal sheet. Returns ``None``: the prims exist, nothing is left for a spawner to do.
        """
        from pxr import Gf, Sdf
        from isaaclab.sim.utils import get_current_stage

        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        m = self._contact_margin_value()
        envs = self._pool_envs(s)
        keys = sorted(
            set(zip(np.round(self._env_size_np[envs], 6), np.round(self._env_slab_np[envs], 6)))
        )
        usd_of = {}
        for size, slab in keys:
            sized = fs.make_pool(size=float(size), thickness=float(slab), hinge=shape.hinge)[shape.name]
            path = usd_dir / f"{shape.name}_{size * 1e3:.2f}mm_{slab * 1e3:.2f}mm.usda"
            fsa.write_shape_usd(sized, path, density=r.density)
            prepare_collider_asset(
                str(path), rest_offset=m, contact_offset=self._contact_offset_value(),
                max_depenetration_velocity=pc.slat_max_depenetration_velocity,
            )
            # What `_chain_spawn_cfg` sets on the chain; no spawner runs here to apply it.
            if author_articulation_props(
                str(path),
                self_collisions=m <= 0.0,
                position_iterations=pc.solver_position_iterations,
                velocity_iterations=pc.solver_velocity_iterations,
            ) != 1:
                raise RuntimeError(f"[fold_pool] {path} does not have exactly one articulation root")
            usd_of[(size, slab)] = str(path)

        stage = get_current_stage()
        pos = Gf.Vec3d(*[float(v) for v in park_local])
        with bulk_edit_layer(stage, f"pool{s}") as layer:
            for i in envs:
                key = (round(float(self._env_size_np[i]), 6), round(float(self._env_slab_np[i]), 6))
                spec = Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{i}/FoldPool{s}")
                spec.specifier = Sdf.SpecifierDef
                spec.typeName = "Xform"
                spec.referenceList.Prepend(Sdf.Reference(usd_of[key]))
                t = Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                t.default = pos
                o = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                o.default = ["xformOp:translate"]
        print(
            f"[fold_pool] {shape.name}: in {len(envs)} envs, {len(keys)} sheet sizes, one USD each",
            flush=True,
        )
        return None

    def _pool_after_spawn(self) -> None:
        # Sizes were authored per env in `_pool_spawn`; nothing to resize.
        pass

    def _pool_plate_mass(self) -> torch.Tensor:
        total = torch.zeros(self.num_envs, device=self.device)
        for s, a in enumerate(self._pool_arts):
            total[self._cloth.members(s)] += a.root_physx_view.get_masses().sum(dim=1).to(self.device)
        return total
