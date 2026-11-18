"""PhysX-side USD and material helpers for the plates: collider offsets, scene-level friction
materials bound per env, articulation properties, GPU buffer sizing.

**Contact matching.** Friction is set with USD physics materials and combine modes -- plates ``min``
at ``rigid_cloth.friction``, fingertips ``max`` at ``rigid_cloth.fingertip_friction``, everything else
PhysX's default 0.5 ``average`` (``physx_chain.friction_model: priority``), or ``cloth_min`` with the
robot and table average-mode. It has to be USD: a tensor-API material write
(``set_material_properties``) creates materials with the DEFAULT combine mode (a ``min`` plate
rewritten to its own value went from 0.354 to the averaged 0.427), so the Play task's tensor material
pass is switched off. Materials are defined ONCE in the scene and bound per env: PhysX caps a scene
at 64K materials, and a material inside a referenced asset is copied into every env. Contact height
is a PhysX rest offset of ``rigid_cloth.contact_margin`` on every plate; plate-plate contact is off
(articulation self-collision disabled) whenever that margin is nonzero.

**Conventions.** Two Newton quirks reach the observation and every policy was trained under them, so
they are reproduced (``tests/test_physx_cloth_conventions.py``): the task's root-pose writes to
``env.object`` arrive reordered wxyz -> xyzw (``reorder_object_pose_writes``), and the fold goal's
orientation is the true one's components relabelled (x, y, z, w) -> read as (w, x, y, z)
(``RigidClothEnv._goal_quat_from_matrix``).
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.backends.usd import bulk_edit_layer

__all__ = [
    "DEFAULT_MU",
    "author_articulation_props",
    "bind_material_in_every_env",
    "bind_materials",
    "collider_paths",
    "define_scene_material",
    "prepare_collider_asset",
    "reorder_object_pose_writes",
    "size_gpu_buffers",
]

#: PhysX's scene-default material, which every collider without a bound material uses: the robot
#: links and the table (``friction_model: priority``).
DEFAULT_MU = 0.5


def _apply_physx_api(prim, schema: str, attrs: dict) -> None:
    """Apply a PhysX API schema by NAME and author its attributes.

    By name rather than through ``pxr.PhysxSchema``: the kit-less stack's usd-core has no PhysX
    schema module (``backends/vbd/newton_support/physx_schema_shim.py`` stubs only part of it), and the authored USD is
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
    carries them; friction does NOT -- see ``ContactsMixin._bind_contact_materials`` for why materials are bound
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
    envs = list(range(num_envs)) if env_ids is None else [int(i) for i in env_ids]
    return bind_materials(
        stage,
        ((f"/World/envs/env_{i}/{asset}/{rel}", material_path) for i in envs for rel in rel_paths),
        tag=f"bind_{asset}",
    )


def bind_materials(stage, bindings, tag: str = "bind") -> int:
    """Bind each ``(prim_path, material_path)`` in ``bindings`` (physics purpose), all in ONE
    ``usd_bulk.bulk_edit_layer``. Returns the number of bindings.

    One layer per call matters: each layer is inserted into the stage's sublayers, which recomposes
    the stage, so 112 calls (7 pool shapes x 16 friction buckets) took 8.5 min at 1536 envs where
    one call per shape took 25 s.
    """
    from pxr import Sdf

    schemas = Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"])
    targets: dict[str, Sdf.Path] = {}
    count = 0
    with bulk_edit_layer(stage, tag) as layer:
        for prim_path, material_path in bindings:
            target = targets.setdefault(material_path, Sdf.Path(material_path))
            spec = Sdf.CreatePrimInLayer(layer, prim_path)
            spec.SetInfo("apiSchemas", schemas)
            rs = spec.relationships.get("material:binding:physics")
            if rs is None:
                rs = Sdf.RelationshipSpec(spec, "material:binding:physics", custom=False)
            rs.targetPathList.explicitItems = [target]
            count += 1
    return count


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


def reorder_object_pose_writes(obj) -> None:
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


def size_gpu_buffers(cfg) -> None:
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
