"""The Isaac Lab 3.0 / Newton fixes the deformable cloth env needs, in one place.

Each section keeps the finding that motivated it, because several of these look like arbitrary
compatibility shims until you know what they cost to discover. ``tasks/cloth/backends/vbd/solver.py``
installs the solver fixes; ``backends/vbd/scene.py`` calls the scene-construction helpers.

* actuator defaults broadcast to (N, J) -- API shape
* replication planning, collision geometry onto mesh prims, USD drive gains -- scene construction
* armature / friction writeback, placeholder inertia, solver refresh, implicit-actuator effort,
  gravcomp and condim on the builder -- physics fidelity
"""

from __future__ import annotations

import importlib

# ============================================================================
# Actuator defaults: broadcasting USD-authored rows
# ============================================================================
#
# Broadcast USD-authored actuator defaults across environments on the Newton backend.
#
# ``ActuatorBase._parse_joint_parameter`` falls back to the USD-authored value whenever the
# config leaves a parameter unset. Upstream's arm actuator specifies only ``stiffness`` and
# ``damping``, so ``armature``, ``friction``, ``effort_limit`` and ``velocity_limit`` all take
# that path.
#
# Under the Newton backend those defaults arrive shaped ``(1, num_joints)`` — one row for the
# articulation — while the validator demands ``(num_envs, num_joints)``::
#
#     ValueError: Invalid default value tensor shape.
#     Got: torch.Size([1, 7])
#     Expected: (4, 7)
#
# The authored values are identical across clones (every environment is spawned from the same
# baked robot USD), so expanding the single row to all environments is exactly the intended
# meaning — the surrounding code comments the tensor branch as "use the same tensor for all
# joints". This patch only widens the accepted shape; it never invents or rescales values, and
# a default whose row count is neither 1 nor ``num_envs`` still raises.
#
# Scope note: this is a Newton-backend shape convention, not a SimToolReal issue, so it is
# patched at the Isaac Lab layer rather than in upstream's task code.

def install_actuator_broadcast() -> None:
    """Allow ``(1, num_joints)`` USD defaults to broadcast to ``(num_envs, num_joints)``."""
    import torch
    from isaaclab.actuators.actuator_base import ActuatorBase

    if getattr(ActuatorBase, "_simtoolreal_broadcast_patch", False):
        return

    original = ActuatorBase._parse_joint_parameter

    def _parse_joint_parameter(self, cfg_value, default_value):
        if (
            cfg_value is None
            and isinstance(default_value, torch.Tensor)
            and default_value.ndim == 2
            and default_value.shape[0] == 1
            and self._num_envs != 1
            and default_value.shape[1] == self.num_joints
        ):
            default_value = default_value.expand(self._num_envs, -1).contiguous()
        return original(self, cfg_value, default_value)

    ActuatorBase._parse_joint_parameter = _parse_joint_parameter
    ActuatorBase._simtoolreal_broadcast_patch = True


# ============================================================================
# Environment replication
# ============================================================================
#
# Perform Isaac Lab 3.0's explicit environment replication, which upstream never calls.
#
# Root cause of ``world_count == 1``
# ----------------------------------
# Newton partitions a scene into one **world** per environment and indexes reset masks by
# environment id. With every body in world 0, ``_scatter_reset_masks_from_ids`` walks off the
# end of a one-element array and the process dies with a CUDA illegal memory access.
#
# Nothing was replicating the environments. Under Isaac Lab 2.x, cloning happened through
# ``InteractiveScene.clone_environments()`` — upstream still refers to it in a comment
# ("setup_student_camera now runs AFTER clone_environments()"), but the method is gone in 3.0.
# Replication is now explicit, as in the shipped Cartpole task::
#
#     plan = cloner.clone_plan_from_env_0(src, dest, num_envs, device, positions)
#     cloner.replicate(plan, stage=scene.stage)
#
# PhysX tolerated the omission because it does not need world partitioning; Newton does not.
#
# Why it is not ``clone_plan_from_env_0``
# ---------------------------------------
# The shipped tasks all build their plan with ``clone_plan_from_env_0``, and so did this port.
# That plan has exactly **one row**, whose source is the whole ``/World/envs/env_0`` subtree, and
# every cfg maps to it. So it means "every environment is a copy of environment 0" -- including
# the object, whichever prims the stage happens to hold.
#
# There was a mitigation here that did not work, and understanding why matters more than the
# three lines it occupied. It walked ``REPLICATION_QUEUE`` and popped the ``/Object`` and
# ``/GoalViz`` cfgs out of ``plan.cfg_rows``, on the documented grounds that ``replicate`` skips
# any cfg absent from that dict. It does. But ``cfg_rows`` is only a *cfg to row-index* map, and
# in a ``clone_plan_from_env_0`` plan there is one row for everything: ``replicate`` takes the
# **set union** of the rows the remaining cfgs own (``replicate_session.py:96``), the robot and
# the table still contribute row 0, and row 0 copies the entire environment, object included.
# Dropping a cfg from a whole-env plan cannot exclude a subtree, because the plan has no
# per-asset granularity to exclude it with. Measured after the drop: all four environments still
# reported env_0's mass and centre of mass.
#
# ``make_clone_plan`` is the constructor that does have that granularity -- one row per asset per
# variant -- so it is the one used here.
#
# Ordering
# --------
# ``make_clone_plan`` **mutates** each cfg's ``spawn_path``/``spawn_paths`` so the next asset
# constructor spawns the prototype into its first active environment, which means upstream's
# constructors would have to run after it. They do not: ``setup_scene`` builds each cfg and
# constructs its asset in the same statement. Rather than restructure upstream's function, the
# multi-asset patch writes exactly the spawn paths ``make_clone_plan`` would have written
# (:func:`_variant_spawn_paths`), the plan is built afterwards from ``REPLICATION_QUEUE``, and
# :func:`_assert_spawn_paths_unchanged` fails the run if the two ever disagree. The check is the
# point: it turns an ordering assumption into an assertion.
#
# Collision filtering is applied only on PhysX, matching Cartpole: ``scene.filter_collisions()``
# is required there for replicated environments, while Newton handles inter-world isolation
# through its world partitioning.


def _plan_groups(cfgs):
    """The cfgs ``make_clone_plan`` will treat as prototype groups, in its own order.

    Mirrors the filter at ``clone_plan.py:274-284`` exactly: a cfg needs a ``prim_path`` under the
    environment root and a spawner. Anything else is skipped by the planner and must therefore be
    skipped when sizing the ``valid_set`` columns too.
    """
    groups = []
    for cfg in cfgs:
        if not hasattr(cfg, "prim_path") or not hasattr(cfg, "spawn") or cfg.spawn is None:
            continue
        if "/World/envs/" not in cfg.prim_path:
            continue
        groups.append(cfg)
    return groups


def _round_robin_valid_set(variant_counts: list[int], num_envs: int, device: str):
    """Diagonal clone combinations, so environment *i* gets variant ``i % n`` of **every** group.

    ``make_clone_plan``'s default ``valid_set`` is the full cartesian product of the groups'
    variants, which ``cloner_strategies.sequential`` then walks in ``itertools.product`` order.
    With four objects and four goal markers that is 16 combinations, and the first four -- the
    ones four environments receive -- all carry object variant 0. The upstream 2.x semantics this
    port reproduces are per-asset round-robin instead, so only the diagonal combinations are
    legal. ``len(rows)`` is the lcm of the variant counts, so ``i % len(rows) % n == i % n`` holds
    for every group and ``sequential`` reproduces the assignment exactly.

    Args:
        variant_counts: Spawn-variant count per group, in ``make_clone_plan`` group order.
        num_envs: Environment count; a variant beyond it is never assigned.
        device: Torch device for the returned tensor.

    Returns:
        A ``[lcm, num_groups]`` long tensor, or ``None`` when every group is single-variant (which
        lets ``make_clone_plan`` take its cheaper whole-environment homogeneous branch).
    """
    import math

    import torch

    assigned = [min(count, num_envs) for count in variant_counts]
    if not assigned or all(count == 1 for count in assigned):
        return None
    period = math.lcm(*assigned)
    rows = [[k % count for count in assigned] for k in range(period)]
    return torch.tensor(rows, dtype=torch.long, device=device)


def _assert_spawn_paths_unchanged(groups, before) -> None:
    """Fail if ``make_clone_plan`` disagrees with where the prototypes were actually spawned.

    The prototypes are spawned before the plan exists, so the plan's own idea of each prototype's
    home has to be checked against reality rather than trusted. A mismatch means environment *i*
    would be cloned from a prim holding a different variant than the one this port intends --
    silently, and only visible as a physics difference much later.
    """
    for cfg, (attr, expected) in zip(groups, before):
        actual = getattr(cfg.spawn, attr)
        if isinstance(expected, list):
            actual = list(actual) if actual is not None else None
        if expected is None:
            # Nothing was pinned for this asset, so nothing can have moved: the planner is
            # assigning a prototype home for the first time, which is it doing its job. This is
            # the case for a single homogeneous asset added by a subclass (the cable), as opposed
            # to the multi-variant object pool, whose `spawn_paths` `install_multi_asset` pins to
            # a list -- there `expected` is that list and the check still bites.
            continue
        if actual != expected:
            raise RuntimeError(
                f"clone plan moved the prototypes for {cfg.prim_path}: spawned at {expected!r}, "
                f"plan wants {actual!r}. Per-environment variant assignment is no longer "
                "reproducible against the 2.3 reference."
            )


# ============================================================================
# Collision geometry: CollisionAPI onto mesh prims (Newton)
# ============================================================================
#
# Push ``UsdPhysics.CollisionAPI`` down onto mesh prims so Newton imports robot collisions.
#
# The problem
# -----------
# Isaac Sim's URDF importer authors each collision as an **Xform** carrying
# ``UsdPhysics.CollisionAPI``, with the actual geometry in a child mesh prim
# (``node_STL_BINARY_`` for STL sources)::
#
#     .../Robot/iiwa14_link_0/collisions/link_0        Xform  + CollisionAPI
#     .../Robot/iiwa14_link_0/collisions/link_0/node_STL_BINARY_   <- real geometry
#
# PhysX descends through the Xform to find the geometry. Newton's USD importer expects
# ``CollisionAPI`` on a ``UsdGeomGPrim`` and skips anything else, logging::
#
#     Warning: CollisionAPI applied to an unknown UsdGeomGPrim type, prim
#       /World/envs/env_0/Robot/iiwa14_link_0/collisions/link_0/node_STL_BINARY_
#
# The result is silent and severe: **every robot collision shape is dropped**. The finalized
# model held 41 shapes (8 envs x 5 non-robot shapes + ground) instead of ~300, with no shape
# attached to any robot body — a robot that cannot touch anything, in a manipulation task. The
# object and table survive because they are authored as primitive geometry, so the failure looks
# like a partial success rather than an error.
#
# The fix
# -------
# Walk the stage and, for every prim carrying ``CollisionAPI`` that is not itself a gprim, apply
# ``CollisionAPI`` to its descendant mesh prims and copy the PhysX-namespaced offsets down.
#
# Applied **in the Newton process only**, against the live stage after spawning. The
# content-addressed USD cache on disk is left untouched, so the PhysX baseline keeps consuming
# byte-identical assets and the M4 comparison stays honest.
#
# Must run before the Newton model is finalized, i.e. before replication.

_CONTACT_OFFSET_ATTR = "physxCollision:contactOffset"
_REST_OFFSET_ATTR = "physxCollision:restOffset"


def author_convex_hull_approximation(stage, root_path: str = "/World/envs") -> int:
    """Mark meshified object colliders as convex hulls, so Newton does not run triangle-mesh
    collision on them.

    Newton's USD importer picks a collision representation from ``physics:approximation`` on
    ``UsdPhysicsMeshCollisionAPI`` (``import_usd.py:514-521`` maps ``"convexhull"`` ->
    ``convex_hull``). Upstream never authors that attribute -- it has no reason to, since its
    tools are box/capsule primitives -- so once ``--meshify`` replaces them with ``<mesh>``
    geometry they import as ``GeoType.MESH``: a general triangle mesh.

    Measured consequence: with meshify on, a *single shared object* rollout that is stable with
    primitives diverges to NaN within one episode (the policy then samples a Normal with
    negative std). The tools are convex by construction -- each is one box or one capsule
    rendered as a hull -- so declaring them convex costs no fidelity and gives MuJoCo the
    contact path it is good at.

    Only the object variants are touched. The robot's collision meshes are already ``MESH`` and
    the backend is stable with them, so changing those would be an unmeasured second change.

    Args:
        stage: USD stage to fix up in place.
        root_path: Subtree to scan.

    Returns:
        Number of collision meshes marked.
    """
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0

    marked = 0
    for prim in Usd.PrimRange(root):
        path = prim.GetPath().pathString
        if "/Object" not in path and "/GoalViz" not in path:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        attr = mesh_api.GetApproximationAttr()
        if not attr:
            attr = mesh_api.CreateApproximationAttr()
        attr.Set(UsdPhysics.Tokens.convexHull)
        marked += 1
    return marked


def push_collision_api_to_meshes(stage, root_path: str = "/World/envs") -> tuple[int, int]:
    """Apply ``CollisionAPI`` to mesh descendants of non-gprim collision prims.

    Args:
        stage: USD stage to fix up in place.
        root_path: Subtree to scan.

    Returns:
        ``(num_source_prims_fixed, num_meshes_marked)``.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return (0, 0)

    fixed = 0
    marked = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # Already a geometric primitive -> Newton handles it directly.
        if prim.IsA(UsdGeom.Gprim):
            continue

        contact_offset = prim.GetAttribute(_CONTACT_OFFSET_ATTR)
        rest_offset = prim.GetAttribute(_REST_OFFSET_ATTR)

        descendant_meshes = [
            child
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
            if child != prim and child.IsA(UsdGeom.Gprim)
        ]
        if not descendant_meshes:
            continue

        # Whether the *parent* collider is enabled. Upstream authors `collisionEnabled=False` on
        # the goal marker -- a full copy of the object parked at the goal pose -- and pushing the
        # API down without this produces children that default to True. The marker then becomes a
        # solid obstacle sitting exactly where the object is trying to go: grasping is untouched
        # and placement is impossible. Measured with `--meshify`, where the goal's colliders are
        # freshly generated mesh prims rather than the authored primitives:
        #
        #     GoalViz  World (Xform)  collisionEnabled=False   Newton shape_flags [0, 0]
        #     GoalViz  mesh  (Mesh)   collisionEnabled=True    Newton shape_flags [6, 6]
        #
        # and the backend scored 0.000 goals/episode at lift 0.78.
        parent_enabled_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        parent_enabled = parent_enabled_attr.Get() if parent_enabled_attr else None

        for mesh in descendant_meshes:
            if not mesh.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(mesh)
            if parent_enabled is not None:
                api = UsdPhysics.CollisionAPI(mesh)
                target = api.GetCollisionEnabledAttr() or api.CreateCollisionEnabledAttr()
                target.Set(bool(parent_enabled))
            # Carry the authored offsets down; Newton reads these PhysX-namespaced names.
            for attr, name in ((contact_offset, _CONTACT_OFFSET_ATTR), (rest_offset, _REST_OFFSET_ATTR)):
                if attr and attr.HasAuthoredValue():
                    target = mesh.GetAttribute(name)
                    if not target:
                        from pxr import Sdf

                        target = mesh.CreateAttribute(name, Sdf.ValueTypeNames.Float)
                    target.Set(attr.Get())
            marked += 1
        fixed += 1

    return (fixed, marked)


# ============================================================================
# USD drive gains (Newton)
# ============================================================================
#
# Author real PD gains into the robot's USD drives before Newton builds its solver.
#
# The actual reason the robot never moves
# ---------------------------------------
# Upstream authors every joint's ``UsdPhysics.DriveAPI`` with **zero** stiffness and damping,
# and supplies the real gains at runtime via ``ImplicitActuatorCfg``. The intent is documented in
# ``scene_utils._robot_joint_drive_cfg``: *"DriveAPI prims must exist for ImplicitActuator runtime
# gains to land."* PhysX honours that runtime injection.
#
# Newton does not work that way. ``SolverMuJoCo`` builds its ``mjModel`` from the USD-derived
# Newton model at construction time, so zero-gain drives produce a robot with no effective joint
# actuation. Everything downstream then looks correct while doing nothing:
#
#     Control.joint_target_q    = -1.570 .. 1.572     targets reach the solver
#     Model.joint_target_ke     = 0.90 .. 600.00      gains on the Newton model
#     Model.joint_target_mode   = POSITION_VELOCITY   on all 58 robot DoFs
#     robot.data.joint_pos delta over 60 steps = 0.0  nothing moves
#
# Measured on the live stage, every one of the 29 joint prims carries::
#
#     drive:angular:physics:stiffness = 0.0
#     drive:angular:physics:damping   = 0.0
#     drive:angular:physics:maxForce  = 300.0
#
# ``notify_model_changed(JOINT_DOF_PROPERTIES)`` does not rescue this: it refreshes cached
# parameter buffers, it cannot create actuation that was never built.
#
# So the gains are written into USD *before* replication and finalization, taken from the same
# tables the PhysX path uses (the env's ``robots.RobotSpec``: ``arm_stiffness`` /
# ``hand_stiffness`` and their damping counterparts). Both backends therefore run the same
# numbers; only the delivery mechanism differs, which is precisely what transfer invariant 3 in
# upstream's own test suite pins down.
#
# Applied to the live stage in the Newton process only, so the content-addressed USD cache stays
# byte-identical and the PhysX baseline is untouched.

_STIFFNESS_ATTR = "drive:angular:physics:stiffness"
_DAMPING_ATTR = "drive:angular:physics:damping"
_MAX_FORCE_ATTR = "drive:angular:physics:maxForce"


def author_drive_gains(stage, robot_spec, root_path: str = "/World/envs") -> tuple[int, int]:
    """Write ``robot_spec``'s PD gains onto every robot joint drive under ``root_path``.

    Returns ``(joints_written, joints_missing_gains)``.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stiffness = {**robot_spec.arm_stiffness, **robot_spec.hand_stiffness}
    damping = {**robot_spec.arm_damping, **robot_spec.hand_damping}
    # Torque limits only where the spec overrides the URDF's (the importer's maxForce otherwise).
    max_force = dict(robot_spec.hand_effort_limit)

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return (0, 0)

    written = 0
    missing = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        name = prim.GetName()
        if name not in stiffness:
            continue
        if not prim.GetAttribute(_STIFFNESS_ATTR):
            missing += 1
            continue

        tables = [(_STIFFNESS_ATTR, stiffness), (_DAMPING_ATTR, damping)]
        if name in max_force:
            tables.append((_MAX_FORCE_ATTR, max_force))
        for attr_name, table in tables:
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            attr.Set(float(table[name]))
        written += 1

    return (written, missing)


# ============================================================================
# Actuator armature and joint friction
# ============================================================================
#
# Write actuator armature and joint friction to the simulator, as Isaac Lab 2.x did.
#
# The defect
# ----------
# Isaac Lab 2.3 pushes every actuator property into the simulation when it processes the actuator
# configs (``articulation.py``)::
#
#     self.write_joint_effort_limit_to_sim(actuator.effort_limit_sim, ...)
#     self.write_joint_velocity_limit_to_sim(actuator.velocity_limit_sim, ...)
#     self.write_joint_armature_to_sim(actuator.armature, ...)
#     self.write_joint_friction_coefficient_to_sim(actuator.friction, ...)
#
# Isaac Lab 3.0 reimplemented ``_process_actuators_cfg`` per backend, and both implementations
# write **stiffness, damping, effort limit and velocity limit — and stop there**. ``armature`` is
# passed *into* the actuator constructor as the USD-authored default and then never written back,
# so an ``ImplicitActuatorCfg(armature=...)`` value is silently discarded and the simulation keeps
# whatever the asset authored, which for a URDF-converted robot is ``0``.
#
# Measured on this scene (``eval/property_audit.py``, identical protocol on both stacks):
#
# ===============================  ===========  ===========
# joint                            2.3          3.0 port
# ===============================  ===========  ===========
# ``iiwa14_joint_*`` (7 arm)       0.0          0.0
# ``left_*_MCP_FE`` (hand)         0.00265      **0.0**
# ``left_*_PIP`` / ``*_IP``        0.00060      **0.0**
# ``left_*_DIP``                   0.00042      **0.0**
# ``left_thumb_CMC_*``             0.00320      **0.0**
# ===============================  ===========  ===========
#
# Everything else the audit compares — stiffness, damping, joint limits, masses, inertias,
# default pose, post-reset state — is identical. Only armature is lost.
#
# Why it matters here specifically
# --------------------------------
# Armature is rotor inertia added directly to the joint-space mass matrix. On this hand it is not
# a numerical nicety: the finger links are grams and the URDF's fingertip placeholder links are
# ``1e-6`` kg, so armature is a large fraction of the effective inertia those joints have. Removing
# it leaves the lowest-stiffness joints (PIP/DIP, stiffness 0.9) nearly inertia-free, and they
# respond to the same command completely differently — visible one tick after reset in an
# open-loop replay of an identical action sequence:
#
# ===================  ==========================
# joint                normalised |diff| at t=0
# ===================  ==========================
# arm (all 7)          <= 0.0004
# ``*_PIP`` (x4)       0.135
# ``*_DIP`` (x4)       0.027
# ===================  ==========================
#
# The arm, which has no armature in either stack, matches to 1e-4. That is the fingerprint: the
# divergence is exactly the set of joints whose armature was dropped.
#
# This is the same class of defect as the per-body gravity attribute (``gravity_patch``): a
# property upstream sets, that 3.0 accepts and then ignores. It affects **both** 3.0 backends,
# which is consistent with OVPhysX and Newton scoring the same 6-9 goal hits against the 2.3
# baseline's 285.
#
# The fix
# -------
# Wrap ``_process_actuators_cfg`` on whichever 3.0 articulation classes are importable and, after
# the original runs, write each actuator's resolved ``armature`` and ``friction`` into the sim and
# into the ``default_joint_*`` buffers — the same four calls 2.3 makes, no new values invented.


# Concrete 3.0 articulation implementations. Both are patched when present so a run does not
# depend on which backend the process happens to import first.
_ARMATURE_TARGETS = (
    ("isaaclab_ov.assets.articulation.articulation", "Articulation"),
    ("isaaclab_newton.assets.articulation.articulation", "Articulation"),
)

_ARMATURE_FLAG = "_simtoolreal_armature_patch"


def install_armature_writeback() -> list[str]:
    """Patch every importable 3.0 articulation class. Returns the ones patched."""
    patched: list[str] = []
    for module_path, class_name in _ARMATURE_TARGETS:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue  # backend not installed in this venv; nothing to patch
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, _ARMATURE_FLAG, False):
            continue
        _patch_armature_class(cls)
        patched.append(f"{module_path}.{class_name}")
    return patched


def _patch_armature_class(cls) -> None:
    original = cls._process_actuators_cfg

    def _process_actuators_cfg(self):
        original(self)
        _write_missing_properties(self)

    cls._process_actuators_cfg = _process_actuators_cfg
    setattr(cls, _ARMATURE_FLAG, True)


def _write_missing_properties(articulation) -> None:
    """Write armature and friction for every actuator, mirroring Isaac Lab 2.x."""
    for name, actuator in getattr(articulation, "actuators", {}).items():
        # `_joint_ids_per_actuator` is an OVPhysX implementation detail; the Newton articulation
        # has no such map, so fall back to the actuator's own indices, which both backends carry.
        per_actuator = getattr(articulation, "_joint_ids_per_actuator", None)
        if per_actuator is not None and name in per_actuator:
            joint_ids = per_actuator[name]
        else:
            joint_ids = getattr(actuator, "joint_indices", None)
        if joint_ids is None:
            continue

        # The 3.0 writers are keyword-only, and each names its value differently.
        for kwarg, value, writer in (
            ("armature", getattr(actuator, "armature", None),
             "write_joint_armature_to_sim_index"),
            ("joint_friction_coeff", getattr(actuator, "friction", None),
             "write_joint_friction_coefficient_to_sim_index"),
            # Viscous joint friction (passive damping), only where the actuator cfg sets it (the
            # RobotSpec's hand_viscous_friction): writing the actuator's default 0 elsewhere could
            # overwrite a USD-authored value. MJWarp has one Coulomb term, so the cfg's
            # dynamic_friction has nowhere to go; the static value above is the frictionloss.
            ("joint_viscous_friction_coeff",
             getattr(actuator, "viscous_friction", None)
             if getattr(getattr(actuator, "cfg", None), "viscous_friction", None) is not None else None,
             "write_joint_viscous_friction_coefficient_to_sim_index"),
        ):
            if value is None:
                continue
            write = getattr(articulation, writer, None)
            if write is None:
                continue
            write(**{kwarg: value}, joint_ids=joint_ids)

        # 2.x also records the configured values as the defaults, so a later reset restores the
        # actuator's armature rather than the asset's zero.
        data = articulation._data
        for attr, value in (("default_joint_armature", getattr(actuator, "armature", None)),
                            ("default_joint_friction_coeff", getattr(actuator, "friction", None))):
            if value is None:
                continue
            buffer = getattr(data, attr, None)
            if buffer is None:
                continue
            torch_view = getattr(buffer, "torch", None)
            target = buffer if torch_view is None or callable(torch_view) else torch_view
            try:
                target[:, joint_ids] = value
            except (TypeError, IndexError, RuntimeError):
                # A read-only or differently-shaped buffer is not worth failing construction
                # over: the sim-side write above is what governs the dynamics.
                pass


# ============================================================================
# Removed: PhysX-only patches
# ============================================================================
#
# Two sections of the reference implementation are deliberately absent here:
#
#   PhysX scene tuning        install_physx_scene / _author_scene_attrs
#   Per-shape friction views  install_physx_view_materials / install_ov_view_materials
#
# They serve the `ovphysx` and `physx3` backends, which existed there to isolate port
# defects from solver defects across four stacks. This repo runs PhysX through Isaac Lab
# 2.3 (`Isaacsimenvs-Play-Direct-v0`), where upstream's own config reaches the scene
# natively and `apply_physx_material_properties` works as written -- so neither patch has
# anything to fix. Recover them from isaac_newton @ beb9efb if a 3.0 PhysX backend is ever
# added.
# ============================================================================
# Placeholder inertia (Newton)
# ============================================================================
#
# Restore the URDF's negligible inertia on massless placeholder links.
#
# The problem
# -----------
# ``ModelBuilder.finalize()`` validates inertia tensors and floors degenerate ones, warning::
#
#     Inertia validation corrected 11 bodies
#
# The corrected bodies are the hand's massless placeholders — the five ``*_MCP_VL`` /
# ``*_CMC_VL`` virtual links and the five fingertips — which the URDF deliberately declares as
# ``mass=1e-6, inertia=2e-12``. Newton raises their inertia to ``1e-6``.
#
# That floor is not negligible here. Compare against a *real* finger link::
#
#     left_thumb_MCP_VL  (massless placeholder)  Idiag = 1.000e-06     <- after correction
#     left_thumb_DP      (real distal phalanx)   Idiag = 2.055e-07
#
# The placeholders end up with roughly **five times the inertia of an actual finger bone**, and
# they sit inside every finger's kinematic chain. The chain's effective inertia is therefore
# dominated by links that should contribute nothing, so the fingers respond sluggishly to the
# same PD gains PhysX uses.
#
# Measured effect, holding a fixed +0.20 rad target for 120 steps:
#
#     PhysX 2.3   hand mean |err| = 0.0517,  moved 0.160 of 0.200
#     Newton      hand mean |err| = 0.1312,  moved 0.115 of 0.200
#
# Since the policy grasps with its fingers, sluggish finger tracking is enough to stop it lifting
# at all, which is what drove lift rate to ~0 and mean reward to 0.15 against 13.
#
# The fix
# -------
# Rewrite those bodies' inertia after finalization to a value far below the smallest real link,
# preserving the URDF's intent (placeholders contribute nothing) while staying non-zero so the
# solver stays conditioned. PhysX runs the original ``2e-12`` without trouble.
#
# Only bodies whose mass is at or below ``mass_threshold`` are touched, so real links are never
# modified.

# Two orders of magnitude below the smallest real finger link (~2e-7), and far above zero.
NEGLIGIBLE_INERTIA = 1e-9
MASS_THRESHOLD = 1e-5


def restore_placeholder_inertia(env, *, inertia: float = NEGLIGIBLE_INERTIA) -> int:
    """Reset inertia on massless placeholder bodies. Returns the number changed."""
    import numpy as np
    import warp as wp

    model = env.sim.physics_manager.get_model()
    mass = model.body_mass.numpy()
    tensor = model.body_inertia.numpy().copy()

    targets = np.nonzero(mass <= MASS_THRESHOLD)[0]
    if targets.size == 0:
        return 0

    for idx in targets:
        tensor[idx] = np.eye(3, dtype=tensor.dtype) * inertia

    model.body_inertia.assign(wp.array(tensor, dtype=model.body_inertia.dtype))

    # Inverse inertia must agree or the solver keeps using the old values.
    inv = getattr(model, "body_inv_inertia", None)
    if inv is not None:
        inv_tensor = inv.numpy().copy()
        for idx in targets:
            inv_tensor[idx] = np.eye(3, dtype=inv_tensor.dtype) * (1.0 / inertia)
        inv.assign(wp.array(inv_tensor, dtype=inv.dtype))

    return int(targets.size)


# ============================================================================
# Solver refresh (Newton)
# ============================================================================
#
# Tell the Newton solver about model data written *after* it was constructed.
#
# Why the robot does not move
# ---------------------------
# ``SolverMuJoCo`` builds its own ``mjModel`` from the Newton model at construction time and
# caches joint gains, targets and modes there. ``Model.joint_target_ke`` is therefore only read
# once; writing it later updates the Newton model but not the solver's copy.
#
# That interacts badly with how upstream configures the arm and hand. The robot's USD ``DriveAPI``
# is deliberately authored with **zero** stiffness and damping — see
# ``scene_utils._robot_joint_drive_cfg``: *"DriveAPI prims must exist for ImplicitActuator runtime
# gains to land"* — and the real gains are supplied at runtime through ``ImplicitActuatorCfg``.
# PhysX accepts that runtime injection. Under Newton the solver is built from the zero-gain USD,
# so MuJoCo's cached actuator gains stay at zero no matter what Isaac Lab writes afterwards, and
# the joints receive no torque even though every observable check passes:
#
#     Control.joint_target_q   = -1.570 .. 1.572   (targets reach the solver)
#     Model.joint_target_ke    = 0.90 .. 600.00    (gains on the Newton model)
#     Model.joint_target_mode  = POSITION_VELOCITY on all robot DoFs
#     robot.data.joint_pos delta over 60 steps = 0.0
#
# Newton provides the escape hatch: ``SolverBase.notify_model_changed(flags)`` refreshes the
# solver's internal buffers without rebuilding it. ``ModelFlags.JOINT_DOF_PROPERTIES`` is
# documented as covering "joint axis limits, targets, modes, DOF state, or force buffers".
#
# The same reasoning applies to :mod:`isaacsimenvs.tasks.cloth.backends.vbd.newton_support.patches`, which writes
# ``Model.shape_material_mu`` after finalization — fingertip friction would otherwise also be
# cached at its pre-write value. ``ModelFlags.SHAPE_PROPERTIES`` covers that.
#
# This is why the shipped Cartpole task never hits the problem: its gains are authored in USD, so
# the solver builds with the correct values and nothing is written afterwards.

def notify_solver(env, *, joints: bool = True, shapes: bool = True) -> list[str]:
    """Refresh solver-cached model data. Returns the flag names applied."""
    import newton

    manager = env.sim.physics_manager
    solver = getattr(manager, "_solver", None)
    if solver is None or not hasattr(solver, "notify_model_changed"):
        return []

    flags = 0
    applied: list[str] = []
    if joints:
        flags |= int(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        applied.append("JOINT_DOF_PROPERTIES")
    if shapes:
        flags |= int(newton.ModelFlags.SHAPE_PROPERTIES)
        applied.append("SHAPE_PROPERTIES")

    if flags:
        solver.notify_model_changed(flags)

    # Explicit geom refresh. NOTE: this proved to be a no-op -- mjw_model.geom_friction (the
    # array actually simulated) already carried the fingertip mu=1.5 written by material_patch.
    # The apparent discrepancy came from reading solver.mj_model, the CPU template, which is
    # stale by design. Kept as defensive hygiene, but it fixed nothing.
    updater = getattr(solver, "_update_geom_properties", None)
    if shapes and callable(updater):
        updater()
        applied.append("geom_properties")

    return applied


# ============================================================================
# Implicit actuators: the PD law applied twice
# ============================================================================
#
# The defect
# ----------
# An ``ImplicitActuator`` is, by definition, handled by the solver: Isaac Lab writes its
# stiffness and damping into the drive and the physics engine integrates the PD law. Its
# ``compute()`` is documented as doing nothing else -- "This function is a no-op and does not
# perform any computation on the input control action", and the torque it stores is explicitly
# "for reward computation ... since PhysX does not compute this quantity explicitly".
#
# Isaac Lab 2.3 respects that. Its ``write_data_to_sim`` sends ``_joint_effort_target_sim`` --
# the user's feed-forward effort, zero for this task -- to ``set_dof_actuation_forces``, and the
# bookkeeping torque never reaches the simulation.
#
# Both Isaac Lab 3.0 backends push that bookkeeping torque into the sim instead::
#
#     at[:, joint_ids] = act.applied_effort          # _apply_actuator_model
#     self._root_view.set_attribute(TT.DOF_ACTUATION_FORCE, effort)   # write_data_to_sim
#     self._root_view.set_attribute(TT.DOF_POSITION_TARGET, pos_target)
#
# while ``_process_actuators_cfg`` has already written the same stiffness and damping into the
# drive. So the same PD law is applied **twice**: once implicitly by the solver, once explicitly
# as a joint actuation force.
#
# How it was found
# ----------------
# With every static property matched between the stacks, a coasting test still diverged. Zero
# the stiffness and damping, give one joint 0.5 rad/s, and step once::
#
#     Isaac Lab 2.3    velocity retained 0.4970 of 0.5   (coasts, as it must)
#     3.0 port         velocity retained 0.2260 of 0.5   (55% gone in one step)
#
# with the gains reading back as 0.0 on both stacks afterwards. Nothing in the *simulation* was
# damping that joint -- the actuator object still held its own stiffness and damping, and the
# explicit torque built from them was still being applied. The same doubling shows up in free
# space as a ~1.5x larger first-step response on the low-gain finger joints, while the arm,
# whose motion is dominated by its own inertia, matches to 0.001 rad.
#
# The fix
# -------
# Zero the applied torque for joints owned by an implicit actuator, after the actuator model
# runs and before it is written, which is exactly 2.3's behaviour. Explicit actuators are left
# alone: their torque is the whole point.


_IMPLICIT_EFFORT_FLAG = "_simtoolreal_implicit_effort_fix"


def install_implicit_effort_fix() -> list[str]:
    """Stop implicit actuators applying their bookkeeping torque. Returns the classes patched."""
    patched: list[str] = []
    for module_path, class_name in (
        ("isaaclab_ov.assets.articulation.articulation", "Articulation"),
        ("isaaclab_newton.assets.articulation.articulation", "Articulation"),
    ):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, _IMPLICIT_EFFORT_FLAG, False):
            continue

        original = cls._apply_actuator_model

        def _apply_actuator_model(self, _original=original):
            _original(self)
            _zero_implicit_effort(self)

        cls._apply_actuator_model = _apply_actuator_model
        setattr(cls, _IMPLICIT_EFFORT_FLAG, True)
        patched.append(f"{module_path}.{class_name}")
    return patched


def _zero_implicit_effort(articulation) -> None:
    from isaaclab.actuators import ImplicitActuator

    applied = getattr(articulation._data, "_applied_torque", None)
    if applied is None:
        return
    import warp as wp

    torch_view = wp.to_torch(applied) if not hasattr(applied, "torch") else applied.torch
    for name, actuator in getattr(articulation, "actuators", {}).items():
        if not isinstance(actuator, ImplicitActuator):
            continue
        per_actuator = getattr(articulation, "_joint_ids_per_actuator", None)
        if per_actuator is not None and name in per_actuator:
            joint_ids = per_actuator[name]
        else:
            joint_ids = getattr(actuator, "joint_indices", None)
        if joint_ids is None:
            continue
        torch_view[:, joint_ids] = 0.0


# ============================================================================
# Contact dimensionality: torsional friction for the fingertips (Newton)
# ============================================================================
#
# MuJoCo's default ``condim`` is 3: sliding friction only. Torsional and rolling friction exist
# in the geom's friction triple but are simply not applied. Measured on the model Newton builds::
#
#     geom_condim   : all 3
#     geom_friction : [0.5, 0.005, 0.0001] and [1.5, 0.005, 0.0001]
#
# So the torsional coefficient the assets carry is inert, and nothing resists a grasped tool
# twisting between the fingertips -- which is exactly the failure this port shows.
#
# The SimToolReal authors' own MuJoCo model of this robot
# (``simtoolreal.github.io``, ``mujoco_wasm/assets/scenes/iiwa_sharpa.xml``) does not use the
# default. It sets ``condim="6"`` on the fingertip elastomer pads, on both object geoms and on
# the table::
#
#     <default class="palmelastomer_geom"><geom condim="6"/></default>
#     <geom name="table_geom"          ... friction="1 0.005 0.0001" condim="6"/>
#     <geom name="object_handle_geom"  ... condim="6"/>
#     <geom name="object_head_geom"    ... condim="6"/>
#
# together with ``cone="elliptic"`` and ``impratio="10"``. This applies the contact-dimension
# half of that; the cone and impratio are defaults on :class:`NewtonSolverKnobs`.


def install_newton_condim(condim: int, grasping: tuple[str, ...]) -> bool:
    """Set MuJoCo contact dimensionality on the grasping shapes *before* the solver compiles.

    Newton exposes this as a builder custom attribute, which is the only point at which it can
    be set: MuJoCo Warp sizes its constraint arrays when it compiles the model, so writing
    ``geom_condim`` afterwards produces NaNs rather than torsional friction.

    ``NewtonManager._prepare_builder_for_finalize`` runs immediately before
    ``ModelBuilder.finalize``, which is where Newton's own examples do the same thing::

        condim_attr = builder.custom_attributes["mujoco:condim"]
        condim_attr.values[shape_idx] = 4

    Args:
        condim: 3 sliding only (MuJoCo default), 4 adds torsional, 6 adds rolling. The
            SimToolReal authors' own MuJoCo model uses 6 on the fingertip pads, tool and table.
        grasping: Fingertip link names supplied by the backend's robot specification.

    Returns:
        Whether the hook was installed.
    """
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except Exception:
        return False
    if getattr(NewtonManager, "_simtoolreal_condim_patch", False):
        return False

    original = NewtonManager._prepare_builder_for_finalize

    def _prepare(cls_or_builder, builder=None):
        # staticmethod/classmethod shape differs across versions; normalise.
        b = builder if builder is not None else cls_or_builder
        if builder is None:
            original(b)
        else:
            original(cls_or_builder, b)
        _apply_condim(b, condim, grasping)

    NewtonManager._prepare_builder_for_finalize = _prepare
    NewtonManager._simtoolreal_condim_patch = True
    return True


def install_newton_gravcomp() -> bool:
    """Disable gravity on the robot natively, the way upstream does, instead of reconstructing it.

    Upstream authors per-body ``disableGravity=True`` on the robot and leaves it on for the
    object. This port long recorded that "Newton has no per-body gravity -- no ``BodyFlags``
    entry, no ``add_body(gravity=...)``", and worked around it by zeroing **global** gravity and
    re-applying the object's as an explicit per-step wrench. That workaround is what produced the
    body-frame-force defect, and then the spurious ``(com - origin) x F`` torque added to patch
    *that*.

    The premise was wrong. MuJoCo expresses exactly this as per-body **gravity compensation**,
    and Newton registers it as a builder custom attribute
    (``newton/_src/solvers/mujoco/solver_mujoco.py:1074-1081``: ``name="gravcomp"``,
    ``AttributeFrequency.BODY``, ``usd_attribute_name="mjc:gravcomp"``), applied per body in the
    solver kernel as ``out[body] = (1.0 - gravcomp) * g`` (``kernels.py:245-266``). So
    ``gravcomp = 1.0`` *is* ``disableGravity``, reachable through the same
    ``_prepare_builder_for_finalize`` hook the condim patch already uses.

    With this installed, global gravity stays at -9.81 and the object falls natively -- no
    reconstruction, no wrench, no compensating torque, and one fewer Newton-only deviation from
    the reference.

    Returns:
        Whether the hook was installed.
    """
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except Exception:
        return False
    if getattr(NewtonManager, "_simtoolreal_gravcomp_patch", False):
        return False

    original = NewtonManager._prepare_builder_for_finalize

    def _prepare(cls_or_builder, builder=None):
        b = builder if builder is not None else cls_or_builder
        if builder is None:
            original(b)
        else:
            original(cls_or_builder, b)
        _apply_gravcomp(b)

    NewtonManager._prepare_builder_for_finalize = _prepare
    NewtonManager._simtoolreal_gravcomp_patch = True
    return True


def _apply_gravcomp(builder) -> None:
    """Set ``mujoco:gravcomp = 1`` on every robot body, leaving the object and table alone."""
    attrs = getattr(builder, "custom_attributes", None)
    if attrs is None or "mujoco:gravcomp" not in attrs:
        print("[gravcomp] builder has no 'mujoco:gravcomp' attribute; skipped", flush=True)
        return
    attr = attrs["mujoco:gravcomp"]
    if attr.values is None:
        attr.values = {}

    labels = getattr(builder, "body_label", None) or getattr(builder, "body_key", None) or []
    n = 0
    for body_idx, label in enumerate(labels):
        path = str(label)
        # Robot bodies only. The object and the table keep gravity, as upstream authors it.
        if "/Robot" in path or "iiwa14" in path or "left_" in path or "sharpa" in path:
            attr.values[body_idx] = 1.0
            n += 1
    print(f"[gravcomp] mujoco:gravcomp=1 on {n}/{len(labels)} bodies before finalize", flush=True)


def _apply_condim(builder, condim: int, grasping: tuple[str, ...]) -> None:
    """Tag the fingertip pads, the tool and the table with ``condim`` on ``builder``."""
    attrs = getattr(builder, "custom_attributes", None)
    if attrs is None or "mujoco:condim" not in attrs:
        print("[condim] builder has no 'mujoco:condim' attribute; skipped", flush=True)
        return
    attr = attrs["mujoco:condim"]
    if attr.values is None:
        attr.values = {}

    labels = getattr(builder, "body_label", None) or getattr(builder, "body_key", None) or []
    labels = [str(label).rsplit("/", 1)[-1] for label in labels]
    shape_body = builder.shape_body

    n = 0
    for shape_idx in range(builder.shape_count):
        body_idx = int(shape_body[shape_idx])
        if body_idx < 0 or body_idx >= len(labels):
            continue
        name = labels[body_idx]
        if any(f in name for f in grasping) or "object" in name.lower() or "box" in name.lower():
            attr.values[shape_idx] = condim
            n += 1
    print(f"[condim] mujoco:condim={condim} on {n} grasping shapes before finalize", flush=True)
