"""Every runtime patch the Isaac Lab 3.0 port applies, in one place.

Upstream SimToolReal is Isaac Lab 2.x code, and this module is the whole of what stands between
it and Isaac Lab 3.0: API renames, warp-backed asset data, quaternion conventions, replication
that 2.x did implicitly, and the physics properties 3.0 accepts and then ignores.

Each section keeps the finding that motivated it, because several of these look like arbitrary
compatibility shims until you know what they cost to discover. Which patches a given backend
applies is decided in :mod:`isaacsimenvs.tasks.play_newton.play_newton_env_cfg` -- deliberately not here, so that a
patch written for one solver cannot leak into another.

Sections in application order:

* multi-asset spawning, actuator defaults, asset data proxying -- API shape
* quaternion math and conventions, fingertip ordering -- convention
* replication, collision geometry, USD drive gains -- scene construction
* armature, PhysX scene tuning, gravity, inertia, friction, solver refresh -- physics fidelity
* meshified collision geometry -- opt-in, off by default; the one section here that deliberately
  *changes* the physics rather than preserving it, so it is only ever installed on request
"""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Any


# ============================================================================
# Multi-asset spawning
# ============================================================================
#
# Restore Isaac Lab 2.x multi-asset semantics: one variant per environment, round-robin.
#
# Upstream gets its object diversity from
# ``MultiUsdFileCfg(usd_path=[...variants...])`` spawned at ``/World/envs/env_.*/Object``.
# Under Isaac Lab 2.x, ``spawn_multi_asset`` expanded the ``env_.*`` regex to every environment
# prim and cycled the variant list *across environments* — one object per env, differing
# between envs. That is the core of the SimToolReal setup: a single policy trained over many
# procedurally generated tools.
#
# Isaac Lab 3.0 redefined the same call. It now spawns **every** asset at **distinct sibling
# paths inside one** environment, which is why it demands ``.*`` in the prim path's base name::
#
#     ValueError: The base name 'Table' in the prim path '/World/envs/env_.*/Table' must
#     contain '.*' ...
#
# Its deprecation text points at ``InteractiveSceneCfg.random_heterogeneous_cloning``, which
# does not exist anywhere in the 3.0 source — it appears only in warning strings and docstrings.
# The genuine successor is the cloner (``CloneCfg.clone_strategy``, with
# ``cloner_strategies.sequential`` documented as round-robin prototype assignment).
#
# Two routes exist in 3.0, and the difference is not cosmetic.
#
# The *shortcut* is ``MultiUsdFileCfg.spawn_paths``, "optional concrete spawn paths, one per USD
# path": enumerate one path per environment, cycle the variants, and every environment's prims
# are authored directly. The USD that comes out is correct, and it is what this port used first.
#
# The *sanctioned* route is prototypes plus the cloner. ``make_clone_plan``
# (``isaaclab/cloner/clone_plan.py:219``) expands a ``MultiUsdFileCfg`` into one **prototype row
# per variant**, builds a ``[num_rows, num_envs]`` clone mask, and rewrites each cfg's
# ``spawn_paths`` so only the prototypes are spawned. ``cloner.replicate`` then materialises the
# rest, per backend.
#
# The shortcut is a trap on every backend that replicates, because *replication is what builds
# the simulated model*, and it works from the plan, not from the stage:
#
# * kit-less PhysX serialises the stage for the ovphysx wheel and, unless asked for the full
#   stage, **strips ``/World/envs/env_<i!=0>`` outright** (``ovphysx_manager.py:757``), then
#   re-creates those environments with ``physx.clone()`` from env_0. Measured: the serialized
#   USDA handed to the wheel holds one ``def Xform "env_0"`` and exactly one object mass, while
#   the live stage holds all four correct ones.
# * Newton's importer calls ``add_usd(..., ignore_paths=["/World/envs", *sources])``
#   (``isaaclab_newton/cloner/replicate.py``) -- everything under the env root is ignored except
#   the plan's own sources, and worlds are built by ``replicate_builder_mapping``. Prims the plan
#   does not name simply do not exist in the model.
#
# So the per-environment variants have to be expressed as *plan rows*, not as authored prims.
#
# Ordering is the part that has to be verified rather than assumed. Upstream 2.x gives
# environment *i* variant ``i % len(usd_paths)``, independently for the object and its goal
# marker. ``make_clone_plan``'s default enumerates the **cartesian product** of every
# variant-carrying cfg, so with 4 objects and 4 goal markers ``cloner_strategies.sequential``
# would hand environments 0..3 the combinations ``(obj 0, goal 0), (obj 0, goal 1), ...`` --
# one object everywhere, which is the failure this section exists to remove, wearing a different
# hat. The fix is an explicit ``valid_set`` holding only the *diagonal* combinations, so every
# group advances together and environment *i* gets variant ``i % n`` of each; see
# :func:`_round_robin_valid_set`.
#
# Determinism matters here more than elegance: if environment *i* holds a different object under
# Newton than it does under PhysX, the M4 comparison is no longer like-for-like, and the
# resulting delta would confound object assignment with physics.
#
# Note this does *not* compose with upstream's ``cfg.scene.replicate_physics=False``, which was
# 2.x's way of saying "parse each environment independently because its contents differ". In 3.0
# that flag only tells :func:`~isaaclab.cloner.replicate` to drop every physics context, leaving
# USD replication -- which kit-less is not registered either, so nothing would be replicated at
# all. Heterogeneity now lives in the plan's rows instead, so :func:`install_cloning` drains its
# own queue with ``replicate_physics=True`` and leaves the scene cfg alone.

def _variant_spawn_paths(base_name: str, num_variants: int, num_envs: int) -> list[str | None]:
    """Prototype spawn path per variant, matching what ``make_clone_plan`` will compute.

    Variant *k* is spawned into the first environment that carries it. Under the round-robin
    assignment (environment *i* holds variant ``i % n``) that environment is *k* itself, and a
    variant past ``num_envs`` is never assigned, so it gets no path at all -- exactly what
    ``make_clone_plan`` writes for an inactive prototype (``clone_plan.py:363``).

    Args:
        base_name: Asset name under the environment prim, e.g. ``"Object"``.
        num_variants: Number of spawn variants declared by the cfg.
        num_envs: Environment count.

    Returns:
        One path (or ``None``) per variant.
    """
    return [f"/World/envs/env_{k}/{base_name}" if k < num_envs else None for k in range(num_variants)]


def install_multi_asset(num_envs: int, *, prototypes_only: bool = True) -> None:
    """Patch upstream's asset-cfg builders so each variant is spawned once, as a prototype.

    ``prototypes_only=False`` restores the older behaviour of authoring one prim per
    environment. The two settings must agree with replication: spawning only prototypes is safe
    exactly when something later replicates them, which is why callers pass
    ``prototypes_only=self.explicit_replication``.

    Args:
        num_envs: environment count; must match ``cfg.scene.num_envs``.
        prototypes_only: spawn one prim per *variant* (True) or one per *environment* (False).
    """
    from isaacsimenvs.newton import compat

    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_multi_asset_patch_installed", False):
        return

    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg

    def build_rigid_object_cfg(prim_path: str, usd_paths: list[str]) -> RigidObjectCfg:
        """Round-robin ``usd_paths`` across environments, spawning one prim per variant."""
        usd_paths = list(usd_paths)
        if not usd_paths:
            raise ValueError(f"no USD paths supplied for {prim_path}")

        # "/World/envs/env_.*/Object" -> base name "Object"
        base_name = prim_path.rsplit("/", 1)[-1]

        if prototypes_only:
            spawn = MultiUsdFileCfg(
                usd_path=usd_paths,
                spawn_paths=_variant_spawn_paths(base_name, len(usd_paths), num_envs),
                random_choice=False,
            )
        else:
            spawn = MultiUsdFileCfg(
                usd_path=[usd_paths[i % len(usd_paths)] for i in range(num_envs)],
                spawn_paths=[f"/World/envs/env_{i}/{base_name}" for i in range(num_envs)],
                random_choice=False,
            )
        return RigidObjectCfg(prim_path=prim_path, spawn=spawn)

    scene_utils.build_rigid_object_cfg = build_rigid_object_cfg

    if prototypes_only:
        # The robot has one variant and no ``spawn_paths``, so ``AssetBase.__init__`` hands the
        # raw ``env_.*`` regex to the spawner and ``sim.utils.prims.clone`` copies the prim into
        # every environment. That is authored-everywhere again: harmless under a whole-env
        # homogeneous plan, but it makes the robot exist at destinations the plan is about to
        # write to, and on the ovphysx full-stage path an existing destination is *overlaid*
        # with an internal reference rather than copied (``ovphysx_manager.py:746``) -- the
        # own-inertia/other-collider hybrid this project has already been bitten by. Pin it to
        # env_0 so it is a prototype like everything else.
        original_robot_cfg = scene_utils.build_robot_articulation_usd_cfg

        def build_robot_articulation_usd_cfg(usd_path: str, **kwargs):
            cfg = original_robot_cfg(usd_path, **kwargs)
            base_name = cfg.prim_path.rsplit("/", 1)[-1]
            cfg.spawn.spawn_path = _variant_spawn_paths(base_name, 1, num_envs)[0]
            return cfg

        scene_utils.build_robot_articulation_usd_cfg = build_robot_articulation_usd_cfg
        _install_object_scale_from_assignment(scene_utils)

    scene_utils._multi_asset_patch_installed = True


def _install_object_scale_from_assignment(scene_utils) -> None:
    """Derive the per-env object-scale tensor from the assignment, not from spawned prims.

    Upstream's ``_build_object_scale_tensor`` enumerates ``/World/envs/env_.*/Object`` and insists
    on finding exactly ``num_envs`` prims -- true when every environment's object is authored in
    USD, false once only the prototypes are. What it is really computing is which pool entry
    environment *i* holds, and that is ``i % num_object_usds`` by construction, on this stack and
    on the 2.3 reference alike (upstream reaches the same answer because ``find_matching_prims``
    returns environments in creation order, so its ``source_idx`` *is* the environment id).

    Note the pool length, not the spawned-variant count, stays the modulus. Under
    ``--single_variant`` the pool still holds every entry while only one is spawned, so this
    reports a different scale per environment for an object that is the same everywhere. That is
    a real wart, and it is the 2.3 reference's wart too: reproducing it is what keeps the control
    comparable. The guard below is sized from what was actually spawned instead.
    """
    import torch

    def _build_object_scale_tensor(env, object_scales_normalized, num_object_usds: int) -> None:
        from isaaclab.sim.utils import find_matching_prim_paths

        spawned = getattr(getattr(env.object.cfg, "spawn", None), "usd_path", None)
        expected = min(len(spawned) if isinstance(spawned, list) else 1, env.num_envs)
        found = len(find_matching_prim_paths("/World/envs/env_.*/Object"))
        if found < expected:
            raise RuntimeError(
                f"Expected at least {expected} Object prototype prims after spawning, got {found}."
            )

        env._object_scale_per_env = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        env._object_asset_index_per_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        for env_id in range(env.num_envs):
            asset_index = env_id % num_object_usds
            env._object_scale_per_env[env_id] = torch.tensor(
                object_scales_normalized[asset_index], device=env.device, dtype=torch.float32,
            )
            env._object_asset_index_per_env[env_id] = asset_index

    scene_utils._build_object_scale_tensor = _build_object_scale_tensor


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
# Asset data: warp ProxyArray -> torch
# ============================================================================
#
# Expose Isaac Lab 3.0 ``ProxyArray`` asset data as torch tensors, as 2.x did.
#
# Isaac Lab 3.0 changed ``asset.data.*`` from torch tensors to warp-backed ``ProxyArray``
# objects with a ``.torch`` accessor (the shipped Cartpole task reads
# ``self.cartpole.data.joint_pos.torch``). Upstream SimToolReal is 2.x code and indexes those
# attributes directly.
#
# Some of it happens to survive: ``body_state_w`` is ``(num_envs, num_bodies)`` of a 13-vector,
# so ``body_state_w[:, palm_id, :]`` still yields ``(N, 13)``. Others do not — ``root_quat_w``
# is ``(num_envs,)`` of a quaternion type rather than ``(num_envs, 4)``, so::
#
#     obj_rot = env.object.data.root_quat_w        # ProxyArray, shape (8,)
#     convert_quat(obj_rot, to="xyzw")
#     ValueError: Expected input quaternion shape mismatch: (8,) != (..., 4).
#
# The mixed behaviour is the hazard: the shapes that *do* survive make the boundary look
# narrower than it is, so patching individual call sites would leave silent wrong-shape reads
# elsewhere in reward, reset and termination code.
#
# This wraps each asset's ``data`` object so attribute reads return ``.torch`` when available,
# restoring 2.x semantics everywhere at once. Writes pass straight through to the wrapped object.


_ASSET_ATTRS = ("robot", "object", "table", "goal_viz", "hole")

# Exactly the asset-data attributes upstream reads (enumerated from the SimToolReal task
# source). Restricting to this allowlist keeps Isaac Lab's own internals -- which read the
# same `data` object expecting warp-backed arrays -- working unchanged.
TORCH_ATTRS = frozenset({
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "root_state_w",
    "body_state_w",
    "joint_pos",
    "joint_vel",
    "joint_pos_limits",
    "default_joint_pos",
    "default_mass",
})


# Fields whose torch view carries quaternions that upstream expects in wxyz order.
# value = None      -> the whole tensor is a quaternion, shape (..., 4)
# value = (lo, hi)  -> quaternion occupies [..., lo:hi] of a wider state vector
# Measured empirically by diffing env-0 observations against the Isaac Lab 2.3 baseline:
# `body_state_w` carries its quaternion as xyzw in 3.0, while `root_quat_w` is still wxyz.
# The conventions are NOT uniform across fields, so each is verified rather than assumed --
# converting root_quat_w as well breaks the object-rotation entries (obs idx 94-97), which
# already matched.
# Isaac Lab 3.0 stores quaternions as xyzw *and* its math helpers are xyzw (verified
# numerically -- see quat_math_patch). Upstream is 2.x code that is wxyz throughout. Both the
# data and the math must therefore be presented as wxyz; converting only one of them leaves the
# two errors cancelling in some fields and compounding in others.
# Verified field by field against the 2.3 baseline observation, because 3.0 is NOT uniform:
#   body_state_w[3:7] -> xyzw, needs converting
#   root_quat_w       -> already wxyz, must be left alone (converting it drives the
#                        object_rot block to max|diff| 1.0 against the baseline)
# The math helpers are separately xyzw for every call; see quat_math_patch.
QUAT_FIELDS: dict[str, tuple[int, int] | None] = {
    "body_state_w": (3, 7),
    # root_quat_w is xyzw in Isaac Lab 3.0 too. An earlier version of this file left it alone,
    # on the evidence that converting it "broke the object-rotation observation against the 2.3
    # baseline" -- but that comparison, and every round-trip check built on it, compared
    # *numbers*. Write and read share the convention, so a wrong one round-trips perfectly.
    #
    # Asked through the physics instead: write (0.7071, 0, 0, 0.7071) -- upstream's wxyz for 90
    # degrees about +Z -- then look at where the simulator itself puts the object's centre of
    # mass, which sits 7.55 cm off the body origin:
    #
    #     2.3 reference   com offset in world = [0, 0.0755, 0]   rotated about Z   -> wxyz
    #     3.0 port        com offset in world = [0.0755, 0, 0]   not rotated at all -> xyzw
    #
    # So the port was storing (w, x, y, z) as (x, y, z, w): a 90 degree turn about X where the
    # task meant Z. Numbers matched everywhere; the object was physically somewhere else.
    "root_quat_w": None,
    "root_state_w": (3, 7),
    "root_link_quat_w": None,
}


def _xyzw_to_wxyz(t):
    """Reorder the trailing quaternion component from (x,y,z,w) to (w,x,y,z)."""
    import torch

    return torch.cat((t[..., 3:4], t[..., 0:3]), dim=-1)


class _TorchDataProxy:
    """Attribute proxy returning ``.torch`` views of warp-backed arrays."""

    __slots__ = ("_wrapped", "_cache")

    def __init__(self, wrapped: Any):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_wrapped"), name)
        if name not in TORCH_ATTRS:
            # Everything else stays a ProxyArray. Isaac Lab's own internals read these
            # attributes expecting warp arrays (`.warp`), so converting indiscriminately
            # breaks stepping with "'Tensor' object has no attribute 'warp'".
            return value
        torch_view = getattr(value, "torch", None)
        if torch_view is None or callable(torch_view):
            return value

        # Isaac Lab 3.0 stores quaternions as xyzw. Upstream is 2.x code: its internal math
        # assumes wxyz and it converts explicitly to xyzw at the policy boundary
        # (`convert_quat(..., to="xyzw")`). Handing it xyzw therefore double-converts, which
        # silently rotates every palm/object/goal orientation the policy sees.
        if name in QUAT_FIELDS:
            span = QUAT_FIELDS[name]
            if span is None:
                return _xyzw_to_wxyz(torch_view)
            lo, hi = span
            converted = torch_view.clone()
            converted[..., lo:hi] = _xyzw_to_wxyz(torch_view[..., lo:hi])
            return converted
        return torch_view

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_wrapped"), name, value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_TorchDataProxy({object.__getattribute__(self, '_wrapped')!r})"


def wrap_env_assets(env) -> list[str]:
    """Wrap ``env.<asset>.data`` for every SimToolReal asset present.

    Returns the names of the assets wrapped.
    """
    wrapped: list[str] = []
    for name in _ASSET_ATTRS:
        asset = getattr(env, name, None)
        if asset is None:
            continue
        data = getattr(asset, "data", None)
        if data is None or isinstance(data, _TorchDataProxy):
            continue
        proxy = _TorchDataProxy(data)
        try:
            asset.data = proxy
        except AttributeError:
            # `data` is a read-only property on the asset class. Swap in a per-instance
            # subclass carrying the proxy, rather than assigning a property onto the shared
            # class -- doing the latter breaks every sibling instance that has not been
            # wrapped yet (table/goal_viz would raise on attribute lookup).
            base = type(asset)
            patched = type(
                f"{base.__name__}TorchData",
                (base,),
                {"data": property(lambda self, _proxy=proxy: _proxy)},
            )
            asset.__class__ = patched
        wrapped.append(name)
    return wrapped


# ============================================================================
# Quaternion math: presenting xyzw helpers as wxyz
# ============================================================================
#
# Give upstream wxyz quaternion math on Isaac Lab 3.0, whose helpers are xyzw.
#
# The bug
# -------
# Isaac Lab 3.0 moved every quaternion to xyzw ordering, including the math helpers in
# ``isaaclab.utils.math``. The docstrings do not state a convention, so this was verified
# numerically -- rotating ``(1,0,0)`` by 90 degrees about Z::
#
#     quat_apply(wxyz form) -> [1.0, 0.0, 0.0]     unrotated, wrong
#     quat_apply(xyzw form) -> [0.0, 1.0, 0.0]     correct
#
# Upstream is 2.x code whose internal convention is wxyz throughout (it converts to xyzw only at
# the policy boundary, via the explicit ``convert_quat(..., to="xyzw")``). Under 3.0 every call it
# makes is therefore given a mis-ordered quaternion:
#
# * ``_apply_local_offset`` -> palm centre offset rotated wrongly (palm_pos differed from the 2.3
#   baseline by up to 0.196 m, which then propagates into every palm-relative observation)
# * ``_keypoints_world`` -> object **and goal** keypoints placed wrongly, which is the quantity the
#   success test and the whole goal-reaching reward are computed from
# * goal sampling (``quat_mul`` / ``random_orientation``) -> goal orientations sampled wrongly
#
# That matches the observed symptom precisely: grasping recovered (lift 0.83) because it depends
# on positions, while goal reaching stayed at zero hits with keypoint distance plateauing near
# 0.09 against a 0.01 tolerance.
#
# The fix
# -------
# Wrap the helpers so they present wxyz semantics to upstream: inputs are reordered to xyzw before
# the call, and quaternion-valued results are reordered back to wxyz. ``convert_quat`` is left
# alone -- it is explicit about ordering and upstream uses it deliberately at the policy boundary.
#
# Patching is applied to the *upstream modules' own globals*, because they bind these names at
# import time (``from isaaclab.utils.math import quat_apply``), so replacing the attribute on
# ``isaaclab.utils.math`` afterwards would have no effect.

_PATCH_FLAG = "_quat_math_patch_installed"

# Upstream modules that import quaternion helpers, and the names they bind.
_TARGET_MODULES = (
    "utils.obs_utils",
    "utils.reset_utils",
    "utils.goal_sampling",
    "utils.scene_utils",
    "utils.reward_utils",
    "utils.termination_utils",
)

# name -> (quaternion argument positions, whether the result is a quaternion)
_WRAPPED: dict[str, tuple[tuple[int, ...], bool]] = {
    "quat_apply": ((0,), False),
    "quat_apply_inverse": ((0,), False),
    "quat_mul": ((0, 1), True),
    "quat_conjugate": ((0,), True),
    "quat_inv": ((0,), True),
    "quat_from_angle_axis": ((), True),
    "random_orientation": ((), True),
    "quat_rotate": ((0,), False),
    "quat_rotate_inverse": ((0,), False),
}


def _to_xyzw(t):
    return t[..., [1, 2, 3, 0]]


def _to_wxyz(t):
    return t[..., [3, 0, 1, 2]]


def _wrap(fn, quat_arg_positions: tuple[int, ...], returns_quat: bool):
    def wrapped(*args, **kwargs):
        args = list(args)
        for pos in quat_arg_positions:
            if pos < len(args) and hasattr(args[pos], "shape") and args[pos].shape[-1] == 4:
                args[pos] = _to_xyzw(args[pos])
        result = fn(*args, **kwargs)
        if returns_quat and hasattr(result, "shape") and result.shape[-1] == 4:
            return _to_wxyz(result)
        return result

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = (
        "wxyz-facing wrapper installed by isaacsimenvs.newton.patches.\n\n"
        + (getattr(fn, "__doc__", "") or "")
    )
    return wrapped


def install_quat_math() -> list[str]:
    """Rebind quaternion helpers inside upstream modules to wxyz semantics."""
    from isaacsimenvs.newton import compat

    patched: list[str] = []
    for module_name in _TARGET_MODULES:
        try:
            module = compat.play_module(module_name)
        except Exception:
            continue
        if getattr(module, _PATCH_FLAG, False):
            continue

        for fn_name, (positions, returns_quat) in _WRAPPED.items():
            original = getattr(module, fn_name, None)
            if original is None or not callable(original):
                continue
            setattr(module, fn_name, _wrap(original, positions, returns_quat))
            patched.append(f"{module_name}.{fn_name}")

        setattr(module, _PATCH_FLAG, True)

    return patched


# ============================================================================
# Quaternion conventions on config-authored poses
# ============================================================================
#
# Convert asset initial-state rotations from Isaac Lab 2.x wxyz to 3.0 xyzw.
#
# The bug this fixes
# ------------------
# Isaac Lab 3.0 changed every quaternion to **xyzw** ordering.
# ``AssetBaseCfg.InitialStateCfg.rot`` is now documented as *"Quaternion rotation (x, y, z, w)"*
# with default ``(0, 0, 0, 1)``. Upstream, written against 2.x, authors the robot's base
# orientation as::
#
#     rot=(1.0, 0.0, 0.0, 0.0)     # wxyz identity
#
# Under 3.0 that same tuple reads as ``x=1, y=0, z=0, w=0`` — a **180 degree rotation about X**.
# The robot is therefore spawned upside-down.
#
# Why it took so long to find: the failure is silent and looks like everything else. Joint angles
# match the training default exactly, the root position is right, ``root_quat_w`` reports
# ``(1,0,0,0)``, and the solver, gains, targets and coordinate mappings all check out. The arm
# simply reaches the wrong way, so the fingertips land ~2 m from the object, the task's
# ``hand_far`` termination (``fingertip distance > 1.5``) fires on **every** step, and the
# environment resets before anything can accumulate. Observations then look frozen and the policy
# scores zero — which reads like a broken state sync rather than a bad spawn pose.
#
# Measured against PhysX at the same joint pose, ``iiwa14_link_7`` relative to the root:
#
#     PhysX  : (0, -0.589, +0.750)
#     Newton : (0, +0.578, -0.743)     exactly negated in y and z -> 180 deg about X
#
# Scope
# -----
# Only the robot articulation authors a non-default ``rot`` in this task; the rigid objects use
# the config default, which is already correct for whichever convention is active. The patch
# converts any explicitly authored rotation rather than hard-coding identity, so a future
# non-identity spawn orientation is handled too.
#
# Applied in the Newton process only. The PhysX venv runs Isaac Lab 2.3, where the original wxyz
# tuple is correct and must be left alone.

def wxyz_to_xyzw(rot: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Reorder a ``(w, x, y, z)`` quaternion to ``(x, y, z, w)``."""
    w, x, y, z = rot
    return (x, y, z, w)


def install_quat_convention() -> None:
    """Reinterpret upstream's wxyz initial rotations as xyzw for Isaac Lab 3.0."""
    from isaacsimenvs.newton import compat

    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_quat_convention_patch_installed", False):
        return

    original = scene_utils.build_robot_articulation_usd_cfg

    def build_with_xyzw_rot(*args, **kwargs):
        cfg = original(*args, **kwargs)
        init = getattr(cfg, "init_state", None)
        rot = getattr(init, "rot", None)
        if init is not None and rot is not None:
            converted = wxyz_to_xyzw(tuple(rot))
            if tuple(rot) != converted:
                init.rot = converted
                print(
                    f"[quat_convention] robot init rot {tuple(rot)} (wxyz) -> {converted} (xyzw)",
                    flush=True,
                )
        return cfg

    scene_utils.build_robot_articulation_usd_cfg = build_with_xyzw_rot
    scene_utils._quat_convention_patch_installed = True


# ============================================================================
# Fingertip body ordering
# ============================================================================
#
# Resolve fingertip bodies in canonical order, not backend body order.
#
# The bug
# -------
# ``reset_utils`` caches the fingertip bodies with a regex::
#
#     FINGERTIP_BODY_REGEX = "left_(index|middle|ring|thumb|pinky)_DP"
#     env._fingertip_body_ids = env.robot.find_bodies(FINGERTIP_BODY_REGEX)[0]
#
# ``find_bodies`` defaults to ``preserve_order=False``, so the result comes back sorted by **body
# index**, which depends on how the backend ordered the articulation's bodies. Under Isaac Lab 2.3
# that happened to coincide with the canonical order; under Newton it does not:
#
#     FINGERTIP_LINK_NAMES : [index, middle, ring, thumb, pinky]     canonical
#     PhysX 2.3 resolves to: [index, middle, ring, thumb, pinky]     ids 24,25,26,28,29
#     Newton resolves to   : [thumb, index, middle, ring, pinky]     ids 12,16,20,24,29
#
# The thumb lands in slot 0 instead of slot 3, so the 15-dim ``fingertip_pos_rel_palm`` block and
# the 5-dim ``closest_fingertip_dist`` block are permuted: the policy reads the thumb's position
# where it expects the index finger's.
#
# This is the same class of failure as the two quaternion bugs — an ordering convention that held
# implicitly under 2.3 and silently changed. It degrades precise manipulation while leaving gross
# grasping intact, which matches the observed symptom (lift rate recovered to 0.87, but goal
# keypoint distance plateaued at ~0.09 against a 0.01 tolerance).
#
# The fix
# -------
# Re-resolve the fingertips by explicit name list with ``preserve_order=True`` so the cached ids
# follow ``FINGERTIP_LINK_NAMES`` on any backend.

def install_fingertip_order() -> None:
    """Reorder ``env._fingertip_body_ids`` to canonical order after they are cached."""
    from isaacsimenvs.newton import compat

    reset_utils = compat.play_module("reset_utils")
    if getattr(reset_utils, "_fingertip_order_patch_installed", False):
        return

    scene_utils = compat.play_module("scene_utils")
    canonical = list(scene_utils.FINGERTIP_LINK_NAMES)

    original = reset_utils.allocate_state_buffers

    def allocate_state_buffers_ordered(env, *args, **kwargs):
        result = original(env, *args, **kwargs)

        body_names = list(env.robot.body_names)
        ordered = [body_names.index(name) for name in canonical]

        current = env._fingertip_body_ids
        current = current.tolist() if hasattr(current, "tolist") else list(current)
        if current != ordered:
            env._fingertip_body_ids = ordered
            print(
                "[fingertip_order] reordered fingertips to canonical order: "
                f"{[body_names[i] for i in current]} -> {canonical}",
                flush=True,
            )
        return result

    reset_utils.allocate_state_buffers = allocate_state_buffers_ordered
    reset_utils._fingertip_order_patch_installed = True


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


def install_cloning(
    *,
    push_collision_api: bool = False,
    write_drive_gains: bool = False,
    filter_inter_env_collisions: bool = True,
    request_full_stage: bool = False,
) -> None:
    """Wrap ``scene_utils.setup_scene`` so environments are replicated after asset creation.

    The two USD rewrites are opt-in per backend rather than decided here. They exist to work
    around Newton's importer and MuJoCo solver, and applying them under PhysX is not neutral:
    pushing ``CollisionAPI`` down doubles the robot's collision prims (68 against 34, measured),
    and the duplicates fall outside the authored ``physics:filteredPairs`` set, breaking the
    adjacent-link self-collision filtering the trained policy depends on. Authoring real gains
    into the drives likewise contradicts upstream's deliberate zero-gain drives, which PhysX
    honours precisely so the ``ImplicitActuatorCfg`` values are the ones that land.

    ``request_full_stage`` is the ovphysx-only half of per-environment variants. The wheel is fed
    a serialized copy of the stage, and by default ``_strip_nonzero_environments``
    (``ovphysx_manager.py:757``) removes every ``env_<i!=0>`` from it before handing it over,
    because re-creating those environments with ``physx.clone()`` is far cheaper than ingesting
    them as independent USD bodies. That optimisation is exactly what destroys the variants: the
    prototypes for variants 1..n-1 live in ``env_1``..``env_{n-1}``. ``require_full_stage()`` is
    the documented opt-out -- "features that need distinct authored physics in every environment
    request the full stage" -- and it is requested only when the plan is genuinely heterogeneous,
    so a single-variant run keeps the fast path untouched.

    Args:
        push_collision_api: apply ``CollisionAPI`` to descendant mesh prims (Newton only).
        write_drive_gains: write canonical PD gains into the USD drives (Newton only).
        filter_inter_env_collisions: call ``scene.filter_collisions`` after replication. PhysX
            needs it; Newton isolates environments through its own world partitioning.
        request_full_stage: ask the ovphysx wheel to ingest every authored environment when the
            plan has more than one prototype per asset.
    """
    from isaacsimenvs.newton import compat

    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_clone_patch_installed", False):
        return

    original_setup_scene = scene_utils.setup_scene

    def setup_scene_with_replication(env) -> None:
        original_setup_scene(env)

        if push_collision_api:
            # Isaac Sim's URDF importer applies CollisionAPI to Xform wrappers rather than to
            # the mesh prims themselves. Newton skips those, silently dropping every robot
            # collision shape. Push the API down before the model is finalized.
            fixed, marked = push_collision_api_to_meshes(env.scene.stage)
            if fixed:
                print(
                    f"[cloning] pushed CollisionAPI onto {marked} mesh prims "
                    f"under {fixed} collision Xforms",
                    flush=True,
                )

            # After the push, not before: the attribute has to land on the *mesh* prims, which
            # only carry CollisionAPI once `push_collision_api_to_meshes` has run.
            n_convex = author_convex_hull_approximation(env.scene.stage)
            if n_convex:
                print(f"[cloning] marked {n_convex} object colliders convexHull", flush=True)

        if write_drive_gains:
            # Upstream authors zero-gain USD drives and injects gains at runtime, which PhysX
            # honours but Newton cannot: SolverMuJoCo bakes actuation from USD at build time.
            # Write the canonical gains before the model is finalized.
            written, missing = author_drive_gains(env.scene.stage, scene_utils)
            if written:
                print(
                    f"[cloning] authored PD gains on {written} joint drives"
                    + (f" ({missing} lacked drive attrs)" if missing else ""),
                    flush=True,
                )

        from isaaclab import cloner

        num_envs = int(env.scene.num_envs)
        device = str(env.device)

        # `make_clone_plan` reads the cfgs, not the stage, so it needs the ones the asset
        # constructors just registered -- in construction order, which is the order its group
        # rows and therefore the `valid_set` columns follow.
        queued = list(cloner.REPLICATION_QUEUE)
        groups = _plan_groups(queued)
        variant_counts = [cloner.num_spawn_variants(cfg.spawn) for cfg in groups]
        valid_set = _round_robin_valid_set(variant_counts, num_envs, device)

        # Snapshot where each prototype was actually spawned, then let the planner rewrite the
        # same fields and compare. See `_assert_spawn_paths_unchanged`.
        before = [
            ("spawn_paths", list(cfg.spawn.spawn_paths))
            if getattr(cfg.spawn, "spawn_paths", None) is not None
            else ("spawn_path", cfg.spawn.spawn_path)
            for cfg in groups
        ]

        plan = cloner.make_clone_plan(
            queued,
            num_envs,
            env.scene.cfg.env_spacing,
            device,
            clone_strategy=cloner.sequential,
            valid_set=valid_set,
        )
        _assert_spawn_paths_unchanged(groups, before)

        heterogeneous = valid_set is not None
        if heterogeneous and request_full_stage:
            # Must precede the first `sim.reset()`, which is what triggers the wheel's stage
            # serialization, and must follow `SimulationContext.initialize()`, which clears the
            # flag. Between the two is exactly here.
            from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

            OvPhysxManager.require_full_stage()

        cloner.replicate(plan, stage=env.scene.stage, replicate_physics=True)

        # PhysX replication needs explicit inter-env collision filtering; Newton isolates
        # environments via worlds, so the call is skipped there (mirrors the Cartpole task).
        if filter_inter_env_collisions:
            env.scene.filter_collisions(global_prim_paths=[])

        print(
            f"[clone_patch] replicated {num_envs} environments from "
            f"{len(plan.sources)} prototype row(s); variant counts {variant_counts}",
            flush=True,
        )

    scene_utils.setup_scene = setup_scene_with_replication
    scene_utils._clone_patch_installed = True


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
# canonical tables the PhysX path uses (``scene_utils.ARM_JOINT_STIFFNESS`` /
# ``HAND_JOINT_STIFFNESS`` and their damping counterparts). Both backends therefore run the same
# numbers; only the delivery mechanism differs, which is precisely what transfer invariant 3 in
# upstream's own test suite pins down.
#
# Applied to the live stage in the Newton process only, so the content-addressed USD cache stays
# byte-identical and the PhysX baseline is untouched.

_STIFFNESS_ATTR = "drive:angular:physics:stiffness"
_DAMPING_ATTR = "drive:angular:physics:damping"


def author_drive_gains(stage, scene_utils, root_path: str = "/World/envs") -> tuple[int, int]:
    """Write canonical PD gains onto every robot joint drive under ``root_path``.

    Returns ``(joints_written, joints_missing_gains)``.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stiffness = dict(scene_utils.ARM_JOINT_STIFFNESS)
    stiffness.update(scene_utils.HAND_JOINT_STIFFNESS)
    damping = dict(scene_utils.ARM_JOINT_DAMPING)
    damping.update(scene_utils.HAND_JOINT_DAMPING)

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

        for attr_name, table in ((_STIFFNESS_ATTR, stiffness), (_DAMPING_ATTR, damping)):
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
# Per-body gravity (Newton)
# ============================================================================
#
# Reproduce upstream's per-body gravity setup, which Newton cannot express natively.
#
# The problem
# -----------
# Upstream disables gravity on the robot and keeps it on the manipulated object. It does this
# through PhysX's per-body attribute (``scene_utils`` bakes ``physxRigidBody:disableGravity`` and
# upstream's transfer invariant 7 pins *"robot gravity disabled"*). The policy was trained in that
# regime.
#
# Newton has **no per-body gravity control** — no ``BodyFlags`` entry, no ``add_body(gravity=...)``
# parameter. Global gravity applies to every dynamic body, so the PhysX attribute is silently
# ignored and the arm and hand sag.
#
# That sag is not cosmetic. The hand's PD gains are tiny (stiffness 0.9-13.2, damping
# 0.028-0.41), so gravity dominates the fingers and produces a steady-state tracking offset:
#
#     holding a fixed +0.20 rad target for 120 steps
#       PhysX 2.3   hand mean |err| = 0.0517,  moved 0.160 of 0.200
#       Newton      hand mean |err| = 0.1312,  moved 0.115 of 0.200
#
# The offset survived every solver knob tried (integrator, substeps, solver iterations,
# self-collision, actuator mode, placeholder inertia), which is the signature of a constant
# external load rather than a solver-response difference.
#
# The fix
# -------
# Zero global gravity so the robot floats as it does under PhysX, then apply gravity explicitly
# to the bodies that should feel it — the free manipulated object. The table and goal marker are
# kinematic and unaffected either way.
#
# This is a faithful reconstruction of the trained regime rather than a tuning choice.
#
# **Newton only.** Kit-less PhysX honours the authored per-body ``disableGravity``
# -------------------------------------------------------------------------------
# An earlier note here claimed both 3.0 backends ignore the attribute. That was **wrong**, and it
# was measured while armature was being dropped and the robot's collision prims were duplicated.
# Re-measured with those fixed, under OVPhysX with native gravity and this reconstruction off:
#
#     robot max joint drift over 7 zero-action steps : identical to the patched run (0.0273 ...
#                                                       0.3587) — the arm does not sag
#     object z(t)                                    : matches the 2.3 baseline step for step
#     object quaternion                              : stays exactly (1, 0, 0, 0)
#
# So on OVPhysX the reconstruction is not merely unnecessary, it is harmful, and this module is
# no longer installed there.
#
# The parasitic torque it was injecting
# -------------------------------------
# The manipulated tool's centre of mass is **7.55 cm** from its body origin (a hammer: the head
# is at one end). Isaac Lab's wrench composer computes a torque *about the CoM*, but the backend
# then applies the resulting wrench at the **link frame**, so a force meant to act at the CoM
# leaks a moment of ``(link_origin - com) x F``. With the same starting pose the object rotated
# about +Y from the very first tick instead of falling straight::
#
#     2.3      quat (1, 0, 0, 0) throughout the fall
#     port     quat w=0.9997 -> 0.9963 -> 0.9838 -> 0.9751,  y = 0.026 -> 0.222
#
# which tipped the tool before the hand ever reached it. The Newton path therefore now adds the
# explicit compensating torque; it is kept honest by the OVPhysX control above, where native
# gravity gives the answer this reconstruction has to reproduce.

GRAVITY_Z = -9.81


def disable_global_gravity(cfg) -> None:
    """Zero simulation gravity; per-body gravity is reapplied at runtime."""
    cfg.sim.gravity = (0.0, 0.0, 0.0)


def install_object_gravity() -> None:
    """Apply object-only gravity every physics step, mirroring PhysX's per-body setup."""
    from isaacsimenvs.newton import compat

    action_utils = compat.play_module("action_utils")
    if getattr(action_utils, "_gravity_patch_installed", False):
        return

    original = action_utils.apply_wrench_dr

    def apply_wrench_dr_with_object_gravity(env) -> None:
        original(env)

        obj = getattr(env, "object", None)
        if obj is None:
            return

        import torch

        masses = obj.data.default_mass
        masses = masses.torch if hasattr(masses, "torch") else masses
        if masses.ndim > 1:
            masses = masses.sum(dim=-1)

        forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
        forces[:, 0, 2] = masses.to(env.device).reshape(-1) * GRAVITY_Z
        torques = torch.zeros_like(forces)

        # No compensating moment. An earlier version added `(com - origin) x F` here, on the
        # theory that the backend applies this wrench at the link frame -- but both ends of the
        # path document the force as acting at the centre of mass:
        #
        #   isaaclab/utils/wrench_composer.py:246-249  "If None, forces are assumed to act at
        #       the body's CoM, independent of the `is_global` flag"
        #   newton/_src/sim/state.py:148-153           body_f is "applied at the body's center
        #       of mass (COM)"
        #
        # so the extra term was a spurious constant torque of |r x mg| ~ 0.16 N.m on the grasped
        # tool, every step, on the one backend whose open failure mode is the tool rotating in
        # the hand. The same function said so fifteen lines further down while contradicting
        # itself here. `positions` stays None, which is what puts the force at the CoM.

        # `is_global=True` is essential and not the default. Isaac Lab's wrench API interprets
        # a force in the **body frame** unless told otherwise, so a constant (0, 0, -mg) vector
        # rotates with the object: the moment the tool is tilted or reoriented, its "gravity"
        # points sideways. That is invisible at reset (the object starts near upright and the
        # error is a cosine) and grows precisely during goal-directed reorientation, which is
        # the failure this port shows. Gravity is a world-frame force; say so.
        #
        # `positions` is left None, which the composer documents as acting at the centre of
        # mass — correct for gravity, and not the same as the body origin for an offset-COM
        # tool like a hammer.
        setter = getattr(obj, "set_external_force_and_torque", None)
        if setter is not None:
            setter(forces, torques, is_global=True)

    action_utils.apply_wrench_dr = apply_wrench_dr_with_object_gravity
    action_utils._gravity_patch_installed = True


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
# Shape friction (Newton)
# ============================================================================
#
# Newton replacement for upstream's PhysX-view-based friction assignment.
#
# ``scene_utils.apply_physx_material_properties`` writes per-shape friction through PhysX view
# APIs that do not exist on the Newton backend::
#
#     env.robot.root_physx_view.get_material_properties() / set_material_properties()
#     env.robot._physics_sim_view.create_rigid_body_view(link_path)
#     view.shared_metatype.link_names, view.link_paths, view.max_shapes
#
# Newton exposes the same physical quantity directly on the model: ``Model.shape_material_mu``
# and ``Model.shape_material_restitution``, indexed by shape, with ``Model.shape_label`` naming
# each shape. This rewrites the assignment against those arrays.
#
# This matters for the comparison rather than being incidental plumbing: fingertip friction is
# one of the parameters a dexterous grasping policy is most sensitive to. Leaving Newton on
# default friction would guarantee an M4 gap that had nothing to do with the contact solver.
#
# Friction model difference (documented, not hidden)
# --------------------------------------------------
# PhysX carries **separate static and dynamic** friction coefficients; upstream sets both to the
# same value (``[friction, friction, 0.0]`` = static, dynamic, restitution). Newton/MuJoCo carry
# a **single** ``mu``. Because upstream sets both PhysX coefficients equal, the collapse to one
# ``mu`` is exact here — no information is lost. That would not hold for a config that set them
# differently.
#
# Unsupported: per-environment bucketed friction randomization
# ------------------------------------------------------------
# Upstream optionally quantizes fingertip/object friction into ``friction_n_buckets`` per-env
# values. Newton's ``shape_material_mu`` is per shape and shapes are already per environment, so
# this is implementable — but it is not implemented here, and the patch **raises** rather than
# silently ignoring it. The M4 protocol disables domain randomization, so the ranges are
# ``(1.0, 1.0)`` and the path is inert; a future randomized run must implement it rather than
# discover the omission from a quiet result.

def install_newton_materials() -> None:
    """Replace ``scene_utils.apply_physx_material_properties`` with a Newton implementation."""
    from isaacsimenvs.newton import compat

    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_material_patch_installed", False):
        return

    fingertip_link_names = set(scene_utils.FINGERTIP_LINK_NAMES)

    def apply_newton_material_properties(env) -> None:
        import warp as wp

        assets_cfg = env.cfg.assets
        if not bool(getattr(assets_cfg, "apply_material_properties", True)):
            return

        dr = env.cfg.domain_randomization
        ft_range = tuple(float(v) for v in dr.fingertip_friction_scale_range)
        obj_range = tuple(float(v) for v in dr.object_friction_scale_range)
        if ft_range != (1.0, 1.0) or obj_range != (1.0, 1.0):
            raise NotImplementedError(
                "Per-env bucketed friction randomization is not implemented for the Newton "
                f"backend (fingertip_friction_scale_range={ft_range}, "
                f"object_friction_scale_range={obj_range}). Disable friction DR, or implement "
                "per-shape randomization against Model.shape_material_mu."
            )

        robot_friction = float(assets_cfg.robot_friction)
        fingertip_friction = float(assets_cfg.finger_tip_friction)

        # `sim.physics_manager` is the manager *class*, not an instance, so the `model`
        # property is not usable here; the class-level accessor is get_model().
        model = env.sim.physics_manager.get_model()

        # Newton names shapes generically ("shape_0", "shape_1", ...), so fingertips cannot be
        # identified from shape_label. Map each shape to its owning body via shape_body and
        # match against body_label, which does carry the URDF link names (left_index_DP, ...).
        body_labels = [label.rsplit("/", 1)[-1] for label in model.body_label]
        shape_body = model.shape_body.numpy()

        mu = model.shape_material_mu.numpy().copy()
        restitution = model.shape_material_restitution.numpy().copy()

        # Every shape in the scene (robot links, table, object, goal viz) takes the base
        # friction; fingertip shapes are then overridden. Matches upstream, which fills the
        # whole material buffer with `default` before writing the fingertip rows.
        mu[:] = robot_friction
        restitution[:] = 0.0

        n_fingertip = 0
        for shape_idx in range(len(mu)):
            body_idx = int(shape_body[shape_idx])
            if body_idx < 0:
                continue  # static/world shape (e.g. ground plane)
            body_name = body_labels[body_idx]
            if any(name in body_name for name in fingertip_link_names):
                mu[shape_idx] = fingertip_friction
                n_fingertip += 1

        if n_fingertip == 0:
            raise RuntimeError(
                "No fingertip shapes matched FINGERTIP_LINK_NAMES in Model.shape_label; "
                "friction would silently stay at the robot default. "
                f"Expected body names containing {sorted(fingertip_link_names)}."
            )

        model.shape_material_mu.assign(wp.array(mu, dtype=model.shape_material_mu.dtype))
        model.shape_material_restitution.assign(
            wp.array(restitution, dtype=model.shape_material_restitution.dtype)
        )

        print(
            f"[material_patch] friction set: {len(mu)} shapes at mu={robot_friction}, "
            f"{n_fingertip} fingertip shapes at mu={fingertip_friction}",
            flush=True,
        )

    scene_utils.apply_physx_material_properties = apply_newton_material_properties
    scene_utils._material_patch_installed = True


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
# The same reasoning applies to :mod:`isaacsimenvs.newton.patches`, which writes
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


def install_newton_condim(condim: int = 6) -> bool:
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

    Returns:
        Whether the hook was installed.
    """
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except Exception:
        return False
    if getattr(NewtonManager, "_simtoolreal_condim_patch", False):
        return False

    from isaacsimenvs.newton import compat

    scene_utils = compat.play_module("scene_utils")
    grasping = tuple(scene_utils.FINGERTIP_LINK_NAMES)
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


def _newton_solver(env):
    manager = env.sim.physics_manager
    for getter in ("get_solver",):
        fn = getattr(manager, getter, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return getattr(manager, "solver", None) or getattr(manager, "_solver", None)



# ============================================================================
# Pose writes: upstream speaks wxyz, Isaac Lab 3.0 stores xyzw
# ============================================================================
#
# The mirror of the read-side conversion. Upstream builds poses as ``(x, y, z, w, qx, qy, qz)``
# -- position then a **wxyz** quaternion -- and hands them to ``write_root_pose_to_sim``. Isaac
# Lab 3.0 expects xyzw, so the components land in the wrong slots and the asset is placed at a
# different orientation than the task asked for. Reading it back returns the same numbers, which
# is why this survived every round-trip check; the physics test in ``QUAT_FIELDS`` is what
# exposes it.
#
# This matters most for the goal marker. The sampled goal is *written* through this path and
# *read* back through the data proxy, so with only one side converted the policy would chase a
# goal at one attitude while the metric scored it against another.


def install_pose_write_conversion(env) -> list[str]:
    """Convert wxyz -> xyzw on every root-pose write. Returns the assets wrapped."""
    import torch

    def to_xyzw(quat):
        return torch.cat((quat[..., 1:4], quat[..., 0:1]), dim=-1)

    wrapped: list[str] = []
    for name in ("robot", "object", "table", "goal_viz", "hole"):
        asset = getattr(env, name, None)
        if asset is None or getattr(asset, "_simtoolreal_write_quat", False):
            continue

        for method, span in (("write_root_pose_to_sim", (3, 7)),
                             ("write_root_state_to_sim", (3, 7)),
                             ("write_root_link_pose_to_sim", (3, 7))):
            original = getattr(asset, method, None)
            if original is None:
                continue

            def converted(value, *args, _orig=original, _span=span, **kwargs):
                lo, hi = _span
                value = value.clone()
                value[..., lo:hi] = to_xyzw(value[..., lo:hi])
                return _orig(value, *args, **kwargs)

            setattr(asset, method, converted)
        asset._simtoolreal_write_quat = True
        wrapped.append(name)
    return wrapped


# ============================================================================
# Meshified collision geometry (opt-in)
# ============================================================================
#
# Represent each procedurally generated tool's *collision* geometry as convex triangle meshes
# instead of the native primitives upstream authors.
#
# Why it exists
# -------------
# A tool is exactly two convex shapes on one link: a handle and a head. Upstream authors each as
# ``<box>`` (3-tuple scale) or ``<cylinder>`` (2-tuple scale) --
# ``isaacsimenvs/tasks/simtoolreal/utils/generate_objects.py:166-252`` -- and converts URDF -> USD
# with ``replace_cylinders_with_capsules=True`` (``scene_utils.py:1703``), so in the simulator the
# collision shapes are of type BOX and/or CAPSULE.
#
# That is what blocks per-environment asset variants under Newton. ``SolverMuJoCo`` requires every
# world to declare the same shape *types* in the same order and raises
#
#     ValueError: SolverMuJoCo requires homogeneous worlds. Shape types mismatch at position 36:
#     world 0 has type 7, but other worlds have types [7, 4, 4].
#
# (``newton/_src/solvers/mujoco/solver_mujoco.py:9153``; GeoType 7 = BOX, 4 = CAPSULE.) Per-world
# mesh *data* is fine on the pipeline this port uses (``use_mujoco_contacts=False``) and the shape
# *count* is already uniform at two per tool, so making every collision shape a MESH makes the
# type vector uniform and removes the only remaining obstacle.
#
# It is off by default and it is a physics change, not a refactor: a convex hull approximating a
# capsule has faceted ends, which is the contact behaviour capsules are chosen to avoid. Whether
# that costs task performance is a measurement, not an argument -- hence ``--meshify`` as an A/B
# arm rather than a new default.
#
# What is and is not changed
# --------------------------
# * Only ``<collision>`` geometry is rewritten. ``<visual>`` keeps its primitives, so renders and
#   the goal marker are untouched.
# * The collision ``<origin>`` (including the -pi/2 rotations that put a cylinder axis on +x/+y)
#   is left exactly as authored, and each mesh is written in the primitive's own local frame. The
#   collider therefore occupies the same place, to the tessellation error.
# * ``<inertial>`` is not touched. Upstream authors an explicit ``<mass>``, ``<inertia>`` and
#   centre-of-mass ``<origin>`` (``generate_objects.py:249-253``), rather than a ``<density>``, so
#   mass and inertia are *inputs* to the importer and cannot be recomputed from the new geometry.
#   This is verified against the live sim, not assumed -- see the probe table in the report.
#
# A cylinder is meshed as the **capsule** it becomes downstream, not as a cylinder: radius r and
# cylindrical section of length equal to the URDF ``length``, with hemispherical caps, which is
# what the importer's ``replace_cylinders_with_capsules`` produces (read back from a cached USD:
# ``Capsule axis=Z height=<cylinder length> radius=<cylinder radius>``).
#
# Tessellation
# ------------
# ``_CAPSULE_SEGMENTS = 16`` azimuthal sectors and ``_CAPSULE_CAP_RINGS = 3`` latitude rings per
# hemispherical cap. That is a uniform 22.5 deg of angular resolution in *both* directions
# (360/16 around the axis; 90/(3+1) from equator to pole), so the maximum surface deviation is
# isotropic at ``r * (1 - cos(11.25 deg)) = 0.0192 r``. The largest handle radius in the hammer
# pool is 0.0148 m, giving 0.28 mm -- under 3% of the 0.01 m success tolerance. It comes to 130
# vertices / 256 triangles, comfortably inside PhysX's 255-vertex hard limit for a convex hull.
#
# Caveat, stated because it is not controlled here: PhysX's default
# ``physxConvexHullCollision:hullVertexLimit`` is 64, so the PhysX stacks may cook a reduced hull
# from these 130 vertices. Both PhysX backends do so identically, and MuJoCo takes its own hull of
# the same mesh; this is not authored either way.
#
# Boxes are meshed exactly: 8 vertices, 12 triangles, zero geometric error.
#
# Cache keying
# ------------
# The URDF -> USD cache is content-addressed on URDF *text* (:func:`usd_cache.cache_key`), which changes
# here because the ``<geometry>`` element changes, so meshified and primitive assets land in
# separate cache entries automatically. The text alone would not cover the mesh *files* it names
# (:mod:`upstream` flags exactly this caveat), so each ``.obj`` is written under a filename
# carrying a hash of its own contents. The URDF text therefore transitively addresses the meshes.

#: Azimuthal sectors around the capsule axis.
_CAPSULE_SEGMENTS = 16

#: Latitude rings per hemispherical cap, excluding the equator and the pole.
_CAPSULE_CAP_RINGS = 3


def _box_mesh(sx: float, sy: float, sz: float):
    """Exact triangle mesh of a URDF ``<box size="sx sy sz">``, centred on the origin."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1),  # -z
        (4, 5, 6), (4, 6, 7),  # +z
        (0, 1, 5), (0, 5, 4),  # -y
        (1, 2, 6), (1, 6, 5),  # +x
        (2, 3, 7), (2, 7, 6),  # +y
        (3, 0, 4), (3, 4, 7),  # -x
    ]
    return verts, faces


def _capsule_mesh(radius: float, height: float, segments: int, cap_rings: int):
    """Triangle mesh of a capsule on the +z axis: cylinder of ``height`` plus hemispherical caps.

    Matches the shape ``replace_cylinders_with_capsules=True`` produces from
    ``<cylinder length="height" radius="radius"/>``: total extent along z is ``height + 2*radius``.
    """
    half = height / 2.0

    # Rings of ``segments`` vertices each, ordered bottom to top. The two equator rings are the
    # ends of the cylindrical section; the cap rings sit on the hemispheres.
    rings: list[tuple[float, float]] = []
    for k in range(cap_rings, 0, -1):
        theta = k * (math.pi / 2.0) / (cap_rings + 1)
        rings.append((-half - radius * math.sin(theta), radius * math.cos(theta)))
    rings.append((-half, radius))
    rings.append((half, radius))
    for k in range(1, cap_rings + 1):
        theta = k * (math.pi / 2.0) / (cap_rings + 1)
        rings.append((half + radius * math.sin(theta), radius * math.cos(theta)))

    verts: list[tuple[float, float, float]] = [(0.0, 0.0, -half - radius)]
    for z, r in rings:
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
    verts.append((0.0, 0.0, half + radius))

    top_pole = len(verts) - 1

    def ring_index(ring: int, j: int) -> int:
        return 1 + ring * segments + (j % segments)

    faces: list[tuple[int, int, int]] = []
    for j in range(segments):  # bottom fan, outward normals point down and out
        faces.append((0, ring_index(0, j + 1), ring_index(0, j)))
    for ring in range(len(rings) - 1):  # bands
        for j in range(segments):
            a, b = ring_index(ring, j), ring_index(ring, j + 1)
            c, d = ring_index(ring + 1, j + 1), ring_index(ring + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(segments):  # top fan
        faces.append((ring_index(len(rings) - 1, j), ring_index(len(rings) - 1, j + 1), top_pole))
    return verts, faces


def _obj_text(verts, faces) -> str:
    """Wavefront OBJ for a vertex/face list. ``%.17g`` so the text round-trips the floats."""
    lines = [f"# generated by simtoolreal_newton.patches (meshify), {len(verts)} verts "
             f"{len(faces)} tris"]
    lines += ["v %.17g %.17g %.17g" % v for v in verts]
    lines += ["f %d %d %d" % (a + 1, b + 1, c + 1) for a, b, c in faces]
    return "\n".join(lines) + "\n"


def _write_collision_mesh(out_dir: Path, kind: str, verts, faces) -> str:
    """Write the mesh under a content-addressed name and return that name.

    The hash is of the OBJ text itself, so the filename recorded in the URDF changes whenever the
    geometry does. That is what lets :func:`usd_cache.cache_key` -- which hashes URDF text only -- stay
    a sound cache key for assets whose collision geometry lives in separate files.
    """
    text = _obj_text(verts, faces)
    name = f"col_{kind}_{hashlib.sha256(text.encode()).hexdigest()[:16]}.obj"
    path = out_dir / name
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)
    return name


def meshify_urdf(urdf_path, *, segments: int = _CAPSULE_SEGMENTS,
                 cap_rings: int = _CAPSULE_CAP_RINGS) -> int:
    """Rewrite one URDF's ``<collision>`` primitives as ``<mesh>`` references, in place.

    Visual geometry and the ``<inertial>`` block are left untouched. Returns the number of
    collision geometries converted.
    """
    import xml.etree.ElementTree as ET

    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    converted = 0
    for collision in root.findall(".//collision"):
        geometry = collision.find("geometry")
        if geometry is None or geometry.find("mesh") is not None:
            continue

        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        if box is not None:
            sx, sy, sz = (float(v) for v in str(box.get("size")).split())
            verts, faces = _box_mesh(sx, sy, sz)
            kind, primitive = "box", box
        elif cylinder is not None:
            # Downstream turns every cylinder into a capsule, so mesh the capsule.
            radius = float(str(cylinder.get("radius")))
            length = float(str(cylinder.get("length")))
            verts, faces = _capsule_mesh(radius, length, segments, cap_rings)
            kind, primitive = "capsule", cylinder
        else:
            continue

        name = _write_collision_mesh(urdf_path.parent, kind, verts, faces)
        geometry.remove(primitive)
        ET.SubElement(geometry, "mesh", {"filename": name})
        converted += 1

    if converted:
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return converted


def install_mesh_collisions(scene_utils, *, segments: int = _CAPSULE_SEGMENTS,
                            cap_rings: int = _CAPSULE_CAP_RINGS) -> None:
    """Make the object pool author convex-mesh colliders instead of box/capsule primitives.

    Wraps whichever ``generate_handle_head_urdfs`` ``scene_utils`` currently holds -- it binds the
    function by name at import (``scene_utils.py:25``), so the module attribute is what has to be
    replaced, not the one in ``generate_objects``.

    Idempotent, and a no-op unless installed: with this patch absent the generated URDFs are
    byte-identical to upstream's.
    """
    current = scene_utils.generate_handle_head_urdfs
    if getattr(current, "_meshified", False):
        return

    def meshified(*args, _current=current, **kwargs):
        paths, scales = _current(*args, **kwargs)
        for path in paths:
            meshify_urdf(path, segments=segments, cap_rings=cap_rings)
        return paths, scales

    meshified._meshified = True
    scene_utils.generate_handle_head_urdfs = meshified
