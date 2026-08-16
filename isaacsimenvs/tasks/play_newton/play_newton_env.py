"""PlayNewtonEnv — the Play task constructed on Newton/MJWarp.

Subclasses :class:`PlayEnv` and overrides nothing that defines the task. ``_pre_physics_step``,
``_apply_action``, ``_get_dones``, ``_get_rewards`` and ``_get_observations`` are all inherited,
so both backends run the same reward, observation and action code and a measured difference
between them is attributable to physics rather than to two diverging copies of the task.

What differs is construction, and it happens in two windows:

* **before** ``super().__init__``, which runs ``_setup_scene`` -- the patches that rewrite
  ``scene_utils``' asset builders, the quaternion conventions, and the USD cache lookup that
  replaces the Kit URDF converter;
* **after** it -- the patches that need live assets, which have no ``.data`` until initialised.

Getting a patch into the wrong window is a silent failure: it installs cleanly and simply never
takes effect on the objects that were already built.
"""

from __future__ import annotations

from isaacsimenvs.newton import compat, patches, physx_schema_shim, usd_cache
from isaacsimenvs.tasks.play.play_env import PlayEnv

from .play_newton_env_cfg import PlayNewtonEnvCfg

__all__ = ["PlayNewtonEnv", "PlayNewtonEnvCfg"]


class PlayNewtonEnv(PlayEnv):
    cfg: PlayNewtonEnvCfg

    def __init__(
        self, cfg: PlayNewtonEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        _install_pre_construction_patches(
            cfg,
            physics_cfg=type(self).build_physics_cfg(cfg),
            pre_clone_hook=type(self).pre_clone_scene_hook,
        )
        super().__init__(cfg, render_mode, **kwargs)
        _install_post_construction_patches(self)

    @staticmethod
    def build_physics_cfg(cfg: PlayNewtonEnvCfg):
        """The physics config this env constructs on. Override to change solver.

        A hook rather than a hardcoded call because a subclass may need a *different solver
        topology* -- the cable env replaces this single MJWarp solve with a coupled MJWarp+VBD
        one. It is a staticmethod so it can run before ``super().__init__``, which is the only
        point at which ``sim.physics`` can still be changed.
        """
        return cfg.newton.build(int(cfg.scene.num_envs))

    #: Optional ``fn(env)`` run at the end of ``setup_scene``, *before* the environments are
    #: replicated. Any asset a subclass adds must be created here rather than after
    #: ``super()._setup_scene()``: replication is also when Newton imports the stage into its
    #: model, so an asset added later exists as USD and is invisible to the solver.
    pre_clone_scene_hook = None

    def _setup_scene(self) -> None:
        # scene_utils.setup_scene has been wrapped by `install_cloning`, which performs the
        # explicit environment replication Isaac Lab 2.x did inside `clone_environments()` and
        # 3.0 no longer does. Without it Newton builds a single world and reset indexing walks
        # off the end of a one-element array with a CUDA illegal memory access.
        super()._setup_scene()


def _install_pre_construction_patches(cfg: PlayNewtonEnvCfg, *, physics_cfg, pre_clone_hook=None) -> None:
    """Everything that must be in place before ``_setup_scene`` runs. Idempotent.

    Order is not arbitrary and is easy to get subtly wrong:

    1. the compat shim re-binds ``isaaclab.utils.configclass``, which importing ``isaaclab_physx``
       shadows, so it has to be re-asserted after any Isaac import;
    2. the USD cache replaces the Kit URDF converter, which the asset bake calls, so it must be
       installed before any asset is built;
    3. the multi-asset patch decides how variants are spawned, and cloning replicates them --
       spawning prototypes only is correct exactly when something replicates them afterwards, so
       the two are one decision.
    """
    compat.install_isaaclab3_compat()

    # `scene_utils._bake_usd` authors PhysX-namespaced USD attributes through `pxr.PhysxSchema`,
    # which ships with Isaac Sim rather than usd-core. Must precede any asset bake. No-ops when
    # the genuine plugin is present, so it never shadows the real one.
    physx_schema_shim.install()

    # Swap PhysX out for Newton. `cfg.sim.physics` is 3.0's backend-neutral field (2.x called it
    # `physx`); the inherited PhysxCfg from PlayEnvCfg is simply replaced, so no PhysX manager is
    # ever constructed. Built by `build_physics_cfg`, which subclasses override -- the cable env
    # substitutes a coupled MJWarp+VBD solver here.
    cfg.sim.physics = physics_cfg

    # The task authors zero-gain USD drives and supply real gains through ImplicitActuatorCfg.
    # PhysX honours that runtime injection; Newton bakes actuation from USD when the solver is
    # built, so Isaac Lab has to synthesise NewtonActuator prims from the Lab actuator configs.
    cfg.sim.use_newton_actuators = True

    # `scene.replicate_physics` is deliberately left at the task default (False). Under 2.x it
    # meant "parse each environment independently because its contents differ"; in 3.0 it only
    # tells `isaaclab.cloner.replicate` to drop physics contexts. `install_cloning` below drains
    # its own queue with replicate_physics=True instead, so flipping the scene flag here would
    # not help and would disable the very replication Newton needs for its world partitioning.

    # Kit-less: conversion is impossible here, so this is a lookup that raises on a miss rather
    # than falling back. A fallback would either crash confusingly or silently convert with a
    # different importer, breaking the like-for-like comparison the cache exists to guarantee.
    usd_cache.install_reader()

    num_envs = int(cfg.scene.num_envs)
    patches.install_multi_asset(num_envs, prototypes_only=True)
    patches.install_quat_convention()
    patches.install_quat_math()
    patches.install_fingertip_order()
    # Newton's USD importer needs CollisionAPI pushed down onto mesh prims (Isaac Sim's URDF
    # importer puts it on Xform wrappers, which Newton skips -- silently dropping every robot
    # collider), and its MuJoCo solver bakes actuation from USD at build time, so the drives must
    # carry real gains. Both rewrite the stage and both are harmful under PhysX, which is why the
    # reference gates them per backend rather than applying them unconditionally.
    #
    # `filter_inter_env_collisions` is correspondingly off: the collision-API push authors its own
    # filtering, and doing both double-filters.
    # Subclass assets go in here, between asset creation and replication.
    if pre_clone_hook is not None:
        _install_pre_clone_hook(pre_clone_hook)

    patches.install_cloning(
        push_collision_api=True,
        write_drive_gains=True,
        filter_inter_env_collisions=False,
        request_full_stage=False,  # ovphysx-only; Newton ingests the authored stage as-is
    )
    patches.install_actuator_broadcast()
    patches.install_armature_writeback()
    patches.install_implicit_effort_fix()

    if cfg.use_gravcomp:
        patches.install_newton_gravcomp()
    else:
        patches.install_object_gravity()
        patches.disable_global_gravity(cfg)

    patches.install_newton_materials()
    patches.install_newton_condim(int(cfg.newton.condim))

    if cfg.meshify:
        scene_utils = compat.play_module("scene_utils")
        patches.install_mesh_collisions(scene_utils)

    # `install_multi_asset` above replaced `build_rigid_object_cfg`, discarding any single-variant
    # wrapper the harness applied before construction. Re-apply on top of ours.
    from isaacsimenvs.eval import protocol

    protocol.reapply_single_variant_if_requested()

    _rebind_play_env_globals()


def _install_pre_clone_hook(hook) -> None:
    """Run ``hook(env)`` at the end of ``setup_scene``, before ``install_cloning`` replicates.

    Installed *before* `install_cloning` wraps the same function, so the resulting call order is
    original setup -> hook -> replication. Getting that order wrong is silent: the asset appears
    on the stage, and the solver simply never sees it.
    """
    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_pre_clone_hook_installed", False):
        return
    original = scene_utils.setup_scene

    def setup_scene_then_hook(env, _original=original):
        _original(env)
        hook(env)

    scene_utils.setup_scene = setup_scene_then_hook
    scene_utils._pre_clone_hook_installed = True


def _rebind_play_env_globals() -> None:
    """Re-point ``play_env``'s module-level names at the patched functions.

    The patches above replace attributes on the ``play.utils.*`` modules, but ``play_env.py``
    binds the names it uses at *its* import time::

        from .utils.scene_utils import apply_physx_material_properties, setup_scene

    and ``play_newton_env`` imports ``PlayEnv`` before any patch is installed. Without this,
    ``PlayEnv.__init__`` and ``_setup_scene`` keep calling the *original* functions and every
    scene-level patch is silently inert -- ``install_cloning`` never replicates the environments,
    ``install_newton_materials`` never runs, and the first symptom is an unrelated-looking
    ``AttributeError: 'Articulation' object has no attribute 'root_physx_view'`` from a function
    that was supposed to have been replaced.

    Rebinding here rather than editing ``play_env.py`` keeps the task code identical between the
    two backends, which is the property the whole comparison rests on. It syncs every name
    ``play_env`` took from a patched module, not just the two known ones, so a future patch does
    not have to remember to come back here.
    """
    import importlib

    play_env = importlib.import_module("isaacsimenvs.tasks.play.play_env")
    for module_name in ("scene_utils", "reset_utils", "obs_utils", "action_utils",
                        "reward_utils", "termination_utils", "logging_utils"):
        module = compat.play_module(module_name)
        for name, current in vars(play_env).items():
            if not callable(current):
                continue
            patched = getattr(module, name, None)
            if patched is not None and patched is not current:
                setattr(play_env, name, patched)


def _install_post_construction_patches(env: PlayNewtonEnv) -> None:
    """Everything that needs live assets. Runs before the first ``reset()``."""
    # Isaac Lab 3.0 asset data are warp-backed ProxyArrays; the task code indexes them as torch
    # tensors throughout. This is the boundary that lets `play/utils/*` stay unmodified.
    patches.wrap_env_assets(env)
    # Mirror of the read-side conversion: the task code writes wxyz, 3.0 stores xyzw.
    patches.install_pose_write_conversion(env)
    # Newton floors degenerate inertia at finalisation, which lifts the hand's massless
    # placeholder links above the inertia of real finger bones and makes the fingers sluggish.
    # Must precede notify_solver so the corrected values are the ones picked up.
    patches.restore_placeholder_inertia(env)
    # SolverMuJoCo caches joint gains and shape materials in its own mjModel at construction, and
    # Isaac Lab writes both after that point.
    patches.notify_solver(env)
