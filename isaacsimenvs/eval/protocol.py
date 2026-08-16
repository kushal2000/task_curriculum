"""Settings that define the comparison protocol, applied identically on every backend.

These are not task configuration and not tuning. Each one removes a source of variation that
would otherwise make two backends' numbers incomparable, so they live in one module that both the
PhysX and Newton runs import -- rather than being re-implemented per backend, which is how a
protocol difference creeps in between two experiments.
"""

from __future__ import annotations


def disable_randomization(cfg) -> None:
    """Turn off every DR and reset-noise source, and pin the object's start pose.

    Domain randomisation is *on* by default in ``cfg/task/Play.yaml`` because that is the
    training regime. For evaluation it has to be off: obs/action delays, object-state noise and
    random wrench impulses all inject variance that has nothing to do with the backend under
    test, which is the comparison this measurement exists to support.
    """
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    rs.fixed_start_pose = (
        0.0,
        0.0,
        rs.table_reset_z + rs.table_object_z_offset,
        1.0,
        0.0,
        0.0,
        0.0,
    )


#: Set when :func:`use_single_object_variant` has been requested, so a backend that patches
#: ``build_rigid_object_cfg`` during construction can re-apply the wrapper on top of its own.
_SINGLE_VARIANT_REQUESTED = False


def single_variant_requested() -> bool:
    return _SINGLE_VARIANT_REQUESTED


def reapply_single_variant_if_requested() -> None:
    """Re-wrap ``build_rigid_object_cfg`` after a backend has installed its own patches.

    Ordering trap, and a silent one. :func:`use_single_object_variant` wraps whatever
    ``build_rigid_object_cfg`` is current, but the Newton backend's ``install_multi_asset``
    *replaces* that function during env construction -- which happens after the harness has
    already applied the wrapper. The wrapper is discarded, every env gets its own variant again,
    and the only symptom is ``SolverMuJoCo requires homogeneous worlds`` from deep inside the
    solver, which reads like a limitation rather than a lost patch.

    Calling this at the end of the backend's patch installation composes the two correctly.
    """
    if _SINGLE_VARIANT_REQUESTED:
        _wrap_build_rigid_object_cfg()


def use_single_object_variant() -> None:
    """Give every environment the same object, by truncating the spawn list to its first entry.

    Two distinct reasons to want this, and they are worth keeping separate:

    * **As a control.** It removes object diversity as a variable, so a difference between two
      stacks cannot be explained by them manipulating different tools.
    * **As a requirement.** ``SolverMuJoCo`` rejects worlds whose collision shapes differ in
      *type*, and the procedural tools mix boxes and capsules -- even a single ``handle_head``
      request emits both a handle-only variant (capsule) and a handle+head one (capsule + box).
      So without this, Newton cannot construct the scene at all::

          ValueError: SolverMuJoCo requires homogeneous worlds. Shape types mismatch at
          position 36: world 0 has type 4, but other worlds have types [7, 4, 7].

      The alternative -- ``meshify``, which makes every collider a convex mesh and so uniform in
      type -- does let Newton run the full pool, but it changes the geometry both backends see
      and therefore adds a confound to the very comparison being made.

    Wraps whichever ``build_rigid_object_cfg`` is currently installed, so it composes with the
    Newton multi-asset patch rather than replacing it. Idempotent.

    Applied to the *shared* ``scene_utils``, so both backends get an identical object. On a
    backend that re-patches ``build_rigid_object_cfg`` during construction, the wrapper is
    re-applied afterwards -- see :func:`reapply_single_variant_if_requested`.
    """
    global _SINGLE_VARIANT_REQUESTED
    _SINGLE_VARIANT_REQUESTED = True
    _wrap_build_rigid_object_cfg()


def _wrap_build_rigid_object_cfg() -> None:
    import importlib

    scene_utils = importlib.import_module("isaacsimenvs.tasks.play.utils.scene_utils")
    current = scene_utils.build_rigid_object_cfg
    if getattr(current, "_single_variant", False):
        return

    def one_variant(prim_path, usd_paths, _current=current):
        return _current(prim_path, list(usd_paths)[:1])

    one_variant._single_variant = True
    scene_utils.build_rigid_object_cfg = one_variant


__all__ = ["disable_randomization", "use_single_object_variant"]
