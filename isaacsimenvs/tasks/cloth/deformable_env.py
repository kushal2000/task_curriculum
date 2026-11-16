"""The fold task on a deformable VBD cloth sheet, on Newton (Isaac Lab 3, ``.venv/isaaclab3``).

Where rigid-body-trained policies are evaluated (or fine-tuned). Sections: ``PlayNewtonEnv`` (the
Play task's env on Newton: MJWarp robot + the patches that swap Isaac Lab's PhysX scene onto Newton),
``BendingPlasticity`` (optional fold memory for the VBD sheet), and ``ClothEnv`` (the sheet, on the
shared ``fold_task.ClothTaskMixin``).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from isaacsimenvs.newton import compat, patches, physx_schema_shim, usd_cache
from isaacsimenvs.tasks.cloth.deformable_env_cfg import ClothEnvCfg, PlayNewtonEnvCfg
from isaacsimenvs.tasks.cloth.fold_task import ClothAsRigidObject, ClothTaskMixin, corner_indices, grid_mesh
from isaacsimenvs.tasks.play.play_env import PlayEnv
from isaacsimenvs.tasks.play.robots import get_robot_spec

__all__ = ['BendingPlasticity', 'ClothEnv', 'PlayNewtonEnv', 'PlayNewtonEnvCfg', 'dihedral_angles']


# ==================================================================================================
# Play on Newton -- PlayNewtonEnv — the Play task constructed on Newton/MJWarp: the base of the VBD cloth env.
# (was newton_base_env.py)
# ==================================================================================================

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
    patches.install_newton_condim(
        int(cfg.newton.condim), tuple(get_robot_spec(cfg.robot).fingertip_link_names)
    )

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


# ==================================================================================================
# Bending plasticity -- Bending plasticity for a VBD sheet: a fold that goes past the yield point stays partly folded.
# (was utils/plasticity.py)
# ==================================================================================================

@wp.func
def _dihedral(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> float:
    """Signed dihedral angle of edge (x2, x3) with opposite vertices x0, x1, exactly as Newton's VBD
    bending evaluates it (``evaluate_dihedral_angle_based_bending_force_hessian``)."""
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)
    if n1_norm < 1.0e-6 or n2_norm < 1.0e-6 or e_norm < 1.0e-6:
        return 0.0
    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    return wp.atan2(wp.dot(wp.cross(n1_hat, n2_hat), e / e_norm), wp.dot(n1_hat, n2_hat))


@wp.kernel
def _dihedral_kernel(
    q: wp.array(dtype=wp.vec3), edge_indices: wp.array2d(dtype=wp.int32), out: wp.array(dtype=float)
):
    e = wp.tid()
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1:
        out[e] = 0.0
        return
    out[e] = _dihedral(q[edge_indices[e, 0]], q[edge_indices[e, 1]], q[edge_indices[e, 2]], q[edge_indices[e, 3]])


@wp.kernel
def _yield_kernel(
    q: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    yield_angle: wp.array(dtype=float),
    set_angle: wp.array(dtype=float),
    rest_angle: wp.array(dtype=float),
):
    e = wp.tid()
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1:
        return
    theta = _dihedral(q[edge_indices[e, 0]], q[edge_indices[e, 1]], q[edge_indices[e, 2]], q[edge_indices[e, 3]])
    d = theta - rest_angle[e]
    # Newton's bending term does not wrap theta - rest; neither does this, so the two agree on what
    # "elastic part" means even for a rest angle near +-pi.
    y = yield_angle[e]
    r = rest_angle[e]
    cand = r
    if d > y:
        cand = theta - y
    elif d < -y:
        cand = theta + y
    # Two-stage gate: growing set (|rest| increasing) needs a sharp bend, |theta| >= set angle;
    # shrinking it (toward flat) is always allowed. set_angle 0 = single-stage, as before.
    if wp.abs(cand) > wp.abs(r) and wp.abs(theta) < set_angle[e]:
        return
    rest_angle[e] = cand


@wp.kernel
def _restore_kernel(
    edge_ids: wp.array(dtype=wp.int32), initial: wp.array(dtype=float), rest_angle: wp.array(dtype=float)
):
    e = edge_ids[wp.tid()]
    rest_angle[e] = initial[e]


def dihedral_angles(q, edge_indices) -> np.ndarray:
    """Per-edge dihedral angle [rad] for positions ``q`` (warp vec3 array), 0 on boundary edges."""
    out = wp.zeros(edge_indices.shape[0], dtype=float, device=q.device)
    wp.launch(_dihedral_kernel, dim=edge_indices.shape[0], inputs=[q, edge_indices, out], device=q.device)
    return out.numpy()


class BendingPlasticity:
    """Yield-curvature plasticity on ``model.edge_rest_angle``.

    ``yield_curvature`` [1/m]: scalar, or one value per edge. ``inf`` disables yielding (the
    elastic sheet, unchanged). Call :meth:`apply` after every solver step with the new state, and
    :meth:`reset` (all edges, or the ids of some) when a sheet is put back flat.
    """

    def __init__(self, model, yield_curvature, rest_q=None, solver_rest_angle=None, set_curvature=0.0) -> None:
        """``solver_rest_angle``: the ``edge_rest_angle`` array the cloth SOLVER reads, when that is
        not ``model.edge_rest_angle`` -- under ``SolverCoupled`` the VBD entry runs on a ModelView
        that owns a projected copy, so writing the model's array alone never reaches the solver.
        The yield update then runs on the solver's array and is mirrored to the model's (which a
        view refresh projects back over the copy).

        ``set_curvature`` [1/m]: scalar or per edge; the two-stage gate (module docstring). 0 = off."""
        self.model = model
        self.edge_indices = model.edge_indices
        self.num_edges = int(model.edge_count)
        self.device = model.device
        #: The array the yield update writes (what the solver bends against) and its mirror.
        self.rest_angle = model.edge_rest_angle if solver_rest_angle is None else solver_rest_angle
        self.mirror = None if self.rest_angle.ptr == model.edge_rest_angle.ptr else model.edge_rest_angle
        if self.rest_angle.shape != model.edge_rest_angle.shape:
            raise ValueError("solver edge_rest_angle has a different edge layout from the model's")
        #: The build-time rest angles, restored by `reset`.
        self.initial_rest = wp.clone(model.edge_rest_angle)
        self.hinge_width = self._hinge_width(model.particle_q.numpy() if rest_q is None else np.asarray(rest_q))
        self.yield_angle = wp.zeros(self.num_edges, dtype=float, device=self.device)
        self.set_yield_curvature(yield_curvature)
        self.set_angle = wp.zeros(self.num_edges, dtype=float, device=self.device)
        self.set_set_curvature(set_curvature)

    def _hinge_width(self, rest_q: np.ndarray) -> np.ndarray:
        """``(A1 + A2) / edge_length`` per edge on the rest positions [m]; 0 on boundary edges."""
        ei = self.edge_indices.numpy()
        w = np.zeros(self.num_edges)
        interior = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
        x0, x1, x2, x3 = (rest_q[np.maximum(ei[:, k], 0)] for k in range(4))
        a1 = 0.5 * np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1)
        a2 = 0.5 * np.linalg.norm(np.cross(x3 - x1, x2 - x1), axis=1)
        length = np.linalg.norm(x3 - x2, axis=1)
        w[interior] = (a1 + a2)[interior] / np.maximum(length[interior], 1e-12)
        return w

    def set_yield_curvature(self, yield_curvature) -> None:
        kappa = np.broadcast_to(np.asarray(yield_curvature, dtype=np.float64), (self.num_edges,))
        # inf * width stays inf (no yield); float32 inf compares correctly in the kernel.
        with np.errstate(invalid="ignore"):
            ang = np.where(np.isinf(kappa), np.inf, kappa * self.hinge_width)
        self.yield_angle.assign(ang.astype(np.float32))

    def set_set_curvature(self, set_curvature) -> None:
        kappa = np.broadcast_to(np.asarray(set_curvature, dtype=np.float64), (self.num_edges,))
        self.set_angle.assign((kappa * self.hinge_width).astype(np.float32))

    def apply(self, state) -> None:
        wp.launch(
            _yield_kernel,
            dim=self.num_edges,
            inputs=[state.particle_q, self.edge_indices, self.yield_angle, self.set_angle, self.rest_angle],
            device=self.device,
        )
        if self.mirror is not None:
            wp.copy(self.mirror, self.rest_angle)

    def reset(self, edge_ids=None) -> None:
        """Restore the build-time rest angles of ``edge_ids`` (a warp int32 array or anything numpy
        takes; ``None`` = every edge)."""
        if edge_ids is None:
            wp.copy(self.rest_angle, self.initial_rest)
            if self.mirror is not None:
                wp.copy(self.mirror, self.initial_rest)
            return
        if not isinstance(edge_ids, wp.array):
            edge_ids = wp.array(np.asarray(edge_ids, dtype=np.int32).reshape(-1), dtype=wp.int32, device=self.device)
        if edge_ids.shape[0] == 0:
            return
        for target in (self.rest_angle, self.mirror):
            if target is not None:
                wp.launch(_restore_kernel, dim=edge_ids.shape[0], inputs=[edge_ids, self.initial_rest, target],
                          device=self.device)

    def wrap_solver(self, solver) -> None:
        """Run :meth:`apply` on the output state after every ``solver.step`` (each VBD substep).

        Patches the instance's ``step``, so a coupled solver that calls the sub-solver's ``step``
        picks it up; a CUDA graph captured after this call contains the update."""
        step = solver.step

        def step_with_plasticity(state_in, state_out, control, contacts, dt):
            step(state_in, state_out, control, contacts, dt)
            self.apply(state_out)

        solver.step = step_with_plasticity

    @staticmethod
    def yield_angle_deg(yield_curvature: float, hinge_width: float) -> float:
        """The hinge angle [deg] at which an edge of ``hinge_width`` yields."""
        return math.degrees(yield_curvature * hinge_width)


# ==================================================================================================
# The deformable cloth env -- ClothEnv — fold half a cloth sheet onto the other half.
# (was cloth_env.py)
# ==================================================================================================

class ClothEnv(ClothTaskMixin, PlayNewtonEnv):
    """Play task with a VBD cloth sheet as the manipuland."""

    cfg: ClothEnvCfg

    def __init__(self, cfg: ClothEnvCfg, render_mode: str | None = None, **kwargs):
        # One rigid-tool variant in every env. The tool is inert here (neutralised once the scene
        # is built), but its shapes stay in the model, and the pool's two hammers (cuboid handle:
        # box + box; cylinder handle: box + capsule) alternate across envs, which SolverMuJoCo
        # rejects: "requires homogeneous worlds. Shape types mismatch". The Newton launchers used to
        # pass --single_variant for this; the task now does it itself. meshify (every tool collider
        # a convex mesh) makes the full pool homogeneous, so it is left alone there.
        if not cfg.meshify:
            from isaacsimenvs.eval.protocol import use_single_object_variant

            use_single_object_variant()
        kappa = float(cfg.cloth.plastic_yield_curvature)
        self._plastic_set_curvature = float(cfg.cloth.plastic_set_curvature)
        self._plasticity = None
        restore = self._install_plasticity_at_capture(kappa) if kappa > 0.0 else None
        restore_sdf = self._install_full_surface_sdf(cfg) if cfg.cloth.full_surface_contact else None
        c = cfg.cloth
        restore_mu = (
            self._install_cloth_friction(cfg) if (c.self_contact_mu > 0.0 or c.table_contact_mu > 0.0) else None
        )
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            for undo in (restore, restore_sdf, restore_mu):
                if undo is not None:
                    undo()
        if restore_mu is not None:
            from isaaclab_newton.physics import NewtonManager
            from newton._src.solvers.coupled.solver_coupled import SolverCoupled

            SolverCoupled._cloth_friction_apply(NewtonManager._solver)  # (3)
        if kappa > 0.0 and self._plasticity is None:  # nothing was captured, so nothing installed it
            self._install_plasticity(kappa)

    @staticmethod
    def _install_cloth_friction(cfg: ClothEnvCfg):
        """Cloth-only shape frictions for ``cloth.self_contact_mu`` / ``cloth.table_contact_mu``.

        ``build_physics`` sets ``model.soft_contact_mu`` to the cloth-on-cloth value, which VBD's
        self-contact reads raw. Cloth-vs-shape contacts use ``sqrt(soft_contact_mu * shape_mu)``,
        so the cloth solver's ModelView gets its OWN ``shape_material_mu``: the parent's value
        scaled by ``mu_cal / mu_cc`` (``mu_cal`` = the preset's ``soft_contact_mu``), which keeps
        every hand pairing where the preset calibrated it, and ``T**2 / mu_cc`` on the table's
        shapes when ``table_contact_mu`` = T. The rigid solver keeps the parent's frictions.

        Timing matters: the task writes its frictions (robot 0.5, fingertips 1.5) into the PARENT
        model after start-up and ``notify_solver`` then re-projects shape attributes into every
        view (``SolverCoupled._refresh_model_view_overrides``), which would overwrite a rescale
        done once at construction. So: (1) ``NewtonVBDManager._create_solver`` is wrapped to make
        sure the view owns a copy before the solver (and any CUDA graph) exists; (2) the rescale is
        re-applied after every view refresh, IN PLACE so captured graphs keep the same pointer;
        (3) ``ClothEnv.__init__`` applies it once more at the end. Returns the undo for (1)."""
        import re

        import numpy as np
        import newton
        import warp as wp
        from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager
        from newton._src.solvers.coupled.solver_coupled import SolverCoupled

        c = cfg.cloth
        mu_cal = float(c.soft_contact_mu)
        mu_cc = float(c.self_contact_mu) if c.self_contact_mu > 0.0 else mu_cal
        table_t = float(c.table_contact_mu)

        def apply(solver) -> None:
            for entry in getattr(solver, "_entries", {}).values():
                if not isinstance(entry.solver, newton.solvers.SolverVBD):
                    continue
                view = entry.view
                parent = view.parent
                if "shape_material_mu" not in view.overrides:
                    raise RuntimeError("cloth view has no own shape_material_mu; rescaling would write the rigid model")
                pmu = parent.shape_material_mu.numpy().astype(np.float64)
                if view.shape_material_mu.shape[0] != pmu.shape[0]:
                    raise RuntimeError("cloth view shape layout differs from the model's; cannot map frictions")
                new = pmu * (mu_cal / mu_cc)
                labels = list(parent.body_label)
                sb = parent.shape_body.numpy()
                # Same match as the proxy set's `/World/envs/env_.*/Table` (a regex SEARCH: the table
                # body may sit below the Table prim).
                table = np.array([b >= 0 and re.search(r"/World/envs/env_[^/]+/Table", str(labels[b])) is not None
                                  for b in sb], dtype=bool)
                if table_t > 0.0:
                    if not table.any():
                        raise RuntimeError("cloth.table_contact_mu is set but no table shapes matched; body labels like "
                                           f"{[str(l) for l in labels[:3]]} ... {[str(l) for l in labels if 'able' in str(l)][:4]}")
                    new[table] = table_t**2 / mu_cc
                view.shape_material_mu.assign(wp.array(new.astype(np.float32), dtype=float, device=view.shape_material_mu.device))
                eff = np.sqrt(mu_cc * new)
                hand = (~table) & (sb >= 0)
                print(f"[cloth] friction: cloth-on-cloth {mu_cc:.3f} (model.soft_contact_mu "
                      f"{float(view.soft_contact_mu):.3f}); cloth-on-hand {sorted({round(float(v), 3) for v in eff[hand]})}; "
                      f"cloth-on-table {sorted({round(float(v), 3) for v in eff[table]})} ({int(table.sum())} shapes)",
                      flush=True)

        # (2) persistent, process-wide: re-apply after every view refresh.
        if not getattr(SolverCoupled, "_cloth_friction_wrapped", False):
            orig_refresh = SolverCoupled._refresh_model_view_overrides

            def _refresh(self, flags):
                orig_refresh(self, flags)
                fn = getattr(SolverCoupled, "_cloth_friction_apply", None)
                if fn is not None:
                    fn(self)

            SolverCoupled._refresh_model_view_overrides = _refresh
            SolverCoupled._cloth_friction_wrapped = True
        SolverCoupled._cloth_friction_apply = staticmethod(apply)

        # (1) the view must own its array before the solver is built.
        orig = NewtonVBDManager.__dict__["_create_solver"]

        def _create(cls, model, solver_cfg):
            if hasattr(model, "_cow_array"):
                model._cow_array("shape_material_mu")
            return orig.__get__(None, cls)(model, solver_cfg)

        NewtonVBDManager._create_solver = classmethod(_create)

        def undo() -> None:
            NewtonVBDManager._create_solver = orig

        return undo

    @staticmethod
    def _install_full_surface_sdf(cfg: ClothEnvCfg):
        """Give every mesh shape the cloth can touch a volume SDF, before ``ModelBuilder.finalize``.

        ``cloth.full_surface_contact`` makes the proxy pipeline sample each participating rigid
        mesh's SDF for edge/face contacts, and Newton raises on a mesh without one. Participating =
        the proxy bodies (``ClothCfg.proxy_body_patterns``) plus static shapes that collide with
        particles; the coupler clears collision flags on every other shape. Wraps
        ``NewtonManager._prepare_builder_for_finalize`` (the coupler manager reaches it via
        ``super()``); returns the undo."""
        import re

        import newton
        from isaaclab_newton.physics import NewtonManager

        from isaacsimenvs.tasks.play.robots import get_robot_spec

        c = cfg.cloth
        pats = [re.compile(p) for p in c.proxy_body_patterns(get_robot_spec(cfg.robot))]
        mesh_types = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        res = int(c.full_surface_sdf_resolution) or None
        orig = NewtonManager.__dict__["_prepare_builder_for_finalize"]

        def _prep(cls, builder) -> None:
            orig.__get__(None, cls)(builder)
            body_label = builder.body_label
            n, kinds = 0, {}
            for s in range(builder.shape_count):
                if int(builder.shape_type[s]) not in mesh_types:
                    continue
                b = int(builder.shape_body[s])
                if b >= 0:
                    lab = body_label[b] or ""
                    if not any(p.search(lab) for p in pats):
                        continue
                    key = lab.rsplit("/", 1)[-1]
                elif int(builder.shape_flags[s]) & collide_particles:
                    key = "static:" + str((builder.shape_label[s] or "").rsplit("/", 1)[-1])
                else:
                    continue
                builder.shape_force_sdf[s] = True
                if res is not None:
                    builder.shape_sdf_max_resolution[s] = res
                n += 1
                kinds[key] = kinds.get(key, 0) + 1
            per_world = {k: v for k, v in sorted(kinds.items())[:24]}
            print(f"[cloth] full_surface_contact: SDF on {n} mesh shapes (max_resolution "
                  f"{res or 'default'}); by body: {per_world}", flush=True)

        NewtonManager._prepare_builder_for_finalize = classmethod(_prep)

        def undo() -> None:
            NewtonManager._prepare_builder_for_finalize = orig

        return undo

    def _install_plasticity_at_capture(self, kappa: float):
        """Install plasticity right before Newton's first CUDA graph capture, so the graph holds it.

        Installing after `super().__init__` and capturing again does not work: the re-captured
        coupled step fails to instantiate ("invalid argument"), the back-to-back capture problem
        `NewtonManager.start_simulation` documents. Every start runs `_capture_or_defer_graph` once
        the model and solver exist, graph or not. Returns the undo."""
        from isaaclab_newton.physics import NewtonManager

        orig = NewtonManager.__dict__["_capture_or_defer_graph"]
        env = self

        def capture(cls):
            if env._plasticity is None:
                env._install_plasticity(kappa)
            return orig.__func__(cls)

        NewtonManager._capture_or_defer_graph = classmethod(capture)
        return lambda: setattr(NewtonManager, "_capture_or_defer_graph", orig)

    def _install_plasticity(self, kappa: float) -> None:
        """Wrap the coupled solver's cloth entry so every VBD substep updates the rest angles."""
        import numpy as np
        import torch
        from isaaclab_newton.physics import NewtonManager


        model = NewtonManager._model
        cloth_solver = NewtonManager._solver.solver("cloth")
        # SolverCoupled runs the VBD entry on a ModelView with its OWN projected edge_rest_angle:
        # the yield update must write that array (the model's alone never reaches the solver, which
        # left every env sheet purely elastic), mirrored to the model's for refreshes and readers.
        views = [e.view for e in getattr(NewtonManager._solver, "_entries", {}).values() if e.solver is cloth_solver]
        solver_rest = views[0].edge_rest_angle if views else None
        kappa_set = float(getattr(self, "_plastic_set_curvature", 0.0))
        plast = BendingPlasticity(model, kappa, solver_rest_angle=solver_rest, set_curvature=kappa_set)
        if kappa_set > 0.0:
            print(f"[cloth] two-stage plasticity: set grows only past {kappa_set:g} /m "
                  f"(yield {kappa:g} /m)", flush=True)
        plast.wrap_solver(cloth_solver)
        print(f"[cloth] plasticity writes the cloth solver's edge_rest_angle "
              f"({'its own view copy, mirrored to the model' if plast.mirror is not None else 'shared with the model'})",
              flush=True)
        ei = model.edge_indices.numpy()
        world = model.particle_world.numpy()[ei[:, 2]]
        world[(ei[:, 0] < 0) | (ei[:, 1] < 0)] = -1  # boundary edges never bend
        # `self.device` may not exist yet: this can run inside `super().__init__`.
        self._edge_env = torch.as_tensor(world, device=str(model.device), dtype=torch.long)
        self._plast_resets_logged = 0
        self._plasticity = plast
        w = plast.hinge_width[world >= 0]
        print(f"[cloth] plasticity: yield curvature {kappa:g} /m = {np.degrees(kappa * w.max()):.1f} deg per grid "
              f"cell, {np.degrees(kappa * w.min()):.1f} deg per diagonal, {int((world >= 0).sum())} hinges", flush=True)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if getattr(self, "_plasticity", None) is None:  # off, or a reset inside super().__init__
            return
        import torch
        import warp as wp

        dev = self._edge_env.device
        ids = torch.arange(self.num_envs, device=dev) if env_ids is None else \
            torch.as_tensor(env_ids, device=dev, dtype=torch.long).reshape(-1)
        edges = torch.nonzero(torch.isin(self._edge_env, ids)).reshape(-1)
        # Log the first few resets of envs that have run, as evidence the substep update is live.
        log = self._plast_resets_logged < 3 and edges.numel() and getattr(self, "common_step_counter", 0) > 0
        if log:
            rest = wp.to_torch(self._plasticity.model.edge_rest_angle)[edges]
            before = (float(rest.abs().max()) * 57.2958, int((rest.abs() > 0.1745).sum()))
        self._plasticity.reset(wp.from_torch(edges.to(torch.int32)))
        if log:
            after = float(wp.to_torch(self._plasticity.model.edge_rest_angle)[edges].abs().max()) * 57.2958
            print(f"[cloth] plasticity reset of {ids.numel()} envs: largest set angle {before[0]:.1f} deg, {before[1]} hinges "
                  f"past 10 deg; after reset {after:.1f} deg", flush=True)
            self._plast_resets_logged += 1

    @staticmethod
    def build_physics_cfg(cfg: "ClothEnvCfg"):
        """Coupled MJWarp (robot) + VBD (cloth) solve, sized for this env count."""
        from isaacsimenvs.tasks.play.robots import get_robot_spec

        return cfg.cloth.build_physics(cfg.newton, int(cfg.scene.num_envs), get_robot_spec(cfg.robot))

    #: Runs at the end of `setup_scene`, before replication -- which is also when Newton imports
    #: the stage. A mesh authored after that point is invisible to the solver.
    pre_clone_scene_hook = staticmethod(lambda env: env._install_cloth())

    def _install_cloth(self) -> None:
        """Author the sheet and register it as a deformable, in place of the rigid tool."""
        # Re-validate here, not only in `ClothCfg.__post_init__`. Hydra constructs the dataclass
        # with its DEFAULTS -- which is when `__post_init__` runs and passes -- and only then
        # assigns the CLI overrides onto the instance. So `env.cloth.thickness=0.03` walks straight
        # past the thickness/spacing guard: it ran against thickness 0.012 and never saw 0.03.
        # That is how a 30 mm radius ended up on a 12.5 mm grid, every particle overlapping its
        # neighbours, which the solver expresses as instability rather than as a config error.
        self.cfg.cloth._check_thickness()
        self.cfg.cloth.check_task()
        if self.cfg.cloth.task == "mixed":
            raise NotImplementedError("cloth.task=mixed is for the PhysX pool and chain; the VBD sheet runs fold or unfold")
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Sdf, UsdGeom, UsdShade

        c = self.cfg.cloth
        verts, indices = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        # Corners of the moving half, not points along one edge: four points spanning an area
        # determine a rotation; four collinear points do not.
        self._cloth_kp_idx = corner_indices(c.resolution, c.fold_axis)

        # The inherited rigid tool is still spawned by `setup_scene` -- the asset pipeline, USD
        # cache and `object_scales` observation are all built around it. Make it inert, or it is a
        # live rigid body loose in the scene. (On the cable task this was visible for hours as a
        # hammer lying on the floor beside the table.)
        self._neutralise_rigid_object()
        self._reshape_goal_marker()

        prim_path = "/World/envs/env_.*/Cloth"
        spawn_z = float(self.cfg.reset.table_reset_z) + c.start_height

        # Author the mesh directly: `DeformableObject` reads vertices and indices off the prim, and
        # a `UsdGeom.Mesh` is what makes it a *surface* (cloth) rather than a volume.
        stage = None
        for env_prim in find_matching_prims("/World/envs/env_.*"):
            stage = env_prim.GetStage()
            mesh_path = env_prim.GetPath().AppendChild("Cloth")
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            # Height is baked into the POINTS, not applied as a transform and not left to
            # `init_state.pos`. Measured: init_state alone left the sheet on the ground (z=0.006),
            # and a translate op on top of it put the sheet at 1.15 m and then exploded to 70 m.
            # Baking it is the one placement that cannot compose with another.
            mesh.CreatePointsAttr([(v[0], v[1], v[2] + spawn_z) for v in verts])
            mesh.CreateFaceVertexIndicesAttr(list(indices))
            mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            mesh.CreateDisplayColorAttr([tuple(c.color)])
            # Doubled-sided: a cloth folded over itself is viewed from both sides, and a
            # single-sided sheet renders as half-invisible once flipped.
            mesh.CreateDoubleSidedAttr(True)
            # NO translate op here. `DeformableObject` reads these points and then applies
            # `init_state.pos` on top, so a transform on the prim composes with it and the sheet
            # spawns at twice the intended height -- which is exactly what it did.

            # `DeformableObject` reads its material off a *bound* physics material carrying
            # `newton:*` attributes, and refuses the prim outright without one. It identifies the
            # material by the presence of `newton:density`, so that attribute is what makes this a
            # Newton deformable material rather than an ordinary shading material.
            mat_path = env_prim.GetPath().AppendChild("ClothMaterial")
            mat = UsdShade.Material.Define(stage, mat_path)
            mp = mat.GetPrim()
            for name, value in (
                ("newton:density", c.density),
                ("newton:particleRadius", c.particle_radius),
                ("newton:triKe", c.tri_ke),
                ("newton:triKa", c.tri_ka),
                ("newton:triKd", c.tri_kd),
                ("newton:edgeKe", c.edge_ke),
                ("newton:edgeKd", c.edge_kd),
            ):
                mp.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))

            binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
            binding.GetDirectBindingRel("physics").SetTargets([mat_path])

        cloth_cfg = DeformableObjectCfg(
            prim_path=prim_path,
            spawn=None,  # the prim is authored above; nothing to spawn
            # Identity: the height is already in the points.
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        cloth = DeformableObject(cloth_cfg)
        self.scene.deformable_objects["cloth"] = cloth
        self._cloth = cloth

        # `PlayEnv` holds the manipuland as `env.object` and every downstream module reads it from
        # there. Without this the policy observes the *inert rigid tool* -- which this env has just
        # made kinematic and collision-free -- while the sheet sits in the scene as a bystander.
        area_density = c.density * c.thickness  # surface density -> sheet mass
        self.object = ClothAsRigidObject(
            cloth,
            num_envs=self.num_envs,
            # LOCAL coordinates (z=0). `write_root_pose_to_sim` adds a centre whose z is
            # `spawn_z`, so baking the height in here too would place the sheet at 2 x spawn_z --
            # which it did, at 1.06 m instead of 0.53 m.
            rest_local=verts,
            keypoint_idx=self._cloth_kp_idx,
            corner_idx=self._cloth_kp_idx,
            corner_rest=[verts[i] for i in self._cloth_kp_idx],
            spawn_z=spawn_z,
            mass=area_density * c.size * c.size,
            device=self.device,
            reset_offset_xy=c.reset_offset_xy,
            # task: unfold -- reset with the + half folded over, the plies one self-contact radius
            # apart (where cloth-on-cloth contact settles; see `_init_fold_targets`).
            fold_start=(
                (0 if c.fold_axis == "x" else 1, list(c.unfold_start_deg), c.self_contact_radius)
                if c.task == "unfold" else None
            ),
        )
        self._write_object_scale()

        print(
            f"[cloth] {c.resolution}x{c.resolution} sheet, {c.size:.3f} m, "
            f"{c.num_particles} particles, {len(indices) // 3} triangles, "
            f"keypoints {self._cloth_kp_idx}",
            flush=True,
        )
