"""Newton setup hooks for cloth friction, full-surface contact, and bending plasticity.

Temporary manager patches surround environment construction. The installed solver hooks
remain active for stepping and model refreshes; physics values come from the environment config.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING

import torch

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import patches
from isaacsimenvs.tasks.cloth.backends.vbd import scene
from isaacsimenvs.tasks.cloth.backends.vbd.plasticity import BendingPlasticity
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT

if TYPE_CHECKING:
    from isaacsimenvs.tasks.cloth.backends.vbd.config import DeformableClothEnvCfg


def configure(cfg: DeformableClothEnvCfg) -> None:
    """Build coupled physics and install the shared Isaac Lab 3 compatibility fixes."""
    cfg.sim.physics = cfg.deformable_cloth.build_physics(cfg.newton, int(cfg.scene.num_envs), ROBOT)
    cfg.sim.use_newton_actuators = True
    patches.install_actuator_broadcast()
    patches.install_armature_writeback()
    patches.install_implicit_effort_fix()
    if not cfg.use_gravcomp:
        raise ValueError("use_gravcomp=false needs the old object-gravity wrench path, which is gone")
    patches.install_newton_gravcomp()
    patches.install_newton_condim(int(cfg.newton.condim), tuple(ROBOT.fingertip_link_names))


@contextmanager
def startup(env, cfg: DeformableClothEnvCfg):
    """Install construction hooks and restore manager methods even if setup fails."""
    cloth = cfg.deformable_cloth
    env._plasticity = None
    env._plastic_set_curvature = float(cloth.plastic_set_curvature)
    with ExitStack() as cleanup:
        if cloth.plastic_yield_curvature > 0.0:
            cleanup.callback(install_plasticity_at_capture(env, float(cloth.plastic_yield_curvature)))
        if cloth.full_surface_contact:
            cleanup.callback(install_full_surface_sdf(cfg))
        if cloth.self_contact_mu > 0.0 or cloth.table_contact_mu > 0.0:
            cleanup.callback(install_cloth_friction(cfg))
        yield


def finish_setup(env, cfg: DeformableClothEnvCfg) -> None:
    """Restore robot materials, refresh solver views, and finish uncaptured plasticity setup."""
    # Newton floors the massless placeholder links' inertia during finalization.
    patches.restore_placeholder_inertia(env)
    scene.set_base_materials(env, ROBOT)
    patches.notify_solver(env)
    cloth = cfg.deformable_cloth
    if cloth.self_contact_mu > 0.0 or cloth.table_contact_mu > 0.0:
        from isaaclab_newton.physics import NewtonManager
        from newton._src.solvers.coupled.solver_coupled import SolverCoupled

        SolverCoupled._cloth_friction_apply(NewtonManager._solver)
    if cloth.plastic_yield_curvature > 0.0 and env._plasticity is None:
        install_plasticity(env, float(cloth.plastic_yield_curvature))


def install_cloth_friction(cfg: DeformableClothEnvCfg):
    """Give the cloth view its own shape friction array; return the construction-hook undo.

    Cloth pairs use sqrt(self_contact_mu * shape_mu). Rescale shape_mu by
    soft_contact_mu / self_contact_mu to preserve hand friction, and use
    table_contact_mu**2 / self_contact_mu for an explicit table override.
    Allocate before CUDA capture and reapply in place after every model-view refresh.
    The refresh hook persists; the temporary solver-construction wrapper is restored."""
    import re

    import newton
    import numpy as np
    import warp as wp
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager
    from newton._src.solvers.coupled.solver_coupled import SolverCoupled

    c = cfg.deformable_cloth
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

    # Reapply after every view refresh for the lifetime of the solver.
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

    # The view must own its array before the solver is built.
    orig = NewtonVBDManager.__dict__["_create_solver"]

    def _create(cls, model, solver_cfg):
        if hasattr(model, "_cow_array"):
            model._cow_array("shape_material_mu")
        return orig.__get__(None, cls)(model, solver_cfg)

    NewtonVBDManager._create_solver = classmethod(_create)

    def undo() -> None:
        NewtonVBDManager._create_solver = orig

    return undo


def install_full_surface_sdf(cfg: DeformableClothEnvCfg):
    """Provision SDFs before model finalization; return the manager-hook undo.

    Full-surface contact needs SDFs on proxy meshes and particle-colliding static
    meshes. The coupler disables collision on the other shapes."""
    import re

    import newton
    from isaaclab_newton.physics import NewtonManager

    c = cfg.deformable_cloth
    pats = [re.compile(p) for p in c.proxy_body_patterns(ROBOT)]
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


def install_plasticity_at_capture(env, kappa: float):
    """Install plasticity before the first CUDA capture; return the manager-hook undo.

    Recapturing after environment construction fails in Newton. This hook also
    runs when graph capture is disabled, once the model and solver exist."""
    from isaaclab_newton.physics import NewtonManager

    orig = NewtonManager.__dict__["_capture_or_defer_graph"]

    def capture(cls):
        if env._plasticity is None:
            install_plasticity(env, kappa)
        return orig.__func__(cls)

    NewtonManager._capture_or_defer_graph = classmethod(capture)
    return lambda: setattr(NewtonManager, "_capture_or_defer_graph", orig)


def install_plasticity(env, kappa: float) -> None:
    """Wrap the coupled solver's cloth entry so every VBD substep updates the rest angles."""
    import numpy as np
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager._model
    cloth_solver = NewtonManager._solver.solver("cloth")
    # SolverCoupled runs the VBD entry on a ModelView with its OWN projected edge_rest_angle:
    # the yield update must write that array (the model's alone never reaches the solver, which
    # left every env cloth purely elastic), mirrored to the model's for refreshes and readers.
    views = [e.view for e in getattr(NewtonManager._solver, "_entries", {}).values() if e.solver is cloth_solver]
    solver_rest = views[0].edge_rest_angle if views else None
    kappa_set = float(getattr(env, "_plastic_set_curvature", 0.0))
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
    # `env.device` may not exist yet: this can run inside `super().__init__`.
    env._edge_env = torch.as_tensor(world, device=str(model.device), dtype=torch.long)
    env._plasticity = plast
    w = plast.hinge_width[world >= 0]
    print(f"[cloth] plasticity: yield curvature {kappa:g} /m = {np.degrees(kappa * w.max()):.1f} deg per grid "
          f"cell, {np.degrees(kappa * w.min()):.1f} deg per diagonal, {int((world >= 0).sum())} hinges", flush=True)


def build_physics(cloth, newton_cfg, num_envs: int, robot_spec=None):
    """Build coupled MJWarp (robot) + VBD (cloth) physics.
    The cloth entry must select all_particles=True: it owns particles, not rigid bodies."""
    from isaaclab_contrib.coupling import (
        CouplerEntryCfg,
        CouplerProxyCfg,
        CouplerProxyMappingCfg,
    )
    from isaaclab_contrib.deformable import NewtonModelCfg, VBDSolverCfg
    from isaaclab_newton.physics import (
        MJWarpSolverCfg,
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonShapeCfg,
    )

    robot_spec = robot_spec or ROBOT
    rigid_bodies = [r"/World/envs/env_.*/Robot"]
    proxy_bodies = cloth.proxy_body_patterns(robot_spec)
    if cloth.proxy_table:
        rigid_bodies.append(r"/World/envs/env_.*/Table")

    budget = cloth.triangle_pair_budget(newton_cfg, num_envs)

    # Use an instance: subclassing changes the module used to resolve its manager.
    vbd = VBDSolverCfg(
        iterations=cloth.vbd_iterations,
        rigid_body_particle_contact_buffer_size=cloth.rigid_body_particle_contact_buffer_size,
        rigid_contact_k_start=cloth.rigid_contact_k_start,
        # Self-contact. Without these the folded flap passes through the half it lands on --
        # `particle_enable_self_contact` defaults to False and this call never set it.
        particle_enable_self_contact=cloth.enable_self_contact,
        particle_self_contact_radius=cloth.self_contact_radius,
        particle_self_contact_margin=cloth.self_contact_margin,
        particle_collision_detection_interval=cloth.self_contact_detection_interval,
    )
    # Undeclared config fields are forwarded through SolverVBD's signature.
    vbd.rigid_body_contact_buffer_size = cloth.rigid_body_contact_buffer_size

    vbd.particle_conservative_bound_relaxation = cloth.conservative_bound_relaxation
    vbd.rigid_avbd_beta = cloth.rigid_avbd_beta
    if cloth.friction_epsilon > 0.0:
        vbd.friction_epsilon = cloth.friction_epsilon

    solver = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=MJWarpSolverCfg(
                    solver=newton_cfg.solver,
                    integrator=newton_cfg.integrator,
                    njmax=newton_cfg.njmax,
                    nconmax=newton_cfg.nconmax,
                    impratio=newton_cfg.impratio,
                    cone=newton_cfg.cone,
                    iterations=newton_cfg.iterations,
                    ls_iterations=newton_cfg.ls_iterations,
                    use_mujoco_contacts=newton_cfg.use_mujoco_contacts,
                    ccd_iterations=newton_cfg.ccd_iterations,
                    disable_sensors=newton_cfg.disable_sensors,
                ),
                bodies=rigid_bodies,
            ),
            CouplerEntryCfg(
                name="cloth",
                solver_cfg=vbd,
                # Particles, not bodies -- see the docstring.
                all_particles=True,
                include_static_shapes=True,
                substeps=cloth.substeps,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="cloth",
                bodies=proxy_bodies,
                mass_scale=cloth.mass_scale,
                mode=cloth.proxy_mode,
                collide_interval=cloth.collide_interval,
                # The proxy builds its OWN pipeline and defaults max_triangle_pairs to a
                # GLOBAL 1e6; the outer collision_cfg never reaches it. On the cable that
                # silently dropped contacts above ~16 envs for hours.
                collision_pipeline=NewtonCollisionPipelineCfg(
                    rigid_contact_max=newton_cfg.rigid_contact_max,
                    max_triangle_pairs=budget,
                    # Edge/face soft contacts -- see `full_surface_contact`.
                    enable_rigid_soft_full_surface_contact=cloth.full_surface_contact,
                ),
            )
        ],
        iterations=cloth.coupler_iterations,
        model_cfg=NewtonModelCfg(
            soft_contact_ke=cloth.soft_contact_ke,
            # `soft_contact_kd` was previously left at the `NewtonModelCfg` default (1.0e-2)
            # because this call only passed `ke` and `mu`.
            soft_contact_kd=cloth.soft_contact_kd,
            # Self-contact reads this raw; cloth-vs-shape pairs are rescaled in the cloth view
            # (`solver.install_cloth_friction`) so they keep using `soft_contact_mu`.
            soft_contact_mu=cloth.self_contact_mu if cloth.self_contact_mu > 0.0 else cloth.soft_contact_mu,
        ),
    )

    return NewtonCfg(
        solver_cfg=solver,
        collision_cfg=NewtonCollisionPipelineCfg(
            rigid_contact_max=newton_cfg.rigid_contact_max,
            max_triangle_pairs=budget,
        ),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=newton_cfg.num_substeps,
        collision_decimation=newton_cfg.collision_decimation,
        use_cuda_graph=bool(cloth.use_cuda_graph),
    )
