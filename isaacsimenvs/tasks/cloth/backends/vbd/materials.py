"""Cloth-specific Newton setup before graph capture and per-world material reset."""

from contextlib import ExitStack, contextmanager
import re
from unittest.mock import patch


@contextmanager
def configure_materials(backend):
    """Install construction hooks while Newton builds this scene."""
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager
    from isaaclab_newton.physics import NewtonManager

    c = backend.cfg.cloth
    use_friction = c.self_contact_mu > 0 or c.table_contact_mu > 0
    with ExitStack() as stack:
        if c.full_surface_contact:
            import newton

            patterns = [
                re.compile(body)
                for proxy in backend.cfg.sim.physics.solver_cfg.proxies
                for body in proxy.bodies
            ]
            prepare = NewtonManager._prepare_builder_for_finalize

            def prepare_sdf(cls, builder):
                prepare(builder)
                for shape, (kind, body) in enumerate(zip(builder.shape_type, builder.shape_body)):
                    if kind not in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                        continue
                    included = (
                        any(p.search(builder.body_label[body] or "") for p in patterns)
                        if body >= 0 else builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES
                    )
                    if included:
                        builder.shape_force_sdf[shape] = True
                        if c.full_surface_sdf_resolution:
                            builder.shape_sdf_max_resolution[shape] = c.full_surface_sdf_resolution

            stack.enter_context(patch.object(NewtonManager, "_prepare_builder_for_finalize", classmethod(prepare_sdf)))

        if use_friction:
            create = NewtonVBDManager.__dict__["_create_solver"]

            def create_cloth_solver(cls, model, solver_cfg):
                model._cow_array("shape_material_mu")
                return create.__func__(cls, model, solver_cfg)

            stack.enter_context(patch.object(NewtonVBDManager, "_create_solver", classmethod(create_cloth_solver)))

        capture = NewtonManager.__dict__["_capture_or_defer_graph"]

        def before_capture(cls):
            backend._install_solver_materials(cls._model, cls._solver)
            return capture.__func__(cls)

        stack.enter_context(patch.object(NewtonManager, "_capture_or_defer_graph", classmethod(before_capture)))
        yield


def configure_friction(solver, view, cloth):
    """Keep the VBD view's friction independent of the rigid model and refresh it in place."""
    import numpy as np
    import warp as wp

    mu = cloth.self_contact_mu or cloth.soft_contact_mu
    parent = view.parent
    table_path = re.compile(r"/World/envs/env_[^/]+/Table(?:/|$)")
    table = np.array([b >= 0 and bool(table_path.search(str(parent.body_label[b])))
                      for b in parent.shape_body.numpy()])
    if cloth.table_contact_mu > 0 and cloth.proxy_table and not table.any():
        raise RuntimeError("No table shapes matched the cloth table friction override")
    refresh = solver._refresh_model_view_overrides

    def update():
        values = parent.shape_material_mu.numpy() * (cloth.soft_contact_mu / mu)
        if cloth.table_contact_mu > 0:
            values[table] = cloth.table_contact_mu**2 / mu
        view.shape_material_mu.assign(wp.array(values, dtype=float, device=view.shape_material_mu.device))

    def refresh_with_friction(flags):
        refresh(flags)
        update()

    solver._refresh_model_view_overrides = refresh_with_friction
    update()


def install_bending(model, entry, c, dt):
    """Install the chosen bending law on the exact arrays read by the cloth solver."""
    view = entry.view
    rest = view._cow_array("edge_rest_angle")
    if c.bending_rigidity > 0:
        from isaacsimenvs.tasks.cloth.utils.hinge_rheology import HingeRheology

        params = {name: getattr(c, name) for name in (
            "bending_rigidity", "friction_ratio",
            "creep_tau", "creep_curvature", "yield_curvature", "natural_curvature",
        )}
        for name in ("creep_tau", "creep_curvature", "yield_curvature"):
            params[name] = params[name] or float("inf")
        params.update(
            stiffness_scale=c.bending_stiffness_scale,
            friction_moment=c.friction_slip_curvature * c.friction_ratio * c.bending_rigidity,
            hardening=c.yield_hardening,
            imperfection_seed=c.mesh_seed,
        )
        material = HingeRheology(
            model, params, dt, solver_rest_angle=rest,
            solver_bending_properties=view._cow_array("edge_bending_properties"),
        )
    else:
        from isaacsimenvs.tasks.cloth.utils.plasticity import BendingPlasticity

        material = BendingPlasticity(
            model, c.plastic_yield_curvature, solver_rest_angle=rest, set_curvature=c.plastic_set_curvature,
        )
    material.wrap_solver(entry.solver)
    return material
