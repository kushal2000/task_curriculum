"""Temporary Newton hooks must not leak when environment construction fails."""

from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def startup_hooks(monkeypatch):
    from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat

    compat.install_isaaclab3_compat()
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager
    from isaaclab_newton.physics import NewtonManager
    from newton._src.solvers.coupled.solver_coupled import SolverCoupled

    from isaacsimenvs.tasks.cloth.backends.vbd import solver

    # Friction's refresh hook intentionally survives startup; isolate it between tests.
    for name in ("_refresh_model_view_overrides", "_cloth_friction_wrapped", "_cloth_friction_apply"):
        monkeypatch.setattr(SolverCoupled, name, SolverCoupled.__dict__.get(name), raising=False)

    methods = (
        (NewtonManager, "_capture_or_defer_graph"),
        (NewtonManager, "_prepare_builder_for_finalize"),
        (NewtonVBDManager, "_create_solver"),
    )
    originals = [owner.__dict__[name] for owner, name in methods]
    cloth = NS(
        plastic_yield_curvature=2.0, plastic_set_curvature=3.0,
        full_surface_contact=True, full_surface_sdf_resolution=32,
        self_contact_mu=0.5, soft_contact_mu=0.25, table_contact_mu=0.0,
        proxy_body_patterns=lambda robot: [r"/Robot/"],
    )
    return solver, NS(deformable_cloth=cloth), methods, originals


@pytest.mark.parametrize("fail_construction", [False, True])
def test_startup_restores_manager_methods(startup_hooks, fail_construction):
    solver, cfg, methods, originals = startup_hooks
    env = NS()
    try:
        with solver.startup(env, cfg):
            assert all(owner.__dict__[name] is not original
                       for (owner, name), original in zip(methods, originals))
            assert env._plasticity is None
            assert env._plastic_set_curvature == 3.0
            if fail_construction:
                raise RuntimeError("scene construction failed")
    except RuntimeError as exc:
        assert fail_construction and str(exc) == "scene construction failed"
    assert all(owner.__dict__[name] is original
               for (owner, name), original in zip(methods, originals))


def test_startup_unwinds_partially_installed_hooks(startup_hooks, monkeypatch):
    solver, cfg, methods, originals = startup_hooks

    def fail(cfg):
        raise RuntimeError("friction setup failed")

    monkeypatch.setattr(solver, "install_cloth_friction", fail)
    with pytest.raises(RuntimeError, match="friction setup failed"), solver.startup(NS(), cfg):
        pytest.fail("construction should not start")
    assert all(owner.__dict__[name] is original
               for (owner, name), original in zip(methods, originals))
