"""Making the cloth task's modules importable under Isaac Lab 3.0.

Getting code to *load*; runtime behaviour is patched in
:mod:`isaacsimenvs.tasks.cloth.backends.vbd.newton_support.patches`. Isaac Lab 3.0 moved several
symbols the task imports by their 2.x paths, and renamed one constructor keyword. They are aliased
back, so one copy of the task code runs on both backends: if the PhysX and Newton envs ran
different reward, observation or action code, a measured difference between them would stop being
about physics.

Adapted from ``simtoolreal_newton.upstream`` in github.com/kushal2000/isaac_newton @ beb9efb.
"""

from __future__ import annotations


def install_isaaclab3_compat() -> list[str]:
    """Re-expose symbols Isaac Lab 3.0 relocated. Idempotent; returns what it applied.

    Must run before any task module is imported, and before anything else imports a Kit-backed
    module -- see the ``configclass`` note below for why the ordering is load-bearing.
    """
    applied: list[str] = []

    import isaaclab.sim as isaaclab_sim

    if not hasattr(isaaclab_sim, "PhysxCfg"):
        from isaaclab_physx.physics import PhysxCfg

        isaaclab_sim.PhysxCfg = PhysxCfg  # type: ignore[attr-defined]
        applied.append("isaaclab.sim.PhysxCfg <- isaaclab_physx.physics.PhysxCfg")

    # `isaaclab.utils` exports a *function* named `configclass` and also contains a *submodule*
    # of the same name. Python binds a submodule onto its parent package as an attribute when
    # imported, so any `import isaaclab.utils.configclass` anywhere -- including inside the
    # `isaaclab_physx` import just above -- silently replaces the function with the module.
    # `from isaaclab.utils import configclass` then yields a module and `@configclass` raises
    # "'module' object is not callable".
    #
    # Re-assert the function binding, after the imports above and before any task module loads.
    import isaaclab.utils as isaaclab_utils
    from isaaclab.utils.configclass import configclass as configclass_fn

    if isaaclab_utils.configclass is not configclass_fn:
        isaaclab_utils.configclass = configclass_fn  # type: ignore[assignment]
        applied.append("isaaclab.utils.configclass <- function (was shadowed by submodule)")

    # 3.0 renamed the backend-specific `SimulationCfg.physx` field to the backend-neutral
    # `physics`. The task config passes `physx=` at class-definition time, so the module cannot be
    # imported without accepting the old keyword. A genuine rename, so it is translated rather
    # than aliased. The PhysX value it builds is inert under Newton: the Newton setup replaces
    # `sim.physics` with a NewtonCfg and no PhysX manager is ever constructed.
    from isaaclab.sim import SimulationCfg

    if not getattr(SimulationCfg, "_play_physx_kwarg_shim", False):
        _orig_init = SimulationCfg.__init__

        def _init_with_physx_alias(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if "physx" in kwargs:
                kwargs.setdefault("physics", kwargs.pop("physx"))
            _orig_init(self, *args, **kwargs)

        SimulationCfg.__init__ = _init_with_physx_alias  # type: ignore[method-assign]
        SimulationCfg._play_physx_kwarg_shim = True  # type: ignore[attr-defined]
        applied.append("SimulationCfg(physx=...) -> SimulationCfg(physics=...)")

    return applied


__all__ = ["install_isaaclab3_compat"]
