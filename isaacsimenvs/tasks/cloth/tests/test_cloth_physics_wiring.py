"""Every cloth physics setting in `Cloth.yaml` actually reaches the solver.

This file exists because the failure mode it guards has already happened here twice. A field can be
declared on `ClothCfg`, pinned in the task YAML, and read back correctly by anything that inspects
the config -- while never being passed to `SolverVBD` at all. It looks configured and does nothing.

Three distinct ways that happens, all covered below:

  * **Never wired.** `rigid_body_contact_buffer_size` was declared and pinned for a while before
    `build_physics` passed it, and `soft_contact_kd` was silently inheriting `NewtonModelCfg`'s
    default because that call only forwarded `ke` and `mu`.
  * **Wrong name.** `particle_conservative_bound_relaxation` and `rigid_avbd_beta` are NOT declared
    fields on `VBDSolverCfg`; they reach the solver only because `_filter_solver_kwargs` filters
    `to_dict()` against `SolverVBD.__init__` and isaaclab's `class_to_dict` walks `obj.__dict__`.
    A rename upstream drops them silently.
  * **Gated by another setting.** `rigid_contact_k_start` is discarded outright when
    `rigid_avbd_beta` is 0: Newton computes `-1.0 if linear_beta == 0.0 else k_start`
    (solver_vbd.py:677) and the kernel then reads `k_floor = avg_ke` (rigid_vbd_kernels.py:3395).
    It sat at 1.0e2 doing nothing for the whole cable-inherited history of this task.

The YAML is loaded here rather than relying on `ClothCfg` defaults so that a value pinned in the
task file but misspelled, or typed as a string by YAML's exponent rules, fails the test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

_YAML = Path(__file__).resolve().parents[1] / "isaacsimenvs" / "cfg" / "task" / "Cloth.yaml"


@pytest.fixture(scope="module")
def solver_kwargs():
    """The kwargs `SolverVBD.__init__` is actually called with, plus the model cfg."""
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from newton.solvers import SolverVBD

    from isaacsimenvs.tasks.cloth.cloth_env_cfg import ClothEnvCfg

    cfg = ClothEnvCfg()
    cfg.from_dict(yaml.safe_load(_YAML.read_text()))
    physics = cfg.cloth.build_physics(cfg.newton, num_envs=32)

    entry = next(e for e in physics.solver_cfg.entries if e.name == "cloth")
    return {
        "kwargs": NewtonManager._filter_solver_kwargs(SolverVBD, entry.solver_cfg),
        "model_cfg": physics.solver_cfg.model_cfg,
        "substeps": entry.substeps,
        "signature": set(inspect.signature(SolverVBD.__init__).parameters),
        "cloth": cfg.cloth,
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("iterations", 12),
        # Self-contact. Off by default in `VBDSolverCfg`, and `build_physics` did not set it, so a
        # folded flap passed through the half beneath it.
        ("particle_enable_self_contact", True),
        ("particle_self_contact_radius", 0.002),
        ("particle_self_contact_margin", 0.003),
        ("particle_collision_detection_interval", -1),
        # Undeclared on VBDSolverCfg -- reaches the solver only via the __dict__ pass-through.
        ("particle_conservative_bound_relaxation", 0.42),
        ("rigid_avbd_beta", 5.0e5),
        ("rigid_contact_k_start", 1.0e2),
        ("rigid_body_contact_buffer_size", 16384),
        ("rigid_body_particle_contact_buffer_size", 65536),
    ],
)
def test_setting_reaches_solver(solver_kwargs, name, expected):
    assert name in solver_kwargs["signature"], (
        f"{name!r} is not a SolverVBD.__init__ parameter in this Newton version. "
        "_filter_solver_kwargs drops it silently, so the setting is inert."
    )
    assert solver_kwargs["kwargs"].get(name) == pytest.approx(expected)


def test_avbd_ramping_is_enabled_so_k_start_is_not_discarded(solver_kwargs):
    """`rigid_contact_k_start` is a no-op unless the linear beta is > 0.

    Newton: `rigid_contact_k_start_value = -1.0 if linear_beta == 0.0 else k_start`, and the kernel
    reads `k_floor = avg_ke if k_start < 0.0 else min(k_start, avg_ke)`. With beta 0 every contact
    starts at full material stiffness and the seed is thrown away.
    """
    kw = solver_kwargs["kwargs"]
    beta = kw["rigid_avbd_beta"]
    assert beta > 0.0, "AVBD contact ramping disabled; rigid_contact_k_start is dead config"
    effective = -1.0 if beta == 0.0 else kw["rigid_contact_k_start"]
    assert effective > 0.0


def test_soft_contact_triplet_is_forwarded(solver_kwargs):
    """`soft_contact_kd` was inheriting NewtonModelCfg's 1.0e-2 because it was never passed."""
    m = solver_kwargs["model_cfg"]
    assert (m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu) == (500.0, 5.0e-3, 0.25)


def test_cloth_substeps_reach_the_coupler_entry(solver_kwargs):
    """Substeps live on the CouplerEntryCfg, not on the solver cfg, so they need their own check."""
    assert solver_kwargs["substeps"] == 6


def test_thickness_and_resolution_stay_on_the_guard_boundary(solver_kwargs):
    """16 mm particles need >= 16.7 mm spacing; resolution 7 is the finest grid that admits it.

    These two are one decision, and the guard only runs in `__post_init__` -- it does NOT re-run
    after the YAML overlay is applied. So a YAML that raised `resolution` without lowering
    `thickness` would produce a sheet that self-collides at rest, with no error.
    """
    c = solver_kwargs["cloth"]
    assert c.thickness <= c.spacing + 1e-9
    assert c.resolution <= c.max_resolution_for_thickness


def test_yaml_numbers_are_numbers():
    """YAML 1.1 parses `1.0e2` as a STRING -- the exponent needs a sign. Caught this in review."""
    block = yaml.safe_load(_YAML.read_text())["cloth"]
    for key, value in block.items():
        if key in ("proxy_mode",):
            continue
        assert not isinstance(value, str), f"cloth.{key} = {value!r} parsed as str, not a number"
