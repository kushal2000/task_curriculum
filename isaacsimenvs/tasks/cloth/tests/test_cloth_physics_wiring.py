"""Every cloth physics setting of the VBD task (config/task.yaml) actually reaches the solver.

This file exists because the failure mode it guards has already happened here twice. A field can be
declared on `ClothCfg`, given a value, and read back correctly by anything that inspects the
config -- while never being passed to `SolverVBD` at all. It looks configured and does nothing.

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

The expected values are written out below, so a default that changes, or a value typed as a string
by YAML's exponent rules in an overlay, fails the test.
"""

from __future__ import annotations

import inspect

import pytest
import yaml


@pytest.fixture(scope="module")
def solver_kwargs():
    """The kwargs `SolverVBD.__init__` is actually called with, plus the model cfg."""
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from newton.solvers import SolverVBD

    from isaacsimenvs.tasks.cloth.backends.vbd.physics import build_physics
    from isaacsimenvs.tasks.cloth.config import vbd_cloth_env_cfg

    cfg = vbd_cloth_env_cfg()
    physics = build_physics(cfg.cloth, cfg.newton, num_envs=32)

    entry = next(e for e in physics.solver_cfg.entries if e.name == "cloth")
    return {
        "kwargs": NewtonManager._filter_solver_kwargs(SolverVBD, entry.solver_cfg),
        "model_cfg": physics.solver_cfg.model_cfg,
        "substeps": entry.substeps,
        "signature": set(inspect.signature(SolverVBD.__init__).parameters),
        "cloth": cfg.cloth,
        "physics": physics,
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


def test_tuned_contact_profile_reaches_coupled_solver(solver_kwargs):
    physics = solver_kwargs["physics"]
    cloth = solver_kwargs["cloth"]
    assert solver_kwargs["substeps"] == 1
    assert physics.num_substeps == 12
    assert cloth.thickness == pytest.approx(0.008)
    assert cloth.table_contact_mu == pytest.approx(0.6)
    assert physics.solver_cfg.proxies[0].collision_pipeline.enable_rigid_soft_full_surface_contact


@pytest.mark.parametrize("suffix", ["measured", "vbd_coarse", "vbd"])
def test_fabric_presets_fit_backend_config(suffix):
    from isaacsimenvs.tasks.cloth.config import apply_values, vbd_cloth_env_cfg
    from isaacsimenvs.tasks.cloth.utils.fabrics import cloth_params

    for fabric in ("denim", "towel", "chiffon", "cotton", "silk", "napkin", "big_napkin"):
        cfg = vbd_cloth_env_cfg()
        apply_values(cfg, {"cloth": cloth_params(f"{fabric}_{suffix}", plastic=True)})
        mesh = cfg.cloth.sheet_mesh()
        assert cfg.cloth.num_particles() == len(mesh.verts)
        assert len(set(mesh.corner_idx)) == 4
        assert set(mesh.corner_idx) <= set(mesh.moving_idx)
        assert mesh.half_mesh()[1]
        assert cfg.cloth.mesh_kind == ("grid" if suffix == "measured" else "irregular")


def _cloth_overlays():
    """``(file, cloth block)`` of every YAML that can set cloth values: run recipes and study
    recipes (under ``env:``) and DR blocks (env overlays themselves)."""
    from isaacsimenvs.conftest import REPO_ROOT

    configs = REPO_ROOT / "isaacsimenvs" / "tasks" / "cloth" / "config"
    recipes = [*(configs / "runs").glob("*.yaml"), *(REPO_ROOT / "experiments").rglob("*.yaml")]
    overlays = [*(configs / "dr").glob("*.yaml")]
    assert recipes and overlays, f"no recipes / DR blocks under {configs}"
    for path in recipes:
        yield path, ((yaml.safe_load(path.read_text()) or {}).get("env") or {}).get("cloth") or {}
    for path in overlays:
        yield path, (yaml.safe_load(path.read_text()) or {}).get("cloth") or {}


def test_yaml_numbers_are_numbers():
    """YAML 1.1 parses `1.0e2` as a STRING -- the exponent needs a sign. Caught this in review."""
    from isaacsimenvs.tasks.cloth.config import vbd_cloth_env_cfg

    defaults = vbd_cloth_env_cfg().cloth  # every cloth field (the rigid pool's are a subset)
    for path, block in _cloth_overlays():
        for key, value in block.items():
            default = getattr(defaults, key, None)
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                assert not isinstance(value, str), f"{path.name}: cloth.{key} = {value!r} parsed as str, not a number"
