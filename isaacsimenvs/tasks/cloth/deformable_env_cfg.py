"""Config of the deformable (VBD) cloth env on Newton (``deformable_env.ClothEnv``), Isaac Lab 3.

Sections: the Newton solver block and ``PlayNewtonEnvCfg`` (the Play task's config on Newton:
MJWarp for the robot, Newton's model/solver settings); ``ClothEnvCfg`` (that + the shared fold task
block ``ClothCfg`` from ``fold_task``); and the real-fabric presets (``isaacsimenvs/cfg/fabric/``),
which also run as a command that prints a preset's hydra overrides::

    .venv/isaaclab3/bin/python -m isaacsimenvs.tasks.cloth.deformable_env_cfg cotton_measured [--plastic]
"""

from __future__ import annotations

import sys
from pathlib import Path

# IMPORT ORDER IS LOAD-BEARING. `play_env_cfg` calls `SimulationCfg(physx=...)` at class-definition
# time, and Isaac Lab 3.0 renamed that keyword to `physics`, so importing it before the compat
# shim is installed raises at import. The shim also re-asserts `isaaclab.utils.configclass` as a
# function after `isaaclab_physx` shadows it with its same-named submodule.
from isaacsimenvs.newton import compat  # isort: skip

compat.install_isaaclab3_compat()

import yaml  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.fold_task import ClothCfg  # noqa: E402
from isaacsimenvs.tasks.play.play_env_cfg import PlayEnvCfg  # noqa: E402

__all__ = ['ClothCfg', 'ClothEnvCfg', 'NewtonSolverCfg', 'PlayNewtonEnvCfg']


# ==================================================================================================
# Play on Newton -- PlayNewtonEnvCfg — the Play task on Isaac Lab 3.0 + Newton/MJWarp: the base of ``ClothEnvCfg``.
# (was newton_base_env_cfg.py)
# ==================================================================================================






@configclass
class NewtonSolverCfg:
    """MJWarp / Newton solver settings.

    Values are the reference port's (github.com/kushal2000/isaac_newton @ beb9efb), which reached
    parity with the PhysX baseline on this exact task. **Almost none of them are library
    defaults**, and the defaults are not merely untuned -- several fail *silently*, dropping work
    rather than raising, which is why each carries its measured justification here.
    """

    # --- The decisive one -------------------------------------------------------------------
    max_triangle_pairs: int | None = None
    """Narrow-phase triangle-pair budget. ``None`` sizes it from the environment count.

    The library default is ``1_000_000`` and it is a **global** budget, not per-world. This robot
    carries 34 mesh collision prims with self-collision enabled, so demand grows linearly with
    environment count: it crosses 1e6 somewhere between 16 and 32 envs and reaches ~1.7M at 64.
    Past the cap the narrow phase **silently drops pairs** -- fingertip/tool contacts simply go
    missing and the grasp cannot be held.

    Measured in the reference port at 64 envs, changing only this value:
    ``1e6 -> 0.12 goals/episode``, ``8e6 -> 14.28``. A 114x effect from a number that appears in
    no config file.

    It also explains why that project spent a day misattributing the backend: every Newton run it
    had ever done used 64-1024 envs, i.e. was entirely inside the saturated regime, so there was
    no working configuration to A/B against and the solver looked uniformly broken rather than
    broken above a threshold.
    """

    per_env_triangle_pairs: int = 65_536
    """Headroom per env when sizing automatically -- about 2.5x the ~26.6k/env measured demand.

    The buffer is a GPU allocation so it is not free, but under-sizing fails silently and
    catastrophically while over-sizing costs memory, which is visible and diagnosable.
    """

    # --- Constraint budgets, same failure mode ------------------------------------------------
    njmax: int = 8192
    """Max constraint rows per world. The library default of 300 is sized for a Franka/Allegro
    lifting a simple object; this scene (22-DoF hand + tool + table) overflows it continuously
    (``nefc overflow - please increase njmax``). Overflow is not benign -- constraints are
    dropped and positions go wrong. The reference ran at 2400 for a long time, still overflowing
    to 2878, and raising it to 8192 moved a "matching" score from 15.97 to 18.61."""

    nconmax: int = 2048
    """Max contact points per world; same failure mode. The reference ran 800 while the solver's
    own message asked for 1081."""

    # --- Contact model, from the task authors' own MuJoCo scene -------------------------------
    cone: str = "elliptic"
    """Friction cone. MuJoCo's default is ``pyramidal``, which approximates the friction limit
    surface with a box and under combined slip lets a held tool rotate more readily. The
    SimToolReal authors' own MuJoCo model of this robot uses elliptic."""

    impratio: float = 10.0
    """Frictional-to-normal constraint impedance. The authors use 10, not MuJoCo's default 1; a
    higher ratio stiffens frictional constraints and stops a grasped tool creeping."""

    condim: int = 6
    """Contact dimensionality on the grasping shapes. The tool and fingertip assets already carry
    torsional and rolling friction coefficients, but MuJoCo ignores them below condim 4 and 6
    respectively. Set before the solver compiles -- MuJoCo Warp sizes its constraint arrays at
    compile time, so writing it afterwards produces NaNs.

    Kept on fidelity grounds (the authors' scene specifies it), **not** performance: measured
    cleanly post-fix it is neutral (19.375 +/- 2.134 at condim 6 against 20.359 +/- 1.918 at
    condim 3, z = 0.34), and an earlier claim that torsional friction was the missing physics was
    retracted."""

    # --- Pipeline -----------------------------------------------------------------------------
    use_mujoco_contacts: bool = False
    """Use Newton's own collision pipeline rather than MuJoCo's. Not a performance preference:
    under MuJoCo's pipeline ``separate_worlds=True`` builds the MuJoCo model from world 0 only, so
    per-env collision *meshes* are **silently ignored** -- every world simulates world 0's mesh
    while reporting its own mass and inertia. That is a physically nonexistent hybrid that passes
    a "do the environments differ?" check. Newton's pipeline reads per-world shape data correctly,
    and on a mesh scene is also better conditioned (0.43 mm penetration against 6.5-9.2 mm) and
    faster."""

    solver: str = "newton"
    integrator: str = "implicitfast"
    iterations: int = 100
    ls_iterations: int = 15
    ccd_iterations: int = 35
    update_data_interval: int = 2
    rigid_contact_max: int = 4_000_000
    num_substeps: int = 2
    collision_decimation: int = 0
    """Re-collide every N substeps; 0 = once per tick. Coupled to :attr:`num_substeps` -- raising
    substeps without also re-colliding more often leaves the solver integrating several steps
    against one stale collision update."""

    use_cuda_graph: bool = True
    disable_sensors: bool = True
    """Required for ``deterministic_mode="run_to_run"``, and safe here: every observation field is
    read through Isaac Lab's articulation and rigid-body state, never from MuJoCo sensors."""

    deterministic_mode: str = "not_guaranteed"

    def resolve_max_triangle_pairs(self, num_envs: int) -> int:
        """Triangle-pair budget for ``num_envs``, honouring an explicit override."""
        if self.max_triangle_pairs is not None:
            return int(self.max_triangle_pairs)
        return max(1_000_000, int(num_envs) * self.per_env_triangle_pairs)

    def build(self, num_envs: int):
        """Return an ``isaaclab_newton`` ``NewtonCfg``.

        ``num_envs`` is load-bearing, not incidental -- it sizes the triangle-pair budget.
        """
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            NewtonShapeCfg,
        )

        return NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver=self.solver,
                integrator=self.integrator,
                njmax=self.njmax,
                nconmax=self.nconmax,
                impratio=self.impratio,
                cone=self.cone,
                update_data_interval=self.update_data_interval,
                iterations=self.iterations,
                ls_iterations=self.ls_iterations,
                use_mujoco_contacts=self.use_mujoco_contacts,
                ccd_iterations=self.ccd_iterations,
                disable_sensors=self.disable_sensors,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=self.rigid_contact_max,
                max_triangle_pairs=self.resolve_max_triangle_pairs(num_envs),
            ),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=self.num_substeps,
            collision_decimation=self.collision_decimation,
            use_cuda_graph=self.use_cuda_graph,
            deterministic_mode=self.deterministic_mode,
        )


@configclass
class PlayNewtonEnvCfg(PlayEnvCfg):
    """Play, constructed on Newton/MJWarp instead of PhysX."""

    newton: NewtonSolverCfg = NewtonSolverCfg()

    use_gravcomp: bool = True
    """Express the robot's per-body ``disableGravity`` as MuJoCo gravity compensation, leaving
    global gravity at -9.81 so the object falls natively.

    The alternative -- zeroing global gravity and re-applying the object's as an explicit wrench --
    is what the reference port did while it believed Newton had no per-body gravity. It does:
    ``mjc:gravcomp`` is a builder attribute applied as ``(1 - gravcomp) * g``. The reconstruction
    route was the origin of two separate defects (a wrench applied in the body frame, and a
    compensating torque on the grasped object), so it is off by default."""

    meshify: bool = False
    """Represent tool collision geometry as convex meshes instead of box/capsule primitives.

    Needed for per-env asset *variants* under Newton -- ``SolverMuJoCo`` requires homogeneous
    shape *types* across worlds, and the procedural tools mix boxes and capsules. Off by default
    because the single-object protocol does not need it."""


# ==================================================================================================
# The deformable cloth env -- ClothEnvCfg — fold half a cloth onto the other half.
# (was cloth_env_cfg.py)
# ==================================================================================================

# IMPORT ORDER IS LOAD-BEARING, as in `cable_env_cfg`: importing `isaaclab_physx` binds
# `isaaclab.utils.configclass` to the same-named *submodule*, shadowing the decorator. Importing
# PlayNewtonEnvCfg first runs the compat shim that re-asserts it.





@configclass
class ClothEnvCfg(PlayNewtonEnvCfg):
    """Play task with a folding cloth in place of the rigid tool."""

    cloth: ClothCfg = ClothCfg()


# ==================================================================================================
# Real-fabric presets -- VBD cloth presets for real fabrics: ``isaacsimenvs/cfg/fabric/<name>.yaml``.
# (was utils/fabrics.py)
# ==================================================================================================

FABRIC_DIR = Path(__file__).resolve().parents[2] / "cfg" / "fabric"

#: Display order: the env default first, then roughly lightest to heaviest.
ORDER = ("base", "chiffon", "silk", "cotton", "napkin", "towel", "denim")


def names() -> list[str]:
    """Every preset, ``ORDER`` first, then any other file alphabetically."""
    found = {p.stem for p in FABRIC_DIR.glob("*.yaml")}
    return [n for n in ORDER if n in found] + sorted(found - set(ORDER))


def load(name: str) -> dict:
    """The preset's whole file."""
    path = FABRIC_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"no fabric preset {name!r} in {FABRIC_DIR} (have {names()})")
    with open(path) as f:
        return yaml.safe_load(f)


def cloth_params(name: str, plastic: bool = False) -> dict[str, float]:
    """``{ClothCfg field: value}``; with ``plastic`` also ``plastic_yield_curvature``."""
    d = load(name)
    out = {k: float(v) for k, v in d["cloth"].items()}
    if plastic:
        out["plastic_yield_curvature"] = float(d.get("plastic_yield_curvature", 0.0))
        if float(d.get("plastic_set_curvature", 0.0)) > 0.0:  # two-stage presets only
            out["plastic_set_curvature"] = float(d["plastic_set_curvature"])
    return out


def overrides(name: str, plastic: bool = False) -> list[str]:
    """Hydra overrides ``env.cloth.<key>=<value>`` for the preset."""
    return [f"env.cloth.{k}={v!r}" for k, v in cloth_params(name, plastic).items()]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plastic"]
    if len(args) != 1:
        sys.exit(f"usage: python -m isaacsimenvs.tasks.cloth.deformable_env_cfg NAME [--plastic]   (NAME in {names()})")
    print(" ".join(overrides(args[0], plastic="--plastic" in sys.argv[1:])))
