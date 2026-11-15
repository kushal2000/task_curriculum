"""URDF generation and URDF->USD conversion options for the hinged-slat chains.

One definition of the chain URDF and of :data:`CONVERT_KWARGS`, shared by the PhysX envs
(``physx_envs.py``) and the analysis scripts that render or probe the chains.

The URDF text is **deterministic**: ``write_chain_urdf`` is pure given a
:class:`~rigid_cloth.ChainSpec` and the three physical arguments below, with no timestamps, no
tempdir paths and no dict-ordering dependence.
"""

from __future__ import annotations

from pathlib import Path

from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc

__all__ = [
    "CONVERT_KWARGS",
    "URDF_DEFAULTS",
    "chain_urdf_stem",
    "write_variant_urdf",
    "write_chain_urdf_for_run",
    "convert_variant",
]


#: The three physical parameters that enter the URDF text, and therefore the cache key.
#:
#: They live **here** rather than only on ``RigidClothCfg`` so the bake can read them without
#: importing the env config. That is not a style choice: ``RigidClothCfg`` inherits from
#: ``ClothEnvCfg`` -> ``PlayEnvCfg``, which constructs a ``SimulationCfg(physics=...)`` that exists
#: only in Isaac Lab 3.0. The bake runs under ``.venv/isaacsim``, where that keyword does not exist,
#: so importing the config chain there raises ``TypeError`` before a single chain is written -- and
#: Kit's shutdown path exits 0, so the bake reported success and baked nothing.
#:
#: ``RigidClothCfg`` takes its defaults from this dict, pinned by a test, so there is still exactly
#: one definition and a run at defaults is guaranteed to hit the baked key.
#: ``joint_damping`` is 0.0 because the URDF's ``<dynamics>`` does **not** survive the importer:
#: whatever is written there, the Newton model reports ``joint_damping = 0.0`` on every slat hinge.
#: Pinning it at zero keeps the URDF from claiming a damping the solver never receives; the damping
#: that actually applies is ``RigidClothCfg.joint_damping``, an Isaac Lab actuator gain.
URDF_DEFAULTS: dict = {
    "density": 2.0,
    "joint_damping": 0.0,
    "joint_limit": None,
}


#: Conversion options, shared by the bake and the run. Every one is keyed into the cache hash
#: (``_KEYED_KWARGS`` plus ``repr(joint_drive)``), so changing any of them invalidates the bake --
#: which is the intended behaviour, not a nuisance.
#:
#: ``self_collision=True`` is **load-bearing and not a default**. The fold is stopped by slat ``i``
#: landing on slat ``i+2``; with self-collision off the two plies pass straight through each other
#: and the chain folds to zero thickness. That failure looks like a policy that has learned to fold
#: perfectly, which is the worst way for it to present.
#:
#: ``replace_cylinders_with_capsules=False`` departs from the rigid tool pool, which sets it True.
#: A slat cylinder lies along ``y`` and spans the sheet's full 100 mm width; capping it with
#: hemispheres would round away the sheet's two side edges and shorten its flat contact with the
#: table by one radius at each end. The tool pool wants capsules because a capsule is cheaper and
#: its handles are round anyway; here the cylinder IS the approximation being measured.
#:
#: ``fix_base=False``: the sheet is free, like the tool it replaces. The drape probes in
#: ``scripts/analysis/rigid_cloth_probe.py`` weld the root, which is a measurement fixture, not this.
CONVERT_KWARGS: dict = {
    "fix_base": False,
    "self_collision": True,
    "replace_cylinders_with_capsules": False,
    "joint_drive": None,
}


def chain_urdf_stem(variant: str) -> str:
    """File stem for a variant's URDF.

    Used as the USD sub-directory name by ``_convert_urdf_to_usd``, so it has to be a valid path
    component and stable across processes.
    """
    return f"rigid_cloth_{variant.replace('-', '_')}"


def write_variant_urdf(
    variant: str,
    out_dir: str | Path,
    *,
    density: float,
    joint_damping: float,
    joint_limit: float | None = None,
) -> Path:
    """Write one variant's URDF into ``out_dir`` and return its path.

    The three physical arguments are the only run-time freedom: they change the URDF text and
    therefore the cache key, so a run that changes ``density`` needs a re-bake. That is deliberate
    -- a silently reused USD with the wrong masses would be indistinguishable from a physics bug.
    """
    spec = rc.variant(variant)
    out_path = Path(out_dir) / f"{chain_urdf_stem(variant)}.urdf"
    return rc.write_chain_urdf(
        spec,
        out_path,
        density=density,
        joint_limit=joint_limit,
        joint_damping=joint_damping,
    )


def write_chain_urdf_for_run(
    variant: str,
    out_dir: str | Path,
    *,
    density: float | None = None,
    joint_limit: float | None = None,
) -> Path:
    """The ONE way a chain URDF is written -- by the bake and by the env alike.

    Exists because the alternative already failed. The bake wrote ``URDF_DEFAULTS["joint_damping"]``
    while the env wrote ``RigidClothCfg.joint_damping``, which had become the *actuator* gain; the two
    URDFs differed by one number, the cache key is a hash of the URDF text, and every single run died
    with a cache miss before building anything.

    So ``joint_damping`` is **not a parameter here**. It cannot be passed, which is the only way to
    guarantee the two callers agree: the URDF's ``<dynamics>`` does not survive the importer (the
    Newton model reports 0.0 on every slat hinge regardless), so there is nothing to express, and the
    damping that acts is an Isaac Lab actuator gain set on the articulation.

    ``density`` and ``joint_limit`` remain overridable because they genuinely change the asset -- and
    a run that overrides either is a different cache entry and needs its own bake, which it will
    announce as a miss naming the variant.
    """
    return write_variant_urdf(
        variant,
        out_dir,
        density=URDF_DEFAULTS["density"] if density is None else density,
        joint_damping=URDF_DEFAULTS["joint_damping"],
        joint_limit=URDF_DEFAULTS["joint_limit"] if joint_limit is None else joint_limit,
    )


def convert_variant(urdf_path: str | Path, usd_work_dir: str | Path) -> str:
    """URDF -> USD through whatever ``scene_utils`` currently provides.

    Deliberately indirect. Under Isaac Sim that attribute is the real Kit converter; under the
    kit-less Newton stack ``usd_cache.install_reader`` has replaced it with a cache lookup. Calling
    the module attribute rather than ``UrdfConverter`` directly is what makes the chain obey the
    same bake-once discipline as every other asset in the scene -- and what makes a missing bake a
    clear cache-miss error instead of an unexplained Kit failure.
    """
    from isaacsimenvs.newton import compat

    scene_utils = compat.play_module("scene_utils")
    return scene_utils._convert_urdf_to_usd(
        str(urdf_path), Path(usd_work_dir), **CONVERT_KWARGS
    )
