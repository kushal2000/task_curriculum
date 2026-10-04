"""Content-addressed URDF -> USD cache, so the kit-less Newton stack can load baked assets.

``scene_utils._convert_urdf_to_usd`` drives ``isaaclab.sim.converters.UrdfConverter``, which needs
the Kit URDF-importer extension. Newton runs kit-less and has no Kit, so conversion cannot happen
there at all. The split is therefore: **bake once under ``.venv_isaacsim``, read under
``.venv_isaaclab3``.**

There is a second, independent reason to do it this way even where Kit *is* available. Both
backends then consume byte-identical geometry, converted by one importer. Without that, a
difference between the PhysX and Newton numbers is confounded with a difference in how their USDs
were produced, and is unattributable.

Populate (Isaac Sim venv)::

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.newton.usd_cache \
        --populate --num_assets_per_type 4

Consume: :func:`install_reader` runs automatically from ``PlayNewtonEnv``.

The key is a hash of URDF *text* plus the conversion options, never a path -- the procedural tool
URDFs are written to a fresh ``tempfile.mkdtemp`` on every launch, so paths are useless as
identity while content is stable. Verified stable across the two interpreters: the 48-URDF pool
for ``num_assets_per_type=4`` hashes identically under numpy 1.26 (3.11) and numpy 2.5 (3.12),
because ``generate_objects`` seeds the legacy ``RandomState`` stream, which is compatibility-
guaranteed across numpy versions.

Adapted from ``simtoolreal_newton.upstream`` in github.com/kushal2000/isaac_newton @ beb9efb.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from isaacsimenvs.newton import compat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / ".usd_cache"

_MANIFEST_NAME = "manifest.json"

#: Conversion options that change the produced USD. Anything not listed here must not affect
#: output, or two different assets would collide on one key.
_KEYED_KWARGS = ("fix_base", "self_collision", "replace_cylinders_with_capsules")


def cache_key(urdf_text: str, kwargs: dict) -> str:
    """Content address for one conversion: URDF text plus the options that affect its output."""
    relevant = {k: kwargs.get(k) for k in _KEYED_KWARGS}
    # joint_drive is a config object; fold in its repr so differing drives key differently.
    relevant["joint_drive"] = repr(kwargs.get("joint_drive"))
    payload = json.dumps(relevant, sort_keys=True) + "\n" + urdf_text
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _entry_dir(cache_dir: Path, key: str) -> Path:
    return cache_dir / key


def install_writer(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Record every conversion into the cache. Run under the Isaac Sim venv; idempotent.

    Conversion still happens normally -- this only copies the result aside.
    """
    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_usd_cache_writer_installed", False):
        return

    original = scene_utils._convert_urdf_to_usd
    cache_dir.mkdir(parents=True, exist_ok=True)
    stored = 0

    def _convert_and_cache(asset_path, usd_work_dir, **kwargs):
        nonlocal stored
        usd_path = original(asset_path, usd_work_dir, **kwargs)

        key = cache_key(Path(asset_path).read_text(), kwargs)
        dest = _entry_dir(cache_dir, key)
        if not dest.exists():
            # Copy the whole directory: `_bake_usd` later writes materials and meshes next to the
            # .usd, so a lone file is not a complete entry.
            shutil.copytree(Path(usd_path).parent, dest)
            (dest / _MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "key": key,
                        "usd_name": Path(usd_path).name,
                        "source_urdf": Path(asset_path).name,
                        "kwargs": {k: repr(v) for k, v in kwargs.items()},
                    },
                    indent=2,
                )
                + "\n"
            )
            stored += 1
        return usd_path

    scene_utils._convert_urdf_to_usd = _convert_and_cache
    scene_utils._usd_cache_writer_installed = True


def install_reader(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Replace conversion with a cache lookup. No Isaac Sim required; idempotent.

    **Raises on a miss rather than falling back to conversion.** A fallback here would either
    crash confusingly (no Kit) or, worse under a Kit-hosted run, silently convert with a
    different importer and quietly break the like-for-like comparison the cache exists to
    guarantee.
    """
    scene_utils = compat.play_module("scene_utils")
    if getattr(scene_utils, "_usd_cache_reader_installed", False):
        return

    if not cache_dir.is_dir():
        raise RuntimeError(
            f"USD cache missing at {cache_dir}. Populate it first:\n"
            "  OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python "
            "-m isaacsimenvs.newton.usd_cache --populate"
        )

    def _lookup(asset_path, usd_work_dir, **kwargs):
        key = cache_key(Path(asset_path).read_text(), kwargs)
        entry = _entry_dir(cache_dir, key)
        manifest_path = entry / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(
                f"USD cache miss for {Path(asset_path).name} (key {key}).\n"
                "The bake and this run disagree on URDF content or conversion options. "
                "Re-populate with the same --num_assets_per_type and --handle_head_types: the "
                "object pool is shuffled over its full length, so a different count is a "
                "different set of objects, not a subset."
            )
        manifest = json.loads(manifest_path.read_text())

        # Copy out of the cache into this run's work dir, so `_bake_usd` can write alongside it
        # without mutating the shared cache.
        dest_dir = Path(usd_work_dir) / Path(asset_path).stem
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(entry, dest_dir)
        (dest_dir / _MANIFEST_NAME).unlink(missing_ok=True)
        return str(dest_dir / manifest["usd_name"])

    scene_utils._convert_urdf_to_usd = _lookup
    scene_utils._usd_cache_reader_installed = True


def bake_rigid_cloth(variants=None, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> list[str]:
    """Convert the hinged-slat chain URDFs so a ``RigidCloth`` run can find them.

    These do not come from a scene build. ``RigidClothEnv`` writes its chain's URDF at construction
    time and converts it through ``scene_utils._convert_urdf_to_usd`` -- the same call the object
    pool uses, so the same cache -- which means the entry must already exist under the kit-less
    stack. Baking them here rather than in a script of their own keeps one definition of the cache
    key and one command to populate it.

    The three URDF parameters come from :data:`rigid_cloth_assets.URDF_DEFAULTS`, which is also
    where ``RigidClothCfg`` takes its defaults, because the URDF text is the cache key: a run that
    overrides any of them is a different asset and needs its own bake. It will say so, as a cache
    miss naming the variant.

    Deliberately does **not** import ``RigidClothCfg``. That config inherits from ``PlayEnvCfg``,
    which builds a ``SimulationCfg(physics=...)`` that only Isaac Lab 3.0 accepts -- and this
    function runs under ``.venv_isaacsim``, where the import raises before anything is baked.
    """
    import tempfile

    from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc
    from isaacsimenvs.tasks.cloth.utils import rigid_cloth_assets as assets

    install_writer(cache_dir)
    names = list(rc.VARIANTS) if variants is None else list(variants)
    unknown = [n for n in names if n not in rc.VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; known: {sorted(rc.VARIANTS)}")

    work = Path(tempfile.mkdtemp(prefix="rigid_cloth_bake_"))
    done = []
    for name in names:
        urdf = assets.write_chain_urdf_for_run(name, work)
        usd = assets.convert_variant(urdf, work)
        key = cache_key(Path(urdf).read_text(), assets.CONVERT_KWARGS)
        print(f"[usd_cache] baked {name} -> {key}  ({Path(usd).name})", flush=True)
        done.append(name)
    return done


def main() -> None:
    """Populate the cache by building the Play scene once under Isaac Sim."""
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Bake Play assets into the USD cache.")
    parser.add_argument("--populate", action="store_true", required=True)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--num_envs", type=int, default=4, help="only affects scene build cost")
    parser.add_argument(
        "--num_assets_per_type",
        type=int,
        default=4,
        help="Must match what the Newton run will use -- the pool is shuffled over its whole "
        "length, so a different count yields a different set of objects.",
    )
    parser.add_argument("--task", default="Isaacsimenvs-Play-Direct-v0")
    parser.add_argument(
        "--rigid_cloth_variants",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Also bake the hinged-slat chains of `rigid_cloth.VARIANTS`; no names means all five. "
        "These are NOT part of any scene build -- a RigidCloth run generates its chain URDF and "
        "converts it through the same cache, so the entry has to exist. Baking them all in one "
        "launch costs one Kit start instead of five.",
    )
    parser.add_argument(
        "--skip_scene_build",
        action="store_true",
        help="Do not build the Play scene; only bake --rigid_cloth_variants. The scene build is "
        "what populates the procedural tool pool, and re-running it with a different "
        "--num_assets_per_type writes a DIFFERENT set of objects into the cache. When the pool is "
        "already baked, skipping it keeps the cache exactly as it is and saves the scene build.",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    install_writer(args.cache_dir)

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = args.num_assets_per_type
        env = gym.make(args.task, cfg=env_cfg)
        entries = len([p for p in args.cache_dir.iterdir() if p.is_dir()])
        print(f"\n[usd_cache] {entries} entries in {args.cache_dir}")
        env.close()

    if args.skip_scene_build:
        if args.rigid_cloth_variants is None:
            parser.error("--skip_scene_build leaves nothing to do without --rigid_cloth_variants")
        print("[usd_cache] skipping the scene build; baking chains only", flush=True)
    else:
        run()

    if args.rigid_cloth_variants is not None:
        bake_rigid_cloth(args.rigid_cloth_variants or None, cache_dir=args.cache_dir)

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
