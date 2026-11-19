"""Content-addressed URDF -> USD cache, so the kit-less Newton stack can load baked assets.

``robots.usd.convert_urdf_to_usd`` drives ``isaaclab.sim.converters.UrdfConverter``, which needs
the Kit URDF-importer extension. Newton runs kit-less and has no Kit, so conversion cannot happen
there at all. The split is therefore: **bake once under ``.venv_isaacsim``, read under
``.venv_isaaclab3``.**

There is a second, independent reason to do it this way even where Kit *is* available. Both
backends then consume byte-identical geometry, converted by one importer. Without that, a
difference between the PhysX and Newton numbers is confounded with a difference in how their USDs
were produced, and is unattributable.

Populate (Isaac Sim venv), once per URDF and self-collision setting::

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.tasks.cloth.backends.vbd.newton_support.usd_cache \
        --populate env.robot_self_collision=false

Consume: :func:`install_reader` runs from ``backends/vbd/newton_setup.py``.

The key is a hash of URDF *text* plus the conversion options, never a path, so a URDF copied or
generated elsewhere still hits.

Adapted from ``simtoolreal_newton.upstream`` in github.com/kushal2000/isaac_newton @ beb9efb.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import importlib

REPO_ROOT = Path(__file__).resolve().parents[6]
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
    robot_usd = importlib.import_module("isaacsimenvs.robots.usd")
    if getattr(robot_usd, "_usd_cache_writer_installed", False):
        return

    original = robot_usd.convert_urdf_to_usd
    cache_dir.mkdir(parents=True, exist_ok=True)
    stored = 0

    def _convert_and_cache(asset_path, usd_work_dir, **kwargs):
        nonlocal stored
        usd_path = original(asset_path, usd_work_dir, **kwargs)

        key = cache_key(Path(asset_path).read_text(), kwargs)
        dest = _entry_dir(cache_dir, key)
        if not dest.exists():
            # Copy the whole directory: `bake_usd` later writes materials and meshes next to the
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

    robot_usd.convert_urdf_to_usd = _convert_and_cache
    robot_usd._usd_cache_writer_installed = True


def install_reader(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Replace conversion with a cache lookup. No Isaac Sim required; idempotent.

    **Raises on a miss rather than falling back to conversion.** A fallback here would either
    crash confusingly (no Kit) or, worse under a Kit-hosted run, silently convert with a
    different importer and quietly break the like-for-like comparison the cache exists to
    guarantee.
    """
    robot_usd = importlib.import_module("isaacsimenvs.robots.usd")
    if getattr(robot_usd, "_usd_cache_reader_installed", False):
        return

    if not cache_dir.is_dir():
        raise RuntimeError(
            f"USD cache missing at {cache_dir}. Populate it first:\n"
            "  OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python "
            "-m isaacsimenvs.tasks.cloth.backends.vbd.newton_support.usd_cache --populate"
        )

    def _lookup(asset_path, usd_work_dir, **kwargs):
        key = cache_key(Path(asset_path).read_text(), kwargs)
        entry = _entry_dir(cache_dir, key)
        manifest_path = entry / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(
                f"USD cache miss for {Path(asset_path).name} (key {key}).\n"
                "The bake and this run disagree on URDF content or conversion options; "
                "re-populate with this run's URDF and self-collision setting (see the module docstring)."
            )
        manifest = json.loads(manifest_path.read_text())

        # Copy out of the cache into this run's work dir, so `bake_usd` can write alongside it
        # without mutating the shared cache.
        dest_dir = Path(usd_work_dir) / Path(asset_path).stem
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(entry, dest_dir)
        (dest_dir / _MANIFEST_NAME).unlink(missing_ok=True)
        return str(dest_dir / manifest["usd_name"])

    robot_usd.convert_urdf_to_usd = _lookup
    robot_usd._usd_cache_reader_installed = True


def main() -> None:
    """Populate the cache by building the PhysX cloth task's scene once under Isaac Sim: the robot
    and table conversions the Newton cloth task looks up. Extra arguments are config overrides."""
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Bake the cloth task's robot and table into the USD cache.")
    parser.add_argument("--populate", action="store_true", required=True)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--num_envs", type=int, default=4, help="only affects scene build cost")
    parser.add_argument("--task", default="Isaacsimenvs-RigidCloth-Direct-v0")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    install_writer(args.cache_dir)

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env = gym.make(args.task, cfg=env_cfg)
        entries = len([p for p in args.cache_dir.iterdir() if p.is_dir()])
        print(f"\n[usd_cache] {entries} entries in {args.cache_dir}")
        env.close()

    run()

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
