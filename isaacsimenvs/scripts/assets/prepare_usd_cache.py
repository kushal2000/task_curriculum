"""Bake assets under Isaac Sim for the VBD backend's kit-less USD cache.

Run from the repository root with:
    .venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache --populate

For the default VBD robot/table assets, add:
    --task Isaacsimenvs-RigidBodyCloth-Direct-v0
    env.assets.table_urdf=assets/urdf/table_narrow.urdf
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support.usd_cache import (
    _MANIFEST_NAME,
    DEFAULT_CACHE_DIR,
    _entry_dir,
    cache_key,
)


def install_writer(cache_dir: Path = DEFAULT_CACHE_DIR, *, cloth: bool = False) -> None:
    """Record every conversion into the cache. Run under the Isaac Sim venv; idempotent.

    Conversion still happens normally -- this only copies the result aside.
    Patch the scene's binding: rigid cloth imports its converter by name from
    backends.assets, whereas Play owns its converter in scene_utils.
    """
    if cloth:
        from isaacsimenvs.tasks.cloth.backends.rigid import scene as target

        name = "convert_urdf"
    else:
        from isaacsimenvs.tasks.play.utils import scene_utils as target

        name = "_convert_urdf_to_usd"
    if getattr(target, "_usd_cache_writer_installed", False):
        return

    original = getattr(target, name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _convert_and_cache(asset_path, usd_work_dir, **kwargs):
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
        return usd_path

    setattr(target, name, _convert_and_cache)
    target._usd_cache_writer_installed = True


def main() -> None:
    """Populate the cache by building a Play or rigid-cloth scene under Isaac Sim."""
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Bake Play or cloth robot/table assets into the USD cache.")
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
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        cloth = getattr(env_cfg, "backend", None) == "rigid"
        if not cloth:
            env_cfg.assets.num_assets_per_type = args.num_assets_per_type
        install_writer(args.cache_dir, cloth=cloth)
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
