"""Boot Kit, import the package, and confirm every task id resolves.

The cheapest thing that can fail after an edit is registration — a typo in an entry point
or a config path only shows up once `isaacsimenvs` is imported, and that import needs a
booted Kit, so `pytest tests/` cannot catch it. This is that check, and it takes about a
minute rather than a training run.

    export OMNI_KIT_ACCEPT_EULA=YES
    .venv_isaacsim/bin/python experiments/check_registration.py

Add --make to also instantiate each env with a handful of envs, which additionally
exercises URDF generation, USD conversion, scene setup and the first reset.
"""

from __future__ import annotations

import argparse
import os
import sys

EXPECTED_TASKS = (
    "Isaacsimenvs-MultiLinkCartpole-Direct-v0",
    "Isaacsimenvs-BottleFlip-Direct-v0",
    "Isaacsimenvs-Play-Direct-v0",
)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Verify isaacsimenvs task registration.")
    parser.add_argument(
        "--make", action="store_true",
        help="Also gym.make each task (slower; exercises asset generation and scene setup).",
    )
    parser.add_argument("--num_envs", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.headless = True

    app = AppLauncher(args_cli).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401  fires gym.register for every task
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    failures: list[str] = []
    for task_id in EXPECTED_TASKS:
        try:
            spec = gym.spec(task_id)
            kwargs = spec.kwargs
            cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            yaml_path = kwargs.get("env_cfg_yaml_entry_point")
            if yaml_path and not os.path.exists(yaml_path):
                raise FileNotFoundError(f"task YAML missing: {yaml_path}")
            for key in ("rl_games_cfg_entry_point", "rl_games_sapg_cfg_entry_point"):
                path = kwargs.get(key)
                if path and not os.path.exists(path):
                    raise FileNotFoundError(f"{key} missing: {path}")
            print(f"  OK   {task_id}  cfg={type(cfg).__name__}")
        except Exception as exc:  # noqa: BLE001 - report every task, not just the first
            failures.append(f"{task_id}: {exc}")
            print(f"  FAIL {task_id}: {exc}")

    if args_cli.make and not failures:
        for task_id in EXPECTED_TASKS:
            try:
                env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
                env_cfg.scene.num_envs = args_cli.num_envs
                env = gym.make(task_id, cfg=env_cfg)
                obs, _ = env.reset()
                policy_obs = obs["policy"]
                print(f"  MADE {task_id}  obs={tuple(policy_obs.shape)}")
                env.close()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"make {task_id}: {exc}")
                print(f"  FAIL make {task_id}: {exc}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for line in failures:
            print(f"  - {line}")
        os._exit(1)
    print("\nall tasks registered.")
    # Kit teardown can hang; force-exit rather than wait for it.
    os._exit(0)


if __name__ == "__main__":
    main()
