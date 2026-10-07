"""cProfile the construction of a PhysX fold env: which calls grow faster than the env count.

    .venv_isaacsim/bin/python scripts/analysis/physx_build_profile.py --headless \\
        --task Isaacsimenvs-RigidClothPhysx-Direct-v0 --num_envs 4096 --out prof_4096.txt

Writes the top functions by cumulative and by own time to ``--out`` (text) and the raw stats next to
it (``.prof``). Construction only: ``gym.make`` and one reset. Exits with ``os._exit`` because Kit's
shutdown hangs on these nodes, so the stats are written first.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    disable_randomization(env_cfg)
    use_single_object_variant()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    prof.disable()
    wall = time.perf_counter() - t0
    buf = io.StringIO()
    buf.write(f"{args_cli.task} num_envs={args_cli.num_envs} construction wall {wall:.1f} s\n\n")
    stats = pstats.Stats(prof, stream=buf)
    stats.sort_stats("cumulative").print_stats(60)
    stats.sort_stats("tottime").print_stats(40)
    stats.dump_stats(os.path.splitext(args_cli.out)[0] + ".prof")
    with open(args_cli.out, "w") as f:
        f.write(buf.getvalue())
    print(f"[build_profile] wall {wall:.1f} s -> {args_cli.out}", flush=True)


run()
os._exit(0)
