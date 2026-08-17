"""Simulation throughput for the cable task, at a given environment count.

    scripts/newton_py -m isaacsimenvs.eval.throughput --num_envs 64

Reports environment-steps per second (``num_envs`` x policy steps per second), which is the number
that matters for training budget, alongside per-step wall time.

Three things this measures carefully, because each will silently corrupt the figure:

* **Warm-up is excluded.** The first steps pay for Warp kernel compilation, CUDA graph capture and
  allocator growth. Timing them reports a number several times too slow.
* **CUDA is synchronised** around the timed region. Without it the host races ahead of the device
  and the measurement times queue submission rather than simulation.
* **Policy inference is separated from stepping.** ``--with_policy`` includes it; by default the
  actions are zeros, isolating the simulator. A training loop pays both, so both are useful, but
  conflating them hides which one is the bottleneck.

Run one environment count per GPU. Two benchmarks sharing a device measure contention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200, help="timed policy steps")
    parser.add_argument("--warmup", type=int, default=60, help="untimed steps first")
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--with_policy", action="store_true", help="include policy inference")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", default="/share/portal/kk837/simtoolreal/pretrained_policy/model.pth"
    )
    parser.add_argument(
        "--policy_config", default="/share/portal/kk837/simtoolreal/pretrained_policy/config.yaml"
    )
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    import torch

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml("Isaacsimenvs-Cable-Direct-v0", "")
    def run(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.assets.num_assets_per_type = args_cli.num_assets_per_type
        disable_randomization(env_cfg)
        use_single_object_variant()

        build_t0 = time.perf_counter()
        env = gym.make("Isaacsimenvs-Cable-Direct-v0", cfg=env_cfg)
        inner = env.unwrapped
        device = inner.device
        build_s = time.perf_counter() - build_t0

        player = None
        if args_cli.with_policy:
            from isaacsimenvs.eval.player import PretrainedPlayer

            player = PretrainedPlayer(
                config_path=args_cli.policy_config,
                checkpoint_path=args_cli.checkpoint,
                num_envs=inner.num_envs,
                device=str(device),
                num_observations=int(inner.cfg.observation_space),
                num_actions=int(inner.cfg.action_space),
            )

        zeros = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
        obs, _ = env.reset()

        def step_once(obs):
            action = player.get_action(obs["policy"], deterministic=True) if player else zeros
            obs, *_ = env.step(action.to(device) if player else action)
            return obs

        # Warp kernel compilation, CUDA graph capture and allocator growth all land in the first
        # steps; timing them would report a number several times too slow.
        for _ in range(args_cli.warmup):
            obs = step_once(obs)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.steps):
            obs = step_once(obs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        policy_sps = args_cli.steps / elapsed
        result = {
            "num_envs": inner.num_envs,
            "with_policy": bool(args_cli.with_policy),
            "timed_steps": args_cli.steps,
            "elapsed_s": round(elapsed, 3),
            "ms_per_step": round(1e3 * elapsed / args_cli.steps, 3),
            "policy_steps_per_s": round(policy_sps, 2),
            "env_steps_per_s": round(policy_sps * inner.num_envs, 1),
            "build_s": round(build_s, 1),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "segments": int(inner.cfg.cable.segments),
            "substeps": int(inner.cfg.cable.cable_substeps),
            "vbd_iterations": int(inner.cfg.cable.vbd_iterations),
            "proxy_mode": str(inner.cfg.cable.proxy_mode),
        }
        print("THROUGHPUT " + json.dumps(result), flush=True)
        if args_cli.out:
            args_cli.out.parent.mkdir(parents=True, exist_ok=True)
            args_cli.out.write_text(json.dumps(result, indent=2))
        env.close()

    run()


if __name__ == "__main__":
    main()
