"""Object speed distribution during a policy rollout, for any manipuland.

    scripts/newton_py -m isaacsimenvs.eval.velocity_stats --task Isaacsimenvs-PlayNewton-Direct-v0

Peak speed alone is a poor stability metric: it is one sample from the tail, and a *successful*
rollout legitimately contains fast motion -- the hand carries the object to a goal, a goal is
scored, a new one is sampled far away, and the object accelerates. Reading a single peak as
"unstable" without a reference for what a working manipulation looks like is how a real solver
problem and ordinary task motion get confused.

So this reports percentiles alongside the max, and is runnable against the rigid-tool and
rigid-rod baselines as well as the cable. The rigid baselines are the reference: whatever speed
distribution the policy produces while actually scoring is, by definition, reasonable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaacsimenvs-Cable-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--single_variant", action="store_true", default=True)
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

    import isaacsimenvs  # noqa: F401  (registers the tasks)
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args_cli.task, "")
    def run(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.assets.num_assets_per_type = args_cli.num_assets_per_type
        env_cfg.termination.eval_success_tolerance = 0.01
        disable_randomization(env_cfg)
        if args_cli.single_variant:
            use_single_object_variant()

        env = gym.make(args_cli.task, cfg=env_cfg)
        inner = env.unwrapped
        device = inner.device

        player = PretrainedPlayer(
            config_path=args_cli.policy_config,
            checkpoint_path=args_cli.checkpoint,
            num_envs=inner.num_envs,
            device=str(device),
            num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space),
        )

        obs, _ = env.reset()
        speeds = []
        goals_before = int(inner._successes.sum())
        for _ in range(args_cli.steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, _term, _trunc, _x = env.step(action.to(device))
            # Segment velocities for a cable, root velocity for a rigid manipuland. Take the
            # per-env max so one env's excursion is not averaged away by 15 quiet ones.
            try:
                v = inner.object._segment_velocities()[..., :3].norm(dim=-1).amax(dim=-1)
            except AttributeError:
                v = inner.object.data.root_lin_vel_w.norm(dim=-1)
            speeds.append(v.detach())

        allv = torch.stack(speeds).flatten().float()
        finite = allv[torch.isfinite(allv)]
        q = torch.tensor([0.5, 0.9, 0.99], device=finite.device)
        p50, p90, p99 = (float(x) for x in torch.quantile(finite, q))
        result = {
            "task": args_cli.task,
            "num_envs": inner.num_envs,
            "steps": args_cli.steps,
            "mean": round(float(finite.mean()), 3),
            "p50": round(p50, 3),
            "p90": round(p90, 3),
            "p99": round(p99, 3),
            "max": round(float(finite.max()), 3),
            "non_finite": int((~torch.isfinite(allv)).sum()),
            "goals_scored": int(inner._successes.sum()) - goals_before,
            "lifted": int(inner._lifted_object.sum()),
        }
        print("VELSTATS " + json.dumps(result), flush=True)
        if args_cli.out:
            args_cli.out.parent.mkdir(parents=True, exist_ok=True)
            args_cli.out.write_text(json.dumps(result, indent=2))
        env.close()

    run()


if __name__ == "__main__":
    main()
