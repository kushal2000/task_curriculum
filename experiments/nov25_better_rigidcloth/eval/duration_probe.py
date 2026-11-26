"""How long do finger penetrations last? Per-step depth of every env's first episode, rule disabled.

    python experiments/nov25_better_rigidcloth/eval/duration_probe.py --headless --rest_mm 1 --out X.json [hydra overrides]

Writes, per env (first episode after the reset): the finger-in-plate depth at every policy step [mm],
whether the episode ended in a held fold, and its length. Rules can then be replayed offline: an
episode the policy folded fails under a rule only if the rule fires before the fold (the rollout up to
that point is the same, the rule being the only change).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pretrained_policy/play_pc_rigidclothRL_64live.pth")
    p.add_argument("--experiment", default="experiments/nov24_play_pc_rl/play_pc_rl_A.yaml")
    p.add_argument("--shapes", default="box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid")
    p.add_argument("--rest_mm", type=float, default=None, help="fixed rest offset [mm]; default: the recipe's")
    p.add_argument("--num_envs", type=int, default=448)
    p.add_argument("--max_steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    AppLauncher.add_app_launcher_args(p)
    args, extra = p.parse_known_args()
    os.chdir(ROOT)
    over = ["env.cloth.keypoint_tolerance=0.015", "env.cloth.nan_policy=reset", "env.termination.success_steps=10",
            f"env.fold_pool.shapes=[{args.shapes}]", "env.fold_pool.one_shape_per_env=true",
            "env.fold_pool.init_fold_prob=0.0", "env.fold_pool.goal_task.mode=half_fold",
            "env.termination.max_finger_penetration=1.0"]
    if args.rest_mm is not None:
        over += [f"env.rigid_cloth.contact_margin={args.rest_mm / 1e3}",
                 "env.domain_randomization.cloth.contact_margin_offset=[0.0,0.0]"]
    sys.argv = [sys.argv[0], *over, *extra]
    AppLauncher(args)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
    from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud, _student_checkpoint
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    task = "Isaacsimenvs-RigidCloth-Direct-v0"
    student = _student_checkpoint(args.checkpoint)

    @hydra_task_config_with_yaml(task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = "cuda:0"
        _check_student_cloud(env_cfg, student["env"])
        env = gym.make(task, cfg=env_cfg)
        inner = env.unwrapped
        pen = inner.backend.penetration
        player = EnvCloudStudentPlayer(inner, student, "cuda:0")
        n = inner.num_envs
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=inner.device))
        player.reset_rnn(torch.arange(n, device=inner.device))
        depth_log = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool, device=inner.device)
        folded = [False] * n
        outcome = [None] * n
        shape = [None] * n
        for _step in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, terminated, truncated, extras = env.step(action)
            # pen.episode_max is per env since its reset; the step's own depth is what the rule reads.
            d = pen.read().cpu()
            ended = terminated.bool() | truncated.bool()
            for e in (~done).nonzero(as_tuple=True)[0].tolist():
                depth_log[e].append(round(float(d[e]) * 1e3, 3))
            fin = ended & ~done
            if bool(fin.any()):
                final = extras["episode_final"]
                for e in fin.nonzero(as_tuple=True)[0].tolist():
                    folded[e] = bool(final["done_fold_held"][e])
                    outcome[e] = next((k[5:] for k in ("done_fold_held", "done_fall", "done_hand_far", "done_timeout",
                                                       "done_nonfinite") if k in final and bool(final[k][e])),
                                      "timeout" if bool(truncated[e]) else "?")
                    shape[e] = next((k[6:] for k, v in final.items() if k.startswith("done_shape_") and bool(v[e])), None)
                done |= fin
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            elif bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if bool(done.all()):
                break
        out = {"rest_mm": args.rest_mm, "experiment": args.experiment, "policy_dt": float(inner.step_dt),
               "physics_dt": float(inner.physics_dt), "decimation": int(inner.cfg.decimation),
               "episodes": [{"folded": folded[e], "outcome": outcome[e], "shape": shape[e], "depth_mm": depth_log[e]}
                            for e in range(n)]}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out))
        print(f"[duration] wrote {args.out}: {sum(folded)}/{n} folded, {int((~done).sum())} unfinished", flush=True)
        env.close()

    run()


if __name__ == "__main__":
    main()
