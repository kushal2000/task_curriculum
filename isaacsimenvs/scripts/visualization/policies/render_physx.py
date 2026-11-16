"""RGB video of a policy on a PhysX (Kit) task, filmed from one env -- e.g. the rigid-body cloth sheet.

    .venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/policies/render_physx.py --headless \\
        --task Isaacsimenvs-RigidBodyCloth-Direct-v0 --experiment rigid_body_cloth \\
        --checkpoint <run>/nn/<name>.pth --out clip.mp4

Kit renders the env's viewport camera (``env_cfg.viewer``, placed relative to the filmed env's
origin) via ``render_mode="rgb_array"``. The rollout matches ``isaacsimenvs/scripts/evaluation/eval.py``: DR off
(``disable_randomization``), deterministic actions, the given SAPG block. ``--stop_on_episode_end``
ends the clip when the filmed env's first episode ends and writes ``<out>.json`` with its outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaacsimenvs.scripts.evaluation.protocol import disable_randomization, use_single_object_variant
from isaacsimenvs import PRETRAINED_CONFIG


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--stride", type=int, default=2, help="capture every Nth policy step")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--world", type=int, default=0, help="which env to film")
    parser.add_argument("--cam_offset", type=float, nargs=3, default=(1.0, -1.0, 1.0),
                        help="camera position relative to the filmed env's origin [m]")
    parser.add_argument("--look_at", type=float, nargs=3, default=(0.0, 0.0, 0.55),
                        help="aim point relative to the filmed env's origin [m]")
    parser.add_argument("--experiment", default=None,
                        help="isaacsimenvs/scripts/configs/runs/ YAML layered over the task YAML (as in training)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy_config", default=str(PRETRAINED_CONFIG))
    parser.add_argument("--sapg_expl_coef", type=float, default=0.0,
                        help="SAPG block conditioning: 0.0 = last block (eval.py's fabric protocol).")
    parser.add_argument("--success_tolerance", type=float, default=0.01)
    parser.add_argument("--single_variant", action="store_true")
    parser.add_argument("--stop_on_episode_end", action="store_true")
    parser.add_argument("--end_hold_s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.enable_cameras = True
    sys.argv = [sys.argv[0]] + hydra_args
    _app = AppLauncher(args).app  # Keep Kit alive for the rollout.

    import gymnasium as gym
    import imageio.v2 as imageio
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer
    from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml

    # The agent entry point is needed only so an experiment YAML's agent: block has a config to merge into.
    @hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = args.success_tolerance
        disable_randomization(env_cfg)
        if args.single_variant:
            use_single_object_variant()
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = args.world
        env_cfg.viewer.eye = tuple(args.cam_offset)
        env_cfg.viewer.lookat = tuple(args.look_at)
        env_cfg.viewer.resolution = (args.width, args.height)

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        inner = env.unwrapped
        device = inner.device
        n = inner.num_envs
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device=device, num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
        )
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=device))
        w = args.world
        shape = None
        if getattr(inner, "_env_shape_np", None) is not None:
            shape = inner._pool[int(inner._env_shape_np[w])].name
        task = None
        if hasattr(inner, "_unfold_envs"):
            task = "unfold" if bool(inner._unfold_envs()[w]) else "fold"

        frames, outcome = [], None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for step in range(args.steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, term, trunc, extras = env.step(action.to(device))
            if step % args.stride == 0:
                frames.append(env.render())
            ended = term.bool() | trunc.bool()
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if bool(ended[w]) and outcome is None:
                final = extras["episode_final"]
                outcome = {
                    "step": step,
                    "reasons": sorted(k[len("done_"):] for k, v in final.items()
                                      if k.startswith("done_") and v.ndim == 1 and bool(v[w])),
                }
                print(f"[render_physx] env {w} episode ended at step {step}: {outcome['reasons']}")
                if args.stop_on_episode_end:
                    frames += [frames[-1]] * int(args.end_hold_s * args.fps)
                    break
        imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1)
        meta = {"task_id": args.task, "checkpoint": args.checkpoint, "world": w, "shape": shape,
                "cloth_task": task, "frames": len(frames), "outcome": outcome,
                "overrides": hydra_args}
        Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
        print(f"[render_physx] wrote {args.out} ({len(frames)} frames) {meta}")
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
