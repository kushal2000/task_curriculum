"""Dump a deterministic rollout of a fold task's observations, to diff the same task across simulators.

A policy trained on PhysX is evaluated on Newton's VBD cloth, so the two must hand the policy the
SAME 140 numbers for the same state -- including every convention quirk the Newton policies were
trained under (``physx_envs`` module docstring). This runs the pinned start (``disable_randomization``)
with zero actions (or a checkpoint, deterministic) and saves, per step: the policy observation, the
fold error, the emulated particle cloud, and the object / goal quaternions as the task reads them.
``cloth_obs_parity_compare.py`` diffs two dumps block by block.

    # Newton (kit-less)
    scripts/newton_py scripts/analysis/cloth_obs_parity.py --task Isaacsimenvs-RigidCloth-Direct-v0 \\
        --out docs/results/parity_newton.pt
    # PhysX
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/analysis/cloth_obs_parity.py \\
        --task Isaacsimenvs-RigidClothPhysx-Direct-v0 --headless --out docs/results/parity_physx.pt
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--out", required=True)
parser.add_argument("--checkpoint", default=None, help="drive with this policy instead of zero actions")
parser.add_argument("--policy_config", default="pretrained_policy/config.yaml")
parser.add_argument("--sapg_expl_coef", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--replay", default=None,
    help="a previous dump: replay its recorded actions open-loop instead of running a policy",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
try:
    app = AppLauncher(args_cli).app
except ImportError as exc:
    app = None
    print(f"[parity] kit-less: {exc}", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    from isaacsimenvs.tasks.play.utils.obs_utils import OBS_FIELD_SIZES

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    disable_randomization(env_cfg)
    use_single_object_variant()
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    n = inner.num_envs

    player = None
    if args_cli.checkpoint:
        from isaacsimenvs.eval.player import PretrainedPlayer

        player = PretrainedPlayer(
            config_path=args_cli.policy_config,
            checkpoint_path=args_cli.checkpoint,
            num_envs=n,
            device=str(device),
            num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space),
            expl_coef=args_cli.sapg_expl_coef,
        )

    def q(t):
        return t.torch if hasattr(t, "torch") else t

    rec = {k: [] for k in ("obs", "fold_err", "particles", "object_quat", "goal_quat", "goal_pos", "actions")}
    replay = torch.load(args_cli.replay)["actions"].to(device) if args_cli.replay else None
    obs, _ = env.reset()
    zero = torch.zeros((n, inner.cfg.action_space), device=device)
    obs, *_ = env.step(zero)
    for _ in range(args_cli.steps):
        rec["obs"].append(obs["policy"].detach().cpu().clone())
        rec["fold_err"].append(inner.fold_error().detach().cpu().clone())
        rec["particles"].append((inner._particles_w() - inner.scene.env_origins.unsqueeze(1)).cpu())
        rec["object_quat"].append(q(inner.object.data.root_quat_w).detach().cpu().clone())
        rec["goal_quat"].append(q(inner.goal_viz.data.root_quat_w).detach().cpu().clone())
        rec["goal_pos"].append(
            (q(inner.goal_viz.data.root_pos_w) - inner.scene.env_origins).detach().cpu().clone()
        )
        t = len(rec["actions"])
        if replay is not None:
            action = replay[min(t, len(replay) - 1)][:n] if t < len(replay) else zero
        elif player is not None:
            action = player.get_action(obs["policy"], deterministic=True)
        else:
            action = zero
        rec["actions"].append(action.detach().cpu().clone())
        obs, *_ = env.step(action.to(device))

    out = {k: torch.stack(v) for k, v in rec.items()}
    out["obs_fields"] = [(f, int(OBS_FIELD_SIZES[f])) for f in inner.cfg.obs.obs_list]
    out["task"] = args_cli.task
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    torch.save(out, args_cli.out)
    print(f"[parity] wrote {args_cli.out}: {args_cli.steps} steps x {n} envs", flush=True)


if __name__ == "__main__":
    run()
    os._exit(0)
