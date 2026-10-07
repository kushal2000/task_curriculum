"""Look at the fold pool's partly folded starts in the simulator: clouds, hinges and the hand.

Resets the pool with ``fold_pool.init_fold_prob`` (default here: 1, every episode folded), holds zero
actions for ``--steps`` policy steps and saves, per step: each env's particle cloud (env frame), its
active shape's hinge angles, and the fingertip positions -- plus each env's drawn shape and start
angles. ``fold_start_probe_plot.py`` draws the clouds and reports how the starts settle and how close
the hand is to a raised flap.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/analysis/fold_start_probe.py --headless \\
        --out docs/results/fold_start_probe.pt env.rigid_cloth.joint_damping=0.02
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysx-Direct-v0")
parser.add_argument("--num_envs", type=int, default=60)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--init_fold_prob", type=float, default=1.0)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pinned", action="store_true",
                    help="the eval / viewer pinned start (disable_randomization) instead of training resets")
parser.add_argument(
    "--skip_start_check", action="store_true",
    help="bypass the startup proof of the start folds, to look at what it would reject",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch
    from isaacsimenvs.tasks.play.utils.obs_utils import OBS_FIELD_SIZES

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.fold_pool.init_fold_prob = args_cli.init_fold_prob
    if args_cli.pinned:
        disable_randomization(env_cfg)
    use_single_object_variant()
    if args_cli.skip_start_check:
        from isaacsimenvs.tasks.cloth.fold_pool_task import FoldPoolTaskMixin

        FoldPoolTaskMixin._verify_start_folds = lambda self, *a, **k: print("[fold_start] start check SKIPPED")
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    n, device = inner.num_envs, inner.device

    # Offsets of the two hand blocks in the policy observation.
    off, cur = {}, 0
    for f in inner.cfg.obs.obs_list:
        off[f] = cur
        cur += int(OBS_FIELD_SIZES[f])

    def hand(obs) -> torch.Tensor:
        o = obs["policy"]
        palm = o[:, off["palm_pos"] : off["palm_pos"] + 3]
        tips = o[:, off["fingertip_pos_rel_palm"] : off["fingertip_pos_rel_palm"] + 15].reshape(n, 5, 3)
        return (tips + palm.unsqueeze(1)).cpu()

    def hinges() -> list[torch.Tensor]:
        active = inner.object.active
        out = []
        for e in range(n):
            s = int(active[e])
            row = int(inner._cloth.local_rows(s, torch.tensor([e], device=device))[1][0])
            out.append(_to_torch(inner._pool_arts[s].data.joint_pos)[row].detach().cpu().clone())
        return out

    obs, _ = env.reset()
    active = inner.object.active.detach().cpu().clone()
    start_q = [
        inner.object._start_q[int(active[e])][e].detach().cpu().clone()
        if int(active[e]) in inner.object._start_q
        else None
        for e in range(n)
    ]
    rec = {"particles": [], "tips": [], "hinges": [], "table_z": []}
    zero = torch.zeros((n, inner.cfg.action_space), device=device)
    for _ in range(args_cli.steps + 1):
        rec["particles"].append((inner._particles_w() - inner.scene.env_origins.unsqueeze(1)).cpu())
        rec["tips"].append(hand(obs))
        rec["hinges"].append(hinges())
        rec["table_z"].append(
            (_to_torch(inner.table.data.root_pos_w)[:, 2] - inner.scene.env_origins[:, 2]).cpu().clone()
        )
        obs, *_ = env.step(zero)

    out = {
        "particles": torch.stack(rec["particles"]),
        "tips": torch.stack(rec["tips"]),
        "hinges": rec["hinges"],
        "table_root_z": torch.stack(rec["table_z"]),
        "spawn_z": float(inner.cfg.reset.table_reset_z) + float(inner.cfg.cloth.start_height),
        "active": active,
        "shapes": [sh.name for sh in inner._pool],
        "start_q": start_q,
        "start_folded": inner._start_folded.detach().cpu().clone(),
        "env_size": torch.as_tensor(inner._env_size_np),
        "table_z": float(inner.cfg.reset.table_reset_z),
        "dt_policy": float(inner.cfg.sim.dt) * int(inner.cfg.decimation),
        "dt_sim": float(inner.cfg.sim.dt),
        "decimation": int(inner.cfg.decimation),
        "pinned": bool(args_cli.pinned),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    torch.save(out, args_cli.out)
    print(f"[fold_start] wrote {args_cli.out}: {args_cli.steps} steps x {n} envs", flush=True)


if __name__ == "__main__":
    run()
    os._exit(0)
