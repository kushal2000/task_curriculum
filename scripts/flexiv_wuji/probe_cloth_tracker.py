"""Cloth tracker DR probe (cloth.tracker): correctness against the true observation, freeze / cover
statistics on a folding policy, and the step-time cost.

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_cloth_tracker.py \
        --headless --experiment flat7RW_trkC --mode exact --ckpt base.pth --out r.json

--mode exact: every tracker range collapsed (no noise, bias, calibration, size error, occlusion; a frame
every step, no latency). On a rigid half the estimate must equal the true object / goal pose; the
report gives the error per shape and over time (non-rigid multi-plate halves differ by design).
--mode full: the recipe as trained. Reports the tracker's episode stats and the frozen share over time.
--mode off: tracker disabled (timing baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0")
parser.add_argument("--experiment", default="flat7RW_trkC")
parser.add_argument("--mode", required=True, choices=("exact", "full", "off"))
parser.add_argument("--ckpt", required=True)
parser.add_argument("--policy_config", default="pretrained_policy/config.yaml")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.eval.player import PretrainedPlayer  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


def exact(tc) -> None:
    tc.cam_pos_noise = (0.0, 0.0, 0.0)
    tc.hand_occlusion = tc.self_occlusion = tc.reacquire = False
    tc.noise_std_range = (0.0, 0.0)
    tc.reacquire_bias_range = (0.0, 0.0)
    tc.calib_xyz = (0.0, 0.0, 0.0)
    tc.calib_rot_deg = 0.0
    tc.frame_period_choices = (1,)
    tc.latency_choices = (0,)
    tc.size_scale_range = (1.0, 1.0)


def qdist_deg(a, b):
    d = (a * b).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    tc = env_cfg.cloth.tracker
    if args_cli.mode == "exact":
        exact(tc)
    elif args_cli.mode == "off":
        tc.enabled = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    dev, n = inner.device, inner.num_envs
    rec = {"exact": [], "t": []}

    if args_cli.mode == "exact":
        orig = inner._tracker_observe

        def wrapped(start):
            out = orig(start)
            o = inner.scene.env_origins
            true_pos = inner.object.data.root_pos_w - o
            true_rot = inner.object.data.root_quat_w
            g_pos = inner.goal_viz.data.root_pos_w - o
            g_rot = inner.goal_viz.data.root_quat_w
            rec["exact"].append(torch.stack((
                (out[0] - true_pos).norm(dim=-1), qdist_deg(out[1], true_rot),
                (out[2] - g_pos).norm(dim=-1), qdist_deg(out[3], g_rot),
            ), dim=-1).cpu())
            return out

        inner._tracker_observe = wrapped

    player = PretrainedPlayer(config_path=args_cli.policy_config, checkpoint_path=args_cli.ckpt, num_envs=n,
                              device=str(dev), num_observations=int(inner.cfg.observation_space),
                              num_actions=int(inner.cfg.action_space), expl_coef=0.0)
    obs, _ = env.reset()
    player.reset_rnn()
    shape = inner.object.active.clone().cpu()
    names = [sh.name for sh in inner._pool]
    finals = []
    frozen_t = []
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(args_cli.max_steps):
        action = player.get_action(obs["policy"], deterministic=True)
        obs, _r, term, trunc, extras = env.step(action.to(dev))
        done = (term | trunc)
        fin = extras.get("episode_final", {})
        if bool(done.any()) and "tracker_frozen" in fin:
            ids = done.nonzero().squeeze(1)
            finals.append({k: v[ids].float().cpu() for k, v in fin.items() if k.startswith(("tracker_", "fold_steps"))})
        tr = getattr(inner, "_tracker", None)
        if tr is not None:
            frozen_t.append(float((~tr.fresh).float().mean()))
    torch.cuda.synchronize()
    wall = time.time() - t0
    report = {"mode": args_cli.mode, "experiment": args_cli.experiment, "num_envs": n, "steps": args_cli.max_steps,
              "wall_s": round(wall, 2), "env_steps_per_s": round(n * args_cli.max_steps / wall, 1)}
    if rec["exact"]:
        e = torch.stack(rec["exact"])  # (T, N, 4)
        cols = ("obj_pos_m", "obj_rot_deg", "goal_pos_m", "goal_rot_deg")
        report["exact_first_step_max"] = {c: round(float(e[0, :, i].max()), 6) for i, c in enumerate(cols)}
        per_shape = {}
        for s, nm in enumerate(names):
            m = shape == s
            if m.any():
                per_shape[nm] = {c: [round(float(e[:, m, i].quantile(q)), 5) for q in (0.5, 0.95, 1.0)]
                                 for i, c in enumerate(cols)}
        report["exact_by_shape_p50_p95_max"] = per_shape
    if finals:
        cat = {k: torch.cat([f[k] for f in finals]) for k in finals[0]}
        report["episodes"] = int(next(iter(cat.values())).numel())
        report["episode_means"] = {k: round(float(v.mean()), 4) for k, v in cat.items()}
    if frozen_t:
        b = 30
        report["not_fresh_share_by_0.5s"] = [round(sum(frozen_t[i:i + b]) / len(frozen_t[i:i + b]), 3)
                                             for i in range(0, len(frozen_t), b)]
    print("[tracker_probe] " + json.dumps(report), flush=True)
    with open(args_cli.out, "w") as f:
        json.dump(report, f, indent=1)
    env.close()


if __name__ == "__main__":
    run()
    import os
    os._exit(0)
