"""Record the Play policy in the training env, every step, for check_rec1.py; optionally compare SAPG blocks.

Runs on the TRAINING branch's code (kushal/2026-10-30_pretraining_real), not this one: the env it
builds is the one the checkpoint was trained in. Randomization is off (protocol.disable_randomization:
no obs / action / object-state delay, no noise, no impulses), but the object's start pose stays random
as in training, and the success tolerance is pinned to training's final 0.01 m.

Every step it saves what the real loop will measure (joint angles and velocities, previous targets,
object, goal, object scales), what the sim computed from it (the 134-dim policy observation, the palm
and fingertip body poses, the keypoint distance and goal-hit test) and what came out (the action, the
next joint targets). check_rec1.py runs those inputs through the deploy code. On the cluster:
scripts/cluster/bos14_simtoolreal_record.sh. By hand:

    cd <branch worktree>
    PYTHONPATH=$PWD:$PWD/rl_games .venv_isaacsim/bin/python \
        /home/yiboc/mit/task_curriculum_deployment/deploy/simtoolreal/checks/record_sim_rollout.py --headless \
        --checkpoint pretrained_policy/nov6_dr.pth --out <dir>

--sweep_coefs 50,40,30,20,10,0 then runs every SAPG block for --sweep_steps on all envs and reports
goals reached and object falls per block.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TASK = "Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0"
DEPLOY_REPO = Path(__file__).resolve().parents[3]   # deploy/simtoolreal/checks -> repo root


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default=str(DEPLOY_REPO / "pretrained_policy" / "config.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--num_assets_per_type", type=int, default=2,
                   help="objects per size distribution; envs = this x 12 distributions")
    p.add_argument("--steps", type=int, default=1200, help="recorded policy steps (60 Hz)")
    p.add_argument("--expl_coef", type=float, default=0.0, help="SAPG block of the recording (0 = last)")
    p.add_argument("--sweep_coefs", default="", help="comma list of SAPG coefficients to compare")
    p.add_argument("--sweep_steps", type=int, default=3600)
    p.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization
    from isaacsimenvs.tasks.play.utils.object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    @hydra_task_config_with_yaml(TASK, "")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.assets.num_assets_per_type = args.num_assets_per_type
        types = set(env_cfg.assets.handle_head_types)
        env_cfg.scene.num_envs = args.num_assets_per_type * sum(d.type in types for d in OBJECT_SIZE_DISTRIBUTIONS)
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = float(env_cfg.termination.target_success_tolerance)
        disable_randomization(env_cfg)
        # Keep training's random start (position noise, random orientation); DR stays off.
        env_cfg.reset.fixed_start_pose = None
        env_cfg.reset.reset_position_noise_x = 0.1
        env_cfg.reset.reset_position_noise_y = 0.1

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        perm = inner._perm_lab_to_canon
        origins = inner.scene.env_origins
        n_obs, n_act = int(inner.cfg.observation_space), int(inner.cfg.action_space)
        assert (n_obs, n_act) == (134, 27), (n_obs, n_act)
        joint_names = [inner.robot.joint_names[i] for i in perm.tolist()]
        print(f"[rec] {TASK}: {n} envs, obs {n_obs}, act {n_act}, device {dev}", flush=True)

        def player(coef: float) -> PretrainedPlayer:
            return PretrainedPlayer(args.policy_config, args.checkpoint, num_envs=n, device=dev,
                                    num_observations=n_obs, num_actions=n_act, expl_coef=coef)

        def state() -> dict:
            """What the real loop measures, and what the sim derived from it, right now."""
            d = inner.robot.data
            body = d.body_state_w
            return {
                "q": d.joint_pos[:, perm], "qd": d.joint_vel[:, perm],
                "prev_targets": inner._prev_targets[:, perm],
                "object_pos": inner.object.data.root_pos_w - origins,
                "object_quat": inner.object.data.root_quat_w,
                "goal_pos": inner.goal_viz.data.root_pos_w - origins,
                "goal_quat": inner.goal_viz.data.root_quat_w,
                "palm_body_pos": body[:, inner._palm_body_id, 0:3] - origins,
                "palm_body_quat": body[:, inner._palm_body_id, 3:7],
                "ft_body_pos": body[:, inner._fingertip_body_ids, 0:3] - origins[:, None],
                "ft_body_quat": body[:, inner._fingertip_body_ids, 3:7],
            }

        def to_np(d: dict) -> dict:
            return {k: v.detach().float().cpu().numpy().copy() for k, v in d.items()}

        # ---------------------------------------------------------------- the recorded rollout
        pl = player(args.expl_coef)
        obs = env.reset()[0]["policy"]
        pl.reset_rnn()
        frames: list[dict] = []
        t0 = time.time()
        for t in range(args.steps):
            s = to_np(state())
            with torch.no_grad():
                action = pl.get_action(obs, deterministic=True)
            obs_next, _rew, term, trunc, _info = env.step(action)
            done = (term | trunc)
            frames.append({
                **s,
                "obs": obs.float().cpu().numpy().copy(),
                "action": action.float().cpu().numpy().copy(),
                # targets the pipeline wrote this step (before any reset overwrote them)
                "targets": inner._cur_targets[:, perm].float().cpu().numpy().copy(),
                "object_pos_after": (inner.object.data.root_pos_w - origins).float().cpu().numpy().copy(),
                "object_quat_after": inner.object.data.root_quat_w.float().cpu().numpy().copy(),
                "kp_max_dist": inner._keypoints_max_dist.float().cpu().numpy().copy(),
                "is_success": inner._is_success.cpu().numpy().copy(),
                "done": done.cpu().numpy().copy(),
                "success_tol": np.float32(inner._current_success_tolerance),
            })
            if done.any():
                pl.reset_rnn(done.nonzero(as_tuple=False).squeeze(-1))
            obs = obs_next["policy"]
            if (t + 1) % 300 == 0:
                print(f"[rec] step {t + 1}/{args.steps} ({time.time() - t0:.0f} s)", flush=True)

        rec = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
        rec["object_scales"] = (inner._object_scale_per_env * inner._object_scale_multiplier).cpu().numpy()
        rec["object_corners"] = inner._object_collision_corners.cpu().numpy()
        rec["table_top_z"] = (inner._table_z_per_env + inner._table_half_height).cpu().numpy()
        np.savez_compressed(out / "rollout.npz", **rec)
        meta = {
            "task": TASK, "checkpoint": str(args.checkpoint), "expl_coef": args.expl_coef,
            "num_envs": n, "steps": args.steps, "joint_names": joint_names,
            # The order the obs lists the fingertips in (find_bodies = PhysX body order), and the
            # pad offset the env applied at each position (robot_spec.fingertip_offsets, by position).
            "fingertip_bodies_obs_order": [inner.robot.body_names[i] for i in inner._fingertip_body_ids],
            "fingertip_offsets_by_position": [list(o) for o in inner.robot_spec.fingertip_offsets],
            "palm_body": inner.robot.body_names[inner._palm_body_id],
            "object_urdfs": [str(u) for u in inner._object_urdf_paths],
            "object_asset_index": inner._object_asset_index_per_env.cpu().tolist(),
            "goals_reached": int(rec["is_success"].sum()), "episodes_ended": int(rec["done"].sum()),
        }
        (out / "rollout_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"[rec] wrote {out / 'rollout.npz'}: {meta['goals_reached']} goals reached, "
              f"{meta['episodes_ended']} episodes ended", flush=True)

        # ---------------------------------------------------------------- SAPG block sweep
        coefs = [float(c) for c in args.sweep_coefs.split(",") if c.strip()]
        results = []
        for coef in coefs:
            pl = player(coef)
            obs = env.reset()[0]["policy"]
            pl.reset_rnn()
            goals = torch.zeros(n, device=dev)
            ended = falls = 0
            for _ in range(args.sweep_steps):
                with torch.no_grad():
                    action = pl.get_action(obs, deterministic=True)
                obs_d, _r, term, trunc, _i = env.step(action)
                goals += inner._is_success.float()
                done = term | trunc
                falls += int(inner._termination_reasons["fall"].sum())
                ended += int(done.sum())
                if done.any():
                    pl.reset_rnn(done.nonzero(as_tuple=False).squeeze(-1))
                obs = obs_d["policy"]
            minutes = args.sweep_steps / 60.0 / 60.0
            r = {"coef": coef, "goals_per_env_minute": float(goals.mean()) / minutes,
                 "episodes_ended": ended, "falls": falls}
            results.append(r)
            print(f"[rec] sweep {r}", flush=True)
        if results:
            (out / "sapg_sweep.json").write_text(json.dumps(results, indent=1) + "\n")

    run()
    os._exit(0)  # Kit shutdown hangs on these nodes


if __name__ == "__main__":
    main()
