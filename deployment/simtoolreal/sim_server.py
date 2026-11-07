"""Isaac Sim stand-in for the real robot and the object tracker, for run.py --robot isaac.

The training env (branch kushal/2026-10-30_pretraining_real) built for one object and driven as a
plant: each request carries the joint targets run.py computed (canonical order); the server holds
them for one policy step (2 physics steps of 1/120 s, through env.step's own replay path, so the
stepping is exactly training's) and answers with what the real loop would measure -- joint angles
and velocities, and the object pose. The goal run.py sends is only drawn (goal_viz). The policy,
action pipeline, goals and success test all live in run.py, as on the real robot.

Randomization is off (delays and noise are run.py's job, --obs_delay / --action_delay); the object's
start pose is random as in training. The env's own goal hits and timeouts are disabled; it still
resets itself when the object falls off the table or the hand ends up far from it, and reports that
as "reset_by_sim" (on the real robot that is a person putting the object back).

    cd <branch worktree>
    PYTHONPATH=$PWD:$PWD/rl_games .venv_isaacsim/bin/python \
        /home/yiboc/mit/task_curriculum_deployment/deployment/simtoolreal/sim_server.py --headless --port 6060
"""

from __future__ import annotations

import argparse
import os
import sys
import time

TASK = "Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0"
AUTHKEY = b"simtoolreal"


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=6060)
    p.add_argument("--object_type", default="eraser",
                   help="procedural object type (hammer screwdriver marker spatula eraser brush)")
    p.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app  # noqa: F841

    from multiprocessing.connection import Listener

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.protocol import disable_randomization
    from isaacsimenvs.tasks.play.utils.object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(TASK, "")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.assets.handle_head_types = [args.object_type]
        env_cfg.assets.num_assets_per_type = 1
        n_dist = sum(d.type == args.object_type for d in OBJECT_SIZE_DISTRIBUTIONS)
        assert n_dist >= 1, f"no size distribution for {args.object_type}"
        env_cfg.scene.num_envs = n_dist  # one object per distribution; env 0 is the one driven
        env_cfg.seed = args.seed
        disable_randomization(env_cfg)
        env_cfg.reset.fixed_start_pose = None
        env_cfg.reset.reset_position_noise_x = 0.1
        env_cfg.reset.reset_position_noise_y = 0.1
        # The env must not end episodes or move goals on its own: run.py owns both.
        env_cfg.episode_length_s = 1.0e6
        env_cfg.termination.episode_length = 10 ** 9
        env_cfg.termination.max_consecutive_successes = 0

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        perm_l2c, perm_c2l = inner._perm_lab_to_canon, inner._perm_canon_to_lab
        origins = inner.scene.env_origins
        zero_action = torch.zeros(n, int(inner.cfg.action_space), device=dev)

        def state(reset_by_sim: bool = False) -> dict:
            d = inner.robot.data
            return {
                "q": d.joint_pos[0, perm_l2c].double().cpu().tolist(),
                "qd": d.joint_vel[0, perm_l2c].double().cpu().tolist(),
                "object_pos": (inner.object.data.root_pos_w[0] - origins[0]).double().cpu().tolist(),
                "object_quat": inner.object.data.root_quat_w[0].double().cpu().tolist(),
                "sim_time": float(inner._sim_step_counter * inner.physics_dt),
                "reset_by_sim": reset_by_sim,
            }

        def set_goal(goal) -> None:
            if goal is None:
                return
            pos, quat = (torch.as_tensor(np.asarray(v), device=dev, dtype=torch.float32) for v in goal)
            pose = torch.cat([pos + origins[0], quat]).unsqueeze(0).expand(n, -1).clone()
            pose[:, :3] += origins - origins[0]
            inner.goal_viz.write_root_pose_to_sim(pose)

        env.reset()
        info = {
            "task": TASK, "object_type": args.object_type,
            "object_urdf": str(inner._object_urdf_paths[int(inner._object_asset_index_per_env[0])]),
            "object_scales": (inner._object_scale_per_env[0] * inner._object_scale_multiplier[0]).cpu().tolist(),
            "object_corners": inner._object_collision_corners[0].cpu().tolist(),
            "table_top_z": float(inner._table_z_per_env[0] + inner._table_half_height),
            "joint_names": [inner.robot.joint_names[i] for i in perm_l2c.tolist()],
            "policy_dt": float(inner.step_dt),
        }
        print(f"[sim_server] {TASK}, object {info['object_urdf']} scales {info['object_scales']}", flush=True)
        # Only plain Python values cross the socket: the two venvs pickle numpy differently (1.x vs 2.x).

        with Listener(("0.0.0.0", args.port), authkey=AUTHKEY) as listener:
            print(f"[sim_server] listening on port {args.port}", flush=True)
            while True:
                conn = listener.accept()
                print(f"[sim_server] client {listener.last_accepted}", flush=True)
                steps, t0 = 0, time.perf_counter()
                try:
                    while True:
                        req = conn.recv()
                        cmd = req["cmd"]
                        if cmd == "info":
                            conn.send(info)
                        elif cmd == "reset":
                            env.reset()
                            set_goal(req.get("goal"))
                            conn.send(state())
                        elif cmd == "step":
                            tgt = torch.as_tensor(np.asarray(req["targets"]), device=dev, dtype=torch.float32)
                            inner._replay_target_lab_order = tgt[perm_c2l].unsqueeze(0).expand(n, -1).clone()
                            set_goal(req.get("goal"))
                            _obs, _r, term, trunc, _i = env.step(zero_action)
                            steps += 1
                            conn.send(state(reset_by_sim=bool((term | trunc)[0])))
                        elif cmd == "close":
                            conn.send({"ok": True})
                            break
                        else:
                            conn.send({"error": f"unknown cmd {cmd}"})
                except EOFError:
                    pass
                finally:
                    conn.close()
                    dt = time.perf_counter() - t0
                    print(f"[sim_server] client done: {steps} steps in {dt:.1f} s "
                          f"({steps / max(dt, 1e-9):.0f} steps/s)", flush=True)
                if req.get("shutdown"):
                    break

    run()
    os._exit(0)  # Kit shutdown hangs on these nodes


if __name__ == "__main__":
    main()
