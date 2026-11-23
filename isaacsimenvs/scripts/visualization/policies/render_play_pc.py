"""Render a point-cloud Play student in Isaac Sim (RTX camera) to an mp4: one env, one tool.

    python -m isaacsimenvs.scripts.visualization.policies.render_play_pc --headless --enable_cameras \
        --student debug_outputs/train_logs/nov/<run>/student_last.pth --out <video>.mp4 [--tool 0] [--steps 900]

The student (``student.SplitStudentNet``, settings from its checkpoint) drives alone, deterministic, with
the env's DR as trained (camera errors, delays, noise). The robot keeps its visual meshes
(``assets.keep_robot_visuals``); the authored tool is brown, its goal translucent green. ``--tool`` picks
the pool entry (``author_tools``: env i holds tool i mod pool, so the scene is ``--tool + 1`` envs and
the camera frames the last one). Policy at 60 Hz, video at 30 fps (every other step).
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tool", type=int, default=0)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import imageio
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student import SplitStudentNet
    from isaacsimenvs.tasks.play.play_pc_env_cfg import PlayPCEnvCfg

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    cfg = PlayPCEnvCfg()
    cfg.scene.num_envs = int(args.tool) + 1
    cfg.scene.env_spacing = 3.0
    cfg.seed = int(args.seed)
    cfg.assets.keep_robot_visuals = True
    cfg.termination.eval_success_tolerance = 0.01      # as the student was distilled
    cfg.viewer.resolution = (int(args.width), int(args.height))
    env = gym.make(ck["env"]["task"], cfg=cfg, render_mode="rgb_array")
    inner = env.unwrapped
    e = int(args.tool)
    # Frame the chosen env: the robot looks along -y at the table; the camera sits in front, to the side.
    org = inner.scene.env_origins[e].cpu().numpy()
    inner.sim.set_camera_view(eye=(org + np.array([0.75, -0.55, 0.85])).tolist(),
                              target=(org + np.array([0.0, 0.32, 0.30])).tolist())
    lay = inner.layout
    n_p, m = lay["n_proprio"], lay["num_points"]
    net = SplitStudentNet(n_p, int(inner.cfg.action_space), 3, int(ck["model"]["aux.2.weight"].shape[0]), **{k: v for k, v in ck["net_args"].items()}).to(inner.device)
    net.load_state_dict(ck["model"])
    net.eval()
    tool = inner._object_urdf_paths[int(inner._object_asset_index_per_env[e])].rsplit("/", 1)[-1]
    print(f"[render] student {args.student} (iter {ck.get('iter')}), env {e}, tool {tool}", flush=True)

    obs, _ = env.reset()
    state = net.initial_state(inner.num_envs, inner.device)
    start = torch.ones(inner.num_envs, dtype=torch.bool, device=inner.device)
    frames, goals = [], 0
    for t in range(int(args.steps)):
        pol = obs["policy"]
        with torch.no_grad():
            mu, _aux, state = net.step(pol[:, :n_p], pol[:, n_p:n_p + 3 * m].reshape(-1, m, 3),
                                       pol[:, n_p + 3 * m:n_p + 4 * m] > 0.5, state, start)
        obs, _r, term, trunc, extras = env.step(mu.float().clamp(-1.0, 1.0))
        start = term.bool() | trunc.bool()
        goals = int(inner._successes[e]) if not bool(start[e]) else goals
        if t % 2 == 0:
            frames.append(env.render())
        if bool(start[e]):
            print(f"[render] episode end at step {t}: {goals} goals", flush=True)
    imageio.mimsave(args.out, frames, fps=30, quality=8)
    print(f"[render] wrote {len(frames)} frames to {args.out}; goals this episode so far {int(inner._successes[e])}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
