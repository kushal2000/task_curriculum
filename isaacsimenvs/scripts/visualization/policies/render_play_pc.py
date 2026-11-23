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
    p.add_argument("--log_contact", action="store_true",
                   help="log, every 10 steps, how deep the robot's surface reaches into the tool (its SDF over the "
                        "robot's surface samples) and the tool's speed; and a per-episode summary")
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

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    # The env config of the student's own task (its gym registration), e.g. the real-limits variant.
    mod, cls = gym.spec(ck["env"]["task"]).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(__import__(mod, fromlist=[cls]), cls)()
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
    contact_log = []

    class _Sub:
        """One env's slice of EnvShapes, for its sdf."""
        def __init__(self, es, i):
            sl = slice(i, i + 1)
            self.pos, self.rot, self.half, self.is_box, self.pad = es.pos[sl], es.rot[sl], es.half[sl], es.is_box[sl], es.pad[sl]
    for t in range(int(args.steps)):
        pol = obs["policy"]
        with torch.no_grad():
            mu, _aux, state = net.step(pol[:, :n_p], pol[:, n_p:n_p + 3 * m].reshape(-1, m, 3),
                                       pol[:, n_p + 3 * m:n_p + 4 * m] > 0.5, state, start)
        obs, _r, term, trunc, extras = env.step(mu.float().clamp(-1.0, 1.0))
        if args.log_contact:
            # Robot surface samples (the camera model's occluders), world -> env -> tool frame, then the SDF.
            from isaacsimenvs.tasks.play.utils.object_surface import matrix_from_quat
            cl, org = inner.cloud, inner.scene.env_origins[e]
            bp = inner.robot.data.body_pos_w[e] - org
            bq = inner.robot.data.body_quat_w[e]
            pts = bp[cl.pool_body] + torch.matmul(matrix_from_quat(bq[cl.pool_body]), cl.pool_off.unsqueeze(-1)).squeeze(-1)
            o_pos = inner.object.data.root_pos_w[e] - org
            local = torch.matmul(pts - o_pos, matrix_from_quat(inner.object.data.root_quat_w[e:e + 1])[0])
            sd = inner.env_shapes.sdf.__func__(_Sub(inner.env_shapes, e), local.unsqueeze(0))[0] * 100.0
            depth = float((-sd).clamp(min=0).max())
            body = inner.robot.body_names[int(cl.pool_body[int(sd.argmin())])]
            speed = float(inner.object.data.root_lin_vel_w[e].norm()) * 100.0
            contact_log.append(depth)
            if t % 10 == 0 or depth > 0.5:
                print(f"[contact] t={t:3d} deepest robot point {depth:.2f} cm inside the tool ({body}), "
                      f"nearest {float(sd.min()):.2f} cm, tool speed {speed:.1f} cm/s", flush=True)
        start = term.bool() | trunc.bool()
        goals = int(inner._successes[e]) if not bool(start[e]) else goals
        if t % 2 == 0:
            frames.append(env.render())
        if bool(start[e]):
            print(f"[render] episode end at step {t}: {goals} goals", flush=True)
    if args.log_contact and contact_log:
        c = torch.tensor(contact_log)
        print(f"[contact] summary: deepest {c.max():.2f} cm; steps with a robot point > 0.5 cm inside: "
              f"{int((c > 0.5).sum())} / {len(c)}; > 1 cm: {int((c > 1.0).sum())}", flush=True)
    imageio.mimsave(args.out, frames, fps=30, quality=8)
    print(f"[render] wrote {len(frames)} frames to {args.out}; goals this episode so far {int(inner._successes[e])}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
