"""Render a point-cloud policy on the VBD evaluation world (Newton) to MP4, with Newton's headless GL viewer.

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/isaaclab3/bin/python \
        experiments/nov25_better_rigidcloth/eval/render_vbd.py --protocol isaacsimenvs/tasks/cloth/config/eval/vbd.yaml \
        --num_envs 4 --episodes 1 --seed 1 [--randomize] --out outputs/vbd_renders

The env is built as ``eval.py --protocol`` builds it (the student's cloud from its checkpoint, its DR only
with --randomize, goal students scored by the surface criterion). Each env's first episode is filmed
from a camera over that env; the clip is labelled with the step, the fold error and the outcome.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
    p.add_argument("--checkpoint", default="pretrained_policy/play_pc_rigidclothRL_64live.pth")
    p.add_argument("--protocol", default="isaacsimenvs/tasks/cloth/config/eval/vbd.yaml")
    p.add_argument("--experiment", default=None,
                   help="build from this training recipe instead of --protocol (e.g. the plate pool on Newton: "
                        "--task Isaacsimenvs-RigidClothNewton-Direct-v0 --experiment <recipe>)")
    p.add_argument("--label", default="VBD eval", help="the text at the start of every frame's label")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--max_steps", type=int, default=900)
    p.add_argument("--stochastic", action="store_true", help="sample actions (seeded by --seed) instead of the mean")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="outputs/vbd_renders")
    p.add_argument("--record", action="store_true",
                   help="also save each env's first episode as <out>/vbd_seed<s>_env<e>_<outcome>.npz for viser_replay.py")
    p.add_argument("--no_video", action="store_true", help="skip the MP4s (much faster; use with --record)")
    AppLauncher.add_app_launcher_args(p)
    args, extra = p.parse_known_args()
    os.chdir(ROOT)
    sys.argv = [sys.argv[0], *extra]
    try:
        AppLauncher(args)
    except ImportError as exc:  # the kit-less Newton backend, as in eval.py
        print(f"[render_vbd] running kit-less: {exc}", flush=True)

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student import apply_student_env
    from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
    from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud, _goal_student_success, _student_checkpoint
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    student = _student_checkpoint(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    @hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point" if args.experiment else "",
                                 experiment=args.experiment or args.protocol)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = "cuda:0"
        if not args.randomize:
            disable_randomization(env_cfg)
        if args.experiment:  # as eval.py with a recipe: the recipe's cloud, checked against the student's
            _check_student_cloud(env_cfg, student["env"])
            _goal_student_success(env_cfg)
        else:
            apply_student_env(env_cfg, student["env"], with_dr=args.randomize)
            _goal_student_success(env_cfg)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n = inner.num_envs
        player = EnvCloudStudentPlayer(inner, student, "cuda:0")
        fps = int(round(1.0 / float(inner.step_dt)))

        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        import newton  # noqa: F401
        from newton.viewer import ViewerGL

        model = NewtonManager._model
        viewer = ViewerGL(width=args.width, height=args.height, headless=True)
        # The robot and table come from collision-only USD shapes: draw collision and static shapes too.
        viewer.show_collision = True
        viewer.show_static = True
        viewer.set_model(model)
        # Newton's viewer spreads worlds apart for display, but only the rigid shapes (robot, table):
        # the cloth particles are drawn where they are. Isaac Lab already places the envs apart, so
        # no display offset -- otherwise every env but 0 shows its robot away from its cloth.
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        print(f"[render_vbd] model: {model.body_count} bodies, {model.shape_count} shapes, {model.particle_count} particles, "
              f"{model.world_count if hasattr(model, 'world_count') else '?'} worlds", flush=True)
        origins = inner.scene.env_origins.cpu().numpy()

        def state():
            return NewtonManager._state_0

        from isaacsimenvs.tasks.cloth.mdp.robot import to_torch

        palm_id = inner.robot.find_bodies(inner.robot_spec.palm_body_name)[0][0]
        view_dir = np.array([0.30, -0.75, 0.60]) / np.linalg.norm([0.30, -0.75, 0.60])
        cam = {}  # per env: smoothed (target, distance), so the camera does not jitter

        def frame_of(e: int) -> np.ndarray:
            centre = inner.cloth_points_w()[e].mean(0).cpu().numpy()
            palm = to_torch(inner.robot.data.body_pos_w)[e, palm_id].cpu().numpy()
            # Aim at the midpoint of the sheet and the palm, far enough back to frame both, smoothed.
            # (Newton camera: yaw about +z from +x, pitch up from horizontal, degrees)
            import math
            t_new = 0.5 * (centre + palm)
            d_new = max(0.55, 1.4 * float(np.linalg.norm(palm - centre)) + 0.35)
            t_old, d_old = cam.get(e, (t_new, d_new))
            target, dist = 0.85 * t_old + 0.15 * t_new, 0.85 * d_old + 0.15 * d_new
            cam[e] = (target, dist)
            eye = target + dist * view_dir
            dx, dy, dz = (target - eye).tolist()
            yaw = math.degrees(math.atan2(dy, dx))
            pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
            viewer.set_camera(pos=wp.vec3(*[float(v) for v in eye]), pitch=pitch, yaw=yaw)
            viewer.begin_frame(0.0)
            viewer.log_state(state())
            viewer.end_frame()
            img = viewer.get_frame().numpy()
            return img[..., :3].astype(np.uint8)

        def label(img, e, step, err_mm, outcome=None):
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, im.width, 34), fill=(0, 0, 0))
            d.text((10, 6), f"{args.label} | env {e} | step {step} | fold error {err_mm:5.1f} mm"
                   + (f" | {outcome}" if outcome else ""), fill=(255, 255, 255), font=font)
            return np.asarray(im)

        # --record: the drawable shapes of each env (collision geometry, body frame) once, and per step the
        # bodies' poses and the cloth's vertices, env-relative; written per env when its episode ends.
        rec = {}
        if args.record:
            from isaacsimenvs.scripts.visualization.live_viser.scene import NewtonScene

            scene = NewtonScene(inner)
            owner = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int64)
            tri = model.tri_indices.numpy() if model.tri_count else np.zeros((0, 3), np.int64)
            moving = inner.task.geometry.moving_idx.cpu().numpy()
            for e in range(n):
                shapes = scene.shapes(e)
                keys = sorted({sh.key for sh in shapes if sh.key is not None})
                pidx = np.nonzero(owner == e)[0]
                if len(pidx):
                    t = tri[(owner[tri] == e).all(axis=1)] - int(pidx[0])
                    in_moving = np.isin(t, moving).sum(axis=1) >= 2
                else:  # no particles (the plate pool): the plates are bodies among the shapes
                    t = np.zeros((0, 3), np.int64)
                    in_moving = np.zeros(0, bool)
                    for sh in shapes:
                        if "FoldPool" in sh.label:
                            sh.rgb = (240, 150, 60)
                rec[e] = {"shapes": shapes, "keys": keys, "pidx": torch.as_tensor(pidx, device=inner.device),
                          "tri_moving": t[in_moving], "tri_stationary": t[~in_moving],
                          "body_q": [], "cloth": [], "err": []}

        def record_step(e: int, err_mm: float) -> None:
            r = rec[e]
            q = wp.to_torch(state().body_q)[torch.as_tensor(r["keys"], device=inner.device)].cpu().numpy().copy()
            q[:, :3] -= origins[e]
            r["body_q"].append(q.astype(np.float32))  # (B, 7): position, quaternion xyzw
            pts = (wp.to_torch(state().particle_q)[r["pidx"]].cpu().numpy() - origins[e]) if len(r["pidx"]) \
                else np.zeros((0, 3), np.float32)
            r["cloth"].append(pts.astype(np.float32))
            r["err"].append(err_mm)

        def write_record(e: int, outcome: str) -> None:
            r = rec[e]
            if not r["cloth"]:
                return
            sh = r["shapes"]
            path = out / f"vbd_seed{args.seed}_env{e}_{outcome}.npz"
            np.savez_compressed(
                path, outcome=outcome, seed=args.seed, env=e, policy_dt=float(inner.step_dt),
                checkpoint=args.checkpoint, protocol=args.protocol,
                shape_key=np.array([-1 if x.key is None else r["keys"].index(x.key) for x in sh]),
                shape_kind=np.array([x.kind for x in sh]), shape_label=np.array([x.label for x in sh]),
                shape_pos=np.array([x.pos - (origins[e] if x.key is None else 0.0) for x in sh]), shape_wxyz=np.array([x.wxyz for x in sh]),
                shape_rgb=np.array([x.rgb for x in sh]),
                shape_verts=np.array([x.verts.astype(np.float32) for x in sh], dtype=object),
                shape_faces=np.array([x.faces.astype(np.int32) for x in sh], dtype=object),
                body_label=np.array([str(model.body_label[k]) for k in r["keys"]]),
                body_q=np.stack(r["body_q"]), cloth=np.stack(r["cloth"]), fold_err_mm=np.array(r["err"]),
                tri_moving=r["tri_moving"], tri_stationary=r["tri_stationary"], env_origin=origins[e])
            print(f"[render_vbd] wrote {path} ({len(r['cloth'])} steps, {outcome})", flush=True)
            for k in ("body_q", "cloth", "err"):
                r[k] = []

        obs, _ = env.reset()
        torch.manual_seed(args.seed)  # the same action noise in every run of this seed
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=inner.device))
        player.reset_rnn(torch.arange(n, device=inner.device))
        frames = {e: [] for e in range(n)}
        done = torch.zeros(n, dtype=torch.bool, device=inner.device)
        for step in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=not args.stochastic)
            obs, _r, terminated, truncated, extras = env.step(action)
            ended = terminated.bool() | truncated.bool()
            err = inner.fold_error().cpu().numpy() * 1e3
            for e in (~done & ~ended).nonzero(as_tuple=True)[0].tolist():
                if args.record:
                    record_step(e, float(err[e]))
                if args.no_video:
                    continue
                frames[e].append(label(frame_of(e), e, step + 1, float(err[e])))
                if step in (20, 120):
                    imageio.imwrite(out / f"peek_seed{args.seed}_env{e}_step{step}.png", frames[e][-1])
            fin = ended & ~done
            if bool(fin.any()):
                final = extras["episode_final"]
                for e in fin.nonzero(as_tuple=True)[0].tolist():
                    outcome = next((k[5:] for k in ("done_fold_held", "done_fall", "done_hand_far", "done_nonfinite")
                                    if k in final and bool(final[k][e])), "timeout" if bool(truncated[e]) else "?")
                    if args.record:
                        write_record(e, outcome)
                    if frames[e]:
                        last = frames[e][-1]
                        frames[e] += [last] * fps
                        path = out / f"vbd_seed{args.seed}_env{e}_{outcome}.mp4"
                        imageio.mimwrite(path, frames[e], fps=fps, quality=8, macro_block_size=8)
                        print(f"[render_vbd] wrote {path} ({len(frames[e])} frames, {outcome})", flush=True)
                    frames[e] = []
                done |= fin
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if bool(done.all()):
                break
        if args.record:  # episodes still running at --max_steps
            for e in (~done).nonzero(as_tuple=True)[0].tolist():
                write_record(e, "running")
        env.close()

    run()


if __name__ == "__main__":
    main()
