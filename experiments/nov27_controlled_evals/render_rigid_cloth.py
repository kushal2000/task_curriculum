"""Render a point-cloud policy on the rigid-cloth evaluation world (PhysX plates, eval/rigid_cloth.yaml) to MP4.

    /share/portal/kk837/task_curriculum/.venv/isaacsim/bin/python \
        experiments/nov27_controlled_evals/render_rigid_cloth.py --headless --enable_cameras \
        --checkpoint pretrained_policy/play_pc_rigidclothRL_64live.pth --episodes 4 --out outputs/renders/rigid_pin_64live

The controlled protocol of eval_pinned_rigid_cloth.sbatch: pinned start (scripts/evaluation/protocol.py
disable_randomization), sampled actions. Env 0 is filmed for --episodes consecutive episodes (each a pinned
reset); every frame is labelled with the step and the Hausdorff distance to the goal configuration of the
cloth (aligned, layered), the clip ends on the episode's outcome.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pretrained_policy/play_pc_rigidclothRL_64live.pth")
    p.add_argument("--experiment", default="isaacsimenvs/tasks/cloth/config/eval/rigid_cloth.yaml")
    p.add_argument("--label", default="PhysX rigid plates, pinned")
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="outputs/renders/rigid_pin")
    AppLauncher.add_app_launcher_args(p)
    args, extra = p.parse_known_args()
    args.enable_cameras = True
    os.chdir(ROOT)
    sys.argv = [sys.argv[0], *extra]
    AppLauncher(args)

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
    from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud, _student_checkpoint
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    task = "Isaacsimenvs-RigidCloth-Direct-v0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student = _student_checkpoint(args.checkpoint)
    torch.manual_seed(args.seed)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    @hydra_task_config_with_yaml(task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = "cuda:0"
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (args.width, args.height)
        disable_randomization(env_cfg)  # the pinned start, no observation / action DR
        _check_student_cloud(env_cfg, student["env"])
        env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        inner = env.unwrapped
        player = EnvCloudStudentPlayer(inner, student, "cuda:0")
        fps = int(round(1.0 / float(inner.step_dt)))
        origin = inner.scene.env_origins[0]
        tol_mm = 1000 * float(inner.task.surface_tolerance()[0])

        def aim() -> None:
            c = (inner.cloth_points_w()[0].mean(0) - origin).tolist()
            inner.viewport_camera_controller.update_view_location(
                eye=(c[0] + 0.22, c[1] - 0.30, c[2] + 0.22), lookat=(c[0], c[1], c[2]))

        def label(img, step, dist_mm, best_mm, outcome=None):
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, im.width, 66), fill=(0, 0, 0))
            d.text((10, 5), args.label, fill=(255, 255, 255), font=font)
            ok = best_mm < tol_mm
            d.text((10, 36), f"step {step} | Hausdorff {dist_mm:.0f} mm (best {best_mm:.0f})"
                   + (f" | {outcome.replace('_', ' ')}" if outcome else ""), fill=(120, 230, 120) if ok else (220, 220, 220), font=font)
            return np.asarray(im)

        names = list(inner.robot.body_names)
        rec = {"body_q": [], "cloth": [], "haus": [], "err": []}

        def record(dist_mm: float) -> None:
            """Env 0's bodies (position, quaternion xyzw) and the sheet's 7 x 7 grid, env-relative, plus the table."""
            pos = (inner.robot.data.body_pos_w[0] - origin).cpu().numpy()
            wxyz = inner.robot.data.body_quat_w[0].cpu().numpy()
            tpos = (inner.table.data.root_pos_w[0] - origin).cpu().numpy()
            twxyz = inner.table.data.root_quat_w[0].cpu().numpy()
            q = np.concatenate([np.concatenate([pos, wxyz[:, 1:], wxyz[:, :1]], 1),
                                np.concatenate([tpos, twxyz[1:], twxyz[:1]])[None]], 0)
            rec["body_q"].append(q.astype(np.float32))
            rec["cloth"].append((inner.fold_view().points_w[0] - origin).cpu().numpy().astype(np.float32))
            rec["haus"].append(dist_mm)
            rec["err"].append(1000 * float(inner.fold_error()[0]))

        def write_record(path: Path, outcome: str) -> None:
            np.savez_compressed(
                path, outcome=outcome, seed=args.seed, env=0, policy_dt=float(inner.step_dt), checkpoint=args.checkpoint,
                protocol=args.experiment, sim="physx", label=args.label,
                body_label=np.array([f"/World/envs/env_0/Robot/{b}" for b in names] + ["/World/envs/env_0/Table/box"]),
                body_q=np.stack(rec["body_q"]), cloth=np.stack(rec["cloth"]), haus_mm=np.array(rec["haus"]),
                fold_err_mm=np.array(rec["err"]), tri_moving=np.zeros((0, 3), np.int32), tri_stationary=np.zeros((0, 3), np.int32))
            for k in rec:
                rec[k] = []

        obs, _ = env.reset()
        aim()
        frames, ep_step, best, clip = [], 0, float("inf"), 0
        for _ in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=False)
            obs, _r, terminated, truncated, extras = env.step(action)
            ended = terminated.bool() | truncated.bool()
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            ep_step += 1
            if not bool(ended[0]):
                dist = 1000 * float(inner.task.surface_dist[0])
                if np.isfinite(dist):
                    best = min(best, dist)
                record(dist)
                raw = env.render()
                last = (ep_step, dist)
                frames.append(label(raw, ep_step, dist, best))
                continue
            final = extras["episode_final"]
            outcome = next((k[5:] for k in ("done_fold_held", "done_fall", "done_hand_far", "done_timeout")
                            if k in final and bool(final[k][0])), "timeout" if bool(truncated[0]) else "?")
            if frames:
                frames += [label(raw, last[0], last[1], best, outcome)] * fps
                clip += 1
                path = out / f"rigid_seed{args.seed}_ep{clip}_{outcome}.mp4"
                imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=8)
                write_record(path.with_suffix(".npz"), outcome)
                print(f"[render] wrote {path} ({len(frames)} frames, best Hausdorff {best:.1f} mm, {outcome})", flush=True)
            if clip >= args.episodes:
                break
            frames, ep_step, best = [], 0, float("inf")
            for k in rec:
                rec[k] = []
            aim()
        env.close()

    run()
    os._exit(0)  # Isaac Sim can hang in its plugin shutdown after the clips are written


if __name__ == "__main__":
    main()
