"""Render rollouts of a point-cloud policy on the rigid sheet and keep episodes where a finger goes into a plate.

    python experiments/nov25_better_rigidcloth/eval/render_penetration.py --headless --enable_cameras \
        --rest_mm 1 --clips 2 --out outputs/penetration_videos

Env 0 is filmed (camera on its sheet, re-aimed at every reset); every frame is labelled with the deepest
finger-in-plate depth this step (FingerPenetration.read: past the plate's collider, beyond the rest
offset shell) and turns red above --flag_mm. The finger-penetration rule is set too large to fire
(1 m), so the episode plays out. An episode is kept when its deepest depth exceeds --flag_mm.
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
    p.add_argument("--checkpoint", default="pretrained_policy/play_pc_rigidclothRL_64live.pth")
    p.add_argument("--experiment", default="experiments/nov24_play_pc_rl/play_pc_rl_A.yaml")
    p.add_argument("--shapes", default="box2-mid")
    p.add_argument("--rest_mm", type=float, default=1.0, help="fixed rest offset (contact shell) [mm]")
    p.add_argument("--flag_mm", type=float, default=0.5, help="the rule's threshold, for the label and selection")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="outputs/penetration_videos")
    AppLauncher.add_app_launcher_args(p)
    args, extra = p.parse_known_args()
    args.enable_cameras = True
    os.chdir(ROOT)
    rest = args.rest_mm / 1e3
    sys.argv = [sys.argv[0],
                "env.cloth.keypoint_tolerance=0.015", "env.cloth.nan_policy=reset", "env.termination.success_steps=10",
                f"env.fold_pool.shapes=[{args.shapes}]", "env.fold_pool.one_shape_per_env=true",
                "env.fold_pool.init_fold_prob=0.0", "env.fold_pool.goal_task.mode=half_fold",
                f"env.rigid_cloth.contact_margin={rest}", "env.domain_randomization.cloth.contact_margin_offset=[0.0,0.0]",
                "env.termination.max_finger_penetration=1.0", *extra]
    AppLauncher(args)

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
    from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud, _student_checkpoint
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    task = "Isaacsimenvs-RigidCloth-Direct-v0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student = _student_checkpoint(args.checkpoint)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
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
        _check_student_cloud(env_cfg, student["env"])
        env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        inner = env.unwrapped
        player = EnvCloudStudentPlayer(inner, student, "cuda:0")
        fps = int(round(1.0 / float(inner.step_dt)))
        origin = inner.scene.env_origins[0]

        def aim() -> None:
            c = (inner.cloth_points_w()[0].mean(0) - origin).tolist()
            inner.viewport_camera_controller.update_view_location(
                eye=(c[0] + 0.22, c[1] - 0.30, c[2] + 0.22), lookat=(c[0], c[1], c[2]))

        def label(img, step, depth_mm, ep_max_mm, outcome=None):
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            bad = depth_mm > args.flag_mm
            col = (255, 40, 40) if bad else (255, 255, 255)
            d.rectangle((0, 0, im.width, 78), fill=(0, 0, 0))
            d.text((16, 8), f"rest offset {args.rest_mm:g} mm | step {step} | finger depth in plate {depth_mm:5.2f} mm"
                   + ("  > PENETRATION" if bad else ""), fill=col, font=font)
            d.text((16, 42), f"episode max {ep_max_mm:5.2f} mm   (rule threshold {args.flag_mm:g} mm, disabled here)"
                   + (f"   outcome: {outcome}" if outcome else ""), fill=(200, 200, 200), font=font)
            if bad:
                d.rectangle((0, 0, im.width - 1, im.height - 1), outline=(255, 40, 40), width=8)
            return np.asarray(im)

        obs, _ = env.reset()
        aim()
        frames, ep_max, ep_step, kept, clip_id = [], 0.0, 0, 0, 0
        for step in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, terminated, truncated, extras = env.step(action)
            depth = float(inner.backend.penetration.read()[0]) * 1e3
            ended = terminated.bool() | truncated.bool()
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            ep_step += 1
            if not bool(ended[0]):
                ep_max = max(ep_max, depth)
                last_depth = depth
                raw = env.render()
                frames.append(label(raw, ep_step, depth, ep_max))
                continue
            final = extras["episode_final"]
            outcome = next((k[5:] for k in ("done_fold_held", "done_fall", "done_hand_far", "done_timeout")
                            if k in final and bool(final[k][0])), "timeout" if bool(truncated[0]) else "?")
            if frames:  # hold the episode's last frame for a second, with its outcome
                frames += [label(raw, ep_step - 1, last_depth, ep_max, outcome)] * fps
            clip_id += 1
            if ep_max > args.flag_mm and frames:
                path = out / f"rest{args.rest_mm:g}mm_ep{clip_id}_max{ep_max:.1f}mm_{outcome}.mp4"
                imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=8)
                kept += 1
                print(f"[render] wrote {path} ({len(frames)} frames, max depth {ep_max:.2f} mm, {outcome})", flush=True)
            else:
                print(f"[render] episode {clip_id}: max depth {ep_max:.2f} mm ({outcome}), not kept", flush=True)
            if kept >= args.clips:
                break
            frames, ep_max, ep_step = [], 0.0, 0
            aim()
        env.close()

    run()


if __name__ == "__main__":
    main()
