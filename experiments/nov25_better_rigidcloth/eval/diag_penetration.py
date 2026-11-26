"""Cross-check the finger-penetration readout against the geometry.

    python experiments/nov25_better_rigidcloth/eval/diag_penetration.py --headless --rest_mm 1 --events 40

Runs the 64live policy (rule disabled) and, whenever an env reads deeper than --flag_mm, dumps every raw
finger-plate contact of that env: finger link, plate body, PhysX separation, normal and point -- the
point and normal also in the plate's frame, against the plate's half thickness. Also prints the rest /
contact offsets PhysX actually holds for the finger links and the plates.
"""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--rest_mm", type=float, default=1.0)
    p.add_argument("--flag_mm", type=float, default=0.5)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--events", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--out", default="outputs/penetration_videos/diag.json")
    AppLauncher.add_app_launcher_args(p)
    args, extra = p.parse_known_args()
    os.chdir(ROOT)
    sys.argv = [sys.argv[0],
                "env.cloth.keypoint_tolerance=0.015", "env.cloth.nan_policy=reset", "env.termination.success_steps=10",
                f"env.fold_pool.shapes=[{args.shapes}]", "env.fold_pool.one_shape_per_env=true",
                "env.fold_pool.init_fold_prob=0.0", "env.fold_pool.goal_task.mode=half_fold",
                f"env.rigid_cloth.contact_margin={args.rest_mm / 1e3}",
                "env.domain_randomization.cloth.contact_margin_offset=[0.0,0.0]",
                "env.termination.max_finger_penetration=1.0", *extra]
    AppLauncher(args)

    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply_inverse

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
    from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud, _student_checkpoint
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    task = "Isaacsimenvs-RigidCloth-Direct-v0"
    student = _student_checkpoint(args.checkpoint)

    @hydra_task_config_with_yaml(task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = 1
        env_cfg.sim.device = "cuda:0"
        _check_student_cloud(env_cfg, student["env"])
        env = gym.make(task, cfg=env_cfg)
        inner = env.unwrapped
        b = inner.backend
        pen = b.penetration
        player = EnvCloudStudentPlayer(inner, student, "cuda:0")
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device))
        pen.read()  # builds the views

        # Offsets PhysX holds.
        robot = b.robot
        names = robot.body_names
        rv = robot.root_physx_view
        rest_r, con_r = rv.get_rest_offsets()[0], rv.get_contact_offsets()[0]
        print(f"[diag] robot shapes: rest offsets {sorted(set(round(float(v) * 1e3, 3) for v in rest_r))} mm, "
              f"contact offsets {sorted(set(round(float(v) * 1e3, 3) for v in con_r))} mm", flush=True)
        for art in b.arts:
            v = art.root_physx_view
            print(f"[diag] {art.cfg.prim_path.rsplit('/', 1)[-1]}: rest {sorted(set(round(float(x) * 1e3, 3) for x in v.get_rest_offsets().flatten()))} mm, "
                  f"contact {sorted(set(round(float(x) * 1e3, 3) for x in v.get_contact_offsets().flatten()))} mm, "
                  f"bodies {art.body_names}", flush=True)
        half_t = 0.5 * float(env_cfg.fold_pool.thickness_range[0])
        print(f"[diag] plate half thickness {half_t * 1e3:.2f} mm", flush=True)

        events = []
        distal = [i for i, n in enumerate(names) if n.endswith("_distal")]
        track = []   # per step: env, link, plate, link-origin z in plate frame [mm], min separation of that pair [mm]
        for step in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, terminated, truncated, _x = env.step(action)
            ended = terminated.bool() | truncated.bool()
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            depth = pen.read()
            # Every distal link's origin in each plate's frame, and the pair's min separation, every step.
            for (view, sensor_env), (s0, art0) in zip(pen.views, [(s, a) for s, a in enumerate(b.arts) if len(b.pool.members(s))]):
                from isaacsimenvs.tasks.cloth.backends.rigid import assets as fsa0
                pb = [n for n in art0.body_names if n.startswith(fsa0.BODY_PREFIX)]
                _f, _p, _n, sep0, cnt0, st0 = view.get_contact_data(dt=pen.physics_dt)
                cnt0, st0, sep0 = cnt0.long().cpu(), st0.long().cpu(), sep0.cpu().reshape(-1)
                links = [pth.rsplit("/", 1)[-1] for pth in view.sensor_paths]
                for si in range(cnt0.shape[0]):
                    link = links[si]
                    if not link.endswith("_distal"):
                        continue
                    e = int(sensor_env[si]); li = names.index(link)
                    lp = robot.data.body_pos_w[e, li]
                    _k, loc = b.pool.local_rows(s0, torch.tensor([e], device=inner.device))
                    for fi, body in enumerate(pb):
                        bi = art0.body_names.index(body)
                        pos, quat = art0.data.body_pos_w[int(loc[0]), bi], art0.data.body_quat_w[int(loc[0]), bi]
                        z = float(quat_apply_inverse(quat[None], (lp - pos)[None])[0, 2]) * 1e3
                        n = int(cnt0[si, fi])
                        ms = float(sep0[int(st0[si, fi]):int(st0[si, fi]) + n].min()) * 1e3 if n else None
                        if n or z < 40:
                            track.append([step, e, link, body, round(z, 3), None if ms is None else round(ms, 3)])
            flagged = (depth > args.flag_mm / 1e3).nonzero(as_tuple=True)[0].tolist()
            if not flagged:
                continue
            for (view, sensor_env), (s, art) in zip(pen.views, [(s, a) for s, a in enumerate(b.arts) if len(b.pool.members(s))]):
                _f, point, normal, sep, counts, starts = view.get_contact_data(dt=pen.physics_dt)
                counts, starts = counts.long().cpu(), starts.long().cpu()
                point, normal, sep = point.cpu(), normal.cpu(), sep.cpu().reshape(-1)
                from isaacsimenvs.tasks.cloth.backends.rigid import assets as fsa
                plate_bodies = [n for n in art.body_names if n.startswith(fsa.BODY_PREFIX)]  # the filter order
                for si in range(counts.shape[0]):
                    env_i = int(sensor_env[si])
                    if env_i not in flagged:
                        continue
                    link = view.sensor_paths[si].rsplit("/", 1)[-1]
                    for fi in range(counts.shape[1]):
                        for k in range(int(counts[si, fi])):
                            c = int(starts[si, fi]) + k
                            body = plate_bodies[fi]
                            _keep, local = b.pool.local_rows(s, torch.tensor([env_i], device=inner.device))
                            inst = int(local[0])
                            bi = art.body_names.index(body)
                            pos = art.data.body_pos_w[inst, bi]
                            quat = art.data.body_quat_w[inst, bi]
                            pt = torch.as_tensor(point[c], device=inner.device).float()
                            nm = torch.as_tensor(normal[c], device=inner.device).float()
                            pl = quat_apply_inverse(quat[None], (pt - pos)[None])[0].tolist()
                            nl = quat_apply_inverse(quat[None], nm[None])[0].tolist()
                            events.append({"step": step, "env": env_i, "depth_mm": round(float(depth[env_i]) * 1e3, 3),
                                           "link": link, "plate": body, "sep_mm": round(float(sep[c]) * 1e3, 3),
                                           "point_plate_frame_mm": [round(1e3 * x, 2) for x in pl],
                                           "normal_plate_frame": [round(x, 3) for x in nl],
                                           "plate_z_world_mm": round(float(pos[2] - inner.scene.env_origins[env_i, 2]) * 1e3, 2)})
            n_ev = len({(e["step"], e["env"]) for e in events})
            if n_ev >= args.events:
                break
        Path(args.out).write_text(json.dumps(events, indent=0))
        Path(args.out.replace(".json", "_track.json")).write_text(json.dumps(track))
        deep = [e for e in events if e["sep_mm"] < -args.flag_mm]
        print(f"[diag] {len(events)} contacts at {n_ev} flagged (step, env); {len(deep)} deeper than {args.flag_mm} mm", flush=True)
        for e in deep[:40]:
            print(f"[diag] step {e['step']:4d} env {e['env']:2d} {e['link']:28s} {e['plate']:12s} sep {e['sep_mm']:7.2f} mm "
                  f"point(plate) {e['point_plate_frame_mm']} normal(plate) {e['normal_plate_frame']}", flush=True)
        env.close()

    run()


if __name__ == "__main__":
    main()
