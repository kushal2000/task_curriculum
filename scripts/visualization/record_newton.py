"""Record one Newton (VBD cloth) policy episode as a replayable trajectory, for `render_physx.py --replay`.

    .venv_isaaclab3/bin/python scripts/visualization/record_newton.py --checkpoint <pth> --out traj.npz [hydra overrides]

The physics is Newton's; only the picture is made elsewhere. Saved per policy step (60 Hz): robot
joint positions (with names, so the replay maps them onto its own articulation by name), the cloth's
vertices (env-local) and, once, its triangles, the robot root pose and the table box. Every world is
recorded until its first episode ends; the one saved is the best by a demo score (a held fold, early,
little sheet drift, the hand clear of the sheet at the end) unless ``--world`` names one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
from isaacsimenvs.utils.paths import PRETRAINED_CONFIG


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default=str(PRETRAINED_CONFIG))
    p.add_argument("--sapg_expl_coef", type=float, default=0.0)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--world", type=int, default=-1, help="world to save; -1 = the best-scoring one")
    p.add_argument("--randomize_reset", action="store_true",
                   help="keep the reset distribution (sheet pose) so the worlds differ; DR stays off")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        import newton
        from isaaclab_newton.physics.newton_manager import NewtonManager

        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.assets.handle_head_types = ("hammer",)
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        # The eval protocol (every DR source off, start pose pinned); --randomize_reset then puts
        # back ONLY the sheet's start pose (xy noise + a yaw per episode), so the worlds run
        # different episodes of the same deterministic, unperturbed policy. Skipping the protocol
        # instead also leaves point-observation noise, hand friction/effort DR and random pushes
        # on -- training perturbations under which all 16 worlds timed out.
        rs = env_cfg.reset
        noise = (rs.reset_position_noise_x, rs.reset_position_noise_y)
        disable_randomization(env_cfg)
        if args.randomize_reset:
            rs.reset_position_noise_x, rs.reset_position_noise_y = noise
            rs.fixed_start_pose = None
            print(f"[record] start pose randomised: xy noise +-{noise[0]}/{noise[1]} m, yaw per episode",
                  flush=True)
        use_single_object_variant()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        N = inner.num_envs
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=N,
            device=args.device, num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
        )
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((N, inner.cfg.action_space), device=inner.device))

        model = NewtonManager.get_model()
        origins = inner.scene.env_origins.detach().cpu().numpy()
        pworld = model.particle_world.numpy()
        mine = [np.nonzero(pworld == w)[0] for w in range(N)]
        remap = -np.ones(model.particle_count, np.int64)
        remap[mine[0]] = np.arange(len(mine[0]))
        tri = model.tri_indices.numpy()
        tri = remap[tri[(remap[tri] >= 0).all(1)]]  # world 0's, world-local; every world is identical
        rest = model.particle_q.numpy()[mine[0]] - origins[0]

        labels = [str(x) for x in model.body_label]
        sb, st = model.shape_body.numpy(), model.shape_type.numpy()
        table = {}
        for s_, b in enumerate(sb):
            if 0 <= b < len(labels) and "Table" in labels[b] and int(st[s_]) == int(newton.GeoType.BOX):
                for w in range(N):
                    if f"env_{w}/" in labels[b]:
                        table[w] = s_
        table_half = model.shape_scale.numpy()[table[0]]

        # Hand bodies: everything after the 7-link arm (+ base). Used to tell whether the hand
        # has cleared the sheet at the end, i.e. whether the fold is visible.
        body_names = list(inner.robot.body_names)
        hand = [i for i, n in enumerate(body_names) if i >= 8]

        joint_names = list(inner.robot.joint_names)
        q, verts, tables = [], [], []

        def snap():
            q.append(inner.robot.data.joint_pos.detach().cpu().numpy().copy())
            pq = NewtonManager._state_0.particle_q.numpy()
            verts.append(np.stack([pq[mine[w]] - origins[w] for w in range(N)]).astype(np.float32))
            bq = NewtonManager._state_0.body_q.numpy()
            xs = model.shape_transform.numpy()
            tables.append(np.stack([bq[sb[table[w]]][:3] + xs[table[w]][:3] - origins[w] for w in range(N)]))

        snap()
        end = np.full(N, -1)
        outcome = ["not_finished"] * N
        clear = np.zeros(N)
        for step in range(args.steps):
            act = player.get_action(obs["policy"], deterministic=True)
            pre_hand = inner.robot.data.body_pos_w[:, hand].detach().cpu().numpy()
            obs, _r, term, trunc, _ = env.step(act)
            done = (term | trunc).cpu().numpy().astype(bool)
            reasons = getattr(inner, "_termination_reasons", {})
            for w in np.nonzero(done & (end < 0))[0]:
                fired = [k for k, v in reasons.items() if bool(v[w])]
                outcome[w] = next((k for k in ("fold_held", "fold", "fall", "hand_far", "nonfinite", "timeout")
                                   if k in fired), "unknown")
                end[w] = len(q)  # frames recorded for w: everything before this step's reset
                v = verts[-1][w] + 0.0
                clear[w] = np.min(np.linalg.norm(pre_hand[w][:, None, :2] - (v[None, :, :2] + origins[w][:2]), axis=-1))
            if bool(done.any()):
                player.reset_rnn(torch.as_tensor(np.nonzero(done)[0], device=inner.device))
            if (end >= 0).all():
                break
            snap()
        end[end < 0] = len(q)

        # Pick: a held fold, as early as possible, little drift of the sheet, hand clear at the end.
        print("[record] world  outcome     frames  drift[cm]  hand_clear[cm]  score", flush=True)
        best, best_score = None, -1e9
        for w in range(N):
            c0 = verts[0][w][:, :2].mean(0)
            c1 = verts[end[w] - 1][w][:, :2].mean(0)
            drift = float(np.linalg.norm(c1 - c0))
            score = (-end[w] / 60.0 - 20.0 * drift + 30.0 * min(clear[w], 0.06)
                     - (0.0 if outcome[w] == "fold_held" else 100.0))
            print(f"[record] {w:5d}  {outcome[w]:10s}  {end[w]:6d}  {100 * drift:9.1f}  {100 * clear[w]:14.1f}  "
                  f"{score:6.2f}", flush=True)
            if score > best_score:
                best, best_score = w, score
        w, T = (args.world if args.world >= 0 else best), end[args.world if args.world >= 0 else best]
        print(f"[record] chose world {w}: {outcome[w]} after {T} frames", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out, joint_pos=np.asarray(q, np.float32)[:T, w], joint_names=np.asarray(joint_names),
            verts=np.asarray(verts, np.float32)[:T, w], tris=tri.astype(np.int32), rest=rest.astype(np.float32),
            table_center=np.asarray(tables, np.float32)[:T, w], table_half=np.asarray(table_half, np.float32),
            root_pos=inner.robot.data.root_pos_w[w].cpu().numpy() - origins[w],
            root_quat=inner.robot.data.root_quat_w[w].cpu().numpy(), step_dt=float(inner.step_dt),
            outcome=str(outcome[w]), world=w,
        )
        print(f"[record] wrote {args.out}: world {w}, {T} frames", flush=True)
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
