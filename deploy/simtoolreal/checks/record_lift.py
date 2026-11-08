"""Record the Play policy lifting the flat eraser 10 cm in its training env, for robot.mode=openloop.

Runs on the TRAINING branch's code (kushal/2026-10-30_pretraining_real), like record_sim_rollout.py.
On the cluster: scripts/cluster/bos14_simtoolreal_lift.sh. By hand:

    cd <branch worktree>
    PYTHONPATH=$PWD:$PWD/rl_games .venv_isaacsim/bin/python \
        /home/yiboc/mit/task_curriculum_deployment/deploy/simtoolreal/checks/record_lift.py --headless \
        --checkpoint <nov6_dr.txt> --out <dir>

The scene (randomization off, as record_sim_rollout.py):
  - the object: the env's procedural eraser (a box), its size range pinned to the DexToolBench
    flat_eraser mesh's bounding box (12.5 x 2.8 x 5 cm; the mesh is a box: hull = bbox to 2 %), density
    --density (DexToolBench's 400 kg/m^3 -> 70 g);
  - its start: lying on its 12.5 x 5 cm face (90 deg about x, as DexToolBench's start poses), long side
    along env x, centre at (--x, --y) on the table top, at rest;
  - the goal: the start pose raised by --lift, the same orientation, fixed (after each hit the env
    sets the same goal again), so the policy lifts and then holds;
  - the robot starts at home (spec default_q), no joint noise; no episode ends but a fall.

Step 1 (policy): --num_envs identical envs run the policy for --steps. An env "holds" if it never
ended and over the last --hold_s the box centre stayed within 2 cm of the goal height and the
env's keypoint distance stayed under --hold_kp_m. It is "clean" if, besides, the box never came
back down to the table (DROPPED_M) once lifted 5 cm: a drop and re-grasp works only with feedback
(the box lands elsewhere each time). policy_envs.npz keeps every env's box trajectory.

Step 2 (open loop in sim, choosing): up to --candidates clean envs (the earliest lifts) have their
joint targets sent unchanged (the env's own _replay_target_lab_order hook, no policy), each to
several envs from the same start. The one that lifts and holds most often is saved. Lifts and
holds here: never ended, and the box stayed more than lift - 3 cm above its start over the last
--hold_s.

Step 3 (open loop in sim, other starts): the saved targets sent to envs whose start differs: the
box shifted in x / y, turned about z, or the hand's kp and kd scaled (the real hand's torque
constant is unknown, so its gains are a guess). It says how well the box has to be placed.

Out (<out>/):
  openloop.npz        what robot.mode=openloop sends (targets) and shows (the sim object and goal);
                      every pose in the env frame, wxyz; object poses both of the box centre and of
                      the DexToolBench mesh's own frame (the one the viewer draws)
  openloop_meta.json  the settings, the chosen env, step-1 counts
  robustness.json     step 3, one row per start condition (step 2 is in openloop_meta.json)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

TASK = "Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0"
DEPLOY_REPO = Path(__file__).resolve().parents[3]   # deploy/simtoolreal/checks -> repo root
MESH = DEPLOY_REPO / "deploy" / "simtoolreal" / "objects" / "flat_eraser" / "flat_eraser.obj"
TABLE_TOP_Z = 0.28           # the env's (spec table_top_z_env); checked against the env below
DROP_M = 0.001               # start this far above the table, so it settles without a jump
DROPPED_M = 0.02             # once lifted (5 cm), back below this above its start = dropped


def mesh_box(path: Path):
    """(extents (3,), centre of the bounding box in the mesh frame (3,)) of an .obj, in metres."""
    import numpy as np
    with open(path) as f:
        v = np.array([line.split()[1:4] for line in f if line.startswith("v ")], dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return hi - lo, 0.5 * (lo + hi)


def conditions() -> list[dict]:
    """Step 2's start conditions: dx, dy (m), yaw (deg, about z), hand gain scale."""
    c = [dict(dx=0.0, dy=0.0, yaw=0.0, kp=1.0)]
    for d in (-0.02, -0.01, 0.01, 0.02):
        c += [dict(dx=d, dy=0.0, yaw=0.0, kp=1.0), dict(dx=0.0, dy=d, yaw=0.0, kp=1.0)]
    for d in (-0.02, -0.01, 0.01, 0.02):
        c += [dict(dx=d, dy=d, yaw=0.0, kp=1.0), dict(dx=d, dy=-d, yaw=0.0, kp=1.0)]
    for y in (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0):
        c += [dict(dx=0.0, dy=0.0, yaw=y, kp=1.0)]
    for k in (0.5, 0.7, 1.4, 2.0):
        c += [dict(dx=0.0, dy=0.0, yaw=0.0, kp=k)]
    return c


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default=str(DEPLOY_REPO / "pretrained_policy" / "config.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=600, help="policy steps (60 Hz) in the rollout")
    p.add_argument("--lift", type=float, default=0.10, help="m, goal above the start")
    p.add_argument("--x", type=float, default=0.0, help="env x of the box centre (table centre: 0)")
    p.add_argument("--y", type=float, default=0.29, help="env y of the box centre (table centre: 0.29)")
    p.add_argument("--yaw", type=float, default=0.0, help="deg about z; 0 = long side along env x")
    p.add_argument("--density", type=float, default=400.0, help="kg/m^3 (DexToolBench: 400)")
    p.add_argument("--hold_s", type=float, default=1.0, help="the end window that must hold")
    p.add_argument("--hold_kp_m", type=float, default=0.03, help="keypoint distance limit while holding")
    p.add_argument("--candidates", type=int, default=8, help="clean rollouts screened open loop in step 2")
    p.add_argument("--expl_coef", type=float, default=0.0, help="SAPG block (0 = last)")
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
    from isaacsimenvs.tasks.play.utils import object_size_distributions as osd
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_envs
    hold_steps = int(round(args.hold_s * 60))

    # the eraser: pin the env's eraser size distribution to the mesh's box (in place: the generator
    # imported the list object itself)
    size, centre_in_mesh = mesh_box(MESH)
    i_eraser = [i for i, d in enumerate(osd.OBJECT_SIZE_DISTRIBUTIONS) if d.type == "eraser"]
    assert len(i_eraser) == 1, i_eraser
    osd.OBJECT_SIZE_DISTRIBUTIONS[i_eraser[0]] = dataclasses.replace(
        osd.OBJECT_SIZE_DISTRIBUTIONS[i_eraser[0]],
        handle_min_lengths=tuple(size), handle_max_lengths=tuple(size),
        handle_min_density=args.density, handle_max_density=args.density)
    mass = float(np.prod(size) * args.density)

    # start and goal (env frame, wxyz): flat = +90 deg about x (the mesh's 2.8 cm y side vertical)
    def yaw_quat(deg: float) -> np.ndarray:
        h = math.radians(deg) / 2
        return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])

    def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    q_flat = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
    start_quat = qmul(yaw_quat(args.yaw), q_flat)
    start_pos = np.array([args.x, args.y, TABLE_TOP_Z + 0.5 * size[1] + DROP_M])
    goal_pos = start_pos + np.array([0.0, 0.0, args.lift])

    @hydra_task_config_with_yaml(TASK, "")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.assets.handle_head_types = ("eraser",)
        env_cfg.assets.num_assets_per_type = n          # n identical boxes, one per env
        env_cfg.scene.num_envs = n
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = float(env_cfg.termination.target_success_tolerance)
        disable_randomization(env_cfg)
        env_cfg.reset.fixed_start_pose = tuple(float(v) for v in (*start_pos, *start_quat))
        env_cfg.reset.fixed_goal_pose = tuple(float(v) for v in (*goal_pos, *start_quat))
        env_cfg.termination.max_consecutive_successes = 0      # no end on goal count
        env_cfg.termination.episode_length = 1_000_000          # nor on time
        env_cfg.episode_length_s = 1.0e5

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        dev = inner.device
        perm_l2c, perm_c2l = inner._perm_lab_to_canon, inner._perm_canon_to_lab
        origins = inner.scene.env_origins
        joint_names = [inner.robot.joint_names[i] for i in perm_l2c.tolist()]
        table_top = (inner._table_z_per_env + inner._table_half_height).cpu().numpy()
        assert np.allclose(table_top, TABLE_TOP_Z, atol=1e-4), table_top
        print(f"[lift] {TASK}: {n} envs; box {np.round(100 * size, 2).tolist()} cm, {1000 * mass:.0f} g; "
              f"start {np.round(start_pos, 4).tolist()} goal {np.round(goal_pos, 4).tolist()}", flush=True)

        def tt(x) -> torch.Tensor:
            """Isaac Lab data may be a ProxyArray (warp); its torch view."""
            return x.torch if hasattr(x, "torch") else x

        def write_gains(kp: torch.Tensor, kd: torch.Tensor) -> None:
            """All envs' joint kp, kd (lab order). Newer Isaac Lab: *_index; this env's: the plain ones."""
            r = inner.robot
            if hasattr(r, "write_joint_stiffness_to_sim_index"):
                r.write_joint_stiffness_to_sim_index(stiffness=kp)
                r.write_joint_damping_to_sim_index(damping=kd)
            else:
                r.write_joint_stiffness_to_sim(kp)
                r.write_joint_damping_to_sim(kd)

        def to_np(t):
            return tt(t).detach().double().cpu().numpy().copy()

        def snapshot() -> dict:
            d = inner.robot.data
            return {"q": to_np(d.joint_pos[:, perm_l2c]), "dq": to_np(d.joint_vel[:, perm_l2c])}

        def object_now() -> dict:
            return {"object_pos": to_np(inner.object.data.root_pos_w - origins),
                    "object_quat": to_np(inner.object.data.root_quat_w),
                    "kp_dist": to_np(inner._keypoints_max_dist)}

        def held(rec: dict) -> np.ndarray:
            """(n,) bool: never ended, and the last hold_steps near the goal."""
            z_ok = np.abs(rec["object_pos"][-hold_steps:, :, 2] - goal_pos[2]) < 0.02
            kp_ok = rec["kp_dist"][-hold_steps:] < args.hold_kp_m
            return ~rec["done"].any(axis=0) & z_ok.all(axis=0) & kp_ok.all(axis=0)

        # ------------------------------------------------------------------ step 1: the policy
        pl = PretrainedPlayer(args.policy_config, args.checkpoint, num_envs=n, device=dev,
                              num_observations=int(inner.cfg.observation_space),
                              num_actions=int(inner.cfg.action_space), expl_coef=args.expl_coef)
        obs = env.reset()[0]["policy"]
        pl.reset_rnn()
        frames = []
        t0 = time.time()
        for t in range(args.steps):
            s = snapshot()
            with torch.no_grad():
                action = pl.get_action(obs, deterministic=True)
            obs_d, _r, term, trunc, _i = env.step(action)
            done = term | trunc
            frames.append({**s, **object_now(), "targets": to_np(inner._cur_targets[:, perm_l2c]),
                           "success": inner._is_success.cpu().numpy().copy(), "done": done.cpu().numpy().copy()})
            if done.any():
                pl.reset_rnn(done.nonzero(as_tuple=False).squeeze(-1))
            obs = obs_d["policy"]
            if (t + 1) % 120 == 0:
                print(f"[lift] policy step {t + 1}/{args.steps} ({time.time() - t0:.0f} s)", flush=True)
        rec = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
        ok = held(rec)
        hits = rec["success"].any(axis=0)
        first_hit = np.where(hits, rec["success"].argmax(axis=0), -1)
        print(f"[lift] policy: {int(ok.sum())}/{n} envs lift and hold; {int(hits.sum())}/{n} hit the goal "
              f"at least once; {int(rec['done'].any(axis=0).sum())} ended (fall)", flush=True)
        meta = {
            "task": TASK, "checkpoint": str(args.checkpoint), "expl_coef": args.expl_coef, "seed": args.seed,
            "num_envs": n, "steps": args.steps, "dt": float(inner.step_dt), "joint_names": joint_names,
            "box_size_m": size.tolist(), "box_centre_in_mesh_m": centre_in_mesh.tolist(), "mesh": str(MESH),
            "density": args.density, "mass_kg": mass,
            "start_pos_env": start_pos.tolist(), "start_quat_wxyz": start_quat.tolist(),
            "goal_pos_env": goal_pos.tolist(), "lift_m": args.lift,
            "envs_holding": int(ok.sum()), "envs_hit_goal": int(hits.sum()),
            "first_hit_step_per_env": first_hit.tolist(),
        }
        # Once lifted (5 cm) the box must stay up to the end. A drop and re-grasp needs feedback (the
        # box lands somewhere else each time), so it cannot work open loop.
        z = rec["object_pos"][:, :, 2] - start_pos[2]
        up = z > 0.05
        first_up = np.where(up.any(axis=0), up.argmax(axis=0), args.steps)
        after_up = np.arange(args.steps)[:, None] >= first_up[None]
        no_drop = ~(after_up & (z < DROPPED_M)).any(axis=0)
        clean = ok & no_drop & (first_up < args.steps)
        print(f"[lift] {int(clean.sum())}/{n} lift and hold without dropping it on the way", flush=True)
        meta.update(envs_clean=int(clean.sum()), first_lift_step_per_env=first_up.tolist())
        # every env's box over time, to see how the policy lifts (and drops)
        np.savez_compressed(out / "policy_envs.npz", object_pos=rec["object_pos"].astype(np.float32),
                            kp_dist=rec["kp_dist"].astype(np.float32), success=rec["success"], done=rec["done"],
                            holds=ok, clean=clean, first_lift_step=first_up, start_pos=start_pos)
        cands = [int(i) for i in np.argsort(first_up, kind="stable") if clean[i]][:args.candidates]
        if not cands:
            (out / "openloop_meta.json").write_text(json.dumps({**meta, "chosen_env": None}, indent=1) + "\n")
            print("[lift] no env lifted and held without a drop: nothing saved", flush=True)
            return

        # ------------------------------------------------------------------ open-loop replay in sim
        hand_lab = perm_l2c[7:]
        kp0, kd0 = tt(inner.robot.data.joint_stiffness).clone(), tt(inner.robot.data.joint_damping).clone()

        def start_pose(c: dict) -> np.ndarray:
            return np.r_[start_pos + np.array([c["dx"], c["dy"], 0.0]), qmul(yaw_quat(args.yaw + c["yaw"]), q_flat)]

        def replay(targets_env: np.ndarray, env_cond: list[dict]) -> dict:
            """Each env gets its own target sequence (n, T, 27), start pose and hand gain scale; no policy.
            ok: never ended, and over the last hold_s the box stayed more than lift - 3 cm above its
            start (height only: a shifted box ends shifted from the goal even when the grasp works)."""
            env.reset()
            ids = torch.arange(n, device=dev)
            pose_t = torch.as_tensor(np.stack([start_pose(c) for c in env_cond]), dtype=torch.float32, device=dev)
            pose_t[:, :3] += tt(origins)
            inner.object.write_root_pose_to_sim(pose_t, env_ids=ids)
            inner.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
            # kp and kd together: the hand's unknown torque constant scales both
            scale = torch.ones_like(kp0)
            scale[:, hand_lab] = torch.tensor([c["kp"] for c in env_cond], device=dev, dtype=kp0.dtype)[:, None]
            write_gains(kp0 * scale, kd0 * scale)
            tg = torch.as_tensor(targets_env, dtype=torch.float32, device=dev)[:, :, perm_c2l]
            frames = []
            for t in range(tg.shape[1]):
                inner._replay_target_lab_order = tg[:, t].contiguous()
                _o, _r, term, trunc, _i = env.step(torch.zeros(n, int(inner.cfg.action_space), device=dev))
                frames.append({**object_now(), "done": (term | trunc).cpu().numpy().copy()})
            inner._replay_target_lab_order = None
            write_gains(kp0, kd0)
            rr = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
            low = rr["object_pos"][-hold_steps:, :, 2].min(axis=0) - start_pos[2]
            rr["ok"] = ~rr["done"].any(axis=0) & (low > args.lift - 0.03)
            return rr

        # step 2: each candidate's targets replayed from the same start in n / len(cands) envs; keep
        # the one that works open loop most often (ties: the earlier lift)
        base = dict(dx=0.0, dy=0.0, yaw=0.0, kp=1.0)
        cand_of_env = np.array([cands[i % len(cands)] for i in range(n)])
        rr = replay(rec["targets"][:, cand_of_env].transpose(1, 0, 2), [base] * n)
        screen = []
        print("[lift] open loop in sim, same start, each candidate in several envs:", flush=True)
        for c in cands:
            m = cand_of_env == c
            screen.append({"env": c, "first_lift_s": float(first_up[c] * inner.step_dt),
                           "open_loop_holds": int(rr["ok"][m].sum()), "of": int(m.sum())})
            print(f"   env {c:2d}: lifted at {screen[-1]['first_lift_s']:.1f} s, open loop holds "
                  f"{screen[-1]['open_loop_holds']}/{screen[-1]['of']}", flush=True)
        best = max(screen, key=lambda s: (s["open_loop_holds"] / s["of"], -s["first_lift_s"]))
        e = best["env"]
        mean_kp = float(rec["kp_dist"][-hold_steps:, e].mean())
        meta.update(chosen_env=e, chosen_first_hit_step=int(first_hit[e]), chosen_first_lift_step=int(first_up[e]),
                    chosen_mean_kp_dist_hold_m=mean_kp, open_loop_screen=screen)

        def mesh_frame(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
            """Box-centre poses -> where the mesh's own frame is: centre - R(quat) c."""
            w, x, y, z = (quat[..., i] for i in range(4))
            c = centre_in_mesh
            # R(q) c for unit wxyz quaternions
            rc = np.stack([
                (1 - 2 * (y * y + z * z)) * c[0] + 2 * (x * y - w * z) * c[1] + 2 * (x * z + w * y) * c[2],
                2 * (x * y + w * z) * c[0] + (1 - 2 * (x * x + z * z)) * c[1] + 2 * (y * z - w * x) * c[2],
                2 * (x * z - w * y) * c[0] + 2 * (y * z + w * x) * c[1] + (1 - 2 * (x * x + y * y)) * c[2],
            ], axis=-1)
            return pos - rc

        np.savez_compressed(
            out / "openloop.npz",
            dt=np.float64(inner.step_dt), joint_names=np.array(joint_names),
            targets=rec["targets"][:, e], q=rec["q"][:, e], dq=rec["dq"][:, e],
            object_pos=rec["object_pos"][:, e], object_quat=rec["object_quat"][:, e],
            object_mesh_pos=mesh_frame(rec["object_pos"][:, e], rec["object_quat"][:, e]),
            kp_dist=rec["kp_dist"][:, e], success=rec["success"][:, e],
            goal_pos=goal_pos, goal_quat=start_quat, goal_mesh_pos=mesh_frame(goal_pos, start_quat),
            start_pos=start_pos, start_quat=start_quat, start_mesh_pos=mesh_frame(start_pos, start_quat),
            box_size=size, box_centre_in_mesh=centre_in_mesh,
            meta=np.array(json.dumps(meta)),
        )
        (out / "openloop_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"[lift] saved env {e} (lifted at {best['first_lift_s']:.1f} s, open loop holds "
              f"{best['open_loop_holds']}/{best['of']}) -> {out / 'openloop.npz'}", flush=True)

        # step 3: the saved targets from other starts
        conds = conditions()
        assert len(conds) <= n, (len(conds), n)
        env_cond = [conds[i] if i < len(conds) else conds[0] for i in range(n)]
        rr = replay(np.broadcast_to(rec["targets"][:, e], (n, *rec["targets"][:, e].shape)), env_cond)
        rows = []
        for i, c in enumerate(conds):
            rows.append({**c, "holds": bool(rr["ok"][i]), "fell": bool(rr["done"][:, i].any()),
                         "max_lift_m": float(rr["object_pos"][:, i, 2].max() - start_pos[2]),
                         "end_lift_m": float(rr["object_pos"][-1, i, 2] - start_pos[2]),
                         "final_kp_dist_m": float(rr["kp_dist"][-1, i])})
        base_repeats = rr["ok"][len(conds):] if n > len(conds) else rr["ok"][:1]
        result = {"chosen_env": e, "conditions": rows,
                  "base_condition_repeats_holding": f"{int(base_repeats.sum())}/{len(base_repeats)}"}
        (out / "robustness.json").write_text(json.dumps(result, indent=1) + "\n")
        print("[lift] open loop in sim (the saved targets, other starts):", flush=True)
        print("   dx cm  dy cm  yaw deg  hand gain  holds  fell  max lift cm  end lift cm", flush=True)
        for r in rows:
            print(f"   {100 * r['dx']:5.0f}  {100 * r['dy']:5.0f}  {r['yaw']:7.0f}  {r['kp']:9.1f}  "
                  f"{'yes' if r['holds'] else 'NO':>5}  {'yes' if r['fell'] else '-':>4}  "
                  f"{100 * r['max_lift_m']:11.1f}  {100 * r['end_lift_m']:11.1f}", flush=True)
        print(f"[lift] base condition repeated in the spare envs: {result['base_condition_repeats_holding']} hold",
              flush=True)

    run()
    os._exit(0)  # Kit shutdown hangs on these nodes


if __name__ == "__main__":
    main()
