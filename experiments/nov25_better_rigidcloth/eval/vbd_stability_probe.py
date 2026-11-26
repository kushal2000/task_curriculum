"""How stable is the VBD cloth? Run a policy on the VBD evaluation world and measure the sheet's jitter.

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYGLET_HEADLESS=1 .venv/isaaclab3/bin/python \
        experiments/nov25_better_rigidcloth/eval/vbd_stability_probe.py --name baseline --out outputs/vbd_stability \
        [hydra overrides, e.g. env.cloth.edge_ke=1.0e-4]

The env is built as ``eval.py --protocol vbd.yaml`` builds it (pinned start unless --randomize; the point-cloud
student brings its cloud). Phase 1: --settle steps of zero action (the hand stays away: the sheet alone on
the table). Phase 2: the policy, closed loop, until every env's first episode ends (or --max_steps). Every
particle of every env is read every policy step (60 Hz). Per env, over its first episode:

  rest_speed_mm_s   RMS particle speed over the second half of the settle (a resting sheet should read ~0)
  accel_p95         per step, the 95th percentile over particles of |x(t+1) - 2 x(t) + x(t-1)| / dt^2 [m/s^2];
                    its median and max over the episode (jitter shows as a high, spiky second difference)
  spike_steps       steps where any particle's acceleration exceeds --spike_g x 9.81 m/s^2
  stretch_max       the longest mesh edge / its rest length, over the episode (1.0 = no stretch)
  sink_mm           how far the lowest particle got below the settled sheet's lowest particle
  outcome           fold_held / fall / timeout / nonfinite ...
Robot side, sampled every PHYSICS step (decimation x the policy rate), split by whether a fingertip is
within --touch_mm of a cloth particle's SURFACE ("touch") or not ("free"). The distance is from the
fingertip's collision mesh (the solver's geometry, posed from Newton's body_q) to each particle, minus the
particle radius -- surface to surface, so a larger particle radius counts as touching from further away:
  hand_err_deg      RMS hand-joint tracking error |target - position| [deg]
  hand_acc          RMS hand-joint acceleration, second difference [rad/s^2] (chatter shows as high touch values)
  hand_flip_hz      hand-joint velocity sign reversals per joint per second, |v| > 0.5 rad/s (oscillation)
  tip_acc_p95       95th percentile fingertip acceleration [m/s^2]
  hand_cmd_acc      RMS acceleration of the hand-joint TARGETS (what the policy commands) [rad/s^2]
  thumb_acc / thumb_cmd_acc / thumb_err_deg   the same for the thumb joints only: actual shaking well above
                    commanded shaking while touching = the physics/coupling, not the policy
Writes <out>/<name>.json (per env and pooled) and prints a one-line summary.
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
    p.add_argument("--name", required=True)
    p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
    p.add_argument("--checkpoint", default="pretrained_policy/play_pc_rigidclothRL_64live.pth")
    p.add_argument("--protocol", default="isaacsimenvs/tasks/cloth/config/eval/vbd.yaml")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--settle", type=int, default=60, help="zero-action steps before the policy starts")
    p.add_argument("--max_steps", type=int, default=700)
    p.add_argument("--spike_g", type=float, default=20.0)
    p.add_argument("--trace", action="store_true",
                   help="also save <out>/<name>_trace.npz: per physics step and env, thumb joint angles / velocities /"
                        " targets, the thumb-to-cloth surface gap and the number of particles inside the thumb")
    p.add_argument("--touch_mm", type=float, default=3.0,
                   help="fingertip-surface to particle-surface gap that counts as touching [mm]")
    p.add_argument("--out", default="outputs/vbd_stability")
    AppLauncher.add_app_launcher_args(p)
    args, extra = p.parse_known_args()
    os.chdir(ROOT)
    sys.argv = [sys.argv[0], *extra]
    try:
        AppLauncher(args)
    except ImportError as exc:
        print(f"[stability] running kit-less: {exc}", flush=True)

    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.distillation.student import apply_student_env
    from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
    from isaacsimenvs.scripts.evaluation.eval import _goal_student_success, _student_checkpoint
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    student = _student_checkpoint(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    @hydra_task_config_with_yaml(args.task, "", experiment=args.protocol)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = "cuda:0"
        if not args.randomize:
            disable_randomization(env_cfg)
        apply_student_env(env_cfg, student["env"], with_dr=args.randomize)
        _goal_student_success(env_cfg)
        c = env_cfg.cloth
        settings = {k: getattr(c, k) for k in ("resolution", "thickness", "edge_ke", "edge_kd", "tri_ke", "tri_kd",
                                               "density", "cloth_table_friction", "proxy_relaxation", "rigid_avbd_beta",
                                               "rigid_contact_k_start", "soft_contact_ke", "soft_contact_kd",
                                               "self_contact_radius", "vbd_iterations", "substeps", "proxy_mode")
                    if hasattr(c, k)}
        print(f"[stability] {args.name}: {settings}", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n = inner.num_envs
        dt = float(inner.step_dt)
        player = EnvCloudStudentPlayer(inner, student, "cuda:0")

        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager._model
        owner = model.particle_world.numpy()
        pidx = [torch.as_tensor(np.nonzero(owner == e)[0], device=inner.device) for e in range(n)]
        tri = model.tri_indices.numpy()
        edges = {}
        for e in range(n):
            t = tri[(owner[tri] == e).all(axis=1)] - int(pidx[e][0])
            ed = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [0, 2]]]), axis=1), axis=0)
            edges[e] = torch.as_tensor(ed, device=inner.device)

        def particles() -> torch.Tensor:  # (n, P, 3) world
            q = wp.to_torch(NewtonManager._state_0.particle_q)
            return torch.stack([q[pi] for pi in pidx])

        def edge_len(x: torch.Tensor) -> torch.Tensor:  # (n, E)
            return torch.stack([(x[e, edges[e][:, 0]] - x[e, edges[e][:, 1]]).norm(dim=-1) for e in range(n)])

        from isaacsimenvs.tasks.cloth.mdp.robot import to_torch

        layout = inner.robot_layout
        hand_ids = torch.as_tensor(layout.hand_joint_ids, device=inner.device)
        hand_names = [inner.robot.joint_names[int(i)] for i in layout.hand_joint_ids]
        thumb_cols = torch.as_tensor([j for j, nm in enumerate(hand_names) if "thumb" in nm], device=inner.device)
        tip_ids = torch.as_tensor(layout.fingertip_body_ids, device=inner.device)
        phys_dt = float(inner.physics_dt)
        # Each env's fingertip collision vertices in their body frames (Newton body index per vertex).
        from isaacsimenvs.scripts.visualization.live_viser.scene import NewtonScene

        def quat_rot(wxyz):
            w, x, y, z = (float(v) for v in wxyz)
            return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

        tip_names = tuple(inner.robot_spec.fingertip_link_names)
        scene = NewtonScene(inner)
        tip_verts = []  # per env: (verts (V, 3) body frame, body key (V,))
        for e in range(n):
            vs, ks = [], []
            for sh in scene.shapes(e):
                if sh.key is not None and sh.label.rsplit("/", 1)[-1] in tip_names:
                    vs.append(sh.verts @ quat_rot(sh.wxyz).T + sh.pos)
                    ks.append(np.full(len(sh.verts), sh.key))
            if not vs:
                raise SystemExit(f"[stability] no fingertip collision shapes for env {e} ({tip_names})")
            V, K = np.concatenate(vs), np.concatenate(ks)
            stride = max(1, len(V) // 20000)  # the collision meshes are dense; ~20k vertices bound the gap well
            tip_verts.append((torch.as_tensor(V[::stride], dtype=torch.float32, device=inner.device),
                              torch.as_tensor(K[::stride], dtype=torch.long, device=inner.device)))
        p_radius = float(env_cfg.cloth.particle_radius())
        thumb_verts = []
        for e in range(n):
            vs, ks = [], []
            for sh in scene.shapes(e):
                if sh.key is not None and sh.label.rsplit("/", 1)[-1] in ("r_thumb_distal", "r_thumb_middle"):
                    vs.append(sh.verts @ quat_rot(sh.wxyz).T + sh.pos)
                    ks.append(np.full(len(sh.verts), sh.key))
            V, K = np.concatenate(vs), np.concatenate(ks)
            stride = max(1, len(V) // 20000)
            thumb_verts.append((torch.as_tensor(V[::stride], dtype=torch.float32, device=inner.device),
                                torch.as_tensor(K[::stride], dtype=torch.long, device=inner.device)))
        print(f"[stability] touch = fingertip collision mesh ({sum(len(v) for v, _ in tip_verts) // n} verts/env) within "
              f"{args.touch_mm} mm of a particle surface (radius {p_radius * 1e3:.1f} mm)", flush=True)

        def posed(v, k, bq):
            q = bq[k]
            qv, qw = q[:, 3:6], q[:, 6:7]
            t = 2.0 * torch.cross(qv, v, dim=1)
            return v + qw * t + torch.cross(qv, t, dim=1) + q[:, :3]

        def thumb_state():  # (n,) thumb-to-particle surface gap [m], (n,) particles closer than their radius
            bq = wp.to_torch(NewtonManager._state_0.body_q)
            x = particles()
            gap, inside = torch.empty(n, device=inner.device), torch.empty(n, device=inner.device)
            for e, (v, k) in enumerate(thumb_verts):
                d = torch.cdist(x[e], posed(v, k, bq)).min(dim=1).values      # per particle
                gap[e] = d.min() - p_radius
                inside[e] = (d < 0.5 * p_radius).sum()
            return gap, inside

        def tip_gap() -> torch.Tensor:  # (n,) closest fingertip-surface to particle-surface gap [m]
            bq = wp.to_torch(NewtonManager._state_0.body_q)
            x = particles()
            out = torch.empty(n, device=inner.device)
            for e, (v, k) in enumerate(tip_verts):
                q = bq[k]                                    # (V, 7): position, quaternion xyzw
                qv, qw = q[:, 3:6], q[:, 6:7]
                t = 2.0 * torch.cross(qv, v, dim=1)
                world = v + qw * t + torch.cross(qv, t, dim=1) + q[:, :3]
                out[e] = torch.cdist(world, x[e]).min() - p_radius
            return out
        rob = {"on": False, "q": [], "tgt": [], "tip": [], "touch": [], "alive": [], "qd": [], "tgap": [], "tin": []}
        alive_now = torch.ones(n, dtype=torch.bool, device=inner.device)
        _apply = inner._apply_action

        def apply_and_record() -> None:  # every physics step: the state the physics step starts from
            _apply()
            if not rob["on"]:
                return
            q = to_torch(inner.robot.data.joint_pos)[:, hand_ids].clone()
            tip = to_torch(inner.robot.data.body_pos_w)[:, tip_ids].clone()          # (n, 5, 3)
            gap = tip_gap()                                                           # (n,)
            rob["q"].append(q)
            rob["tgt"].append(inner.controller.targets[:, hand_ids].clone())
            rob["tip"].append(tip)
            rob["touch"].append((gap < args.touch_mm * 1e-3).unsqueeze(1))
            if args.trace:
                rob["qd"].append(to_torch(inner.robot.data.joint_vel)[:, hand_ids].clone())
                tg, ti = thumb_state()
                rob["tgap"].append(tg)
                rob["tin"].append(ti)
            rob["alive"].append(alive_now.clone())

        inner._apply_action = apply_and_record

        obs, _ = env.reset()
        zero = torch.zeros((n, inner.cfg.action_space), device=inner.device)
        hist = [particles()]
        settle_speed = []
        for k in range(args.settle):
            obs, *_ = env.step(zero)
            hist.append(particles())
            if k >= args.settle // 2:
                settle_speed.append(((hist[-1] - hist[-2]).norm(dim=-1) / dt).pow(2).mean(dim=1).sqrt())
        rest_len = edge_len(hist[-1])
        z_floor = hist[-1][..., 2].min(dim=1).values
        player.reset_rnn(torch.arange(n, device=inner.device))
        hist = hist[-2:]
        rob["on"] = True

        done = torch.zeros(n, dtype=torch.bool, device=inner.device)
        acc_p95, spikes, stretch, sink = [[] for _ in range(n)], [0] * n, [1.0] * n, [0.0] * n
        outcome = [None] * n
        nonfinite_steps = 0
        for step in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, terminated, truncated, extras = env.step(action)
            x = particles()
            ended = terminated.bool() | truncated.bool()
            if not bool(torch.isfinite(x).all()):
                nonfinite_steps += 1
            acc = (x - 2 * hist[-1] + hist[-2]).norm(dim=-1) / (dt * dt)  # (n, P)
            p95 = torch.quantile(torch.nan_to_num(acc, nan=1e9), 0.95, dim=1)
            amax = torch.nan_to_num(acc, nan=1e9).max(dim=1).values
            ratio = (edge_len(x) / rest_len).max(dim=1).values
            low = z_floor - x[..., 2].min(dim=1).values
            live = (~done & ~ended).nonzero(as_tuple=True)[0].tolist()
            for e in live:  # the step an episode ends shows the reset, not physics
                acc_p95[e].append(float(p95[e]))
                spikes[e] += int(float(amax[e]) > args.spike_g * 9.81)
                stretch[e] = max(stretch[e], float(ratio[e]))
                sink[e] = max(sink[e], float(low[e]) * 1e3)
            fin = ended & ~done
            alive_now &= ~ended
            if bool(fin.any()):
                final = extras["episode_final"]
                for e in fin.nonzero(as_tuple=True)[0].tolist():
                    outcome[e] = next((kk[5:] for kk in ("done_fold_held", "done_fall", "done_hand_far", "done_nonfinite")
                                       if kk in final and bool(final[kk][e])), "timeout" if bool(truncated[e]) else "?")
                done |= fin
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            hist = [hist[-1], x]
            if bool(done.all()):
                break

        rest = torch.stack(settle_speed).mean(dim=0) if settle_speed else torch.zeros(n)

        # Robot side: samples where the env is still on its first episode; contact = any fingertip touching.
        Q, TG, TP = torch.stack(rob["q"]), torch.stack(rob["tgt"]), torch.stack(rob["tip"])   # (T, n, ...)
        TOUCH, ALIVE = torch.stack(rob["touch"]).any(dim=2), torch.stack(rob["alive"])          # (T, n)
        err = (TG - Q).abs().pow(2).mean(dim=2).sqrt()                                          # (T, n) rad
        qacc = torch.zeros_like(err)
        qacc[1:-1] = ((Q[2:] - 2 * Q[1:-1] + Q[:-2]) / phys_dt**2).pow(2).mean(dim=2).sqrt()
        v = torch.diff(Q, dim=0) / phys_dt                                                      # (T-1, n, J)
        flip = torch.zeros_like(err)
        flip[1:-1] = ((v[1:] * v[:-1] < 0) & (v[1:].abs() > 0.5) & (v[:-1].abs() > 0.5)).float().mean(dim=2)
        tacc = torch.zeros_like(err)
        tacc[1:-1] = ((TP[2:] - 2 * TP[1:-1] + TP[:-2]) / phys_dt**2).norm(dim=-1).max(dim=2).values
        # Commanded vs actual, hand joints and thumb joints only: if the targets are smooth while the joints
        # shake, the physics (contact / coupling) shakes them; if the targets shake too, the policy does.
        def acc_rms(X):  # (T, n, J) -> (T, n)
            a = torch.zeros(X.shape[:2], device=X.device)
            a[1:-1] = ((X[2:] - 2 * X[1:-1] + X[:-2]) / phys_dt**2).pow(2).mean(dim=2).sqrt()
            return a

        tacc_cmd = acc_rms(TG)
        th_acc, th_cmd = acc_rms(Q[:, :, thumb_cols]), acc_rms(TG[:, :, thumb_cols])
        th_err = (TG[:, :, thumb_cols] - Q[:, :, thumb_cols]).pow(2).mean(dim=2).sqrt()
        valid = ALIVE.clone()
        valid[0], valid[-1] = False, False

        def split(metric, e, how):
            out = {}
            for tag, m in (("touch", valid[:, e] & TOUCH[:, e]), ("free", valid[:, e] & ~TOUCH[:, e])):
                vals = metric[m, e]
                if vals.numel() == 0:
                    out[tag] = None
                elif how == "rms":
                    out[tag] = float(vals.pow(2).mean().sqrt())
                elif how == "mean":
                    out[tag] = float(vals.mean())
                else:
                    out[tag] = float(torch.quantile(vals, 0.95))
            return out
        per_env = []
        for e in range(n):
            a = np.array(acc_p95[e]) if acc_p95[e] else np.array([np.nan])
            per_env.append({"rest_speed_mm_s": round(float(rest[e]) * 1e3, 3),
                            "accel_p95_median": round(float(np.median(a)), 2), "accel_p95_max": round(float(np.max(a)), 1),
                            "spike_steps": spikes[e], "steps": len(acc_p95[e]),
                            "stretch_max": round(stretch[e], 3), "sink_mm": round(sink[e], 2), "outcome": outcome[e],
                            "touch_frac": round(float((valid[:, e] & TOUCH[:, e]).float().sum() / valid[:, e].float().sum().clamp(min=1)), 3),
                            "hand_err_deg": {k: None if x is None else round(x * 57.2958, 2) for k, x in split(err, e, "rms").items()},
                            "hand_acc": {k: None if x is None else round(x, 1) for k, x in split(qacc, e, "rms").items()},
                            "hand_flip_hz": {k: None if x is None else round(x / phys_dt, 2) for k, x in split(flip, e, "mean").items()},
                            "tip_acc_p95": {k: None if x is None else round(x, 2) for k, x in split(tacc, e, "p95").items()},
                            "hand_cmd_acc": {k: None if x is None else round(x, 1) for k, x in split(tacc_cmd, e, "rms").items()},
                            "thumb_acc": {k: None if x is None else round(x, 1) for k, x in split(th_acc, e, "rms").items()},
                            "thumb_cmd_acc": {k: None if x is None else round(x, 1) for k, x in split(th_cmd, e, "rms").items()},
                            "thumb_err_deg": {k: None if x is None else round(x * 57.2958, 2) for k, x in split(th_err, e, "rms").items()}})

        def med(k):
            return float(np.median([r[k] for r in per_env]))

        def med2(k, tag):
            vals = [r[k][tag] for r in per_env if r[k][tag] is not None]
            return float(np.median(vals)) if vals else float("nan")

        pooled = {
            "rest_speed_mm_s_median": med("rest_speed_mm_s"),
            "accel_p95_median": med("accel_p95_median"), "accel_p95_max_median": med("accel_p95_max"),
            "spike_frac": sum(r["spike_steps"] for r in per_env) / max(1, sum(r["steps"] for r in per_env)),
            "stretch_max_median": med("stretch_max"), "stretch_max_max": max(r["stretch_max"] for r in per_env),
            "sink_mm_median": med("sink_mm"), "sink_mm_max": max(r["sink_mm"] for r in per_env),
            "touch_frac_median": med("touch_frac"),
            **{f"{k}_{tag}": med2(k, tag) for k in ("hand_err_deg", "hand_acc", "hand_cmd_acc", "hand_flip_hz", "tip_acc_p95", "thumb_acc",
                                                        "thumb_cmd_acc", "thumb_err_deg") for tag in ("touch", "free")},
            "folded": sum(r["outcome"] == "fold_held" for r in per_env), "fell": sum(r["outcome"] == "fall" for r in per_env),
            "nonfinite": sum(r["outcome"] == "nonfinite" for r in per_env) + (1 if nonfinite_steps else 0) * 0,
            "envs": n, "policy_dt": dt, "physics_dt": phys_dt}
        if args.trace:
            tc = thumb_cols.cpu().numpy()
            np.savez_compressed(out / f"{args.name}_trace.npz", physics_dt=phys_dt,
                                joint_names=np.array([hand_names[j] for j in tc]),
                                q=Q[:, :, thumb_cols].cpu().numpy(), target=TG[:, :, thumb_cols].cpu().numpy(),
                                qd=torch.stack(rob["qd"])[:, :, thumb_cols].cpu().numpy(),
                                thumb_gap=torch.stack(rob["tgap"]).cpu().numpy(), particles_inside=torch.stack(rob["tin"]).cpu().numpy(),
                                alive=ALIVE.cpu().numpy(), particle_radius=p_radius)
            print(f"[stability] wrote {out / (args.name + '_trace.npz')}", flush=True)
        rec = {"name": args.name, "overrides": extra, "settings": settings, "pooled": pooled, "per_env": per_env}
        (out / f"{args.name}.json").write_text(json.dumps(rec, indent=1))
        print(f"[stability] {args.name}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                                      for k, v in pooled.items()), flush=True)
        env.close()

    run()


if __name__ == "__main__":
    main()
