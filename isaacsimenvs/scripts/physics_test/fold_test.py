"""Drive a cloth's fold hinge closed, let go, and check the fold is reached and stays.

    # one case (optionally with video)
    .venv/isaacsim/bin/python isaacsimenvs/scripts/physics_test/fold_test.py --headless \\
        --config <run>/.hydra/config.yaml --design strip_4 --size 0.1 0.2 --video fold.mp4 --out fold.json
    # the run's whole randomization: every DR corner + random draws, at the size-range extremes
    ... --design strip_4 --dr_random 64 --thickness 0.002 --out sweep.json

The scene is the training scene (``--config`` plus ``key=value`` overrides, same loader and
``ClothEnv`` as ``eval_cloth.py``) with the robot's colliders disabled. Materials and sizes are
chosen exactly as in ``press_test.py`` (``common.configure``): one material and size by default, or
with ``--dr_random N`` every corner of the run's randomization box plus N random draws, each at
every corner of the size grid. Materials go through the training randomization code.

Script, identical for all envs:
  1. settle flat on the table;
  2. FOLD: every joint is driven to the fold pose. The stationary half (panels with x < 0) is
     pinned -- the articulation root, which is the stationary panel at the centre crease, is held at
     its settled pose -- and every hinge's drive becomes a stiff motor: the fold hinge's target sweeps
     0 -> 180 deg over ``--fold_s`` (smooth) and holds for ``--hold_s``; every other hinge is held
     at 0 deg, so both halves stay flat and rigid while the moving half is carried over;
  3. RELEASE: pin and motors off; every hinge gets its training bending gains and flat target back,
     and nothing touches the cloth for ``--release_s`` (``--pin_release`` keeps the base pinned).

The task's own ``is_folded`` (fold error < tolerance and footprint < max) is the criterion:
``reached`` = folded at the end of the hold, ``stays`` = still folded at the end of the release.
An env FAILS unless it both reaches and stays.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


from isaacsimenvs.scripts.evaluation.eval_cloth import launch
from isaacsimenvs.scripts.physics_test import dr_samples
from isaacsimenvs.scripts.physics_test.common import (add_material_args, add_video_args, configure,
                                                      disable_robot, set_camera)


def parse():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_material_args(p)
    p.add_argument("--motor_stiffness", type=float, default=1.0, help="fold/pin motor gain [N m/rad]")
    p.add_argument("--motor_damping", type=float, default=0.05, help="fold/pin motor damping [N m s/rad]")
    p.add_argument("--settle_s", type=float, default=0.5)
    p.add_argument("--fold_s", type=float, default=1.0)
    p.add_argument("--hold_s", type=float, default=0.5)
    p.add_argument("--release_s", type=float, default=3.0)
    p.add_argument("--pin_release", action="store_true", help="keep the stationary half pinned during the release")
    p.add_argument("--out", type=Path, help="JSON results path")
    add_video_args(p, slowmo=2.0)
    p.add_argument("overrides", nargs="*", help="key=value edits to --config")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def make_env(cfg, render_mode=None):
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv

    class NoRobotEnv(ClothEnv):
        def _setup_scene(self):
            super()._setup_scene()
            from isaaclab.sim.utils import get_current_stage

            disable_robot(get_current_stage(), self.scene.env_prim_paths, hide=render_mode is not None)

    return NoRobotEnv(cfg, render_mode=render_mode)


def main() -> None:
    args = parse()
    if args.video:
        args.enable_cameras = True
    app = launch(args)

    import torch

    cfg, values, labels, ranges = configure(args)
    rc = cfg.rigid_cloth
    rc.designs = [f"{args.design}.yaml"]
    thickness = float(rc.thickness)
    G, M = len(rc.geometries), len(values)
    cfg.scene.num_envs = G * M  # env e: geometry e % G (the backend's own), material e // G
    print(f"[fold_test] {args.design}: {M} materials x {G} geometries = {cfg.scene.num_envs} envs", flush=True)
    if args.video:
        set_camera(cfg, args, (0.35, -0.45, 0.35))

    env = make_env(cfg, "rgb_array" if args.video else None)
    n, dev, dt = env.num_envs, env.device, env.physics_dt
    backend = env.backend
    (g,) = backend.groups
    a, rows = g.articulation, backend._env_local[g.env_ids]
    order = g.env_ids  # env id of each selected row
    e_all = torch.arange(n, device=dev)
    mat_of = (e_all // G).cpu().numpy()
    dims = backend._dims.cpu().numpy()

    def physics_step():
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=dt)

    def fold_angle():
        out = torch.empty(n, device=dev)
        out[order] = a.data.joint_pos[rows, g.fold_joint_id]
        return out

    def set_targets(angle):
        target = torch.zeros(len(rows), a.num_joints, device=dev)
        target[:, g.fold_joint_id] = angle
        a.set_joint_position_target(target, env_ids=rows)

    def folded_now():
        state, view = env._now()
        return env.task.is_folded(state, view), view.fold_error, view.footprint

    frames = []
    stride = max(1, round(1 / (dt * 30 * args.slowmo)))

    def record(i):
        if args.video and i % stride == 0:
            frames.append(env.render())

    env.reset()
    with torch.inference_mode():
        dr_samples.apply_materials(backend, e_all, values[mat_of])
        q = torch.zeros(len(rows), a.num_joints, device=dev)
        a.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=rows)  # start flat
        set_targets(0.0)
        for _ in range(int(args.settle_s / dt)):
            physics_step()
        if args.video:
            for _ in range(4):  # first renders are black or grainy while the renderer warms up
                env.render()

        # FOLD: base pinned, every hinge a motor (fold hinge -> 180 deg, all others held at 0).
        pinned_pose = a.data.root_link_pose_w[rows].clone()
        k0, d0 = a.data.joint_stiffness[rows].clone(), a.data.joint_damping[rows].clone()
        k, d = k0.clone(), d0.clone()
        for j in g.joint_ids:
            k[:, j], d[:, j] = args.motor_stiffness, args.motor_damping
        a.write_joint_stiffness_to_sim(k, env_ids=rows)
        a.write_joint_damping_to_sim(d, env_ids=rows)

        def pin():
            a.write_root_link_pose_to_sim(pinned_pose, env_ids=rows)
            a.write_root_velocity_to_sim(torch.zeros(len(rows), 6, device=dev), env_ids=rows)

        trace_t, trace_angle, trace_folded = [], [], []
        step = 0
        fold_steps, hold_steps = int(args.fold_s / dt), int(args.hold_s / dt)
        for i in range(fold_steps + hold_steps):
            s = min(1.0, (i + 1) / fold_steps)
            set_targets(math.pi * 0.5 * (1 - math.cos(math.pi * s)))  # smooth 0 -> 180 deg
            pin()
            physics_step()
            record(step)
            step += 1
            if i % 6 == 0:
                trace_t.append(round((i + 1) * dt, 4))
                trace_angle.append(fold_angle().cpu())
                trace_folded.append(folded_now()[0].float().cpu())
        reached, err_hold, foot_hold = folded_now()
        angle_hold = fold_angle()

        # RELEASE: the training hinges again (bending gains, flat targets), nothing touches it.
        a.write_joint_stiffness_to_sim(k0, env_ids=rows)
        a.write_joint_damping_to_sim(d0, env_ids=rows)
        set_targets(0.0)
        unfolded_at = torch.full((n,), float("nan"), device=dev)
        for i in range(int(args.release_s / dt)):
            if args.pin_release:
                pin()
            physics_step()
            record(step)
            step += 1
            f = folded_now()[0]
            unfolded_at[reached & ~f & torch.isnan(unfolded_at)] = (i + 1) * dt
            if i % 6 == 0:
                trace_t.append(round(args.fold_s + args.hold_s + (i + 1) * dt, 4))
                trace_angle.append(fold_angle().cpu())
                trace_folded.append(f.float().cpu())
        stays, err_end, foot_end = folded_now()
        angle_end = fold_angle()

    per_env = []
    for e in range(n):
        m = int(mat_of[e])
        unfolded = float(unfolded_at[e])
        per_env.append({
            "env": e, "kind": labels[m], "material_index": m,
            "material": {p: float(values[m][j]) for j, p in enumerate(dr_samples.PARAMS)},
            "length": float(dims[e][0]), "width": float(dims[e][1]),
            "reached": bool(reached[e]), "stays": bool(stays[e]), "failed": not (bool(reached[e]) and bool(stays[e])),
            "unfolded_after_s": None if math.isnan(unfolded) else round(unfolded, 3),
            "fold_angle_deg_hold": round(math.degrees(float(angle_hold[e])), 1),
            "fold_angle_deg_end": round(math.degrees(float(angle_end[e])), 1),
            "fold_error_mm_hold": round(float(err_hold[e]) * 1000, 1),
            "fold_error_mm_end": round(float(err_end[e]) * 1000, 1),
            "footprint_hold": round(float(foot_hold[e]), 3), "footprint_end": round(float(foot_end[e]), 3),
        })
    n_fail = sum(r["failed"] for r in per_env)
    worst = max(per_env, key=lambda r: (r["failed"], not r["reached"], r["fold_error_mm_end"]))
    result = {"test": "fold", "config": str(args.config), "overrides": list(args.overrides), "design": args.design,
              "thickness_m": thickness, "dr_ranges": ranges, "dr_random": args.dr_random,
              "motor": [args.motor_stiffness, args.motor_damping],
              "driven": "all hinges (fold -> 180 deg, others 0) with the base pinned during fold+hold"
                        + ("; base pinned during release" if args.pin_release else ""),
              "phases_s": {"settle": args.settle_s, "fold": args.fold_s, "hold": args.hold_s, "release": args.release_s},
              "fold_tolerance_m": float(cfg.task.keypoint_tolerance),
              "max_folded_footprint": float(cfg.task.max_folded_footprint), "num_envs": n,
              "failed_envs": n_fail, "not_reached_envs": sum(not r["reached"] for r in per_env),
              "failed_corner_envs": sum(r["failed"] for r in per_env if r["kind"] == "corner"),
              "failed_random_envs": sum(r["failed"] for r in per_env if r["kind"] == "random"),
              "random_envs": sum(r["kind"] == "random" for r in per_env),
              "failing_ranges": dr_samples.describe_failures(per_env, "failed"),
              "worst_env": worst, "per_env": per_env,
              "trace": {"t": trace_t, "fold_angle_rad": torch.stack(trace_angle).T.numpy().round(4).tolist(),
                        "folded": torch.stack(trace_folded).T.numpy().tolist()}}

    print("\n" + "=" * 96)
    print(f"fold_test  {args.design}  t={thickness*1000:.1f} mm  tolerance {cfg.task.keypoint_tolerance*1000:.0f} mm  "
          f"footprint < {cfg.task.max_folded_footprint}  fail: not reached, or not folded {args.release_s:g} s after release")
    print(f"failed {n_fail}/{n} envs (not reached {result['not_reached_envs']}; corners {result['failed_corner_envs']}, "
          f"random {result['failed_random_envs']}/{result['random_envs']})")
    dr_samples.print_failing_ranges(result["failing_ranges"])
    print(f"worst: env {worst['env']} {worst['length']*100:.0f}x{worst['width']*100:.0f} cm, reached={worst['reached']} "
          f"stays={worst['stays']}, unfolded after {worst['unfolded_after_s']} s, end angle {worst['fold_angle_deg_end']} deg, "
          f"material {worst['material']}")
    print("=" * 96 + "\n", flush=True)
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=30, macro_block_size=1)
        print(f"[fold_test] wrote {args.video} (env {args.video_env}, {args.slowmo:g}x slow motion)", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"[fold_test] wrote {args.out}", flush=True)

    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
