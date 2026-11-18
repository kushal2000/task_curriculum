"""Lift half the sheet to an angle, let go, and record how it falls -- VBD cloth vs hinged plates.

The question is which hinge physics (damping, armature, stiffness) makes the rigid stand-in move
like the cloth under gravity alone. The training value ``joint_damping = 0.05`` was chosen for
stability, never against the cloth, and an order-of-magnitude estimate says it is far from it: a
10 cm half plate (10 g, lever 25 mm) carries 2.5e-3 N m of gravity torque, which against 0.05 N m s
gives a terminal speed of 0.05 rad/s -- a raised flap that should drop in a fraction of a second
would take tens of seconds. This measures it, on both simulators, with one protocol:

  1. pinned reset (no noise), sheet shifted ``--dy`` away from the robot so a raised flap cannot meet
     the resting hand (fingertip clearance is recorded and flagged anyway), settle flat;
  2. raise the +x half to ``theta0`` about the crease:
       cloth  -- the moving half's particles become kinematic and are swept up over ``--lift_steps``
                 and held ``--hold_steps`` (the cloth_fold_probe mechanism), so they are at rest;
       plates -- the hinge angle is written directly at zero velocity;
  3. release and record ``--release_steps`` policy steps.

Measured per env per step, from the particle grid (the plates' grid is the same 7 x 7 layout
reconstructed from body poses, so both are read by the same code): the FOLD ANGLE between the two
halves (0 = flat, 180 = folded; mean over the grid rows of the angle at the crease particle between
the two edge particles), the flap's SAG (largest distance of the flap's interior particles from the
crease-edge chord, 0 for a rigid plate) and the crease height above its flat height.

On PhysX the hinge physics is set PER ENV (``write_joint_damping/stiffness/armature_to_sim``), so one
job sweeps a full grid: every (damping, armature, stiffness) set x every ``theta0`` x ``--reps``.

    scripts/newton_py scripts/analysis/hinge_release_probe.py --task Isaacsimenvs-Cloth-Direct-v0 \\
        --out docs/results/hinge_release/vbd.pt
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/analysis/hinge_release_probe.py \\
        --headless --task Isaacsimenvs-FoldPoolPhysx-Direct-v0 --size 0.10 \\
        --out docs/results/hinge_release/physx_s010.pt
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

POOL_TASKS = ("Isaacsimenvs-FoldPoolPhysx-Direct-v0", "Isaacsimenvs-FoldPool-Direct-v0")

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
p.add_argument("--out", required=True)
p.add_argument("--angles", default="45,90,135,170", help="release angles theta0 [deg]")
p.add_argument("--reps", type=int, default=2, help="envs per (physics set, angle)")
p.add_argument("--dy", type=float, default=-0.08, help="sheet shift in y [m]; < 0 is away from the robot")
p.add_argument("--settle_steps", type=int, default=60)
p.add_argument("--lift_steps", type=int, default=30, help="cloth only: sweep up to theta0")
p.add_argument("--hold_steps", type=int, default=30, help="cloth only: hold at theta0 before release")
p.add_argument("--release_steps", type=int, default=240)
p.add_argument("--size", type=float, default=0.10, help="plates only: sheet side [m] (box2-mid)")
p.add_argument("--damping", default="0.05,0.02,0.005,0.001,0", help="plates: hinge damping grid [N m s/rad]")
p.add_argument("--armature", default="1e-4,3e-4,1e-3,3e-3", help="plates: hinge armature grid [kg m^2]")
p.add_argument("--stiffness", default="0,5e-4,2e-3", help="plates: hinge stiffness grid [N m/rad]")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sim_device", default="cuda:0")
try:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
except ImportError:
    AppLauncher = None
args, hydra_args = p.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
pool = args.task in POOL_TASKS
if AppLauncher is not None:
    try:
        AppLauncher(args)
    except ImportError as exc:  # the Newton venv runs kit-less
        print(f"[release] kit-less: {exc}", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


def floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


angles = floats(args.angles)
if pool:
    sets = list(itertools.product(floats(args.damping), floats(args.armature), floats(args.stiffness)))
else:
    sets = [(float("nan"), float("nan"), float("nan"))]  # the cloth has no hinge
# Env e -> (set, angle, rep), set-major.
layout = [(s, a, r) for s in range(len(sets)) for a in range(len(angles)) for r in range(args.reps)]


@hydra_task_config_with_yaml(args.task, "")
def run(env_cfg, agent_cfg) -> None:
    from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch
    from isaacsimenvs.tasks.play.utils.obs_utils import OBS_FIELD_SIZES

    n = len(layout)
    env_cfg.scene.num_envs = n
    env_cfg.sim.device = args.sim_device
    env_cfg.seed = args.seed
    disable_randomization(env_cfg)
    rs = env_cfg.reset
    pose = list(rs.fixed_start_pose)
    pose[1] += args.dy
    rs.fixed_start_pose = tuple(pose)
    if pool:
        env_cfg.cloth.nan_policy = "reset"  # recorded below; one bad hinge set must not end the sweep
        env_cfg.fold_pool.shapes = ["box2-mid"]
        env_cfg.fold_pool.weights = None
        env_cfg.fold_pool.size_range = [args.size, args.size]
        env_cfg.fold_pool.init_fold_prob = 0.0
        env_cfg.rigid_cloth.randomization.enabled = False
    use_single_object_variant()
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    dev = inner.device
    c = inner.cfg.cloth
    res = int(c.resolution)
    zero = torch.zeros((n, int(inner.cfg.action_space)), device=dev)
    dt_policy = float(inner.cfg.sim.dt) * int(inner.cfg.decimation)

    # No termination: a reset mid-measurement restores the flat sheet and reads as the physics.
    _orig = inner._get_dones

    def _no_dones():
        term, trunc = _orig()
        return torch.zeros_like(term), torch.zeros_like(trunc)

    inner._get_dones = _no_dones

    # Grid: particle i * res + j sits at x = i * step - size / 2 (cloth_geometry.grid_mesh), so i
    # indexes the fold direction and i = res // 2 is the crease; the moving half is i > crease.
    mid = res // 2
    col = lambda i: torch.arange(res, device=dev) + i * res  # noqa: E731

    def measure(parts: torch.Tensor) -> dict:
        crease = parts[:, col(mid)]                        # (N, res, 3)
        edge_s = parts[:, col(0)]
        edge_m = parts[:, col(res - 1)]
        a = torch.nn.functional.normalize(edge_s - crease, dim=-1)
        b = torch.nn.functional.normalize(edge_m - crease, dim=-1)
        inner_angle = torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)))
        fold = (180.0 - inner_angle).mean(dim=1)           # (N,)
        # Sag: interior flap particles' distance from the crease -> edge chord, per row.
        chord = edge_m - crease
        L = chord.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        u = chord / L
        sag = torch.zeros(parts.shape[0], device=dev)
        for i in range(mid + 1, res - 1):
            r = parts[:, col(i)] - crease
            perp = r - (r * u).sum(-1, keepdim=True) * u
            sag = torch.maximum(sag, perp.norm(dim=-1).amax(dim=1))
        return {"fold": fold, "sag": sag, "crease_z": crease[..., 2].mean(dim=1),
                "edge_z": edge_m[..., 2].mean(dim=1)}

    off, cur = {}, 0
    for f in inner.cfg.obs.obs_list:
        off[f] = cur
        cur += int(OBS_FIELD_SIZES[f])

    def tips(obs) -> torch.Tensor | None:
        if "palm_pos" not in off or "fingertip_pos_rel_palm" not in off:
            return None
        o = obs["policy"]
        palm = o[:, off["palm_pos"]: off["palm_pos"] + 3]
        t = o[:, off["fingertip_pos_rel_palm"]: off["fingertip_pos_rel_palm"] + 15].reshape(n, 5, 3)
        return t + palm.unsqueeze(1)

    obs, _ = env.reset()
    for _ in range(args.settle_steps):
        obs, *_ = env.step(zero)
    flat = inner._particles_w().clone()
    flat_m = measure(flat)
    print(f"[release] settled flat: fold {float(flat_m['fold'].abs().max()):.2f} deg max, "
          f"crease z {float(flat_m['crease_z'].mean()):.4f}", flush=True)

    theta = torch.tensor([math.radians(angles[a]) for _, a, _ in layout], device=dev)
    info = {"task": args.task, "angles": angles, "sets": sets, "layout": layout,
            "dt_policy": dt_policy, "size": float(args.size) if pool else float(c.size),
            "dy": args.dy, "resolution": res}

    if pool:
        art = inner._pool_arts[0]
        nj = art.num_joints
        print(f"[release] plates: joints {list(art.joint_names)}, {len(sets)} hinge sets x "
              f"{len(angles)} angles x {args.reps} reps = {n} envs", flush=True)
        damp = torch.tensor([sets[s][0] for s, _, _ in layout], device=dev).unsqueeze(1).expand(-1, nj)
        arm = torch.tensor([sets[s][1] for s, _, _ in layout], device=dev).unsqueeze(1).expand(-1, nj)
        stiff = torch.tensor([sets[s][2] for s, _, _ in layout], device=dev).unsqueeze(1).expand(-1, nj)
        art.write_joint_damping_to_sim(damp.contiguous())
        art.write_joint_armature_to_sim(arm.contiguous())
        art.write_joint_stiffness_to_sim(stiff.contiguous())
        art.set_joint_position_target(torch.zeros((n, nj), device=dev))
        # Which sign lifts the flap UP is the simulator's joint frame (cf. _calibrate_start_signs):
        # write +20 deg everywhere for one step and see whether the sheet's top rose.
        z0 = flat[..., 2].amax(dim=1)
        inner.object.write_joint_positions(0, torch.full((n, nj), math.radians(20.0), device=dev))
        obs, *_ = env.step(zero)
        rise = float((inner._particles_w()[..., 2].amax(dim=1) - z0).median())
        sign = 1.0 if rise > 0.0 else -1.0
        print(f"[release] +20 deg raised the top {rise * 1e3:+.1f} mm -> up sign {sign:+.0f}", flush=True)
        inner.object.write_joint_positions(0, torch.zeros((n, nj), device=dev))
        for _ in range(args.settle_steps):
            obs, *_ = env.step(zero)
        inner.object.write_joint_positions(0, sign * theta.unsqueeze(1).expand(-1, nj).contiguous())
        info["up_sign"] = sign
        joint_trace = []
    else:
        # Kinematic lift of the moving half about the live crease, world +z up.
        P = flat.shape[1]
        moving = inner._moving_idx
        mv_i = moving // res
        mv_j = moving % res
        crease_pts = flat[:, mid * res + mv_j]                                  # (N, M, 3)
        outward = flat[:, col(res - 1)].mean(dim=1) - flat[:, col(mid)].mean(dim=1)
        outward[:, 2] = 0.0
        u = torch.nn.functional.normalize(outward, dim=-1).unsqueeze(1)         # (N, 1, 3)
        dist = ((flat[:, moving] - crease_pts) * u).sum(-1, keepdim=True)       # (N, M, 1)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev).view(1, 1, 3)

        def lifted(frac: float) -> torch.Tensor:
            th = (theta * frac).view(-1, 1, 1)
            return crease_pts + dist * (torch.cos(th) * u + torch.sin(th) * up)

        def write(pos_mv: torch.Tensor | None) -> None:
            tgt = torch.zeros((n, P, 4), device=dev)
            tgt[..., :3] = inner._particles_w()
            tgt[..., 3] = 1.0
            if pos_mv is not None:
                tgt[:, moving, :3] = pos_mv
                tgt[:, moving, 3] = 0.0
            inner._cloth.write_nodal_kinematic_target_to_sim_index(tgt, None)

        for k in range(args.lift_steps):
            write(lifted((k + 1) / args.lift_steps))
            obs, *_ = env.step(zero)
        for _ in range(args.hold_steps):
            write(lifted(1.0))
            obs, *_ = env.step(zero)
        held = measure(inner._particles_w())
        print(f"[release] held: fold {[round(float(v), 1) for v in held['fold'][:: args.reps]]}", flush=True)
        info["held_fold"] = held["fold"].cpu()
        write(None)

    trace = {k: [] for k in ("fold", "sag", "crease_z", "edge_z")}
    clear = []
    finite = torch.ones(n, dtype=torch.bool, device=dev)
    for k in range(args.release_steps + 1):
        parts = inner._particles_w()
        finite &= torch.isfinite(parts).all(dim=-1).all(dim=-1)
        m = measure(torch.nan_to_num(parts))
        for key in trace:
            trace[key].append(m[key].cpu())
        t = tips(obs)
        if t is not None:
            # palm_pos is in the env frame (cf. fold_start_probe), the particles are world.
            rel = torch.nan_to_num(parts) - inner.scene.env_origins.unsqueeze(1)
            d = torch.cdist(t, rel).amin(dim=(1, 2))
            clear.append(d.cpu())
        if pool:
            joint_trace.append(_to_torch(art.data.joint_pos).detach().cpu().clone())
        if k < args.release_steps:
            obs, *_ = env.step(zero)

    out = {**info, **{k: torch.stack(v, dim=1) for k, v in trace.items()},
           "finite": finite.cpu(), "flat_crease_z": flat_m["crease_z"].cpu()}
    if clear:
        out["tip_clearance"] = torch.stack(clear, dim=1)
    if pool:
        out["joint_pos"] = torch.stack(joint_trace, dim=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)

    fold = out["fold"]
    for a, ang in enumerate(angles):
        sel = [e for e, (s, aa, _) in enumerate(layout) if aa == a and s == 0]
        f = fold[sel].mean(dim=0)
        print(f"[release] theta0 {ang:5.1f}: fold at 0/0.1/0.25/0.5/1/2/4 s = "
              + " ".join(f"{float(f[min(int(t / dt_policy), f.numel() - 1)]):6.1f}"
                         for t in (0, 0.1, 0.25, 0.5, 1, 2, 4))
              + (f"  (set {sets[0]})" if pool else ""), flush=True)
    if clear:
        print(f"[release] fingertip clearance min {float(out['tip_clearance'].min()) * 1e3:.1f} mm", flush=True)
    print(f"[release] nonfinite envs {int((~finite).sum())}/{n}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    run()
    os._exit(0)
