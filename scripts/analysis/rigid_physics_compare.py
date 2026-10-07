"""The rigid manipuland's physics, measured the same way on Newton and on PhysX, without the robot.

The PhysX port passes observation parity (``cloth_obs_parity.py``) but Newton-trained policies score
lower on it -- the sheets slide further and leave the table. This isolates the manipuland from the
policy and compares, per simulator and per shape:

  settle   drop flat from the spawn height onto the table: mid-plane height above the table top,
           flatness (max vertex deviation from the centred flat grid, after a rigid fit)
  slide    a uniform 0.4 m/s along x on the settled sheet: slide distance -> mu = v^2 / (2 g d)
  press    a downward force (0.5 / 2 / 8 N in total) on the sheet's bodies: how far the mid-plane sinks,
           i.e. the contact's stiffness against the table
  release  every hinge written to +60 deg (the shape's own half fold, scaled), then free:
           fold-angle and flatness trace, i.e. how the passive sheet falls back under gravity

    scripts/newton_py scripts/analysis/rigid_physics_compare.py --task Isaacsimenvs-RigidCloth-Direct-v0 --out a.json
    .venv_isaacsim/bin/python scripts/analysis/rigid_physics_compare.py --headless \\
        --task Isaacsimenvs-RigidClothPhysx-Direct-v0 --out b.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

G = 9.81
parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out", required=True)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--slide_speed", type=float, default=0.4)
parser.add_argument("--slide_steps", type=int, default=60)
parser.add_argument("--release_steps", type=int, default=90)
parser.add_argument("--release_deg", type=float, default=60.0)
parser.add_argument("--press_newtons", type=float, nargs="*", default=[0.5, 2.0, 8.0])
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
try:
    app = AppLauncher(args).app
except ImportError:
    app = None

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


@hydra_task_config_with_yaml(args.task, "")
def run(env_cfg, agent_cfg) -> None:
    from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.cloth.nan_policy = "reset"
    disable_randomization(env_cfg)
    env_cfg.rigid_cloth.randomization.enabled = False
    if hasattr(env_cfg, "fold_pool"):
        # Every env at the nominal size, so per-shape numbers are comparable across simulators.
        env_cfg.fold_pool.size_range = [0.10, 0.10]
    use_single_object_variant()
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    dev, n = inner.device, inner.num_envs
    dt = float(inner.cfg.sim.dt)
    origins = inner.scene.env_origins
    obj = inner.object
    pool = hasattr(inner, "_pool_arts")
    arts = list(inner._pool_arts) if pool else [inner._chain]
    shapes = [sh.name for sh in inner._pool] if pool else [inner.cfg.rigid_cloth.variant]
    rest = torch.tensor(inner._cloth_rest_local, device=dev, dtype=torch.float32)
    rest_c = rest - rest.mean(dim=0, keepdim=True)
    table_top = float(inner.cfg.reset.table_reset_z) + float(inner.cfg.cloth.table_half_thickness)

    env.reset()  # puts the table at its working height

    def raw(k: int, record=None) -> list:
        out = []
        for _ in range(k):
            if pool:
                obj.park()
            inner.sim.step(render=False)
            inner.scene.update(dt)
            if record is not None:
                out.append(record())
        return out

    def cloud() -> torch.Tensor:
        return inner._particles_w()

    def flat_dev(c: torch.Tensor) -> torch.Tensor:
        """Max vertex distance from the flat grid after the best rigid fit (Kabsch), per env."""
        q = c - c.mean(dim=1, keepdim=True)
        h = torch.einsum("pi,npj->nij", rest_c, q)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        d = torch.ones((c.shape[0], 3), device=dev)
        d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
        fit = torch.einsum("nij,pj->npi", r, rest_c)
        return (q - fit).norm(dim=-1).amax(dim=-1)

    def place() -> None:
        pose = torch.zeros((n, 7), device=dev)
        pose[:, :3] = origins
        pose[:, 3] = 1.0
        type(obj).write_root_pose_to_sim(obj, pose)
        obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev))

    results = {"task": args.task, "num_envs": n, "shapes": {}}
    all_ids = torch.arange(n, device=dev)
    for s, name in enumerate(shapes):
        art = arts[s]
        if pool:
            obj.select(all_ids, torch.full((n,), s, device=dev, dtype=torch.long))
        place()
        raw(args.settle_steps)
        c = cloud()
        z = c[..., 2]
        r = {
            "midplane_above_table_mm": round((float(z.median()) - table_top) * 1e3, 2),
            "z_spread_mm_median": round(float((z.amax(1) - z.amin(1)).median()) * 1e3, 2),
            "flat_dev_mm_median": round(float(flat_dev(c).median()) * 1e3, 3),
            "flat_dev_mm_max": round(float(flat_dev(c).max()) * 1e3, 3),
        }

        # press: a downward force spread over the sheet's bodies, then how far the mid-plane sinks
        z_rest = float(cloud()[..., 2].median())
        nb = int(art.num_bodies)
        press = {}
        for total in args.press_newtons:
            f = torch.zeros((n, nb, 3), device=dev)
            f[..., 2] = -float(total) / nb
            art.set_external_force_and_torque(f, torch.zeros_like(f), is_global=True)
            for _ in range(40):
                art.write_data_to_sim()
                inner.sim.step(render=False)
                inner.scene.update(dt)
            press[str(total)] = round((z_rest - float(cloud()[..., 2].median())) * 1e3, 3)
        zero = torch.zeros((n, nb, 3), device=dev)
        art.set_external_force_and_torque(zero, zero, is_global=True)
        art.write_data_to_sim()
        raw(30)
        r["press_sink_mm"] = press

        # slide
        c0 = cloud().mean(dim=1)
        v = torch.zeros((n, 6), device=dev)
        v[:, 0] = args.slide_speed
        art.write_root_velocity_to_sim(v, all_ids)
        trace = raw(args.slide_steps, lambda: float((cloud().mean(dim=1)[:, 0] - c0[:, 0]).mean()))
        d = cloud().mean(dim=1)[:, 0] - c0[:, 0]
        mu = args.slide_speed**2 / (2.0 * G * d.clamp(min=1e-5))
        r["slide_mm_median"] = round(float(d.median()) * 1e3, 2)
        r["mu_effective_median"] = round(float(mu.median()), 3)
        r["slide_trace_mm"] = [round(x * 1e3, 1) for x in trace[::6]]
        r["flat_dev_after_slide_mm_median"] = round(float(flat_dev(cloud()).median()) * 1e3, 3)

        # release from a partial fold
        place()
        raw(args.settle_steps)
        nj = int(art.num_joints)
        names = list(art.joint_names)
        q = torch.zeros((n, nj), device=dev)
        if pool:
            sh = inner._pool[s]
            from isaacsimenvs.tasks.cloth.utils import fold_shapes as fsh

            stage = sh.folds[fsh.HALF_FOLD][0]
            frac = math.radians(args.release_deg) / math.pi
            for child, ang in stage.items():
                jn = f"slat_joint_{child}"
                if jn in names:
                    q[:, names.index(jn)] = float(ang) * frac
            obj.write_joint_positions(s, q)
        else:
            from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc

            spec = inner.cfg.rigid_cloth.spec
            angles = rc.chain_fold_angles(spec, inner.cfg.rigid_cloth.joint_limit)
            frac = math.radians(args.release_deg) / max(abs(a) for a in angles if a)
            for i, a in enumerate(angles):
                jn = rc.slat_joint_name(i)
                if a and jn in names:
                    q[:, names.index(jn)] = float(a) * frac
            obj.write_joint_positions(q)
        moved = (q.abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0]

        def hinge_deg() -> float:
            jp = _to_torch(art.data.joint_pos)
            return math.degrees(float(jp[:, moved].abs().amax(dim=1).median()))

        tr = raw(args.release_steps, hinge_deg)
        r["release_written_deg"] = round(math.degrees(float(q.abs().max())), 1)
        r["release_hinge_trace_deg"] = [round(x, 1) for x in tr[::6]]
        r["release_final_flat_dev_mm_median"] = round(float(flat_dev(cloud()).median()) * 1e3, 3)
        r["nonfinite_envs"] = int((~torch.isfinite(cloud()).all(dim=-1).all(dim=-1)).sum())

        # the full half fold, driven kinematically, then left to settle on its own
        place()
        raw(args.settle_steps)
        q = q / frac
        if pool:
            obj.write_joint_positions(s, q)
        else:
            obj.write_joint_positions(q)
        best = torch.full((n,), float("inf"), device=dev)

        def track():
            nonlocal best
            best = torch.minimum(best, inner.fold_error())
            return None

        raw(args.release_steps, track)
        fe, fp = inner.fold_error(), inner.footprint_ratio()
        r["halffold_best_err_mm_median"] = round(float(best.median()) * 1e3, 2)
        r["halffold_final_err_mm_median"] = round(float(fe.median()) * 1e3, 2)
        r["halffold_final_footprint_median"] = round(float(fp.median()), 3)
        r["halffold_folded_frac"] = round(float(inner.is_folded().float().mean()), 3)
        if pool and inner._pool[s].seams:
            from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _quat_apply

            sh = inner._pool[s]
            pos, quat = inner._cloth._pose(art)
            bnames = list(art.body_names)
            gap = torch.zeros(n, device=dev)
            for seam in sh.seams:
                for pt in (seam.a, seam.b):
                    x = torch.tensor([pt[0], pt[1], sh.hinge_z], device=dev).expand(n, 3)
                    i1, i2 = bnames.index("slat_" + seam.plate1), bnames.index("slat_" + seam.plate2)
                    w1 = pos[:, i1] + _quat_apply(quat[:, i1], x)
                    w2 = pos[:, i2] + _quat_apply(quat[:, i2], x)
                    gap = torch.maximum(gap, (w1 - w2).norm(dim=-1))
            r["halffold_seam_gap_mm_median"] = round(float(gap.median()) * 1e3, 3)
            r["halffold_seam_gap_mm_max"] = round(float(gap.max()) * 1e3, 3)
        results["shapes"][name] = r
        print(f"[compare] {args.task} {name}: {json.dumps(r)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[compare] wrote {args.out}", flush=True)


if __name__ == "__main__":
    run()
    os._exit(0)
