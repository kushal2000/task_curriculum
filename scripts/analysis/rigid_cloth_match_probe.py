"""Does the chain now match the cloth where it matters for transfer? Measured, both manipulands.

Three fixes went into `RigidClothEnv` (heading at reset, contact height, friction) plus per-env
physics randomisation. Each one is a number that can silently be wrong while training still runs,
so each is measured here on the live solver, and the SAME protocol is run on the VBD cloth so the
chain's numbers have a target rather than an expectation:

  heading     write a different requested yaw into every env through the task's own reset calls
              (`object.write_root_pose_to_sim` then `write_root_velocity_to_sim`, the order
              `reset_utils` uses), step, and recover the achieved heading from the particle grid.
              Also a real `env.reset()` with reset noise ON: the spread of headings it produces.
  height      settle flat, report the mid-plane z of the grid; and the contact margins involved.
  friction    give the settled sheet a uniform 0.4 m/s slide along x and measure how far it goes:
              mu = v^2 / (2 g d). The cloth's own VBD friction is the target.
              Plus the solver's RESOLVED per-pair coefficient (priority rule, as MJWarp computes it)
              for slat-table, slat-fingertip and fingertip-table.
  fold        (chain) drive the hinges to the fold and settle: best fold error, and where the top ply
              ends up -- the check that slat-slat contact is gone and the plies still close.
  randomise   (chain, if enabled) per-env readback of what MJWarp is actually simulating --
              friction, margin, mass, hinge damping and stiffness -- against what was drawn, the
              per-env slide distance against the drawn friction, and the cost of a re-draw.

    scripts/newton_py -m scripts.analysis.rigid_cloth_match_probe --variant box3-mid --out x.json
    scripts/newton_py -m scripts.analysis.rigid_cloth_match_probe \
        --task Isaacsimenvs-Cloth-Direct-v0 --out y.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

G = 9.81


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-RigidCloth-Direct-v0")
    p.add_argument("--variant", default=None)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--settle_steps", type=int, default=90)
    p.add_argument("--slide_speed", type=float, default=0.4)
    p.add_argument("--slide_steps", type=int, default=60)
    p.add_argument("--randomize", action="store_true", help="env.rigid_cloth.randomization.enabled")
    p.add_argument("--sim_device", default="cuda:0")
    p.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    chain = args.task == "Isaacsimenvs-RigidCloth-Direct-v0"
    if args.variant:
        hydra_args += [f"env.rigid_cloth.variant={args.variant}"]
    if args.randomize:
        hydra_args += ["env.rigid_cloth.randomization.enabled=true"]
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        AppLauncher(args)
    except ImportError as exc:
        print(f"[match] kit-less: {exc}", flush=True)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.protocol import use_single_object_variant
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    out: dict = {"task": args.task, "variant": args.variant, "num_envs": args.num_envs,
                 "randomize": bool(args.randomize), "errors": {}}

    def section(name):
        def deco(fn):
            def run(*a, **k):
                try:
                    out[name] = fn(*a, **k)
                except Exception as exc:  # recorded, and the remaining sections still run
                    import traceback

                    traceback.print_exc()
                    out["errors"][name] = f"{type(exc).__name__}: {exc}"
                    print(f"[match] SECTION {name} FAILED: {exc}", flush=True)
            return run
        return deco

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.sim_device
        env_cfg.cloth.nan_policy = "reset"
        # The heading section needs the REAL reset distribution, so reset noise stays on for the
        # first env.reset() and is pinned only afterwards, by hand, for the measurements.
        use_single_object_variant()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        dev = inner.device
        n = inner.num_envs
        res = int(inner.cfg.cloth.resolution)
        origins = inner.scene.env_origins
        act = torch.zeros((n, int(inner.cfg.action_space)), device=dev)
        obj = inner.object

        def cloud() -> torch.Tensor:
            return inner._cloth.data.nodal_pos_w.reshape(n, -1, 3)

        def heading_deg(c: torch.Tensor) -> torch.Tensor:
            # Along one grid row: vertex (0, res-1) minus vertex (0, 0). The sheet's own x axis.
            d = c[:, res - 1, :2] - c[:, 0, :2]
            return torch.rad2deg(torch.atan2(d[:, 1], d[:, 0]))

        def place(yaw: torch.Tensor) -> None:
            pose = torch.zeros((n, 7), device=dev)
            pose[:, :3] = origins
            pose[:, 2] += float(inner.cfg.reset.table_object_z_offset)
            pose[:, 3] = torch.cos(0.5 * yaw)   # (w, x, y, z), the order both adapters decode
            pose[:, 6] = torch.sin(0.5 * yaw)
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev))

        def settle(k: int) -> None:
            for _ in range(k):
                env.step(act)

        env.reset()

        @section("heading")
        def s_heading() -> dict:
            from isaacsimenvs.tasks.cloth.utils.cloth_adapter import task_yaw

            # (a) the real reset, noise on: how spread are the headings it produces?
            env.reset()
            for _ in range(2):
                env.step(act)
            real = heading_deg(cloud())
            # (b) random reset quaternions through the task's own (wrapped) write path, against the
            # heading the SHARED decode predicts for them -- i.e. "does this manipuland start where
            # the other one would for the same draw". The wrapper reorders wxyz -> xyzw first.
            g = torch.Generator(device=dev).manual_seed(0)
            q = torch.nn.functional.normalize(torch.randn((n, 4), device=dev, generator=g), dim=-1)
            pose = torch.zeros((n, 7), device=dev)
            pose[:, :3] = origins
            pose[:, 3:7] = q
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev))
            env.step(act)
            got = heading_deg(cloud())
            conv = pose.clone()
            conv[:, 3:7] = torch.cat((q[:, 1:4], q[:, 0:1]), dim=-1)
            want = torch.rad2deg(task_yaw(conv))
            d = (got - want).remainder(360.0)
            offset = float(d.median())          # the grid row's heading at decode 0, a constant
            err = (d - offset + 180.0).remainder(360.0) - 180.0
            r = {
                "decoded_deg": [round(float(v), 1) for v in want[:: max(n // 8, 1)]],
                "achieved_deg": [round(float(v), 1) for v in got[:: max(n // 8, 1)]],
                "grid_row_offset_deg": round(offset, 2),
                "max_abs_error_deg": round(float(err.abs().max()), 3),
                "real_reset_heading_std_deg": round(float(real.std()), 1),
                "real_reset_heading_range_deg": [round(float(real.min()), 1), round(float(real.max()), 1)],
            }
            print(f"[match] heading: {r}", flush=True)
            return r

        s_heading()

        # Everything below starts from the pinned pose: centred, axis-aligned.
        place(torch.zeros(n, device=dev))
        settle(args.settle_steps)

        @section("height")
        def s_height() -> dict:
            c = cloud()
            table_top = float(inner.cfg.reset.table_reset_z) + float(inner.cfg.cloth.table_half_thickness)
            z = c[..., 2]
            # Per-env MEDIAN vertex height: robust to the few envs whose sheet the robot's resting
            # hand is touching, which drag a plain mean (the cloth's own mean read 8.6 and 12.0 mm
            # in two runs of the same scene).
            med = z.median(dim=1).values
            spread = z.amax(dim=1) - z.amin(dim=1)
            flat = spread < 0.004
            r = {
                "table_surface_z": round(table_top, 4),
                "midplane_z_mean": round(float(z.mean()), 4),
                "midplane_z_min": round(float(z.min()), 4),
                "midplane_z_max": round(float(z.max()), 4),
                "median_above_table_mm": round((float(med.median()) - table_top) * 1e3, 2),
                "flat_envs": int(flat.sum()),
                "flat_envs_median_above_table_mm": round(
                    (float(med[flat].median()) - table_top) * 1e3, 2) if bool(flat.any()) else None,
            }
            model = inner.sim.physics_manager.get_model()
            margin = model.shape_margin.numpy()
            labels = [s.rsplit("/", 1)[-1] for s in model.body_label]
            sb = model.shape_body.numpy()
            by = {}
            for s, b in enumerate(sb):
                lab = labels[b] if b >= 0 else "<static>"
                key = ("slat" if lab.startswith("slat_") else
                       "fingertip" if any(t in lab for t in ("_DP", "tip")) else
                       "table" if "able" in lab or lab == "<static>" else "other")
                by.setdefault(key, set()).add(round(float(margin[s]) * 1e3, 3))
            r["shape_margin_mm"] = {k: sorted(v)[:6] for k, v in by.items()}
            if chain and args.randomize:
                m = inner.dr_sample["contact_margin"]
                zz = med
                r["per_env_margin_vs_height_corr"] = round(
                    float(torch.corrcoef(torch.stack((m, zz)))[0, 1]), 3
                )
            print(f"[match] height: {r}", flush=True)
            return r

        s_height()

        @section("friction")
        def s_friction() -> dict:
            c0 = cloud().mean(dim=1)
            v = torch.zeros((n, 6), device=dev)
            v[:, 0] = args.slide_speed
            if chain:
                ids = torch.arange(n, device=dev)
                inner._chain.write_root_velocity_to_sim(v, ids)
            else:
                pv = torch.zeros_like(cloud())
                pv[..., 0] = args.slide_speed
                inner._cloth.write_nodal_velocity_to_sim_index(pv, torch.arange(n, device=dev))
            xs = []
            for _ in range(args.slide_steps):
                env.step(act)
                xs.append(float((cloud().mean(dim=1)[:, 0] - c0[:, 0]).mean()))
            d = cloud().mean(dim=1)[:, 0] - c0[:, 0]
            mu = args.slide_speed ** 2 / (2.0 * G * d.clamp(min=1e-5))
            r = {
                "slide_speed": args.slide_speed,
                "slide_mm_mean": round(float(d.mean()) * 1e3, 2),
                "slide_mm_min": round(float(d.min()) * 1e3, 2),
                "slide_mm_max": round(float(d.max()) * 1e3, 2),
                "mu_table_effective_mean": round(float(mu.mean()), 3),
                "mu_table_effective_median": round(float(mu.median()), 3),
                "trace_mm": [round(x * 1e3, 1) for x in xs[:: max(len(xs) // 12, 1)]],
            }
            if chain and args.randomize:
                s = inner.dr_sample["slat_friction"]
                ok = (d > 0.005) & (d < 0.1)      # drop the envs the resting hand interferes with
                r["per_env_mu_vs_drawn_corr"] = round(float(torch.corrcoef(torch.stack((s[ok], mu[ok])))[0, 1]), 3)
                r["per_env_mu_over_drawn_median"] = round(float((mu[ok] / s[ok]).median()), 3)
            if chain:
                r["resolved_pairs"] = resolved_pairs()
            print(f"[match] friction: {r}", flush=True)
            return r

        def resolved_pairs() -> dict:
            """MJWarp's contact_params rule on the solver's own arrays, world 0."""

            solver = inner.sim.physics_manager._solver
            # `.numpy()`, not `wp.to_torch`: torch views of MJWarp solver arrays fault.
            prio = torch.as_tensor(solver.mjw_model.geom_priority.numpy())
            fr = torch.as_tensor(solver.mjw_model.geom_friction.numpy())
            # `newton_shape_to_mjc_geom` faults on host reads; invert the other map instead.
            g2s = solver.mjc_geom_to_newton_shape.numpy()[0]
            geom_of = torch.full((int(inner.sim.physics_manager.get_model().shape_count),), -1, dtype=torch.long)
            for g, s_ in enumerate(g2s):
                if s_ >= 0:
                    geom_of[int(s_)] = g
            model = inner.sim.physics_manager.get_model()
            labels = [s.rsplit("/", 1)[-1] for s in model.body_label]
            sb = model.shape_body.numpy()
            table = [s for s, b in enumerate(sb) if b >= 0 and "able" in labels[b]] or \
                    [s for s, b in enumerate(sb) if b < 0]
            slat = int(inner._slat_shapes[0, 0])
            tip = int(inner._tip_shapes[0, 0])

            def mu(s1, s2):
                g1, g2 = int(geom_of[s1]), int(geom_of[s2])
                if g1 < 0 or g2 < 0:
                    return None
                p1 = int(prio[g1] if prio.dim() == 1 else prio[0, g1])
                p2 = int(prio[g2] if prio.dim() == 1 else prio[0, g2])
                f1, f2 = float(fr[0, g1, 0]), float(fr[0, g2, 0])
                return round(f1 if p1 > p2 else f2 if p2 > p1 else max(f1, f2), 3)

            t = table[0]
            return {"slat_table": mu(slat, t), "slat_fingertip": mu(slat, tip),
                    "fingertip_table": mu(tip, t)}

        s_friction()

        @section("fold")
        def s_fold() -> dict | None:
            if not chain:
                return None
            from scripts.analysis.rigid_cloth_env_probe import _fold_the_chain

            place(torch.zeros(n, device=dev))
            settle(30)
            spec = inner.cfg.rigid_cloth.spec
            r = _fold_the_chain(inner, spec, 120)
            c = cloud()
            bad = ~torch.isfinite(c).all(dim=-1).all(dim=-1)
            r["nonfinite_envs"] = int(bad.sum())
            if bool(bad.any()) and args.randomize:
                r["nonfinite_draws"] = {
                    k: [round(float(v), 5) for v in inner.dr_sample[k][bad]] for k in inner.dr_sample
                }
            good = c[~bad]
            r["top_z_max"] = round(float(good[..., 2].max()), 4) if good.numel() else None
            r["dropped_slat_pairs"] = int(getattr(inner, "_dropped_slat_pairs", 0))
            print(f"[match] fold: {r}", flush=True)
            return r

        s_fold()

        @section("randomization")
        def s_dr() -> dict | None:
            if not (chain and args.randomize):
                return None

            solver = inner.sim.physics_manager._solver
            mm = solver.mjw_model
            def host(a):
                return torch.as_tensor(a.numpy(), device=dev)

            fr = host(mm.geom_friction)
            gm = host(mm.geom_margin)
            bm = host(mm.body_mass)
            bias = host(mm.actuator_biasprm)
            g2s = solver.mjc_geom_to_newton_shape.numpy()
            b2n = host(solver.mjc_body_to_newton).long()
            a2n = host(solver.mjc_actuator_to_newton_idx).long()
            model = inner.sim.physics_manager.get_model()
            dpw = model.joint_dof_count // n
            ds = inner.dr_sample
            ids = torch.arange(n, device=dev)
            g_slat = int(inner._slat_geoms[0])
            g_tip = int(inner._tip_geoms[0])
            body0 = inner._slat_bodies[:, 0]
            mjb = torch.stack([(b2n[w] == body0[w]).nonzero()[0, 0] for w in range(n)])
            base_mass = inner._mass_base[:, 0]

            def corr(a, b):
                return round(float(torch.corrcoef(torch.stack((a.float(), b.float())))[0, 1]), 4)

            r = {
                "friction_slat": corr(fr[ids, g_slat, 0], ds["slat_friction"]),
                "friction_slat_max_abs_err": round(float((fr[ids, g_slat, 0] - ds["slat_friction"]).abs().max()), 5),
                "friction_tip_max_abs_err": round(float((fr[ids, g_tip, 0] - ds["fingertip_friction"]).abs().max()), 5),
                "margin_max_abs_err_mm": round(float((gm[ids, g_slat] - inner._slat_margin_base[:, 0] - ds["contact_margin"]).abs().max()) * 1e3, 4),
                "mass_max_rel_err": round(float((bm[ids, mjb] / (base_mass * ds["mass"]) - 1.0).abs().max()), 5),
            }
            # The chain's hinge actuators are the last ones built. Find, among the tail, the
            # coefficient that tracks each drawn gain across envs, and how exactly it tracks.
            gain = host(mm.actuator_gainprm)
            best = {"hinge_damping": (0.0, None), "hinge_stiffness": (0.0, None)}
            for a in range(max(gain.shape[1] - 6, 0), gain.shape[1]):
                for k in range(3):
                    for src_, arr in (("gain", gain), ("bias", bias)):
                        col = arr[:, a, k].abs()
                        if float(col.std()) == 0.0:
                            continue
                        for key in best:
                            c = corr(col, ds[key])
                            if c > best[key][0]:
                                err = float((col / ds[key].clamp(min=1e-12) - 1.0).abs().max())
                                best[key] = (c, f"act {a} {src_}prm[{k}] corr {c} max rel err {err:.2e}")
            r["damping_readback"] = best["hinge_damping"][1] or "no actuator coefficient tracks it"
            r["stiffness_readback"] = best["hinge_stiffness"][1] or "no actuator coefficient tracks it"
            r["drawn_ranges"] = {k: [round(float(v.min()), 5), round(float(v.max()), 5)] for k, v in ds.items()}
            # Cost of one re-draw at a typical per-step reset batch.
            batch = max(n // 16, 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(50):
                inner._randomize_physics(torch.randperm(n, device=dev)[:batch])
            torch.cuda.synchronize()
            r["redraw_ms"] = round((time.perf_counter() - t0) / 50 * 1e3, 3)
            print(f"[match] randomization: {r}", flush=True)
            return r

        s_dr()

        @section("timing")
        def s_timing() -> dict:
            for _ in range(10):
                env.step(act)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                env.step(act)
            torch.cuda.synchronize()
            w = time.perf_counter() - t0
            return {"env_steps_per_s": round(100 * n / w, 1), "step_ms": round(w / 100 * 1e3, 3)}

        s_timing()
        print(f"[match] timing: {out.get('timing')}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"[match] wrote {args.out}", flush=True)
        print(f"[match] errors: {out['errors'] or 'none'}", flush=True)

    run()


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
