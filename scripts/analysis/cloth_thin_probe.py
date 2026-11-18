"""Thinner VBD cloth and a grippier table: does the sim break (fingers through the sheet, sheet
through the table, NaNs, jitter)?

    scripts/newton_py scripts/analysis/cloth_thin_probe.py --task <id> --checkpoint <pth> --out x.json \
        [--table_mu 2.0] [hydra: env.cloth.thickness=0.004 ...]

Three phases in one process, same env:

1. **settle** (no policy): arm held, hand at its reset pose (well above the table), the sheet drops
   from its spawn height and rests. Reports the particle-to-table-top gap (should equal
   ``particle_radius``), the lowest particle, and rest jitter.
2. **slide**: every particle gets the same horizontal velocity ``--slide_v``; the sheet decelerates
   by table friction only, so ``mu_eff = v^2 / (2 g d)`` from the stopping distance ``d``. This checks
   that the requested table friction actually reaches the cloth (Newton's material patch sets every
   non-fingertip shape, the table included, to ``assets.robot_friction``; ``--table_mu`` rewrites the
   table shapes' ``shape_material_mu`` after build -- probe-only, the repo patch is untouched).
3. **policy**: one episode per env with the checkpoint; per step: finger pierce (finger-link axis
   crosses own cloth triangle, ``utils/pierce.py``), particles below the table top (centre under the
   surface = sheet through the table), outcome per env.

``--keep_obs_scale`` (default on) keeps the ``object_scales`` obs at the 16 mm thickness the policies
were trained with, so only the physics changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUTCOMES = ("fold_held", "fall", "hand_far", "nonfinite", "timeout")


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=900)
    p.add_argument("--table_mu", type=float, default=None, help="table shape mu (default: leave as built)")
    p.add_argument("--slide_v", type=float, default=0.4)
    p.add_argument("--settle_steps", type=int, default=120)
    p.add_argument("--slide_steps", type=int, default=90)
    p.add_argument("--keep_obs_scale", type=int, default=1)
    p.add_argument("--random_start", type=int, default=1, help="DR off but keep reset start-pose noise")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        AppLauncher(args)
    except ImportError:
        pass

    import copy

    import gymnasium as gym
    import torch
    import warp as wp

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.newton.patches import notify_solver
    from isaacsimenvs.tasks.cloth.utils.pierce import PierceDetector
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = 0.01
        rs0 = copy.deepcopy(env_cfg.reset)
        disable_randomization(env_cfg)
        if args.random_start:
            env_cfg.reset = rs0
        use_single_object_variant()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        c = inner.cfg.cloth
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        res = {"task": args.task, "checkpoint": args.checkpoint, "overrides": hydra_args,
               "num_envs": n, "seed": args.seed, "thickness": float(c.thickness),
               "particle_radius": float(c.particle_radius), "resolution": int(c.resolution),
               "spacing": float(c.spacing), "soft_contact_mu": float(c.soft_contact_mu)}

        # --- table friction ----------------------------------------------------------------
        labels = [str(x) for x in model.body_label]
        shape_body = model.shape_body.numpy()
        mu = model.shape_material_mu.numpy().copy()
        table_shapes = [s for s in range(len(mu)) if shape_body[s] >= 0 and "/Table" in labels[shape_body[s]]]
        mu_before = sorted({round(float(mu[s]), 4) for s in table_shapes})
        if args.table_mu is not None:
            mu[table_shapes] = args.table_mu
            model.shape_material_mu.assign(wp.array(mu, dtype=model.shape_material_mu.dtype))
            notify_solver(inner, joints=False, shapes=True)
        res["table_shapes"] = len(table_shapes)
        res["table_mu_before"] = mu_before
        res["table_mu"] = sorted({round(float(model.shape_material_mu.numpy()[s]), 4) for s in table_shapes})
        res["table_body_labels"] = sorted({labels[shape_body[s]].split("/env_")[-1].split("/", 1)[-1]
                                           for s in table_shapes})[:6]
        tm = res["table_mu"][0] if res["table_mu"] else float("nan")
        res["expected_cloth_table_mu"] = round(float(np.sqrt(c.soft_contact_mu * tm)), 3)
        print(f"[thin] table shapes {len(table_shapes)} mu {mu_before} -> {res['table_mu']} "
              f"(bodies {res['table_body_labels']}); expected cloth-table mu "
              f"{res['expected_cloth_table_mu']}; thickness {c.thickness * 1e3:.1f} mm, res {c.resolution}", flush=True)

        base = float(inner.cfg.reward.object_base_size)

        def fix_obs_scale():
            if args.keep_obs_scale:
                inner._object_scale_per_env[:, 2] = 0.016 / base

        table_half = float(c.table_half_thickness)

        def table_top():
            return (inner._table_z_per_env + table_half + inner.scene.env_origins[:, 2])[:, None]  # (E,1)

        def parts():
            return inner.object._particles(), inner.object._velocities()

        tip_ids, _ = inner.robot.find_bodies(inner.robot_spec.fingertip_body_regex)

        # --- 1. settle -------------------------------------------------------------------------
        obs, _ = env.reset()
        fix_obs_scale()
        zero = torch.zeros((n, inner.cfg.action_space), device=dev)
        # Hold every joint target where reset put it (the action pipeline's replay hook).
        inner._replay_target_lab_order = inner._cur_targets.clone()
        z_hist, nonfinite_settle, resets_passive = [], 0, 0
        t0 = time.time()
        for k in range(args.settle_steps):
            obs, _r, term, trunc, _x = env.step(zero)
            resets_passive += int((term.bool() | trunc.bool()).sum())
            fix_obs_scale()
            x, v = parts()
            if k >= args.settle_steps - 30:
                z_hist.append((x[..., 2] - table_top()).clone())
            nonfinite_settle += int((~torch.isfinite(x)).any(dim=(1, 2)).sum())
        settle_wall = time.time() - t0
        gap = torch.stack(z_hist)                       # (T, E, P)
        tips_z = inner.robot.data.body_pos_w[:, tip_ids, 2] - table_top()
        x, v = parts()
        res["settle"] = {
            "gap_median_mm": round(float(gap[-1].median()) * 1e3, 2),
            "gap_min_mm": round(float(gap.min()) * 1e3, 2),
            "gap_max_mm": round(float(gap[-1].max()) * 1e3, 2),
            "frac_particles_below_top": round(float((gap[-1] < 0).float().mean()), 4),
            "jitter_z_std_mm": round(float(gap.std(dim=0).median()) * 1e3, 3),
            "jitter_z_std_p99_mm": round(float(torch.quantile(gap.std(dim=0).flatten().float(), 0.99)) * 1e3, 3),
            "rest_speed_p99_mmps": round(float(torch.quantile(v.norm(dim=-1).flatten(), 0.99)) * 1e3, 2),
            "fingertip_min_height_mm": round(float(tips_z.min()) * 1e3, 1),
            "nonfinite_env_steps": nonfinite_settle,
            "wall_s_per_step": round(settle_wall / args.settle_steps, 4),
        }
        print("[thin] settle " + json.dumps(res["settle"]), flush=True)

        # --- 2. slide ----------------------------------------------------------------------------
        ids = torch.arange(n, device=dev)
        P = x.shape[1]
        vel = torch.zeros((n, P, 3), device=dev)
        vel[..., 0] = args.slide_v
        inner.object._cloth.write_nodal_velocity_to_sim_index(vel, ids)
        c0 = x.mean(dim=1).clone()
        trace = []
        for k in range(args.slide_steps):
            obs, _r, term, trunc, _x = env.step(zero)
            resets_passive += int((term.bool() | trunc.bool()).sum())
            fix_obs_scale()
            x, v = parts()
            trace.append(((x.mean(dim=1) - c0)[:, 0]).clone())
        tr = torch.stack(trace)                         # (T, E) displacement along x
        d = tr[-1]
        g = 9.81
        mu_eff = (args.slide_v ** 2) / (2 * g * d.clamp_min(1e-5))
        # stop time: first step where speed along x < 2 cm/s
        sp = (tr[1:] - tr[:-1]) / float(inner.step_dt)
        stopped = (sp < 0.02).float()
        t_stop = torch.where(stopped.any(dim=0), stopped.argmax(dim=0).float() * float(inner.step_dt),
                             torch.full_like(d, float("nan")))
        creep = (tr[-1] - tr[-21]) / (20 * float(inner.step_dt))
        res["slide"] = {
            "v0": args.slide_v,
            "dist_mm_median": round(float(d.median()) * 1e3, 2),
            "dist_mm_min_max": [round(float(d.min()) * 1e3, 2), round(float(d.max()) * 1e3, 2)],
            "mu_eff_median": round(float(mu_eff.median()), 3),
            "mu_eff_from_stop_time_median": round(float((args.slide_v / (g * t_stop)).nanmedian()), 3),
            "creep_last_20_steps_mmps": round(float(creep.median()) * 1e3, 3),
            "gap_after_min_mm": round(float((x[..., 2] - table_top()).min()) * 1e3, 2),
            "env_resets_during_settle_slide": resets_passive,
        }
        inner._replay_target_lab_order = None
        print("[thin] slide " + json.dumps(res["slide"]), flush=True)

        # --- 3. policy ----------------------------------------------------------------------------
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=0.0,
        )
        det = PierceDetector(model, n, dev, robot_spec=inner.robot_spec)
        names = det.names
        F = len(names)
        res["fingers"] = names
        res["finger_radius_mm"] = {k: round(v * 1e3, 1) for k, v in det.radius.items()}

        obs, _ = env.reset()
        fix_obs_scale()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=dev))
        fix_obs_scale()
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        pierce_steps = torch.zeros(n, F, dtype=torch.long, device=dev)
        run_len = torch.zeros(n, F, dtype=torch.long, device=dev)
        longest = torch.zeros(n, F, dtype=torch.long, device=dev)
        onsets = torch.zeros(n, F, dtype=torch.long, device=dev)
        below_steps = torch.zeros(n, dtype=torch.long, device=dev)   # any particle centre under table top
        deep_below_steps = torch.zeros(n, dtype=torch.long, device=dev)  # ... by more than 5 mm
        min_gap = torch.full((n,), 1.0, device=dev)
        # Sheet THROUGH the table: a particle centre more than 3 mm under the top while inside the
        # table footprint (0.475 x 0.4 box, 5 mm in from the edge). A sheet hanging over the edge
        # is below the top too but is not a penetration.
        thru_steps = torch.zeros(n, dtype=torch.long, device=dev)
        thru_depth = torch.zeros(n, device=dev)
        tc = inner.table.data.root_pos_w[:, :2]
        length = torch.zeros(n, dtype=torch.long, device=dev)
        outcome = [""] * n
        step = 0
        t0 = time.time()
        while step < args.max_steps and not bool(done.all()):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            fix_obs_scale()
            ended = term.bool() | trunc.bool()
            live = ~done & ~ended
            hit, _a, _b = det(NewtonManager._state_0)
            hit = hit & live[:, None]
            new = hit & (run_len == 0)
            onsets += new.long()
            run_len = torch.where(hit, run_len + 1, torch.zeros_like(run_len))
            longest = torch.maximum(longest, run_len)
            pierce_steps += hit.long()
            x, _v = parts()
            gz = (x[..., 2] - table_top()).nan_to_num(nan=1.0).min(dim=1).values
            below_steps += ((gz < 0) & live).long()
            deep_below_steps += ((gz < -0.005) & live).long()
            min_gap = torch.where(live, torch.minimum(min_gap, gz), min_gap)
            dxy = (x[..., :2] - tc[:, None, :]).abs()
            inside = (dxy[..., 0] < 0.2325) & (dxy[..., 1] < 0.195)
            pen = (table_top() - x[..., 2]).nan_to_num(nan=0.0)          # depth under the top
            pen_in = torch.where(inside & (pen > 0.003), pen, torch.zeros_like(pen)).max(dim=1).values
            thru_steps += ((pen_in > 0) & live).long()
            thru_depth = torch.where(live, torch.maximum(thru_depth, pen_in), thru_depth)
            length += (~done).long()
            fin = ended & ~done
            if bool(fin.any()):
                reasons = inner._termination_reasons
                for e in fin.nonzero(as_tuple=True)[0].tolist():
                    fired = [k for k in OUTCOMES if k in reasons and bool(reasons[k][e])]
                    outcome[e] = fired[0] if fired else ("timeout" if bool(trunc[e]) else "?")
                done |= fin
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            step += 1
        policy_wall = time.time() - t0
        out = np.array(outcome)
        fin_mask = out != ""
        pierced = (pierce_steps.sum(dim=1) > 0).cpu().numpy()
        longest_any = longest.max(dim=1).values.cpu().numpy()
        below = (below_steps > 0).cpu().numpy()

        def split(mask):
            m = mask & fin_mask
            return {k: int(((out == k) & m).sum()) for k in OUTCOMES} | {"n": int(m.sum())}

        res["policy"] = {
            "steps_run": step,
            "wall_s_per_step": round(policy_wall / max(step, 1), 4),
            "episodes_finished": int(fin_mask.sum()),
            "outcome": split(np.ones(n, dtype=bool)),
            "episodes_with_pierce": int((pierced & fin_mask).sum()),
            "episodes_with_pierce_ge_30_steps": int(((longest_any >= 30) & fin_mask).sum()),
            "pierce_finger_steps_per_episode": round(float(pierce_steps.sum()) / max(int(fin_mask.sum()), 1), 2),
            "pierce_onsets_per_episode": round(float(onsets.sum()) / max(int(fin_mask.sum()), 1), 2),
            "pierce_onsets_per_finger": dict(zip(names, onsets.sum(dim=0).tolist())),
            "outcome_if_pierced": split(pierced),
            "outcome_if_clean": split(~pierced),
            "episodes_sheet_below_table_top": int((below & fin_mask).sum()),
            "episodes_sheet_below_table_5mm": int(((deep_below_steps > 0).cpu().numpy() & fin_mask).sum()),
            "episodes_through_table": int(((thru_steps > 0).cpu().numpy() & fin_mask).sum()),
            "through_table_env_steps": int(thru_steps.sum()),
            "through_table_depth_mm_max": round(float(thru_depth.max()) * 1e3, 1),
            "through_table_depth_mm_per_ep": [round(float(v) * 1e3, 1) for v in thru_depth[thru_depth > 0].tolist()][:20],
            "min_particle_gap_mm_p01": round(float(torch.quantile(min_gap, 0.01)) * 1e3, 2),
            "min_particle_gap_mm_median": round(float(min_gap.median()) * 1e3, 2),
        }
        print("[thin] policy " + json.dumps(res["policy"]), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
