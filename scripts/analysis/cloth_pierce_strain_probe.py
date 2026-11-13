"""How often does a finger go THROUGH the VBD cloth, and what does it cost the fold?

    scripts/newton_py -m scripts.analysis.cloth_pierce_probe --checkpoint <pth> --out x.json [hydra...]

The cloth feels rigid bodies only through its 49 PARTICLES (radius 8 mm, spacing 16.7 mm): Newton's
VBD has particle-vs-shape contact and nothing else -- no triangle or edge sees a finger. A fingertip
(~14.5 mm across) that spreads two particles apart is past the sheet, after which the stretch
springs close the sheet around it. This counts that event directly.

**Pierce** = the axis of a finger link (distal or middle phalanx; the principal axis of its collision
mesh, end to end) crosses one of its own env's cloth triangles. In ordinary contact the particle
radius keeps the finger SURFACE 8 mm off the cloth mid-surface and the axis sits another ~7 mm
inside the finger, so a crossing is never a grazing touch -- it is a finger on both sides of the sheet.

One episode per env, as `episodes.py` (same reset, zero-action tick, player, success settings).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
from isaacsimenvs.tasks.cloth.utils.pierce import PierceDetector

OUTCOMES = ("fold_held", "fall", "hand_far", "nonfinite", "timeout")


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--sapg_expl_coef", type=float, default=0.0)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=900)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        AppLauncher(args)
    except ImportError:
        pass

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    task = "Isaacsimenvs-Cloth-Direct-v0"

    @hydra_task_config_with_yaml(task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = 0.01
        if not args.randomize:
            disable_randomization(env_cfg)
        use_single_object_variant()
        import time

        t_setup = time.perf_counter()
        env = gym.make(task, cfg=env_cfg)
        setup_s = time.perf_counter() - t_setup
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
        )

        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        det = PierceDetector(model, n, dev)
        names, seg, radius = det.names, det.seg_np, det.radius

        print(f"[pierce] fingers {names}; axis length mm "
              + ", ".join(f"{k}:{np.linalg.norm(seg[i, 1] - seg[i, 0]) * 1e3:.1f}" for i, k in enumerate(names))
              + "; radius mm " + ", ".join(f"{k}:{v * 1e3:.1f}" for k, v in radius.items()), flush=True)

        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=dev))

        F = len(names)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        pierce_steps = torch.zeros(n, F, dtype=torch.long, device=dev)
        run_len = torch.zeros(n, F, dtype=torch.long, device=dev)
        longest = torch.zeros(n, F, dtype=torch.long, device=dev)
        onsets = torch.zeros(n, F, dtype=torch.long, device=dev)
        first_pierce_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        outcome = [""] * n
        length = torch.zeros(n, dtype=torch.long, device=dev)
        onset_speed = []  # finger speed [m/s] at each pierce onset
        onset_height = []  # lowest point of the finger axis above the table top [m], at onset
        pierced_height = []  # the same, at every pierced finger-step
        table_half = float(inner.cfg.cloth.table_half_thickness)
        prev_pos = None
        # Grip: does the hand move the cloth at all? Highest cloth node above the table top and the
        # largest node displacement from where it lay after reset, max over the first episode.
        def nodal():
            q = inner._cloth.data.nodal_pos_w
            return (q.torch if hasattr(q, "torch") else q).float()
        # The sheet spawns above the table and drops onto it: start measuring once it has settled.
        GRIP_WARM = 30
        q0 = nodal().clone()
        # Grid edges + rest lengths from the flat spawn sheet (before it touches anything).
        qs = q0
        N = qs.shape[1]
        tri = model.tri_indices.numpy().reshape(-1, 3)
        tri = tri[(tri < N).all(axis=1)]  # env 0's sheet; same topology in every env
        E = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
        ei = torch.as_tensor(E[:, 0], device=dev, dtype=torch.long); ej = torch.as_tensor(E[:, 1], device=dev, dtype=torch.long)
        keep = torch.ones(len(E), dtype=torch.bool); h = (qs[0, ei] - qs[0, ej]).norm(dim=-1).min()
        L0 = (qs[:, ei] - qs[:, ej]).norm(dim=-1).clamp(min=1e-6)
        print(f'[strain] {qs.shape[1]} nodes, {int(keep.sum())} grid edges, spacing {float(h) * 1e3:.2f} mm, z-spread {float(qs[0, :, 2].max() - qs[0, :, 2].min()) * 1e3:.2f} mm', flush=True)
        top0 = (inner._table_z_per_env + table_half) + inner.scene.env_origins[:, 2]
        cloth_lift = torch.zeros(n, device=dev)
        cloth_move = torch.zeros(n, device=dev)
        # Stretch: grid-edge strain vs the settled (flat) sheet at GRIP_WARM.
        strain_max = torch.zeros(n, device=dev)
        strain_mean_max = torch.zeros(n, device=dev)
        step_smax = []
        step = 0
        step_s = []  # env.step wall time [s], synced
        while step < args.max_steps and not bool(done.all()):
            action = player.get_action(obs["policy"], deterministic=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            torch.cuda.synchronize()
            step_s.append(time.perf_counter() - t0)
            ended = (term.bool() | trunc.bool())

            hit, a, b = det(NewtonManager._state_0)              # (E, F)
            pos = 0.5 * (a + b)
            # The env resets inside `step`, so this frame of an ended env is the NEXT episode's
            # start: score only envs still on their first episode and not ending now.
            live = ~done & ~ended
            hit = hit & live[:, None]
            speed = ((pos - prev_pos).norm(dim=-1) / float(inner.step_dt)) if prev_pos is not None else torch.zeros_like(pos[..., 0])
            new = hit & (run_len == 0)
            # Height of the finger axis's lowest end above this env's table top (the collision
            # surface sits table_half_thickness above the table body's z).
            top = (inner._table_z_per_env + table_half)[:, None]
            low = torch.minimum(a[..., 2], b[..., 2]) - inner.scene.env_origins[:, 2:3] - top
            if bool(new.any()):
                onset_speed += speed[new].tolist()
                onset_height += low[new].tolist()
            if bool(hit.any()):
                pierced_height += low[hit].tolist()
            onsets += new.long()
            run_len = torch.where(hit, run_len + 1, torch.zeros_like(run_len))
            longest = torch.maximum(longest, run_len)
            pierce_steps += hit.long()
            first_pierce_step = torch.where((first_pierce_step < 0) & hit.any(dim=1) & live,
                                            torch.full_like(first_pierce_step, step), first_pierce_step)
            prev_pos = pos
            q = nodal()
            if step == GRIP_WARM:
                q0 = q.clone()
            first = ~done & ~ended & (step >= GRIP_WARM)
            lift = q[..., 2].amax(dim=1) - top0
            move = (q - q0).norm(dim=-1).amax(dim=1)
            cloth_lift = torch.where(first, torch.maximum(cloth_lift, lift), cloth_lift)
            cloth_move = torch.where(first, torch.maximum(cloth_move, move), cloth_move)
            if L0 is not None:
                s = (q[:, ei] - q[:, ej]).norm(dim=-1) / L0 - 1.0
                smax, smean = s.amax(dim=1), s.mean(dim=1)
                strain_max = torch.where(first, torch.maximum(strain_max, smax), strain_max)
                strain_mean_max = torch.where(first, torch.maximum(strain_mean_max, smean), strain_mean_max)
                if bool(first.any()):
                    step_smax += smax[first].tolist()
            length += (~done).long()

            fin = ended & ~done
            if bool(fin.any()):
                reasons = inner._termination_reasons
                for e in fin.nonzero(as_tuple=True)[0].tolist():
                    fired = [k for k in OUTCOMES if k in reasons and bool(reasons[k][e])]
                    outcome[e] = fired[0] if fired else ("timeout" if bool(trunc[e]) else "?")
                done |= fin
                player.reset_rnn(fin.nonzero(as_tuple=True)[0])
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            step += 1

        pierced = (pierce_steps.sum(dim=1) > 0).cpu().numpy()
        longest_any = longest.max(dim=1).values.cpu().numpy()
        out = np.array(outcome)
        fin_mask = out != ""

        def split(mask):
            m = mask & fin_mask
            return {k: int(((out == k) & m).sum()) for k in OUTCOMES} | {"n": int(m.sum())}

        res = {
            "checkpoint": args.checkpoint, "randomize": args.randomize, "seed": args.seed,
            "num_envs": n, "overrides": hydra_args, "steps_run": step,
            "fingers": names,
            "timing": {
                "env_setup_s": round(setup_s, 1),
                "env_step_ms_mean": round(float(np.mean(step_s[20:] or step_s)) * 1e3, 2),
                "env_step_ms_median": round(float(np.median(step_s[20:] or step_s)) * 1e3, 2),
                "warmup_steps_excluded": 20,
            },
            "episodes_finished": int(fin_mask.sum()),
            "episodes_with_pierce": int((pierced & fin_mask).sum()),
            "episodes_with_pierce_ge_30_steps": int(((longest_any >= 30) & fin_mask).sum()),
            "env_steps_pierced_frac": round(float((pierce_steps.sum(dim=1) > 0).float().mean()), 4),
            "pierce_env_steps": int((pierce_steps > 0).any(dim=1).sum()),
            "pierce_steps_per_finger": dict(zip(names, pierce_steps.sum(dim=0).tolist())),
            "pierce_onsets_per_finger": dict(zip(names, onsets.sum(dim=0).tolist())),
            "longest_pierce_steps": longest_any.tolist(),
            "first_pierce_step": first_pierce_step.cpu().tolist(),
            "episode_length": length.cpu().tolist(),
            "outcome": outcome,
            "outcome_if_pierced": split(pierced),
            "outcome_if_clean": split(~pierced),
            "onset_height_mm": {
                "n": len(onset_height),
                "median": round(float(np.median(onset_height)) * 1e3, 1) if onset_height else None,
                "frac_below_12mm": round(float(np.mean(np.array(onset_height) < 0.012)), 3) if onset_height else None,
            },
            "pierced_steps_height_mm": {
                "n": len(pierced_height),
                "frac_below_12mm": round(float(np.mean(np.array(pierced_height) < 0.012)), 3) if pierced_height else None,
                "frac_above_30mm": round(float(np.mean(np.array(pierced_height) > 0.030)), 3) if pierced_height else None,
            },
            "grip": {
                "settle_steps": GRIP_WARM,
                "cloth_lift_cm_median": round(float(cloth_lift.median()) * 100, 2),
                "cloth_lift_cm_p90": round(float(cloth_lift.quantile(0.9)) * 100, 2),
                "frac_lifted_over_3cm": round(float((cloth_lift > 0.03).float().mean()), 3),
                "cloth_move_cm_median": round(float(cloth_move.median()) * 100, 2),
                "frac_moved_over_3cm": round(float((cloth_move > 0.03).float().mean()), 3),
            },
            "onset_speed_mps": {
                "n": len(onset_speed),
                "median": round(float(np.median(onset_speed)), 3) if onset_speed else None,
                "p90": round(float(np.percentile(onset_speed, 90)), 3) if onset_speed else None,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sm = strain_max.cpu().numpy(); smm = strain_mean_max.cpu().numpy()
        res['strain'] = {
            'episode_max_edge_strain_median': round(float(np.median(sm)), 4),
            'episode_max_edge_strain_p90': round(float(np.percentile(sm, 90)), 4),
            'frac_episodes_max_strain_over_20pct': round(float((sm > 0.2).mean()), 4),
            'episode_max_mean_strain_median': round(float(np.median(smm)), 4),
            'per_step_max_strain_median': round(float(np.median(step_smax)), 4) if step_smax else None,
            'per_step_max_strain_p90': round(float(np.percentile(step_smax, 90)), 4) if step_smax else None,
        }
        print('[strain] ' + json.dumps(res['strain']), flush=True)
        args.out.write_text(json.dumps(res, indent=2))
        print("[pierce] " + json.dumps({k: res[k] for k in (
            "episodes_finished", "episodes_with_pierce", "episodes_with_pierce_ge_30_steps", "grip",
            "timing", "outcome_if_pierced", "outcome_if_clean", "pierce_onsets_per_finger", "onset_speed_mps",
            "onset_height_mm", "pierced_steps_height_mm")}),
            flush=True)
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
