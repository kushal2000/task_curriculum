"""Check ``cloth.release_touch=contact`` (the solver-contact touch test) against the origin distance.

    scripts/newton_py -m scripts.analysis.cloth_touch_probe --checkpoint <pth> --out x.json [hydra...]

Runs a policy and, every step and env, records: the contact touch test
(``ClothTaskMixin.hand_cloth_touching``), the number of hand-cloth contact records the cloth solver
holds for the hand, and the old origin test (fingertips + iiwa14_link_7 origins, and every
hand link ``left_.*|iiwa14_link_7``, to the nearest particle). Summary: how often each test says
touching, and how often the old test called the hand clear (> release_clearance) while the solver
had it in contact -- the release loophole.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="checkpoints/config.yaml")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=3)
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
    import warp as wp

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    task = "Isaacsimenvs-Cloth-Direct-v0"

    @hydra_task_config_with_yaml(task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        disable_randomization(env_cfg)
        use_single_object_variant()
        env = gym.make(task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=0.0,
        )
        all_ids, all_names = inner.robot.find_bodies("left_.*|iiwa14_link_7")
        clearance = float(inner.cfg.cloth.release_clearance)
        print(f"[touch] {len(all_ids)} hand links; full_surface_contact={inner.cfg.cloth.full_surface_contact}",
              flush=True)

        obs, _ = env.reset()
        rows = {k: [] for k in ("touch", "n_rec", "min_gap", "d_default", "d_all", "live")}
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        t_touch, t_step = [], []
        for step in range(args.max_steps):
            action = player.get_action(obs["policy"], deterministic=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            torch.cuda.synchronize()
            t_step.append((time.perf_counter() - t1) * 1e3)
            done |= term.bool() | trunc.bool()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            touch = inner.hand_cloth_touching()
            torch.cuda.synchronize()
            t_touch.append((time.perf_counter() - t0) * 1e3)
            d_def = inner.hand_cloth_distance()
            hand = inner.robot.data.body_state_w[:, all_ids, 0:3]
            hand = hand.torch if hasattr(hand, "torch") else hand
            d_all = torch.cdist(hand.float(), inner._particles_w().float()).amin(dim=(1, 2))
            # per-env hand-record count and smallest gap, from the check's own last evaluation
            n_rec = torch.zeros(n, device=dev)
            min_gap = torch.full((n,), float("inf"), device=dev)
            g, e = getattr(inner, "_touch_last_gap", None), getattr(inner, "_touch_last_env", None)
            if g is not None and g.numel():
                e = e.clamp(0, n - 1)
                n_rec.index_add_(0, e, torch.ones_like(g))
                min_gap.scatter_reduce_(0, e, g, reduce="amin")
                inner._touch_last_gap = None
            rows["min_gap"].append(min_gap.cpu())
            rows["touch"].append(touch.cpu())
            rows["n_rec"].append(n_rec.cpu())
            rows["d_default"].append(d_def.cpu())
            rows["d_all"].append(d_all.cpu())
            rows["live"].append((~done).cpu())
            if step % 50 == 0:
                print(f"[touch] step {step}: touching {int(touch.sum())}/{n}, hand records "
                      f"{int(n_rec.sum())}, origin<{clearance*100:.0f}cm default {int((d_def <= clearance).sum())} "
                      f"all-links {int((d_all <= clearance).sum())}, live {int((~done).sum())}", flush=True)
            if bool(done.all()):
                break

        T = {k: torch.stack(v) for k, v in rows.items()}
        live = T["live"]
        touch, d_def, d_all = T["touch"] & live, T["d_default"], T["d_all"]
        rec = (T["n_rec"] > 0) & live
        old_clear = (d_def > clearance) & live
        all_clear = (d_all > clearance) & live
        s = {
            "checkpoint": args.checkpoint, "num_envs": n, "steps": int(live.shape[0]),
            "overrides": hydra_args,
            "live_env_steps": int(live.sum()),
            "touch_env_steps": int(touch.sum()),
            "records_env_steps": int(rec.sum()),
            "touch_without_records": int((touch & ~rec).sum()),
            "old_clear_env_steps": int(old_clear.sum()),
            "old_clear_but_touching": int((old_clear & touch).sum()),
            "alllinks_clear_but_touching": int((all_clear & touch).sum()),
            "touching_origin_dist_mm": {
                "median": float(d_def[touch].median() * 1e3) if touch.any() else None,
                "p90": float(d_def[touch].quantile(0.9) * 1e3) if touch.any() else None,
                "max": float(d_def[touch].max() * 1e3) if touch.any() else None,
            },
            "not_touching_origin_dist_mm_min": float(d_def[live & ~touch].min() * 1e3) if (live & ~touch).any() else None,
            "envs_ever_touching": int(touch.any(0).sum()),
            # does the test track proximity? touching fraction by nearest hand-link origin distance
            "touch_frac_by_alllinks_origin_cm": {
                f"{lo}-{hi}": (float(touch[(d_all >= lo / 100) & (d_all < hi / 100) & live].float().mean())
                               if ((d_all >= lo / 100) & (d_all < hi / 100) & live).any() else None)
                for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 100))},
            "min_gap_when_touching_mm": {
                "median": float(T["min_gap"][touch].median() * 1e3) if touch.any() else None,
                "min": float(T["min_gap"][touch].min() * 1e3) if touch.any() else None},
            "timing_ms": {"touch_check_mean": sum(t_touch[5:]) / max(len(t_touch[5:]), 1),
                          "touch_check_max": max(t_touch[5:], default=0.0),
                          "env_step_mean": sum(t_step[5:]) / max(len(t_step[5:]), 1)},
        }
        print("[touch] summary " + json.dumps(s), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(s, indent=1))
        env.close()

    run()


if __name__ == "__main__":
    main()
