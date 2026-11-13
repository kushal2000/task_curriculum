"""How fast does the VBD sheet fall onto the table after reset? Zero actions, one env.

    scripts/newton_py -m scripts.analysis.cloth_drop_probe --out drop.json [hydra overrides...]

Records, per env step, the mean and min particle height above the table's collision surface and
the mean vertical velocity, and compares the fall to free fall from the spawn height
(``cloth.start_height - table_half_thickness`` above the surface). Used to check whether a cloth
preset (density, damping, contact radii) slows the spawn drop so the policy can act on a sheet that
is still in the air.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=90)
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
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    task = "Isaacsimenvs-Cloth-Direct-v0"

    @hydra_task_config_with_yaml(task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        disable_randomization(env_cfg)
        use_single_object_variant()
        env = gym.make(task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        c = inner.cfg.cloth
        dt = float(inner.step_dt)
        table_half = float(c.table_half_thickness)
        top = (inner._table_z_per_env + table_half + inner.scene.env_origins[:, 2])[0].item()

        def nodal():
            q = inner._cloth.data.nodal_pos_w
            return (q.torch if hasattr(q, "torch") else q).float()[0]

        env.reset()
        zero = torch.zeros((n, inner.cfg.action_space), device=dev)
        rows = []
        prev = nodal()[:, 2].mean().item()
        rows.append({"step": 0, "t": 0.0, "mean_h": prev - top, "min_h": nodal()[:, 2].min().item() - top, "vz": 0.0})
        for k in range(1, args.steps + 1):
            env.step(zero)
            z = nodal()[:, 2]
            m = z.mean().item()
            rows.append({"step": k, "t": k * dt, "mean_h": m - top, "min_h": z.min().item() - top,
                         "vz": (m - prev) / dt})
            prev = m
        h0 = rows[0]["mean_h"]
        g = 9.81
        land = next((r for r in rows if r["min_h"] <= float(c.thickness) * 0.75 + 1e-3), None)
        res = {
            "dt": dt, "drop_height_m": h0, "free_fall_time_s": (2 * h0 / g) ** 0.5,
            "landing_step": land["step"] if land else None, "landing_time_s": land["t"] if land else None,
            "density": float(c.density), "thickness": float(c.thickness),
            "self_contact_radius": float(c.self_contact_radius), "rows": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print(f"[drop] dt {dt:.4f} s, drop {h0 * 100:.1f} cm, free fall {res['free_fall_time_s']:.3f} s, "
              f"landed at step {res['landing_step']} ({res['landing_time_s']})", flush=True)
        for r in rows[:: max(1, len(rows) // 18)]:
            print(f"[drop] step {r['step']:3d} t {r['t']:.3f} mean_h {r['mean_h'] * 100:6.2f} cm "
                  f"min_h {r['min_h'] * 100:6.2f} cm vz {r['vz']:6.3f} m/s", flush=True)
        env.close()

    run()


if __name__ == "__main__":
    main()
