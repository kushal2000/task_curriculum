"""Free-space joint tracking probe: the same open-loop action sequence on PhysX or Newton/MJWarp.

    # PhysX (Isaac Lab 2.3):
    $ISAACSIM_PY experiments/nov25_better_rigidcloth/freespace/probe.py --task Isaacsimenvs-RigidCloth-Direct-v0 \
        --headless --out outputs/freespace_compare/physx.npz
    # Newton (Isaac Lab 3):
    .venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/freespace/probe.py \
        --task Isaacsimenvs-RigidClothNewton-Direct-v0 --out outputs/freespace_compare/newton.npz [env.hand_sysid=true]

Each of the 8 envs runs one signal (``signals.py``) through ``env.step`` (the policy's action space,
canonical joint order), from the deterministic reset pose. Logged per policy step and per physics step
(by wrapping ``inner._apply_action``): joint pos / vel, controller targets, fingertip pads. Everything
is stored by joint NAME (canonical order), so the two backends' arrays line up.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RECIPE = "experiments/nov24_play_pc_rl/play_pc_rl_A.yaml"
FIXED_OVERRIDES = [
    "env.fold_pool.shapes=[box2-mid]",
    "env.fold_pool.one_shape_per_env=false",
    "env.fold_pool.size_range=[0.15,0.15]",
    "env.episode_length_s=30.0",
]


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=600)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0], *FIXED_OVERRIDES, *hydra_args]
    try:
        app = AppLauncher(args).app
    except ImportError as exc:
        app = None
        print(f"[probe] kit-less: {exc}", flush=True)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.tasks.cloth.mdp.robot import to_torch
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    import signals

    @hydra_task_config_with_yaml(args.task, "", experiment=RECIPE)
    def run(env_cfg, _agent) -> None:
        env_cfg.scene.num_envs = signals.NUM_ENVS
        env_cfg.seed = 0
        disable_randomization(env_cfg)
        info = {
            "task": args.task,
            "hand_sysid": bool(env_cfg.hand_sysid),
            "sim_dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "newton": str(getattr(env_cfg, "newton", None)),
            "action_cfg": str(env_cfg.action),
            "dr_action": str(env_cfg.domain_randomization.action),
            "dr_robot": str(env_cfg.domain_randomization.robot),
            "table_top_z": float(env_cfg.reset.table_reset_z) + float(env_cfg.cloth.table_half_thickness),
            "overrides": sys.argv[1:],
        }
        print("[probe] cfg", info, flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        dev = inner.device
        lay = inner.robot_layout
        robot = inner.robot
        canon = list(lay.spec.joint_names_canonical)
        lab = list(lay.joint_names)
        l2c = lay.perm_lab_to_canon  # canonical[i] = lab[l2c[i]]

        # Drive parameters as the simulator holds them (canonical order), for the report.
        def _get(name):
            v = getattr(robot.data, name, None)
            return None if v is None else to_torch(v)[0][l2c].detach().cpu().numpy()

        drive = {k: _get(k) for k in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff",
                                       "joint_effort_limits", "joint_vel_limits", "joint_dynamic_friction_coeff",
                                       "joint_viscous_friction_coeff")}
        drive = {k: v for k, v in drive.items() if v is not None}
        for k, v in drive.items():
            print(f"[probe] {k}: {np.array2string(v, precision=5, max_line_width=250)}", flush=True)

        log_phys = {"q": [], "qd": [], "tgt": []}
        orig_apply = inner._apply_action

        def apply_wrapped():
            orig_apply()
            log_phys["q"].append(to_torch(robot.data.joint_pos)[:, l2c].clone())
            log_phys["qd"].append(to_torch(robot.data.joint_vel)[:, l2c].clone())
            log_phys["tgt"].append(inner.controller.targets[:, l2c].clone())

        env.reset()
        q0 = to_torch(robot.data.joint_pos)[:, l2c].clone()
        lower = lay.joint_pos_limits[0, :, 0][l2c]
        upper = lay.joint_pos_limits[0, :, 1][l2c]
        default = lay.default_joint_pos[0][l2c]
        print("[probe] reset q0 - default max |d|", float((q0 - default).abs().max()), flush=True)
        acts = signals.build(canon, default.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), args.steps)
        acts_t = torch.as_tensor(acts, device=dev, dtype=torch.float32)

        inner._apply_action = apply_wrapped
        rec = {k: [] for k in ("q", "qd", "tgt", "tips", "done")}
        for t in range(args.steps):
            _obs, _r, term, trunc, _ = env.step(acts_t[t])
            rec["q"].append(to_torch(robot.data.joint_pos)[:, l2c].clone())
            rec["qd"].append(to_torch(robot.data.joint_vel)[:, l2c].clone())
            rec["tgt"].append(inner.controller.targets[:, l2c].clone())
            st = inner.sim_state()
            rec["tips"].append(inner.task.fingertip_pads_w(st) - inner.scene.env_origins.unsqueeze(1))
            rec["done"].append(term | trunc)
        out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
        out.update({f"phys_{k}": torch.stack(v).cpu().numpy() for k, v in log_phys.items()})
        out.update({f"drive_{k}": v for k, v in drive.items()})
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(
            args.out, actions=acts, joint_names=np.array(canon), lower=lower.cpu().numpy(), upper=upper.cpu().numpy(),
            default=default.cpu().numpy(), q0=q0.cpu().numpy(), info=np.array(repr(info)),
            signal_names=np.array(signals.SIGNAL_NAMES), policy_dt=float(inner.step_dt),
            phys_dt=float(inner.physics_dt), **out,
        )
        print(f"[probe] wrote {args.out}; any done: {bool(out['done'].any())}; min tip z "
              f"{float(out['tips'][..., 2].min()):.4f} (table top {info['table_top_z']:.4f})", flush=True)
        env.close()

    run()
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
