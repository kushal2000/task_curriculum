"""Export what the real-robot joint-reach loop (deploy/joint_reach) needs from the sim, as JSON.

Everything is read from a live env, in the policy's canonical joint order: joint limits (they normalize
the observation and clamp the targets), the action pipeline's constants, the nominal PD gains and
effort limits, the default pose, and the observation layout. The deploy code then holds no hand-copied
numbers.

    .venv_isaacsim/bin/python -m isaacsimenvs.eval.joint_reach_export_spec --headless \
        --out deploy/joint_reach/spec_flexiv_wuji_hl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaacsimenvs.eval.joint_reach_eval import TASK


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(TASK, "")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.scene.num_envs = 1
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        env.reset()
        perm = inner._perm_lab_to_canon
        data = inner.robot.data

        def canon(t) -> list[float]:
            t = t.torch if hasattr(t, "torch") else t
            return [float(v) for v in t[0, perm]]

        a, rr = inner.cfg.action, inner.cfg.reach_reward
        origin = inner.scene.env_origins[0]
        spec = {
            "task": TASK,
            "robot_urdf": str(env_cfg.assets.robot_urdf),
            "joint_names": list(inner.robot_spec.joint_names_canonical),
            "num_arm_joints": 7,
            "joint_lower": [float(v) for v in inner._joint_lower_canon],
            "joint_upper": [float(v) for v in inner._joint_upper_canon],
            "default_q": [float(v) for v in inner.default_goal()],
            "stiffness": canon(inner._base_k_lab),
            "damping": canon(inner._base_d_lab),
            "effort_limit": canon(data.joint_effort_limits),
            "policy_dt": float(inner.step_dt),
            "sim_dt": float(inner.cfg.sim.dt),
            "decimation": int(inner.cfg.decimation),
            "dof_speed_scale": float(a.dof_speed_scale),
            "arm_moving_average": float(a.arm_moving_average),
            "hand_moving_average": float(a.hand_moving_average),
            "obs_clip": float(inner.cfg.obs.clamp_abs_observations),
            "obs_layout": [
                "joint_pos normalized to [-1, 1] by joint_lower/upper",
                "joint_vel [rad/s]",
                "previous joint targets [rad]",
                "goal normalized to [-1, 1] by joint_lower/upper",
            ],
            "arm_tol_deg": float(rr.arm_tol_deg),
            "hand_tol_deg": float(rr.hand_tol_deg),
            "robot_gravity": "off in sim (the real arm's controller compensates it)",
            "robot_root_pos_env": [float(v) for v in data.root_pos_w[0] - origin],
            "robot_root_quat_wxyz": [float(v) for v in data.root_quat_w[0]],
            "table_size": [float(v) for v in env_cfg.goal.table_box_size],
            "table_pos_env": [float(v) for v in inner.table.data.root_pos_w[0] - origin],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"[jr_spec] wrote {out}", flush=True)
        with torch.no_grad():
            print(json.dumps({k: spec[k] for k in ("stiffness", "damping", "effort_limit")}), flush=True)

    run()
    import os

    os._exit(0)  # Kit shutdown hangs on these nodes


if __name__ == "__main__":
    main()
