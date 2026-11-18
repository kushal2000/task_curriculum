"""Do adjacent fingers (ring / pinky) collide, on PhysX and on Newton, in the same poses?

    PhysX:  PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
                scripts/flexiv_wuji/probe_finger_contact.py --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --headless
    Newton: scripts/newton_py scripts/flexiv_wuji/probe_finger_contact.py \
                --task Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0 'env.assets.handle_head_types=[hammer]'

Randomization off, hand joint friction zeroed, the passive spring off (only the PD acts), the hand
held at a target for --settle_steps. Per pose, the max |q - target| [mrad] on the ring and pinky
joints: 0 means nothing pushes them (no contact force), > 0 means contact.
  A: flexion joints at rest + 0.5 rad, abduction at the default (the spring probes' pose)
  B: as A, with ring and pinky abduction closed toward each other to the joint limits
  C: as A, with ring and pinky abduction spread apart to the joint limits
PhysX also reports the robot's physxArticulation:enabledSelfCollisions as authored on the stage.
"""

from __future__ import annotations

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=400)
parser.add_argument("--out", default=None)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
try:
    app = AppLauncher(args_cli).app
except ImportError as exc:  # Newton: kit-less
    app = None
    print(f"[finger_contact] kit-less: {exc}", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402


def _t(x):
    v = getattr(x, "torch", None)
    if v is None:
        return x
    return v() if callable(v) else v


def _self_collision_flags() -> dict:
    """physxArticulation:enabledSelfCollisions on env_0's robot (PhysX / Kit only)."""
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd
    except ImportError:
        return {"note": "no Kit stage"}
    stage = get_current_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    out = {}
    for prim in Usd.PrimRange(root):
        attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if attr and attr.HasAuthoredValue():
            out[str(prim.GetPath())] = bool(attr.Get())
    return out


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.assets.num_assets_per_type = 1
    disable_randomization(env_cfg)
    use_single_object_variant()
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    robot, device = inner.robot, inner.device
    names = list(robot.data.joint_names)
    hand = [int(i) for i in inner._hand_joint_ids]
    env.reset()
    rep = {"task": args_cli.task, "robot_self_collision": bool(getattr(env_cfg, "robot_self_collision", None)),
           "authored_self_collision": _self_collision_flags()}
    print("[finger_contact] self-collision", json.dumps(rep), flush=True)

    inner._spring_ks = None  # PD only
    robot.set_joint_effort_target(torch.zeros_like(_t(robot.data.joint_pos)))
    if hasattr(robot, "root_physx_view"):
        view = robot.root_physx_view
        fr = view.get_dof_friction_properties().clone()
        fr[:] = 0.0
        view.set_dof_friction_properties(fr, indices=torch.arange(inner.num_envs, dtype=torch.int32))
    else:
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.zeros(inner.num_envs, len(hand), device=device), joint_ids=hand)

    lo, hi = (_t(robot.data.soft_joint_pos_limits)[0, :, k] for k in (0, 1))
    spec = inner.robot_spec
    ref = torch.tensor([float(spec.hand_spring_ref.get(n, 0.0)) for n in names], device=device)
    flex = [i for i in hand if ("flex" in names[i] or "pip" in names[i] or "dip" in names[i]
                                 or names[i].endswith("_mcp") or names[i].endswith("_ip"))]
    rp = [i for i in hand if ("ring" in names[i] or "pinky" in names[i])]
    ring_abd, pinky_abd = names.index("r_ring_finger_mcp_abd"), names.index("r_pinky_mcp_abd")
    base = _t(robot.data.default_joint_pos).clone()
    base[:, flex] = (ref[flex] + 0.5).unsqueeze(0)
    base[:, hand] = torch.clamp(base[:, hand], lo[hand] + 0.02, hi[hand] - 0.02)
    dt = float(inner.cfg.sim.dt)

    def settle(a):
        robot.write_joint_state_to_sim(a, torch.zeros_like(a))
        for _ in range(args_cli.settle_steps):
            robot.set_joint_position_target(a)
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            inner.scene.update(dt)
        dq = (_t(robot.data.joint_pos) - a)[:, rp].abs().max(0).values
        return {names[i]: round(1e3 * float(dq[k]), 2) for k, i in enumerate(rp)}

    poses = {"A_default_abd": base.clone()}
    b = base.clone()  # which sign closes the gap depends on the joint axes: try both, report both
    b[:, ring_abd], b[:, pinky_abd] = lo[ring_abd], hi[pinky_abd]
    poses["B_ring_lo_pinky_hi"] = b
    c = base.clone()
    c[:, ring_abd], c[:, pinky_abd] = hi[ring_abd], lo[pinky_abd]
    poses["C_ring_hi_pinky_lo"] = c
    rep["abd_limits_rad"] = {"ring": [float(lo[ring_abd]), float(hi[ring_abd])],
                             "pinky": [float(lo[pinky_abd]), float(hi[pinky_abd])]}
    rep["poses"] = {}
    for k, a in poses.items():
        rep["poses"][k] = settle(a)
        print(f"[finger_contact] {k}: max |q - target| mrad", json.dumps(rep["poses"][k]), flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(rep, f, indent=1)
    import os

    os._exit(0)


if __name__ == "__main__":
    run()
