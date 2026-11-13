"""The rigid2deform policy node: plate poses + joint state in, joint targets out, at 60 Hz.

    .venv_deploy/bin/python deploy/rigid2deform/policy.py config=deploy/rigid2deform/config.yaml [overrides]
Usually started by run.py (with the robot, viewer and recorder, deploy/nodes/). config.yaml's `task`
picks the loop (TASKS below).

task=fold (run_fold): a fold-pool policy. The objects' perception (object.name) picks the sheet source
(sheets/: two FoundationPose plates on object_poses, or CoTracker cloth points), which turns each
perception message into the sheet's grid points; fold.py turns those into the observation (the moving
half gives the object, the stationary half the goal) and the success test. Subscribes the source's
topic and joint_state. The run stops when the sheet is folded (the training episode ends there too),
on timeout (fold.timeout_steps), when part of the sheet falls off the table, or when the source says
its tracking is lost (rigid plates: their hinge edges drift apart). Publishes joint_target, and
goal_pose (where the moving half lands when folded) with fold_error / footprint, the source's numbers
and a status line for the viewer and recorder.

The policy works in the env frame (the training env's); the bus carries robot-base-frame poses.
When this node stops, it simply stops publishing: the robot node holds its last target.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.*
from deploy.rigid2deform.fold import fold_geometry, observe, score                                  # noqa: E402
from deploy.rigid2deform.sheets import sheet_source                                                 # noqa: E402
from deploy.utils.config import load_config, repo_path                                              # noqa: E402
from deploy.utils.link import age, open_publisher, open_subscriber, recv_latest, send               # noqa: E402
from deploy.utils.observation import NUM_OBS, build_obs                                              # noqa: E402
from deploy.utils.policy_net import (OBJECT_LOST_BELOW_TABLE, action_to_targets,         # noqa: E402
                                     load_network, wait_for_inputs, wait_step)
from deploy.utils.robot_model import load_robot_model                                                # noqa: E402
from deploy.utils.spec import env_to_base, load_spec                                                 # noqa: E402


def run_fold(cfg, spec, model, net, pub) -> None:
    """task=fold: the moving half is the object, the stationary half gives the goal; see the docstring."""
    src = sheet_source(cfg)
    geom = fold_geometry(spec, src.sheet_size(cfg))
    sheet = src.init(cfg, spec, geom)
    lift = sheet["target_lift"]

    dt = spec["policy_dt"]
    max_steps = int(round(cfg.run.duration_s / dt))
    timeout_steps = cfg.fold.timeout_steps or spec["episode_length_steps"]
    lost_z = spec["table_top_z_env"] - OBJECT_LOST_BELOW_TABLE

    bus = cfg.bus
    sub_state = open_subscriber(bus.host, bus.joint_state_port, "joint_state")
    sub_object = open_subscriber(bus.host, bus.object_pose_port, src.TOPIC)

    def status(sc: dict, s: dict) -> str:
        """goal_pose's line for the viewer."""
        return (f"fold error (corners, worst): **{100 * sc['fold_error_real']:.1f} cm** "
                f"(success below {100 * geom['tolerance']:.1f} cm) · footprint {sc['footprint']:.2f} "
                f"(at most {geom['max_footprint']:.2f}) · {s['status']}")

    print(f"[policy] fold: {sheet['describe']}, sheet {100 * geom['size']:.0f} cm; "
          f"waiting for robot.py ready (hand home) and {src.TOPIC} ...", flush=True)
    joints, msg = wait_for_inputs(sub_state, sub_object)
    s = src.start(sheet, msg)
    sc = score(geom, observe(geom, s["particles"]), lift)
    print(f"[policy] fold: sheet found ({s['status']}), fold error {100 * sc['fold_error_real']:.1f} cm, "
          f"footprint {sc['footprint']:.2f}", flush=True)

    prev_targets = joints["q"].copy()
    with torch.no_grad():
        net.get_action(torch.zeros(1, NUM_OBS), deterministic=True)   # warm up (the first call is slow)
    net.reset_rnn()                                                    # ... and forget it

    stop_reason = "duration"
    step = 0
    print(f"[policy] running for {cfg.run.duration_s} s (episode {timeout_steps * dt:.0f} s); Ctrl-C stops", flush=True)
    t_next = time.perf_counter()
    try:
        for step in range(max_steps):
            # 1. newest inputs
            joints = recv_latest(sub_state) or joints
            msg = recv_latest(sub_object) or msg
            if joints.get("tripped"):
                stop_reason = f"robot safety trip: {joints['tripped']}"
                break
            if age(joints) > cfg.policy.joint_state_max_age_s:
                stop_reason = f"joint_state {1000 * age(joints):.0f} ms old"
                break
            if age(msg) > cfg.perception.max_age_s:
                stop_reason = f"{src.TOPIC} {1000 * age(msg):.0f} ms old"
                break
            s = src.update(sheet, msg, s)
            if s["stop"]:
                stop_reason = s["stop"]
                break
            o = observe(geom, s["particles"])
            sc = score(geom, o, lift)

            # 2. the state the last targets produced: folded / fell / timeout end the episode
            if step > 0:
                if sc["folded"]:
                    stop_reason = "folded"
                    break
                if s["particles"][:, 2].min() < lost_z:
                    stop_reason = "part of the sheet off the table"
                    break
                if step >= timeout_steps:
                    stop_reason = "episode timeout"
                    break

            # 3. observation -> network -> targets
            obs = build_obs(spec, model, joints["q"], joints["dq"], prev_targets, o["object_pos"], o["object_quat"],
                            o["goal_pos"], o["goal_quat_obs"], geom["object_scales"],
                            keypoints=(o["object_kp"], o["goal_kp"]))
            with torch.no_grad():
                action = net.get_action(torch.from_numpy(obs)[None], deterministic=True)[0].numpy().astype(np.float64)
            targets = action_to_targets(spec, prev_targets, action)
            prev_targets = targets

            # 4. publish
            now = time.time()
            send(pub, "joint_target", {"seq": step, "t": now, "targets": targets})
            goal = {"seq": step, "t": now, "goal_index": 0, "fold_error": sc["fold_error_real"],
                    "fold_error_train": sc["fold_error"], "footprint": sc["footprint"],
                    "status": status(sc, s), **s["info"]}
            if s["goal_mesh_pose"] is not None:
                goal["pos"], goal["quat_wxyz"] = env_to_base(spec, *s["goal_mesh_pose"])
            else:   # no mesh for the goal: the fold target's centre, the folded orientation
                goal["pos"], goal["quat_wxyz"] = env_to_base(spec, o["goal_pos"], o["goal_quat_true"])
            send(pub, "goal_pose", goal)

            t_next = wait_step(t_next, dt, step)   # -> 60 Hz
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    finally:
        pub.close(linger=0)

    print(f"[policy] stopped ({stop_reason}) after {step * dt:.1f} s: fold error {100 * sc['fold_error_real']:.1f} cm "
          f"(success below {100 * geom['tolerance']:.1f}), footprint {sc['footprint']:.2f} "
          f"(success at most {geom['max_footprint']:.2f})", flush=True)


TASKS = {"fold": run_fold}   # config.yaml `task` -> its loop (cfg, spec, model, net, pub)


def check(cfg) -> None:
    """The config's task and object, before anything starts (run.py calls it too)."""
    if cfg.task not in TASKS:
        raise SystemExit(f"[policy] task={cfg.task}: {' | '.join(TASKS)}")
    sheet_source(cfg).check(cfg)   # object.name: one implemented perception, the objects it needs


def main() -> None:
    cfg = load_config(path=HERE / "config.yaml")
    check(cfg)
    # publish port first: subscribers (recorder, viewer) connect while the network loads,
    # so they get the first message too
    pub = open_publisher(cfg.bus.policy_port)
    spec = load_spec(repo_path(cfg.spec_path))
    model = load_robot_model(spec)
    net = load_network(repo_path(cfg.policy.checkpoint), repo_path(cfg.policy.config), cfg.policy.expl_coef,
                       num_actions=len(spec["joint_names"]))
    TASKS[cfg.task](cfg, spec, model, net, pub)


if __name__ == "__main__":
    main()
