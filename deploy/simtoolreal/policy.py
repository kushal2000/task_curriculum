"""The policy node: object pose + joint state in, joint targets out, at 60 Hz.

    .venv_deploy/bin/python deploy/simtoolreal/policy.py config=deploy/simtoolreal/config.yaml [overrides]
Usually started by run.py (with the robot, viewer and recorder, deploy/nodes/).

Subscribes  object_pose (perception), joint_state (robot)
Publishes   joint_target (to the robot), goal_pose (for the viewer and the recorder)

Every step (1/60 s):
  1. take the newest joint_state and object_pose; stop if either is too old
  2. goal test on the state the last targets produced: reached -> next goal;
     too long on one goal or object off the table -> the episode ends (run.on_episode_end)
  3. observation -> network -> action -> joint targets
  4. publish the targets and the goal

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
from deploy.simtoolreal.goals import first_goal, is_near, next_goal, object_corners, rollout_goals   # noqa: E402
from deploy.utils.config import load_config, object_names, repo_path                                # noqa: E402
from deploy.utils.link import age, open_publisher, open_subscriber, recv_latest, send               # noqa: E402
from deploy.utils.observation import NUM_OBS, build_obs                                              # noqa: E402
from deploy.utils.policy_net import (OBJECT_LOST_BELOW_TABLE, action_to_targets,         # noqa: E402
                                     load_network, wait_for_inputs, wait_step)
from deploy.utils.robot_model import load_robot_model                                                # noqa: E402
from deploy.utils.spec import base_to_env, env_to_base, load_spec                                    # noqa: E402


def main() -> None:
    cfg = load_config(path=HERE / "config.yaml")
    # publish port first: subscribers (recorder, viewer) connect while the network loads,
    # so they get the first message too
    pub = open_publisher(cfg.bus.policy_port)
    spec = load_spec(repo_path(cfg.spec_path))
    model = load_robot_model(spec)
    net = load_network(repo_path(cfg.policy.checkpoint), repo_path(cfg.policy.config), cfg.policy.expl_coef,
                       num_actions=len(spec["joint_names"]))
    rng = np.random.default_rng(cfg.run.seed)

    obj_cfg = cfg.objects[object_names(cfg)[0]]   # with several objects: the first
    scales = np.asarray(obj_cfg.scale, dtype=np.float64)
    corners = object_corners(repo_path(obj_cfg.mesh))

    dt = spec["policy_dt"]
    max_steps = int(round(cfg.run.duration_s / dt))
    timeout_steps = cfg.goals.timeout_steps or spec["episode_length_steps"]
    tolerance = cfg.goals.success_tolerance or spec["success_tolerance_final"]
    if cfg.run.on_episode_end not in ("stop", "next_goal"):
        raise SystemExit(f"[policy] run.on_episode_end={cfg.run.on_episode_end}: only stop | next_goal so far")

    bus = cfg.bus
    sub_state = open_subscriber(bus.host, bus.joint_state_port, "joint_state")
    sub_object = open_subscriber(bus.host, bus.object_pose_port, "object_pose")

    print("[policy] waiting for robot.py ready (hand home) and object_pose ...", flush=True)
    state, obj = wait_for_inputs(sub_state, sub_object)

    # start: targets = where the robot is; a first goal; empty LSTM memory
    prev_targets = state["q"].copy()
    if cfg.goals.from_rollout:
        # a sim rollout's goals, in its order; after the last one the run stops
        fixed_goals, _ = rollout_goals(repo_path(cfg.goals.from_rollout))
        goal_pos, goal_quat = fixed_goals[0]
        print(f"[policy] {len(fixed_goals)} goals from {cfg.goals.from_rollout}", flush=True)
    else:
        fixed_goals = None
        goal_pos, goal_quat = first_goal(spec, rng, corners, cfg.goals.table_clearance_m)
    goal_index = 0
    with torch.no_grad():
        net.get_action(torch.zeros(1, NUM_OBS), deterministic=True)   # warm up (the first call is slow)
    net.reset_rnn()                                                    # ... and forget it

    near_steps = steps_on_goal = goals_reached = 0
    stop_reason = "duration"
    print(f"[policy] running for {cfg.run.duration_s} s; Ctrl-C stops", flush=True)
    t_next = time.perf_counter()
    try:
        for step in range(max_steps):
            # 1. newest inputs
            state = recv_latest(sub_state) or state
            obj = recv_latest(sub_object) or obj
            if state.get("tripped"):
                stop_reason = f"robot safety trip: {state['tripped']}"
                break
            if age(state) > cfg.policy.joint_state_max_age_s:
                stop_reason = f"joint_state {1000 * age(state):.0f} ms old"
                break
            if age(obj) > cfg.perception.max_age_s:
                stop_reason = f"object_pose {1000 * age(obj):.0f} ms old"
                break
            obj_pos, obj_quat = base_to_env(spec, obj["pos"], obj["quat_wxyz"])

            # 2. goal test
            if step > 0:
                near_steps += is_near(spec, obj_pos, obj_quat, goal_pos, goal_quat, tolerance)
                steps_on_goal += 1
                reached = near_steps >= spec["success_steps"]
                ended = None
                if not reached and obj_pos[2] < spec["table_top_z_env"] - OBJECT_LOST_BELOW_TABLE:
                    ended = "object off the table"
                elif not reached and steps_on_goal >= timeout_steps:
                    ended = "goal timeout"
                if reached:
                    goals_reached += 1
                    print(f"[policy] t={step * dt:6.1f} s  goal {goals_reached} reached", flush=True)
                    if goals_reached == cfg.run.max_goals:
                        stop_reason = "max_goals"
                        break
                if ended:
                    print(f"[policy] t={step * dt:6.1f} s  episode ended: {ended}", flush=True)
                    if cfg.run.on_episode_end == "stop":
                        stop_reason = ended
                        break
                if reached or ended:
                    goal_index += 1
                    if fixed_goals is None:
                        goal_pos, goal_quat = next_goal(spec, rng, corners, goal_pos, goal_quat,
                                                        cfg.goals.table_clearance_m)
                    elif goal_index == len(fixed_goals):
                        stop_reason = "all rollout goals done"
                        break
                    else:
                        goal_pos, goal_quat = fixed_goals[goal_index]
                    near_steps = steps_on_goal = 0

            # 3. observation -> network -> targets
            obs = build_obs(spec, model, state["q"], state["dq"], prev_targets, obj_pos, obj_quat,
                            goal_pos, goal_quat, scales)
            with torch.no_grad():
                action = net.get_action(torch.from_numpy(obs)[None], deterministic=True)[0].numpy().astype(np.float64)
            targets = action_to_targets(spec, prev_targets, action)
            prev_targets = targets

            # 4. publish
            now = time.time()
            send(pub, "joint_target", {"seq": step, "t": now, "targets": targets})
            goal_pos_base, goal_quat_base = env_to_base(spec, goal_pos, goal_quat)
            send(pub, "goal_pose", {"seq": step, "t": now, "pos": goal_pos_base, "quat_wxyz": goal_quat_base,
                                    "goal_index": goals_reached})

            t_next = wait_step(t_next, dt, step)   # -> 60 Hz
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    finally:
        pub.close(linger=0)

    print(f"[policy] stopped ({stop_reason}) after {step * dt:.1f} s, {goals_reached} goals reached", flush=True)


if __name__ == "__main__":
    main()
