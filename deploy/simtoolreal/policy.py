"""The policy node: object pose + joint state in, joint targets out, at 60 Hz.

    .venv_deploy/bin/python deploy/simtoolreal/policy.py [config overrides]
Usually started by run.py (with the robot, viewer and recorder).

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

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from utils.config import REPO, load_config, repo_path                            # noqa: E402
from utils.goals import first_goal, is_near, next_goal, object_corners           # noqa: E402
from utils.link import age, open_publisher, open_subscriber, recv_latest, send   # noqa: E402
from utils.observation import NUM_OBS, build_obs                                 # noqa: E402
from utils.robot_model import load_robot_model                                   # noqa: E402
from utils.spec import base_to_env, env_to_base, load_spec                       # noqa: E402

OBJECT_LOST_BELOW_TABLE = 0.05   # m: object this far below the table top = fell off


def load_network(checkpoint: Path, config: Path, expl_coef: float, num_actions: int):
    """The trained policy: an LSTM (rl_games SAPG checkpoint), one robot, on the CPU.

    Use: net.get_action(obs (1, 134) tensor, deterministic=True) -> (1, 27) actions in [-1, 1], once
    per step, in order (the LSTM remembers past steps); net.reset_rnn() clears that memory.

    Built through the training repo's isaacsimenvs/eval/player.py (rl_games reads
    pretrained_policy/config.yaml). SAPG trains several policy "blocks" that differ only in one extra
    input, the exploration coefficient; expl_coef 0.0 is the one trained on the task reward only.
    """
    # load player.py by path: importing the isaacsimenvs package would pull in Isaac Lab
    path = REPO / "isaacsimenvs" / "eval" / "player.py"
    module_spec = importlib.util.spec_from_file_location("_play_player", path)
    player = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(player)
    return player.PretrainedPlayer(config_path=config, checkpoint_path=checkpoint, num_envs=1, device="cpu",
                                   num_observations=NUM_OBS, num_actions=num_actions, expl_coef=expl_coef)


def action_to_targets(spec: dict, prev_targets: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Policy action (27 numbers in [-1, 1]) -> joint position targets (rad), as the training env
    (action_utils.apply_action_pipeline, no delay):

    Arm (first 7): a velocity command, target = prev + dof_speed_scale * dt * action.
    Hand (last 20): an absolute position, -1 -> lower limit, +1 -> upper limit.
    Both then smoothed, new = 0.1 * target + 0.9 * prev, and clamped to the joint limits.
    """
    n = spec["num_arm_joints"]
    lo, hi = spec["joint_lower"], spec["joint_upper"]
    arm = np.clip(prev_targets[:n] + spec["dof_speed_scale"] * spec["policy_dt"] * action[:n], lo[:n], hi[:n])
    hand = lo[n:] + 0.5 * (action[n:] + 1.0) * (hi[n:] - lo[n:])
    a_arm, a_hand = spec["arm_moving_average"], spec["hand_moving_average"]
    arm = a_arm * arm + (1.0 - a_arm) * prev_targets[:n]
    hand = a_hand * hand + (1.0 - a_hand) * prev_targets[n:]
    return np.clip(np.concatenate([arm, hand]), lo, hi)


def wait_for_inputs(sub_state, sub_object, timeout_s: float = 30.0) -> tuple[dict, dict]:
    """Block until robot.py is ready (hand home, forwarding targets) and an object_pose has arrived.
    Starting earlier takes prev_targets from a hand still moving home: the first targets then jump."""
    state = obj = None
    t_end = time.time() + timeout_s
    while state is None or obj is None:
        if time.time() > t_end:
            missing = [name for name, m in (("ready joint_state", state), ("object_pose", obj)) if m is None]
            raise SystemExit(f"[policy] no {' and no '.join(missing)} after {timeout_s:.0f} s: is the publisher running?")
        new = recv_latest(sub_state)
        if new is not None and new.get("ready"):
            state = new
        obj = recv_latest(sub_object) or obj
        time.sleep(0.01)
    return state, obj


def main() -> None:
    cfg = load_config()
    # publish port first: subscribers (recorder, viewer) connect while the network loads,
    # so they get the first message too
    pub = open_publisher(cfg.bus.policy_port)
    spec = load_spec(repo_path(cfg.spec_path))
    model = load_robot_model(spec)
    net = load_network(repo_path(cfg.policy.checkpoint), repo_path(cfg.policy.config), cfg.policy.expl_coef,
                       num_actions=len(spec["joint_names"]))
    rng = np.random.default_rng(cfg.run.seed)

    obj_cfg = cfg.objects[cfg.object.name]
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
    goal_pos, goal_quat = first_goal(spec, rng, corners)
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
                    goal_pos, goal_quat = next_goal(spec, rng, corners, goal_pos, goal_quat)
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

            # wait for the next tick -> 60Hz
            t_next += dt
            late = time.perf_counter() - t_next
            if late < 0:
                time.sleep(-late)
            elif late > dt:   # more than a whole step behind: don't try to catch up
                print(f"[policy] step {step} {1000 * late:.0f} ms late", flush=True)
                t_next = time.perf_counter()
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    finally:
        pub.close(linger=0)

    print(f"[policy] stopped ({stop_reason}) after {step * dt:.1f} s, {goals_reached} goals reached", flush=True)


if __name__ == "__main__":
    main()
