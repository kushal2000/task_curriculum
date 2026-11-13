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

task=fold (run_fold): a fold-pool policy. Subscribes object_poses (both plates) instead; every step
the moving plate gives the object and the stationary one the goal (utils/fold.py); the run stops when
the sheet is folded (the training episode ends there too), on timeout, or when the two plates' hinge
edges drift apart (fold.max_hinge_gap_m: a tracker flipped or lost). goal_pose is where the moving
plate's mesh lands when folded, plus fold_error / footprint / hinge_gap for the viewer and recorder.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from utils.config import REPO, load_config, object_names, repo_path                         # noqa: E402
from utils.fold import FoldGeometry, PlateFrames, mesh_to_sheet                  # noqa: E402
from utils.goals import first_goal, is_near, next_goal, object_corners, rollout_goals   # noqa: E402
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
    # One CPU thread. The net is tiny (0.6 ms on 8 threads, 2 ms on 1, idle), but torch's default
    # thread pool spins on every core: with FoundationPose, the recorder and the viewer busy on the
    # same PC, it took 21 ms median, 50 ms p99, and the loop fell to 30 Hz. 1 thread under that load: 2.3 ms.
    torch.set_num_threads(1)
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
    if cfg.task == "fold":
        run_fold(cfg, spec, model, net, pub)
        return
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


def run_fold(cfg, spec, model, net, pub) -> None:
    """task=fold: the moving plate is the object, the stationary one gives the goal; see the docstring."""
    names = object_names(cfg)
    if len(names) != 2:
        raise SystemExit(f"[policy] task=fold: object.name must be [moving, stationary], got {names}")
    fc = cfg.fold
    geom = FoldGeometry(spec, fc.sheet_size, fc.hinge_z)
    m2s = mesh_to_sheet(fc.hinge_axis_mesh)
    frames = PlateFrames(geom, m2s, fc.max_hinge_gap_m)

    dt = spec["policy_dt"]
    max_steps = int(round(cfg.run.duration_s / dt))
    timeout_steps = cfg.goals.timeout_steps or spec["episode_length_steps"]
    lost_z = spec["table_top_z_env"] - OBJECT_LOST_BELOW_TABLE

    bus = cfg.bus
    sub_state = open_subscriber(bus.host, bus.joint_state_port, "joint_state")
    sub_object = open_subscriber(bus.host, bus.object_pose_port, "object_poses")

    def plates(msg) -> list[tuple]:
        """[moving, stationary] as (centre, mesh quat wxyz), env frame."""
        out, sent = [], [str(n) for n in msg["names"]]   # the bus may deliver a numpy array
        for name in names:
            if name not in sent:
                raise SystemExit(f"[policy] object_poses has {sent}, no {name}: perception's object.name differs")
            i = sent.index(name)
            out.append(base_to_env(spec, np.asarray(msg["pos"][i]), np.asarray(msg["quat_wxyz"][i])))
        return out

    print(f"[policy] fold: moving {names[0]}, stationary {names[1]}, sheet {100 * fc.sheet_size:.0f} cm; "
          "waiting for robot.py ready (hand home) and object_poses ...", flush=True)
    state, obj = wait_for_inputs(sub_state, sub_object)
    f = frames.start(*plates(obj))
    score = geom.score(geom.observe(f["moving"], f["stationary"]), f["stationary"])
    print(f"[policy] fold: plates side by side, hinge gap {1000 * f['hinge_gap']:.1f} mm, fold error "
          f"{100 * score['fold_error_real']:.1f} cm, footprint {score['footprint']:.2f}", flush=True)

    prev_targets = state["q"].copy()
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
            state = recv_latest(sub_state) or state
            obj = recv_latest(sub_object) or obj
            if state.get("tripped"):
                stop_reason = f"robot safety trip: {state['tripped']}"
                break
            if age(state) > cfg.policy.joint_state_max_age_s:
                stop_reason = f"joint_state {1000 * age(state):.0f} ms old"
                break
            if age(obj) > cfg.perception.max_age_s:
                stop_reason = f"object_poses {1000 * age(obj):.0f} ms old"
                break
            f = frames.update(*plates(obj))
            if f["hinge_gap"] > fc.max_hinge_gap_m:
                stop_reason = (f"the plates' hinge edges {100 * f['hinge_gap']:.1f} cm apart "
                               f"(fold.max_hinge_gap_m {100 * fc.max_hinge_gap_m:.1f}): tracking lost or flipped")
                break
            o = geom.observe(f["moving"], f["stationary"])
            score = geom.score(o, f["stationary"])

            # 2. the state the last targets produced: folded / fell / timeout end the episode
            if step > 0:
                if score["folded"]:
                    stop_reason = "folded"
                    break
                if min(f["moving"][0][2], f["stationary"][0][2]) < lost_z:
                    stop_reason = "a plate off the table"
                    break
                if step >= timeout_steps:
                    stop_reason = "episode timeout"
                    break

            # 3. observation -> network -> targets
            obs = build_obs(spec, model, state["q"], state["dq"], prev_targets, o["object_pos"], o["object_quat"],
                            o["goal_pos"], o["goal_quat_obs"], geom.object_scales,
                            keypoints=(o["object_kp"], o["goal_kp"]))
            with torch.no_grad():
                action = net.get_action(torch.from_numpy(obs)[None], deterministic=True)[0].numpy().astype(np.float64)
            targets = action_to_targets(spec, prev_targets, action)
            prev_targets = targets

            # 4. publish
            now = time.time()
            send(pub, "joint_target", {"seq": step, "t": now, "targets": targets})
            goal_pos_base, goal_quat_base = env_to_base(spec, *geom.goal_plate_pose(f["stationary"], m2s))
            send(pub, "goal_pose", {"seq": step, "t": now, "pos": goal_pos_base, "quat_wxyz": goal_quat_base,
                                    "goal_index": 0, "fold_error": score["fold_error_real"],
                                    "fold_error_train": score["fold_error"], "footprint": score["footprint"],
                                    "hinge_gap": f["hinge_gap"]})

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

    print(f"[policy] stopped ({stop_reason}) after {step * dt:.1f} s: fold error {100 * score['fold_error_real']:.1f} cm "
          f"(success below {100 * geom.tolerance:.1f}), footprint {score['footprint']:.2f} "
          f"(success at most {geom.max_footprint:.2f})", flush=True)


if __name__ == "__main__":
    main()
