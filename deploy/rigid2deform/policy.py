"""The rigid2deform policy node: perception + joint state in, joint targets out, at 60 Hz.

    .venv_deploy/bin/python deploy/rigid2deform/policy.py config=deploy/rigid2deform/config.yaml [overrides]
Usually started by run.py (with the robot, viewer and recorder, deploy/nodes/). config.yaml's `task`
and the kind of policy.checkpoint (deploy/utils/policy_net.py: keypoints | pointcloud) pick the loop
(TASKS below).

task=fold, a keypoint checkpoint (run_fold_keypoints): a fold-pool policy. The objects' perception (object.name) picks the sheet source
(sheets/: two FoundationPose plates on object_poses, or CoTracker cloth points), which turns each
perception message into the sheet's grid points; fold.py turns those into the observation (the moving
half gives the object, the stationary half the goal) and the success test. Subscribes the source's
topic and joint_state. The run stops when the sheet is folded (the training episode ends there too),
on timeout (fold.timeout_steps), or when the source says its tracking is lost (rigid plates: their
hinge edges drift apart). Publishes joint_target, and goal_pose (where the moving half lands when folded) with fold_error / footprint, the source's numbers,
object_kp / goal_kp (4, 3) (the observed keypoints and their goals, base frame) and a status line for the viewer
and recorder.

task=fold, a point-cloud checkpoint (run_fold_cloud): a distilled point-cloud student. Its perception
(perception/cloud_camera.py) sends point_cloud: the live cloud, at most 1024 points already sampled, base frame,
t = its older camera frame; and in every message the same non-live group, taken once at the start, which
one the checkpoint's mode says (cloud_obs.mode_of):
  label  the reference, the flat cloth alone (clicked corners -> each point's half): ref_xyz (R <= 192, 3)
         base frame, ref_moving (R,) bool (True = on the half that folds over)
  goal   the goal, the sheet folded by hand and photographed before the run (then unfolded where it
         was): goal_xyz (R <= 192, 3) base frame
Each step: build_proprio (103) + cloud_obs.featurize (live + that group, which the network gets only
from step pc_args.ref_step on, 12: training masked it before) -> the network -> action_to_targets.
There is no success test (no tracked sheet): the run stops on timeout (fold.timeout_steps),
run.duration_s, a safety trip, or a stale joint_state / point_cloud (perception.max_age_s). Publishes
joint_target, and goal_pose with a status line for the viewer (its pose: where the moving half lands,
the reference's stationary half's centre or the goal cloud's centre; identity orientation).

The policy works in the env frame (the training env's); the bus carries robot-base-frame poses.
When this node stops, it simply stops publishing: the robot node holds its last target.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.*
from deploy.rigid2deform import cloud_obs                                                            # noqa: E402
from deploy.rigid2deform.fold import fold_geometry, observe, score                                  # noqa: E402
from deploy.rigid2deform.sheets import sheet_source                                                 # noqa: E402
from deploy.utils.config import load_config, repo_path                                              # noqa: E402
from deploy.utils.link import (CLOUD_PORT, CLOUD_TOPIC, age, open_publisher, open_subscriber,      # noqa: E402
                               recv_cloud, recv_latest, send)
from deploy.utils.observation import build_keypoints_obs, build_proprio                              # noqa: E402
from deploy.utils.policy_net import (action_to_targets, checkpoint_kind,                 # noqa: E402
                                     load_policy_net, wait_for_inputs, wait_step)
from deploy.utils.robot_model import load_robot_model                                                # noqa: E402
from deploy.utils.spec import env_to_base, load_spec                                                 # noqa: E402

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def run_fold_keypoints(cfg, spec, model, net, pub) -> None:
    """task=fold: the moving half is the object, the stationary half gives the goal; see the docstring."""
    src = sheet_source(cfg)
    size = src.sheet_size(cfg)   # None: the source measures it (cloth: its clicked corners), from the first message

    dt = spec["policy_dt"]
    max_steps = int(round(cfg.run.duration_s / dt))
    timeout_steps = cfg.fold.timeout_steps or spec["episode_length_steps"]

    bus = cfg.bus
    sub_state = open_subscriber(bus.host, bus.joint_state_port, "joint_state")
    sub_object = open_subscriber(bus.host, bus.object_pose_port, src.TOPIC)

    print(f"[policy] fold: waiting for robot.py ready (hand home) and {src.TOPIC} ...", flush=True)
    joints, msg = wait_for_inputs(sub_state, sub_object)
    if size is None:
        size = src.measured_size(msg)
    geom = fold_geometry(spec, size)
    if cfg.fold.success_tolerance_m is not None:   # the success test's corner distance (training: keypoint_tolerance)
        geom["tolerance"] = float(cfg.fold.success_tolerance_m)
    sheet = src.init(cfg, spec, geom)
    lift = sheet["target_lift"]
    # older perception = lost: the source's own limit if it has one (CoTracker answers in batches)
    max_age = src.max_age_s(sheet) if hasattr(src, "max_age_s") else cfg.perception.max_age_s

    def status(sc: dict, s: dict) -> str:
        """goal_pose's line for the viewer."""
        return (f"fold error (corners, worst): **{100 * sc['fold_error_real']:.1f} cm** "
                f"(success below {100 * geom['tolerance']:.1f} cm) · footprint {sc['footprint']:.2f} "
                f"(at most {geom['max_footprint']:.2f}) · {s['status']}")

    print(f"[policy] fold: {sheet['describe']}, sheet {100 * geom['size']:.1f} cm", flush=True)
    s = src.start(sheet, msg)
    sc = score(geom, observe(geom, s["particles"]), lift)
    print(f"[policy] fold: sheet found ({s['status']}), fold error {100 * sc['fold_error_real']:.1f} cm, "
          f"footprint {sc['footprint']:.2f}", flush=True)

    prev_targets = joints["q"].copy()
    net.warm_up()   # the first call is slow; then forgotten

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
            if age(msg) > max_age:
                stop_reason = f"{src.TOPIC} {1000 * age(msg):.0f} ms old"
                break
            s = src.update(sheet, msg, s)
            if s["stop"]:
                stop_reason = s["stop"]
                break
            o = observe(geom, s["particles"])
            sc = score(geom, o, lift)

            # 2. the state the last targets produced: folded / timeout end the episode
            if step > 0:
                if sc["folded"]:
                    stop_reason = "folded"
                    break
                if step >= timeout_steps:
                    stop_reason = "episode timeout"
                    break

            # 3. observation -> network -> targets
            obs = build_keypoints_obs(spec, model, joints["q"], joints["dq"], prev_targets, o["object_pos"], o["object_quat"],
                                      o["goal_pos"], o["goal_quat_obs"], geom["object_scales"],
                                      keypoints=(o["object_kp"], o["goal_kp"]))
            action = net.act(obs)
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
            # the 4 keypoints the policy observed and their goals (keypoints_rel_goal), base frame
            goal["object_kp"] = env_to_base(spec, o["object_kp"], IDENTITY)[0]
            goal["goal_kp"] = env_to_base(spec, o["goal_kp"], IDENTITY)[0]
            send(pub, "goal_pose", goal)

            t_next = wait_step(t_next, dt, step)   # -> 60 Hz
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    finally:
        pub.close(linger=0)

    print(f"[policy] stopped ({stop_reason}) after {step * dt:.1f} s: fold error {100 * sc['fold_error_real']:.1f} cm "
          f"(success below {100 * geom['tolerance']:.1f}), footprint {sc['footprint']:.2f} "
          f"(success at most {geom['max_footprint']:.2f})", flush=True)


def non_live_group(msg: dict, pc: dict) -> dict:
    """point_cloud's reference (mode label) or goal (mode goal), as cloud_obs.featurize's keyword arguments
    ("group"), with "centre" (where the moving half lands, base frame) and "describe"; SystemExit if the
    message does not carry it."""
    def points(key: str) -> np.ndarray:
        xyz = np.asarray(msg[key], dtype=np.float64).reshape(-1, 3)
        if not 0 < len(xyz) <= cloud_obs.N_REF:
            raise SystemExit(f"[policy] point_cloud {key}: {len(xyz)} points (want 1 to {cloud_obs.N_REF})")
        return xyz

    if cloud_obs.mode_of(pc) == "label":
        if "ref_xyz" not in msg or "ref_moving" not in msg:
            raise SystemExit("[policy] point_cloud carries no reference (ref_xyz, ref_moving), which this "
                             "mode label checkpoint needs: the perception node takes it at its start (click the "
                             "flat cloth's corners); see this file's docstring")
        xyz, moving = points("ref_xyz"), np.asarray(msg["ref_moving"], dtype=bool).reshape(-1)
        if len(moving) != len(xyz):
            raise SystemExit(f"[policy] point_cloud reference: {len(xyz)} points, {len(moving)} labels")
        if moving.all() or not moving.any():
            raise SystemExit(f"[policy] point_cloud reference: all {len(xyz)} points on one half")
        return {"group": {"ref_base": xyz, "ref_moving": moving}, "centre": xyz[~moving].mean(axis=0),
                "name": "reference", "describe": f"reference {len(xyz)} points ({int(moving.sum())} moving, "
                                                 f"{int((~moving).sum())} stationary)"}
    if "goal_xyz" not in msg:
        raise SystemExit("[policy] point_cloud carries no goal (goal_xyz), which this mode goal checkpoint "
                         "needs: the perception node takes it at its start (the sheet folded by hand); see this "
                         "file's docstring")
    xyz = points("goal_xyz")
    return {"group": {"goal_base": xyz}, "centre": xyz.mean(axis=0), "name": "goal",
            "describe": f"goal {len(xyz)} points (the folded sheet)"}


def run_fold_cloud(cfg, spec, model, net, pub) -> None:
    """task=fold with a point-cloud policy; see the docstring."""
    pc = net.pc_args
    ref_step = int(pc["ref_step"])
    dt = spec["policy_dt"]
    max_steps = int(round(cfg.run.duration_s / dt))
    timeout_steps = cfg.fold.timeout_steps or spec["episode_length_steps"]
    max_age = float(cfg.perception.max_age_s)

    bus = cfg.bus
    sub_state = open_subscriber(bus.host, bus.joint_state_port, "joint_state")
    sub_cloud = open_subscriber(bus.host, bus[CLOUD_PORT], CLOUD_TOPIC, latest_only=False)

    print(f"[policy] fold (point cloud): waiting for robot.py ready (hand home) and {CLOUD_TOPIC} ...", flush=True)
    joints, cloud = wait_for_inputs(sub_state, sub_cloud, recv_object=lambda s: recv_cloud(s, newest=True),
                                    what=CLOUD_TOPIC)
    non_live = non_live_group(cloud, pc)   # frozen for the run: taken once, at the start
    ref_seq = cloud["seq"]
    centre, name = non_live["centre"], non_live["name"]   # goal_pose: where the moving half lands
    print(f"[policy] fold (point cloud, mode {cloud_obs.mode_of(pc)}): {non_live['describe']}, the network gets it from "
          f"step {ref_step}; live cloud {len(cloud['xyz'])} points", flush=True)

    prev_targets = joints["q"].copy()
    net.warm_up()

    stop_reason = "duration"
    step = 0
    n_ages, sum_age = 0, 0.0
    print(f"[policy] running for {cfg.run.duration_s} s (episode {timeout_steps * dt:.0f} s; no success test: "
          "stop it yourself); Ctrl-C stops", flush=True)
    t_next = time.perf_counter()
    try:
        for step in range(max_steps):
            # 1. newest inputs
            joints = recv_latest(sub_state) or joints
            cloud = recv_cloud(sub_cloud, newest=True) or cloud
            if joints.get("tripped"):
                stop_reason = f"robot safety trip: {joints['tripped']}"
                break
            if age(joints) > cfg.policy.joint_state_max_age_s:
                stop_reason = f"joint_state {1000 * age(joints):.0f} ms old"
                break
            if age(cloud) > max_age:
                stop_reason = f"{CLOUD_TOPIC} {1000 * age(cloud):.0f} ms old"
                break
            if step >= timeout_steps:
                stop_reason = "episode timeout"
                break

            # 2. observation -> network -> targets
            proprio = build_proprio(spec, model, joints["q"], joints["dq"], prev_targets)
            group = non_live["group"] if step >= ref_step else {}
            points, mask = cloud_obs.featurize(cloud["xyz"], pc, **group)
            action = net.act((proprio, points, mask))
            targets = action_to_targets(spec, prev_targets, action)
            prev_targets = targets

            # 3. publish
            now = time.time()
            n_ages, sum_age = n_ages + 1, sum_age + age(cloud)
            send(pub, "joint_target", {"seq": step, "t": now, "targets": targets})
            send(pub, "goal_pose", {
                "seq": step, "t": now, "goal_index": 0, "pos": centre, "quat_wxyz": IDENTITY,
                "cloud_points": int(mask[:cloud_obs.N_POINTS].sum()), "cloud_seq": cloud["seq"],
                "cloud_age": age(cloud), "reference_in": bool(step >= ref_step), "reference_seq": ref_seq,
                "status": (f"point-cloud policy · live cloud **{int(mask[:cloud_obs.N_POINTS].sum())}** points, "
                           f"{1000 * age(cloud):.0f} ms old · {name} "
                           f"{int(mask[cloud_obs.N_POINTS:].sum()) if step >= ref_step else 0} points "
                           f"({'in' if step >= ref_step else f'from step {ref_step}'}) · no success test")})

            t_next = wait_step(t_next, dt, step)   # -> 60 Hz
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    finally:
        pub.close(linger=0)

    mean_age = f", point_cloud {1000 * sum_age / n_ages:.0f} ms old on average" if n_ages else ""
    print(f"[policy] stopped ({stop_reason}) after {step * dt:.1f} s{mean_age}", flush=True)


# config.yaml `task` -> the loop of each kind of network (policy_net.py) -> fn(cfg, spec, model, net, pub)
TASKS = {"fold": {"keypoints": run_fold_keypoints, "pointcloud": run_fold_cloud}}


def check(cfg) -> None:
    """The config's task, checkpoint and object, before anything starts (run.py calls it too)."""
    if cfg.task not in TASKS:
        raise SystemExit(f"[policy] task={cfg.task}: {' | '.join(TASKS)}")
    kind = checkpoint_kind(repo_path(cfg.policy.checkpoint))   # the network decides the perception
    if kind not in TASKS[cfg.task]:
        raise SystemExit(f"[policy] task={cfg.task} has no loop for a {kind} policy ({cfg.policy.checkpoint})")
    if kind == "keypoints":
        sheet_source(cfg).check(cfg)   # object.name: one implemented perception, the objects it needs
        return
    if CLOUD_PORT not in cfg.bus:
        raise SystemExit(f"[policy] a point-cloud policy needs bus.{CLOUD_PORT} (its perception's port)")
    spec = load_spec(repo_path(cfg.spec_path))
    net = load_policy_net(cfg, num_actions=len(spec["joint_names"]), kinds=("pointcloud",))
    cloud_obs.check_supported(net.pc_args, net.point_dim)   # a cloud cloud_obs.py builds


def main() -> None:
    cfg = load_config(path=HERE / "config.yaml")
    check(cfg)
    # publish port first: subscribers (recorder, viewer) connect while the network loads,
    # so they get the first message too
    pub = open_publisher(cfg.bus.policy_port)
    spec = load_spec(repo_path(cfg.spec_path))
    model = load_robot_model(spec)
    net = load_policy_net(cfg, num_actions=len(spec["joint_names"]), kinds=tuple(TASKS[cfg.task]))
    TASKS[cfg.task][net.kind](cfg, spec, model, net, pub)


if __name__ == "__main__":
    main()
