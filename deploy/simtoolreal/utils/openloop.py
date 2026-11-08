"""The sim rollout that robot.mode=openloop sends: checks/record_lift.py's openloop.npz.

    roll = load_rollout(repo_path(cfg.openloop.path), spec)
    roll["targets"]                      # (T, 27) joint targets, spec joint order, one per policy step
    roll["object_pos"], ["object_quat"]  # (T, 3), (T, 4): the sim object after each step (base frame)
    print(placement_text(roll))          # where to put the real object before the run

Poses are in the robot base frame, of the object mesh's own frame (objects/, the one the viewer
draws). The file stores them in the env frame.
"""

import json
from pathlib import Path

import numpy as np

from .quat import quat_apply
from .spec import env_to_base

START_TOLERANCE_DEG = 0.5   # the rollout's first joint angles vs spec default_q (home)


def load_rollout(path: Path, spec: dict) -> dict:
    """Read and check an openloop.npz. Exits with the reason if it does not fit this spec."""
    if not Path(path).exists():
        raise SystemExit(f"[openloop] no rollout at {path}: record one with checks/record_lift.py")
    d = np.load(path)
    names = [str(n) for n in d["joint_names"]]
    if names != list(spec["joint_names"]):
        raise SystemExit(f"[openloop] {path}: joint names differ from the spec's")
    dt = float(d["dt"])
    if abs(dt - spec["policy_dt"]) > 1e-6:
        raise SystemExit(f"[openloop] {path}: dt {dt} differs from the spec's policy_dt {spec['policy_dt']}")
    # robot.py starts it only with the robot at home, so the rollout must start there too
    err = np.degrees(np.abs(d["q"][0] - spec["default_q"]))
    if err.max() > START_TOLERANCE_DEG:
        i = int(np.argmax(err))
        raise SystemExit(f"[openloop] {path}: starts {err[i]:.2f} deg from home at {names[i]}")

    def base(pos, quat):
        p, q = env_to_base(spec, np.asarray(pos, np.float64), np.asarray(quat, np.float64))
        return p, q

    obj = [base(p, q) for p, q in zip(d["object_mesh_pos"], d["object_quat"])]
    return {
        "path": Path(path), "dt": dt, "targets": d["targets"].astype(np.float64),
        "q_sim": d["q"].astype(np.float64),
        "object_pos": np.array([o[0] for o in obj]), "object_quat": np.array([o[1] for o in obj]),
        "goal": base(d["goal_mesh_pos"], d["goal_quat"]),
        "start": base(d["start_mesh_pos"], d["start_quat"]),
        "start_centre": base(d["start_pos"], d["start_quat"])[0],
        "box_size": d["box_size"].astype(np.float64),
        "meta": json.loads(str(d["meta"])),
    }


def placement_text(roll: dict) -> str:
    """Where the real object has to be, as the sim had it at the start."""
    c = roll["start_centre"]
    m = roll["meta"]
    long_axis = quat_apply(roll["start"][1], np.array([1.0, 0.0, 0.0]))   # the mesh's long (x) side
    yaw = np.degrees(np.arctan2(long_axis[1], long_axis[0]))
    sx, sy, sz = 100 * roll["box_size"]
    first = m.get("chosen_first_hit_step")
    hit = f", goal first reached at {first * roll['dt']:.1f} s" if first is not None and first >= 0 else ""
    return (f"[openloop] {roll['path']}: {len(roll['targets'])} steps ({len(roll['targets']) * roll['dt']:.1f} s){hit}.\n"
            f"[openloop] Put the {sx:.1f} x {sy:.1f} x {sz:.1f} cm eraser flat on its {sx:.1f} x {sz:.1f} face, "
            f"long side {yaw:+.0f} deg from the robot base x axis,\n"
            f"[openloop]   centre at base x = {100 * c[0]:+.1f} cm, y = {100 * c[1]:+.1f} cm "
            f"(the sim's; lift {100 * m['lift_m']:.0f} cm). The viewer draws it in blue.")
