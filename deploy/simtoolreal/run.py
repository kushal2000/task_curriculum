"""The one entry point: robot.py in the chosen robot.mode, plus the viewer and the recorder, plus the
policy in policy mode.

    PY=.venv_deploy/bin/python; R=deploy/simtoolreal/run.py; ARM="robot.arm=<ROBOT_SN> robot.hand=<HAND_SN>"
    $PY $R robot.mode=home       $ARM    # 1. move to home (or free_drive: move it there by hand)
    (start perception.py)                # 2.
    $PY $R $ARM                          # 3. robot.mode=policy (the default): the policy run
        record.name=eraser_try1          record it (recordings/<date>_<time>_eraser_try1.jsonl)
        record.camera=true               ... with the camera images
        run.viewer=false                 no viewer
        run.duration_s=60, ...           any other config key

| robot.mode | starts                                   | the robot                                  |
|------------|------------------------------------------|--------------------------------------------|
| policy     | viewer, recorder, robot, then policy     | refuses to start away from home; follows   |
| home       | viewer, recorder, robot                  | moves to home, then robot.py exits          |
| free_drive | viewer, recorder, robot                  | movable by hand until Ctrl-C               |
| openloop   | viewer, recorder, robot                  | refuses to start away from home; sends a   |
|            |                                          | sim rollout's targets (openloop.path)      |
(recorder only with record.name; viewer unless run.viewer=false.) Viewer and recorder start first,
so they listen from the beginning; the policy waits for the robot's first joint_state.

While it runs:
  - the policy ends (duration, max goals, goal timeout, object off the table, safety trip): the
    robot holds where it is;
  - robot.py exits (home reached, or the Flexiv E-stop / enabling button / a fault): the policy
    stops too;
  in both cases the viewer and the recorder stay up. Ctrl-C ends everything: policy, robot (holds,
  then stops: the hand goes limp), recorder, viewer.
"""

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from robot import MODES                                       # noqa: E402
from utils.config import load_config, repo_path               # noqa: E402
from utils.link import start_node, stop_node                  # noqa: E402

# asked before real hardware moves
CONFIRM_WARNINGS = {"home": "the arm moves in a straight joint-space line, NOT around the table",
                    "free_drive": "the arm goes soft: the hand must be the Flexiv tool payload, or it sags",
                    "policy": "the policy drives the robot",
                    "openloop": "a sim rollout's joint targets drive the robot, blind: the object must be "
                                "where the sim had it"}


def main() -> None:
    cfg = load_config()
    mode = cfg.robot.mode
    if mode not in MODES:
        raise SystemExit(f"[run] robot.mode={mode}: {' | '.join(MODES)}")
    if mode == "openloop":                              # where to put the object, before 'go'
        from utils.openloop import load_rollout, placement_text
        from utils.spec import load_spec
        print(placement_text(load_rollout(repo_path(cfg.openloop.path), load_spec(repo_path(cfg.spec_path)))),
              flush=True)
    if cfg.robot.arm != "mujoco" and cfg.run.confirm_hardware:
        answer = input(f"[run] {mode}, arm {cfg.robot.arm}, hand {cfg.robot.hand}: {CONFIRM_WARNINGS[mode]}. "
                       "Type 'go': ")
        if answer.strip() != "go":
            raise SystemExit("[run] not confirmed")

    nodes = {}
    if cfg.run.viewer:
        nodes["viewer"] = start_node("viewer.py")
    if cfg.record.name:
        nodes["recorder"] = start_node("recorder.py")
    nodes["robot"] = start_node("robot.py")
    if mode == "policy":
        nodes["policy"] = start_node("policy.py")

    watched = [n for n in ("robot", "policy") if n in nodes]
    exited = set()
    try:
        while True:
            for name in watched:
                if name not in exited and nodes[name].poll() is not None:
                    exited.add(name)
                    print(f"[run] {name}.py exited (code {nodes[name].returncode})", flush=True)
                    if name == "robot":
                        stop_node(nodes.get("policy"))   # it would stop by itself (no joint_state), sooner
                    if exited == set(watched):
                        still = [n for n in ("viewer", "recorder") if n in nodes]
                        print(f"[run] {' and '.join(watched)} stopped; {' and '.join(still) or 'nothing'} "
                              "still running. Ctrl-C to quit.", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("[run] stopping ...", flush=True)
        stop_node(nodes.get("policy"))                  # stops sending targets
        stop_node(nodes.get("robot"))                   # holds, then stops arm and hand
        if "recorder" in nodes:
            time.sleep(0.3)                             # the last messages
            stop_node(nodes["recorder"], signal.SIGTERM)
        stop_node(nodes.get("viewer"))
        print("[run] done", flush=True)


if __name__ == "__main__":
    main()
