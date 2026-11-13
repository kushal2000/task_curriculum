"""A task's run.py: the shared nodes (deploy/nodes/) + the task's policy node, on one config.

    run(config_path, policy_script)     # deploy/<task>/run.py calls this, nothing else

robot.py in the chosen robot.mode, plus the viewer and the recorder, plus the task's policy in policy
mode. Every node gets this process's command line and config=<the task's config>, so all of them
read the same config.

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
  - the policy ends (duration, goals, timeout, object off the table, safety trip): the robot holds
    where it is;
  - robot.py exits (home reached, or the Flexiv E-stop / enabling button / a fault): the policy
    stops too;
  in both cases the viewer and the recorder stay up. Ctrl-C ends everything: policy, robot (holds,
  then stops: the hand goes limp), recorder, viewer.
"""

import signal
import time
from pathlib import Path

from deploy.nodes.robot import MODES

from .config import REPO, load_config, repo_path
from .link import start_node, stop_node

# asked before real hardware moves
CONFIRM_WARNINGS = {"home": "the arm moves in a straight joint-space line, NOT around the table",
                    "free_drive": "the arm goes soft: the hand must be the Flexiv tool payload, or it sags",
                    "policy": "the policy drives the robot",
                    "openloop": "a sim rollout's joint targets drive the robot, blind: the object must be "
                                "where the sim had it"}


def run(config_path: Path, policy_script: Path, check=None) -> None:
    """Load config_path (+ the command line), then check(cfg) if given (a task's own tests, e.g. its
    task name), then start and watch the nodes until Ctrl-C."""
    cfg = load_config(path=config_path)
    if check is not None:
        check(cfg)
    config_arg = f"config={Path(config_path).resolve().relative_to(REPO)}"
    mode = cfg.robot.mode
    if mode not in MODES:
        raise SystemExit(f"[run] robot.mode={mode}: {' | '.join(MODES)}")
    if mode == "openloop":                              # where to put the object, before 'go'
        if "openloop" not in cfg:
            raise SystemExit(f"[run] robot.mode=openloop: {cfg.config} has no openloop section")
        from .openloop import load_rollout, placement_text
        from .spec import load_spec
        print(placement_text(load_rollout(repo_path(cfg.openloop.path), load_spec(repo_path(cfg.spec_path)))),
              flush=True)
    if cfg.robot.arm != "mujoco" and cfg.run.confirm_hardware:
        answer = input(f"[run] {mode}, arm {cfg.robot.arm}, hand {cfg.robot.hand}: {CONFIRM_WARNINGS[mode]}. "
                       "Type 'go': ")
        if answer.strip() != "go":
            raise SystemExit("[run] not confirmed")

    nodes = {}
    if cfg.run.viewer:
        nodes["viewer"] = start_node("viewer.py", config_arg)
    if cfg.record.name:
        nodes["recorder"] = start_node("recorder.py", config_arg)
    nodes["robot"] = start_node("robot.py", config_arg)
    if mode == "policy":
        nodes["policy"] = start_node(Path(policy_script).resolve(), config_arg)

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
