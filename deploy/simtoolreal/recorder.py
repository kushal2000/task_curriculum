"""The recorder node: every message on the bus, appended to a .jsonl file as it arrives.

run.py starts it when record.name is set and stops it when the run ends (SIGTERM):
    .venv_deploy/bin/python deploy/simtoolreal/run.py record.name=eraser_try1 [record.camera=true]
It can also run alone (Ctrl-C stops):
    .venv_deploy/bin/python deploy/simtoolreal/recorder.py record.name=free_drive_test

File: <record.dir>/<date>_<time>_<name>.jsonl, one JSON object per line:
    line 1:  {"topic": "meta", "msg": {"t": <start time>, "config": <the full config of this run>}}
    then:    {"topic": "<topic>", "msg": <the message exactly as sent on the bus>}
Recorded: joint_state, joint_target, goal_pose, object_pose, and camera_image with record.camera.
About 15 MB/min without the camera (joint_state at 200 Hz is most of it), +24 MB/min with it.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import zmq
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from utils.config import load_config, repo_path              # noqa: E402
from utils.link import TOPIC_PORTS, open_subscriber          # noqa: E402


def main() -> None:
    cfg = load_config()
    rc = cfg.record
    if not rc.name:
        raise SystemExit("[recorder] set record.name")
    topics = [t for t in TOPIC_PORTS if t != "camera_image" or rc.camera]

    out_dir = repo_path(rc.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{rc.name}.jsonl"

    # every message, not only the newest (latest_only=False)
    subs = [open_subscriber(cfg.bus.host, cfg.bus[TOPIC_PORTS[t]], t, latest_only=False) for t in topics]
    poller = zmq.Poller()
    for s in subs:
        poller.register(s, zmq.POLLIN)

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    parent = os.getppid()

    counts = dict.fromkeys(topics, 0)
    print(f"[recorder] {', '.join(topics)} -> {path}", flush=True)
    with open(path, "w") as f:
        meta = {"t": time.time(), "config": OmegaConf.to_container(cfg, resolve=True)}
        f.write(json.dumps({"topic": "meta", "msg": meta}) + "\n")
        t_flush = time.time()
        while not stop:
            ready = dict(poller.poll(100))
            for s in subs:
                if s in ready:
                    _drain(s, f, counts)
            if time.time() - t_flush > 1.0:
                f.flush()
                t_flush = time.time()
                if os.getppid() != parent:   # run.py died without stopping us
                    break
        for s in subs:                       # whatever arrived during the last poll
            _drain(s, f, counts)

    size = path.stat().st_size / 1e6
    print(f"[recorder] saved {path} ({size:.1f} MB): " + ", ".join(f"{t} {n}" for t, n in counts.items()), flush=True)


def _drain(sock, f, counts: dict) -> None:
    """Write every queued message of one socket, without parsing it."""
    while True:
        try:
            s = sock.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return
        topic, body = s.split(" ", 1)
        f.write(f'{{"topic": "{topic}", "msg": {body}}}\n')
        counts[topic] += 1


if __name__ == "__main__":
    main()
