"""The recorder node: every message on the bus, appended to a .jsonl file as it arrives.

A task's run.py starts it when record.name is set and stops it when the run ends (SIGTERM):
    .venv_deploy/bin/python deploy/simtoolreal/run.py record.name=eraser_try1 [record.camera=true]
It can also run alone (Ctrl-C stops):
    .venv_deploy/bin/python deploy/nodes/recorder.py config=deploy/simtoolreal/config.yaml record.name=free_drive_test

File: <record.dir>/<date>/<date>_<time>_<name>.jsonl, one JSON object per line:
    line 1:  {"topic": "meta", "msg": {"t": <start time>, "config": <the full config of this run>}}
    then:    {"topic": "<topic>", "msg": <the message exactly as sent on the bus>}
Recorded: joint_state, joint_target, goal_pose, sim_object_pose, object_pose, object_poses. Not camera_image (the
viewer's 10 Hz JPEGs).

record.camera=true: every camera frame (camera_frame from perception.py, at the camera's fps) goes to
<same name>.camera.mp4 (H.264, deploy/utils/video.py; record.camera_codec, record.camera_quality), and each
frame's time to the .jsonl:
    {"topic": "camera_frame", "msg": {"seq": <perception's>, "t": <camera frame time>, "frame": <index in the mp4>}}
About 15 MB/min without the camera (joint_state at 200 Hz is most of it).
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import zmq
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for deploy.*
from deploy.utils.config import load_config, repo_path              # noqa: E402
from deploy.utils.link import FRAME_PORT, FRAME_TOPIC, TOPIC_PORTS, open_subscriber, recv_frame   # noqa: E402
from deploy.utils.video import VideoWriter                                                     # noqa: E402


def main() -> None:
    cfg = load_config()
    rc = cfg.record
    if not rc.name:
        raise SystemExit("[recorder] set record.name")
    topics = [t for t in TOPIC_PORTS if t != "camera_image"]

    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = repo_path(rc.dir) / stamp[:10]                       # one folder per day
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}_{rc.name}.jsonl"

    # every message, not only the newest (latest_only=False)
    subs = [open_subscriber(cfg.bus.host, cfg.bus[TOPIC_PORTS[t]], t, latest_only=False) for t in topics]
    poller = zmq.Poller()
    for s in subs:
        poller.register(s, zmq.POLLIN)
    camera = None
    if rc.camera:
        camera = Camera(open_subscriber(cfg.bus.host, cfg.bus[FRAME_PORT], FRAME_TOPIC, latest_only=False),
                        path.with_suffix(".camera.mp4"), cfg)
        poller.register(camera.sub, zmq.POLLIN)

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    parent = os.getppid()

    counts = dict.fromkeys(topics, 0)
    print(f"[recorder] {', '.join(topics + ([FRAME_TOPIC] if camera else []))} -> {path}", flush=True)
    with open(path, "w") as f:
        meta = {"t": time.time(), "config": OmegaConf.to_container(cfg, resolve=True)}
        f.write(json.dumps({"topic": "meta", "msg": meta}) + "\n")
        t_flush = time.time()
        while not stop:
            ready = dict(poller.poll(100))
            for s in subs:
                if s in ready:
                    _drain(s, f, counts)
            if camera and camera.sub in ready:
                camera.drain(f)
            if time.time() - t_flush > 1.0:
                f.flush()
                t_flush = time.time()
                if os.getppid() != parent:   # run.py died without stopping us
                    break
        for s in subs:                       # whatever arrived during the last poll
            _drain(s, f, counts)
        if camera:
            camera.drain(f)

    size = path.stat().st_size / 1e6
    print(f"[recorder] saved {path} ({size:.1f} MB): " + ", ".join(f"{t} {n}" for t, n in counts.items()), flush=True)
    if camera:
        camera.close()


class Camera:
    """camera_frame -> the mp4 (opened at the first frame, its size) + one line per frame in the .jsonl."""

    def __init__(self, sub, path: Path, cfg) -> None:
        self.sub, self.path, self.cfg = sub, path, cfg
        self.video, self.failed = None, False

    def drain(self, f) -> None:
        while (got := recv_frame(self.sub)) is not None:
            header, rgb = got
            if self.failed:
                continue
            try:
                if self.video is None:
                    rc = self.cfg.record
                    self.video = VideoWriter(self.path, header["width"], header["height"],
                                             self.cfg.perception.camera_fps, rc.camera_codec, rc.camera_quality)
                k = self.video.n
                self.video.write(rgb)
            except (BrokenPipeError, ValueError) as e:
                print(f"[recorder] camera video stopped: {e!r} (ffmpeg's error above)", flush=True)
                self.failed = True
                continue
            f.write(json.dumps({"topic": FRAME_TOPIC, "msg": {"seq": header["seq"], "t": header["t"], "frame": k}})
                    + "\n")

    def close(self) -> None:
        if self.video is None:
            print("[recorder] record.camera: no camera_frame arrived (perception.py running, with "
                  "perception.frame_stream=true?); no camera video", flush=True)
            return
        code = self.video.close()
        size = self.path.stat().st_size / 1e6 if self.path.exists() else 0.0
        print(f"[recorder] saved {self.path} ({size:.1f} MB, {self.video.n} frames"
              f"{'' if code == 0 else f', ffmpeg exit code {code}'})", flush=True)


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
