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

point_cloud (a point-cloud policy's rigid2deform/perception/cloud_camera.py; configs with bus.point_cloud_port) is always recorded,
to <same name>.cloud.npz, written when the recorder stops (~12 kB per cloud: ~20 MB/min at 30 Hz):
    xyz (T, max n, 3) float32, base frame, NaN past each cloud's n; n (T,); t (T,); seq (T,);
    t_cams (T, cameras); depth (raw | foundationstereo, the last cloud's); and the non-live group the clouds
    carry (the first cloud's; the node never changes it): ref_xyz (R, 3), ref_moving (R,), ref_corners (4, 3)
    (a mode label policy) or goal_xyz (R, 3), goal_corners (4, 3) (mode goal)
and one line per cloud to the .jsonl (the header as sent without that group, "index" = its row in the npz):
    {"topic": "point_cloud", "msg": {"seq": ..., "t": ..., "n": ..., "depth": ..., "t_cams": [...], "index": k}}
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for deploy.*
from deploy.utils.config import load_config, repo_path              # noqa: E402
from deploy.utils.link import (CLOUD_PORT, CLOUD_TOPIC, FRAME_PORT, FRAME_TOPIC, TOPIC_PORTS,     # noqa: E402
                               open_subscriber, recv_cloud, recv_frame)
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
    cloud = None
    if CLOUD_PORT in cfg.bus:
        cloud = Clouds(open_subscriber(cfg.bus.host, cfg.bus[CLOUD_PORT], CLOUD_TOPIC, latest_only=False),
                       path.with_suffix(".cloud.npz"))
        poller.register(cloud.sub, zmq.POLLIN)

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    parent = os.getppid()

    counts = dict.fromkeys(topics, 0)
    extra = ([FRAME_TOPIC] if camera else []) + ([CLOUD_TOPIC] if cloud else [])
    print(f"[recorder] {', '.join(topics + extra)} -> {path}", flush=True)
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
            if cloud and cloud.sub in ready:
                cloud.drain(f)
            if time.time() - t_flush > 1.0:
                f.flush()
                t_flush = time.time()
                if os.getppid() != parent:   # run.py died without stopping us
                    break
        for s in subs:                       # whatever arrived during the last poll
            _drain(s, f, counts)
        if camera:
            camera.drain(f)
        if cloud:
            cloud.drain(f)

    size = path.stat().st_size / 1e6
    print(f"[recorder] saved {path} ({size:.1f} MB): " + ", ".join(f"{t} {n}" for t, n in counts.items()), flush=True)
    if camera:
        camera.close()
    if cloud:
        cloud.close()


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


REF_FIELDS = ("ref_xyz", "ref_moving", "ref_corners", "goal_xyz", "goal_corners")   # once, in the npz


class Clouds:
    """point_cloud -> kept in memory, one line per cloud in the .jsonl; the .npz at close (none if none came)."""

    def __init__(self, sub, path: Path) -> None:
        self.sub, self.path = sub, path
        self.xyz, self.header, self.ref = [], [], {}

    def drain(self, f) -> None:
        while (msg := recv_cloud(self.sub)) is not None:
            xyz = msg.pop("xyz")
            for k in REF_FIELDS:
                if k in msg:
                    v = msg.pop(k)
                    self.ref.setdefault(k, np.asarray(v))
            msg["index"] = len(self.xyz)
            self.xyz.append(xyz.copy())
            self.header.append(msg)
            f.write(json.dumps({"topic": CLOUD_TOPIC, "msg": msg}) + "\n")

    def close(self) -> None:
        if not self.xyz:
            return
        m = max(len(x) for x in self.xyz)
        xyz = np.full((len(self.xyz), m, 3), np.nan, np.float32)
        for i, x in enumerate(self.xyz):
            xyz[i, :len(x)] = x
        h = self.header
        np.savez(self.path, xyz=xyz, n=np.array([m["n"] for m in h]), t=np.array([m["t"] for m in h]),
                 seq=np.array([m["seq"] for m in h]), t_cams=np.array([m.get("t_cams", []) for m in h]),
                 depth=str(h[-1].get("depth", "")), **self.ref)
        print(f"[recorder] saved {self.path} ({self.path.stat().st_size / 1e6:.1f} MB, {len(h)} clouds)", flush=True)


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
