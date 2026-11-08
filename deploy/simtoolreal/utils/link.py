"""Messages between the deploy processes, over ZMQ.

Each publishing process binds its own port (config.yaml `bus`); any number of processes subscribe.
A message is one string, "<topic> <json>", and arrives as a dict; lists come back as numpy arrays.
Every message has seq (counts up from 0, the sender counts) and t (time.time(), see below).

| topic        | sender                      | port               | fields                                        |
|--------------|-----------------------------|--------------------|-----------------------------------------------|
| object_pose  | perception.py               | object_pose_port   | pos (3), quat_wxyz (4); t = camera frame time  |
| camera_image | perception.py (camera)      | object_pose_port   | jpeg (base64 text); t = camera frame time      |
| joint_state  | robot.py                    | joint_state_port   | q (27), dq (27), tripped (None or the safety   |
|              |                             |                    | trip's reason); t = when measured              |
| joint_target | policy.py; robot.py in      | policy_port        | targets (27); t = when sent                    |
|              | robot.mode=openloop         |                    |                                               |
| goal_pose    | policy.py; robot.py         | policy_port        | pos (3), quat_wxyz (4), goal_index; t = sent   |
|              | (openloop: the sim's goal)  |                    |                                               |
| sim_object_  | robot.py (openloop)         | policy_port        | pos (3), quat_wxyz (4): where the sim rollout  |
| pose         |                             |                    | had the object at this step; t = sent          |

Poses are in the robot base frame; joints in the spec's joint order (7 arm, then 20 hand), rad.

Publisher:
    pub = open_publisher(port)
    send(pub, "joint_target", {"seq": seq, "t": time.time(), "targets": targets})

Subscriber that only needs the newest message (policy, viewer):
    sub = open_subscriber(host, port, "object_pose", latest_only=True)
    new = recv_latest(sub)          # each tick; None = nothing new since the last call
    if new is not None:
        pose = new                  # keep the last one yourself
    if pose is None or age(pose) > max_age_s: ...   # lost

Subscriber that needs every message (recorder):
    sub = open_subscriber(host, port, "object_pose", latest_only=False)
    msg = recv_wait(sub, timeout_s=1.0)   # None on timeout

One subscriber socket per topic: latest_only keeps one message per socket, whatever its topic.
Needs only numpy + pyzmq, so it also runs in the FoundationPose environment.
"""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq

NODES_DIR = Path(__file__).resolve().parents[1]   # deploy/simtoolreal

# every topic -> its publisher's port key in config.yaml `bus`
TOPIC_PORTS = {"joint_state": "joint_state_port", "joint_target": "policy_port", "goal_pose": "policy_port",
               "sim_object_pose": "policy_port", "object_pose": "object_pose_port",
               "camera_image": "object_pose_port"}


def open_publisher(port: int) -> zmq.Socket:
    """Start publishing on tcp://*:port."""
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    return sock


def send(sock: zmq.Socket, topic: str, msg: dict) -> None:
    """Send one message. Never waits: with no subscriber it is dropped."""
    sock.send_string(f"{topic} {json.dumps(msg, default=_to_json)}")


def open_subscriber(host: str, port: int, topic: str, latest_only: bool = True) -> zmq.Socket:
    """Connect to tcp://host:port and receive `topic` only.
    latest_only=True keeps only the newest message (older ones are dropped); False queues them all."""
    sock = zmq.Context.instance().socket(zmq.SUB)
    if latest_only:
        sock.setsockopt(zmq.CONFLATE, 1)   # must be set before connect
    sock.setsockopt(zmq.RECONNECT_IVL, 10)  # ms; retry fast while the publisher is not up yet (default 100)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic + " ")
    sock.connect(f"tcp://{host}:{port}")
    return sock


def recv_latest(sock: zmq.Socket) -> dict | None:
    """The newest message that arrived since the last call, without waiting. None if nothing new."""
    try:
        return _parse(sock.recv_string(zmq.NOBLOCK))
    except zmq.Again:
        return None


def recv_wait(sock: zmq.Socket, timeout_s: float = 1.0) -> dict | None:
    """The next message, waiting up to timeout_s. None on timeout."""
    if sock.poll(int(timeout_s * 1000)) == 0:
        return None
    return _parse(sock.recv_string())


def age(msg: dict) -> float:
    """Seconds since the message's t."""
    return time.time() - msg["t"]


def start_node(script: str, *overrides: str) -> subprocess.Popen:
    """Start another node (e.g. "viewer.py") with this process's command-line overrides (so both read
    the same config), then `overrides` (these win). Its own session: a Ctrl-C in the terminal reaches
    only this process, which ends the child with stop_node."""
    return subprocess.Popen([sys.executable, str(NODES_DIR / script), *sys.argv[1:], *overrides],
                            start_new_session=True)


def stop_node(proc: subprocess.Popen | None, sig=signal.SIGINT, timeout_s: float = 10.0) -> None:
    """End a node started by start_node: sig (SIGINT = as Ctrl-C, so it cleans up), then kill."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(sig)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()


def encode_jpeg(rgb: np.ndarray, quality: int = 80) -> str:
    """camera_image's "jpeg": RGB image -> JPEG as base64 text. 640 x 480 at quality 80: ~40 kB. Needs cv2."""
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg(text: str) -> np.ndarray:
    """camera_image's "jpeg" -> RGB image. Needs cv2."""
    import base64
    import cv2
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(text), np.uint8), cv2.IMREAD_COLOR)
    return bgr[:, :, ::-1]


def _to_json(v):
    """json.dumps fallback for numpy values."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"cannot send {type(v)}")


def _parse(s: str) -> dict:
    """"<topic> <json>" -> dict, lists as numpy arrays."""
    _, body = s.split(" ", 1)
    return from_json(json.loads(body))


def from_json(d: dict) -> dict:
    """A message as json.loads gives it -> lists as numpy arrays (as received over the bus)."""
    return {k: np.array(v) if isinstance(v, list) else v for k, v in d.items()}
