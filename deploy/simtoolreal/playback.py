"""Play back a recording (recorder.py) in viser, and export it as an mp4. Never moves the robot.

    .venv_deploy/bin/python deploy/simtoolreal/playback.py <recording.jsonl> [viewer.port=8092]

Open the page: the scene is the live viewer's (viewer.py), at the time on the slider. "play" runs it
at the chosen speed. "export mp4": set the 3D view in the browser first; every frame is rendered by
that browser tab (viser get_render) at record.video_fps, the camera image (if recorded) is put on
its right, and the video is written next to the recording (<recording>.mp4).

The config comes from the recording (spec, object, success tolerance as they were); command-line
overrides apply on top (e.g. viewer.port, record.video_fps).
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from utils.config import repo_path                            # noqa: E402
from utils.link import decode_jpeg, from_json                 # noqa: E402
from utils.spec import load_spec                              # noqa: E402
from viewer import build_scene, draw, goal_text, success_tolerance   # noqa: E402


def load_recording(path: Path) -> tuple[dict, dict]:
    """(the run's config, {topic: (times (N,), messages [N])})."""
    config, topics = None, {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["topic"] == "meta":
                config = d["msg"]["config"]
            else:
                topics.setdefault(d["topic"], []).append(from_json(d["msg"]))
    return config, {k: (np.array([m["t"] for m in msgs]), msgs) for k, msgs in topics.items()}


def at(rec: dict, t: float) -> dict:
    """The newest message of every topic sent at or before time t (None if none yet)."""
    out = {}
    for topic, (times, msgs) in rec.items():
        i = np.searchsorted(times, t, side="right") - 1
        out[topic] = msgs[i] if i >= 0 else None
    return out


def export_mp4(client, server, scene, spec, rec, t0: float, duration: float, cfg, out: Path, progress) -> None:
    """Render every frame in the client's browser and write the mp4."""
    import cv2

    fps, h = cfg.record.video_fps, cfg.record.video_height
    w3d = int(round(h * 4 / 3))
    has_camera = "camera_image" in rec
    width = w3d + (w3d if has_camera else 0)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, h))
    n = int(duration * fps) + 1
    camera, camera_seq = np.zeros((h, w3d, 3), np.uint8), None
    try:
        for k in range(n):
            msgs = at(rec, t0 + k / fps)
            draw(server, scene, spec, msgs)
            server.flush()
            frame = client.get_render(height=h, width=w3d)[:, :, :3]
            if has_camera and msgs["camera_image"] is not None and msgs["camera_image"]["seq"] != camera_seq:
                img = decode_jpeg(msgs["camera_image"]["jpeg"])
                camera = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))[:, :w3d]
                camera = np.pad(camera, ((0, 0), (0, w3d - camera.shape[1]), (0, 0)))
                camera_seq = msgs["camera_image"]["seq"]
            if has_camera:
                frame = np.hstack([frame, camera])
            writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
            if k % fps == 0:
                progress(f"exporting: {k / fps:.0f} / {duration:.0f} s")
    finally:
        writer.release()
    progress(f"saved **{out}** ({n} frames)")


def main() -> None:
    import viser

    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1]).resolve()
    config, rec = load_recording(path)
    cfg = OmegaConf.merge(OmegaConf.create(config), OmegaConf.from_dotlist(sys.argv[2:]))
    spec = load_spec(repo_path(cfg.spec_path))
    tolerance = success_tolerance(cfg, spec)

    t0 = min(times[0] for times, _ in rec.values())
    duration = max(times[-1] for times, _ in rec.values()) - t0
    counts = ", ".join(f"{k} {len(v[1])}" for k, v in rec.items())
    print(f"[playback] {path.name}: {duration:.1f} s; {counts}", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=cfg.viewer.port)
    scene = build_scene(server, spec, repo_path(cfg.objects[cfg.object.name].mesh))
    slider = server.gui.add_slider("time (s)", min=0.0, max=max(duration, 0.01), step=1.0 / cfg.record.video_fps,
                                   initial_value=0.0)
    play = server.gui.add_checkbox("play", False)
    speed = server.gui.add_dropdown("speed", ("0.25", "0.5", "1", "2", "4"), initial_value="1")
    export = server.gui.add_button("export mp4 (uses this tab's 3D view)")
    note = server.gui.add_markdown("")
    busy = threading.Lock()   # the export and the playback loop both move the scene

    @export.on_click
    def _(event) -> None:
        if not busy.acquire(blocking=False):
            return
        play.value = False
        try:
            def progress(text: str) -> None:
                note.content = text
            export_mp4(event.client, server, scene, spec, rec, t0, duration, cfg, path.with_suffix(".mp4"), progress)
        finally:
            busy.release()

    print(f"[playback] http://localhost:{cfg.viewer.port}", flush=True)
    shown, camera_seq, dt = None, None, 1.0 / cfg.viewer.hz
    while True:
        t_tick = time.perf_counter()
        if play.value:
            slider.value = min(slider.value + dt * float(speed.value), slider.max)
            if slider.value >= slider.max:
                play.value = False
        if slider.value != shown and busy.acquire(blocking=False):
            try:
                shown = slider.value
                msgs = at(rec, t0 + shown)
                draw(server, scene, spec, msgs)
                cam = msgs.get("camera_image")
                if cam is not None and cam["seq"] != camera_seq:
                    scene["image"].image = decode_jpeg(cam["jpeg"])
                    camera_seq = cam["seq"]
                scene["status"].content = f"t = {shown:.2f} s / {duration:.1f} s\n\n" + goal_text(spec, msgs, tolerance)
            finally:
                busy.release()
        time.sleep(max(0.0, dt - (time.perf_counter() - t_tick)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
