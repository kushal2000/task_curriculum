"""Two-camera cloth tracking for a policy that uses the newest points: one loop at stereo_rate_hz.

    stereo_tracker            (the ~/.bashrc alias), or
    ~/miniconda3/envs/cotracker/bin/python deploy/rigid2deform/perception/cotracker_stereo_node.py object.name=[cloth]

Start: click the cloth in A's window, check the points in B's window (cotracker_stereo_picker.py), SPACE.

Then, every step:
  1. wait for camera A's next frame, take it and B's newest (the cameras run free at camera_fps)
  2. run CoTracker3 on each, in parallel: the new frame + the last 15 frames this loop processed
     (stride 1: every step gives a result for the new frame; only the new frame is encoded)
  3. triangulate the points; a point is good if both cameras see it and it reprojects within
     stereo_max_reproj_px in both views. A point that fails keeps its last good position.
  4. publish them as object_points (t = camera A's frame time, so the policy sees how old they are;
     "fresh" marks the points triangulated this step, "visible" the ones with any position yet)
If a step takes longer than 1 / stereo_rate_hz, the loop just runs as fast as it can.
Depth is used once, at the start (the cloth's plane, to carry A's points into B).
"""

from __future__ import annotations

import collections
import copy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform.perception.cotracker_node import GREEN, RED, WHITE, Node, draw_grid, panel, pt   # noqa: E402
from deploy.rigid2deform.sheets import cloth_points, sheet_perception                         # noqa: E402
from deploy.rigid2deform.perception.cotracker_stereo_geometry import base_to_cam, projection_matrix, triangulate   # noqa: E402
from deploy.rigid2deform.perception.cotracker_stereo_picker import CameraBPicker                                   # noqa: E402
from deploy.utils.config import load_config, repo_path                                        # noqa: E402
from deploy.utils.link import encode_jpeg, send                                               # noqa: E402

CONFIG = HERE.parent / "config.yaml"
REPORT_EVERY_S = 5.0


class StereoTracker:
    def __init__(self, cfg):
        c = cfg.perception.cotracker
        if str(c.mode) not in ("grid", "kp8"):
            raise SystemExit(f"[cotracker_stereo_node] mode {c.mode}: stereo needs grid or kp8 (B's points come from A's clicks)")
        self.cfg, self.cot = cfg, c
        self.rate_hz = float(c.stereo_rate_hz)

        cfg_b = copy.deepcopy(cfg)                       # camera B: its own extrinsics, no publishing
        cfg_b.perception.extrinsics = str(c.stereo_extrinsics)
        cfg_b.perception.camera_serial = None
        cfg_b.perception.cotracker.save_dir = None
        self.a = Node(cfg, half=True, stride=1)
        self.b = Node(cfg_b, publish=False, half=True, stride=1)
        for cam in (self.a, self.b):
            if any(abs(k) > 1e-9 for k in cam.intr.coeffs):
                raise SystemExit(f"[cotracker_stereo_node] camera {cam.cam.serial} has lens distortion: undistort first")
        self.T_b_calib = self.b.T_base_cam.copy()
        self.Pa = self.Pb = None                         # projection matrices, set in start()
        self.last_good = None                            # (N, 3) each point's last triangulated position
        self.pool = ThreadPoolExecutor(max_workers=2)    # the two trackers run in parallel
        self.stats = collections.Counter()
        self.seq = 0

    # ---- start: click in A, check in B ----------------------------------------------------------------
    def start(self) -> bool:
        """True once both trackers are running; False means go back to clicking in A."""
        a, b = self.a, self.b
        a.select()
        if not a.start_tracking():
            return False
        a.capture.recording = False                      # this loop reads capture.newest, no frame queue
        pick = CameraBPicker(a, b, self.T_b_calib, self.cot).run()
        if pick is None:
            a.end_run()
            a.frozen = None
            return False
        b.T_base_cam = pick.T_base_cam
        self.Pa = projection_matrix(a.intr, a.T_base_cam)
        self.Pb = projection_matrix(b.intr, b.T_base_cam)
        b.tracker.start(pick.frame, pick.queries)
        self.last_good = np.full((len(pick.queries), 3), np.nan)
        print(f"[cotracker_stereo_node] tracking {len(pick.queries)} points at {self.rate_hz:.0f} Hz (the first "
              f"{a.tracker.window - 1} steps fill CoTracker's window); camera B moved {100 * pick.shift_m:.1f} cm "
              f"to match A ({pick.calib_error_px:.1f} -> {pick.aligned_error_px:.1f} px)", flush=True)
        return True

    # ---- the loop -------------------------------------------------------------------------------------
    def run(self):
        while not self.start():
            pass
        period = 1.0 / self.rate_hz
        t_next = t_report = time.perf_counter()
        while True:
            frame_a = self.next_frame_a()                # start the moment a fresh frame arrives
            frame_b = self.b.capture.newest
            r = self.step(frame_a, frame_b)
            if r is not None:
                send(self.a.pub, "object_points", dict(r["msg"], seq=self.seq))
                self.seq += 1
                self.stats.update(published=1, age_ms=1000 * (time.time() - frame_a[2]))
                self.send_camera_image(r)

            now = time.perf_counter()
            if now - t_report >= REPORT_EVERY_S:
                self.report(r, now - t_report)
                t_report = now
            t_next = max(t_next + period, now)           # running late: carry on from now
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def next_frame_a(self):
        """Wait for camera A's next frame (at most one frame period) so a step never starts on an old one."""
        old_t = self.a.capture.newest[2]
        while self.a.capture.newest[2] == old_t:
            time.sleep(0.0005)
        return self.a.capture.newest

    def step(self, frame_a, frame_b) -> dict | None:
        """Both trackers on their newest frame, then triangulation. None while the window is filling."""
        t0 = time.perf_counter()
        job_a = self.pool.submit(self.a.tracker.add, frame_a)
        job_b = self.pool.submit(self.b.tracker.add, frame_b)
        out_a, out_b = job_a.result(), job_b.result()
        self.stats.update(steps=1, step_ms=1000 * (time.perf_counter() - t0),
                          gap_ms=1000 * abs(frame_a[2] - frame_b[2]))
        if not out_a:
            return None
        (frame_a, uv_a, vis_a), (frame_b, uv_b, vis_b) = out_a[-1], out_b[-1]
        vis_a, vis_b = vis_a.astype(bool), vis_b.astype(bool)
        X, good, err = triangulate(self.Pa, self.Pb, uv_a, uv_b, float(self.cot.stereo_max_reproj_px))
        fresh = vis_a & vis_b & good
        self.last_good[fresh] = X[fresh]                 # a failed point keeps its last good position
        has_pos = np.all(np.isfinite(self.last_good), axis=1)
        self.stats.update(points=len(fresh), triangulated=int(fresh.sum()), held=int((has_pos & ~fresh).sum()),
                          err_sum=float(err[fresh].sum()))
        r = self.a.emit(frame_a, uv_a, has_pos, base_to_cam(self.a.T_base_cam, self.last_good), send_now=False)
        r["msg"]["fresh"] = fresh
        r["b"] = {"frame": frame_b, "uv": uv_b, "vis": vis_b, "failed": vis_a & vis_b & ~good}
        return r

    # ---- viewer image and terminal report ---------------------------------------------------------------
    def send_camera_image(self, r: dict):
        """At image_hz: A's overlay above B's image (B's points: hollow = not seen, red = not triangulated)."""
        a, p = self.a, self.cfg.perception
        t = r["frame"][2]
        due = p.image_hz > 0 and t - a.t_image >= 1.0 / p.image_hz
        if not due and a.log is None:
            return
        rb = r["b"]
        img_b = rb["frame"][0].copy()
        if a.mode == "grid":
            draw_grid(img_b, a.geom, rb["uv"], hollow=~rb["vis"])
        else:
            for uv, seen in zip(rb["uv"], rb["vis"]):
                cv2.circle(img_b, pt(uv), 6, WHITE, -1 if seen else 1, cv2.LINE_AA)
        for uv in rb["uv"][rb["failed"]]:
            cv2.circle(img_b, pt(uv), 6, RED, 2, cv2.LINE_AA)
        panel(img_b, [f"B  seen {rb['vis'].sum()}/{len(rb['vis'])}  not triangulated {rb['failed'].sum()}"],
              RED if rb["failed"].any() else GREEN)
        img = np.vstack([a.render_track(r), img_b])
        if due:
            send(a.pub, "camera_image", {"seq": self.seq, "t": t, "jpeg": encode_jpeg(img, p.image_jpeg_quality)})
            a.t_image = t
        a.save_frame(img, fps=self.rate_hz)

    def report(self, r: dict | None, dt: float):
        s, self.stats = self.stats, collections.Counter()
        steps, published, points = max(s["steps"], 1), max(s["published"], 1), max(s["points"], 1)
        fold = ("filling the window" if r is None else
                "a half has < 3 points" if r["sc"] is None else f"fold {r['angle']:.0f} deg")
        print(f"[cotracker_stereo_node] {s['steps'] / dt:.0f} Hz, step {s['step_ms'] / steps:.0f} ms, points "
              f"{s['age_ms'] / published:.0f} ms old when sent, A/B frames {s['gap_ms'] / steps:.0f} ms apart | "
              f"triangulated {100 * s['triangulated'] / points:.0f}%, held {100 * s['held'] / points:.0f}%, "
              f"{s['err_sum'] / max(s['triangulated'], 1):.1f} px "
              f"| {fold}", flush=True)

    def close(self):
        self.pool.shutdown()
        self.a.close()
        self.b.close()


def main():
    cfg = load_config(path=CONFIG)
    if sheet_perception(cfg) != "cotracker":
        raise SystemExit(f"[cotracker_stereo_node] object.name={cfg.object.name}: not a cotracker object (use object.name=[cloth])")
    cloth_points.check(cfg)
    if not repo_path(cfg.perception.cotracker.stereo_extrinsics).is_file():
        raise SystemExit(f"[cotracker_stereo_node] {cfg.perception.cotracker.stereo_extrinsics}: no such file")
    tracker = StereoTracker(cfg)
    try:
        tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
