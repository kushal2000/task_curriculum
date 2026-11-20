"""Camera B's point picking for cotracker_stereo_node.py: an OpenCV window that finds, in camera B's image, the
points you clicked in camera A.

How a point gets from A to B:
  1. carry   A's clicked pixel's ray is cut by the cloth's plane (fitted to A's depth inside the clicked
             corners: the cloth lies flat at the start) and the 3D point projected into B with B's
             calibration (grey dots).
  2. snap    The two calibrations disagree by a few cm (left_camera_nov7 / right_camera_nov10: 2 cm at the
             cloth, 13 px), so the 4 outer corners snap to image corners in B: first all 4 together by
             the one shift within stereo_snap_radius that lands the most corner strength under them,
             then each one on its own within snap_radius.
  3. follow  The other points follow the 4 corners by a homography (yellow).
  4. align   stereo_align: camera B is moved (its position only) so A's 3D points project onto B's
             points. Without it a matched pair reprojects ~13 px off and nothing triangulates.
             Moving B more than ALIGN_WARN_M means a corner is probably on the wrong thing.

Keys in the window: click = move the nearest corner there, SPACE = track, m = snap on / off, r = a new
B frame, c = click A again, q = quit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from deploy.rigid2deform.perception.cotracker_node import GREEN, PERIMETER, RED, WHITE, YELLOW, draw_grid, panel, pt, ray_plane
from deploy.rigid2deform.perception.cotracker_stereo_geometry import cam_to_base, move_camera_onto, project, projection_matrix

ALIGN_WARN_M = 0.05         # stereo_align moved B further than this: probably the wrong corners
GREY = (160, 160, 160)


@dataclass
class PickResult:
    frame: tuple                 # camera B's frame the queries are on: (rgb, depth, t)
    queries: np.ndarray          # (N, 2) px in B, in the same order as A's queries
    T_base_cam: np.ndarray       # B's extrinsics to track with (moved onto A's points if stereo_align)
    calib_error_px: float        # mean px between A's points seen through B's calibration and the queries
    aligned_error_px: float      # the same after moving B
    shift_m: float               # how far stereo_align moved B


class CameraBPicker:
    """Shows A's clicked points in camera B's image and lets you fix B's 4 outer corners.

    a, b: cotracker_node.Node for each camera; A must already have its clicks and layout
    (a.start_tracking() done). T_b_calib: B's calibration as loaded. run() returns a PickResult on SPACE,
    None when you want to click A again, and raises KeyboardInterrupt on q."""

    WINDOW = "cotracker_stereo_node: camera B"

    def __init__(self, a, b, T_b_calib: np.ndarray, cotracker_cfg):
        self.a, self.b, self.T_b_calib, self.cfg = a, b, T_b_calib, cotracker_cfg
        self.snap = bool(cotracker_cfg.stereo_snap)
        self.frame = None            # B's frame being picked on
        self.X = None                # (N, 3) A's queries on the cloth's plane, base frame
        self.carried = None          # (N, 2) X in B by B's calibration
        self.carried_corners = None  # (4, 2) A's 4 outer clicks in B by B's calibration
        self.corners = None          # (4, 2) where the outer corners are in B: carried, snapped or clicked
        self.message = ""            # why there is nothing to pick on, if so

    # ---- the window ------------------------------------------------------------------------------
    def run(self) -> PickResult | None:
        self._new_view()
        cv2.namedWindow(self.WINDOW)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        try:
            while True:
                result = self._solve()
                cv2.imshow(self.WINDOW, cv2.cvtColor(self._draw(result), cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord(" ") and result is not None:
                    return result
                if key == ord("c"):
                    return None
                if key == ord("m"):
                    self.snap = not self.snap
                    self._new_view()
                if key == ord("r"):
                    self._new_view()
        finally:
            cv2.destroyWindow(self.WINDOW)
            cv2.waitKey(1)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        """Left click: the nearest outer corner goes there (snapped within snap_radius while snap is on)."""
        if event != cv2.EVENT_LBUTTONDOWN or self.corners is None:
            return
        p = np.array([[float(x), float(y)]])
        if self.snap:
            p = self._snap_to_corners(p, int(self.cfg.snap_radius))
        nearest = np.argmin(np.linalg.norm(self.corners - p, axis=1))
        self.corners[nearest] = p[0]

    # ---- A's points into B -----------------------------------------------------------------------
    def _new_view(self) -> None:
        """A fresh B frame, A's points carried into it, the corners snapped if snap is on."""
        self.frame = self.b.capture.newest
        self.message = self._carry()
        self.corners = None
        if self.message is None:
            self.corners = self.carried_corners.copy()
            if self.snap:
                self.corners = self._snap_to_corners(self.corners, int(self.cfg.snap_radius),
                                                     int(self.cfg.stereo_snap_radius))

    def _carry(self) -> str | None:
        """A's queries and 4 outer clicks -> the cloth's plane -> B's image. None, or why it failed."""
        a = self.a
        if a.lay["plane"] is None:
            return "no depth on the cloth in A: click again"
        # The corners are A's clicks, not queries[PERIMETER]: in grid mode the queries are the 49 grid points.
        px = np.vstack([a.lay["queries"], np.asarray(a.clicks)[PERIMETER[a.mode]]]).astype(np.float64)
        X = cam_to_base(a.T_base_cam, ray_plane(a.intr, px, a.lay["plane"]))
        uv, depth = project(projection_matrix(self.b.intr, self.T_b_calib), X)
        h, w = self.frame[0].shape[:2]
        inside = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if not inside.all():
            return f"{int((~inside).sum())} point(s) outside B's image"
        self.X, self.carried, self.carried_corners = X[:-4], uv[:-4], uv[-4:]
        return None

    def _snap_to_corners(self, px: np.ndarray, radius: int, search: int = 0) -> np.ndarray:
        """Each pixel (K, 2) moved to the strongest image corner within radius, or left where it is.

        search > 0: first move all of them together by the one shift (within search px) that puts the most
        corner strength under them. A calibration error moves every point alike, and snapping each on its
        own that far would grab a neighbour's corner."""
        b = self.b
        gray = cv2.cvtColor(self.frame[0], cv2.COLOR_RGB2GRAY)
        strength = b._strength_map(gray)
        px = np.asarray(px, dtype=np.float64)
        if search:
            px = px + self._best_common_shift(strength, px, search)
        return np.array([b._snap(gray, strength, float(u), float(v), radius)[:2] for u, v in px], dtype=np.float64)

    @staticmethod
    def _best_common_shift(strength: np.ndarray, px: np.ndarray, search: int) -> tuple[int, int]:
        size = 2 * search + 1
        padded = cv2.copyMakeBorder(cv2.dilate(strength, np.ones((3, 3), np.uint8)), search, search, search, search,
                                    cv2.BORDER_CONSTANT, value=0.0)
        score = np.zeros((size, size))
        h, w = strength.shape
        for u, v in np.clip(np.round(px).astype(int), 0, [w - 1, h - 1]):
            window = padded[v: v + size, u: u + size]                         # centred on (u, v)
            score += np.sqrt(window / max(float(window.max()), 1e-12))        # each corner counts alike
        d = np.arange(-search, search + 1)
        score *= np.exp(-(d[:, None] ** 2 + d[None] ** 2) / (2.0 * search ** 2))   # a tie: the smaller shift
        iv, iu = np.unravel_index(np.argmax(score), score.shape)
        return d[iu], d[iv]

    # ---- the result ------------------------------------------------------------------------------
    def _solve(self) -> PickResult | None:
        """B's queries for the current corners, and B's extrinsics (moved onto A's points if stereo_align)."""
        if self.corners is None:
            return None
        H, _ = cv2.findHomography(self.carried_corners, self.corners)
        queries = (self.carried if H is None else cv2.perspectiveTransform(self.carried[None], H)[0]).astype(np.float32)
        T = self.T_b_calib
        if self.cfg.stereo_align:
            T = move_camera_onto(self.b.intr, T, self.X, queries.astype(np.float64))
        after = np.linalg.norm(project(projection_matrix(self.b.intr, T), self.X)[0] - queries, axis=1).mean()
        return PickResult(frame=self.frame, queries=queries, T_base_cam=T,
                          calib_error_px=float(np.linalg.norm(self.carried - queries, axis=1).mean()),
                          aligned_error_px=float(after),
                          shift_m=float(np.linalg.norm(T[:3, 3] - self.T_b_calib[:3, 3])))

    def _draw(self, result: PickResult | None) -> np.ndarray:
        img = self.frame[0].copy()
        if result is None:
            panel(img, [f"B: {self.message}", "c: click A again  r: new B frame  q: quit"], RED)
            return img
        if self.a.mode == "grid":
            draw_grid(img, self.a.geom, result.queries)
        for p0, p in zip(self.carried, result.queries):           # grey: by calibration, yellow: placed
            cv2.line(img, pt(p0), pt(p), WHITE, 1, cv2.LINE_AA)
            cv2.circle(img, pt(p0), 3, GREY, -1, cv2.LINE_AA)
            cv2.circle(img, pt(p), 4, YELLOW, 1, cv2.LINE_AA)
        for p0, p in zip(self.carried_corners, self.corners):
            cv2.line(img, pt(p0), pt(p), WHITE, 1, cv2.LINE_AA)
            cv2.circle(img, pt(p), 8, YELLOW, 2, cv2.LINE_AA)
        if self.cfg.stereo_align:
            align = (f"B moved {100 * result.shift_m:.1f} cm: "
                     f"{result.calib_error_px:.1f} -> {result.aligned_error_px:.1f} px")
        else:
            align = f"B's calibration {result.calib_error_px:.1f} px off, not aligned"
        panel(img, [f"B: {len(result.queries)} points from A  |  snap {'ON' if self.snap else 'OFF'} (m)  |  {align}",
                    "SPACE: track  click: move the nearest corner  r: new B frame  c: click A again  q: quit"],
              RED if result.shift_m > ALIGN_WARN_M else GREEN)
        return img
