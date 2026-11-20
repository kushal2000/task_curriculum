"""Geometry for two calibrated cameras: projecting base-frame points into a camera, triangulating matched
pixels, and nudging a camera's position so its view agrees with the other camera's.

All points are in the robot base frame unless a name says otherwise. T_base_cam is a camera's 4 x 4 pose
in the base frame (its extrinsics). Pixels are undistorted (the RealSense colour stream reports no
distortion; cotracker_stereo_node.py checks that).
"""

from __future__ import annotations

import cv2
import numpy as np


def projection_matrix(intr, T_base_cam: np.ndarray) -> np.ndarray:
    """3 x 4 matrix: homogeneous base-frame point -> homogeneous pixel."""
    K = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])
    return K @ np.linalg.inv(T_base_cam)[:3]


def project(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points X (N, 3) through projection matrix P -> (pixels (N, 2), depth along the camera's
    axis (N,), m). Points behind the camera get a negative depth."""
    x = np.c_[X, np.ones(len(X))] @ P.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return x[:, :2] / x[:, 2:], x[:, 2]


def triangulate(Pa: np.ndarray, Pb: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray,
                max_reproj_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matched pixels in cameras A and B -> (X (N, 3), good (N,), err (N,) px).

    X is the linear (DLT) triangulation. err is the worse of the two views' reprojection errors. good means
    the point lies in front of both cameras and reprojects within max_reproj_px in both."""
    Xh = cv2.triangulatePoints(Pa, Pb, uv_a.T.astype(np.float64), uv_b.T.astype(np.float64))   # (4, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        X = (Xh[:3] / Xh[3]).T
    (pa, za), (pb, zb) = project(Pa, X), project(Pb, X)
    err = np.maximum(np.linalg.norm(pa - uv_a, axis=1), np.linalg.norm(pb - uv_b, axis=1))
    good = np.isfinite(err) & (za > 0) & (zb > 0) & (err < max_reproj_px)
    return X, good, err


def cam_to_base(T_base_cam: np.ndarray, xyz_cam: np.ndarray) -> np.ndarray:
    return xyz_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def base_to_cam(T_base_cam: np.ndarray, xyz_base: np.ndarray) -> np.ndarray:
    return (xyz_base - T_base_cam[:3, 3]) @ T_base_cam[:3, :3]


def move_camera_onto(intr, T_base_cam: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """T_base_cam with the camera moved (rotation kept) so that base points X (N, 3) project onto pixels
    uv (N, 2), by least squares.

    It is linear in the camera-frame translation t (x_cam = R^T X + t):
        fx (x + tx) = (u - cx) (z + tz),   fy (y + ty) = (v - cy) (z + tz).
    Seen from a small flat cloth, a small rotation looks like a shift, so moving the camera corrects both
    near the cloth."""
    R = T_base_cam[:3, :3]
    x = X @ R                                             # rotated into the camera, not yet translated
    du, dv, zero = uv[:, 0] - intr.cx, uv[:, 1] - intr.cy, np.zeros(len(X))
    A = np.r_[np.c_[np.full(len(X), intr.fx), zero, -du], np.c_[zero, np.full(len(X), intr.fy), -dv]]
    rhs = np.r_[du * x[:, 2] - intr.fx * x[:, 0], dv * x[:, 2] - intr.fy * x[:, 1]]
    t = np.linalg.lstsq(A, rhs, rcond=None)[0]
    out = T_base_cam.copy()
    out[:3, 3] = -R @ t
    return out
