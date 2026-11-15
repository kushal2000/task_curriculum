"""Quaternion helpers. Quaternions are (w, x, y, z) everywhere, except the policy's observation,
which uses (x, y, z, w) (wxyz_to_xyzw). All functions broadcast over leading axes.
"""

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b: rotate by b first, then by a."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """The inverse rotation (for unit quaternions)."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by q (same formula as Isaac Lab's quat_apply)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_from_angle_axis(angle, axis: np.ndarray) -> np.ndarray:
    """Rotation by angle (rad) about a unit axis."""
    half = 0.5 * np.asarray(angle)[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


def quat_from_rpy(rpy) -> np.ndarray:
    """URDF rpy (fixed axes: roll about x, then pitch about y, then yaw about z)."""
    r, p, y = rpy
    qx = quat_from_angle_axis(r, np.array([1.0, 0.0, 0.0]))
    qy = quat_from_angle_axis(p, np.array([0.0, 1.0, 0.0]))
    qz = quat_from_angle_axis(y, np.array([0.0, 0.0, 1.0]))
    return quat_mul(qz, quat_mul(qy, qx))


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation (as Isaac Lab: a normalized Gaussian 4-vector)."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)
