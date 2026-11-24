"""Fused Warp kernels for ``PointCloud.visible``: the same visibility test, without the memory traffic.

The torch version materialises a dozen ``(n, 2, P)`` tensors per call (camera-frame coordinates, pixel
indices, depths, masks) and scatters every point into the depth images with ``scatter_reduce``; at
24576 envs x 2 cameras x ~17k candidates that is ~800M points a step and the cloud cost ~1.2 s. Here
each point is projected in a register and written once:

1. ``_splat``   -- project, ``atomic_min`` its depth into its camera's depth image;
2. ``_dilate``  -- each pixel takes the nearest depth of its 3x3 neighbourhood (in-image only:
                   ``max_pool2d``'s implicit -inf padding);
3. ``_test``    -- project again, visible when inside the image, beyond ``min_depth`` and within
                   ``depth_tol`` of its neighbourhood's nearest surface.

Same semantics as the torch path (``PointCloud.visible``), compared in tests/test_visibility_warp.py.
"""

from __future__ import annotations

import torch
import warp as wp

__all__ = ["visible_warp"]

_INF = 1.0e30


@wp.func
def _project(p: wp.vec3, cam: wp.vec3, r: wp.mat33, f: float, w: int, h: int, min_depth: float):
    """Pixel (u, v), depth z and whether the point is in the image beyond ``min_depth``."""
    d = p - cam
    # local_j = sum_i d_i R[i][j]  (R's columns are the camera axes): R^T d
    lx = d[0] * r[0, 0] + d[1] * r[1, 0] + d[2] * r[2, 0]
    ly = d[0] * r[0, 1] + d[1] * r[1, 1] + d[2] * r[2, 1]
    lz = d[0] * r[0, 2] + d[1] * r[1, 2] + d[2] * r[2, 2]
    inv = f / wp.max(lz, 1.0e-3)
    u = int(wp.floor(lx * inv + 0.5 * float(w)))
    v = int(wp.floor(ly * inv + 0.5 * float(h)))
    ok = (lz > min_depth) and (u >= 0) and (u < w) and (v >= 0) and (v < h) and wp.isfinite(lz)
    return u, v, lz, ok


@wp.kernel
def _splat(pts: wp.array2d(dtype=wp.vec3), cams: wp.array2d(dtype=wp.vec3), rots: wp.array2d(dtype=wp.mat33),
           f: float, w: int, h: int, min_depth: float, zbuf: wp.array2d(dtype=float)):
    e, c, p = wp.tid()
    u, v, z, ok = _project(pts[e, p], cams[e, c], rots[e, c], f, w, h, min_depth)
    if ok:
        wp.atomic_min(zbuf, e * 2 + c, v * w + u, z)


@wp.kernel
def _splat_bodies(body_pos: wp.array2d(dtype=wp.vec3), body_quat: wp.array2d(dtype=wp.vec4),
                  pool_body: wp.array(dtype=wp.int32), pool_off: wp.array(dtype=wp.vec3),
                  cams: wp.array2d(dtype=wp.vec3), rots: wp.array2d(dtype=wp.mat33),
                  f: float, w: int, h: int, min_depth: float, zbuf: wp.array2d(dtype=float)):
    """The robot's surface samples, moved by their bodies' poses in a register (wxyz quaternions)."""
    e, c, p = wp.tid()
    b = pool_body[p]
    qv = body_quat[e, b]
    q = wp.quat(qv[1], qv[2], qv[3], qv[0])                  # warp quats are xyzw
    x = body_pos[e, b] + wp.quat_rotate(q, pool_off[p])
    u, v, z, ok = _project(x, cams[e, c], rots[e, c], f, w, h, min_depth)
    if ok:
        wp.atomic_min(zbuf, e * 2 + c, v * w + u, z)


@wp.kernel
def _dilate(zbuf: wp.array2d(dtype=float), w: int, h: int, near: wp.array2d(dtype=float)):
    img, pix = wp.tid()
    v = pix // w
    u = pix - v * w
    best = float(_INF)
    for dv in range(-1, 2):
        for du in range(-1, 2):
            uu = u + du
            vv = v + dv
            if uu >= 0 and uu < w and vv >= 0 and vv < h:
                best = wp.min(best, zbuf[img, vv * w + uu])
    near[img, pix] = best


@wp.kernel
def _test(pts: wp.array2d(dtype=wp.vec3), cams: wp.array2d(dtype=wp.vec3), rots: wp.array2d(dtype=wp.mat33),
          f: float, w: int, h: int, min_depth: float, tol: float, near: wp.array2d(dtype=float),
          out: wp.array3d(dtype=wp.uint8)):
    e, c, p = wp.tid()
    u, v, z, ok = _project(pts[e, p], cams[e, c], rots[e, c], f, w, h, min_depth)
    vis = wp.uint8(0)
    if ok:
        if z <= near[e * 2 + c, v * w + u] + tol:
            vis = wp.uint8(1)
    out[e, c, p] = vis


def visible_warp(pts: torch.Tensor, cams: torch.Tensor, rots: torch.Tensor, f: float, w: int, h: int,
                 min_depth: float, depth_tol: float, test_upto: int | None = None,
                 body_pos: torch.Tensor | None = None, body_quat: torch.Tensor | None = None,
                 pool_body: torch.Tensor | None = None, pool_off: torch.Tensor | None = None) -> torch.Tensor:
    """``(n, 2, P)`` bool, as ``PointCloud.visible``. Every point occludes; only the first ``test_upto``
    are tested (the rest come back False) -- a cloud that keeps only the manipuland skips the robot's.

    With ``body_pos`` ``(n, B, 3)`` / ``body_quat`` ``(n, B, 4)`` wxyz and the robot's surface pool
    (``pool_body`` ``(R,)``, ``pool_off`` ``(R, 3)``), the robot occludes too without ever being built
    as an ``(n, R, 3)`` tensor (it is not tested)."""
    wp.init()                                   # no-op once initialised (Kit initialises it; a plain process may not)
    n, p = int(pts.shape[0]), int(pts.shape[1])
    q = p if test_upto is None else int(test_upto)
    dev = pts.device
    pts = pts.detach().float().contiguous()
    cams_c = cams.detach().float().contiguous()
    rots_c = rots.detach().float().contiguous()
    zbuf = torch.full((n * 2, h * w), _INF, device=dev, dtype=torch.float32)
    near = torch.empty_like(zbuf)
    out = torch.zeros((n, 2, p), device=dev, dtype=torch.uint8)
    stream = wp.stream_from_torch(torch.cuda.current_stream(dev))
    wd = wp.device_from_torch(dev)
    a_pts = wp.from_torch(pts, dtype=wp.vec3)
    a_cams = wp.from_torch(cams_c, dtype=wp.vec3)
    a_rots = wp.from_torch(rots_c, dtype=wp.mat33)
    a_z, a_near = wp.from_torch(zbuf), wp.from_torch(near)
    wp.launch(_splat, dim=(n, 2, p), inputs=[a_pts, a_cams, a_rots, float(f), int(w), int(h), float(min_depth), a_z],
              device=wd, stream=stream)
    if body_pos is not None:
        bp = body_pos.detach().float().contiguous()
        bq = body_quat.detach().float().contiguous()
        wp.launch(_splat_bodies, dim=(n, 2, int(pool_body.numel())),
                  inputs=[wp.from_torch(bp, dtype=wp.vec3), wp.from_torch(bq, dtype=wp.vec4),
                          wp.from_torch(pool_body.to(torch.int32).contiguous(), dtype=wp.int32),
                          wp.from_torch(pool_off.float().contiguous(), dtype=wp.vec3),
                          a_cams, a_rots, float(f), int(w), int(h), float(min_depth), a_z],
                  device=wd, stream=stream)
    wp.launch(_dilate, dim=(n * 2, h * w), inputs=[a_z, int(w), int(h), a_near], device=wd, stream=stream)
    if q > 0:
        a_out = wp.from_torch(out[:, :, :q] if q == p else out, dtype=wp.uint8)
        wp.launch(_test, dim=(n, 2, q),
                  inputs=[a_pts, a_cams, a_rots, float(f), int(w), int(h), float(min_depth), float(depth_tol), a_near, a_out],
                  device=wd, stream=stream)
    return out.bool()
