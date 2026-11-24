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

#: False: always dilate the whole depth image (the reference path; tests compare the two).
FUSE_DILATION = True
#: False: every robot point is written into the depth images (the reference path).
CULL_BODIES = True


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
def _pixel_box(hull: wp.array2d(dtype=wp.vec3), cams: wp.array2d(dtype=wp.vec3), rots: wp.array2d(dtype=wp.mat33),
               f: float, w: int, h: int, box: wp.array2d(dtype=wp.int32)):
    """Per image, the pixel rectangle ``hull``'s points project into, ``box[img] = (u min, v min, -u max,
    -v max)`` (not clipped to the image); the whole image when a point is not in front of the camera."""
    e, c, p = wp.tid()
    d = hull[e, p] - cams[e, c]
    r = rots[e, c]
    lx = d[0] * r[0, 0] + d[1] * r[1, 0] + d[2] * r[2, 0]
    ly = d[0] * r[0, 1] + d[1] * r[1, 1] + d[2] * r[2, 1]
    lz = d[0] * r[0, 2] + d[1] * r[1, 2] + d[2] * r[2, 2]
    img = e * 2 + c
    u = 0
    v = 0
    nu = 1 - w
    nv = 1 - h
    if lz > 1.0e-3 and wp.isfinite(lz) and wp.isfinite(lx) and wp.isfinite(ly):
        inv = f / lz
        fu = wp.clamp(wp.floor(lx * inv + 0.5 * float(w)), -4.0, float(w) + 4.0)
        fv = wp.clamp(wp.floor(ly * inv + 0.5 * float(h)), -4.0, float(h) + 4.0)
        u = int(fu)
        v = int(fv)
        nu = -u
        nv = -v
    wp.atomic_min(box, img, 0, u)
    wp.atomic_min(box, img, 1, v)
    wp.atomic_min(box, img, 2, nu)
    wp.atomic_min(box, img, 3, nv)


@wp.kernel
def _splat_bodies(body_pos: wp.array2d(dtype=wp.vec3), body_quat: wp.array2d(dtype=wp.vec4),
                  pool_body: wp.array(dtype=wp.int32), pool_off: wp.array(dtype=wp.vec3),
                  cams: wp.array2d(dtype=wp.vec3), rots: wp.array2d(dtype=wp.mat33),
                  f: float, w: int, h: int, min_depth: float, box: wp.array2d(dtype=wp.int32),
                  zbuf: wp.array2d(dtype=float)):
    """The robot's surface samples, moved by their bodies' poses in a register (wxyz quaternions).
    Written only where a tested point can read them: within one pixel of ``box`` (``_pixel_box``)."""
    e, c, p = wp.tid()
    b = pool_body[p]
    qv = body_quat[e, b]
    q = wp.quat(qv[1], qv[2], qv[3], qv[0])                  # warp quats are xyzw
    x = body_pos[e, b] + wp.quat_rotate(q, pool_off[p])
    u, v, z, ok = _project(x, cams[e, c], rots[e, c], f, w, h, min_depth)
    if ok:
        img = e * 2 + c
        if u >= box[img, 0] - 1 and v >= box[img, 1] - 1 and u <= 1 - box[img, 2] and v <= 1 - box[img, 3]:
            wp.atomic_min(zbuf, img, v * w + u, z)


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


@wp.kernel
def _test_dilated(pts: wp.array2d(dtype=wp.vec3), cams: wp.array2d(dtype=wp.vec3), rots: wp.array2d(dtype=wp.mat33),
                  f: float, w: int, h: int, min_depth: float, tol: float, zbuf: wp.array2d(dtype=float),
                  out: wp.array3d(dtype=wp.uint8)):
    """``_dilate`` + ``_test`` for the tested points only: the 3x3 minimum is taken at each point's own
    pixel instead of at every pixel of every image."""
    e, c, p = wp.tid()
    u, v, z, ok = _project(pts[e, p], cams[e, c], rots[e, c], f, w, h, min_depth)
    vis = wp.uint8(0)
    if ok:
        img = e * 2 + c
        best = float(_INF)
        for dv in range(-1, 2):
            for du in range(-1, 2):
                uu = u + du
                vv = v + dv
                if uu >= 0 and uu < w and vv >= 0 and vv < h:
                    best = wp.min(best, zbuf[img, vv * w + uu])
        if z <= best + tol:
            vis = wp.uint8(1)
    out[e, c, p] = vis


def visible_warp(pts: torch.Tensor, cams: torch.Tensor, rots: torch.Tensor, f: float, w: int, h: int,
                 min_depth: float, depth_tol: float, test_upto: int | None = None,
                 body_pos: torch.Tensor | None = None, body_quat: torch.Tensor | None = None,
                 pool_body: torch.Tensor | None = None, pool_off: torch.Tensor | None = None,
                 hull: torch.Tensor | None = None) -> torch.Tensor:
    """``(n, 2, P)`` bool, as ``PointCloud.visible``. Every point occludes; only the first ``test_upto``
    are tested (the rest come back False) -- a cloud that keeps only the manipuland skips the robot's.

    With ``body_pos`` ``(n, B, 3)`` / ``body_quat`` ``(n, B, 4)`` wxyz and the robot's surface pool
    (``pool_body`` ``(R,)``, ``pool_off`` ``(R, 3)``), the robot occludes too without ever being built
    as an ``(n, R, 3)`` tensor (it is not tested). ``hull`` ``(n, H, 3)``: points whose convex hull holds
    every ``pts`` (the sheet's particles for samples interpolated on it); robot points that project
    outside its pixel rectangle cannot hide a tested point and are skipped (same result, fewer writes)."""
    wp.init()                                   # no-op once initialised (Kit initialises it; a plain process may not)
    n, p = int(pts.shape[0]), int(pts.shape[1])
    q = p if test_upto is None else int(test_upto)
    dev = pts.device
    pts = pts.detach().float().contiguous()
    cams_c = cams.detach().float().contiguous()
    rots_c = rots.detach().float().contiguous()
    zbuf = torch.full((n * 2, h * w), _INF, device=dev, dtype=torch.float32)
    out = torch.zeros((n, 2, p), device=dev, dtype=torch.uint8)
    stream = wp.stream_from_torch(torch.cuda.current_stream(dev))
    wd = wp.device_from_torch(dev)
    a_pts = wp.from_torch(pts, dtype=wp.vec3)
    a_cams = wp.from_torch(cams_c, dtype=wp.vec3)
    a_rots = wp.from_torch(rots_c, dtype=wp.mat33)
    a_z = wp.from_torch(zbuf)
    wp.launch(_splat, dim=(n, 2, p), inputs=[a_pts, a_cams, a_rots, float(f), int(w), int(h), float(min_depth), a_z],
              device=wd, stream=stream)
    if body_pos is not None:
        # The robot's points are never tested here, so only those within a pixel of a tested point's
        # can change a verdict: they are written inside the pixel rectangle of ``hull`` only (the whole
        # image without a hull, or when not every point is tested).
        box = torch.tensor([0, 0, 1 - w, 1 - h], device=dev, dtype=torch.int32).repeat(n * 2, 1)
        a_box = wp.from_torch(box, dtype=wp.int32)
        if CULL_BODIES and q == p and hull is not None:
            box[:] = torch.tensor([w + 8, h + 8, 8, 8], device=dev, dtype=torch.int32)
            hull_c = hull.detach().float().contiguous()
            wp.launch(_pixel_box, dim=(n, 2, int(hull_c.shape[1])),
                      inputs=[wp.from_torch(hull_c, dtype=wp.vec3), a_cams, a_rots, float(f), int(w), int(h), a_box],
                      device=wd, stream=stream)
        bp = body_pos.detach().float().contiguous()
        bq = body_quat.detach().float().contiguous()
        wp.launch(_splat_bodies, dim=(n, 2, int(pool_body.numel())),
                  inputs=[wp.from_torch(bp, dtype=wp.vec3), wp.from_torch(bq, dtype=wp.vec4),
                          wp.from_torch(pool_body.to(torch.int32).contiguous(), dtype=wp.int32),
                          wp.from_torch(pool_off.float().contiguous(), dtype=wp.vec3),
                          a_cams, a_rots, float(f), int(w), int(h), float(min_depth), a_box, a_z],
                  device=wd, stream=stream)
    if q == 0:
        return out.bool()
    a_out = wp.from_torch(out[:, :, :q] if q == p else out, dtype=wp.uint8)
    if FUSE_DILATION and 9 * q < h * w:
        # Fewer tested points than pixels: the 3x3 minimum at each point's pixel only (same result).
        wp.launch(_test_dilated, dim=(n, 2, q),
                  inputs=[a_pts, a_cams, a_rots, float(f), int(w), int(h), float(min_depth), float(depth_tol), a_z, a_out],
                  device=wd, stream=stream)
    else:
        near = torch.empty_like(zbuf)
        a_near = wp.from_torch(near)
        wp.launch(_dilate, dim=(n * 2, h * w), inputs=[a_z, int(w), int(h), a_near], device=wd, stream=stream)
        wp.launch(_test, dim=(n, 2, q),
                  inputs=[a_pts, a_cams, a_rots, float(f), int(w), int(h), float(min_depth), float(depth_tol), a_near, a_out],
                  device=wd, stream=stream)
    return out.bool()
