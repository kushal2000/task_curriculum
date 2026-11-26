"""Monkeypatches for the observation audit (none applied unless named).

bodyquat   ClothEnv.sim_state's ``body_quat_w`` is read raw from Isaac Lab 3.0 (``data.body_quat_w``), which
           the Newton data proxy does NOT convert (backends/vbd/newton_support/patches.py QUAT_FIELDS lists
           body_state_w, root_* only) -> xyzw; the point cloud reads it as wxyz for the robot's link poses
           (perception/pointcloud_obs.py:106), i.e. the occluding hand is mis-rotated. The patch takes the
           quaternion from ``body_state_w[..., 3:7]``, which the proxy does convert to wxyz.
"""

import torch


def apply(name: str, env_cfg) -> None:
    if name not in ("none", "", "bodyquat"):
        raise ValueError(f"unknown patch {name!r}")


def apply_env(name: str, inner) -> None:
    if name != "bodyquat":
        return
    original = inner.sim_state

    def sim_state_fixed():
        s = original()
        s.body_quat_w = s.body_state_w[..., 3:7].clone()
        return s

    inner.sim_state = sim_state_fixed
    print("[audit] patch bodyquat: sim_state.body_quat_w <- body_state_w[..., 3:7] (wxyz)", flush=True)


def diagnose(inner) -> dict:
    """The raw body_quat_w vs body_state_w's quaternion, and how many cloth candidates change visibility
    when the occluder uses one or the other, for the current state."""
    from isaacsimenvs.tasks.cloth.mdp.robot import to_torch

    data = inner.robot.data
    q_raw = to_torch(data.body_quat_w)
    q_state = to_torch(data.body_state_w)[..., 3:7]
    as_xyzw = torch.cat((q_state[..., 1:4], q_state[..., 0:1]), dim=-1)
    out = {
        "max_abs_raw_minus_state": float((q_raw - q_state).abs().max()),
        "max_abs_raw_minus_state_as_xyzw": float((q_raw - as_xyzw).abs().max()),
        "palm_raw": q_raw[0, inner.robot_layout.palm_body_id].tolist(),
        "palm_state_wxyz": q_state[0, inner.robot_layout.palm_body_id].tolist(),
    }
    # Visibility of the same cloth candidates with the two occluder poses.
    cloud = inner.observations.pointcloud.cloud
    origins = inner.scene.env_origins
    pts_w, _ = inner.backend.read_points()
    parts = pts_w - origins.unsqueeze(1)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    cloth, _, _ = cloud.cloth_candidates(parts)
    body_pos = to_torch(data.body_pos_w) - origins.unsqueeze(1)
    from isaacsimenvs.tasks.cloth.perception.visibility_warp import visible_warp

    res = {}
    for tag, q in (("raw", q_raw), ("fixed", q_state)):
        if cloud.use_warp:
            vis = visible_warp(cloth, cloud.cams, cloud.rots, cloud.f, cloud.w, cloud.h, float(cloud.cfg.min_depth),
                               float(cloud.cfg.depth_tol), body_pos=body_pos, body_quat=q, pool_body=cloud.pool_body,
                               pool_off=cloud.pool_off, hull=parts)
        else:
            pts = torch.cat((cloth, cloud.robot_points(body_pos, q)), dim=1)
            vis = cloud.visible(pts, cloud.cams, cloud.rots)[:, :, : cloth.shape[1]]
        res[tag] = (vis[:, 0] | vis[:, 1])
    out["visible_frac_raw"] = res["raw"].float().mean(1).tolist()
    out["visible_frac_fixed"] = res["fixed"].float().mean(1).tolist()
    out["visibility_disagree_frac"] = (res["raw"] != res["fixed"]).float().mean(1).tolist()
    # Where the robot surface samples land (fingertip link) under the two quaternions.
    rp_raw = cloud.robot_points(body_pos, q_raw)
    rp_fix = cloud.robot_points(body_pos, q_state)
    out["robot_surface_point_shift_mm_median"] = float((rp_raw - rp_fix).norm(dim=-1).median()) * 1e3
    out["robot_surface_point_shift_mm_max"] = float((rp_raw - rp_fix).norm(dim=-1).max()) * 1e3
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    return out
