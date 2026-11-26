"""Compare captured observations (capture.py) of the plate and VBD worlds: numbers to JSON, plots to PNG.

    python experiments/nov25_better_rigidcloth/obs_audit/analyze.py outputs/obs_audit/plates_pinned.pt \
        outputs/obs_audit/vbd_pinned.pt --tag pinned
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COL = {"live": "#2b6cb0", "goal": "#dd6b20", "grid": "#555555"}


def cloud_stats(xyz, mask, table_z, yaw=None, ref_c=None):
    p = xyz[mask].numpy()
    if len(p) == 0:
        return {"n": 0}
    c = p.mean(0)
    out = {"n": int(len(p)), "centroid": c.round(4).tolist(),
           "x_extent": [round(float(p[:, 0].min()), 4), round(float(p[:, 0].max()), 4)],
           "y_extent": [round(float(p[:, 1].min()), 4), round(float(p[:, 1].max()), 4)],
           "z_above_table_median_mm": round(float(np.median(p[:, 2] - table_z)) * 1e3, 2),
           "z_above_table_p10_p90_mm": [round(float(np.percentile(p[:, 2] - table_z, q)) * 1e3, 2) for q in (10, 90)]}
    if yaw is not None and ref_c is not None:   # in the sheet's own (stationary-frame) axes, about the sheet centre
        cy, sy = math.cos(-yaw), math.sin(-yaw)
        d = p[:, :2] - np.asarray(ref_c[:2])
        loc = np.stack((cy * d[:, 0] - sy * d[:, 1], sy * d[:, 0] + cy * d[:, 1]), 1)
        out["sheet_frame_centroid"] = loc.mean(0).round(4).tolist()
        out["sheet_frame_x_extent"] = [round(float(loc[:, 0].min()), 4), round(float(loc[:, 0].max()), 4)]
    return out


def proprio_split(meta, vec):
    out, at = {}, 0
    for name, w in meta["proprio_fields"]:
        out[name] = vec[at:at + w]
        at += w
    return out


def teacher_split(meta, vec):
    out, at = {}, 0
    for name, w in meta["teacher_fields"]:
        out[name] = vec[at:at + w]
        at += w
    return out


def frozen_index(rec):
    for i, r in enumerate(rec):
        if bool(r["frozen"][0]) and int(r["mask"][0, 64:].sum()) > 0:
            return i
    return None


def summarise(d, env=0):
    m, rec = d["meta"], d["rec"]
    tz = float(m["table_z_cloud"])
    fi = frozen_index(rec)
    r0, rf, rl = rec[0], rec[fi], rec[-1]
    P0 = r0["points"][env]
    mov, st = m["moving_idx"], m["stationary_idx"]
    out = {
        "world": m["world"], "rand": m["rand"], "patch": m["patch"],
        "config": {k: (m[k].tolist() if torch.is_tensor(m[k]) else m[k]) for k in (
            "robot_root_pos", "hand_sysid", "table_reset_z", "table_half", "start_height", "success", "surface_layered",
            "surface_ref_step", "keypoint_tolerance", "footprint_max", "fold_lift", "table_z_cloud", "scale",
            "keypoint_idx", "decimation", "dt", "episode_length_s")},
        "centre": m["centre"].tolist(), "cams": m["cams"].tolist(),
        "dr": {"pointcloud": m["dr_pc"], "proprio": m["dr_proprio"], "action": m["dr_action"], "robot": m["dr_robot"]},
        "obs_pc": m["obs_pc"], "layout": m["layout"],
        "goal_tasks": None if m["goal_tasks"] is None else {"names": m["goal_tasks"]["names"],
                                                            "kp": m["goal_tasks"]["kp"].tolist()},
        "object_scale_env0": None if m["object_scale"] is None else m["object_scale"][env].tolist(),
        "frozen_step_record": fi, "frozen_episode_step": rf["step"],
        "folded_first_episode": int(m["folded_first_episode"].sum()), "num_envs": int(m["folded_first_episode"].numel()),
    }
    yaw0 = float(r0["frame_yaw"][env])
    out["reset"] = {
        "stationary_frame_yaw_deg": round(math.degrees(yaw0), 2),
        "sheet_centre": P0.mean(0).numpy().round(4).tolist(),
        "moving_half_centre": P0[mov].mean(0).numpy().round(4).tolist(),
        "stationary_half_centre": P0[st].mean(0).numpy().round(4).tolist(),
        "grid_z_above_table_mm": round(float(P0[:, 2].mean() - tz) * 1e3, 2),
        "grid_point_0": P0[0].numpy().round(4).tolist(), "grid_point_6": P0[6].numpy().round(4).tolist(),
        "grid_point_42": P0[42].numpy().round(4).tolist(), "grid_point_48": P0[48].numpy().round(4).tolist(),
        "keypoints": r0["keypoints"][env].numpy().round(4).tolist(),
    }
    Pf = rf["points"][env]
    yawf = float(rf["frame_yaw"][env])
    live = cloud_stats(rf["xyz"][env, :64], rf["mask"][env, :64], tz, yawf, Pf.mean(0).numpy())
    goal = cloud_stats(rf["xyz"][env, 64:], rf["mask"][env, 64:], tz, yawf, Pf.mean(0).numpy())
    G = rf["goal_ref"][env]
    # Goal's yaw relative to the live sheet: the goal grid's stationary part vs the sheet's stationary part.
    gst = G[st] - G[st].mean(0)
    pst = Pf[st] - Pf[st].mean(0)
    ang = math.degrees(math.atan2(float((pst[:, 0] * gst[:, 1] - pst[:, 1] * gst[:, 0]).sum()),
                                  float((pst[:, :2] * gst[:, :2]).sum())))
    out["frozen"] = {
        "live_cloud": live, "goal_cloud": goal,
        "goal_grid_moving_half_centre": G[mov].mean(0).numpy().round(4).tolist(),
        "goal_grid_stationary_half_centre": G[st].mean(0).numpy().round(4).tolist(),
        "sheet_stationary_half_centre": Pf[st].mean(0).numpy().round(4).tolist(),
        "sheet_moving_half_centre": Pf[mov].mean(0).numpy().round(4).tolist(),
        "goal_moving_half_z_above_table_mm": round(float(G[mov, 2].mean() - tz) * 1e3, 2),
        "goal_stationary_half_z_above_table_mm": round(float(G[st, 2].mean() - tz) * 1e3, 2),
        "goal_stationary_vs_sheet_stationary_xy_mm": round(float((G[st, :2] - Pf[st, :2]).norm(dim=-1).max()) * 1e3, 2),
        "goal_yaw_vs_sheet_deg": round(ang, 2),
        "goal_moving_half_lands_on": "stationary half" if float((G[mov].mean(0) - Pf[st].mean(0))[:2].norm()) < 0.01 else "elsewhere",
        "fold_error": round(float(rf["fold_error"][env]), 4), "layered_dist": None if rf["layered"] is None else round(float(rf["layered"][env]), 4),
        "aligned_dist": round(float(rf["aligned"][env]), 4), "in_place_dist": round(float(rf["in_place"][env]), 4),
        "footprint": round(float(rf["footprint"][env]), 4),
    }
    # The policy phase: best signals per env.
    zs = sum(1 for r in rec if r["step"] <= rec[fi]["step"] + 7)  # not used; zero phase is the first records
    fe = torch.stack([r["fold_error"] for r in rec])
    lay = torch.stack([r["layered"] if r["layered"] is not None else r["aligned"] for r in rec])
    out["episode"] = {
        "best_fold_error_per_env": fe.min(0).values.numpy().round(4).tolist(),
        "best_layered_dist_per_env": lay.min(0).values.numpy().round(4).tolist(),
        "final_sheet_centre_env0": rl["points"][env].mean(0).numpy().round(4).tolist(),
        "final_frame_yaw_deg_env0": round(math.degrees(float(rl["frame_yaw"][env])), 2),
        "live_n_valid_mean_over_steps": round(float(torch.stack([r["mask"][:, :64].float().sum(1) for r in rec]).mean()), 2),
        "live_z_above_table_median_mm_over_policy_phase": round(float(np.median(np.concatenate(
            [(r["xyz"][:, :64][r["mask"][:, :64]][:, 2] - tz).numpy() for r in rec[20:]]))) * 1e3, 2),
    }
    t = teacher_split(m, rf["teacher"][env])
    out["teacher_at_frozen"] = {k: v.numpy().round(4).tolist() for k, v in t.items() if k in ("object_rot", "object_scales", "keypoints_rel_goal")}
    out["proprio_at_record0"] = {k: v.numpy().round(4).tolist() for k, v in proprio_split(m, r0["policy"][env]).items()
                                 if k in ("palm_pos", "palm_rot")}
    return out


def plot_pair(ds, names, idx_fn, title, path, env=0):
    fig, axes = plt.subplots(1, len(ds), figsize=(6 * len(ds), 5.6), squeeze=False)
    for ax, d, name in zip(axes[0], ds, names):
        rec = d["rec"]
        i = idx_fn(rec)
        r = rec[i]
        live, lm = r["xyz"][env, :64], r["mask"][env, :64]
        goal, gm = r["xyz"][env, 64:], r["mask"][env, 64:]
        P = r["points"][env]
        mv = d["meta"]["moving_idx"]
        ax.scatter(P[:, 0], P[:, 1], s=14, facecolors="none", edgecolors=COL["grid"], lw=0.7, label="sheet grid (49)")
        ax.scatter(P[mv, 0], P[mv, 1], s=4, c=COL["grid"], label="moving half (grid)")
        ax.scatter(live[lm, 0], live[lm, 1], s=10, c=COL["live"], label=f"live cloud ({int(lm.sum())})")
        ax.scatter(goal[gm, 0], goal[gm, 1], s=10, c=COL["goal"], marker="x", label=f"goal cloud ({int(gm.sum())})")
        for k in (0, 6, 42, 48):
            ax.annotate(str(k), P[k, :2].numpy(), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_title(f"{name}  (episode step {r['step']}, yaw {math.degrees(float(r['frame_yaw'][env])):+.0f} deg)", fontsize=10)
        ax.set_xlabel("env x [m]")
        ax.set_ylabel("env y [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_side(ds, names, idx_fn, path, env=0):
    fig, axes = plt.subplots(1, len(ds), figsize=(6 * len(ds), 3.6), squeeze=False)
    for ax, d, name in zip(axes[0], ds, names):
        rec, tz = d["rec"], float(d["meta"]["table_z_cloud"])
        r = rec[idx_fn(rec)]
        live, lm = r["xyz"][env, :64], r["mask"][env, :64]
        goal, gm = r["xyz"][env, 64:], r["mask"][env, 64:]
        P = r["points"][env]
        ax.scatter(P[:, 0], (P[:, 2] - tz) * 1e3, s=14, facecolors="none", edgecolors=COL["grid"], label="grid")
        ax.scatter(live[lm, 0], (live[lm, 2] - tz) * 1e3, s=10, c=COL["live"], label="live")
        ax.scatter(goal[gm, 0], (goal[gm, 2] - tz) * 1e3, s=10, c=COL["goal"], marker="x", label="goal")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name}: height above table top (step {r['step']})", fontsize=10)
        ax.set_xlabel("env x [m]")
        ax.set_ylabel("z - table [mm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_signals(ds, names, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for d, name, ls in zip(ds, names, ("-", "--", ":", "-.")):
        rec = d["rec"]
        st = [r["step"] for r in rec]
        for key, ax in (("fold_error", axes[0]), ("layered", axes[1])):
            v = torch.stack([r[key] if r[key] is not None else r["aligned"] for r in rec]).clamp(max=0.2).numpy()
            ax.plot(st, v.mean(1), ls, label=name)
            ax.set_title(f"{key} (env mean, clamp 0.2)")
        n = [r["mask"][:, :64].float().sum(1).mean() for r in rec]
        axes[2].plot(st, n, ls, label=name)
        axes[2].set_title("valid live points (env mean)")
    for ax in axes:
        ax.set_xlabel("episode step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def proprio_diff(da, db):
    """Per proprio field, max |a - b| at the first record (pinned starts: the same robot state)."""
    ra, rb = da["rec"][0]["policy"], db["rec"][0]["policy"]
    pa, pb = proprio_split(da["meta"], ra[0]), proprio_split(db["meta"], rb[0])
    return {k: round(float((pa[k] - pb[k]).abs().max()), 5) for k in pa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--tag", default="pinned")
    ap.add_argument("--outdir", default="outputs/obs_audit")
    a = ap.parse_args()
    ds = [torch.load(f, weights_only=False) for f in a.files]
    names = a.names or [f"{d['meta']['world']}{'' if d['meta']['patch'] == 'none' else '+' + d['meta']['patch']}" for d in ds]
    res = {"files": a.files, "summaries": {n: summarise(d) for n, d in zip(names, ds)}}
    if len(ds) >= 2:
        res["proprio_max_abs_diff_record0"] = proprio_diff(ds[0], ds[1])
    plot_pair(ds, names, frozen_index, f"top-down at the frozen goal step ({a.tag}, env 0)", f"{a.outdir}/topdown_frozen_{a.tag}.png")
    plot_pair(ds, names, lambda rec: 0, f"top-down at the first observation ({a.tag}, env 0)", f"{a.outdir}/topdown_reset_{a.tag}.png")
    plot_pair(ds, names, lambda rec: len(rec) - 1, f"top-down at the last captured step ({a.tag}, env 0)", f"{a.outdir}/topdown_last_{a.tag}.png")
    plot_side(ds, names, frozen_index, f"{a.outdir}/side_frozen_{a.tag}.png")
    plot_signals(ds, names, f"{a.outdir}/signals_{a.tag}.png")
    with open(f"{a.outdir}/audit_{a.tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:12000])


if __name__ == "__main__":
    main()
