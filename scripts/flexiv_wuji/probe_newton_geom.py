"""Newton's ring / pinky collision geometry at the reset pose (hand at 0), vs the URDF's.

    scripts/newton_py scripts/flexiv_wuji/probe_newton_geom.py \
        --task Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0 'env.assets.handle_head_types=[hammer]'

No stepping. For world 0's ring and pinky shapes:
  1. Newton's own geometry: shape type, scale, margin, gap, vertex count; world-space convex hulls
     from shape_source vertices x scale, shape_transform, body_q; ring-pinky signed distance
     (+ overlap / - gap, mm). The URDF's hulls give -2.3 mm at this pose (closest pair, proximal).
  2. What MJWarp collides: each shape's MuJoCo geom type, size, margin, gap, mesh vertex count.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0")
parser.add_argument("--out", default=None)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
try:
    app = AppLauncher(args_cli).app
except ImportError as exc:
    app = None
    print(f"[newton_geom] kit-less: {exc}", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.scripts.evaluation.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml  # noqa: E402


def _quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _tf(t):
    t = np.asarray(t, dtype=float)
    return t[:3], _quat_xyzw_to_mat(t[3:7])


def _signed(a, b):
    import trimesh

    best = -np.inf
    for ma, mb in itertools.product(a, b):
        for p, q in ((ma, mb), (mb, ma)):
            np.random.seed(0)
            pts = np.vstack([p.vertices, p.sample(400)])
            best = max(best, float(trimesh.proximity.signed_distance(q, pts).max()))
    return best


@hydra_task_config_with_yaml(args_cli.task, "")
def run(env_cfg, agent_cfg) -> None:
    import trimesh
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonManager

    env_cfg.scene.num_envs = 2
    env_cfg.seed = 0
    env_cfg.assets.num_assets_per_type = 1
    disable_randomization(env_cfg)
    use_single_object_variant()
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    model = NewtonManager.get_model()
    state = NewtonManager._state_0 if hasattr(NewtonManager, "_state_0") else None
    body_q = (state.body_q if state is not None else model.body_q).numpy()
    labels = [lab or "" for lab in model.body_label]
    shape_labels = [lab or "" for lab in getattr(model, "shape_label", [])]
    sb = model.shape_body.numpy()
    stype = model.shape_type.numpy()
    sscale = model.shape_scale.numpy()
    stf = model.shape_transform.numpy()
    smargin = model.shape_margin.numpy() if getattr(model, "shape_margin", None) is not None else None
    sgap = model.shape_gap.numpy() if getattr(model, "shape_gap", None) is not None else None

    rep = {"bodies": {}, "shapes": []}
    hulls = {}
    for s in range(len(sb)):
        b = int(sb[s])
        if b < 0 or "env_0/Robot/r_" not in labels[b] or not any(k in labels[b] for k in ("ring", "pinky")):
            continue
        name = labels[b].rsplit("/", 1)[-1]
        src = model.shape_source[s]
        verts = np.asarray(getattr(src, "vertices", np.zeros((0, 3))), dtype=float)
        row = {"shape": s, "label": shape_labels[s] if s < len(shape_labels) else "", "body": name,
               "type": int(stype[s]), "scale": sscale[s].round(5).tolist(), "n_verts": int(len(verts)),
               "margin": None if smargin is None else float(smargin[s]),
               "gap": None if sgap is None else float(sgap[s]),
               "local_pos": stf[s][:3].round(5).tolist()}
        if len(verts):
            pb, Rb = _tf(body_q[b])
            ps, Rs = _tf(stf[s])
            w = (Rb @ (Rs @ (verts * sscale[s]).T + ps[:, None])).T + pb
            hull = trimesh.Trimesh(w).convex_hull
            hulls.setdefault(name, []).append(hull)
            row["world_extent_mm"] = (1e3 * (w.max(0) - w.min(0))).round(2).tolist()
        rep["shapes"].append(row)

    ring = [n for n in hulls if "ring" in n]
    pinky = [n for n in hulls if "pinky" in n]
    pairs = sorted(((_signed(hulls[a], hulls[b]), a, b) for a in ring for b in pinky), reverse=True)
    rep["newton_ring_pinky_signed_mm"] = [[round(1e3 * d, 2), a, b] for d, a, b in pairs[:6]]
    print("[newton_geom] Newton geometry ring-pinky (+ overlap / - gap mm):", json.dumps(rep["newton_ring_pinky_signed_mm"]), flush=True)
    for row in rep["shapes"]:
        print("[newton_geom] shape", json.dumps(row), flush=True)

    try:
        solver = NewtonManager._solver
        entries = getattr(solver, "_entries", None)
        rigid = entries["rigid"].solver if entries else solver
        mjm = rigid.mjw_model
        g2s = wp.to_torch(rigid.mjc_geom_to_newton_shape).cpu().numpy()[0]
        def arr(x):
            a = np.asarray(x.numpy())
            return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a

        gtype, gsize, gmargin, ggap, gdata = (arr(getattr(mjm, k)) for k in
                                              ("geom_type", "geom_size", "geom_margin", "geom_gap", "geom_dataid"))
        mvnum = arr(mjm.mesh_vertnum) if hasattr(mjm, "mesh_vertnum") else None
        mj = []
        shapes = {row["shape"]: row["body"] for row in rep["shapes"]}
        for g, s in enumerate(g2s):
            if int(s) in shapes:
                gm, gg, gs = gmargin[g], ggap[g], gsize[g]
                mj.append({"geom": g, "shape": int(s), "body": shapes[int(s)], "type": int(gtype[g]),
                           "size": np.asarray(gs).round(5).tolist(), "margin": float(gm), "gap": float(gg),
                           "mesh_verts": None if mvnum is None or gdata[g] < 0 else int(mvnum[gdata[g]])})
        rep["mujoco_geoms"] = mj
        for row in mj:
            print("[newton_geom] mjgeom", json.dumps(row), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        rep["mujoco_geoms"] = f"failed: {e}"
    # ---- settle at pose A (flexion = spring rest + 0.5 rad, PD only, no joint friction), then compare
    import torch

    inner = env.unwrapped
    robot, device = inner.robot, inner.device

    def _t(x):
        v = getattr(x, "torch", None)
        return x if v is None else (v() if callable(v) else v)

    names = list(robot.data.joint_names)
    hand = [int(i) for i in inner._hand_joint_ids]
    spec = inner.robot_spec
    inner._spring_ks = None
    robot.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=torch.zeros(inner.num_envs, len(hand), device=device), joint_ids=hand)
    lo, hi = (_t(robot.data.soft_joint_pos_limits)[0, :, k] for k in (0, 1))
    ref = torch.tensor([float(spec.hand_spring_ref.get(n, 0.0)) for n in names], device=device)
    flex = [i for i in hand if ("flex" in names[i] or "pip" in names[i] or "dip" in names[i]
                                 or names[i].endswith("_mcp") or names[i].endswith("_ip"))]
    a = _t(robot.data.default_joint_pos).clone()
    a[:, flex] = (ref[flex] + 0.5).unsqueeze(0)
    a[:, hand] = torch.clamp(a[:, hand], lo[hand] + 0.02, hi[hand] - 0.02)
    robot.write_joint_state_to_sim(a, torch.zeros_like(a))
    dt = float(inner.cfg.sim.dt)
    for _ in range(300):
        robot.set_joint_position_target(a)
        robot.set_joint_effort_target(torch.zeros_like(a))
        robot.write_data_to_sim()
        inner.sim.step(render=False)
        inner.scene.update(dt)
    st = NewtonManager._state_0
    bq = st.body_q.numpy()
    shape_rows = [r for r in rep["shapes"] if r["n_verts"]]
    posed = {}
    for r in shape_rows:
        s_ = r["shape"]
        b = int(sb[s_])
        verts = np.asarray(model.shape_source[s_].vertices, dtype=float)
        pb, Rb = _tf(bq[b])
        ps, Rs = _tf(stf[s_])
        w = (Rb @ (Rs @ (verts * sscale[s_]).T + ps[:, None])).T + pb
        posed.setdefault(r["body"], []).append(trimesh.Trimesh(w).convex_hull)
    exact = {f"{x} <-> {y}": round(1e3 * _signed(posed[x], posed[y]), 2)
             for x in posed if "ring" in x for y in posed if "pinky" in y}
    rep["settled_A"] = {"q_minus_target_mrad": {names[i]: round(1e3 * float((_t(robot.data.joint_pos) - a)[0, i]), 1)
                                                for i in hand if "ring" in names[i] or "pinky" in names[i]},
                        "exact_hull_signed_mm": dict(sorted(exact.items(), key=lambda kv: -kv[1])[:8])}
    try:
        solver = NewtonManager._solver
        entries = getattr(solver, "_entries", None)
        rigid = entries["rigid"].solver if entries else solver
        mjd = rigid.mjw_data
        nacon = int(wp.to_torch(mjd.nacon)[0])
        geom = wp.to_torch(mjd.contact.geom).long()[:nacon].cpu().numpy()
        world = wp.to_torch(mjd.contact.worldid).long()[:nacon].cpu().numpy()
        dist = wp.to_torch(mjd.contact.dist)[:nacon].cpu().numpy()
        g2s_all = wp.to_torch(rigid.mjc_geom_to_newton_shape).long().cpu().numpy()
        reported = {}
        for k in range(nacon):
            if world[k] != 0:
                continue
            bs = [labels[int(sb[g2s_all[0, geom[k, j]]])].rsplit("/", 1)[-1] if geom[k, j] >= 0 and g2s_all[0, geom[k, j]] >= 0 else "static"
                  for j in (0, 1)]
            if not (any("ring" in b for b in bs) and any("pinky" in b for b in bs)):
                continue
            key = " <-> ".join(sorted(bs, key=lambda b: "pinky" in b))
            reported[key] = round(min(reported.get(key, 1e9), 1e3 * float(dist[k])), 2)
        rep["settled_A"]["reported_contact_min_dist_mm"] = reported
    except Exception as e:  # noqa: BLE001
        rep["settled_A"]["reported_contact_min_dist_mm"] = f"failed: {e}"
    print("[newton_geom] settled A", json.dumps(rep["settled_A"]), flush=True)

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(rep, f, indent=1)
    import os

    os._exit(0)


if __name__ == "__main__":
    run()
