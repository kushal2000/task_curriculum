"""Measure interpenetration during policy rollouts of one rigid cloth design.

    .venv/isaacsim/bin/python experiments/nov18/cloth_penetration/measure.py --headless \\
        --config <run>/.hydra/config.yaml --checkpoint <ckpt> --num_envs 64 --design strip_4 --out x.json

Env and policy are built by eval_cloth.build (same settings as the evaluation), restricted to one
design. Over each env's FIRST episode, per policy step (the last physics substep):

* panel-panel: exact box separating-axis overlap depth for every panel pair, split into hinge
  neighbours (collision disabled by design) and non-neighbours (collision enabled);
* panel-table: depth of the lowest panel corner below the table top, and whether a whole panel
  is below it (a pass-through);
* cloth-robot: PhysX's own minimum contact separation (negative = penetration);
* hinge angles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from isaacsimenvs.scripts.evaluation.eval_cloth import add_common_args, build, launch  # noqa: E402


def sat_penetration(ca, ra, ha, cb, rb, hb):
    """Overlap depth of oriented boxes (centers c, rotation matrices r with box axes as columns,
    half extents h), batched over leading dims. 0 when separated."""
    import torch

    axes = [ra[..., :, k] for k in range(3)] + [rb[..., :, k] for k in range(3)]
    for i in range(3):
        for j in range(3):
            axes.append(torch.linalg.cross(ra[..., :, i], rb[..., :, j]))
    t = cb - ca
    best = torch.full(t.shape[:-1], float("inf"), device=t.device)
    for axis in axes:
        norm = axis.norm(dim=-1, keepdim=True)
        valid = norm[..., 0] > 1e-6
        axis = axis / norm.clamp_min(1e-9)
        proj_a = (ha * (ra.transpose(-1, -2) @ axis[..., None])[..., 0].abs()).sum(-1)
        proj_b = (hb * (rb.transpose(-1, -2) @ axis[..., None])[..., 0].abs()).sum(-1)
        overlap = proj_a + proj_b - (t * axis).sum(-1).abs()
        best = torch.where(valid, torch.minimum(best, overlap), best)
    return best.clamp_min(0.0)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser, num_envs=64)
    parser.add_argument("--design", default="strip_4")
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.overrides = [f"env.rigid_cloth.designs=[{args.design}.yaml]", *args.overrides]
    app = launch(args)

    import torch
    from isaacsimenvs.tasks.cloth.mdp.geometry import rotations
    from isaacsimenvs.tasks.cloth.backends.rigid.contacts import contact_indices

    b = build(args)
    env, inner = b.env, b.inner
    n, dev = inner.num_envs, inner.device
    (group,) = inner.backend.groups
    design = group.design
    if list(group.env_ids.cpu().tolist()) != list(range(n)):
        raise RuntimeError("expected one design group covering every env in order")
    plates = [p.name for p in design.plates]
    P = len(plates)
    neighbours = {frozenset((h.parent, h.child)) for h in design.hinges}
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    is_neighbour = torch.tensor([frozenset((plates[i], plates[j])) in neighbours for i, j in pairs], device=dev)

    # Per env, per plate: collider box center in the body frame and half extents.
    center_local = torch.zeros(n, P, 3, device=dev)
    half = torch.zeros(n, P, 3, device=dev)
    for e in range(n):
        spec = design.resolve(*b.dims[e])
        for k, plate in enumerate(spec.plates):
            if len(plate.vertices) != 4:
                raise RuntimeError("only box panels are supported")
            xs, ys = zip(*plate.vertices)
            center_local[e, k] = torch.tensor([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.0])
            half[e, k] = torch.tensor(plate.half_extents)

    H = len(design.hinges)
    hinge_a = torch.zeros(n, H, 2, device=dev)
    hinge_u = torch.zeros(n, H, 2, device=dev)
    for e in range(n):
        for h, hinge in enumerate(design.resolve(*b.dims[e]).hinges):
            hinge_a[e, h] = torch.tensor(hinge.anchor[:2])
            u = torch.tensor(hinge.axis[:2])
            hinge_u[e, h] = u / u.norm()
    link_names = list(inner.robot.body_names)
    series = {"robot_depth": [], "hinge_dist": [], "z_frac": [], "link": [], "plate": []}

    rows = inner.backend._env_local  # env -> row of the group's articulation (rows are not env-ordered)
    guard = inner._episode_guard
    contacts = getattr(guard, "contacts", None) if guard is not None else None

    worst = {k: torch.zeros(n, device=dev) for k in ("neighbour", "non_neighbour", "table", "robot")}
    passthrough = torch.zeros(n, dtype=torch.bool, device=dev)
    hinge_min = torch.full((n,), float("inf"), device=dev)
    hinge_max = torch.full((n,), -float("inf"), device=dev)
    events = []  # (depth, kind, env, step)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    success = torch.zeros_like(done)

    def measure(step):
        live = ~done
        art = group.articulation.data
        pos = art.body_link_pos_w[rows][:, group.body_ids]
        quat = art.body_link_quat_w[rows][:, group.body_ids]
        rot = rotations.to_matrix(quat)  # (n, P, 3, 3), columns = body axes
        centers = pos + rotations.apply(quat, center_local)
        i, j = zip(*pairs)
        pen = sat_penetration(centers[:, i], rot[:, i], half[:, i], centers[:, j], rot[:, j], half[:, j])
        nb = pen[:, is_neighbour].amax(1) if bool(is_neighbour.any()) else torch.zeros(n, device=dev)
        nn = pen[:, ~is_neighbour].amax(1) if bool((~is_neighbour).any()) else torch.zeros(n, device=dev)
        state, _ = inner._now()
        top = state.table_top_z.reshape(n, 1)
        highest = centers[..., 2] + (half * rot[..., 2, :].abs()).sum(-1)
        lowest = centers[..., 2] - (half * rot[..., 2, :].abs()).sum(-1)
        table = (top - lowest).clamp_min(0).amax(1)
        passthrough[live] |= (highest < top).any(1)[live]
        robot = torch.zeros(n, device=dev)
        hd = torch.full((n,), float("nan"), device=dev)
        zf = torch.full((n,), float("nan"), device=dev)
        lk = torch.full((n,), -1, dtype=torch.long, device=dev)
        pl = torch.full((n,), -1, dtype=torch.long, device=dev)
        if contacts is not None:
            _, point, _, sep, counts, starts = contacts.view.get_contact_data(dt=contacts.dt)
            pair, index = contact_indices(counts, starts, contacts.capacity)
            partner = pair % contacts.view.filter_count
            sensor = pair // contacts.view.filter_count
            is_robot = partner < contacts.num_robot_links
            env_of, plate_of = contacts.env_ids[sensor], contacts.plate_ids[sensor]
            depth = (-sep.flatten()[index]).clamp_min(0)
            robot.scatter_reduce_(0, env_of[is_robot], depth[is_robot], reduce="amax")
            # The deepest robot contact per env: where on the cloth, and which link.
            if bool(is_robot.any()):
                e_r, p_r, d_r = env_of[is_robot], plate_of[is_robot], depth[is_robot]
                pt, lnk = point[index][is_robot], partner[is_robot]
                order = torch.argsort(d_r)  # ascending; later writes win -> deepest kept
                e_r, p_r, d_r, pt, lnk = e_r[order], p_r[order], d_r[order], pt[order], lnk[order]
                q = quat[e_r, p_r]
                rest = rotations.apply(q * q.new_tensor([1., -1., -1., -1.]), pt - pos[e_r, p_r])
                rel = rest[:, None, :2] - hinge_a[e_r]
                u = hinge_u[e_r]
                dist = (rel[..., 0] * u[..., 1] - rel[..., 1] * u[..., 0]).abs().amin(1)
                hd[e_r], zf[e_r] = dist, rest[:, 2] / half[e_r, p_r, 2]
                lk[e_r], pl[e_r] = lnk, p_r
        for key, value in (("robot_depth", robot), ("hinge_dist", hd), ("z_frac", zf), ("link", lk), ("plate", pl)):
            series[key].append(torch.where(live, value, torch.full_like(value, -1 if value.dtype == torch.long else float("nan"))).cpu())
        angles = art.joint_pos[rows][:, group.joint_ids]
        hinge_min[live] = torch.minimum(hinge_min, angles.amin(1))[live]
        hinge_max[live] = torch.maximum(hinge_max, angles.amax(1))[live]
        for kind, value in (("neighbour", nb), ("non_neighbour", nn), ("table", table), ("robot", robot)):
            worst[kind][live] = torch.maximum(worst[kind], value)[live]
            big = (value > 0.005) & live
            for e in big.nonzero().flatten().tolist():
                events.append((float(value[e]), kind, e, step))

    obs, _ = env.reset()
    b.player.reset_rnn()
    step = 0
    with torch.inference_mode():
        measure(step)
        while step < 2 * int(inner.max_episode_length) and not bool(done.all()):
            obs, _, terminated, truncated, extras = env.step(b.act(obs))
            step += 1
            ended = terminated | truncated
            first = ended & ~done
            # Measure BEFORE marking done: for an env that just ended, poses are post-reset,
            # so exclude it from this step's measurement instead.
            done |= first
            success[first] = extras["episode_final"]["successes"][first] > 0
            measure(step)
            if bool(ended.any()):
                b.player.reset_rnn(ended.nonzero(as_tuple=True)[0])

    def dist(x):
        x = x.cpu().numpy() * 1000
        return {"median_mm": round(float(np.median(x)), 2), "p90_mm": round(float(np.percentile(x, 90)), 2),
                "max_mm": round(float(x.max()), 2), "envs_over_5mm": int((x > 5).sum()),
                "envs_over_10mm": int((x > 10).sum())}

    events.sort(reverse=True)
    result = {
        "design": args.design, "num_envs": n, "thickness_m": b.thickness, "plates": plates,
        "pairs": {"neighbour": [f"{plates[i]}-{plates[j]}" for (i, j), m in zip(pairs, is_neighbour.tolist()) if m],
                  "non_neighbour": [f"{plates[i]}-{plates[j]}" for (i, j), m in zip(pairs, is_neighbour.tolist()) if not m]},
        "success_rate": float(success.float().mean()),
        "worst_per_env_episode": {k: dist(v) for k, v in worst.items()},
        "envs_with_panel_fully_below_table_top": int(passthrough.sum()),
        "hinge_angle_deg": {"min": round(float(hinge_min.min().rad2deg()), 1),
                            "max": round(float(hinge_max.max().rad2deg()), 1),
                            "envs_bent_below_minus_30": int((hinge_min < -0.5236).sum())},
        "top_events": [{"depth_mm": round(d * 1000, 1), "kind": k, "env": e, "step": s,
                        "design_dims_m": b.dims[e]} for d, k, e, s in events[:15]],
        "robot_contacts_measured": contacts is not None,
    }
    S = {k: torch.stack(v) for k, v in series.items()}  # (steps, n)
    depth_s = S["robot_depth"].nan_to_num(0)
    deep = depth_s > 0.005
    contact = depth_s > 0
    hdist, zfrac = S["hinge_dist"], S["z_frac"]
    near = hdist < half[..., 2].amax().cpu() * 2  # within ~one thickness of a hinge line
    result["robot_contacts"] = {
        "steps_with_any_robot_contact": int(contact.sum()),
        "fraction_of_contact_steps_near_hinge": round(float((near & contact).sum() / max(1, contact.sum())), 3),
        "fraction_of_deep_steps_near_hinge": round(float((near & deep).sum() / max(1, deep.sum())), 3),
        "deep_steps_(>5mm)": int(deep.sum()),
        "deep_steps_contact_below_midplane": int((deep & (zfrac < 0)).sum()),
        "deep_links": {link_names[i]: int(((S["link"] == i) & deep).sum()) for i in S["link"][deep].unique().tolist() if i >= 0},
        "near_hinge_threshold_m": round(float(half[..., 2].amax()) * 2, 4),
    }
    worst_envs = depth_s.amax(0).argsort(descending=True)[:5].tolist()
    result["worst_env_timeseries"] = {}
    for e in worst_envs:
        steps = contact[:, e].nonzero().flatten().tolist()
        result["worst_env_timeseries"][e] = {
            "dims_m": b.dims[e],
            "steps": steps,
            "depth_mm": [round(float(depth_s[t, e]) * 1000, 1) for t in steps],
            "hinge_dist_mm": [round(float(hdist[t, e]) * 1000, 1) for t in steps],
            "z_frac": [round(float(zfrac[t, e]), 2) for t in steps],
            "link": [link_names[int(S["link"][t, e])] for t in steps],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("design", "success_rate", "worst_per_env_episode",
                                             "envs_with_panel_fully_below_table_top", "hinge_angle_deg")},
                     indent=2), flush=True)
    print(json.dumps(result["robot_contacts"], indent=1), flush=True)
    print("top events:", json.dumps(result["top_events"][:8]), flush=True)
    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
