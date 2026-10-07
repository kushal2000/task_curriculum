"""Where do the DRAWN rigid plates overlap the fingers and the table, and by how much?

    scripts/newton_py -m scripts.analysis.rigid_overlap_probe --checkpoint <pth> --out x.npz [hydra...]

Same env construction as `live_viser.py` (pinned start, nominal physics, even sizes off), same
player. Every step, for every env, on the collision geometry the viewer draws:

* finger -> plate: robot mesh vertices inside an active plate's convex hull (half-space test),
  deepest one, and which robot body it belongs to. Also the minimum SIGNED distance robot->plate
  (negative = inside the drawn slab), so the contact shell (`contact_margin`) shows up as the
  typical resting value.
* plate -> table: lowest plate vertex above the table top, for vertices over the table footprint.

Plus the per-env Newton contact count and contact margins, to tell "contact present but soft"
from "contact missing".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="Isaacsimenvs-FoldPool-Direct-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mass_scale", type=float, default=1.0)
    p.add_argument("--contact_ke", type=float, default=0.0, help=">0: every shape's material ke (and kd)")
    p.add_argument("--contact_kd", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        AppLauncher(args)
    except ImportError:
        pass

    import gymnasium as gym
    import torch
    import warp as wp

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.live_viser import _capture_geometry
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = 0.01
        env_cfg.termination.success_steps = 10
        env_cfg.cloth.nan_policy = "reset"
        rd = env_cfg.rigid_cloth.randomization
        rd.enabled = True
        for f, v in (("slat_friction_scale", 1.0), ("fingertip_friction_scale", 1.0),
                     ("hinge_damping_scale", 1.0), ("hinge_stiffness", 0.0),
                     ("contact_margin_offset", 0.0), ("mass_scale", args.mass_scale)):
            setattr(rd, f, [v, v])  # the viewer's nominal physics
        disable_randomization(env_cfg)
        use_single_object_variant()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        shapes = _capture_geometry(model)
        solver = inner._solver()
        ke_t, kd_t = wp.to_torch(model.shape_material_ke), wp.to_torch(model.shape_material_kd)
        if args.contact_ke > 0:
            import newton
            ke_t.fill_(args.contact_ke)
            kd_t.fill_(args.contact_kd)
            inner._notify(int(newton.ModelFlags.SHAPE_PROPERTIES))
        sl = int(inner._slat_shapes[0, 0])
        gsr = wp.to_torch(solver.mjw_model.geom_solref)
        print(f"[probe] shape ke/kd slat {float(ke_t[sl]):.1f}/{float(kd_t[sl]):.1f}, range ke "
              f"{float(ke_t.min()):.1f}..{float(ke_t.max()):.1f}; geom_solref range "
              f"{gsr.reshape(-1, 2).min(0).values.tolist()}..{gsr.reshape(-1, 2).max(0).values.tolist()}; "
              f"solimp {wp.to_torch(solver.mjw_model.geom_solimp).reshape(-1, 5)[0].tolist()}; "
              f"dt {float(solver.mjw_model.opt.timestep.numpy().reshape(-1)[0]):.5f}", flush=True)
        pm = wp.to_torch(model.body_mass)[inner._slat_bodies[0]]
        print(f"[probe] plate body mass (world 0) min {float(pm.min()) * 1e3:.1f} g, max {float(pm.max()) * 1e3:.1f} g", flush=True)
        print(f"[probe] slat margin {float(inner._margin[inner._slat_shapes].mean()) * 1e3:.2f} mm, "
              f"all-shape margin max {float(inner._margin.max()) * 1e3:.2f} mm", flush=True)
        robot_margin = inner._margin[torch.as_tensor(
            [i for i, b in enumerate(model.shape_body.numpy()) if b >= 0 and "Robot" in model.body_label[b]],
            device=dev)]
        print(f"[probe] robot shape margins: min {float(robot_margin.min()) * 1e3:.2f} "
              f"max {float(robot_margin.max()) * 1e3:.2f} mm", flush=True)

        slat_bodies = set(inner._slat_bodies.reshape(-1).tolist())

        def to_t(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)

        def qrot(q, v):  # q xyzw (..., 4), v (..., 3)
            u, w = q[..., :3], q[..., 3:4]
            t = 2.0 * torch.cross(u, v, dim=-1)
            return v + w * t + torch.cross(u, t, dim=-1)

        def wxyz_to_xyzw(q):
            return np.array([q[1], q[2], q[3], q[0]])

        per_world = []
        for w in range(n):
            mine = [s for s in shapes if s.world == w and s.body >= 0]
            robot = [s for s in mine if "Robot" in s.label]
            plates = [s for s in mine if s.body in slat_bodies]
            table = [s for s in shapes if s.world in (w, -1) and ("Table" in s.label or (s.body < 0 and s.world == w))]
            # robot: verts in body frame, plus body index per vertex
            rv, rb = [], []
            for s in robot:
                q = wxyz_to_xyzw(s.wxyz)
                v = qrot(to_t(q)[None], to_t(s.verts)) + to_t(s.pos)[None]
                rv.append(v)
                rb.append(torch.full((len(v),), s.body, dtype=torch.long, device=dev))
            # plates: half-spaces in body frame + verts in body frame
            pl = []
            for s in plates:
                q = wxyz_to_xyzw(s.wxyz)
                v = qrot(to_t(q)[None], to_t(s.verts)) + to_t(s.pos)[None]
                f = torch.as_tensor(s.faces, device=dev).long()
                a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
                nrm = torch.cross(b - a, c - a, dim=-1)
                ok = nrm.norm(dim=-1) > 1e-12
                nrm = nrm[ok] / nrm[ok].norm(dim=-1, keepdim=True)
                d = (nrm * a[ok]).sum(-1)
                cen = v.mean(0)
                flip = (nrm * cen).sum(-1) > d
                nrm[flip] *= -1
                d[flip] *= -1
                pl.append((s.body, v, nrm, d))
            per_world.append(dict(rv=torch.cat(rv), rb=torch.cat(rb), plates=pl, table=table))

        labels = [str(x) for x in model.body_label]

        def world_pts(body_ids, local):
            bq = wp.to_torch(NewtonManager._state_0.body_q)[body_ids]
            return qrot(bq[:, 3:7], local) + bq[:, :3]

        def table_top(w):
            tb = per_world[w]["table"]
            pts = []
            for s in tb:
                q = wxyz_to_xyzw(s.wxyz)
                v = qrot(to_t(q)[None], to_t(s.verts)) + to_t(s.pos)[None]
                if s.body >= 0:
                    v = world_pts(torch.full((len(v),), s.body, device=dev, dtype=torch.long), v)
                pts.append(v)
            v = torch.cat(pts)
            return float(v[:, 2].max()), v[:, :2].min(0).values, v[:, :2].max(0).values

        for w in range(min(n, 2)):
            print(f"[probe] world {w}: {len(per_world[w]['plates'])} plate shapes, "
                  f"{len(per_world[w]['rv'])} robot verts, table top z {table_top(w)[0]:.4f}", flush=True)

        # ---- which shape pairs can produce a contact at all
        shape_body = model.shape_body.numpy()
        shape_world = model.shape_world.numpy()
        S = int(model.shape_count)
        pipeline = NewtonManager._collision_pipeline
        print(f"[probe] broad phase {getattr(pipeline, 'broad_phase_mode', '?')}", flush=True)
        body_shapes = {}
        for i, b in enumerate(shape_body):
            body_shapes.setdefault(int(b), []).append(i)
        table_shapes = [[i for i in range(S) if shape_world[i] == w and (
            (shape_body[i] < 0) or "Table" in labels[shape_body[i]])] for w in range(n)]
        pairs = wp.to_torch(pipeline.shape_pairs_filtered).view(-1, 2).long() if getattr(
            pipeline, "shape_pairs_filtered", None) is not None else None
        if pairs is not None:
            pk = torch.minimum(pairs[:, 0], pairs[:, 1]) * S + torch.maximum(pairs[:, 0], pairs[:, 1])

            def n_pairs(A, B):
                A = torch.as_tensor(A, device=dev).long()
                B = torch.as_tensor(B, device=dev).long()
                k = (torch.minimum(A[:, None], B[None]) * S + torch.maximum(A[:, None], B[None])).reshape(-1)
                return int(torch.isin(k, pk).sum()), int(k.numel())

            plates0 = inner._slat_shapes[0].tolist()
            print(f"[probe] pairs plate-table (world 0): {n_pairs(plates0, table_shapes[0])}", flush=True)
            rob = sorted({int(b) for b in shape_body if b >= 0 and "Robot" in labels[b] and shape_world[body_shapes[b][0]] == 0})
            for b in rob:
                print(f"[probe] pairs plate-{labels[b].split('/')[-1]}: {n_pairs(plates0, body_shapes[b])}", flush=True)
        stat = dict(tab_over=0, tab_over_contact=0, fin_in=0, fin_in_contact=0)

        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=0.0,
        )
        obs, _ = env.reset()
        T = args.steps
        finger_depth = np.zeros((T, n), np.float32)    # max depth of a robot vertex inside a drawn plate
        finger_sd = np.full((T, n), np.nan, np.float32)  # min signed dist robot vertex -> plate
        finger_body = np.full((T, n), -1, np.int64)
        table_gap = np.full((T, n), np.nan, np.float32)  # min (plate vertex z - table top), over footprint
        n_contacts = np.zeros((T, n), np.int32)
        ended_a = np.zeros((T, n), bool)
        folded_a = np.zeros((T, n), bool)
        for t in range(T):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            ended = (term.bool() | trunc.bool())
            ended_a[t] = ended.cpu().numpy()
            reasons = getattr(inner, "_termination_reasons", {})
            if "fold_held" in reasons:
                folded_a[t] = reasons["fold_held"].bool().cpu().numpy()
            if ended.any():
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            c = NewtonManager._contacts
            if c is not None and t % 10 == 0:
                cnt = int(c.rigid_contact_count.numpy()[0])
                s0 = c.rigid_contact_shape0.numpy()[:cnt]
                sw = model.shape_world.numpy()[s0]
                n_contacts[t] = np.bincount(sw[sw >= 0], minlength=n)[:n]
            ck = None
            if c is not None:
                cnt = int(c.rigid_contact_count.numpy()[0])
                a0 = wp.to_torch(c.rigid_contact_shape0)[:cnt].long()
                a1 = wp.to_torch(c.rigid_contact_shape1)[:cnt].long()
                ck = torch.minimum(a0, a1) * S + torch.maximum(a0, a1)

            def has_contact(A, B):
                if ck is None or not len(ck):
                    return False
                A = torch.as_tensor(A, device=dev).long()
                B = torch.as_tensor(B, device=dev).long()
                k = (torch.minimum(A[:, None], B[None]) * S + torch.maximum(A[:, None], B[None])).reshape(-1)
                return bool(torch.isin(k, ck).any())

            for w in range(n):
                W = per_world[w]
                rvw = world_pts(W["rb"], W["rv"])
                top, lo, hi = table_top(w)
                best_d, best_sd, best_b, gap = 0.0, np.inf, -1, np.inf
                for body, pv, nrm, d in W["plates"]:
                    bq = wp.to_torch(NewtonManager._state_0.body_q)[body]
                    if abs(float(bq[2])) > 1.6 or (bq[:2] - torch.as_tensor(inner.scene.env_origins[w, :2])).abs().max() > 1.5:
                        continue  # parked
                    # robot verts into plate body frame
                    qi = bq[3:7].clone()
                    qi[:3] *= -1
                    loc = qrot(qi[None], rvw - bq[None, :3])
                    sd = (loc @ nrm.T - d[None]).max(dim=1).values  # >0 outside (lower bound), <0 inside
                    k = int(sd.argmin())
                    if float(sd[k]) < 0:
                        stat["fin_in"] += 1
                        stat["fin_in_contact"] += has_contact(body_shapes[body], body_shapes[int(W["rb"][k])])
                    if float(sd[k]) < best_sd:
                        best_sd = float(sd[k])
                        if best_sd < 0:
                            best_d, best_b = -best_sd, int(W["rb"][k])
                    # plate verts to world, vs table top
                    pw = qrot(bq[None, 3:7], pv) + bq[None, :3]
                    over = ((pw[:, :2] >= lo) & (pw[:, :2] <= hi)).all(dim=1)
                    if over.any():
                        q_ = pw[over]
                        # depth into the table BOX: distance to its nearest face (top or a side);
                        # positive outside the top, negative inside
                        inside = torch.stack((top - q_[:, 2], q_[:, 0] - lo[0], hi[0] - q_[:, 0],
                                              q_[:, 1] - lo[1], hi[1] - q_[:, 1]), dim=1)
                        dep = inside.min(dim=1).values
                        dep = torch.where(q_[:, 2] >= top, top - q_[:, 2], dep)
                        k_ = int(dep.argmax())
                        g_ = -float(dep[k_])
                        if g_ < 0 and int(inside[k_].argmin()) > 0:
                            stat["side"] = stat.get("side", 0) + 1
                        gap = min(gap, g_)
                        if -0.3 < g_ < -0.002:
                            stat["tab_over"] += 1
                            stat["tab_over_contact"] += has_contact(body_shapes[body], table_shapes[w])
                finger_depth[t, w] = best_d
                finger_sd[t, w] = best_sd
                finger_body[t, w] = best_b
                table_gap[t, w] = gap
            if t % 250 == 0:
                print(f"[probe] t={t} finger sd min {np.nanmin(finger_sd[:t+1]) * 1e3:.1f} mm, "
                      f"table gap min {np.nanmin(table_gap[:t+1]) * 1e3:.1f} mm", flush=True)
        np.savez(args.out, finger_depth=finger_depth, finger_sd=finger_sd, finger_body=finger_body,
                 table_gap=table_gap, n_contacts=n_contacts, ended=ended_a, folded=folded_a,
                 labels=np.array(labels, dtype=object))
        print(f"[probe] contact presence at overlap: {stat}", flush=True)
        fs = finger_sd[np.isfinite(finger_sd)]
        tg = table_gap[np.isfinite(table_gap)]
        print(f"[probe] finger->plate signed dist [mm]: pct 1/5/50 = {np.percentile(fs, [1, 5, 50]) * 1e3}")
        print(f"[probe]   steps with a finger inside a drawn plate: {(fs < 0).mean() * 100:.1f}%  "
              f"(>2 mm {(fs < -0.002).mean() * 100:.1f}%, >5 mm {(fs < -0.005).mean() * 100:.1f}%)")
        print(f"[probe] plate->table gap [mm]: pct 1/5/50 = {np.percentile(tg, [1, 5, 50]) * 1e3}")
        print(f"[probe]   steps with a plate below the table top: {(tg < 0).mean() * 100:.1f}%  "
              f"(>2 mm {(tg < -0.002).mean() * 100:.1f}%)")
        bodies, cnt = np.unique(finger_body[finger_body >= 0], return_counts=True)
        for b, c_ in sorted(zip(bodies, cnt), key=lambda x: -x[1])[:10]:
            print(f"[probe]   inside-plate body {labels[b].split('/')[-1]}: {c_} env-steps")
        import os
        os._exit(0)

    run()


if __name__ == "__main__":
    main()
