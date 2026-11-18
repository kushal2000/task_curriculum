"""Do the hand's links pass through each other while a policy runs with robot self-collision OFF?

    scripts/newton_py scripts/analysis/finger_self_pen_probe.py --task <id> --checkpoint <pth> --out x.json [hydra...]

With `robot_self_collision=false` every robot shape pair is in the collision filter, so the fingers
feel nothing when they meet. This probe runs the policy exactly as trained (self-collision off) and,
every PHYSICS step, measures how deep the shape pairs that self-collision ON would check overlap:
all robot shape pairs with at least one hand link (r_* body), minus same-body pairs and minus the
spec's `self_collision_filter` (the adjacent links). Depth is a point-in-convex-hull test: dense
surface samples of shape A expressed in shape B's frame, depth = distance to B's hull boundary
(exact for a sample inside B; edge-edge crossings are seen through the surface samples, so it is a
slight underestimate). Both directions, max taken. Collision shapes must be (convex) meshes or boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _quat_rotate(q, v):
    """q (..., 4) xyzw, v (..., 3)."""
    import torch

    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _quat_conj(q):
    import torch

    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=600, help="policy steps (envs auto-reset)")
    p.add_argument("--samples", type=int, default=400, help="surface samples per shape")
    p.add_argument("--random_start", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        AppLauncher(args)
    except ImportError:
        pass

    import copy

    import gymnasium as gym
    import torch
    import trimesh
    from scipy.spatial import ConvexHull

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        rs0 = copy.deepcopy(env_cfg.reset)
        disable_randomization(env_cfg)
        if args.random_start:
            env_cfg.reset = rs0
        use_single_object_variant()
        print(f"[pen] robot_self_collision = {env_cfg.robot_self_collision}", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=0.0,
        )
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton import GeoType

        model = NewtonManager.get_model()
        blabels = list(model.body_label)
        bidx = {lab: i for i, lab in enumerate(blabels)}
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_tf = model.shape_transform.numpy()          # (S, 7) p + q xyzw, in the body frame
        filt_model = {tuple(sorted(map(int, pr))) for pr in (model.shape_collision_filter_pairs or [])}

        def robot_link(lab):
            return lab.split("/Robot/", 1)[1].split("/")[0] if "/env_0/Robot/" in lab else None

        # env-0 robot shapes
        shapes = [s for s in range(len(shape_body))
                  if shape_body[s] >= 0 and robot_link(blabels[shape_body[s]]) is not None]
        link_of = {s: robot_link(blabels[shape_body[s]]) for s in shapes}
        spec_filter = inner.robot_spec.self_collision_filter
        filt_spec = {tuple(sorted((a, b))) for a, bs in spec_filter.items() for b in bs}
        hand = lambda link: link.startswith(("r_", "l_"))  # noqa: E731
        pairs = []
        for i, a in enumerate(shapes):
            for b in shapes[i + 1:]:
                la, lb = link_of[a], link_of[b]
                if la == lb or not (hand(la) or hand(lb)) or tuple(sorted((la, lb))) in filt_spec:
                    continue
                pairs.append((a, b))
        in_model_filter = sum(tuple(sorted(pr)) in filt_model for pr in pairs)
        print(f"[pen] {len(shapes)} env-0 robot shapes, {len(pairs)} would-be self-collision pairs with a hand "
              f"link; {in_model_filter} of them are in the model's collision filter", flush=True)

        # geometry -> hull planes + surface samples, in each shape's own frame
        geo = {}
        rng = np.random.default_rng(0)
        for s in sorted({x for pr in pairs for x in pr}):
            t = int(shape_type[s])
            sc = shape_scale[s]
            if t in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
                v = np.asarray(model.shape_source[s].vertices, dtype=np.float64) * sc
            elif t == int(GeoType.BOX):
                v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * sc
            else:
                raise RuntimeError(f"shape {s} ({link_of[s]}) has type {GeoType(t).name}; only meshes/boxes handled")
            hull = ConvexHull(v)
            tm = trimesh.Trimesh(v, hull.simplices).convex_hull
            pts = np.concatenate([tm.vertices, trimesh.sample.sample_surface(tm, args.samples, seed=rng)[0]])
            # one plane per hull FACET (coplanar triangles merged), n.x + d <= 0 inside
            eq = np.unique(np.round(hull.equations, 5), axis=0)
            c = tm.vertices.mean(0)
            geo[s] = (torch.tensor(pts, dtype=torch.float32, device=dev),
                      torch.tensor(eq, dtype=torch.float32, device=dev),
                      torch.tensor(c, dtype=torch.float32, device=dev),
                      float(np.linalg.norm(tm.vertices - c, axis=1).max()))
        P = max(g[0].shape[0] for g in geo.values())
        F = max(g[1].shape[0] for g in geo.values())
        print(f"[pen] {len(geo)} shapes in pairs; samples/shape <= {P}, planes/shape <= {F}", flush=True)

        def pad(x, m, fill):
            if x.shape[0] == m:
                return x
            return torch.cat([x, fill.expand(m - x.shape[0], x.shape[1])], 0)

        A = torch.tensor([pr[0] for pr in pairs], device=dev)
        B = torch.tensor([pr[1] for pr in pairs], device=dev)
        pts = {s: pad(g[0], P, g[0][:1]) for s, g in geo.items()}
        pln = {s: pad(g[1], F, torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=dev)) for s, g in geo.items()}
        ptsA = torch.stack([pts[a] for a, _ in pairs]); ptsB = torch.stack([pts[b] for _, b in pairs])   # (K, P, 3)
        plA = torch.stack([pln[a] for a, _ in pairs]); plB = torch.stack([pln[b] for _, b in pairs])     # (K, F, 4)
        cA = torch.stack([geo[a][2] for a, _ in pairs]); cB = torch.stack([geo[b][2] for _, b in pairs])  # (K, 3)
        rAB = torch.tensor([geo[a][3] + geo[b][3] for a, b in pairs], device=dev)                          # (K,)
        stf = torch.tensor(shape_tf, dtype=torch.float32, device=dev)

        # per-env body index of each pair's two bodies
        def env_body(s, e):
            lab = blabels[shape_body[s]].replace("/env_0/", f"/env_{e}/", 1)
            return bidx[lab]
        bA = torch.tensor([[env_body(a, e) for a, _ in pairs] for e in range(n)], device=dev)   # (n, K)
        bB = torch.tensor([[env_body(b, e) for _, b in pairs] for e in range(n)], device=dev)

        def to_world(local, s_tf, body_q):
            """local (M, P, 3) in the shape frame -> world, via shape tf (M, 7) and body pose (M, 7)."""
            x = _quat_rotate(s_tf[:, None, 3:], local) + s_tf[:, None, :3]
            return _quat_rotate(body_q[:, None, 3:], x) + body_q[:, None, :3]

        def to_shape(world, s_tf, body_q):
            x = _quat_rotate(_quat_conj(body_q[:, None, 3:]), world - body_q[:, None, :3])
            return _quat_rotate(_quat_conj(s_tf[:, None, 3:]), x - s_tf[:, None, :3])

        tfA, tfB = stf[A], stf[B]
        chunk = max(1, int(1.5e8 // (P * F)))
        stats = {"cand": 0, "calls": 0}

        def depth_now():
            bq = torch.as_tensor(NewtonManager.get_state_0().body_q.numpy(), device=dev)   # (Nb, 7)
            qa, qb = bq[bA], bq[bB]                                                         # (n, K, 7)
            # broadphase: bounding spheres
            K = len(pairs)
            wa = to_world(cA[None].expand(n, -1, -1).reshape(-1, 1, 3), tfA.repeat(n, 1), qa.reshape(-1, 7))[:, 0]
            wb = to_world(cB[None].expand(n, -1, -1).reshape(-1, 1, 3), tfB.repeat(n, 1), qb.reshape(-1, 7))[:, 0]
            cand = ((wa - wb).norm(dim=-1) < rAB.repeat(n)).nonzero(as_tuple=True)[0]          # flat (e*K + k)
            out = torch.zeros(n * K, device=dev)
            stats["cand"] += int(cand.numel()); stats["calls"] += 1
            qa_f, qb_f = qa.reshape(-1, 7), qb.reshape(-1, 7)
            for i in range(0, cand.numel(), chunk):
                idx = cand[i:i + chunk]
                k = idx % K

                def one(pts_from, tf_from, q_from, planes_to, tf_to, q_to):
                    loc = to_shape(to_world(pts_from, tf_from, q_from), tf_to, q_to)          # (M, P, 3)
                    d = torch.einsum("mpi,mfi->mpf", loc, planes_to[..., :3]) + planes_to[:, None, :, 3]
                    return (-d.max(dim=-1).values).clamp(min=0.0).max(dim=-1).values         # (M,)

                out[idx] = torch.maximum(
                    one(ptsA[k], tfA[k], qa_f[idx], plB[k], tfB[k], qb_f[idx]),
                    one(ptsB[k], tfB[k], qb_f[idx], plA[k], tfA[k], qa_f[idx]))
            return out.view(n, K)

        log = []
        orig_apply = inner._apply_action

        def apply():
            log.append(depth_now().cpu())
            orig_apply()

        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=dev))
        rest = depth_now().cpu().numpy()          # pinned start pose (env 0..n-1)
        inner._apply_action = apply
        ended_n = 0
        for _step in range(args.steps):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            ended = (term.bool() | trunc.bool())
            ended_n += int(ended.sum())
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])

        D = torch.stack(log).numpy()              # (T, n, K) metres
        T = D.shape[0]
        names = [f"{link_of[a]} | {link_of[b]}" for a, b in pairs]
        print(f"[pen] broadphase: {stats['cand'] / max(stats['calls'], 1):.0f} candidate (env, pair)/step of {n * len(pairs)}", flush=True)
        res = {"task": args.task, "checkpoint": args.checkpoint, "overrides": hydra_args, "physics_steps": T,
               "num_envs": n, "episodes_ended": ended_n, "pairs": len(pairs),
               "pairs_in_model_filter": in_model_filter, "rest_max_mm": float(rest.max() * 1e3),
               "any_pair": {}, "per_pair": []}
        anyd = D.max(axis=-1)                    # (T, n)
        for thr in (0.001, 0.003, 0.005, 0.01):
            res["any_pair"][f">{thr * 1e3:g}mm_step_frac"] = float((anyd > thr).mean())
            res["any_pair"][f">{thr * 1e3:g}mm_env_frac"] = float((anyd > thr).any(axis=0).mean())
        res["any_pair"]["max_mm"] = float(anyd.max() * 1e3)
        res["any_pair"]["p99_mm"] = float(np.percentile(anyd, 99) * 1e3)
        for k, nm in enumerate(names):
            d = D[..., k]
            if d.max() < 0.001 and rest[:, k].max() < 0.001:
                continue
            # longest run of consecutive physics steps > 2 mm in any env
            run_max = 0
            for e in range(n):
                r = 0
                for x in d[:, e] > 0.002:
                    r = r + 1 if x else 0
                    run_max = max(run_max, r)
            res["per_pair"].append({
                "pair": nm, "max_mm": float(d.max() * 1e3), "p99_mm": float(np.percentile(d, 99) * 1e3),
                "step_frac_gt1mm": float((d > 0.001).mean()), "step_frac_gt3mm": float((d > 0.003).mean()),
                "env_frac_gt3mm": float((d > 0.003).any(axis=0).mean()),
                "longest_gt2mm_s": run_max * float(inner.physics_dt), "rest_mm": float(rest[:, k].max() * 1e3)})
        res["per_pair"].sort(key=lambda r: -r["step_frac_gt3mm"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        np.savez_compressed(args.out.with_suffix(".npz"), depth=D.astype(np.float16), rest=rest, names=np.array(names))
        print(f"[pen] wrote {args.out}: any pair >3mm in {res['any_pair']['>3mm_step_frac'] * 100:.1f}% of env-steps, "
              f"max {res['any_pair']['max_mm']:.1f} mm, rest max {res['rest_max_mm']:.1f} mm", flush=True)
        for r in res["per_pair"][:12]:
            print(f"[pen]   {r['pair']:<45} >3mm {r['step_frac_gt3mm'] * 100:5.1f}%  max {r['max_mm']:5.1f} mm  "
                  f"longest {r['longest_gt2mm_s']:.2f} s  rest {r['rest_mm']:.1f}", flush=True)
        import os

        os._exit(0)

    run()


if __name__ == "__main__":
    main()
