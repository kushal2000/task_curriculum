"""How far does a POLICY's fold spring back once the hand lets go, and where on the sheet is the bend?

    scripts/newton_py -m scripts.analysis.cloth_springback_probe --checkpoint <pth> --out x.json [hydra...]

Needs ``env.cloth.release_persist=true`` and plasticity on. One episode per env (as
``cloth_pierce_probe.py``). Every step, per env: the flap's opening angle (0 = folded flat,
180 = flat sheet; angle at the crease row between the far moving row and the far stationary row),
fold error, the release test's hand distance (fingertips + iiwa14_link_7 origins) and the nearest
of ALL hand links (``left_.*`` + iiwa14_link_7) with its name.

At the release latch and at the episode's last live step, the hinge picture: for every interior
edge, its current dihedral angle ``theta`` and plastic rest angle ``rest`` (set part), binned by
the edge midpoint's rest distance from the crease (half-cell bins) and split into crease-parallel
grid edges / diagonals / crease-perpendicular grid edges. ``theta - rest`` is the elastic part,
which springs back once nothing holds the sheet (at most the yield angle per hinge).

Output: --out JSON (summary + per-episode records) and --out with .npz (per-step traces).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant

OUTCOMES = ("persist", "persist_fail", "fold_held", "fall", "hand_far", "nonfinite", "timeout")


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="checkpoints/config.yaml")
    p.add_argument("--sapg_expl_coef", type=float, default=0.0)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--yank", choices=("none", "release", "entry"), default="none",
                   help="at the release latch / the fold entry, take the arm from the policy and drive it "
                        "back to its start pose over --yank_steps (fingers frozen): does the fold hold alone?")
    p.add_argument("--yank_steps", type=int, default=30)
    p.add_argument("--signed_trace", type=int, default=0,
                   help="record SIGNED dihedral / rest angles of the crease-region hinges (|bin| <= 2) of the "
                        "first N triggered envs, 3 steps before to 20 steps after the trigger, plus the flap's "
                        "mean particle speed")
    p.add_argument("--zero_elastic", action="store_true",
                   help="at the trigger, set that env's rest angles to its current dihedral angles (no "
                        "bending energy left): isolates whether bending drives the flap")
    p.add_argument("--remove", choices=("arm", "contact"), default="arm",
                   help="arm: drive the arm back to its start pose (it can drag the sheet); contact: switch "
                        "off that env's hand-cloth contact at the trigger (the hand passes through, the "
                        "policy keeps acting), so only the cloth's own state decides")
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
    import newton as newton_mod
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.tasks.cloth.utils.plasticity import _dihedral_kernel
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    task = "Isaacsimenvs-Cloth-Direct-v0"

    @hydra_task_config_with_yaml(task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = 0.01
        disable_randomization(env_cfg)
        use_single_object_variant()
        env = gym.make(task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        c = inner.cfg.cloth
        assert c.release_persist, "run with env.cloth.release_persist=true"
        plast = inner._plasticity
        assert plast is not None, "run with plasticity on (env.cloth.plastic_yield_curvature > 0)"
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
        )
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        R, H = int(c.resolution), float(c.size) / (int(c.resolution) - 1)
        P = R * R
        ax = 0 if c.fold_axis == "x" else 1
        rest = np.asarray(inner._cloth_rest_local, dtype=np.float64).reshape(P, 3)

        # --- model edges -> (env, local particles); check the particle order against the adapter.
        ei = model.edge_indices.numpy()
        pw = model.particle_world.numpy()
        first = {w: int(np.flatnonzero(pw == w).min()) for w in range(n)}
        cloth_world = pw[np.maximum(ei[:, 2], 0)]
        interior = (ei[:, 0] >= 0) & (ei[:, 1] >= 0) & (cloth_world >= 0) & (cloth_world < n)
        eids = np.flatnonzero(interior)
        e_env = cloth_world[eids]
        off = np.array([first[w] for w in e_env])
        la, lb = ei[eids, 2] - off, ei[eids, 3] - off
        assert la.min() >= 0 and lb.max() < P
        q0 = NewtonManager._state_0.particle_q.numpy()
        mine = inner._particles_w()[0].cpu().numpy()
        assert np.allclose(q0[first[0]:first[0] + P], mine, atol=1e-5), "particle order mismatch"
        d = rest[lb] - rest[la]
        mid = 0.5 * (rest[la] + rest[lb])[:, ax]           # crease at 0 (sheet centred)
        par = np.abs(d[:, ax]) < 1e-6                       # runs along the crease: bends the fold
        perp = np.abs(d[:, 1 - ax]) < 1e-6
        kind = np.where(par, 0, np.where(perp, 2, 1))       # 0 crease-parallel, 1 diagonal, 2 perpendicular
        bins = np.round(mid / (0.5 * H)).astype(int)        # half-cell bins, 0 = the crease row
        BINS = list(range(int(bins.min()), int(bins.max()) + 1))
        KINDS = ("parallel", "diagonal", "perpendicular")
        e_env_t = torch.as_tensor(e_env, device=dev)
        eids_wp = wp.array(eids.astype(np.int32), dtype=wp.int32, device=str(model.device))
        theta_all = wp.zeros(ei.shape[0], dtype=float, device=str(model.device))
        yield_deg = np.degrees(plast.yield_angle.numpy()[eids])
        print(f"[spring] {len(eids)} interior hinges ({len(eids) // n}/env), bins {BINS}, "
              f"yield deg parallel {yield_deg[kind == 0].mean():.1f} diagonal {yield_deg[kind == 1].mean():.1f}", flush=True)

        crease_sel = np.abs(bins) <= 2
        flap_idx = torch.as_tensor(sorted(set(np.flatnonzero(rest[:, 0 if c.fold_axis == "x" else 1] > 1e-6))), device=dev)
        sig_buf, sig_rec, sig_open = {}, {}, {}

        def signed_now():
            wp.launch(_dihedral_kernel, dim=ei.shape[0], inputs=[NewtonManager._state_0.particle_q,
                      model.edge_indices, theta_all], device=str(model.device))
            return (np.degrees(theta_all.numpy()[eids]), np.degrees(model.edge_rest_angle.numpy()[eids]))

        def hinge_picture(env_mask: np.ndarray):
            """{env: {kind: {bin: [mean |theta| deg, mean |rest| deg, mean |theta-rest| deg, n]}}}."""
            wp.launch(_dihedral_kernel, dim=ei.shape[0], inputs=[NewtonManager._state_0.particle_q,
                      model.edge_indices, theta_all], device=str(model.device))
            th = np.degrees(theta_all.numpy()[eids])
            rs = np.degrees(model.edge_rest_angle.numpy()[eids])
            out = {}
            for e in np.flatnonzero(env_mask):
                m = e_env == e
                out[int(e)] = {
                    KINDS[k]: {int(b): [round(float(np.abs(th[s]).mean()), 1), round(float(np.abs(rs[s]).mean()), 1),
                                        round(float(np.abs(th[s] - rs[s]).mean()), 1), int(s.sum())]
                               for b in BINS for s in [m & (kind == k) & (bins == b)] if s.any()}
                    for k in range(3)
                }
            return out

        # --- rows for the opening angle
        coord = rest[:, ax]
        crease_row = torch.as_tensor(np.flatnonzero(np.abs(coord) < 1e-6), device=dev)
        far_mv = torch.as_tensor(np.flatnonzero(np.abs(coord - coord.max()) < 1e-6), device=dev)
        far_st = torch.as_tensor(np.flatnonzero(np.abs(coord - coord.min()) < 1e-6), device=dev)

        def opening_deg(parts):
            cr = parts[:, crease_row].mean(1)
            a = parts[:, far_mv].mean(1) - cr
            b = parts[:, far_st].mean(1) - cr
            cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-9)
            return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))

        ids_all, names_all = inner.robot.find_bodies("left_.*|iiwa14_link_7")

        def all_hand_dist(parts):
            hand = inner.robot.data.body_state_w[:, ids_all, 0:3]
            hand = hand.torch if hasattr(hand, "torch") else hand
            dd = torch.cdist(hand.float(), parts.float()).amin(dim=2)       # (N, B)
            v, i = dd.min(dim=1)
            return v, i

        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=dev))

        # Which edge_rest_angle does the cloth SOLVER read? (plasticity writes model.edge_rest_angle)
        cloth_views = [en.view for en in getattr(NewtonManager._solver, "_entries", {}).values()
                       if isinstance(en.solver, newton_mod.solvers.SolverVBD)]
        for cv in cloth_views:
            print(f"[spring] cloth view edge_rest_angle ptr {cv.edge_rest_angle.ptr} vs model {model.edge_rest_angle.ptr} "
                  f"same={cv.edge_rest_angle.ptr == model.edge_rest_angle.ptr}; 'edge_rest_angle' in overrides: "
                  f"{'edge_rest_angle' in getattr(cv, 'overrides', {})}; view edge_count {cv.edge_count} model {model.edge_count}",
                  flush=True)

        def rest_check(tag):
            for cv in cloth_views:
                a, b = np.degrees(cv.edge_rest_angle.numpy()), np.degrees(model.edge_rest_angle.numpy())
                print(f"[spring] rest check {tag}: solver view max|rest| {np.abs(a).max():.1f} deg, model max|rest| "
                      f"{np.abs(b).max():.1f} deg", flush=True)

        # --- yank: override the arm targets of triggered envs after the policy's action pipeline.
        arm_ids = torch.as_tensor(inner._arm_joint_ids, device=dev, dtype=torch.long).reshape(-1)
        hand_ids = torch.as_tensor(inner._hand_joint_ids, device=dev, dtype=torch.long).reshape(-1)
        home_arm = inner._cur_targets[:, arm_ids].clone()
        yank_k = torch.full((n,), -1, dtype=torch.long, device=dev)
        yank_from = torch.zeros_like(home_arm)
        yank_hand = inner._cur_targets[:, hand_ids].clone()
        yank_step = [-1] * n
        orig_pre = inner._pre_physics_step

        def pre_with_yank(actions):
            orig_pre(actions)
            m = yank_k >= 0
            if bool(m.any()):
                idx = m.nonzero(as_tuple=True)[0]
                a = ((yank_k[idx] + 1).float() / float(args.yank_steps)).clamp(max=1.0)[:, None]
                cur = inner._cur_targets
                cur[idx[:, None], arm_ids[None, :]] = yank_from[idx] + a * (home_arm[idx] - yank_from[idx])
                cur[idx[:, None], hand_ids[None, :]] = yank_hand[idx]
                inner._prev_targets = cur.clone()
                yank_k[idx] += 1

        if args.yank != "none" and args.remove == "arm":
            inner._pre_physics_step = pre_with_yank

        # --- remove=contact: clear COLLIDE_PARTICLES on the env's robot shapes, in the model and in
        # the cloth solver's own (copy-on-write) view, in place. Newton applies the flag per thread
        # in the particle and edge/face soft-contact kernels, so it takes effect on the next step.
        import newton
        import re as _re

        flag_arrays = [model.shape_flags]
        for entry in getattr(NewtonManager._solver, "_entries", {}).values():
            if isinstance(entry.solver, newton.solvers.SolverVBD):
                fa = entry.view.shape_flags
                if fa.ptr != model.shape_flags.ptr:
                    flag_arrays.append(fa)
        sw, sb = model.shape_world.numpy(), model.shape_body.numpy()
        blabels = list(model.body_label)
        is_robot = np.array([b >= 0 and _re.search(r"/World/envs/env_[^/]+/Table", str(blabels[b])) is None
                             for b in sb], dtype=bool)
        CP = int(newton.ShapeFlags.COLLIDE_PARTICLES)

        def vanish(e: int) -> int:
            ids = np.flatnonzero(is_robot & (sw == e))
            for fa in flag_arrays:
                f = fa.numpy()
                f[ids] &= ~CP
                fa.assign(wp.array(f, dtype=fa.dtype, device=fa.device))
            return int(ids.size)
        n_vanished = []
        was_entry = torch.zeros(n, dtype=torch.bool, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        was_rel = torch.zeros(n, dtype=torch.bool, device=dev)
        outcome = [""] * n
        T = {k: [] for k in ("opening", "fold_err", "folded", "released", "d_release", "d_all", "d_all_body",
                             "press_N", "pen_max_mm")}
        # Hand press estimate: VBD's soft contact is a penalty, normal force = k * penetration per
        # record (rigid_vbd_kernels._eval_soft_ef_contact), k = mean(soft_contact_ke, shape ke) with
        # rigid_avbd_beta = 0. Summed over the hand's records; damping and friction left out.
        shape_ke = getattr(model, "shape_material_ke", None)
        ke_shape = float(np.median(shape_ke.numpy())) if shape_ke is not None and shape_ke.shape[0] else 2.5e3
        k_eff = 0.5 * (float(inner.cfg.cloth.soft_contact_ke) + ke_shape)
        sheet_mass = float(model.particle_mass.numpy()[: model.particle_count // n].sum())
        print(f"[spring] press estimate: k_eff {k_eff:.0f} N/m (cloth {inner.cfg.cloth.soft_contact_ke:g}, shape "
              f"{ke_shape:g}); sheet mass {sheet_mass * 1e3:.1f} g, weight {sheet_mass * 9.81:.3f} N", flush=True)

        def press_now():
            inner.hand_cloth_touching()
            g, ev = getattr(inner, "_touch_last_gap", None), getattr(inner, "_touch_last_env", None)
            f = torch.zeros(n, device=dev)
            pm = torch.zeros(n, device=dev)
            if g is not None and g.numel():
                pen = (-g).clamp(min=0.0).float()
                ev = ev.long().clamp(0, n - 1)
                f.index_add_(0, ev, k_eff * pen)
                pm.scatter_reduce_(0, ev, pen * 1e3, reduce="amax")
            return f, pm
        at_release, at_end, release_step, last_live = {}, {}, [-1] * n, {}
        step = 0
        while step < args.max_steps and not bool(done.all()):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            ended = term.bool() | trunc.bool()
            live = ~done & ~ended
            parts = inner._particles_w()
            ang = opening_deg(parts)
            fe = inner.fold_error()
            folded = inner.is_folded()
            rel = inner._released.clone()
            d_rel = inner.hand_cloth_distance()
            d_all, b_all = all_hand_dist(parts)
            press, pen_max = press_now()
            for k, v in (("opening", ang), ("fold_err", fe), ("folded", folded), ("released", rel),
                         ("d_release", d_rel), ("d_all", d_all), ("d_all_body", b_all),
                         ("press_N", press), ("pen_max_mm", pen_max)):
                T[k].append(torch.where(live, v.float(), torch.full_like(v.float(), float("nan"))).cpu().numpy())
            newly = (rel & ~was_rel & live).cpu().numpy()
            if newly.any():
                at_release.update(hinge_picture(newly))
                for e in np.flatnonzero(newly):
                    release_step[e] = step
            if args.yank != "none":
                trig = (rel & ~was_rel) if args.yank == "release" else (inner._entry_paid & ~was_entry)
                trig = trig & live & (yank_k < 0) & ~done
                for e in trig.nonzero(as_tuple=True)[0].tolist():
                    yank_step[e] = step
                    if args.zero_elastic:
                        th_z, _ = signed_now()
                        ra = model.edge_rest_angle.numpy()
                        sel = e_env == e
                        ra[eids[sel]] = np.radians(th_z[sel])
                        model.edge_rest_angle.assign(wp.array(ra, dtype=float, device=model.edge_rest_angle.device))
                    if args.remove == "contact":
                        n_vanished.append(vanish(e))
                        if len(n_vanished) == 1:
                            print(f"[spring] contact off for env {e}: {n_vanished[0]} robot shapes in "
                                  f"{len(flag_arrays)} flag arrays", flush=True)
                        continue
                    yank_from[e] = inner._cur_targets[e, arm_ids]
                    yank_hand[e] = inner._cur_targets[e, hand_ids]
                    yank_k[e] = 0
                was_entry = was_entry | (inner._entry_paid & live)
                yank_k[ended] = -1  # the env was reset inside step: give it back to the policy
            was_rel = rel & live | was_rel
            # End snapshot: an env ending now has already been reset, so keep each live env's
            # latest picture only on demand -- take it for envs that are released (the interesting
            # window) every 10 steps, plus every step in the last 3 before a typical end.
            snap = (live & was_rel).cpu().numpy() & (step % 5 == 0)
            if snap.any():
                for e, v in hinge_picture(snap).items():
                    last_live[e] = (step, v)
            fin = ended & ~done
            if bool(fin.any()):
                reasons = inner._termination_reasons
                for e in fin.nonzero(as_tuple=True)[0].tolist():
                    fired = [k for k in OUTCOMES if k in reasons and bool(reasons[k][e])]
                    outcome[e] = fired[0] if fired else ("timeout" if bool(trunc[e]) else "?")
                    if e in last_live:
                        at_end[e] = {"step": last_live[e][0], "hinges": last_live[e][1]}
                done |= fin
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if args.signed_trace > 0:
                th_s, rs_s = signed_now()
                vel = inner.object._velocities()[:, flap_idx].norm(dim=-1).mean(dim=1).cpu().numpy()
                # Heights above the table top [mm] (particle centres; at rest one radius, 8 mm, up).
                top = (inner._table_z_per_env + float(c.table_half_thickness))[:, None]
                zrel = ((parts[..., 2] - inner.scene.env_origins[:, 2:3] - top) * 1e3).cpu().numpy()
                st_np = np.asarray(inner._stationary_idx.cpu())
                hand_pos = inner.robot.data.body_state_w[:, ids_all, 0:3]
                hand_pos = hand_pos.torch if hasattr(hand_pos, "torch") else hand_pos
                mv_np = np.asarray(inner._moving_idx.cpu())
                for e in range(n):
                    if done[e] or ended[e]:
                        continue
                    m = (e_env == e) & crease_sel
                    snap = {"step": step, "theta": np.round(th_s[m], 1).tolist(), "rest": np.round(rs_s[m], 1).tolist(),
                            "flap_speed_mps": round(float(vel[e]), 4), "opening": round(float(ang[e]), 1),
                            "z_bottom_mm": [round(float(zrel[e, st_np].min()), 2), round(float(zrel[e, st_np].mean()), 2)],
                            "z_flap_mm": [round(float(zrel[e, mv_np].min()), 2), round(float(zrel[e, mv_np].mean()), 2)],
                            "z_all_min_mm": round(float(zrel[e].min()), 2),
                            "q_mm": np.round((parts[e] - inner.scene.env_origins[e]).cpu().numpy() * 1e3, 2).tolist(),
                            "qd": np.round(inner.object._velocities()[e].cpu().numpy(), 4).tolist(),
                            "hand_mm": np.round((hand_pos[e] - inner.scene.env_origins[e]).cpu().numpy() * 1e3, 1).tolist()}
                    if e in sig_rec:
                        if len(sig_rec[e]["after"]) < 20:
                            sig_rec[e]["after"].append(snap)
                    elif yank_step[e] == step and len(sig_rec) < args.signed_trace:
                        sig_rec[e] = {"before": list(sig_buf.get(e, [])), "after": [snap]}
                    else:
                        sig_buf.setdefault(e, []).append(snap)
                        sig_buf[e] = sig_buf[e][-3:]
            if step in (100, 200, 300):
                rest_check(f"step {step}")
            step += 1

        tr = {k: np.stack(v, axis=1) for k, v in T.items()}   # (N, steps)
        episodes = []
        for e in range(n):
            r0 = release_step[e]
            ep = {"env": e, "outcome": outcome[e], "release_step": r0}
            pr, fo = tr["press_N"][e], tr["folded"][e]
            fin_ = np.isfinite(pr)
            first_fold = int(np.flatnonzero(fo == 1)[0]) if (fo == 1).any() else -1
            # the fold-making window: first contact to release (or episode end)
            touch_steps = np.flatnonzero(fin_ & (pr > 0))
            t0_ = int(touch_steps[0]) if touch_steps.size else -1
            t1_ = r0 if r0 >= 0 else int(np.flatnonzero(fin_)[-1]) + 1 if fin_.any() else 0
            win = pr[t0_:t1_] if t0_ >= 0 and t1_ > t0_ else np.array([])
            win = win[np.isfinite(win)]
            hold = pr[first_fold:t1_] if 0 <= first_fold < t1_ else np.array([])
            hold = hold[np.isfinite(hold)]
            ep |= {"first_fold_step": first_fold, "first_touch_step": t0_,
                   "press_peak_N": round(float(win.max()), 3) if win.size else 0.0,
                   "press_p90_N": round(float(np.percentile(win, 90)), 3) if win.size else 0.0,
                   "press_mean_touching_N": round(float(win[win > 0].mean()), 3) if (win > 0).any() else 0.0,
                   "press_impulse_Ns": round(float(win.sum() * float(inner.step_dt)), 4) if win.size else 0.0,
                   "touch_frac_before_release": round(float((win > 0).mean()), 3) if win.size else 0.0,
                   "press_while_folded_mean_N": round(float(hold.mean()), 3) if hold.size else None,
                   "press_while_folded_peak_N": round(float(hold.max()), 3) if hold.size else None,
                   "pen_max_mm": round(float(np.nanmax(tr["pen_max_mm"][e])), 2) if fin_.any() else None}
            if r0 >= 0:
                op = tr["opening"][e]
                post = op[r0:]
                post = post[np.isfinite(post)]
                pre_rel = op[max(0, r0 - 30):r0 + 1]
                ep |= {
                    "opening_at_release": round(float(op[r0]), 1),
                    "opening_min_30_before": round(float(np.nanmin(pre_rel)), 1),
                    "opening_max_after": round(float(post.max()), 1) if post.size else None,
                    "opening_last": round(float(post[-1]), 1) if post.size else None,
                    "steps_after_release": int(post.size),
                    "fold_err_max_after_mm": round(float(np.nanmax(tr["fold_err"][e][r0:])) * 1e3, 1),
                    "d_all_min_after_mm": round(float(np.nanmin(tr["d_all"][e][r0:])) * 1e3, 1),
                    "nearest_body_after": names_all[int(tr["d_all_body"][e][r0 + int(np.nanargmin(tr["d_all"][e][r0:]))])],
                    "hinges_at_release": at_release.get(e),
                    "hinges_at_end": at_end.get(e),
                }
            y0 = yank_step[e]
            if y0 >= 0:
                fe_e, fo_e, da_e, op_e = tr["fold_err"][e], tr["folded"][e], tr["d_all"][e], tr["opening"][e]
                live_after = np.isfinite(fe_e[y0:])
                nlive = int(live_after.sum())

                def at(k, arr, scale=1.0):
                    i = y0 + k
                    return round(float(arr[i]) * scale, 1) if i < arr.size and np.isfinite(arr[i]) else None
                ep |= {"yank_step": y0, "yank_live_steps": nlive,
                       "yank_fold_err_mm": {f"+{k}": at(k, fe_e, 1e3) for k in (0, 15, 30, 60, 120, 179)},
                       "yank_opening_deg": {f"+{k}": at(k, op_e) for k in (0, 30, 60, 120, 179)},
                       "yank_d_all_mm": {f"+{k}": at(k, da_e, 1e3) for k in (0, 15, 30, 60)},
                       "yank_folded_frac": round(float(np.nanmean(fo_e[y0:y0 + 180])), 3) if nlive else None,
                       "yank_first_unfold": (int(np.flatnonzero(fo_e[y0:y0 + 180] == 0)[0])
                                             if (fo_e[y0:y0 + 180] == 0).any() else None)}
            episodes.append(ep)

        rel_eps = [x for x in episodes if x["release_step"] >= 0]

        def agg(sel, key):
            v = [x[key] for x in sel if x.get(key) is not None]
            return {"n": len(v), "median": round(float(np.median(v)), 1) if v else None,
                    "p25": round(float(np.percentile(v, 25)), 1) if v else None,
                    "p75": round(float(np.percentile(v, 75)), 1) if v else None}

        def bend_profile(sel, which):
            """Mean over episodes of |theta|, |rest|, |theta-rest| per (kind, bin)."""
            acc = {}
            for x in sel:
                h = x.get(which)
                h = h.get("hinges") if which == "hinges_at_end" and h else h
                if not h:
                    continue
                for k, bb in h.items():
                    for b, v in bb.items():
                        acc.setdefault(k, {}).setdefault(int(b), []).append(v[:3])
            return {k: {b: [round(float(np.mean([v[i] for v in vs])), 1) for i in range(3)] + [len(vs)]
                        for b, vs in sorted(bb.items())} for k, bb in acc.items()}

        def yank_summary(eps):
            ys = [x for x in eps if x.get("yank_step", -1) >= 0]
            # held = still folded at every live step of the 180 after the yank, and lived >= 179 of them
            # (or the episode ended as a persist success, which needs 180 folded & clear steps).
            held = [x for x in ys if x["outcome"] == "persist" or
                    (x["yank_first_unfold"] is None and x["yank_live_steps"] >= 179)]
            def med(key, k):
                v = [x[key][k] for x in ys if x[key].get(k) is not None]
                return round(float(np.median(v)), 1) if v else None
            return {"n": len(ys), "held_180": len(held),
                    "outcomes": {o: sum(x["outcome"] == o for x in ys) for o in OUTCOMES},
                    "first_unfold_step_median": (round(float(np.median([x["yank_first_unfold"] for x in ys
                                                 if x["yank_first_unfold"] is not None])), 1)
                                                 if any(x["yank_first_unfold"] is not None for x in ys) else None),
                    "fold_err_mm_median": {k: med("yank_fold_err_mm", k) for k in ("+0", "+15", "+30", "+60", "+120", "+179")},
                    "opening_deg_median": {k: med("yank_opening_deg", k) for k in ("+0", "+30", "+60", "+120", "+179")},
                    "d_all_mm_median": {k: med("yank_d_all_mm", k) for k in ("+0", "+15", "+30", "+60")}}

        by_out = {}
        for o in OUTCOMES:
            sel = [x for x in rel_eps if x["outcome"] == o]
            if sel:
                by_out[o] = {"n": len(sel), **{k: agg(sel, k) for k in (
                    "opening_at_release", "opening_min_30_before", "opening_max_after", "opening_last",
                    "fold_err_max_after_mm", "d_all_min_after_mm", "press_peak_N", "press_p90_N",
                    "press_mean_touching_N", "press_impulse_Ns", "press_while_folded_mean_N",
                    "press_while_folded_peak_N", "pen_max_mm")},
                    "bend_at_release": bend_profile(sel, "hinges_at_release")}
        res = {
            "checkpoint": args.checkpoint, "seed": args.seed, "num_envs": n, "overrides": hydra_args,
            "steps_run": step, "yield_deg": {"parallel": round(float(yield_deg[kind == 0].mean()), 1),
                                            "diagonal": round(float(yield_deg[kind == 1].mean()), 1)},
            "bins_half_cells": BINS, "hinge_value_order": ["abs_theta_deg", "abs_rest_deg", "abs_elastic_deg", "n"],
            "outcomes": {o: outcome.count(o) for o in set(outcome)},
            "press_k_eff_N_per_m": k_eff, "sheet_mass_g": round(sheet_mass * 1e3, 2),
            "sheet_weight_N": round(sheet_mass * 9.81, 4),
            "released": len(rel_eps),
            "by_outcome": by_out,
            "hand_bodies": names_all,
            "episodes": episodes,
            "yank": args.yank, "yank_steps": args.yank_steps, "remove": args.remove,
            "yank_summary": yank_summary(episodes) if args.yank != "none" else None,
            "signed_trace": ({"edges": {"bin": bins[crease_sel & (e_env == 0)].tolist(),
                                        "kind": [KINDS[k] for k in kind[crease_sel & (e_env == 0)]]},
                              "envs": {int(e): v for e, v in sig_rec.items()}} if args.signed_trace > 0 else None),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        np.savez_compressed(args.out.with_suffix(".npz"), **tr, release_step=np.array(release_step),
                            hand_bodies=np.array(names_all))
        print("[spring] " + json.dumps({k: res[k] for k in ("outcomes", "released")}), flush=True)
        for o, v in by_out.items():
            print(f"[spring] {o}: n={v['n']} " + " ".join(f"{k}={v[k]['median']}" for k in (
                "opening_at_release", "opening_max_after", "opening_last", "d_all_min_after_mm",
                "press_peak_N", "press_while_folded_mean_N", "press_impulse_Ns", "pen_max_mm")), flush=True)
        if args.yank != "none":
            print("[spring] yank " + json.dumps(res["yank_summary"]), flush=True)
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
