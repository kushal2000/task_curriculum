"""Is a hand joint moving because the policy commands it, or because the physics throws it?

    .venv_isaaclab3/bin/python isaacsimenvs/tasks/cloth/tools/hand_tracking_probe.py \
        --task Isaacsimenvs-Cloth-Direct-v0 --policy_config checkpoints/config.yaml \
        --checkpoint <pth> --out x.json [--hold_hand] [--joints thumb] [hydra overrides...]

Per step and env, for the watched joints (name contains --joints): the commanded position target
(`_cur_targets`, after the action moving-average -- what the PD drive is told), the joint position and
velocity. A step where the joint moves more than --jump_deg is a JUMP; it is COMMANDED if the target moved
by at least half as much in the same direction that step, UNCOMMANDED otherwise. Large target-position
gaps (tracking error) also point at contact forces fighting the drive.

--hold_hand freezes every hand joint's target at its value after reset (the hand holds its reset pose;
the arm still follows the policy). Any remaining hand motion is then physics, not commands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from isaacsimenvs.scripts.evaluation.protocol import disable_randomization


def _force_stats(frc, rec, wnames, inner, jump, commanded, near, A, finite) -> dict:
    """Cloth torque about each thumb joint vs its torque limit, overall and at uncommanded jumps."""
    tau = np.abs(np.nan_to_num(np.stack(rec["tau"]), nan=0.0))              # (S, E, J)
    lf = np.nan_to_num(np.stack(rec["link_f"]), nan=0.0)                     # (S, E, L)
    alive = (A[..., 0].cpu().numpy()) & finite.cpu().numpy()[None, :]         # (S, E)
    # A jump at index t-1 is the motion over step t; take the larger torque of the step's two ends.
    tau_j = np.maximum(tau[1:], tau[:-1])
    unc = (jump & ~commanded).cpu().numpy()                                  # (S-1, E, J)
    near_cloth = near["cloth"].cpu().numpy() if near is not None else np.zeros(unc.shape[:2], bool)
    live = alive[1:] & alive[:-1]
    limits = dict(getattr(inner.robot_spec, "hand_effort_limit", {}) or {})
    q = lambda x, p: float(np.quantile(x, p)) if x.size else float("nan")  # noqa: E731
    out = {"cloth_weight_n": frc.cloth_weight_n, "sample": "last coupled substep of each policy step",
           "joints": {}, "links": {}}
    for j, nm in enumerate(wnames):
        lim = float(limits.get(nm, float("nan")))
        allv = tau[..., j][alive]
        touch = tau[..., j][alive & (tau[..., j] > 1e-6)]
        at_unc = tau_j[..., j][unc[..., j] & live]
        at_unc_cloth = tau_j[..., j][unc[..., j] & live & near_cloth]
        near_steps = tau_j[..., j][live & near_cloth]
        out["joints"][nm] = {
            "torque_limit_nm": lim,
            "frac_steps_loaded": float(touch.size / max(allv.size, 1)),
            "tau_loaded_median_nm": q(touch, 0.5), "tau_loaded_p95_nm": q(touch, 0.95),
            "tau_loaded_p99_nm": q(touch, 0.99), "tau_max_nm": float(allv.max()) if allv.size else float("nan"),
            "frac_steps_over_limit": float((allv > lim).mean()) if allv.size else float("nan"),
            "tau_near_cloth_median_nm": q(near_steps, 0.5), "tau_near_cloth_p95_nm": q(near_steps, 0.95),
            "tau_at_uncmd_jump_median_nm": q(at_unc, 0.5), "tau_at_uncmd_jump_p95_nm": q(at_unc, 0.95),
            "tau_at_uncmd_jump_near_cloth_median_nm": q(at_unc_cloth, 0.5),
            "tau_at_uncmd_jump_near_cloth_p95_nm": q(at_unc_cloth, 0.95),
            "n_uncmd_jumps_near_cloth": int(at_unc_cloth.size),
            "frac_uncmd_jumps_near_cloth_over_limit": float((at_unc_cloth > lim).mean()) if at_unc_cloth.size else float("nan"),
        }
        r = out["joints"][nm]
        print(f"[force] {nm:24s} limit {lim:.3f}  loaded {r['frac_steps_loaded']:.2f} of steps, tau med/p95/max "
              f"{r['tau_loaded_median_nm']:.3f}/{r['tau_loaded_p95_nm']:.3f}/{r['tau_max_nm']:.3f} N m, over limit "
              f"{r['frac_steps_over_limit']:.3f}; at near-cloth uncmd jumps med/p95 "
              f"{r['tau_at_uncmd_jump_near_cloth_median_nm']:.3f}/{r['tau_at_uncmd_jump_near_cloth_p95_nm']:.3f}", flush=True)
    for k, nm in enumerate(frc.link_names):
        v = lf[..., k][alive]
        loaded = v[v > 1e-6]
        out["links"][nm] = {"frac_steps_loaded": float(loaded.size / max(v.size, 1)),
                            "force_loaded_median_n": q(loaded, 0.5), "force_loaded_p95_n": q(loaded, 0.95),
                            "force_max_n": float(v.max()) if v.size else float("nan")}
    # Contact records per link: how many share the load, and how deep the deepest sits in the contact
    # distance (particle radius 8 mm + margin). The force is summed over a link's records.
    if rec.get("link_n"):
        n_last = np.stack(rec["link_n"]); n_max = np.stack(rec["link_n_max"])                    # (S, E, L)
        dep = np.stack(rec["link_depth_mm"])
        for k, nm in enumerate(frc.link_names):
            m = alive & (lf[..., k] > 1e-6)
            n_l, n_m, d = n_last[..., k][m], n_max[..., k][m], dep[..., k][m]
            d = d[np.isfinite(d)]
            per = lf[..., k][m & (n_last[..., k] > 0)] / n_last[..., k][m & (n_last[..., k] > 0)]
            out["links"][nm].update({
                "records_loaded_median": q(n_l, 0.5), "records_loaded_p95": q(n_l, 0.95),
                "records_loaded_max": float(n_l.max()) if n_l.size else float("nan"),
                "records_step_max_p95": q(n_m, 0.95),
                "force_per_record_p95_n": q(per, 0.95), "force_per_record_max_n": float(per.max()) if per.size else float("nan"),
                "depth_loaded_median_mm": q(d, 0.5), "depth_loaded_p95_mm": q(d, 0.95),
                "depth_max_mm": float(d.max()) if d.size else float("nan"),
                "frac_loaded_depth_over_4mm": float((d > 4.0).mean()) if d.size else float("nan"),
                "frac_loaded_depth_over_6mm": float((d > 6.0).mean()) if d.size else float("nan"),
            })
            r = out["links"][nm]
            if n_l.size:
                print(f"[force] {nm:24s} loaded: records med/p95/max {r['records_loaded_median']:.0f}/"
                      f"{r['records_loaded_p95']:.0f}/{r['records_loaded_max']:.0f}, force/record p95 "
                      f"{r['force_per_record_p95_n']:.2f} N, depth med/p95/max {r['depth_loaded_median_mm']:.2f}/"
                      f"{r['depth_loaded_p95_mm']:.2f}/{r['depth_max_mm']:.2f} mm", flush=True)
    # Where is the thumb when the cloth pushes on it? Per step, from the thumb capsules' SURFACES
    # (ThumbForceRecorder.sample_location), not the link origins the jump labels use:
    #   touching          a cloth particle centre within particle radius + 4 mm of a thumb capsule surface
    #   pressing_on_table touching, and a touching particle rests on the table (centre <= radius + 3 mm up)
    #   cloth_lifted      touching, none of the touching particles on the table
    #   table_only        not touching, lowest capsule surface point within 20 mm of the table top
    #   free              the rest
    # The force is VBD's cloth -> thumb feedback only (MuJoCo's direct thumb-table contact is not in it), so
    # table_only / free should read ~0 -- a check on the labels.
    if rec.get("th_gap"):
        gap = np.stack(rec["th_gap"]); sup = np.stack(rec["th_sup"]); low = np.stack(rec["th_low"])     # (S, E)
        S = min(gap.shape[0], lf.shape[0])
        r_p = frc.particle_radius
        touch = gap[:S] < r_p + 0.004
        press = touch & (sup[:S] <= r_p + 0.003)
        F = lf[:S].sum(-1)                                                    # (S, E) summed over thumb links
        al = alive[:S]
        dmax = None
        if rec.get("link_depth_mm"):
            dd = np.stack(rec["link_depth_mm"])[:S]
            dmax = np.where(np.isfinite(dd).any(-1), np.nanmax(np.where(np.isfinite(dd), dd, -1e9), axis=-1), np.nan)
        total = float(F[al].sum()) or 1.0
        out["by_location"] = {"definition": "thumb capsule surfaces (tools/thumb_force.py sample_location)",
                              "gap_mm_p05_p50": [q(gap[:S][al] * 1e3, 0.05), q(gap[:S][al] * 1e3, 0.5)],
                              "low_mm_p05_p50": [q(low[:S][al] * 1e3, 0.05), q(low[:S][al] * 1e3, 0.5)]}
        for name, msk in (("pressing_on_table", press), ("cloth_lifted", touch & ~press),
                          ("table_only", ~touch & (low[:S] < 0.020)), ("free", ~touch & (low[:S] >= 0.020))):
            m = msk & al
            v = F[m]
            row = {"frac_env_steps": float(m.sum() / max(al.sum(), 1)), "share_of_thumb_force": float(v.sum() / total),
                   "force_median_n": q(v, 0.5), "force_p95_n": q(v, 0.95),
                   "force_max_n": float(v.max()) if v.size else float("nan")}
            if dmax is not None:
                dv = dmax[m]; dv = dv[np.isfinite(dv)]
                row["depth_p95_mm"] = q(dv, 0.95)
            out["by_location"][name] = row
            print(f"[force] where {name:17s} {row['frac_env_steps']:.2f} of steps, {row['share_of_thumb_force']:.2f} of the "
                  f"force; thumb force med/p95/max {row['force_median_n']:.2f}/{row['force_p95_n']:.2f}/"
                  f"{row['force_max_n']:.2f} N; depth p95 {row.get('depth_p95_mm', float('nan')):.2f} mm", flush=True)
        print(f"[force] thumb surface gap to cloth p05/p50 {out['by_location']['gap_mm_p05_p50']} mm, lowest surface "
              f"above table p05/p50 {out['by_location']['low_mm_p05_p50']} mm", flush=True)
    print(f"[force] cloth weight {frc.cloth_weight_n * 1000:.1f} mN; thumb link force p95 / max [N]: "
          + ", ".join(f"{nm} {out['links'][nm]['force_loaded_p95_n']:.3f}/{out['links'][nm]['force_max_n']:.3f}"
                      for nm in frc.link_names), flush=True)
    return out


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--joints", default="thumb", help="substring of the joint names to watch")
    p.add_argument("--jump_deg", type=float, default=5.0, help="per-step joint motion counted as a jump")
    p.add_argument("--hold_hand", action="store_true")
    p.add_argument("--excursion_deg", type=float, default=10.0, help="tracking error that starts an excursion")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forces", action="store_true",
                   help="also record the cloth's torque about each watched joint and force on each thumb link "
                        "(tools/thumb_force.py; VBD only)")
    p.add_argument("--health", action="store_true",
                   help="also record cloth health per step: particle speed, edge strain, exit kick / cap cuts, "
                        "self-contact overflow (tools/cloth_health.py; VBD only)")
    p.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        AppLauncher(args)
    except ImportError:
        pass

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        disable_randomization(env_cfg)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n, device="cuda:0",
            num_observations=int(inner.cfg.observation_space), num_actions=int(inner.cfg.action_space), expl_coef=0.0,
        )
        names = list(inner.robot.joint_names)
        hand = list(inner.robot_layout.hand_joint_ids)
        watch = [i for i in hand if args.joints in names[i]]
        if not watch:
            raise SystemExit(f"no hand joint contains {args.joints!r}: {[names[i] for i in hand]}")
        wnames = [names[i] for i in watch]
        widx = torch.tensor(watch, device=dev)
        lo = inner.robot_layout.hand_lower[:, [hand.index(i) for i in watch]]
        hi = inner.robot_layout.hand_upper[:, [hand.index(i) for i in watch]]
        print(f"[probe] watching {wnames}; hold_hand={args.hold_hand}", flush=True)

        obs, _ = env.reset()
        n_arm = len(inner.robot_layout.arm_joint_ids)
        hold = None
        if args.hold_hand:
            hp = inner.controller.targets[:, inner.robot_layout.hand_joint_ids].clone()
            hold = (2.0 * (hp - inner.robot_layout.hand_lower) / (inner.robot_layout.hand_upper - inner.robot_layout.hand_lower) - 1.0).clamp(-1, 1)

        rec = {k: [] for k in ("tgt", "q", "qd")}
        # Where is the thumb tip at each step? Height above the table top and distance to the nearest cloth
        # particle (VBD only). A jump is then labelled near-table / near-cloth / free.
        tip_ids = inner.robot.find_bodies(".*thumb.*")[0]   # every thumb link: contact can land on any of them
        tip = tip_ids if tip_ids else None
        table_half = float(getattr(getattr(inner.cfg, "cloth", None), "table_half_thickness", 0.0) or 0.0)
        has_cloth = args.task == "Isaacsimenvs-Cloth-Direct-v0"
        rec["tip_h"], rec["tip_cloth"] = [], []
        frc = None
        if args.forces:
            from isaacsimenvs.tasks.cloth.tools.thumb_force import ThumbForceRecorder
            frc = ThumbForceRecorder(inner, wnames)
            rec["tau"], rec["link_f"] = [], []
        hlt = None
        if args.health:
            from isaacsimenvs.tasks.cloth.tools.cloth_health import ClothHealthRecorder
            hlt = ClothHealthRecorder(inner)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        for step in range(args.max_steps):
            a = player.get_action(obs["policy"], deterministic=True).to(dev)
            if hold is not None:
                a = a.clone()
                a[:, n_arm:] = hold
            obs, _r, term, trunc, _x = env.step(a)
            ended = term.bool() | trunc.bool()
            alive &= ~ended  # first episode only; a reset teleports the hand
            rec["tgt"].append(inner.controller.targets[:, widx].clone())
            rec["q"].append(inner.robot.data.joint_pos[:, widx].clone())
            rec["qd"].append(inner.robot.data.joint_vel[:, widx].clone())
            rec.setdefault("alive", []).append(alive.clone())
            if hlt is not None:
                hlt.sample()
            if frc is not None:
                tau_s, link_s = frc.sample()
                rec["tau"].append(tau_s); rec["link_f"].append(link_s)
                rr = frc.sample_records()
                if rr is not None:
                    for k, v in zip(("link_n", "link_n_max", "link_depth_mm"), rr):
                        rec.setdefault(k, []).append(v)
                top_z = inner.scene.env_origins[:, 2] + inner.task.table_z + table_half
                g_s, sup_s, low_s = frc.sample_location(inner.backend.read_points()[0], top_z)
                for k, v in (("th_gap", g_s), ("th_sup", sup_s), ("th_low", low_s)):
                    rec.setdefault(k, []).append(v.detach().cpu().numpy())
            if tip is not None:
                tp = inner.robot.data.body_pos_w[:, tip]                      # (E, L, 3)
                top = inner.scene.env_origins[:, 2] + inner.task.table_z + table_half
                rec["tip_h"].append((tp[..., 2] - top[:, None]).amin(dim=1))  # lowest thumb link above the table
                if has_cloth:
                    pw = inner.backend.read_points()[0]                                  # (E, P, 3)
                    rec["tip_cloth"].append(torch.cdist(tp, pw).amin(dim=(1, 2)))
            if not bool(alive.any()):
                break
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])

        T = torch.stack(rec["tgt"]); Q = torch.stack(rec["q"]); QD = torch.stack(rec["qd"])  # (S, E, J)
        A = torch.stack(rec["alive"])[..., None].expand_as(Q)
        # An env whose robot went non-finite is dropped entirely (its NaNs would poison every statistic).
        finite = torch.isfinite(Q).all(dim=0).all(dim=-1) & torch.isfinite(T).all(dim=0).all(dim=-1)  # (E,)
        n_nonfinite = int((~finite).sum())
        A = A & finite[None, :, None]
        Q = torch.nan_to_num(Q); T = torch.nan_to_num(T); QD = torch.nan_to_num(QD)
        dq, dt = Q[1:] - Q[:-1], T[1:] - T[:-1]
        live = A[1:] & A[:-1]
        jump = (dq.abs() > math.radians(args.jump_deg)) & live
        commanded = jump & (torch.sign(dt) == torch.sign(dq)) & (dt.abs() >= 0.5 * dq.abs())
        err = (Q - T).abs()
        over = torch.maximum(lo - Q, Q - hi).clamp(min=0)
        deg = lambda x: float(math.degrees(float(x)))  # noqa: E731
        out = {"checkpoint": args.checkpoint, "hold_hand": args.hold_hand, "overrides": hydra_args,
               "steps": int(T.shape[0]), "envs": n, "nonfinite_envs_dropped": n_nonfinite, "joints": {}}
        near = None
        if rec["tip_h"]:
            H = torch.nan_to_num(torch.stack(rec["tip_h"]), nan=1e9)[1:]  # (S-1, E)
            C = torch.nan_to_num(torch.stack(rec["tip_cloth"]), nan=1e9)[1:] if rec["tip_cloth"] else torch.full_like(H, 1e9)
            near_cloth = C < 0.025       # thumb tip within 25 mm of a cloth particle (radius + tip + slack)
            near_table = (H < 0.020) & ~near_cloth
            near = {"cloth": near_cloth, "table": near_table, "free": ~near_cloth & ~near_table}
        for j, nm in enumerate(wnames):
            lj = live[..., j]
            m = A[..., j]
            e = err[..., j][m]
            out["joints"][nm] = {
                "target_step_change_deg_mean": deg(dt[..., j][lj].abs().mean()),
                "joint_step_change_deg_mean": deg(dq[..., j][lj].abs().mean()),
                "jumps": int(jump[..., j].sum()),
                "jumps_commanded": int(commanded[..., j].sum()),
                "jumps_uncommanded": int((jump[..., j] & ~commanded[..., j]).sum()),
                "track_err_deg_median": deg(e.median()), "track_err_deg_p95": deg(e.quantile(0.95)),
                "track_err_deg_max": deg(e.max()),
                "speed_dps_p99": deg(QD[..., j][m].abs().quantile(0.99)),
                "limit_over_deg_max": deg(over[..., j][m].max()),
                "frac_steps_over_limit_1deg": float((over[..., j][m] > math.radians(1.0)).float().mean()),
            }
            if near is not None:
                unc = jump[..., j] & ~commanded[..., j]
                out["joints"][nm]["uncommanded_jumps_by_location"] = {k: int((unc & v).sum()) for k, v in near.items()}
                out["joints"][nm]["commanded_jumps_by_location"] = {k: int((commanded[..., j] & v).sum()) for k, v in near.items()}
            r = out["joints"][nm]
            print(f"[probe] {nm:28s} jumps {r['jumps']:4d} (cmd {r['jumps_commanded']}, uncmd "
                  f"{r['jumps_uncommanded']})  track err med/p95/max {r['track_err_deg_median']:.1f}/"
                  f"{r['track_err_deg_p95']:.1f}/{r['track_err_deg_max']:.1f} deg  limit over max "
                  f"{r['limit_over_deg_max']:.1f}  uncmd by loc {r.get('uncommanded_jumps_by_location')}", flush=True)
        # SURFACE LOCATION (2026-11-26, needs --forces): the same jumps labelled from the thumb capsules' surfaces
        # (ThumbForceRecorder.sample_location) instead of the link origins above, which sit at the joints and miss
        # pad contact. A jump at index t-1 is the motion over step t; it counts as contact if the thumb touched the
        # cloth at EITHER end of that step (a throw can end the step clear of the cloth). Same definitions as
        # forces.by_location: touching = a particle centre within radius + 4 mm of a capsule surface; pressing =
        # touching and a touching particle rests on the table; table = not touching, capsule within 20 mm of the
        # table top. The old labels are kept unchanged for comparison with earlier rounds.
        if frc is not None and rec.get("th_gap"):
            r_p = frc.particle_radius
            g_ = np.stack(rec["th_gap"]); sp_ = np.stack(rec["th_sup"]); lo_ = np.stack(rec["th_low"])   # (S, E)
            tch = g_ < r_p + 0.004
            prs = tch & (sp_ <= r_p + 0.003)
            ends = lambda x: x[1:] | x[:-1]  # noqa: E731
            t_any, p_any = ends(tch), ends(prs)
            tab = ~t_any & ends(lo_ < 0.020)
            surf = {"cloth_pressing": p_any, "cloth_lifted": t_any & ~p_any, "table": tab, "free": ~t_any & ~tab}
            J = jump.cpu().numpy(); Cm = commanded.cpu().numpy()
            for j, nm in enumerate(wnames):
                unc = J[..., j] & ~Cm[..., j]
                out["joints"][nm]["uncommanded_jumps_by_surface_location"] = {k: int((unc & v).sum()) for k, v in surf.items()}
                out["joints"][nm]["commanded_jumps_by_surface_location"] = {k: int((Cm[..., j] & v).sum()) for k, v in surf.items()}
                print(f"[probe] {nm:28s} uncmd by SURFACE loc {out['joints'][nm]['uncommanded_jumps_by_surface_location']}", flush=True)
            live_np = live[..., 0].cpu().numpy()
            out["surface_location_steps"] = {k: float((v & live_np).sum() / max(live_np.sum(), 1)) for k, v in surf.items()}
        # EXCURSION ONSETS: a tracking error that first exceeds --excursion_deg. Its cause is judged at onset:
        # "target" if the target jumped >= jump_deg that step (the drive could not follow a commanded move),
        # "contact" if any thumb link was near cloth/table within +-2 steps of onset, both, or neither. Every jump
        # (commanded or not) inside the excursion is attributed to that onset cause.
        if near is not None:
            thr = math.radians(args.excursion_deg)
            E_ = err.cpu().numpy(); Tn = T.cpu().numpy(); An = A.cpu().numpy()
            contact = (near["cloth"] | near["table"]).cpu().numpy()        # (S-1, E), aligned to step 1..S-1
            contact = np.concatenate([contact[:1], contact], 0)             # (S, E)
            Jn = np.concatenate([np.zeros_like(jump[:1].cpu().numpy()), jump.cpu().numpy()], 0)  # (S, E, J)
            S_, Ee, Jj = E_.shape
            cause_names = ("target", "contact", "both", "neither")
            for j, nm in enumerate(wnames):
                onsets = dict.fromkeys(cause_names, 0); jumps_by = dict.fromkeys(cause_names, 0)
                for e in range(Ee):
                    cur = None
                    for t in range(1, S_):
                        if not An[t, e, j]:
                            cur = None; continue
                        if E_[t, e, j] >= thr and E_[t - 1, e, j] < thr:
                            tj = abs(Tn[t, e, j] - Tn[t - 1, e, j]) >= math.radians(args.jump_deg)
                            ct = bool(contact[max(0, t - 2):t + 3, e].any())
                            cur = "both" if (tj and ct) else "target" if tj else "contact" if ct else "neither"
                            onsets[cur] += 1
                        elif E_[t, e, j] < 0.5 * thr:
                            cur = None
                        if cur is not None and Jn[t, e, j]:
                            jumps_by[cur] += 1
                out["joints"][nm]["excursion_onsets_by_cause"] = onsets
                out["joints"][nm]["jumps_in_excursions_by_onset_cause"] = jumps_by
                print(f"[probe] {nm:28s} excursions>{args.excursion_deg:.0f}deg onsets {onsets}  jumps inside {jumps_by}", flush=True)
        if frc is not None:
            out["forces"] = _force_stats(frc, rec, wnames, inner, jump, commanded, near, A, finite)
        if hlt is not None:
            unc_near = None
            if near is not None:
                unc_near = ((jump & ~commanded) & near["cloth"][..., None]).sum(-1).cpu().numpy()
            out["cloth_health"] = hlt.summary(torch.stack(rec["alive"]).cpu().numpy(), finite.cpu().numpy(), unc_near)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
