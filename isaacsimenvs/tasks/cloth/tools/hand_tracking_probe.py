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
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
