"""Evaluate a joint-reach policy in sim, and export the goal sequence for the real robot.

Goals are held-out random-walk chains (JointReachEnv.build_goal_chains, a seed training did not use):
half start at the default pose, half at held-out workspace configurations. Every env starts at its
chain's first configuration and follows --segments goals, each held for the arm's travel time at the
pipeline's speed limit plus --settle_s. Per goal segment it records
the error at the end of the hold, whether every joint is within tolerance there, the time to first
enter tolerance, and the mean joint speed over the last 0.5 s (oscillation). One run per condition:

    dr        training DR (delays, noise, random gains)
    nodr      no delays, no noise, nominal gains
    gain0.5   training DR, every gain x0.5
    gain2     training DR, every gain x2
    handS2R   training DR, arm nominal, hand stiffness x3.5 / damping x2.5 (Wuji's sim2real drive)

Then the real-robot sequence: one held-out chain of --real_goals goals from the default pose (so it
starts small and wanders further), held for travel + --real_settle_s, rolled out with no DR in env 0, written with the full sim trajectory (obs, action, joint targets,
joint positions) so the deploy code can be checked against it offline and the hardware compared
against it online.

    .venv_isaacsim/bin/python -m isaacsimenvs.eval.joint_reach_eval --headless \
        --checkpoint <run>/nn/<ckpt>.pth --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TASK = "Isaacsimenvs-JointReachFlexivWujiHumanLike-Direct-v0"
CONDITIONS = {
    "dr": dict(dr=True, gains=(0.0, 0.0, 0.0, 0.0)),
    "nodr": dict(dr=False, gains=(1.0, 1.0, 1.0, 1.0)),
    "gain0.5": dict(dr=True, gains=(0.5, 0.5, 0.5, 0.5)),
    "gain2": dict(dr=True, gains=(2.0, 2.0, 2.0, 2.0)),
    "handS2R": dict(dr=True, gains=(1.0, 3.5, 1.0, 2.5)),
}


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy_config", default="pretrained_policy/config.yaml",
                        help="Any config.yaml of this network; only the network section is read.")
    parser.add_argument("--sapg_expl_coef", type=float, default=0.0)
    parser.add_argument("--num_envs", type=int, default=500)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--settle_s", type=float, default=2.0)
    parser.add_argument("--test_seed", type=int, default=1)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--real_goals", type=int, default=20)
    parser.add_argument("--real_settle_s", type=float, default=2.5)
    parser.add_argument("--out", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in args.conditions.split(",") if c]

    @hydra_task_config_with_yaml(TASK, "")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.episode_length_s = 3600.0  # the rollouts end themselves
        env_cfg.goal.default_start_prob = 1.0
        env_cfg.reset.reset_dof_pos_random_interval_arm = 0.0
        env_cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        env_cfg.reset.reset_dof_vel_random_interval = 0.0
        dr_on = env_cfg.domain_randomization.to_dict()

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        inner.external_goals = True
        dev = inner.device
        n_env = inner.num_envs
        all_ids = torch.arange(n_env, device=dev)
        names = list(inner.robot_spec.joint_names_canonical)

        test_bank = inner.build_goal_bank(n_env, args.test_seed)
        test_chains = inner.build_goal_chains(
            inner.chain_starts(test_bank, 0.5, args.test_seed), args.segments + 1, args.test_seed
        )
        default_starts = inner.default_goal().unsqueeze(0).repeat(n_env, 1)
        real_chain = inner.build_goal_chains(default_starts, args.real_goals + 1, args.test_seed + 7)[0]
        torch.save({"goals": test_chains.cpu(), "joint_names": names, "seed": args.test_seed}, out / "test_chains.pt")
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n_env,
            device=str(dev), num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
        )
        arm_tol, hand_tol = inner._arm_tol, inner._hand_tol

        def set_condition(name: str) -> None:
            c = CONDITIONS[name]
            dr = inner.cfg.domain_randomization
            for k, v in dr_on.items():
                setattr(dr, k, v)
            if not c["dr"]:
                dr.use_obs_delay = False
                dr.use_action_delay = False
                dr.action_noise_std = 0.0
                dr.joint_pos_obs_noise_std = 0.0
                dr.joint_velocity_obs_noise_std = 0.0
            inner.cfg.gain_dr.fixed_scales = tuple(c["gains"])

        def rollout(chains: torch.Tensor, settle_s: float, record: bool):
            """chains: (num_envs, K + 1, 27); env i starts at chains[i, 0] and follows chains[i, 1:].
            Returns per-segment metrics (num_envs, K) and, if `record`, env 0's trajectory."""
            k_seg = chains.shape[1] - 1
            env.reset()
            player.reset_rnn()
            inner.place_robot(all_ids, chains[:, 0])
            settle = torch.full((n_env,), settle_s, device=dev)
            seg = torch.zeros(n_env, dtype=torch.long, device=dev)
            inner._goal[:] = chains[:, 1]
            hold = inner.hold_steps(chains[:, 0], chains[:, 1], settle)
            t_in = torch.zeros(n_env, dtype=torch.long, device=dev)
            first_in = torch.full((n_env,), -1.0, device=dev)
            speed_sum = torch.zeros(n_env, device=dev)
            speed_n = torch.zeros(n_env, device=dev)
            obs = inner._get_observations()  # the obs queue is still flushed at episode start
            res = {key: torch.full((n_env, k_seg), float("nan"), device=dev) for key in
                   ("arm_err_deg", "hand_err_deg", "reached", "settle_s", "end_speed", "arm_move_rad")}
            traj = {key: [] for key in ("obs", "action", "target", "q", "qd", "goal")} if record else None
            max_steps = int(inner.hold_steps(chains[:, :-1].reshape(-1, 27), chains[:, 1:].reshape(-1, 27),
                                             torch.full((n_env * k_seg,), settle_s, device=dev))
                            .reshape(n_env, k_seg).sum(dim=1).max()) + 10
            for _ in range(max_steps):
                active = seg < k_seg
                if not bool(active.any()):
                    break
                action = player.get_action(obs["policy"], deterministic=True)
                if record:
                    traj["obs"].append(obs["policy"][0].cpu())
                    traj["action"].append(action[0].cpu())
                obs, *_ = env.step(action.to(dev))
                q = inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
                qd = inner.robot.data.joint_vel[:, inner._perm_lab_to_canon]
                if record:
                    traj["target"].append(inner._prev_targets[0, inner._perm_lab_to_canon].cpu())
                    traj["q"].append(q[0].cpu())
                    traj["qd"].append(qd[0].cpu())
                    traj["goal"].append(inner._goal[0].cpu())
                err = (inner._goal - q).abs()
                arm_e, hand_e = err[:, :7].max(-1).values, err[:, 7:].max(-1).values
                in_tol = (arm_e < arm_tol) & (hand_e < hand_tol)
                t_in += 1
                first_in = torch.where(in_tol & (first_in < 0) & active, t_in.float() / 60.0, first_in)
                tail = (hold - t_in < 30) & active
                speed_sum += torch.where(tail, qd.abs().mean(dim=-1), torch.zeros_like(speed_sum))
                speed_n += tail.float()
                ended = (t_in >= hold) & active
                if bool(ended.any()):
                    ids = ended.nonzero(as_tuple=False).squeeze(-1)
                    k = seg[ids]
                    prev_goal = chains[ids, k]
                    res["arm_err_deg"][ids, k] = torch.rad2deg(arm_e[ids])
                    res["hand_err_deg"][ids, k] = torch.rad2deg(hand_e[ids])
                    res["reached"][ids, k] = in_tol[ids].float()
                    res["settle_s"][ids, k] = torch.where(first_in[ids] < 0, float("nan"), first_in[ids])
                    res["end_speed"][ids, k] = speed_sum[ids] / speed_n[ids].clamp(min=1)
                    res["arm_move_rad"][ids, k] = (chains[ids, k + 1, :7] - prev_goal[:, :7]).abs().max(-1).values
                    seg[ids] += 1
                    more = ids[seg[ids] < k_seg]
                    if more.numel() > 0:
                        nxt = chains[more, seg[more] + 1]
                        hold[more] = inner.hold_steps(inner._goal[more], nxt, settle[more])
                        inner._goal[more] = nxt
                    t_in[ids] = 0
                    first_in[ids] = -1.0
                    speed_sum[ids] = 0.0
                    speed_n[ids] = 0.0
            if record:
                traj = {key: torch.stack(v) for key, v in traj.items()}
            return res, traj

        def summarise(res) -> dict:
            flat = {k: v.flatten() for k, v in res.items()}
            move = flat["arm_move_rad"]
            out_d = {}
            for label, mask in (
                ("all", torch.ones_like(move, dtype=torch.bool)),
                ("arm_move<0.1rad", move < 0.1),
                ("arm_move0.1-0.25rad", (move >= 0.1) & (move < 0.25)),
                ("arm_move>=0.25rad", move >= 0.25),
            ):
                if int(mask.sum()) == 0:
                    continue
                settle = flat["settle_s"][mask]
                settle = settle[~torch.isnan(settle)]
                out_d[label] = {
                    "segments": int(mask.sum()),
                    "reached": float(flat["reached"][mask].mean()),
                    "arm_err_deg_median": float(flat["arm_err_deg"][mask].median()),
                    "arm_err_deg_p90": float(flat["arm_err_deg"][mask].quantile(0.9)),
                    "hand_err_deg_median": float(flat["hand_err_deg"][mask].median()),
                    "hand_err_deg_p90": float(flat["hand_err_deg"][mask].quantile(0.9)),
                    "settle_s_median": float(settle.median()) if settle.numel() else float("nan"),
                    "end_speed_rad_s_median": float(flat["end_speed"][mask].median()),
                }
            return out_d

        summary = {"checkpoint": args.checkpoint, "sapg_expl_coef": args.sapg_expl_coef,
                   "settle_s": args.settle_s, "segments": args.segments, "num_envs": n_env,
                   "arm_tol_deg": math.degrees(arm_tol), "hand_tol_deg": math.degrees(hand_tol),
                   "conditions": {}}
        for name in conditions:
            set_condition(name)
            res, _ = rollout(test_chains, args.settle_s, record=False)
            summary["conditions"][name] = summarise(res)
            torch.save({k: v.cpu() for k, v in res.items()}, out / f"segments_{name}.pt")
            print(f"[jr_eval] {name}: {json.dumps(summary['conditions'][name]['all'])}", flush=True)

        # Real-robot sequence: a held-out chain from the default pose, no DR, env 0 recorded.
        set_condition("nodr")
        res, traj = rollout(real_chain.unsqueeze(0).repeat(n_env, 1, 1), args.real_settle_s, record=True)
        q_default = inner.default_goal()
        dist = (real_chain[1:, :7] - q_default[:7]).abs().max(dim=-1).values
        hold = inner.hold_steps(real_chain[:-1], real_chain[1:], torch.full((args.real_goals,), args.real_settle_s, device=dev))
        torch.save(
            {
                "start": real_chain[0].cpu(), "goals": real_chain[1:].cpu(), "hold_steps": hold.cpu(),
                "settle_s": args.real_settle_s, "joint_names": names, "default_q": q_default.cpu(),
                "arm_dist_from_default_rad": dist.cpu(),
                "sim_trajectory": traj, "sim_segments": {k: v[0].cpu() for k, v in res.items()},
                "policy_dt": inner.step_dt, "checkpoint": args.checkpoint,
                "sapg_expl_coef": args.sapg_expl_coef,
            },
            out / "real_test_sequence.pt",
        )
        summary["real_sequence"] = {
            "goals": args.real_goals,
            "duration_s": round(float(hold.sum()) * inner.step_dt, 1),
            "arm_dist_from_default_rad": [round(float(v), 3) for v in dist],
            "reached": [int(v) for v in res["reached"][0]],
            "arm_err_deg": [round(float(v), 2) for v in res["arm_err_deg"][0]],
            "hand_err_deg": [round(float(v), 2) for v in res["hand_err_deg"][0]],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[jr_eval] wrote {out / 'summary.json'}", flush=True)

    run()
    import os

    os._exit(0)  # Kit shutdown hangs on these nodes


if __name__ == "__main__":
    main()
