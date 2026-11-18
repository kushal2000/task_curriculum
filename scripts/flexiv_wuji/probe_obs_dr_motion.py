"""Does an obs-DR-trained policy stop moving, and which observation-DR term causes it?

Runs fold-pool policies on the PhysX rigid pool (the env they trained in) for one episode per env,
under one observation condition (--cond), and records per env: success, best fold error, palm path
length / max displacement, arm and hand joint path, mean |action|, the fraction of steps the robot
is still, and the closest any fingertip came to the sheet. Plus a time profile (30-step bins).

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_obs_dr_motion.py \
        --headless --cond off --ckpts tagA=path/a.pth tagB=path/b.pth --out r.json

The env is always built from --experiment (default flat7RW_obsDR, the obs-DR run's own recipe); the
recipe's DR block is saved, everything is switched off with the eval protocol
(disable_randomization), then --cond turns selected terms back on. Reset noise stays at training
values (random starts) in every condition.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from isaaclab.app import AppLauncher

CONDS = (
    "off", "train_full", "obs_all", "pt_lag", "pt_lag0", "pt_lag2", "pt_lag5",
    "pt_common_bias", "pt_common_bias_xy", "pt_point_noise_bias", "proprio_delay", "joint_noise",
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0")
parser.add_argument("--experiment", default="flat7RW_obsDR")
parser.add_argument("--cond", required=True, choices=CONDS)
parser.add_argument("--ckpts", nargs="+", required=True, help="tag=path pairs")
parser.add_argument("--policy_config", default="pretrained_policy/config.yaml")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_steps", type=int, default=650)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", required=True)
parser.add_argument("--pinned", action="store_true", help="eval protocol start: pinned pose, no reset noise")
parser.add_argument("--single_variant", action="store_true")
parser.add_argument("--success_tolerance", type=float, default=None, help="termination.eval_success_tolerance")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
try:  # Kit for PhysX; the Newton (VBD) venv runs kit-less, as eval/episodes.py does
    app = AppLauncher(args_cli).app
except ImportError as exc:
    app = None
    print(f"[motion] kit-less: {exc}", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer  # noqa: E402
from isaacsimenvs.scripts.evaluation.protocol import disable_randomization, use_single_object_variant  # noqa: E402
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml  # noqa: E402

POINT_ZERO = dict(point_noise_std=0.0, point_bias_xyz=(0.0, 0.0, 0.0), point_common_bias_xyz=(0.0, 0.0, 0.0))


def apply_cond(dr, rec, cond: str) -> None:
    """``dr`` is already protocol-off; ``rec`` is the recipe's DR block."""
    def point_simple(lag: int, **kw):
        dr.point_dr_mode = "simple"
        dr.point_delay_range = (lag, lag)
        for k, v in {**POINT_ZERO, **kw}.items():
            setattr(dr, k, v)

    def point_realistic(**kw):
        dr.point_dr_mode = "realistic"
        for k in ("point_base_delay_range", "point_delay_jitter", "point_delay_max", "point_hold_choices"):
            setattr(dr, k, getattr(rec, k))
        for k, v in {**POINT_ZERO, **kw}.items():
            setattr(dr, k, v)

    if cond == "off":
        return
    if cond == "obs_all":
        point_realistic(point_noise_std=rec.point_noise_std, point_bias_xyz=rec.point_bias_xyz,
                        point_common_bias_xyz=rec.point_common_bias_xyz)
        dr.use_obs_delay, dr.obs_delay_max, dr.delay_mode = rec.use_obs_delay, rec.obs_delay_max, rec.delay_mode
        dr.joint_pos_obs_noise_std = rec.joint_pos_obs_noise_std
        dr.joint_velocity_obs_noise_std = rec.joint_velocity_obs_noise_std
    elif cond == "pt_lag":
        point_realistic()
    elif cond.startswith("pt_lag"):
        point_simple(int(cond[len("pt_lag"):]))
    elif cond == "pt_common_bias":
        point_simple(0, point_common_bias_xyz=rec.point_common_bias_xyz)
    elif cond == "pt_common_bias_xy":
        b = rec.point_common_bias_xyz
        point_simple(0, point_common_bias_xyz=(b[0], b[1], 0.0))
    elif cond == "pt_point_noise_bias":
        point_simple(0, point_noise_std=rec.point_noise_std, point_bias_xyz=rec.point_bias_xyz)
    elif cond == "proprio_delay":
        point_simple(0)  # point mode on, so use_obs_delay delays only the proprio fields, as in training
        dr.use_obs_delay, dr.obs_delay_max, dr.delay_mode = True, rec.obs_delay_max, rec.delay_mode
    elif cond == "joint_noise":
        dr.joint_pos_obs_noise_std = rec.joint_pos_obs_noise_std
        dr.joint_velocity_obs_noise_std = rec.joint_velocity_obs_noise_std
    else:
        raise ValueError(cond)


class _Only:
    """Hand disable_randomization only one section of the cfg (its reset half)."""

    def __init__(self, cfg, keep):
        import types
        self.domain_randomization = types.SimpleNamespace()
        setattr(self, keep, getattr(cfg, keep))


def r4(x) -> float:
    return round(float(x), 4)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment or None)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    rec = copy.deepcopy(env_cfg.domain_randomization)
    rs = env_cfg.reset
    reset_keep = {k: getattr(rs, k) for k in (
        "reset_position_noise_x", "reset_position_noise_y", "reset_position_noise_z",
        "reset_dof_pos_random_interval_arm", "reset_dof_pos_random_interval_fingers",
        "reset_dof_vel_random_interval", "table_reset_z_range", "fixed_start_pose")}
    if args_cli.cond == "train_full":
        pass  # the recipe's DR block as trained
    else:
        disable_randomization(env_cfg)
        apply_cond(env_cfg.domain_randomization, rec, args_cli.cond)
    if args_cli.pinned:
        disable_randomization(_Only(env_cfg, "reset"))
    else:
        for k, v in reset_keep.items():
            setattr(rs, k, v)
    if args_cli.success_tolerance is not None:
        env_cfg.termination.eval_success_tolerance = args_cli.success_tolerance
    if args_cli.single_variant:
        use_single_object_variant()
    dr = env_cfg.domain_randomization
    dr_report = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(dr).items()
                 if k.startswith(("point_", "use_", "delay_mode", "obs_delay", "action_delay", "joint_"))}
    print(f"[motion] cond={args_cli.cond} dr={json.dumps(dr_report)}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    dev, n = inner.device, inner.num_envs
    n_act = int(inner.cfg.action_space)
    arm = torch.zeros(inner.robot.num_joints, dtype=torch.bool, device=dev)
    arm[inner._arm_joint_ids] = True
    hand = ~arm
    dt = float(inner.step_dt)
    BIN = 30

    report = {"pinned": args_cli.pinned, "overrides": hydra_args, "cond": args_cli.cond, "experiment": args_cli.experiment, "task": args_cli.task,
              "num_envs": n, "seed": args_cli.seed, "dr": dr_report, "policies": {}}

    for spec in args_cli.ckpts:
        tag, path = spec.split("=", 1)
        torch.manual_seed(args_cli.seed)
        player = PretrainedPlayer(config_path=args_cli.policy_config, checkpoint_path=path, num_envs=n,
                                  device=str(dev), num_observations=int(inner.cfg.observation_space),
                                  num_actions=n_act, expl_coef=0.0)
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, n_act), device=dev))
        player.reset_rnn()

        alive = torch.ones(n, dtype=torch.bool, device=dev)
        success = torch.zeros(n, dtype=torch.bool, device=dev)
        length = torch.zeros(n, dtype=torch.long, device=dev)
        palm_path = torch.zeros(n, device=dev)
        palm_maxd = torch.zeros(n, device=dev)
        arm_path = torch.zeros(n, device=dev)
        hand_path = torch.zeros(n, device=dev)
        act_abs = torch.zeros(n, device=dev)
        still = torch.zeros(n, device=dev)
        ft_min = torch.full((n,), float("inf"), device=dev)
        best_fe = torch.full((n,), float("inf"), device=dev)
        bins: dict[int, list[float]] = {}

        def palm():
            return _t(inner.robot.data.body_state_w)[:, inner._palm_body_id, 0:3].clone()

        palm0, p_prev, q_prev = palm(), palm(), _t(inner.robot.data.joint_pos).clone()
        for step in range(args_cli.max_steps):
            if not bool(alive.any()):
                break
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _r, term, trunc, extras = env.step(action.to(dev))
            p, q = palm(), _t(inner.robot.data.joint_pos).clone()
            dp = (p - p_prev).norm(dim=-1)
            dq = (q - q_prev).abs()
            darm, dhand = dq[:, arm].sum(-1), dq[:, hand].sum(-1)
            a = alive.float()
            palm_path += dp * a
            palm_maxd = torch.where(alive, torch.maximum(palm_maxd, (p - palm0).norm(dim=-1)), palm_maxd)
            arm_path += darm * a
            hand_path += dhand * a
            act_abs += action.to(dev).abs().mean(-1) * a
            is_still = (dp / dt < 0.01) & (dq[:, hand].amax(-1) / dt < 0.1) & (dq[:, arm].amax(-1) / dt < 0.05)
            still += is_still.float() * a
            ft = _t(inner.robot.data.body_state_w)[:, inner._fingertip_body_ids, 0:3]
            parts = _t(inner._particles_w())
            ft_now = torch.cdist(ft, parts).amin(dim=(1, 2))
            fe_now = _t(inner.fold_error())
            ft_min = torch.where(alive, torch.minimum(ft_min, ft_now), ft_min)
            best_fe = torch.where(alive, torch.minimum(best_fe, fe_now), best_fe)
            palm_z = p[:, 2] - _t(inner.scene.env_origins)[:, 2]
            length += alive.long()
            b = step // BIN
            if int(alive.sum()) > 0:
                acc = bins.setdefault(b, [0.0] * 7)
                acc[0] += float((dp / dt)[alive].sum())
                acc[1] += float((dq[:, hand].mean(-1) / dt)[alive].sum())
                acc[2] += float(is_still[alive].float().sum())
                acc[3] += float(alive.sum())
                acc[4] += float(ft_now[alive].sum())
                acc[5] += float(fe_now[alive].sum())
                acc[6] += float(palm_z[alive].sum())
            done = (term.bool() | trunc.bool()) & alive
            if bool(done.any()):
                success |= done & (extras["episode_final"]["successes"] > 0)
                alive &= ~done
            p_prev, q_prev = p, q

        L = length.clamp(min=1).float()
        per = {
            "success": success.float(), "length": length.float(), "palm_path_m": palm_path,
            "palm_max_disp_m": palm_maxd, "arm_joint_path_rad": arm_path, "hand_joint_path_rad": hand_path,
            "mean_abs_action": act_abs / L, "still_frac": still / L, "ft_min_dist_m": ft_min,
            "best_fold_err_m": best_fe,
        }
        summ = {k: {"mean": r4(v.mean()), "p10": r4(v.quantile(0.1)), "p50": r4(v.median()),
                    "p90": r4(v.quantile(0.9))} for k, v in per.items()}
        # "Frozen": never moved the palm 3 cm from its start, or still >= 80% of the episode.
        summ["frozen_frac_palm_lt3cm"] = r4((palm_maxd < 0.03).float().mean())
        summ["frozen_frac_still_ge80"] = r4((still / L >= 0.8).float().mean())
        summ["censored"] = int(alive.sum())
        prof = [{"t_s": r4(b * BIN * dt), "palm_speed": r4(v[0] / v[3]), "hand_qd": r4(v[1] / v[3]),
                 "still": r4(v[2] / v[3]), "alive": int(v[3] / BIN),
                 "ft_dist": r4(v[4] / v[3]), "fold_err": r4(v[5] / v[3]), "palm_z": r4(v[6] / v[3])} for b, v in sorted(bins.items())]
        report["policies"][tag] = {"checkpoint": path, "summary": summ, "profile": prof,
                                   "per_env": {k: [r4(x) for x in v.cpu()] for k, v in per.items()}}
        print(f"[motion] {args_cli.cond} {tag}: success {summ['success']['mean']} palm_path "
              f"{summ['palm_path_m']['p50']} maxdisp {summ['palm_max_disp_m']['p50']} still "
              f"{summ['still_frac']['p50']} frozen {summ['frozen_frac_palm_lt3cm']}/{summ['frozen_frac_still_ge80']} "
              f"ft_min {summ['ft_min_dist_m']['p50']} fe {summ['best_fold_err_m']['p50']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[motion] wrote {args_cli.out}", flush=True)
    env.close()


run()
if app is not None:
    del app
sys.stdout.flush()
os._exit(0)
