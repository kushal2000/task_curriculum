"""GPU checks of the point-level observation DR (point_dr_mode; utils/point_dr.py, obs_utils).

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_point_dr.py \
        --check equiv|offpath|lags --experiment <EXP> --headless --num_envs 256 [--out r.json] [hydra overrides]

equiv    (launch with point_dr_mode=simple, lag [0,0], no noise/bias, obs/action delay off, obs_delay_max 4)
         check 1: point mode vs off on the same state, per policy field; object_rot sign.
         check 5: common bias only / per-point noise only / use_obs_delay with a fixed lag 3.
offpath  check 2: HEAD's build_observations (isaacsimenvs/tasks/play/utils/_obs_utils_head.py, a temporary
         copy made by the sbatch script) vs the working tree's, bitwise, same state and RNG.
lags     check 4: lag statistics from env._point_lag_last over --steps steps, plus an end-to-end check
         that keypoints_rel_goal is the clean one from `lag` steps ago (noise and biases zeroed at runtime).

--patch_off_init: works around reset_utils._init_point_dr swallowing the tail of allocate_state_buffers
when point_dr_mode is off (see report); without it an off-mode env cannot be built.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHumanLike-Direct-v0")
parser.add_argument("--experiment", default="flat7FlexivWujiRHL")
parser.add_argument("--check", required=True, choices=("equiv", "offpath", "lags"))
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--patch_off_init", action="store_true")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.tasks.play.utils import obs_utils as new_obs  # noqa: E402
from isaacsimenvs.tasks.play.utils import reset_utils  # noqa: E402
from isaacsimenvs.tasks.play.utils.obs_utils import OBS_FIELD_SIZES, _PER_JOINT_FIELDS  # noqa: E402
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml  # noqa: E402

print(f"[probe] obs_utils from {new_obs.__file__}", flush=True)

if args_cli.patch_off_init:
    _orig_init = reset_utils._init_point_dr

    def _patched_init(env):
        dr = env.cfg.domain_randomization
        if str(dr.point_dr_mode) != "off":
            return _orig_init(env)
        keys = ("point_dr_mode", "use_object_state_delay_noise", "object_pose_bias_xyz",
                "object_pose_bias_rot_degrees", "point_delay_range")
        saved = {k: getattr(dr, k) for k in keys}
        dr.point_dr_mode, dr.use_object_state_delay_noise = "simple", False
        dr.object_pose_bias_xyz, dr.object_pose_bias_rot_degrees, dr.point_delay_range = (0.0, 0.0, 0.0), 0.0, (0, 0)
        try:
            _orig_init(env)  # runs the swallowed tail (wrench DR state, caches, reward_buf)
        finally:
            for k, v in saved.items():
                setattr(dr, k, v)
            for a in ("_point_queue_len", "_point_bias", "_point_common_bias", "_point_queue", "_proprio_queue"):
                if hasattr(env, a):
                    delattr(env, a)
        print("[probe] patched _init_point_dr ran the allocate_state_buffers tail for point_dr_mode=off", flush=True)

    reset_utils._init_point_dr = _patched_init


def r6(x):
    return float(f"{float(x):.6g}")


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    dr = env_cfg.domain_randomization
    report = {"check": args_cli.check, "experiment": args_cli.experiment, "overrides": hydra_args,
              "num_envs": args_cli.num_envs,
              "dr": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(dr).items()
                     if k.startswith(("point_", "use_", "delay_mode", "obs_delay", "action_delay", "object_", "joint_"))}}
    print("[probe] dr", json.dumps(report["dr"]), flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    dev, n = inner.device, inner.num_envs
    dr = inner.cfg.domain_randomization
    n_act = int(inner.cfg.action_space)

    sizes = {**OBS_FIELD_SIZES, **{f: inner.robot.num_joints for f in _PER_JOINT_FIELDS}}
    slices, off = {}, 0
    for f in inner.cfg.obs.obs_list:
        slices[f] = slice(off, off + sizes[f])
        off += sizes[f]
    report["obs_list"] = list(inner.cfg.obs.obs_list)

    act = torch.zeros(n, n_act, device=dev)

    def step():
        nonlocal act
        act = (act + 0.3 * torch.randn_like(act)).clamp(-1, 1)
        obs, _, term, trunc, _ = env.step(act)
        return obs["policy"], term | trunc

    def per_field(a, b, mask=None):
        out = {}
        for f, s in slices.items():
            d = (a[:, s] - b[:, s]).abs()
            if mask is not None:
                d = d[mask]
            out[f] = r6(d.max()) if d.numel() else None
        return out

    def clean_build():
        """Off path, no delay: touches no queue (pose DR is off in these recipes)."""
        saved = (dr.point_dr_mode, dr.use_obs_delay)
        dr.point_dr_mode, dr.use_obs_delay = "off", False
        try:
            return new_obs.build_observations(inner)
        finally:
            dr.point_dr_mode, dr.use_obs_delay = saved

    def finite(t):
        return torch.isfinite(t).all(dim=-1)

    env.reset()
    for _ in range(args_cli.warmup):
        step()

    if args_cli.check == "equiv":
        assert dr.point_dr_mode == "simple" and tuple(dr.point_delay_range) == (0, 0)
        # ---- check 1 -------------------------------------------------------------------------------
        worst, worst_crit, sign = {}, 0.0, {"min_dot": 1.0, "flips": 0}
        track = []
        for snap in range(5):
            for _ in range(10):
                step()
            dr.point_dr_mode = "simple"
            p = new_obs.build_observations(inner)
            dr.point_dr_mode = "off"
            c = new_obs.build_observations(inner)
            dr.point_dr_mode = "simple"
            ok = finite(p["policy"]) & finite(c["policy"])
            pf = per_field(p["policy"], c["policy"], ok)
            for f, v in pf.items():
                worst[f] = max(worst.get(f, 0.0), v)
            worst_crit = max(worst_crit, float((p["critic"] - c["critic"]).abs()[ok].max()))
            s = slices["object_rot"]
            dot = (p["policy"][:, s] * c["policy"][:, s]).sum(-1)[ok]
            sign["min_dot"] = min(sign["min_dot"], r6(dot.min()))
            sign["flips"] += int((dot < 0).sum())
            # Are the tracked corners the rigid reconstruction?
            origins = inner.scene.env_origins
            tracked = inner._tracked_points_w() - origins.unsqueeze(1)
            kp_off = inner._keypoint_offsets * inner._object_scale_multiplier.unsqueeze(1)
            obj_kp = new_obs._keypoints_world(inner.object.data.root_pos_w - origins,
                                              inner.object.data.root_quat_w, kp_off)
            track.append(float((tracked - obj_kp).norm(dim=-1)[ok].max()))
            print(f"[probe] check1 snap {snap}: non-finite envs {int((~ok).sum())} {json.dumps(pf)}", flush=True)
        report["check1"] = {"max_abs_diff_per_field": worst, "critic_max_abs_diff": r6(worst_crit),
                            "object_rot_sign": sign,
                            "tracked_corners_minus_pose_keypoints_max_m": r6(max(track)),
                            "snapshots": 5}

        # ---- check 5a: common bias only -----------------------------------------------------------
        half = torch.tensor((0.015, 0.015, 0.01), device=dev)
        b = (torch.rand(n, 3, device=dev) * 2 - 1) * half
        inner._point_bias.zero_()
        inner._point_common_bias[:] = b
        p = new_obs.build_observations(inner)["policy"]
        c = clean_build()["policy"]
        ok = finite(p) & finite(c)
        sg, sp = slices["keypoints_rel_goal"], slices["keypoints_rel_palm"]
        rel_palm_off = (p[:, sp] - c[:, sp]).view(n, -1, 3) - b.unsqueeze(1)
        report["check5a_common_bias"] = {
            "bias_abs_max_m": [r6(v) for v in b.abs().amax(0)],
            "keypoints_rel_goal_max_abs_diff": r6((p[:, sg] - c[:, sg]).abs()[ok].max()),
            "keypoints_rel_palm_minus_clean_minus_bias_max_abs": r6(rel_palm_off.abs()[ok].max()),
            "keypoints_rel_palm_minus_clean_max_abs (= bias)": r6((p[:, sp] - c[:, sp]).abs()[ok].max()),
            "object_rot_max_abs_diff": r6((p[:, slices["object_rot"]] - c[:, slices["object_rot"]]).abs()[ok].max()),
            "all_fields": per_field(p, c, ok),
        }
        inner._point_common_bias.zero_()

        # ---- check 5b: per-point noise only -------------------------------------------------------
        dr.point_noise_std = 0.005
        c = clean_build()["policy"]
        diffs, rot_err = [], []
        for _ in range(20):
            p = new_obs.build_observations(inner)["policy"]
            ok = finite(p) & finite(c)
            d = (p[:, sp] - c[:, sp]).view(n, -1, 3)[ok]  # noisy kp - clean kp (palm identical)
            diffs.append(d)
            q1, q2 = p[:, slices["object_rot"]][ok], c[:, slices["object_rot"]][ok]
            rot_err.append(torch.rad2deg(2 * torch.acos((q1 * q2).sum(-1).abs().clamp(max=1.0))))
        d = torch.cat(diffs)
        re = torch.cat(rot_err)
        report["check5b_point_noise"] = {
            "point_noise_std_m": 0.005,
            "builds": 20,
            "std_noisy_minus_true_kp_m_per_axis": [r6(v) for v in d.reshape(-1, 3).std(0)],
            "std_noisy_minus_true_kp_m_all": r6(d.std()),
            "rms_3d_error_m": r6(d.norm(dim=-1).pow(2).mean().sqrt()),
            "expected_std_rigid_fit_K4 (sigma*sqrt(1/2))": r6(0.005 * 0.5 ** 0.5),
            "object_rot_error_deg_mean": r6(re.mean()), "object_rot_error_deg_p95": r6(re.quantile(0.95)),
        }
        dr.point_noise_std = 0.0

        # ---- check 5c: use_obs_delay with a fixed lag 3 -------------------------------------------
        LAG = 3
        dr.use_obs_delay, dr.delay_mode = True, "per_episode"
        inner._obs_delay_idx[:] = LAG
        hist = []
        stats = {"prev_action_targets_max_abs_diff": 0.0, "palm_pos_vs_clean_t_minus_3_max_abs": 0.0,
                 "joint_pos_vs_clean_t_minus_3_max_abs": 0.0, "palm_pos_vs_clean_now_mean_abs": [],
                 "frac_envs_palm_pos_differs_from_now_gt_1e-4": [],
                 "kp_rel_palm_plus_palm_vs_clean_max_abs": 0.0,
                 "keypoints_rel_goal_vs_clean_now_max_abs": 0.0, "object_rot_vs_clean_now_max_abs": 0.0,
                 "palm_vel_vs_clean_t_minus_3_max_abs": 0.0}
        n_used = 0
        for t in range(40):
            pol, _ = step()
            inner._obs_delay_idx[:] = LAG
            c = clean_build()["policy"]
            hist.append(c.clone())
            if t < 8:
                continue
            m = (inner.episode_length_buf > LAG + 1) & finite(pol) & finite(c) & finite(hist[-1 - LAG])
            n_used += int(m.sum())
            old = hist[-1 - LAG]

            def mx(key, a, bb):
                stats[key] = max(stats[key], float((a - bb).abs()[m].max()))
            mx("prev_action_targets_max_abs_diff", pol[:, slices["prev_action_targets"]], c[:, slices["prev_action_targets"]])
            mx("palm_pos_vs_clean_t_minus_3_max_abs", pol[:, slices["palm_pos"]], old[:, slices["palm_pos"]])
            mx("joint_pos_vs_clean_t_minus_3_max_abs", pol[:, slices["joint_pos"]], old[:, slices["joint_pos"]])
            if "palm_vel" in slices:
                mx("palm_vel_vs_clean_t_minus_3_max_abs", pol[:, slices["palm_vel"]], old[:, slices["palm_vel"]])
            mx("keypoints_rel_goal_vs_clean_now_max_abs", pol[:, sg], c[:, sg])
            mx("object_rot_vs_clean_now_max_abs", pol[:, slices["object_rot"]], c[:, slices["object_rot"]])
            a = pol[:, sp].view(n, -1, 3) + pol[:, slices["palm_pos"]].unsqueeze(1)
            bb = c[:, sp].view(n, -1, 3) + c[:, slices["palm_pos"]].unsqueeze(1)
            stats["kp_rel_palm_plus_palm_vs_clean_max_abs"] = max(
                stats["kp_rel_palm_plus_palm_vs_clean_max_abs"], float((a - bb).abs()[m].max()))
            dn = (pol[:, slices["palm_pos"]] - c[:, slices["palm_pos"]]).abs()[m]
            stats["palm_pos_vs_clean_now_mean_abs"].append(float(dn.mean()))
            stats["frac_envs_palm_pos_differs_from_now_gt_1e-4"].append(float((dn.amax(-1) > 1e-4).float().mean()))
        stats["palm_pos_vs_clean_now_mean_abs"] = r6(sum(stats["palm_pos_vs_clean_now_mean_abs"]) / len(stats["palm_pos_vs_clean_now_mean_abs"]))
        stats["frac_envs_palm_pos_differs_from_now_gt_1e-4"] = r6(min(stats["frac_envs_palm_pos_differs_from_now_gt_1e-4"]))
        report["check5c_obs_delay_lag3"] = {k: (r6(v) if isinstance(v, float) else v) for k, v in stats.items()} | {
            "env_steps_used": n_used}

    elif args_cli.check == "offpath":
        from isaacsimenvs.tasks.play.utils import _obs_utils_head as old_obs
        print(f"[probe] HEAD obs_utils from {old_obs.__file__}", flush=True)
        assert dr.point_dr_mode == "off"
        qnames = [q for q in ("_obs_queue", "_object_state_queue", "_action_queue") if hasattr(inner, q)]
        results = []
        for snap in range(8):
            for _ in range(5):
                step()
            saved = {q: getattr(inner, q).clone() for q in qnames}
            torch.manual_seed(1000 + snap)
            new = new_obs.build_observations(inner)
            q_new = {q: getattr(inner, q).clone() for q in qnames}
            for q in qnames:
                setattr(inner, q, saved[q].clone())
            torch.manual_seed(1000 + snap)
            old = old_obs.build_observations(inner)
            q_old = {q: getattr(inner, q).clone() for q in qnames}
            res = {"policy_bitwise_equal": bool(torch.equal(new["policy"], old["policy"])),
                   "critic_bitwise_equal": bool(torch.equal(new["critic"], old["critic"])),
                   "queues_bitwise_equal": {q: bool(torch.equal(q_new[q], q_old[q])) for q in qnames},
                   "policy_max_abs_diff": r6((new["policy"] - old["policy"]).abs().nan_to_num(0).max()),
                   "nan_envs": int((~finite(new["policy"])).sum())}
            results.append(res)
            print(f"[probe] offpath snap {snap}: {json.dumps(res)}", flush=True)
        report["check2"] = {"snapshots": results,
                            "all_policy_equal": all(r["policy_bitwise_equal"] for r in results),
                            "all_critic_equal": all(r["critic_bitwise_equal"] for r in results),
                            "all_queues_equal": all(all(r["queues_bitwise_equal"].values()) for r in results),
                            "queues_compared": qnames}
        # Prove the DR is active: policy obs differ from the DR-free build on some fields.
        p = new_obs.build_observations(inner)["policy"]
        saved = {k: getattr(dr, k) for k in ("use_obs_delay", "use_object_state_delay_noise",
                                            "joint_pos_obs_noise_std", "joint_velocity_obs_noise_std")}
        for k in saved:
            setattr(dr, k, False if k.startswith("use_") else 0.0)
        b_pos, b_q = inner._object_bias_pos.clone(), inner._object_bias_quat.clone()
        inner._object_bias_pos.zero_(); inner._object_bias_quat.zero_(); inner._object_bias_quat[:, 0] = 1.0
        c = new_obs.build_observations(inner)["policy"]
        inner._object_bias_pos[:], inner._object_bias_quat[:] = b_pos, b_q
        for k, v in saved.items():
            setattr(dr, k, v)
        report["check2"]["dr_active_max_abs_diff_vs_dr_free_per_field"] = per_field(p, c, finite(p) & finite(c))

    else:  # lags
        mode = str(dr.point_dr_mode)
        # Zero the noise and biases at runtime so the end-to-end check sees only the lag.
        dr.point_noise_std = 0.0
        lags, eplen, bases, ks, fs, ts = [], [], [], [], [], []
        hist = []  # clean keypoints_rel_goal per step
        e2e_max, e2e_n = 0.0, 0
        sg = slices["keypoints_rel_goal"]
        for t in range(args_cli.steps):
            pol, _ = step()
            inner._point_bias.zero_()
            inner._point_common_bias.zero_()
            lag = inner._point_lag_last.clone()
            lags.append(lag)
            eplen.append(inner.episode_length_buf.clone())
            if mode == "realistic":
                L = inner._point_lags
                bases.append(L.base.clone()); ks.append(L.k.clone()); fs.append(L.f.clone()); ts.append(L.t.clone())
            c = clean_build()["policy"]
            hist.append(c[:, sg].clone())
            # End-to-end: pol rel_goal == clean rel_goal from `lag` steps ago, for envs whose episode
            # started more than max-lag steps ago AND whose biases were already zero at that time.
            if t >= 12:
                el = inner.episode_length_buf
                m = (el > 12) & finite(pol)
                idx = torch.arange(n, device=dev)
                ref = torch.stack(hist[-11:], 0)  # (11, n, D); ref[-1 - lag]
                sel = ref[10 - lag.clamp(max=10), idx]
                m = m & finite(sel)
                if bool(m.any()):
                    e2e_max = max(e2e_max, float((pol[:, sg] - sel).abs()[m].max()))
                    e2e_n += int(m.sum())
        lags = torch.stack(lags).cpu()
        eplen = torch.stack(eplen).cpu()
        hist_counts = Counter(lags.flatten().tolist())
        tot = lags.numel()
        rep = {"mode": mode, "steps": args_cli.steps,
               "lag_hist": {int(k): hist_counts[k] for k in sorted(hist_counts)},
               "lag_frac": {int(k): r6(hist_counts[k] / tot) for k in sorted(hist_counts)},
               "lag_mean": r6(lags.float().mean()),
               "e2e_rel_goal_vs_clean_t_minus_lag_max_abs": r6(e2e_max), "e2e_env_steps": e2e_n,
               "resets_seen": int((eplen[1:] < eplen[:-1]).sum())}
        if mode == "realistic":
            B, K, F, T = (torch.stack(x).cpu() for x in (bases, ks, fs, ts))
            same_ep = (T[1:] == T[:-1] + 1)  # consecutive steps within one episode
            back = (F[1:] < F[:-1]) & same_ep
            changed = (F[1:] != F[:-1]) & same_ep
            rep["frame_steps_back_within_episode"] = int(back.sum())
            rep["t_equals_episode_length"] = bool(torch.equal(T, eplen.long()))
            rep["lag_equals_t_minus_f"] = bool(torch.equal(lags.long(), (T - F)))
            per_k = {}
            for kv in (1, 2, 3):
                mk = (K[1:] == kv) & same_ep
                per_k[kv] = {"env_steps": int(mk.sum()),
                             "frame_change_rate": r6(changed[mk].float().mean()) if bool(mk.any()) else None,
                             "expected_1_over_k": r6(1 / kv),
                             "mean_lag_minus_base": r6((lags[1:] - B[1:])[mk].float().mean()) if bool(mk.any()) else None,
                             "expected_approx_(k-1)/2": (kv - 1) / 2}
            rep["per_k"] = per_k
            per_base = {}
            for bv in range(int(B.min()), int(B.max()) + 1):
                mb = B == bv
                c2 = Counter(lags[mb].tolist())
                per_base[bv] = {"env_steps": int(mb.sum()), "mean_lag": r6(lags[mb].float().mean()),
                                "lag_hist": {int(k): c2[k] for k in sorted(c2)}}
            rep["per_base"] = per_base
            last = B[-1], K[-1]
            rep["draws_at_last_step"] = {"base_counts": dict(Counter(last[0].tolist())),
                                         "k_counts": dict(Counter(last[1].tolist()))}
            rep["lag_min"], rep["lag_max"] = int(lags.min()), int(lags.max())
        report["check4"] = rep

    text = json.dumps(report, indent=2, default=str)
    print(text, flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write(text + "\n")
    env.close()


if __name__ == "__main__":
    run()
    import os
    os._exit(0)
