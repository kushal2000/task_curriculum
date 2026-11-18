"""Probe table-height DR and the robot base height on a cloth task, with the recipe's DR ON.

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_table_dr.py \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHumanLike-Direct-v0 --experiment flat7FlexivWujiRHL \
        --headless --num_envs 256 --resets 4 [--out report.json]

Per reset (all envs, the training reset distribution):
  * table top relative to the robot base: ``_table_z_per_env + 0.15 - base z``, and the table body's
    sim z against ``_table_z_per_env``
  * the sheet after ``--settle_steps`` (1 s): lowest particle z minus table top, minus the per-env
    contact margin draw; correlation of the lowest particle z with the table top
  * arm drift while holding the (noisy) reset pose
Then the HOME pose (robot default joints) written into every env with the table where it is:
  * arm drift while holding it, palm centre height above the table top, observed palm_pos
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHumanLike-Direct-v0")
parser.add_argument("--experiment", default="flat7FlexivWujiRHL")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--resets", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--home_steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.tasks.play.utils.obs_utils import OBS_FIELD_SIZES, _PER_JOINT_FIELDS  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

TABLE_HALF = 0.15


def _st(t):
    t = t.flatten().float()
    return {"min": round(float(t.min()), 4), "max": round(float(t.max()), 4),
            "mean": round(float(t.mean()), 4), "std": round(float(t.std()), 4)}


def _corr(a, b):
    a, b = a.flatten().float() - a.float().mean(), b.flatten().float() - b.float().mean()
    return round(float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12)), 4)


# The experiment carries an agent block, so an agent cfg must be loaded for it to merge into.
@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point", experiment=args_cli.experiment)
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    report = {"task": args_cli.task, "experiment": args_cli.experiment, "num_envs": args_cli.num_envs,
              "resolved_cfg": {
                  "robot_root_pos": list(env_cfg.robot_root_pos),
                  "table_reset_z": env_cfg.reset.table_reset_z,
                  "table_reset_z_range": env_cfg.reset.table_reset_z_range,
                  "cloth_start_height": env_cfg.cloth.start_height,
                  "reset_dof_pos_random_interval_arm": env_cfg.reset.reset_dof_pos_random_interval_arm,
              }}
    print("[probe] resolved cfg", report["resolved_cfg"], flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    spec, robot, device = inner.robot_spec, inner.robot, inner.device
    n, n_act = inner.num_envs, int(inner.cfg.action_space)
    origins = inner.scene.env_origins
    arm, hand = inner._arm_joint_ids, inner._hand_joint_ids
    lo, hi = inner._hand_lower, inner._hand_upper
    canon_hand = inner._perm_canon_to_lab[hand]

    # palm_pos's slice of the policy observation.
    sizes = {**OBS_FIELD_SIZES, **{f: robot.num_joints for f in _PER_JOINT_FIELDS}}
    off = 0
    for f in inner.cfg.obs.obs_list:
        if f == "palm_pos":
            break
        off += sizes[f]
    palm_slice = slice(off, off + 3) if "palm_pos" in inner.cfg.obs.obs_list else None

    def hold_action():
        a = torch.zeros((n, n_act), device=device)
        a[:, canon_hand] = (2.0 * (robot.data.joint_pos[:, hand] - lo) / (hi - lo) - 1.0).clamp(-1, 1)
        return a

    def base_z():
        return robot.data.root_pos_w[:, 2] - origins[:, 2]

    def palm_centre():
        b = robot.data.body_state_w[:, inner._palm_body_id]
        o = torch.tensor(spec.palm_center_offset, device=device).expand(n, 3)
        return b[:, :3] + quat_apply(b[:, 3:7], o) - origins

    def policy(obs):
        return obs["policy"] if isinstance(obs, dict) else obs

    root = (robot.data.root_pos_w - origins)
    report["robot_root_pos_sim"] = _st(root[:, 2]) | {"xy_env0": [round(float(v), 4) for v in root[0, :2]]}

    resets = []
    all_top, all_res, all_minz = [], [], []
    for r in range(args_cli.resets):
        obs, _ = env.reset()
        table_z = inner._table_z_per_env.clone()
        top = table_z + TABLE_HALF
        table_sim_z = inner.table.data.root_pos_w[:, 2] - origins[:, 2]
        q0 = robot.data.joint_pos.clone()
        # Hand clearance over the spawned sheet: the lowest fingertip pad and the lowest hand/arm
        # body origin, minus the sheet's highest particle, right after the reset (before any step).
        st = robot.data.body_state_w[:, inner._fingertip_body_ids]
        po = torch.tensor(spec.fingertip_offset, device=device).expand(st.shape[0], st.shape[1], 3)
        pads_z = (st[..., :3] + quat_apply(st[..., 3:7].reshape(-1, 4), po.reshape(-1, 3)).view_as(po))[..., 2]
        sheet_top0 = inner._particles_w()[..., 2].amax(dim=1)
        clear_pad = pads_z.amin(dim=1) - sheet_top0
        clear_body = robot.data.body_pos_w[..., 2].amin(dim=1) - sheet_top0
        act = hold_action()
        # Envs that terminate during the settle are re-spawned (new arm pose, new sheet in the air):
        # they are left out of the drift and sheet numbers and counted separately.
        reset_mid = torch.zeros(n, dtype=torch.bool, device=device)
        term_mid = torch.zeros(n, dtype=torch.bool, device=device)
        drift_e = torch.zeros(n, device=device)
        sheet_min_at_term = []
        for _ in range(args_cli.settle_steps):
            pre_min = inner._particles_w()[..., 2].amin(dim=1) - origins[:, 2] - top
            obs, _, term, trunc, _ = env.step(act)
            new = (term | trunc) & ~reset_mid
            if bool(new.any()):
                sheet_min_at_term.append(pre_min[new])
            term_mid |= term
            reset_mid |= term | trunc
            d = (robot.data.joint_pos - q0)[:, arm].abs().amax(dim=1)
            drift_e = torch.where(reset_mid, drift_e, torch.maximum(drift_e, d))
        drift = math.degrees(float(drift_e.max()))
        # Which envs drift, and is it the table: lowest robot body below the table top at reset?
        # Bodies near the base sit below the table top by design, so take only those within the
        # table's footprint in y (env frame; the table spans y in [-0.2, 0.2]).
        bp = robot.data.body_pos_w - origins.unsqueeze(1)
        over = bp[..., 1] < 0.2
        low = torch.where(over, bp[..., 2], torch.full_like(bp[..., 2], 9.0)).amin(dim=1) - top
        dd = torch.rad2deg(drift_e)
        drift_rep = {"envs_gt_5deg": int((dd > 5).sum()), "envs_gt_2deg": int((dd > 2).sum()),
                     "lowest_body_over_table_minus_top_mm_of_gt_5deg": [round(float(v), 1) for v in low[(dd > 5) & ~reset_mid] * 1e3],
                     "drift_deg_of_envs_with_body_below_top_plus_2cm": [round(float(v), 2) for v in dd[(low < 0.02) & ~reset_mid]],
                     "envs_with_body_over_table_below_top_plus_2cm": int(((low < 0.02) & ~reset_mid).sum())}
        n_reset_mid = int(reset_mid.sum())
        keep = ~reset_mid
        parts = inner._particles_w()
        min_z = parts[..., 2].amin(dim=1) - origins[:, 2]
        margin = inner.dr_sample["contact_margin"] if "contact_margin" in inner.dr_sample else torch.zeros(n, device=device)
        resid = min_z - top
        resid_m = resid - margin
        rel = top - base_z()
        smt = torch.cat(sheet_min_at_term) * 1e3 if sheet_min_at_term else torch.zeros(0, device=device)
        terminated_rep = {"count": n_reset_mid, "terminated_not_truncated": int(term_mid.sum()),
                          "sheet_min_minus_top_mm_just_before": [round(float(v), 1) for v in smt]}
        thr = 0.03
        lowc = clear_pad < thr
        clearance_rep = {
            "pad_minus_sheet_top_at_spawn_mm": _st(clear_pad * 1e3),
            "body_minus_sheet_top_at_spawn_mm": _st(clear_body * 1e3),
            "pad_clearance_of_envs_that_reset_mm": [round(float(v), 1) for v in clear_pad[reset_mid] * 1e3],
            f"envs_pad_clearance_lt_{int(thr*1e3)}mm": int(lowc.sum()),
            f"reset_rate_if_pad_clearance_lt_{int(thr*1e3)}mm": round(float(reset_mid[lowc].float().mean()), 3) if bool(lowc.any()) else None,
            f"reset_rate_if_pad_clearance_ge_{int(thr*1e3)}mm": round(float(reset_mid[~lowc].float().mean()), 3),
        }
        disturbed = keep & ((resid_m - resid_m[keep].median()).abs() > 5e-3)
        clearance_rep["pad_clearance_of_kept_but_off_table_sheets_mm"] = [round(float(v), 1) for v in clear_pad[disturbed] * 1e3]
        top, min_z, resid, resid_m, rel, parts = top[keep], min_z[keep], resid[keep], resid_m[keep], rel[keep], parts[keep]
        orig_k = origins[keep]
        all_top.append(top), all_res.append(resid_m), all_minz.append(min_z)
        resets.append({
            "table_top_rel_base_m": _st(rel),
            "table_sim_z_minus_table_z_per_env_max_abs_mm": round(float((table_sim_z - table_z).abs().max()) * 1e3, 3),
            "sheet_min_particle_z_minus_table_top_mm": _st(resid * 1e3),
            "sheet_min_minus_top_minus_margin_mm": _st(resid_m * 1e3),
            "corr_sheet_min_z_vs_table_top": _corr(min_z, top),
            "sheet_max_particle_z_minus_table_top_mm": _st((parts[..., 2].amax(dim=1) - orig_k[:, 2] - top) * 1e3),
            "sheets_resting_gt_5mm_off_median": int(((resid_m - resid_m.median()).abs() > 5e-3).sum()),
            "envs_kept": int(keep.sum()),
            "arm_drift_deg_noisy_reset_pose": round(drift, 3),
            "arm_drift_detail": drift_rep,
            "envs_reset_during_settle": terminated_rep,
            "hand_sheet_clearance": clearance_rep,
        })
        print(f"[probe] reset {r}: {json.dumps(resets[-1])}", flush=True)
    top, res, minz = torch.cat(all_top), torch.cat(all_res), torch.cat(all_minz)
    report["resets"] = resets
    report["pooled"] = {
        "table_top_rel_base_m": _st(top - float(base_z()[0])),
        "corr_sheet_min_z_vs_table_top": _corr(minz, top),
        "sheet_min_minus_top_minus_margin_mm": _st(res * 1e3),
        "max_abs_residual_after_mean_mm": round(float((res - res.mean()).abs().max()) * 1e3, 3),
        "fraction_sheets_on_table_(resid_m_within_5mm)": round(float(((res - res.mean()).abs() < 5e-3).float().mean()), 4),
    }

    # HOME pose with the table where the last reset put it.
    home = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    inner._prev_targets[:] = home
    inner._cur_targets[:] = home
    robot.set_joint_position_target(home)
    top = inner._table_z_per_env + TABLE_HALF
    act = hold_action()
    # hold_action reads joint_pos, which is stale until a step; compute from home instead.
    act[:, canon_hand] = (2.0 * (home[:, hand] - lo) / (hi - lo) - 1.0).clamp(-1, 1)
    drift, obs_palm0 = 0.0, None
    for i in range(args_cli.home_steps):
        obs, *_ = env.step(act)
        if i == 0 and palm_slice is not None:
            obs_palm0 = policy(obs)[:, palm_slice].clone()
        drift = max(drift, math.degrees(float((robot.data.joint_pos - home)[:, arm].abs().max())))
    worst = int((robot.data.joint_pos - home)[:, arm].abs().max(dim=0).values.argmax())
    pc = palm_centre()
    home_rep = {
        "steps": args_cli.home_steps,
        "arm_max_drift_deg": round(drift, 4),
        "worst_arm_joint": robot.data.joint_names[arm[worst]],
        "palm_centre_env0": [round(float(v), 4) for v in pc[0]],
        "palm_centre_above_table_top_m": _st(pc[:, 2] - top),
        "table_top_rel_base_m": _st(top - base_z()),
        "palm_pos_obs_computed_env0": [round(float(v), 4) for v in (pc[0] - root[0] + torch.tensor((0.0, 0.8, 0.0), device=device))],
    }
    if palm_slice is not None:
        o = policy(obs)[:, palm_slice]
        home_rep["palm_pos_obs_env0_after_hold"] = [round(float(v), 4) for v in o[0]]
        home_rep["palm_pos_obs_env0_first_step"] = [round(float(v), 4) for v in obs_palm0[0]]
        home_rep["palm_pos_obs_spread_across_envs_mm"] = round(float((o - o.mean(0)).abs().max()) * 1e3, 3)
    report["home"] = home_rep

    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write(text + "\n")
    env.close()


if __name__ == "__main__":
    run()
    app.close()
