"""Why does the hand bounce when it touches the VBD cloth? Gain, sim coupling, or the policy?

    scripts/newton_py scripts/analysis/cloth_bounce_probe.py --task <id> --checkpoint <pth> --out x.json [hydra...]

Logs every PHYSICS step (decimation 2 -> 120 Hz) the hand joint targets the env sends, the hand joint
positions, the fingertip velocities and (Newton coupled solver only) the cloth->hand coupling wrench
harvested by the VBD proxy. Then summarises how "jittery" each signal is.

Modes:
  policy  -- the policy drives the whole episode.
  freeze  -- the policy drives until this env's hand first feels the cloth (coupling force > --contact_n,
             VBD only) or until step --freeze_step (other sims); from then on the joint TARGETS of that env
             are held constant. If the hand still bounces with a constant command, the sim is doing it.

--proxy_relaxation r (<1) under-relaxes the VBD -> hand feedback force: r * new + (1 - r) * old.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np


def _find_coupled_solver():
    for o in gc.get_objects():
        try:
            if type(o).__name__ == "SolverCoupledProxy":
                return o
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=400, help="policy steps")
    p.add_argument("--mode", choices=("policy", "freeze"), default="policy")
    p.add_argument("--freeze_step", type=int, default=60)
    p.add_argument("--contact_n", type=float, default=2.0)
    p.add_argument("--proxy_relaxation", type=float, default=1.0)
    p.add_argument("--hand_armature_scale", type=float, default=1.0)
    p.add_argument("--hand_kp_scale", type=float, default=1.0, help="also scales Kd by sqrt")
    p.add_argument("--randomize", action="store_true", help="keep all DR on")
    p.add_argument("--random_start", action="store_true", help="DR off but keep the reset (start-pose) noise")
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
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        if args.hand_armature_scale != 1.0 or args.hand_kp_scale != 1.0:
            import dataclasses
            import importlib

            from isaacsimenvs.tasks.play import robots as _robots

            _orig = _robots.get_robot_spec

            def _scaled(name):
                spec = _orig(name)
                a, k = args.hand_armature_scale, args.hand_kp_scale
                return dataclasses.replace(
                    spec,
                    hand_armature={j: v * a for j, v in spec.hand_armature.items()},
                    hand_stiffness={j: v * k for j, v in spec.hand_stiffness.items()},
                    hand_damping={j: v * k ** 0.5 for j, v in spec.hand_damping.items()},
                )

            for mod in ("isaacsimenvs.tasks.play.robots", "isaacsimenvs.tasks.play.play_env",
                        "isaacsimenvs.tasks.play.utils.scene_utils", "isaacsimenvs.tasks.cloth.cloth_env"):
                m = importlib.import_module(mod)
                if getattr(m, "get_robot_spec", None) is _orig:
                    m.get_robot_spec = _scaled
            print(f"[bounce] hand armature x{args.hand_armature_scale}, kp x{args.hand_kp_scale}", flush=True)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = 0.01
        if not args.randomize:
            import copy
            rs0 = copy.deepcopy(env_cfg.reset)
            disable_randomization(env_cfg)
            if args.random_start:
                env_cfg.reset = rs0
        use_single_object_variant()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n,
            device="cuda:0", num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=0.0,
        )
        robot = inner.robot
        hand_ids = inner._hand_joint_ids
        tip_ids, tip_names = robot.find_bodies(inner.robot_spec.fingertip_body_regex)
        hand_names = [robot.joint_names[i] for i in (hand_ids.tolist() if torch.is_tensor(hand_ids) else hand_ids)]
        print(f"[bounce] hand joints {hand_names}; tips {tip_names}", flush=True)
        _sp = inner.robot_spec
        print(f"[bounce] spec {_sp.name}: armature {sorted(set(round(v, 5) for v in _sp.hand_armature.values()))} "
              f"kp {min(_sp.hand_stiffness.values()):.3f}-{max(_sp.hand_stiffness.values()):.3f}", flush=True)

        # --- coupled solver (Newton VBD only) ------------------------------------------------
        solver = _find_coupled_solver()
        proxy_env = proxy_ids = None
        mapping = None
        if solver is not None:
            import warp as wp

            model = solver.model
            mapping = solver._proxy_mappings[0]
            ids = mapping.proxy_ids_global.numpy()
            labels = list(model.body_label) if hasattr(model, "body_label") else list(model.body_key)
            world = model.body_world.numpy()
            hand = np.array(["Table" not in labels[i] for i in ids])
            proxy_ids = torch.as_tensor(ids[hand], device=dev, dtype=torch.long)
            proxy_env = torch.as_tensor(world[ids[hand]], device=dev, dtype=torch.long)
            print(f"[bounce] coupled solver: {len(ids)} proxies ({int(hand.sum())} hand), mode {int(mapping.mode)}, "
                  f"relax {mapping.proxy_relaxation}; e.g. {[labels[i].split('/')[-1] for i in ids[hand][:4]]}",
                  flush=True)
            if args.proxy_relaxation != 1.0:
                for m in solver._proxy_mappings:
                    m.proxy_relaxation = float(args.proxy_relaxation)
                    if m.coupling_forces_previous is None:
                        m.coupling_forces_previous = wp.zeros(
                            m.proxy_ids_global.shape[0], dtype=m.coupling_forces.dtype, device=m.coupling_forces.device)
                print(f"[bounce] proxy_relaxation -> {args.proxy_relaxation}", flush=True)

        # --- per-physics-step logger ---------------------------------------------------------
        log = {"tgt": [], "q": [], "tipv": [], "f": [], "frozen": [], "alive": []}
        state = {"alive": torch.ones(n, dtype=torch.bool, device=dev),
                 "frozen": torch.zeros(n, dtype=torch.bool, device=dev),
                 "ftgt": None}
        orig_apply = inner._apply_action
        orig_pre = inner._pre_physics_step

        def pre(actions):
            orig_pre(actions)
            if args.mode == "freeze" and state["ftgt"] is not None:
                fr = state["frozen"]
                if bool(fr.any()):
                    inner._cur_targets[fr] = state["ftgt"][fr]
                    inner._prev_targets = inner._cur_targets.clone()

        def apply():
            orig_apply()
            log["tgt"].append(inner._cur_targets[:, hand_ids].float().cpu())
            log["q"].append(robot.data.joint_pos[:, hand_ids].float().cpu())
            log["tipv"].append(robot.data.body_lin_vel_w[:, tip_ids].float().cpu())
            if mapping is not None:
                cf = torch.as_tensor(mapping.coupling_forces.numpy(), device=dev)[proxy_ids]  # (P, 6)
                F = torch.zeros(n, 3, device=dev).index_add_(0, proxy_env, cf[:, :3])
                Fabs = torch.zeros(n, device=dev).index_add_(0, proxy_env, cf[:, :3].norm(dim=-1))
                log["f"].append(torch.cat([F, Fabs[:, None]], dim=1).cpu())
            log["frozen"].append(state["frozen"].cpu().clone())
            log["alive"].append(state["alive"].cpu().clone())

        inner._pre_physics_step = pre
        inner._apply_action = apply

        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, inner.cfg.action_space), device=dev))
        for k in log:
            log[k].clear()

        for step in range(args.steps):
            action = player.get_action(obs["policy"], deterministic=True)
            if args.mode == "freeze":
                if mapping is not None:
                    touch = (log["f"][-1][:, 3].to(dev) > args.contact_n) if log["f"] else torch.zeros(n, dtype=torch.bool, device=dev)
                else:
                    touch = torch.full((n,), step >= args.freeze_step, dtype=torch.bool, device=dev)
                new = touch & ~state["frozen"] & state["alive"]
                if bool(new.any()):
                    if state["ftgt"] is None:
                        state["ftgt"] = inner._cur_targets.clone()
                    state["ftgt"][new] = inner._cur_targets[new]
                    state["frozen"] |= new
            obs, _r, term, trunc, _x = env.step(action.to(dev))
            ended = term.bool() | trunc.bool()
            state["alive"] &= ~ended
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if not bool(state["alive"].any()):
                break

        # --- summary -------------------------------------------------------------------------
        dt = float(inner.physics_dt)
        tgt = torch.stack(log["tgt"]).numpy()        # (T, n, J)
        q = torch.stack(log["q"]).numpy()
        tipv = torch.stack(log["tipv"]).numpy()      # (T, n, 5, 3)
        alive = torch.stack(log["alive"]).numpy()    # (T, n)
        frozen = torch.stack(log["frozen"]).numpy()
        f = torch.stack(log["f"]).numpy() if log["f"] else None   # (T, n, 4)
        T = q.shape[0]

        def acc2(x):  # second difference magnitude, rad per step^2 -> mean abs
            return np.abs(x[2:] - 2 * x[1:-1] + x[:-2])

        # window masks over t in [1, T-2] (second-difference centre)
        a_mid = alive[2:] & alive[1:-1] & alive[:-2]

        def masks(extra):
            m = a_mid & extra[1:-1]
            return m

        contact = (f[..., 3] > args.contact_n) if f is not None else np.zeros_like(alive)
        sel = {
            "all": masks(np.ones_like(alive)),
            "contact": masks(contact),
            "no_contact": masks(~contact),
        }
        if args.mode == "freeze":
            sel["frozen"] = masks(frozen)
            sel["frozen_contact"] = masks(frozen & contact)
        aq, at = acc2(q), acc2(tgt)                  # (T-2, n, J)
        # fingertip velocity reversals: v_t . v_{t-1} < 0 with both speeds > 3 cm/s
        sp = np.linalg.norm(tipv, axis=-1)
        dot = (tipv[1:] * tipv[:-1]).sum(-1)
        rev = (dot < 0) & (sp[1:] > 0.03) & (sp[:-1] > 0.03)   # (T-1, n, 5)
        rev = rev[1:]                                           # align with T-2
        # joint velocity sign flips (finite diff of q) with |dq|/dt > 0.5 rad/s
        dq = np.diff(q, axis=0) / dt
        flip = (dq[1:] * dq[:-1] < 0) & (np.abs(dq[1:]) > 0.5) & (np.abs(dq[:-1]) > 0.5)   # (T-2, n, J)
        dtg = np.diff(tgt, axis=0) / dt
        tflip = (dtg[1:] * dtg[:-1] < 0) & (np.abs(dtg[1:]) > 0.5) & (np.abs(dtg[:-1]) > 0.5)
        res = {"task": args.task, "checkpoint": args.checkpoint, "mode": args.mode,
               "proxy_relaxation": args.proxy_relaxation, "physics_dt": dt, "T": T, "num_envs": n,
               "overrides": hydra_args, "hand_joints": hand_names, "windows": {}}
        for k, m in sel.items():
            cnt = int(m.sum())
            if cnt == 0:
                res["windows"][k] = {"steps": 0}
                continue
            w = {
                "steps": cnt,
                "q_acc_rad_s2": float(aq[m].mean() / dt**2),
                "tgt_acc_rad_s2": float(at[m].mean() / dt**2),
                "q_flip_per_joint_s": float(flip[m].mean() / dt),
                "tgt_flip_per_joint_s": float(tflip[m].mean() / dt),
                "tip_reversal_per_tip_s": float(rev[m].mean() / dt),
                "track_err_rad": float(np.abs(q - tgt)[1:-1][m].mean()),
            }
            if f is not None:
                F = f[1:-1][m]            # (cnt, 4)
                w["force_abs_median"] = float(np.median(F[:, 3]))
                w["force_abs_p99"] = float(np.percentile(F[:, 3], 99))
                w["force_abs_max"] = float(F[:, 3].max())
                fv = f[..., :3]
                fdot = (fv[1:] * fv[:-1]).sum(-1)[1:]            # (T-2, n)
                big = (np.linalg.norm(fv, axis=-1)[1:-1] > args.contact_n) & (np.linalg.norm(fv, axis=-1)[:-2] > args.contact_n)
                mm = m & big
                w["force_dir_flip_frac"] = float((fdot[mm] < 0).mean()) if mm.any() else None
                dF = np.abs(f[2:, :, 3] - f[1:-1, :, 3])
                w["force_step_change_median"] = float(np.median(dF[m]))
            res["windows"][k] = w
        # one env's raw trace for plotting (most frozen-contact steps, else most contact steps)
        score = (sel.get("frozen_contact", sel["contact"])).sum(0) if f is not None else sel["all"].sum(0)
        e = int(np.argmax(score))
        res["trace_env"] = e
        res["trace"] = {"tgt": tgt[:, e].round(4).tolist(), "q": q[:, e].round(4).tolist(),
                        "tip_speed": sp[:, e].round(4).tolist(),
                        "frozen": frozen[:, e].tolist(), "alive": alive[:, e].tolist()}
        if f is not None:
            res["trace"]["force"] = f[:, e].round(3).tolist()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res))
        raw = {"tgt": tgt.astype(np.float32), "q": q.astype(np.float32), "tipv": tipv.astype(np.float32),
               "alive": alive, "frozen": frozen, "decimation": np.int32(inner.cfg.decimation),
               "physics_dt": np.float32(dt)}
        if f is not None:
            raw["f"] = f.astype(np.float32)
        np.savez_compressed(args.out.with_suffix(".npz"), **raw)
        print("[bounce] " + json.dumps({k: res[k] for k in ("task", "mode", "proxy_relaxation", "T")}), flush=True)
        for k, w in res["windows"].items():
            print(f"[bounce] {k}: " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in w.items()}), flush=True)
        env.close()

    run()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
