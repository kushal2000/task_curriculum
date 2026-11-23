"""Distil the keypoint Play prior into a point-cloud student on the point-cloud Play task (online DAgger).

    python -m isaacsimenvs.scripts.distillation.distill_play_pc --headless \
        --teacher pretrained_policy/flexiv_wuji/model.pth --out debug_outputs/train_logs/nov/<date>_<name>_<job> \
        --num_envs 24576 --wandb_project play_pc --wandb_name <name>

The env (``Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0``, ``tasks/play/play_pc_env.py``) gives
the student ``policy`` = ``[proprio | live cloud | goal cloud | mask]`` and the teacher ``teacher``, the
clean keypoint vector with the goal turned to the symmetric equivalent nearest the tool. The student
(``student.SplitStudentNet``: one PointNet per cloud, then the teacher's LSTM + MLP, initialised from
the teacher) drives the env -- the teacher with probability ``beta``, one draw per episode, falling from 1 to 0 over
``--beta_iters`` -- the teacher labels every visited state, and the student regresses the label and,
from an auxiliary head, the teacher's keypoint fields. Truncated BPTT over ``--horizon`` steps.

Logged every ``--log_every`` iterations to ``<out>/log.jsonl`` and wandb: losses, auxiliary errors,
goals reached per episode (student- and teacher-driven envs), the clouds' fill and visibility, and the
goal distance; the interactive viewer (``play_pc_viewer``: robot, tool, goal, live and goal clouds of
env ``--viewer_env``) every ``--viewer_every`` env steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TASK = "Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0"


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description="Point-cloud Play student (DAgger from the keypoint prior).")
    p.add_argument("--task", default=TASK)
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", default=None, help="warm start from a student checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--num_envs", type=int, default=24576)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--success_tolerance", type=float, default=0.01,
                   help="fixed success tolerance [m] (the prior finished its curriculum here); <= 0: Play's curriculum")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--mb_envs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta_iters", type=int, default=200)
    p.add_argument("--aux_coef", type=float, default=1.0,
                   help="weight of the auxiliary keypoint regression; 0 trains on the actions alone (the keypoint\n"
                        "labels are ambiguous under a tool's symmetries)")
    p.add_argument("--grad_norm", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    # wandb + interactive viewer
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_name", default="")
    p.add_argument("--wandb_group", default="")
    p.add_argument("--viewer_every", type=int, default=3200, help="env steps between viewer captures (0: off)")
    p.add_argument("--viewer_len", type=int, default=300)
    p.add_argument("--viewer_env", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.distillation.student import SplitStudentNet, TeacherNet, split_columns
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml
    from isaacsimenvs.tasks.play.play_pc_env import AUX_FIELDS, OBJECT_FIELDS
    from isaacsimenvs.tasks.play.utils.obs_utils import compute_obs_dim

    @hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.seed = int(args.seed)
        if args.success_tolerance > 0:
            env_cfg.termination.eval_success_tolerance = float(args.success_tolerance)
        torch.manual_seed(args.seed)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        wb = None
        if args.wandb_project:
            import wandb

            try:
                env_dict = env_cfg.to_dict()
            except Exception:
                env_dict = {}
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name or out.name,
                            group=args.wandb_group or None, dir=str(out),
                            config={"args": vars(args), "env": env_dict, "hydra_overrides": hydra_args})
            wandb.define_metric("distill/*", step_metric="distill/iter")

        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        if args.viewer_every > 0:
            from isaacsimenvs.scripts.visualization.interactive_viewer.play_pc_viewer import PlayPCViewerWrapper

            env = PlayPCViewerWrapper(env, output_dir=out / "interactive_viewer", capture_len=args.viewer_len,
                                      capture_interval=args.viewer_every, env_id=args.viewer_env,
                                      wandb_key="interactive_viewer")
        device, n = inner.device, inner.num_envs
        lay = inner.layout
        num_actions = int(inner.cfg.action_space)
        n_p, m = lay["n_proprio"], lay["num_points"]
        obs_list = list(inner.cfg.obs.obs_list)
        sizes = {f: compute_obs_dim([f], num_actions) for f in obs_list}
        proprio_cols, aux_cols = split_columns(obs_list, sizes, OBJECT_FIELDS, AUX_FIELDS)
        if len(proprio_cols) != n_p:
            raise ValueError(f"policy has {n_p} proprio columns, the teacher's vector {len(proprio_cols)}")
        aux_cols = torch.tensor(aux_cols, dtype=torch.long, device=device)
        scale = float(inner.cfg.pointcloud.scale)

        ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        ckpt = ckpt[0] if 0 in ckpt else ckpt
        teacher = TeacherNet(ckpt["model"]).to(device).eval()
        if teacher.obs_dim != lay["teacher_dim"]:
            raise ValueError(f"teacher reads {teacher.obs_dim} columns, the env's teacher vector has {lay['teacher_dim']}")
        net_args = {"n_live": lay["n_live"], "widths": list(lay["widths"]), "compile": bool(lay["compile"])}
        student = SplitStudentNet(n_p, num_actions, 3, int(aux_cols.numel()), **net_args).to(device)
        first_iter = 0
        if args.student:
            sc = torch.load(args.student, map_location="cpu", weights_only=False)
            student.load_state_dict(sc["model"])
            first_iter = int(sc.get("iter", 0))
        else:
            student.load_teacher(teacher, torch.tensor(proprio_cols))
        opt = torch.optim.Adam(student.parameters(), lr=float(args.lr))
        if args.student and "opt" in sc:
            opt.load_state_dict(sc["opt"])
            for g in opt.param_groups:
                g["lr"] = float(args.lr)
        amp = not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def autocast():
            return torch.autocast("cuda", dtype=torch.float16, enabled=amp)

        def split(obs):
            pol = torch.nan_to_num(obs["policy"])
            return (pol[:, :n_p], pol[:, n_p:n_p + 3 * m].reshape(n, m, 3), pol[:, n_p + 3 * m:n_p + 4 * m] > 0.5,
                    torch.nan_to_num(obs["teacher"]))

        h = int(args.horizon)
        buf = {
            "proprio": torch.zeros(h, n, n_p, device=device),
            "points": torch.zeros(h, n, m, 3, device=device, dtype=torch.float16),
            "mask": torch.zeros(h, n, m, dtype=torch.bool, device=device),
            "act": torch.zeros(h, n, num_actions, device=device),
            "aux": torch.zeros(h, n, int(aux_cols.numel()), device=device),
            "start": torch.zeros(h, n, dtype=torch.bool, device=device),
        }
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, num_actions), device=device))
        t_state, s_state = teacher.initial_state(n, device), student.initial_state(n, device)
        start = torch.ones(n, dtype=torch.bool, device=device)
        meta = {"env": {"task": args.task, "layout": {k: v for k, v in lay.items()},
                        "pointcloud": inner.cfg.pointcloud.to_dict()},
                "net": "SplitStudentNet", "net_args": net_args, "teacher": args.teacher,
                "aux_coef": float(args.aux_coef)}
        (out / "student_meta.json").write_text(json.dumps(meta, indent=1, default=str))

        def save(it: int, name: str) -> None:
            torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "iter": it, **meta}, out / name)

        win: dict = {}

        def add(k, v):
            win[k] = win.get(k, 0.0) + float(v)

        log_path = out / "log.jsonl"
        t0, frames = time.time(), 0
        print(f"[distill_play] {n} envs x {h} steps/iter, {args.iters} iters, minibatch {args.mb_envs}, "
              f"beta 1->0 over {args.beta_iters}, amp {amp}, student {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M",
              flush=True)
        def beta_at(it: int) -> float:
            return max(0.0, 1.0 - it / max(int(args.beta_iters), 1))

        # Who drives an env is drawn once per episode (redrawn at its reset), so an episode is all teacher
        # or all student and the per-driver goal counts are each driver's own.
        use_teacher = torch.rand(n, device=device) < beta_at(first_iter)
        for it in range(first_iter, int(args.iters)):
            beta = beta_at(it)
            state0 = (s_state[0].detach().clone(), s_state[1].detach().clone())
            student.eval()
            t_roll = time.time()
            for t in range(h):
                proprio, points, mask, tobs = split(obs)
                with torch.no_grad():
                    t_act, t_state = teacher.step(tobs, t_state, start)
                    with autocast():
                        mu, _aux, s_state = student.step(proprio, points, mask, s_state, start)
                s_act = torch.nan_to_num(mu.float()).clamp(-1.0, 1.0)
                act = torch.where(use_teacher.unsqueeze(1), t_act, s_act)
                buf["proprio"][t], buf["points"][t], buf["mask"][t] = proprio, points.half(), mask
                buf["act"][t], buf["aux"][t], buf["start"][t] = t_act, tobs[:, aux_cols] * scale, start
                add("live_pts", mask[:, :lay["n_live"]].float().sum(1).mean() / h)
                add("tool_visible", inner.cloud.last_visible_frac.mean() / h)
                add("tool_share", inner.cloud.last_cloth_share.mean() / h)
                add("goal_dist_cm", inner._keypoints_max_dist.mean() * 100.0 / h)
                obs, _r, terminated, truncated, extras = env.step(act)
                ended = terminated.bool() | truncated.bool()
                if bool(ended.any()):
                    goals = extras["episode_final"]["successes"]
                    for who, sel in (("s", ended & ~use_teacher), ("t", ended & use_teacher)):
                        add(f"{who}_ep", sel.sum())
                        add(f"{who}_goals", (goals * sel).sum())
                start = ended
                if bool(ended.any()):
                    use_teacher = torch.where(ended, torch.rand(n, device=device) < beta, use_teacher)
            frames += n * h
            add("t_roll", time.time() - t_roll)

            t_upd = time.time()
            student.train()
            for mb in torch.randperm(n, device=device).split(int(args.mb_envs)):
                with autocast():
                    mu, aux = student.forward_seq(buf["proprio"][:, mb], buf["points"][:, mb].float(), buf["mask"][:, mb],
                                                  (state0[0][:, mb], state0[1][:, mb]), buf["start"][:, mb])
                    loss_act = ((mu.float() - buf["act"][:, mb]) ** 2).mean()
                    loss_aux = ((aux.float() - buf["aux"][:, mb]) ** 2).mean()
                    loss = loss_act + float(args.aux_coef) * loss_aux if args.aux_coef > 0 else loss_act
                if not bool(torch.isfinite(loss)):
                    add("skipped", 1)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_norm))
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    d = ((aux.float() - buf["aux"][:, mb]) / scale).view(*aux.shape[:-1], 2, -1, 3).norm(dim=-1).mean(dim=-1)
                    add("upd", 1)
                    add("loss_act", loss_act)
                    add("loss_aux", loss_aux)
                    add("aux_palm_cm", d[..., 0].mean() * 100.0)
                    add("aux_goal_cm", d[..., 1].mean() * 100.0)
            add("t_upd", time.time() - t_upd)
            add("it", 1)
            if win.get("skipped", 0) > 50:
                raise RuntimeError(f"{win['skipped']} non-finite minibatches by iter {it}")

            if (it + 1) % int(args.log_every) == 0 or it + 1 == int(args.iters):
                u, k = max(win.get("upd", 0), 1), max(win.get("it", 0), 1)
                row = {
                    "iter": it + 1, "frames": frames, "beta": round(beta, 3),
                    "sps": round(frames / (time.time() - t0), 1),
                    "s_per_iter_rollout": round(win.get("t_roll", 0) / k, 2), "s_per_iter_update": round(win.get("t_upd", 0) / k, 2),
                    "loss_act": win.get("loss_act", 0) / u, "loss_aux": win.get("loss_aux", 0) / u,
                    "aux_palm_cm": win.get("aux_palm_cm", 0) / u, "aux_goal_cm": win.get("aux_goal_cm", 0) / u,
                    "student_goals_per_ep": win["s_goals"] / win["s_ep"] if win.get("s_ep") else None,
                    "teacher_goals_per_ep": win["t_goals"] / win["t_ep"] if win.get("t_ep") else None,
                    "student_episodes": int(win.get("s_ep", 0)), "teacher_episodes": int(win.get("t_ep", 0)),
                    "live_points": win.get("live_pts", 0) / k, "tool_visible_frac": win.get("tool_visible", 0) / k,
                    "tool_share_of_cloud": win.get("tool_share", 0) / k, "goal_dist_cm": win.get("goal_dist_cm", 0) / k,
                    "skipped_minibatches": int(win.get("skipped", 0)),
                    "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
                print("[distill_play] " + json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if wb is not None:
                    # no explicit step: the viewer logs its HTML through the same run
                    wb.log({f"distill/{k2}": v for k2, v in row.items() if v is not None})
                win = {"skipped": win.get("skipped", 0)}
            if (it + 1) % int(args.save_every) == 0:
                save(it + 1, f"student_it{it + 1}.pth")
                save(it + 1, "student_last.pth")
        save(int(args.iters), "student_last.pth")
        print(f"[distill_play] done: {out / 'student_last.pth'}", flush=True)
        env.close()
        if wb is not None:
            wb.finish()

    run()
    if app is not None:
        del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
