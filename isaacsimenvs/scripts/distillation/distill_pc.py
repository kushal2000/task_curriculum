"""Distil a state (keypoint) fold policy into a point-cloud student (DAgger). Training only: students
are evaluated by experiments/eval.sh (scripts/evaluation/eval.py), like every fold policy.

    # a recipe with env.obs.object: pointcloud (its cloud and DR), the keypoint teacher
    python -m isaacsimenvs.scripts.distillation.distill_pc --headless \
        --task Isaacsimenvs-RigidCloth-Direct-v0 --experiment experiments/<study>/<recipe>.yaml \
        --teacher <snap.pth> --out outputs/distill/<name> env.scene.num_envs=4096

Online DAgger: the student's action drives the env (the teacher's with probability ``beta``, which
falls linearly from 1 to 0 over ``--beta_iters``), the frozen teacher labels every visited state with
its deterministic action, and the student regresses that action plus -- from an auxiliary head --
the keypoint observations the teacher read. Truncated BPTT over ``--horizon`` steps, the teacher's
``seq_length``.

The env builds both inputs (``obs.object: pointcloud``, ``tasks/cloth/mdp/observations.py``): the
student reads ``policy`` = ``[proprio | points | mask]``, the teacher reads ``teacher``, the clean
keypoint vector. The cloud (``obs.pointcloud``: goal group, forward-kinematics points) and its DR
(``domain_randomization.proprio`` / ``pointcloud`` / ``action``) come from the recipe, which
``args.json`` records next to the checkpoints; the student also keeps them (``student.student_env``)
for the VBD evaluation and the viewer. The network is the
teacher's LSTM + MLP with a PointNet in front (``scripts/distillation/student.py``).

A student is fine-tuned by RL with ``train.py`` on a point-cloud recipe (``pc_rl.py``); ``--student``
also takes such an rl_games checkpoint as a warm start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCALE_CM = 100.0


def _t(x):
    """Newton hands back warp arrays with a ``.torch`` view; PhysX hands back tensors."""
    return x.torch if hasattr(x, "torch") else x


class DragTracker:
    """How far each env's stationary half has been dragged (xy, yaw) since its episode started."""

    def __init__(self, inner) -> None:
        import torch

        self.inner, n, dev = inner, inner.num_envs, inner.device
        self.start_xy, self.start_yaw = torch.zeros(n, 2, device=dev), torch.zeros(n, device=dev)
        self.drag_xy, self.drag_yaw = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        self.max_drag_xy, self.max_drag_yaw = torch.zeros(n, device=dev), torch.zeros(n, device=dev)

    def _stationary(self):
        import torch

        view = self.inner.fold_view()
        rot = view.frame_r
        return view.frame_c - self.inner.scene.env_origins, torch.atan2(rot[:, 1, 0], rot[:, 0, 0])

    def reset(self, ids) -> None:
        if ids.numel() == 0:
            return
        centroid, yaw = self._stationary()
        self.start_xy[ids], self.start_yaw[ids] = centroid[ids, :2], yaw[ids]
        self.max_drag_xy[ids] = 0.0
        self.max_drag_yaw[ids] = 0.0

    def update(self) -> None:
        import torch

        centroid, yaw = self._stationary()
        self.drag_xy = (centroid[:, :2] - self.start_xy).norm(dim=-1)
        dyaw = yaw - self.start_yaw
        self.drag_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
        self.max_drag_xy = torch.maximum(self.max_drag_xy, self.drag_xy)
        self.max_drag_yaw = torch.maximum(self.max_drag_yaw, self.drag_yaw)


class StudentInput:
    """Splits the env's observations: the student's ``[proprio | points | mask]`` (``policy``), the
    teacher's keypoint vector (``teacher``) and the auxiliary targets in it."""

    def __init__(self, inner) -> None:
        import torch

        from isaacsimenvs.scripts.distillation.student import split_columns
        from isaacsimenvs.tasks.cloth.mdp.observations import obs_dim
        from isaacsimenvs.tasks.cloth.perception.keypoints import AUX_FIELDS, OBJECT_FIELDS

        self.inner = inner
        obs_list = list(inner.cfg.obs.obs_list)
        num_joints = int(inner.cfg.action_space)
        self.teacher_dim = obs_dim(obs_list, num_joints)
        sizes = {f: obs_dim([f], num_joints) for f in obs_list}
        proprio, aux = split_columns(obs_list, sizes, OBJECT_FIELDS, AUX_FIELDS)
        self.proprio_cols = torch.tensor(proprio, dtype=torch.long, device=inner.device)
        self.aux_cols = torch.tensor(aux, dtype=torch.long, device=inner.device)
        self.pc = inner.observations.pointcloud
        if self.pc is None:
            return
        lay = inner.observations.pointcloud_layout
        self.n_proprio, self.num_points, self.point_dim = lay["n_proprio"], lay["num_points"], lay["point_dim"]
        self.scale = float(self.pc.setup.scale)
        if self.n_proprio != len(proprio):
            raise ValueError(f"the point-cloud vector has {self.n_proprio} proprio columns, the teacher's {len(proprio)}")

    @staticmethod
    def teacher_obs(obs):
        return obs.get("teacher", obs["policy"])

    def __call__(self, obs) -> dict:
        import torch

        pol = obs["policy"]
        n_p, m, d = self.n_proprio, self.num_points, self.point_dim
        teacher = torch.nan_to_num(obs["teacher"], nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "proprio": pol[:, :n_p], "points": pol[:, n_p:n_p + m * d].reshape(-1, m, d),
            "mask": pol[:, n_p + m * d:n_p + m * d + m] > 0.5, "aux": teacher[:, self.aux_cols] * self.scale,
        }


def _kp_err_cm(pred, target, scale: float):
    """Mean keypoint distance [cm] of the two aux groups (rel palm, rel goal): ``(N,), (N,)``."""
    d = ((pred.float() - target.float()) / scale).view(*pred.shape[:-1], 2, -1, 3).norm(dim=-1).mean(dim=-1)
    return d[..., 0] * SCALE_CM, d[..., 1] * SCALE_CM


DRAG_BINS = ((0.0, 0.01), (0.01, 0.03), (0.03, 1e9))
DRAG_NAMES = ("drag<1cm", "drag1-3cm", "drag>3cm")


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description="Point-cloud student distillation (DAgger).")
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", default=None, help="run recipe: the point-cloud env (cloud, DR) the student trains in")
    p.add_argument("--teacher", required=True, help="teacher checkpoint (.pth)")
    p.add_argument("--student", default=None, help="a student checkpoint to start from (distilled or RL-trained)")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--seed", type=int, default=1)
    # student
    p.add_argument("--no_rnn_state", action="store_true", help="memory ablation: LSTM state dropped every step")
    p.add_argument("--no_teacher_init", action="store_true")
    # DAgger
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--mb_envs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta_iters", type=int, default=200)
    p.add_argument("--aux_coef", type=float, default=1.0)
    p.add_argument("--grad_norm", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--rl_device", default="cuda:0")
    p.add_argument("--sim_device", default="cuda:0")
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    try:
        app = AppLauncher(args).app
    except ImportError as exc:
        app = None
        print(f"[distill] running kit-less (no Isaac Sim runtime): {exc}", flush=True)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.distillation.student import StudentNet, TeacherNet, student_env
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
    def run(env_cfg, _agent_cfg) -> None:
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.sim_device
        env_cfg.seed = args.seed
        from isaacsimenvs.scripts.distillation.pc_rl import load_student_any

        student_ckpt = load_student_any(args.student) if args.student else None   # distilled or RL-trained
        if env_cfg.obs.object != "pointcloud":
            raise ValueError("a student reads the point cloud: use a recipe with env.obs.object: pointcloud")
        torch.manual_seed(args.seed)

        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        device, n = inner.device, inner.num_envs
        num_actions = int(inner.cfg.action_space)
        sin = StudentInput(inner)
        drag = DragTracker(inner)

        ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        ckpt = ckpt[0] if 0 in ckpt else ckpt
        teacher = TeacherNet(ckpt["model"]).to(device).eval()
        if teacher.obs_dim != sin.teacher_dim:
            raise ValueError(f"teacher reads {teacher.obs_dim} obs columns, the env's keypoint vector has {sin.teacher_dim}")

        net_args = dict(use_rnn_state=not args.no_rnn_state)
        env_settings = student_env(env_cfg)
        point_dim = sin.point_dim
        student = StudentNet(sin.n_proprio, num_actions, point_dim, int(sin.aux_cols.numel()), **net_args).to(device)
        if student_ckpt is not None:
            # an RL-trained actor has no auxiliary head: that part keeps its initial values
            student.load_state_dict(student_ckpt["model"], strict=not student_ckpt.get("from_rl", False))
            print(f"[distill] student <- {args.student} (iter {student_ckpt.get('iter')})", flush=True)
        else:
            student.load_teacher(teacher, sin.proprio_cols.cpu())   # the normaliser is always the teacher's
            if args.no_teacher_init:
                fresh = StudentNet(student.n_proprio, num_actions, point_dim, student.aux_dim, **net_args)
                keep = {k: v for k, v in student.state_dict().items() if k in ("mean", "var")}
                student.load_state_dict({**fresh.state_dict(), **keep})
                student.to(device)
        amp = (not args.no_amp) and str(device).startswith("cuda")

        def autocast():
            return torch.autocast("cuda", dtype=torch.float16, enabled=amp)

        all_ids = torch.arange(n, device=device)
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, num_actions), device=device))
        drag.reset(all_ids)
        t_state = teacher.initial_state(n, device)
        s_state = student.initial_state(n, device)
        start = torch.ones(n, dtype=torch.bool, device=device)

        def policy_step(tobs, so, use_teacher=None):
            """Teacher label and student output for this observation; returns the action to apply."""
            nonlocal t_state, s_state
            t_act, t_state = teacher.step(torch.nan_to_num(tobs, nan=0.0, posinf=0.0, neginf=0.0), t_state, start)
            with torch.no_grad(), autocast():
                mu, aux, s_state = student.step(so["proprio"], so["points"], so["mask"], s_state, start)
            s_act = torch.nan_to_num(mu.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            act = s_act if use_teacher is None else torch.where(use_teacher.unsqueeze(1), t_act, s_act)
            return act, t_act, aux.float()

        # ================================================================== train (DAgger)
        out_dir = Path(args.out or "outputs/distill/run")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps({**vars(args), "hydra": hydra_args}, indent=1, default=str))
        opt = torch.optim.Adam(student.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        h, m_pts = int(args.horizon), sin.num_points
        buf = {
            "proprio": torch.zeros(h, n, sin.n_proprio, device=device),
            "points": torch.zeros(h, n, m_pts, point_dim, device=device, dtype=torch.float16),
            "mask": torch.zeros(h, n, m_pts, dtype=torch.bool, device=device),
            "act": torch.zeros(h, n, num_actions, device=device),
            "aux": torch.zeros(h, n, int(sin.aux_cols.numel()), device=device),
            "start": torch.zeros(h, n, dtype=torch.bool, device=device),
            "drag": torch.zeros(h, n, device=device),
        }
        log_path = out_dir / "log.jsonl"
        win = {"s_ep": 0, "s_ok": 0, "t_ep": 0, "t_ok": 0, "loss_act": 0.0, "loss_aux": 0.0, "upd": 0,
               "e_palm": 0.0, "e_goal": 0.0, "vis": 0.0, "pts": 0.0, "cloth": 0.0, "low": 0.0, "outl": 0.0, "it": 0, "skipped": 0,
               "bin_sum": torch.zeros(len(DRAG_BINS), device=device), "bin_cnt": torch.zeros(len(DRAG_BINS), device=device)}
        first_iter = int(student_ckpt.get("iter", 0)) if student_ckpt is not None else 0
        t0 = time.time()
        frames = 0

        def save(it: int, name: str) -> None:
            torch.save({"model": student.state_dict(), "iter": it, "env": env_settings, "net_args": net_args,
                        "teacher": args.teacher, "task": args.task}, out_dir / name)

        print(
            f"[distill] train: {n} envs x {h} steps = {n * h} samples/iter, {args.iters} iters, "
            f"minibatch {args.mb_envs} envs, lr {args.lr}, beta 1->0 over {args.beta_iters} iters, amp {amp}",
            flush=True,
        )
        for it in range(first_iter, args.iters):
            beta = max(0.0, 1.0 - it / max(args.beta_iters, 1))
            use_teacher = torch.rand(n, device=device) < beta
            state0 = (s_state[0].detach().clone(), s_state[1].detach().clone())
            student.eval()
            for t in range(h):
                tobs = sin.teacher_obs(obs)
                so = sin(obs)
                drag.update()
                act, t_act, _aux = policy_step(tobs, so, use_teacher)
                buf["proprio"][t], buf["points"][t], buf["mask"][t] = so["proprio"], so["points"].to(torch.float16), so["mask"]
                buf["act"][t], buf["aux"][t], buf["start"][t], buf["drag"][t] = t_act, so["aux"], start, drag.drag_xy
                cloud = sin.pc.cloud
                win["vis"] += float(cloud.last_visible_frac.mean()) / h
                win["pts"] += float(cloud.last_count.mean()) / h
                win["cloth"] += float(cloud.last_cloth_share.mean()) / h
                win["low"] += float(cloud.last_robot_low.mean()) / h
                win["outl"] += float(cloud.last_outlier.float().sum(dim=1).mean()) / h
                obs, _rew, terminated, truncated, extras = env.step(act)
                ended = terminated.bool() | truncated.bool()
                if bool(ended.any()):
                    ok = extras["episode_final"]["successes"] > 0
                    win["s_ep"] += int((ended & ~use_teacher).sum())
                    win["s_ok"] += int((ended & ~use_teacher & ok).sum())
                    win["t_ep"] += int((ended & use_teacher).sum())
                    win["t_ok"] += int((ended & use_teacher & ok).sum())
                drag.reset(ended.nonzero(as_tuple=True)[0])
                start = ended
            frames += n * h

            student.train()
            for mb in torch.randperm(n, device=device).split(int(args.mb_envs)):
                with autocast():
                    mu, aux = student.forward_seq(
                        buf["proprio"][:, mb], buf["points"][:, mb].float(), buf["mask"][:, mb],
                        (state0[0][:, mb], state0[1][:, mb]), buf["start"][:, mb],
                    )
                    loss_act = ((mu.float() - buf["act"][:, mb]) ** 2).mean()
                    loss_aux = ((aux.float() - buf["aux"][:, mb]) ** 2).mean()
                    loss = loss_act + float(args.aux_coef) * loss_aux
                if not bool(torch.isfinite(loss)):
                    win["skipped"] += 1
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_norm))
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    e_palm, e_goal = _kp_err_cm(aux, buf["aux"][:, mb], sin.scale)
                    win["loss_act"] += float(loss_act)
                    win["loss_aux"] += float(loss_aux)
                    win["e_palm"] += float(e_palm.mean())
                    win["e_goal"] += float(e_goal.mean())
                    win["upd"] += 1
                    mb_drag = buf["drag"][:, mb]
                    for b, (lo, hi) in enumerate(DRAG_BINS):
                        sel = (mb_drag >= lo) & (mb_drag < hi)
                        win["bin_sum"][b] += e_goal[sel].sum()
                        win["bin_cnt"][b] += sel.sum()
            if win["skipped"] > 50:
                raise RuntimeError(f"{win['skipped']} non-finite minibatches by iter {it}")
            win["it"] += 1

            if (it + 1) % args.log_every == 0 or it + 1 == args.iters:
                u, k = max(win["upd"], 1), max(win["it"], 1)
                row = {
                    "iter": it + 1, "frames": frames, "beta": round(beta, 3),
                    "sps": round(frames / (time.time() - t0), 1),
                    "loss_act": round(win["loss_act"] / u, 5), "loss_aux": round(win["loss_aux"] / u, 5),
                    "aux_palm_cm": round(win["e_palm"] / u, 3), "aux_goal_cm": round(win["e_goal"] / u, 3),
                    "aux_goal_cm_by_drag": {
                        nm: (round(float(win["bin_sum"][b] / win["bin_cnt"][b]), 3) if float(win["bin_cnt"][b]) > 0 else None)
                        for b, nm in enumerate(DRAG_NAMES)
                    },
                    "drag_share": {nm: round(float(win["bin_cnt"][b] / win["bin_cnt"].sum().clamp(min=1)), 3)
                                   for b, nm in enumerate(DRAG_NAMES)},
                    "student_success": round(win["s_ok"] / win["s_ep"], 4) if win["s_ep"] else None,
                    "student_episodes": win["s_ep"],
                    "teacher_success": round(win["t_ok"] / win["t_ep"], 4) if win["t_ep"] else None,
                    "teacher_episodes": win["t_ep"],
                    "visible_cloth_fraction": round(win["vis"] / k, 4),
                    "cloud_points": round(win["pts"] / k, 1), "cloth_share": round(win["cloth"] / k, 4),
                    "robot_points_low_3cm": round(win["low"] / k, 2), "outlier_points": round(win["outl"] / k, 2),
                    "skipped_minibatches": win["skipped"],
                    "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                }
                print("[distill] " + json.dumps(row), flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                for key in ("s_ep", "s_ok", "t_ep", "t_ok", "upd", "it"):
                    win[key] = 0
                for key in ("loss_act", "loss_aux", "e_palm", "e_goal", "vis", "pts", "cloth", "low", "outl"):
                    win[key] = 0.0
                win["bin_sum"].zero_()
                win["bin_cnt"].zero_()
            if (it + 1) % args.save_every == 0:
                save(it + 1, f"student_it{it + 1}.pth")
                save(it + 1, "student_last.pth")
        save(args.iters, "student_last.pth")
        print(f"[distill] done: {out_dir / 'student_last.pth'}", flush=True)
        env.close()

    run()
    if app is not None:
        del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
