"""Distil a state (keypoint) fold policy into a point-cloud student, and evaluate either one.

    # train (PhysX fold pool, the teacher's own recipe)
    python -m isaacsimenvs.distill.distill_pc --mode train --headless \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0 --experiment flat7RW_base \
        --teacher <snap.pth> --vis cams --goal g1 --out outputs/distill/<name> \
        env.scene.num_envs=4096 env.cloth.keypoint_tolerance=0.015 env.termination.success_steps=10

    # evaluate a student (or the teacher: --policy teacher) -- first episode of every env
    scripts/newton_py -m isaacsimenvs.distill.distill_pc --mode eval --policy student \
        --task Isaacsimenvs-ClothFlexivWujiRightHLRL-Direct-v0 --teacher <snap.pth> \
        --student outputs/distill/<name>/student_last.pth --num_envs 64 --json <out.json> <overrides>

Online DAgger: the student's action drives the env (the teacher's with probability ``beta``, which
falls linearly from 1 to 0 over ``--beta_iters``), the frozen teacher labels every visited state with
its deterministic action, and the student regresses that action plus -- from an auxiliary head --
the keypoint observations the teacher read. Truncated BPTT over ``--horizon`` steps, the teacher's
``seq_length``.

Student input: the teacher's proprioceptive observation columns and a point cloud
(``tasks/cloth/utils/cloth_pointcloud.py``) in place of the tracker columns. The network is the
teacher's LSTM + MLP with a PointNet in front (``distill/student.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

SCALE_CM = 100.0


def _t(x):
    """Newton hands back warp arrays with a ``.torch`` view; PhysX hands back tensors."""
    return x.torch if hasattr(x, "torch") else x


class StudentObs:
    """Builds the student's observation from the env's state, and tracks how far the stationary
    half has been dragged since the episode started."""

    def __init__(self, inner, pc_cfg, proprio_delay: int = 0) -> None:
        import torch

        from isaacsimenvs.distill.student import split_columns
        from isaacsimenvs.tasks.cloth.utils.cloth_pointcloud import ClothPointCloud, hand_capsules
        from isaacsimenvs.tasks.cloth.utils.scene_pointcloud import LagBuffer, SceneCloudCfg
        from isaacsimenvs.tasks.play.utils.obs_utils import compute_obs_dim

        self.inner, self.device, self.n = inner, inner.device, inner.num_envs
        if inner.cfg.cloth.fold_axis != "x":
            raise ValueError("the point cloud assumes cloth.fold_axis = x")
        obs_list = list(inner.cfg.obs.obs_list)
        num_joints = int(inner.cfg.action_space)
        sizes = {f: compute_obs_dim([f], num_joints) for f in obs_list}
        if sum(sizes.values()) != int(inner.cfg.observation_space):
            raise ValueError(f"obs_list sizes {sizes} do not add up to {inner.cfg.observation_space}")
        proprio, aux = split_columns(obs_list, sizes)
        self.proprio_cols = torch.tensor(proprio, dtype=torch.long, device=self.device)
        self.aux_cols = torch.tensor(aux, dtype=torch.long, device=self.device)

        root = torch.as_tensor(inner.cfg.robot_root_pos, dtype=torch.float32)
        cams = torch.stack((torch.as_tensor(pc_cfg.cam_a_base) + root, torch.as_tensor(pc_cfg.cam_b_base) + root))
        names = list(inner.robot.body_names)
        self.scene = isinstance(pc_cfg, SceneCloudCfg)
        self.scale = float(pc_cfg.scale)
        dev = self.device
        self.start_xy = torch.zeros(self.n, 2, device=dev)
        self.start_yaw = torch.zeros(self.n, device=dev)
        self.drag_xy = torch.zeros(self.n, device=dev)
        self.drag_yaw = torch.zeros(self.n, device=dev)
        self.max_drag_xy = torch.zeros(self.n, device=dev)
        self.max_drag_yaw = torch.zeros(self.n, device=dev)
        # The student's proprioception is up to ``proprio_delay`` policy steps old (per-episode lag).
        self.proprio_delay = int(proprio_delay)
        self.p_lag = torch.zeros(self.n, dtype=torch.long, device=dev)
        self.fresh = torch.ones(self.n, dtype=torch.bool, device=dev)
        self.p_buf = LagBuffer(self.n, self.proprio_delay, (len(proprio),), dev)
        if self.scene:
            self._init_scene(inner, pc_cfg, cams, names, len(proprio), len(aux))
            return
        regex = str(inner.robot_spec.hand_joint_regex)
        spec = hand_capsules(pc_cfg, regex[:2] if regex[:2] in ("r_", "l_") else "")
        missing = sorted({b for a, _, c, _, _ in spec for b in (a, c) if b not in names})
        if missing:
            raise ValueError(f"robot has no bodies {missing} for the hand capsules (bodies: {names})")
        dev = self.device
        self.cap_ia = torch.tensor([names.index(a) for a, *_ in spec], dtype=torch.long, device=dev)
        self.cap_oa = torch.tensor([o for _, o, *_ in spec], dtype=torch.float32, device=dev).view(-1, 3)
        self.cap_ib = torch.tensor([names.index(b) for _, _, b, _, _ in spec], dtype=torch.long, device=dev)
        self.cap_ob = torch.tensor([o for *_, o, _ in spec], dtype=torch.float32, device=dev).view(-1, 3)
        radii = torch.tensor([r for *_, r in spec], dtype=torch.float32, device=dev)
        self.pc = ClothPointCloud(pc_cfg, self.n, int(inner.cfg.cloth.resolution), cams, radii, dev)
        print(
            f"[distill] point cloud: {pc_cfg.n_cloth} cloth + {self.pc.n_hand} hand ({len(spec)} capsules) + "
            f"{self.pc.n_extra} goal points, visibility {pc_cfg.visibility}, goal {pc_cfg.goal_mode}; "
            f"proprio {len(proprio)} of {int(inner.cfg.observation_space)} obs columns, aux {len(aux)}",
            flush=True,
        )

    def _init_scene(self, inner, pc_cfg, cams, names, n_proprio: int, n_aux: int) -> None:
        import torch

        from isaacsimenvs.tasks.cloth.utils import robot_surface
        from isaacsimenvs.tasks.cloth.utils.scene_pointcloud import LagBuffer, SceneCloud

        dev = self.device
        surf = robot_surface.load(inner.cfg.assets.robot_urdf, names)
        fk_idx = robot_surface.fk_subset(surf, pc_cfg.fk_points) if pc_cfg.fk_points > 0 else None
        rots = None
        if pc_cfg.cam_a_rot is not None and pc_cfg.cam_b_rot is not None:
            rots = torch.tensor([pc_cfg.cam_a_rot, pc_cfg.cam_b_rot], dtype=torch.float32).view(2, 3, 3)
        table_z = float(inner.cfg.reset.table_reset_z) + float(inner.cfg.cloth.table_half_thickness)
        self.pc = SceneCloud(
            pc_cfg, self.n, int(inner.cfg.cloth.resolution), cams, table_z,
            surf["body"], surf["offset"], surf["weight"], dev, cam_rots=rots, fk_idx=fk_idx,
        )
        n_body = len(names)
        self.body_buf = LagBuffer(self.n, self.proprio_delay if pc_cfg.fk_points > 0 else 0, (n_body, 7), dev)
        print(
            f"[distill] scene cloud: {pc_cfg.n_points} live points from {pc_cfg.n_cloth_cand} cloth + "
            f"{self.pc.n_robot} robot candidates ({len(surf['links'])} links), image {self.pc.w}x{self.pc.h}, "
            f"goal {pc_cfg.goal_mode}, fk {pc_cfg.fk_points}, labels {self.pc.labels or 'none'}, "
            f"cloud delay <= {pc_cfg.delay_max}, proprio delay <= {self.proprio_delay}; "
            f"proprio {n_proprio} of {int(inner.cfg.observation_space)} obs columns, aux {n_aux}",
            flush=True,
        )

    def _stationary(self):
        import torch

        parts_w = self.inner._particles_w()
        rot, centroid = self.inner._stationary_frame(parts_w)
        origins = self.inner.scene.env_origins
        return parts_w - origins.unsqueeze(1), centroid - origins, torch.atan2(rot[:, 1, 0], rot[:, 0, 0])

    def reset(self, ids) -> None:
        """Call for the envs whose episode just started, before ``__call__``."""
        if ids.numel() == 0:
            return
        import torch

        parts, centroid, yaw = self._stationary()
        if self.scene:
            self.pc.reset(ids)
        else:
            self.pc.reset(ids, parts[ids].mean(dim=1))
        self.p_lag[ids] = torch.randint(0, self.proprio_delay + 1, (int(ids.numel()),), device=self.device)
        self.fresh[ids] = True
        self.start_xy[ids], self.start_yaw[ids] = centroid[ids, :2], yaw[ids]
        self.max_drag_xy[ids] = 0.0
        self.max_drag_yaw[ids] = 0.0

    def __call__(self, teacher_obs):
        import torch
        from isaaclab.utils.math import quat_apply

        inner, n = self.inner, self.n
        origins = inner.scene.env_origins
        parts, centroid, yaw = self._stationary()
        self.drag_xy = (centroid[:, :2] - self.start_xy).norm(dim=-1)
        dyaw = yaw - self.start_yaw
        self.drag_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
        self.max_drag_xy = torch.maximum(self.max_drag_xy, self.drag_xy)
        self.max_drag_yaw = torch.maximum(self.max_drag_yaw, self.drag_yaw)

        pos, quat = _t(inner.robot.data.body_pos_w), _t(inner.robot.data.body_quat_w)
        proprio = torch.nan_to_num(teacher_obs[:, self.proprio_cols], nan=0.0, posinf=0.0, neginf=0.0)
        proprio = self.p_buf.push(proprio, self.p_lag, self.fresh)
        aux = torch.nan_to_num(teacher_obs[:, self.aux_cols], nan=0.0, posinf=0.0, neginf=0.0) * self.scale
        if self.scene:
            body = torch.cat((pos - origins.unsqueeze(1), quat), dim=-1)
            old = self.body_buf.push(body, self.p_lag, self.fresh)      # what the delayed joint angles give
            self.fresh[:] = False
            points, mask = self.pc.observe(
                inner.episode_length_buf, parts, body[..., :3], body[..., 3:], old[..., :3], old[..., 3:]
            )
            return {"proprio": proprio, "points": points, "mask": mask, "aux": aux}
        self.fresh[:] = False

        def ends(ids, off):
            q = quat[:, ids].reshape(-1, 4)
            o = off.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
            return pos[:, ids] + quat_apply(q, o).view(n, -1, 3) - origins.unsqueeze(1)

        caps_a, caps_b = ends(self.cap_ia, self.cap_oa), ends(self.cap_ib, self.cap_ob)
        pid = inner._palm_body_id
        offset = torch.as_tensor(inner.robot_spec.palm_center_offset, dtype=pos.dtype, device=pos.device)
        palm = pos[:, pid] + quat_apply(quat[:, pid].reshape(-1, 4), offset.expand(n, 3)) - origins
        tips = pos[:, inner._fingertip_body_ids] - origins.unsqueeze(1)
        goal = inner.fold_targets_w() - origins.unsqueeze(1)
        points, mask = self.pc.observe(
            inner.episode_length_buf, parts, palm.reshape(n, 3), goal, centroid, caps_a, caps_b, tips
        )
        # An env whose sim went non-finite (cloth.nan_policy resets it next step) must not put NaN
        # into the networks: its bad points are masked out and every input is made finite.
        ok = torch.isfinite(points).all(dim=-1) & (points[..., :3].abs() < 50.0).all(dim=-1)
        mask = mask & ok
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0) * mask.unsqueeze(-1)
        return {"proprio": proprio, "points": points, "mask": mask, "aux": aux}


def _kp_err_cm(pred, target, scale: float):
    """Mean keypoint distance [cm] of the two aux groups (rel palm, rel goal): ``(N,), (N,)``."""
    d = ((pred.float() - target.float()) / scale).view(*pred.shape[:-1], 2, -1, 3).norm(dim=-1).mean(dim=-1)
    return d[..., 0] * SCALE_CM, d[..., 1] * SCALE_CM


DRAG_BINS = ((0.0, 0.01), (0.01, 0.03), (0.03, 1e9))
DRAG_NAMES = ("drag<1cm", "drag1-3cm", "drag>3cm")


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description="Point-cloud student distillation (DAgger) and evaluation.")
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", default=None, help="cfg/experiment recipe (the teacher's), train mode")
    p.add_argument("--teacher", required=True, help="teacher checkpoint (.pth)")
    p.add_argument("--student", default=None, help="student checkpoint: eval, or a train warm start")
    p.add_argument("--policy", choices=("student", "teacher"), default="student", help="eval: who drives")
    p.add_argument("--out", default=None, help="train: output directory")
    p.add_argument("--json", default=None, help="eval: result file")
    p.add_argument("--seed", type=int, default=1)
    # point cloud
    p.add_argument("--vis", choices=("cams", "none"), default="cams")
    p.add_argument("--cloud", choices=("legacy", "scene"), default="legacy",
                   help="legacy: labelled cloth + hand points; scene: everything in the box over the table, xyz only")
    p.add_argument("--goal", choices=("g0", "g3", "g1", "gl", "none", "first"), default="g0",
                   help="legacy: g0 g3 g1 gl; scene: none | first (first-frame cloth cloud, moving / stationary)")
    p.add_argument("--n_points", type=int, default=1024, help="scene: live points")
    p.add_argument("--dump_cloud", default=None, help="eval, scene: save envs 0-3's cloud at steps 0/60/150 to this .npz")
    p.add_argument("--fk_points", type=int, default=0, help="scene: forward-kinematics points (forearm + hand)")
    p.add_argument("--pc_delay", type=int, default=0, help="scene: the cloud is up to this many steps old")
    p.add_argument("--proprio_delay", type=int, default=0, help="the student's proprioception is up to this many steps old")
    p.add_argument("--action_delay", type=int, default=0, help="env: actions applied up to this many steps late")
    p.add_argument("--env_obs_delay", type=int, default=0, help="env: the teacher's whole observation delayed (teacher eval)")
    p.add_argument("--noise", action="store_true", help="version C point-cloud noise (noise_preset)")
    p.add_argument("--g3_noise", action="store_true", help="g3 estimate error (g3_noise_preset)")
    p.add_argument("--eval_noise", action="store_true", help="eval: version C noise whatever the student trained with")
    p.add_argument("--eval_clean", action="store_true", help="eval: no point-cloud noise whatever the student trained with")
    p.add_argument("--eval_noise_only", default=None, choices=("occl", "calib", "depth", "drop", "tip"),
                   help="eval: one part of the version C noise (occl = camera position + capsule margin)")
    p.add_argument("--eval_g3_scale", type=float, default=None, help="eval: g3 estimate error = preset x this")
    p.add_argument("--hand_cam", action="store_true", help="hand points as the cameras see them (g1 / gl)")
    p.add_argument("--n_cloth", type=int, default=192)
    p.add_argument("--ref_step", type=int, default=0)
    p.add_argument("--pc_chunk", type=int, default=1024)
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
    # eval protocol (as eval/episodes.py)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=900)
    p.add_argument("--success_tolerance", type=float, default=0.01)
    p.add_argument("--randomize", action="store_true", help="eval: keep DR and reset noise on")
    p.add_argument("--single_variant", action="store_true")
    p.add_argument("--num_assets_per_type", type=int, default=1)
    p.add_argument("--allow_overflow", action="store_true")
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
    from isaacsimenvs.distill.student import StudentNet, TeacherNet
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.tasks.cloth.utils.cloth_pointcloud import PointCloudCfg, g3_noise_preset, noise_preset
    from isaacsimenvs.tasks.cloth.utils.scene_pointcloud import SceneCloudCfg, scene_noise_preset
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    def delays_of(a: dict) -> tuple[int, int]:
        """(cloud, proprio) delay for these pc_args; eval flags override what the student trained with."""
        pc_d, pr_d = int(a.get("pc_delay", 0)), int(a.get("proprio_delay", 0))
        if args.mode == "eval":
            if args.eval_clean:
                pc_d, pr_d = 0, 0
            if args.eval_noise:
                pc_d, pr_d = max(pc_d, args.pc_delay), max(pr_d, args.proprio_delay)
        return pc_d, pr_d

    def pc_cfg_from(a: dict):
        if a.get("cloud", "legacy") == "scene":
            cfg = SceneCloudCfg(
                n_points=int(a["n_points"]), goal_mode=a["goal"], ref_step=int(a["ref_step"]),
                fk_points=int(a["fk_points"]), chunk=int(args.pc_chunk),
            )
            noisy = (a["noise"] and not (args.mode == "eval" and args.eval_clean)) or (args.mode == "eval" and args.eval_noise)
            if noisy:
                scene_noise_preset(cfg)
            cfg.delay_max = delays_of(a)[0]
            return cfg
        cfg = PointCloudCfg(
            n_cloth=int(a["n_cloth"]), visibility=a["vis"], goal_mode=a["goal"],
            ref_step=int(a["ref_step"]), chunk=int(args.pc_chunk),
            hand_cam=bool(a.get("hand_cam", False)),
        )
        if a["noise"] and not (args.mode == "eval" and (args.eval_clean or args.eval_noise_only)):
            noise_preset(cfg)
        if args.mode == "eval" and args.eval_noise_only:
            full = noise_preset(PointCloudCfg())
            fields = {"occl": ("cam_pos_noise", "margin_range"), "calib": ("calib_xyz", "calib_rot_deg"),
                      "depth": ("depth_noise_std_range",), "drop": ("drop_prob_range",),
                      "tip": ("tip_drop_prob",)}[args.eval_noise_only]
            for f in fields:
                setattr(cfg, f, getattr(full, f))
        if a["g3_noise"]:
            g3_noise_preset(cfg)
        if args.mode == "eval":                       # stress tests: input unlike the training one
            if args.eval_noise:
                noise_preset(cfg)
            if args.eval_g3_scale is not None and a["goal"] == "g3":
                g3_noise_preset(cfg)
                k = float(args.eval_g3_scale)
                cfg.g3_bias_xy, cfg.g3_bias_yaw_deg = cfg.g3_bias_xy * k, cfg.g3_bias_yaw_deg * k
                cfg.g3_noise_xy, cfg.g3_noise_yaw_deg = cfg.g3_noise_xy * k, cfg.g3_noise_yaw_deg * k
        return cfg

    @hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
    def run(env_cfg, _agent_cfg) -> None:
        train = args.mode == "train"
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = args.num_assets_per_type
        env_cfg.sim.device = args.sim_device
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = args.success_tolerance
        if not train and not args.randomize:
            disable_randomization(env_cfg)
        from isaacsimenvs.distill.pc_rl import load_student_any

        student_ckpt = load_student_any(args.student) if args.student else None   # distilled or RL-trained
        dr = env_cfg.domain_randomization
        action_delay = int(args.action_delay)
        if not train and student_ckpt is not None and not args.eval_clean:   # as the student trained
            action_delay = max(action_delay, int(student_ckpt["pc_args"].get("action_delay", 0)))
        if action_delay > 0:
            dr.use_action_delay, dr.action_delay_max, dr.delay_mode = True, action_delay, "per_episode"
        if args.env_obs_delay > 0:
            dr.use_obs_delay, dr.obs_delay_max, dr.delay_mode = True, int(args.env_obs_delay), "per_episode"
        if args.single_variant:
            use_single_object_variant()
        torch.manual_seed(args.seed)

        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        device, n = inner.device, inner.num_envs
        num_actions = int(inner.cfg.action_space)

        ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        ckpt = ckpt[0] if 0 in ckpt else ckpt
        teacher = TeacherNet(ckpt["model"]).to(device).eval()
        if teacher.obs_dim != int(inner.cfg.observation_space):
            raise ValueError(f"teacher reads {teacher.obs_dim} obs columns, the env gives {inner.cfg.observation_space}")

        pc_args = dict(vis=args.vis, goal=args.goal, noise=args.noise, g3_noise=args.g3_noise,
                       n_cloth=args.n_cloth, ref_step=args.ref_step, hand_cam=args.hand_cam,
                       cloud=args.cloud, n_points=args.n_points, fk_points=args.fk_points,
                       pc_delay=args.pc_delay, proprio_delay=args.proprio_delay, action_delay=args.action_delay)
        if args.cloud == "scene" and args.goal not in ("none", "first"):
            raise ValueError("--cloud scene takes --goal none or first")
        net_args = dict(use_rnn_state=not args.no_rnn_state)
        if student_ckpt is not None and not train:
            pc_args, net_args = student_ckpt["pc_args"], student_ckpt["net_args"]   # the student's own input
        sobs = StudentObs(inner, pc_cfg_from(pc_args), proprio_delay=delays_of(pc_args)[1])
        point_dim = int(sobs.pc.point_dim)
        need_student = train or args.policy == "student"
        student = None
        if need_student:
            student = StudentNet(
                int(sobs.proprio_cols.numel()), num_actions, point_dim, int(sobs.aux_cols.numel()), **net_args
            ).to(device)
            if student_ckpt is not None:
                # an RL-trained actor has no auxiliary head: that part keeps its initial values
                student.load_state_dict(student_ckpt["model"], strict=not student_ckpt.get("from_rl", False))
                print(f"[distill] student <- {args.student} (iter {student_ckpt.get('iter')})", flush=True)
            elif not args.no_teacher_init:
                student.load_teacher(teacher, sobs.proprio_cols.cpu())
            elif train:
                student.load_teacher(teacher, sobs.proprio_cols.cpu())   # the normaliser is always the teacher's
                fresh = StudentNet(student.n_proprio, num_actions, point_dim, student.aux_dim, **net_args)
                keep = {k: v for k, v in student.state_dict().items() if k in ("mean", "var")}
                student.load_state_dict({**fresh.state_dict(), **keep})
                student.to(device)
        amp = (not args.no_amp) and str(device).startswith("cuda")

        def autocast():
            return torch.autocast("cuda", dtype=torch.float16, enabled=amp)

        all_ids = torch.arange(n, device=device)
        obs, _ = env.reset()
        if not train and not args.allow_overflow:
            from isaacsimenvs.newton.contact_guard import assert_no_buffer_overflow

            assert_no_buffer_overflow(env)
            env.reset()
            obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((n, num_actions), device=device))
        sobs.reset(all_ids)
        t_state = teacher.initial_state(n, device)
        s_state = student.initial_state(n, device) if student is not None else None
        start = torch.ones(n, dtype=torch.bool, device=device)

        def policy_step(tobs, so, use_teacher=None):
            """Teacher label and student output for this observation; returns the action to apply."""
            nonlocal t_state, s_state
            t_act, t_state = teacher.step(torch.nan_to_num(tobs, nan=0.0, posinf=0.0, neginf=0.0), t_state, start)
            if student is None:
                return t_act, t_act, None
            with torch.no_grad(), autocast():
                mu, aux, s_state = student.step(so["proprio"], so["points"], so["mask"], s_state, start)
            s_act = torch.nan_to_num(mu.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            act = s_act if use_teacher is None else torch.where(use_teacher.unsqueeze(1), t_act, s_act)
            return act, t_act, aux.float()

        # ================================================================== eval
        if not train:
            capture = capture_cm = None
            if not args.allow_overflow:
                from isaacsimenvs.newton.contact_guard import capture_fd_output, raise_if_overflowed

                capture_cm = capture_fd_output()
                capture = capture_cm.__enter__()
            done_once = torch.zeros(n, dtype=torch.bool, device=device)
            goals = torch.full((n,), -1, dtype=torch.long, device=device)
            length = torch.zeros(n, dtype=torch.long, device=device)
            reasons: dict[str, torch.Tensor] = {}
            ep_drag_xy = torch.zeros(n, device=device)
            ep_drag_yaw = torch.zeros(n, device=device)
            best_err = torch.full((n,), float("inf"), device=device)
            err_sum = torch.zeros(2, device=device)
            bin_sum = torch.zeros(len(DRAG_BINS), 2, device=device)
            bin_cnt = torch.zeros(len(DRAG_BINS), device=device)
            vis_sum, steps_cnt = 0.0, 0.0
            action_gap = 0.0
            dump: dict = {}
            step = 0
            try:
                while step < args.max_steps and not bool(done_once.all()):
                    tobs = obs["policy"]
                    so = sobs(tobs)
                    live = ~done_once
                    act, t_act, aux = policy_step(tobs, so, None if args.policy == "student" else torch.ones_like(start))
                    w = live.float()
                    if aux is not None:
                        e_palm, e_goal = _kp_err_cm(aux, so["aux"], sobs.scale)
                        err_sum += torch.stack(((e_palm * w).sum(), (e_goal * w).sum()))
                        for b, (lo, hi) in enumerate(DRAG_BINS):
                            sel = w * ((sobs.drag_xy >= lo) & (sobs.drag_xy < hi)).float()
                            bin_sum[b] += torch.stack(((e_palm * sel).sum(), (e_goal * sel).sum()))
                            bin_cnt[b] += sel.sum()
                        action_gap += float((((act - t_act) ** 2).mean(dim=1) * w).sum())
                    else:
                        for b, (lo, hi) in enumerate(DRAG_BINS):
                            bin_cnt[b] += (w * ((sobs.drag_xy >= lo) & (sobs.drag_xy < hi)).float()).sum()
                    vis_sum += float((sobs.pc.last_visible_frac * w).sum())
                    steps_cnt += float(w.sum())
                    best_err = torch.where(live, torch.minimum(best_err, inner.fold_error()), best_err)
                    drag_xy_now, drag_yaw_now = sobs.max_drag_xy.clone(), sobs.max_drag_yaw.clone()

                    if args.dump_cloud and sobs.scene and step in (0, 60, 150):
                        pos_w, quat_w = _t(inner.robot.data.body_pos_w), _t(inner.robot.data.body_quat_w)
                        org = inner.scene.env_origins
                        dump[f"robot_all_{step}"] = sobs.pc.robot_points(pos_w[:4] - org[:4].unsqueeze(1), quat_w[:4]).cpu().numpy()
                        dump[f"body_pos_{step}"] = (pos_w[:4] - org[:4].unsqueeze(1)).cpu().numpy()
                        dump[f"particles_{step}"] = (inner._particles_w()[:4] - org[:4].unsqueeze(1)).cpu().numpy()
                        dump[f"tips_{step}"] = (pos_w[:4][:, inner._fingertip_body_ids] - org[:4].unsqueeze(1)).cpu().numpy()
                        for key in ("last_xyz", "last_mask", "last_label", "last_cam"):
                            dump[f"{key}_{step}"] = getattr(sobs.pc, key)[:4].cpu().numpy()
                        dump["cams"], dump["rots"] = sobs.pc.cams[:4].cpu().numpy(), sobs.pc.rots[:4].cpu().numpy()
                        dump["box_lo"], dump["box_hi"] = sobs.pc.box_lo.cpu().numpy(), sobs.pc.box_hi.cpu().numpy()
                        dump["body_names"] = list(inner.robot.body_names)
                        import numpy as np

                        np.savez_compressed(args.dump_cloud, **dump)
                    obs, _rew, terminated, truncated, extras = env.step(act)
                    length += live.long()
                    ended = terminated.bool() | truncated.bool()
                    first = ended & live
                    if bool(first.any()):
                        final = extras["episode_final"]
                        goals[first] = final["successes"][first].long()
                        ep_drag_xy[first], ep_drag_yaw[first] = drag_xy_now[first], drag_yaw_now[first]
                        for key, value in final.items():
                            if key.startswith("done_"):
                                r = reasons.setdefault(key[5:], torch.zeros(n, dtype=torch.bool, device=device))
                                r[first] = value[first].bool()
                        done_once |= first
                    sobs.reset(ended.nonzero(as_tuple=True)[0])
                    start = ended
                    step += 1
                    if capture is not None and step % 25 == 0:
                        raise_if_overflowed(capture(), n, f"the rollout (step {step})")
                    if step % 100 == 0:
                        print(f"[distill] eval step {step}: {int(done_once.sum())}/{n} envs done", flush=True)
            finally:
                if capture is not None:
                    pending = capture()
                    capture_cm.__exit__(None, None, None)
                    sys.stdout.write(pending)
                    sys.stdout.flush()

            m = done_once.cpu()
            g = goals.cpu()[m]
            q = lambda t, v: round(float(torch.quantile(t[done_once].float(), v)), 4) if bool(done_once.any()) else None  # noqa: E731
            cnt = max(steps_cnt, 1.0)
            result = {
                "task": args.task, "policy": args.policy, "teacher": args.teacher, "student": args.student,
                "pc_args": pc_args, "net_args": net_args, "num_envs": n, "seed": args.seed,
                "eval_flags": {"clean": bool(args.eval_clean), "noise": bool(args.eval_noise),
                               "action_delay": action_delay, "env_obs_delay": int(args.env_obs_delay),
                               "cloud_delay": int(getattr(sobs.pc.cfg, "delay_max", 0)), "proprio_delay": sobs.proprio_delay},
                "randomization": bool(args.randomize), "steps_run": step, "censored": int(n - m.sum()),
                "keypoint_tolerance": float(inner.cfg.cloth.keypoint_tolerance),
                "success_rate": round(float((g > 0).float().mean()), 4) if len(g) else None,
                "n": int(m.sum()),
                "mean_episode_length": round(float(length[done_once].float().mean()), 1) if len(g) else None,
                "termination_reasons": {k: int(v[done_once].sum()) for k, v in sorted(reasons.items())},
                "best_fold_err_mean_m": round(float(best_err.mean()), 4),
                "max_drag_xy_m": {f"p{int(v * 100)}": q(ep_drag_xy, v) for v in (0.1, 0.5, 0.9)},
                "max_drag_yaw_deg": {
                    f"p{int(v * 100)}": (None if q(ep_drag_yaw, v) is None else round(math.degrees(q(ep_drag_yaw, v)), 2))
                    for v in (0.1, 0.5, 0.9)
                },
                "share_of_steps_by_drag": {nm: round(float(bin_cnt[b] / cnt), 4) for b, nm in enumerate(DRAG_NAMES)},
                "visible_cloth_fraction": round(vis_sum / cnt, 4),
                "per_env_success": [int(v > 0) if v >= 0 else None for v in goals.cpu().tolist()],
                "per_env_max_drag_xy_m": [round(float(v), 4) for v in ep_drag_xy.cpu()],
            }
            if student is not None:
                result["aux_err_cm"] = {"keypoints_rel_palm": round(float(err_sum[0] / cnt), 3),
                                        "keypoints_rel_goal": round(float(err_sum[1] / cnt), 3)}
                result["aux_goal_err_cm_by_drag"] = {
                    nm: (round(float(bin_sum[b, 1] / bin_cnt[b]), 3) if float(bin_cnt[b]) > 0 else None)
                    for b, nm in enumerate(DRAG_NAMES)
                }
                result["action_mse_vs_teacher"] = round(action_gap / cnt, 5)
            print("\n" + "=" * 72)
            print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_env")}, indent=1))
            print("=" * 72, flush=True)
            if args.json:
                out = Path(args.json)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=1))
                print(f"[distill] wrote {out}", flush=True)
            env.close()
            return

        # ================================================================== train (DAgger)
        out_dir = Path(args.out or "outputs/distill/run")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps({**vars(args), "hydra": hydra_args}, indent=1, default=str))
        opt = torch.optim.Adam(student.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        h, m_pts = int(args.horizon), sobs.pc.num_points
        buf = {
            "proprio": torch.zeros(h, n, int(sobs.proprio_cols.numel()), device=device),
            "points": torch.zeros(h, n, m_pts, point_dim, device=device, dtype=torch.float16),
            "mask": torch.zeros(h, n, m_pts, dtype=torch.bool, device=device),
            "act": torch.zeros(h, n, num_actions, device=device),
            "aux": torch.zeros(h, n, int(sobs.aux_cols.numel()), device=device),
            "start": torch.zeros(h, n, dtype=torch.bool, device=device),
            "drag": torch.zeros(h, n, device=device),
        }
        log_path = out_dir / "log.jsonl"
        win = {"s_ep": 0, "s_ok": 0, "t_ep": 0, "t_ok": 0, "loss_act": 0.0, "loss_aux": 0.0, "upd": 0,
               "e_palm": 0.0, "e_goal": 0.0, "vis": 0.0, "pts": 0.0, "cloth": 0.0, "it": 0, "skipped": 0,
               "bin_sum": torch.zeros(len(DRAG_BINS), device=device), "bin_cnt": torch.zeros(len(DRAG_BINS), device=device)}
        first_iter = int(student_ckpt.get("iter", 0)) if student_ckpt is not None else 0
        t0 = time.time()
        frames = 0

        def save(it: int, name: str) -> None:
            torch.save({"model": student.state_dict(), "iter": it, "pc_args": pc_args, "net_args": net_args,
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
                tobs = obs["policy"]
                so = sobs(tobs)
                act, t_act, _aux = policy_step(tobs, so, use_teacher)
                buf["proprio"][t], buf["points"][t], buf["mask"][t] = so["proprio"], so["points"].to(torch.float16), so["mask"]
                buf["act"][t], buf["aux"][t], buf["start"][t], buf["drag"][t] = t_act, so["aux"], start, sobs.drag_xy
                win["vis"] += float(sobs.pc.last_visible_frac.mean()) / h
                if sobs.scene:
                    win["pts"] += float(sobs.pc.last_count.mean()) / h
                    win["cloth"] += float(sobs.pc.last_cloth_share.mean()) / h
                obs, _rew, terminated, truncated, extras = env.step(act)
                ended = terminated.bool() | truncated.bool()
                if bool(ended.any()):
                    ok = extras["episode_final"]["successes"] > 0
                    win["s_ep"] += int((ended & ~use_teacher).sum())
                    win["s_ok"] += int((ended & ~use_teacher & ok).sum())
                    win["t_ep"] += int((ended & use_teacher).sum())
                    win["t_ok"] += int((ended & use_teacher & ok).sum())
                sobs.reset(ended.nonzero(as_tuple=True)[0])
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
                    e_palm, e_goal = _kp_err_cm(aux, buf["aux"][:, mb], sobs.scale)
                    win["loss_act"] += float(loss_act)
                    win["loss_aux"] += float(loss_aux)
                    win["e_palm"] += float(e_palm.mean())
                    win["e_goal"] += float(e_goal.mean())
                    win["upd"] += 1
                    drag = buf["drag"][:, mb]
                    for b, (lo, hi) in enumerate(DRAG_BINS):
                        sel = (drag >= lo) & (drag < hi)
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
                    "skipped_minibatches": win["skipped"],
                    "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                }
                print("[distill] " + json.dumps(row), flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                for key in ("s_ep", "s_ok", "t_ep", "t_ok", "upd", "it"):
                    win[key] = 0
                for key in ("loss_act", "loss_aux", "e_palm", "e_goal", "vis", "pts", "cloth"):
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
