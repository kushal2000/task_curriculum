"""Observation audit: build the plate world (RigidClothNewton, the policy's recipe) or the VBD world
(Cloth, eval/vbd.yaml protocol + the student's cloud), as eval.py does, and record what the point-cloud
policy receives: ZERO actions for --zero_steps, then the policy for --policy_steps.

    python experiments/nov25_better_rigidcloth/obs_audit/capture.py --world plates|vbd [--rand] \
        --out outputs/obs_audit/plates_pinned.pt [--patch NAME] [hydra overrides...]

Saved per step (env frame = world - env origin): obs["policy"], the decoded live / goal clouds, the
sheet's grid points, the goal sheet now / frozen, keypoints, stationary-frame yaw, fold_error,
aligned / layered / in-place distances, fold flags, the teacher (keypoint) vector, palm pos.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

CKPT = "pretrained_policy/play_pc_rigidclothRL_64live.pth"
RECIPE = "experiments/nov24_play_pc_rl/play_pc_rl_A.yaml"
PROTOCOL = "isaacsimenvs/tasks/cloth/config/eval/vbd.yaml"
PLATE_OVR = [
    "env.cloth.keypoint_tolerance=0.015", "env.cloth.nan_policy=reset", "env.termination.success_steps=10",
    "env.fold_pool.shapes=[box2-mid]", "env.fold_pool.one_shape_per_env=false", "env.fold_pool.init_fold_prob=0.0",
    "env.fold_pool.goal_task.mode=half_fold", "env.fold_pool.size_range=[0.10,0.10]", "env.robot_self_collision=false",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", choices=("plates", "vbd"), required=True)
    p.add_argument("--rand", action="store_true", help="random starts + the policy's DR (eval --randomize)")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--zero_steps", type=int, default=20)
    p.add_argument("--policy_steps", type=int, default=100)
    p.add_argument("--patch", default="none")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=CKPT)
    args, hydra = p.parse_known_args()
    task_id = "Isaacsimenvs-RigidClothNewton-Direct-v0" if args.world == "plates" else "Isaacsimenvs-Cloth-Direct-v0"
    layer = RECIPE if args.world == "plates" else PROTOCOL
    sys.argv = [sys.argv[0]] + (PLATE_OVR if args.world == "plates" else []) + hydra

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.scripts.evaluation import eval as ev
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    student = ev._student_checkpoint(args.checkpoint)
    out = {}

    @hydra_task_config_with_yaml(task_id, "", experiment=layer)
    def run(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = "cuda:0"
        env_cfg.seed = args.seed
        if not args.rand:
            disable_randomization(env_cfg)
        from isaacsimenvs.scripts.distillation.student import apply_student_env

        if args.world == "vbd":
            apply_student_env(env_cfg, student["env"], with_dr=args.rand)
            ev._goal_student_success(env_cfg)
        else:
            ev._check_student_cloud(env_cfg, student["env"])
        import patches_obs_audit as pa

        pa.apply(args.patch, env_cfg)
        env = gym.make(task_id, cfg=env_cfg)
        inner = env.unwrapped
        pa.apply_env(args.patch, inner)
        dev, n = inner.device, inner.num_envs
        from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer

        player = EnvCloudStudentPlayer(inner, student, "cuda:0")
        lay = dict(inner.observations.pointcloud_layout)
        cloud = inner.observations.pointcloud.cloud
        from isaacsimenvs.tasks.cloth.mdp.observations import FIELD_SIZES
        from isaacsimenvs.tasks.cloth.perception.proprio import PER_JOINT_FIELDS

        nj = inner.robot_layout.num_joints
        fields = [(f, nj if f in PER_JOINT_FIELDS else FIELD_SIZES[f]) for f in inner.observations.proprio_fields]
        tfields = [(f, nj if f in PER_JOINT_FIELDS else FIELD_SIZES[f]) for f in inner.cfg.obs.obs_list]
        geo = inner.task.geometry
        meta = dict(
            world=args.world, rand=args.rand, patch=args.patch, layout=lay, proprio_fields=fields, teacher_fields=tfields,
            centre=cloud.centre.cpu(), scale=float(cloud.cfg.scale), table_z_cloud=float(cloud.table_z),
            cams=cloud.cams0.cpu(), box_lo=cloud.box_lo.cpu(), box_hi=cloud.box_hi.cpu(),
            keypoint_idx=geo.keypoint_idx.cpu(), moving_idx=geo.moving_idx.cpu(), stationary_idx=geo.stationary_idx.cpu(),
            rest_local=geo.rest_local.cpu(), fold_lift=float(geo.layout.fold_lift),
            object_scale=getattr(inner.backend, "object_scale", None),
            robot_root_pos=list(inner.cfg.robot_root_pos), hand_sysid=bool(inner.cfg.hand_sysid),
            table_reset_z=float(inner.cfg.reset.table_reset_z), table_half=float(inner.cfg.cloth.table_half_thickness),
            start_height=float(inner.cfg.cloth.start_height), fixed_start_pose=inner.cfg.reset.fixed_start_pose,
            success=str(inner.cfg.termination.success), surface_layered=bool(inner.task.surface_layered),
            surface_ref_step=int(inner.cfg.termination.surface.ref_step), tol=inner.task.surface_tolerance().cpu(),
            keypoint_tolerance=float(inner.cfg.cloth.keypoint_tolerance), footprint_max=float(inner.cfg.cloth.max_folded_footprint),
            dr_pc=str(inner.cfg.domain_randomization.pointcloud), dr_proprio=str(inner.cfg.domain_randomization.proprio),
            dr_action=str(inner.cfg.domain_randomization.action), dr_robot=str(inner.cfg.domain_randomization.robot),
            obs_pc=str(inner.cfg.obs.pointcloud), obs_list=list(inner.cfg.obs.obs_list),
            action_cfg=str(inner.cfg.action), decimation=int(inner.cfg.decimation), dt=float(inner.cfg.sim.dt),
            episode_length_s=float(inner.cfg.episode_length_s),
            goal_tasks=None if geo.goal_tasks is None else dict(names=geo.goal_tasks.names, kp=geo.goal_tasks.kp.cpu(),
                                                                moving=geo.goal_tasks.moving.cpu()),
        )
        if getattr(inner.backend, "env_size", None) is not None:
            meta["env_size"] = inner.backend.env_size.cpu()
        print("[audit] layout", lay, flush=True)

        obs, _ = env.reset()
        obs, _ = env.reset()
        zero = torch.zeros((n, inner.cfg.action_space), device=dev)
        obs, *_ = env.step(zero)
        # The reset pose the sheet got (the env's start): yaw of the stationary frame at this point.
        rec = []
        diag = []
        origins = inner.scene.env_origins
        n_p, m, d = lay["n_proprio"], lay["num_points"], lay["point_dim"]
        total = args.zero_steps + args.policy_steps
        succ_any = torch.zeros(n, dtype=torch.bool, device=dev)
        ep_done = torch.zeros(n, dtype=torch.bool, device=dev)
        t0 = time.time()
        for step in range(total):
            pol = obs["policy"]
            state = inner.sim_state()
            view = inner.task.derive(state)
            pts = pol[:, n_p:n_p + m * d].reshape(n, m, d)
            mask = pol[:, n_p + m * d:n_p + m * d + m] > 0.5
            xyz = pts / meta["scale"] + cloud.centre  # env frame
            goal_now = inner.task.goal_now_w(view) - origins.unsqueeze(1)
            goal_ref = inner.task.goal_reference_w(view) - origins.unsqueeze(1)
            r = view.frame_r
            rec.append(dict(
                step=int(inner.episode_length_buf[0]),
                policy=pol.cpu(), teacher=obs["teacher"].cpu(), xyz=xyz.cpu(), mask=mask.cpu(),
                points=(view.points_w - origins.unsqueeze(1)).cpu(), goal_now=goal_now.cpu(), goal_ref=goal_ref.cpu(),
                frozen=inner.task.goal_is_frozen.cpu().clone(),
                keypoints=(view.keypoints_w - origins.unsqueeze(1)).cpu(), targets=(view.targets_w - origins.unsqueeze(1)).cpu(),
                frame_yaw=torch.atan2(r[:, 1, 0], r[:, 0, 0]).cpu(), fold_error=view.fold_error.cpu(),
                footprint=view.footprint.cpu(),
                aligned=inner.task.aligned_dist.cpu().clone(),
                layered=(inner.task.layered_dist.cpu().clone() if inner.task.surface_layered else None),
                in_place=inner.task.in_place_dist.cpu().clone(),
                palm=(inner.sim_state().body_pos_w[:, inner.robot_layout.palm_body_id] - origins).cpu()
                if hasattr(inner.robot_layout, "palm_body_id") else None,
                joint_pos=state.joint_pos.cpu(), table_z=inner.task.table_z.cpu().clone(),
                episode_len=inner.episode_length_buf.cpu().clone(),
            ))
            if step in (0, 19, 60, 100, 150, 200):
                diag.append(dict(step=int(inner.episode_length_buf[0]), **pa.diagnose(inner)))
                print(f"[audit] diag {diag[-1]['step']}: quat raw-state {diag[-1]['max_abs_raw_minus_state']:.3f}, raw-xyzw(state) "
                      f"{diag[-1]['max_abs_raw_minus_state_as_xyzw']:.3f}, surface shift median {diag[-1]['robot_surface_point_shift_mm_median']:.1f} mm, "
                      f"visibility disagree {sum(diag[-1]['visibility_disagree_frac']) / n:.3f}", flush=True)
            if step < args.zero_steps:
                action = zero
                player.get_action(pol)  # keep the LSTM's state as it would be
            else:
                action = player.get_action(pol)
            obs, _r, term, trunc, extras = env.step(action.to(dev))
            ended = (term.bool() | trunc.bool())
            if bool(ended.any()):
                fin = extras["episode_final"]
                newly = ended & ~ep_done
                succ_any |= newly & (fin["done_fold"].bool() if "done_fold" in fin else fin["successes"].bool())
                ep_done |= ended
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])
        print(f"[audit] {total} steps in {time.time() - t0:.1f}s; folded (first episode) {int(succ_any.sum())}/{n}", flush=True)
        meta["folded_first_episode"] = succ_any.cpu()
        out.update(meta=meta, rec=rec, diag=diag)
        torch.save(out, args.out)
        print(f"[audit] wrote {args.out}", flush=True)
        os._exit(0)

    run()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
