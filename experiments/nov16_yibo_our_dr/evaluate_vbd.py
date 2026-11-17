"""Evaluate Yibo's unchanged policy on eight native VBD cloth material presets.

The default runs one isolated Newton process per material, sequentially on one GPU.
Each process scores only the first episode of each of 32 environments. Physical
fold success is reported; whole-robot release clearance is not implemented for VBD.
Use --check-config for a CPU-only configuration/cache preflight.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_MATERIALS = ('base', 'big_napkin_measured', 'chiffon_measured', 'cotton_measured',
                     'denim_measured', 'napkin_measured', 'silk_measured', 'towel_measured')


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--material', choices=DEFAULT_MATERIALS, help='Run one material; otherwise run all eight.')
    cli.add_argument('--num-envs', type=int, default=32)
    cli.add_argument('--steps', type=int, default=600, help='Policy steps per episode (600 = 10 seconds).')
    cli.add_argument('--size', type=float, default=.15)
    cli.add_argument('--resolution', type=int, default=7)
    cli.add_argument('--seed', type=int, default=0)
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--checkpoint', type=Path, default=ROOT/'pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth')
    cli.add_argument('--policy-config', type=Path, default=ROOT/'pretrained_policy/flexiv_wuji/config.yaml')
    cli.add_argument('--expl-coef', type=float, default=0.)
    cli.add_argument('--experiment', type=Path, help='Run recipe with env overrides applied after the material preset.')
    cli.add_argument('--out-dir', type=Path, default=ROOT/'experiments/nov16_yibo_our_dr/results/vbd_materials')
    cli.add_argument('--check-config', action='store_true')
    cli.add_argument('--skip-cache-check', action='store_true', help='Only with --check-config; inspect configs before asset baking.')
    return cli


def build_config(args, material):
    from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat
    compat.install_isaaclab3_compat()
    from isaacsimenvs import CONFIG_DIR
    from isaacsimenvs.scripts.evaluation.player import read_policy_cfg
    from isaacsimenvs.tasks.cloth.backends.vbd.config import DeformableClothEnvCfg
    from isaacsimenvs.tasks.cloth.backends.vbd import fabrics
    import yaml

    cfg = DeformableClothEnvCfg()
    cfg.from_dict(yaml.safe_load((CONFIG_DIR/'task/Cloth.yaml').read_text()))
    saved = read_policy_cfg(args.policy_config, args.device)['env']
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    # MuJoCo capacities are global. Reserve enough constraints/contacts for all hands.
    # The default 8192 constraints overflowed at 32 envs; these only size buffers.
    cfg.newton.njmax = max(cfg.newton.njmax, args.num_envs * 1024)
    cfg.newton.nconmax = max(cfg.newton.nconmax, args.num_envs * 256)
    cfg.robot_preset = 'yibo'
    cfg.robot_root_pos = (0., .51, .20)
    cfg.assets.robot_urdf = 'assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf'
    cfg.assets.table_urdf = 'assets/urdf/table_fold_pool.urdf'
    cfg.assets.robot_friction = .5
    cfg.assets.finger_tip_friction = 1.5
    cfg.assets.table_friction = .5  # Preset effective friction was calibrated against this shape coefficient.
    cfg.action.from_dict(saved['action'])
    cfg.obs.from_dict(saved['obs'])
    cfg.obs.keypoint_mode = 'yibo'
    cfg.obs.yibo_contact_thickness = .016
    cfg.obs.yibo_goal_lift = 0.
    cfg.reward.object_base_size = float(saved['reward']['object_base_size'])
    cfg.sim.dt, cfg.decimation = 1/120, 2
    cfg.episode_length_s = args.steps * cfg.sim.dt * cfg.decimation
    cfg.task.mode = 'fold'
    cfg.task.nan_policy = 'raise'
    cfg.task.release.enabled = False
    cfg.reset.table_reset_z, cfg.reset.table_reset_z_range = .38, 0.
    cfg.reset.table_reset_xy = (0., 0.)
    cfg.reset.reset_position_noise_x = cfg.reset.reset_position_noise_y = .05
    cfg.reset.random_yaw = True
    cfg.reset.yaw_range_deg = [0., 360.]
    dr = cfg.domain_randomization
    dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = dr.joint_pos_obs_noise_std = dr.action_noise_std = 0.
    dr.object_state_xyz_noise_std = dr.object_state_rotation_noise_degrees = 0.
    cloth = cfg.deformable_cloth
    cloth.size, cloth.resolution, cloth.start_height = args.size, args.resolution, .20
    cloth.from_dict(fabrics.cloth_params(material, plastic=True))
    if args.experiment is not None:
        from isaacsimenvs.scripts.configs.loader import load_experiment

        cfg.from_dict(load_experiment(str(args.experiment)).get('env', {}))
    cfg.task.check()
    cloth.check_thickness()
    return cfg


def check_cache(cfg):
    from isaacsimenvs.tasks.cloth.backends import assets
    from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import usd_cache
    for path, kwargs in (
        (cfg.assets.robot_urdf, dict(fix_base=True, self_collision=cfg.robot_self_collision, joint_drive=assets.position_drive())),
        (cfg.assets.table_urdf, dict(fix_base=False)),
    ):
        key = usd_cache.cache_key((ROOT/path).read_text(), kwargs)
        if not (usd_cache.DEFAULT_CACHE_DIR/key/'manifest.json').is_file():
            raise RuntimeError(f'USD cache miss for {path}; prepare this asset under Isaac Sim before submitting VBD.')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')
    temporary.replace(path)


def evaluate(args):
    import torch
    from isaacsimenvs.scripts.evaluation.contact_guard import capture_fd_output, raise_if_overflowed
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer
    from isaacsimenvs.tasks.cloth.backends.vbd import fabrics

    started = time.perf_counter()
    cfg = build_config(args, args.material)
    check_cache(cfg)
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv

    class EvaluationEnv(ClothEnv):
        def _get_dones(self):
            result = super()._get_dones()
            # Capture physical diagnostics BEFORE DirectRLEnv autoresets completed rows.
            state = self.backend.read_state()
            view = self.task.derive(state)
            self.eval_fold_error = view.fold_error.clone()
            self.eval_footprint = view.footprint.clone()
            self.eval_finite = all(bool(torch.isfinite(value).all()) for value in
                                   (state.points_w, state.points_vel_w, state.joint_pos, state.joint_vel,
                                    view.fold_error, view.footprint))
            from isaacsimenvs.tasks.cloth.mdp.geometry.yibo import yibo_object_features
            cg = self.backend.cloth_grid
            points, velocities = state.points_w, state.points_vel_w
            actual_corners = points[:, cg.corner_idx]
            actor_corners = yibo_object_features(cg, state, self.cfg.obs.yibo_goal_lift)[4]
            local = points-self.scene.env_origins[:, None]
            relative_xy = local[..., :2]-local.new_tensor(self.cfg.reset.table_reset_xy)
            edge_vectors = points[:, self.eval_edges[:, 1]]-points[:, self.eval_edges[:, 0]]
            stretch = edge_vectors.norm(dim=-1)/self.eval_rest_edge_lengths
            self.eval_metrics = {
                'joint_displacement_rad': (state.joint_pos-self.eval_initial_q).abs().amax(1),
                'arm_joint_displacement_rad': (state.joint_pos[:, self.controller._arm_ids]-self.eval_initial_q[:, self.controller._arm_ids]).abs().amax(1),
                'hand_joint_displacement_rad': (state.joint_pos[:, self.controller._hand_ids]-self.eval_initial_q[:, self.controller._hand_ids]).abs().amax(1),
                'palm_displacement_m': (state.palm_pos_w-self.eval_initial_palm).norm(dim=-1),
                'arm_target_tracking_rmse_rad': (state.joint_pos[:, self.controller._arm_ids]-self.controller._cur_targets[:, self.controller._arm_ids]).square().mean(1).sqrt(),
                'action_max_abs': self.eval_actions.abs().amax(1),
                'moving_height_above_table_m': (points[:, cg.moving_idx, 2]-state.table_top_z[:, None]).amax(1),
                'centroid_height_above_table_m': points[..., 2].mean(1)-state.table_top_z,
                'particle_speed_m_s': velocities.norm(dim=-1).amax(1),
                'edge_stretch_ratio': stretch.amax(1),
                'p95_edge_stretch_ratio': torch.quantile(stretch, .95, dim=1),
                'actor_marker_fit_rms_m': ((actor_corners-actual_corners).square().sum(-1).mean(-1)).sqrt(),
                'any_particle_outside_table_xy': (relative_xy.abs()>self.eval_table_half_xy).any(dim=(1, 2)),
                'centroid_outside_table_xy': (relative_xy.mean(1).abs()>self.eval_table_half_xy).any(dim=1),
                'any_particle_near_ground': (local[..., 2]<.1).any(dim=1),
            }
            return result

    result = dict(status='running', material=args.material,
                  material_preset=fabrics.load(args.material), checkpoint=str(args.checkpoint.resolve()),
                  policy_config=str(args.policy_config.resolve()),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'), seed=args.seed,
                  hostname=os.uname().nodename, gpu_name=torch.cuda.get_device_name(args.device),
                  gpu_total_memory_bytes=torch.cuda.get_device_properties(args.device).total_memory,
                  num_envs=args.num_envs, max_policy_steps=args.steps, deterministic=True,
                  sapg_expl_coef=args.expl_coef, actor_encoding='yibo', robot_preset='yibo',
                  release_evaluated=False,
                  limitation='Physical fold only. Whole-robot 10 cm release clearance is unsupported for VBD.',
                  experiment=str(args.experiment) if args.experiment is not None else None,
                  config=cfg.to_dict())
    destination = args.out_dir/f'{args.material}.json'
    write_json(destination, result)
    env = None
    try:
        with capture_fd_output() as captured:
            try:
                env = EvaluationEnv(cfg)
                player = PretrainedPlayer(args.policy_config, args.checkpoint, args.num_envs, device=args.device,
                                          num_observations=134, num_actions=27, expl_coef=args.expl_coef)
                # Runner initialization can reseed global RNG; restore requested evaluation seed.
                env.seed(args.seed)
                obs, _ = env.reset()
                player.reset_rnn()
                device, n = env.device, args.num_envs
                done_once = torch.zeros(n, dtype=torch.bool, device=device)
                folded = torch.zeros_like(done_once)
                lengths = torch.zeros(n, dtype=torch.long, device=device)
                best_error = torch.full((n,), float('inf'), device=device)
                best_footprint = torch.full((n,), float('inf'), device=device)
                final_error, final_footprint = torch.zeros(n, device=device), torch.zeros(n, device=device)
                reasons = {}
                if not all(torch.isfinite(value).all() for value in obs.values()):
                    raise RuntimeError('Nonfinite reset observation')
                # Geometric scoring controls: a flat sheet must fail, a noiseless perfect fold must pass.
                from isaacsimenvs.tasks.cloth.mdp.geometry import fold, grid
                cg = env.backend.cloth_grid
                perfect = grid.folded_start(cg, torch.full((n,), torch.pi, device=device))
                rotation, center = fold.stationary_frame(cg, perfect)
                target = fold.corner_targets(cg, rotation, center, torch.zeros(n, dtype=torch.bool, device=device))
                analytic_error = fold.fold_error(perfect[:, cg.corner_idx], target)
                analytic_footprint = fold.footprint_ratio(cg, perfect, rotation)
                assert float(analytic_error.max()) < 2e-6
                assert bool((analytic_footprint < cfg.task.max_folded_footprint).all())
                assert not bool(env.is_folded().any()), 'Initial flat sheet was scored as folded'
                result['analytic_perfect_fold_error_m'] = float(analytic_error.max())
                result['initial_joint_positions'] = env.backend.read_joint_pos().cpu().tolist()
                result['initial_cloth_centers_m'] = (env.backend.cloth_points()[0].mean(1)-env.scene.env_origins).cpu().tolist()
                from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
                triangles = torch.tensor(grid.grid_triangles(cg.resolution), device=device).reshape(-1, 3)
                edges = torch.cat((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
                env.eval_edges = torch.unique(torch.sort(edges, dim=1).values, dim=0)
                env.eval_rest_edge_lengths = (cg.rest[:, env.eval_edges[:, 1]]-cg.rest[:, env.eval_edges[:, 0]]).norm(dim=-1)
                env.eval_table_half_xy = torch.tensor(urdf_box_size(cfg.assets.table_urdf)[:2], device=device)/2
                initial_state = env.backend.read_state()
                from isaacsimenvs.tasks.cloth.backends.base import as_torch

                result['initial_table_centers_m'] = (
                    as_torch(env.backend.table.data.root_pos_w)-env.scene.env_origins
                ).cpu().tolist()
                result['initial_table_top_z_m'] = (initial_state.table_top_z-env.scene.env_origins[:, 2]).cpu().tolist()
                env.eval_initial_q = initial_state.joint_pos.clone()
                env.eval_initial_palm = initial_state.palm_pos_w.clone()
                tracking_sq_sum = torch.zeros(n, device=device)
                tracking_count = torch.zeros(n, device=device)
                peaks, events, samples = {}, {}, []
                result['startup_seconds'] = time.perf_counter()-started
                rollout_started = time.perf_counter()
                for step in range(args.steps):
                    with torch.no_grad():
                        action = player.get_action(obs['policy'], deterministic=True)
                        if not bool(torch.isfinite(action).all()):
                            raise RuntimeError(f'Nonfinite policy action at step {step}')
                        env.eval_actions = action.detach()
                        obs, reward, terminated, truncated, extras = env.step(action)
                    if not env.eval_finite or not all(bool(torch.isfinite(value).all()) for value in obs.values()):
                        raise RuntimeError(f'Nonfinite state/observation at step {step}')
                    if not bool(torch.isfinite(reward).all()) or env.task._nonfinite_count or env.task._insane_count:
                        raise RuntimeError(f'Invalid reward/state counter at step {step}')
                    active = ~done_once
                    tracking_sq_sum += env.eval_metrics['arm_target_tracking_rmse_rad'].square()*active
                    tracking_count += active
                    for key, values in env.eval_metrics.items():
                        if values.dtype == torch.bool:
                            events.setdefault(key, torch.zeros(n, dtype=torch.bool, device=device))
                            events[key] |= values & active
                        else:
                            if not bool(torch.isfinite(values).all()):
                                raise RuntimeError(f'Nonfinite diagnostic {key} at step {step}')
                            peaks.setdefault(key, torch.zeros(n, device=device))
                            peaks[key] = torch.where(active, torch.maximum(peaks[key], values), peaks[key])
                    if step % 20 == 0:
                        samples.append(dict(step=step+1, active=active.cpu().tolist(),
                                            fold_error_m=env.eval_fold_error.cpu().tolist(),
                                            **{key:value.cpu().tolist() for key,value in env.eval_metrics.items()}))
                    best_error = torch.where(active, torch.minimum(best_error, env.eval_fold_error), best_error)
                    best_footprint = torch.where(active, torch.minimum(best_footprint, env.eval_footprint), best_footprint)
                    ended = terminated.bool() | truncated.bool()
                    first = ended & active
                    if bool(first.any()):
                        final = extras['episode_final']
                        folded[first] = final['done_fold'][first].bool()
                        lengths[first] = step+1
                        final_error[first] = env.eval_fold_error[first]
                        final_footprint[first] = env.eval_footprint[first]
                        for key, values in final.items():
                            if key.startswith('done_'):
                                reasons.setdefault(key, torch.zeros(n, dtype=torch.bool, device=device))[first] = values[first].bool()
                        done_once |= first
                    if bool(ended.any()):
                        player.reset_rnn(ended.nonzero(as_tuple=True)[0])
                    if (step+1) % 25 == 0 or bool(done_once.all()):
                        torch.cuda.synchronize()
                        raise_if_overflowed(captured(), n, f'{args.material}, step {step+1}')
                    if (step+1) % 100 == 0:
                        print(f'[vbd-eval] {args.material} step={step+1} completed={int(done_once.sum())}/{n} physical_folds={int(folded.sum())}', flush=True)
                    if bool(done_once.all()):
                        break
                torch.cuda.synchronize()
                result.update(status='complete', rollout_seconds=time.perf_counter()-rollout_started,
                              steps_run=step+1, completed_episodes=int(done_once.sum()), censored=int((~done_once).sum()),
                              physical_folds=int(folded.sum()), physical_fold_fraction=float(folded.float().mean()),
                              finite_checks_passed=True, overflow_checks_passed=True,
                              per_env_folded=folded.cpu().tolist(), per_env_steps=lengths.cpu().tolist(),
                              per_env_best_fold_error_m=best_error.cpu().tolist(),
                              per_env_best_footprint=best_footprint.cpu().tolist(),
                              per_env_final_fold_error_m=final_error.cpu().tolist(),
                              per_env_final_footprint=final_footprint.cpu().tolist(),
                              termination_reasons={key:int(value.sum()) for key,value in reasons.items()},
                              resolved_cloth=cfg.deformable_cloth.to_dict(), resolved_newton=cfg.newton.to_dict(),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                              plasticity_active=env.backend._plasticity is not None,
                              per_env_peak_metrics={key:value.cpu().tolist() for key,value in peaks.items()},
                              per_env_events={key:value.cpu().tolist() for key,value in events.items()},
                              diagnostic_samples=samples,
                              per_env_arm_target_tracking_rmse_rad=(tracking_sq_sum/tracking_count.clamp_min(1)).sqrt().cpu().tolist())
                classes = []
                for index in range(n):
                    if folded[index]:
                        label = 'physical_fold'
                    elif events['centroid_outside_table_xy'][index] or events['any_particle_near_ground'][index]:
                        label = 'off_table_or_near_ground_without_fold'
                    elif peaks['centroid_height_above_table_m'][index] > .08:
                        label = 'height_excursion_without_fold'
                    else:
                        label = 'no_large_height_excursion_without_fold'
                    classes.append(label)
                result['diagnostic_classes'] = classes
                result['diagnostic_class_definition'] = (
                    'Descriptive only, not causal: height excursion means centroid >8cm above table; '
                    'near ground means any particle <10cm above environment origin. No class asserts grasp/contact.')
            finally:
                output = captured()
            raise_if_overflowed(output, args.num_envs, f'{args.material}, complete evaluation')
        if result['censored']:
            raise RuntimeError(f"{result['censored']} first episodes did not finish within the requested horizon")
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            result['total_seconds'] = time.perf_counter()-started
            write_json(destination, result)
    print(f'[vbd-eval] RESULT {args.material}: {result["physical_folds"]}/{args.num_envs} physical folds; release not evaluated',flush=True)


def main():
    args = parser().parse_args()
    if args.skip_cache_check and not args.check_config:
        raise SystemExit('--skip-cache-check is only valid with --check-config')
    if args.num_envs < 1 or args.steps < 1 or args.resolution < 5 or args.size <= 0:
        raise SystemExit('Require positive envs/steps/size and resolution>=5 for a nondegenerate stationary fit')
    os.chdir(ROOT)
    if args.check_config:
        for material in (args.material,) if args.material else DEFAULT_MATERIALS:
            cfg = build_config(args, material)
            if not args.skip_cache_check:
                check_cache(cfg)
            print(f'{material}: {cfg.deformable_cloth.to_dict()}')
        print('All requested configs verified; no GPU simulation started. Cache checked:', not args.skip_cache_check)
    elif args.material:
        evaluate(args)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        failed = []
        for material in DEFAULT_MATERIALS:
            with (args.out_dir/f'{material}.log').open('w') as log:
                proc = subprocess.run([*command, '--material', material], stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                failed.append(material)
            print(f'[vbd-sweep] {material}: exit={proc.returncode}',flush=True)
        summary = {material:json.loads((args.out_dir/f'{material}.json').read_text())
                   for material in DEFAULT_MATERIALS if (args.out_dir/f'{material}.json').is_file()}
        write_json(args.out_dir/'summary.json', summary)
        if failed:
            raise SystemExit(f'Material evaluations failed: {failed}; inspect per-material logs/results')


if __name__ == '__main__':
    main()
