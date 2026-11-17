"""Evaluate the fold checkpoint on an explicitly selected cloth recipe.

Without --preset, retain the current recipe and planar/Play observation controls.
--preset yibo/yibo_our_dr uses the corresponding recipe's robot, physics and actor.
The first episode of each env is the sampling unit; metrics are captured before
DirectRLEnv autoresets, including fold entry independently from full release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CASES = {
    'hybrid': {},
    'sensors_off': {'domain_randomization': {'use_obs_delay': False, 'use_action_delay': False,
        'use_object_state_delay_noise': False, 'joint_velocity_obs_noise_std': 0.,
        'joint_pos_obs_noise_std': 0., 'action_noise_std': 0.}},
    'delays_off': {'domain_randomization': {'use_obs_delay': False, 'use_action_delay': False,
        'object_state_delay_max': 1}},
    'noise_off': {'domain_randomization': {'object_state_xyz_noise_std': 0.,
        'object_state_rotation_noise_degrees': 0., 'joint_velocity_obs_noise_std': 0.,
        'joint_pos_obs_noise_std': 0., 'action_noise_std': 0.}},
    'thickness_16mm': {'obs': {'yibo_contact_thickness': .016}},
    'density_2': {'rigid_cloth': {'randomization': {'density': [2., 2.]}}},
    'hinges_yibo': {'rigid_cloth': {'hinge_stiffness_override': 0.,
        'hinge_damping_override': .05, 'hinge_armature': .001}},
    'damping_yibo': {'rigid_cloth': {'hinge_damping_override': .05}},
    'armature_yibo': {'rigid_cloth': {'hinge_armature': .001}},
    'sizes_small': {'rigid_cloth': {'length_range': [.1, .2], 'width_range': [.1, .2]}},
    'contacts_yibo': {'rigid_cloth': {'rest_offset': .007, 'contact_offset': .012,
        'randomization': {'profile': 'yibo'}}},
    'contacts_thickness_16mm': {'obs': {'yibo_contact_thickness': .016},
        'rigid_cloth': {'rest_offset': .007, 'contact_offset': .012,
            'randomization': {'profile': 'yibo'}}},
    'historical_physics': {'rigid_cloth': {'rest_offset': .007, 'contact_offset': .012,
        'hinge_stiffness_override': 0., 'hinge_damping_override': .05, 'hinge_armature': .001,
        'randomization': {'profile': 'yibo', 'density': [2., 2.]}}},
    'yibo_with_our_sensors': {'domain_randomization': {'use_obs_delay': True,
        'use_action_delay': True, 'use_object_state_delay_noise': True,
        'object_state_xyz_noise_std': .01, 'object_state_rotation_noise_degrees': 5.,
        'joint_velocity_obs_noise_std': .1}},
    'yibo_with_thickness_2mm': {'obs': {'yibo_contact_thickness': .002}},
    'yibo': {},
}


def case_overlay(name):
    """One restored factor on the hybrid; yibo is an explicitly confounded reference."""
    import copy
    return copy.deepcopy(CASES[name])


def case_preset(name):
    return 'yibo' if name == 'yibo' or name.startswith('yibo_with_') else 'yibo_our_dr'


def merge_config(base, overlay):
    return {**base, **{key: merge_config(base[key], value)
                      if isinstance(value, dict) and isinstance(base.get(key), dict) else value
                      for key, value in overlay.items()}}


def install_contact_only_profile(seed):
    """Evaluation-local composition: modern mass/gain DR plus historical contacts.

    The yibo flag authors min/average materials at scene creation. Modern DR is
    run under its normal flag, then only the old contact coefficients/rest shell
    are restored. Extra contact draws use an independent RNG stream. No production
    randomizer is modified and every case runs in its own Python process.
    """
    import torch
    from isaacsimenvs.tasks.cloth.backends.rigid import randomization, yibo_randomization
    original_init, original_randomize = randomization.rigid_init, randomization.rigid_randomize
    counter = [0]

    def isolated_contact_draw(function, env, *args):
        devices = [torch.device(env.device).index or 0] if str(env.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 900000 + counter[0])
            counter[0] += 1
            function(env, *args)

    def initialize(env):
        rd = env.cfg.rigid_cloth.randomization
        rd.profile = 'material'
        try:
            original_init(env)
        finally:
            rd.profile = 'yibo'
        isolated_contact_draw(yibo_randomization.initialize, env)

    def randomize(env, ids):
        rd = env.cfg.rigid_cloth.randomization
        rd.profile = 'material'
        try:
            original_randomize(env, ids)
        finally:
            rd.profile = 'yibo'
        if not getattr(env, '_yibo_initialized', False):
            return
        cpu = ids.cpu()
        for name, asset, buffer in (('robot', env.robot, env._robot_mats), ('table', env.table, env._table_mats)):
            mu = getattr(env, f'_yibo_{name}_friction')[cpu]
            buffer[cpu, :, 0] = mu[:, None]
            buffer[cpu, :, 1] = mu[:, None]
            buffer[cpu, :, 2] = 0.
            asset.root_physx_view.set_material_properties(buffer, cpu)
        isolated_contact_draw(yibo_randomization.reset, env, ids)
    randomization.rigid_init, randomization.rigid_randomize = initialize, randomize


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=1200)
    parser.add_argument('--mode', choices=['planar', 'play', 'yibo'], default=None)
    parser.add_argument('--preset', choices=['yibo', 'yibo_our_dr'], default=None)
    parser.add_argument('--case', choices=list(CASES), default='hybrid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expl_coef', type=float, default=0.)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--pinned', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', default=str(ROOT/'pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth'))
    parser.add_argument('--policy_config', default=str(ROOT/'pretrained_policy/flexiv_wuji/config.yaml'))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app

    import torch
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer, read_policy_cfg
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.scripts.evaluation.contact_guard import capture_fd_output, raise_if_overflowed
    from isaacsimenvs.scripts.visualization.rigid_cloth.reset import load_training_env_config

    cfg = ClothCfg.for_backend('rigid')
    preset = args.preset or case_preset(args.case)
    recipe = ROOT / f'isaacsimenvs/scripts/configs/runs/{preset}.yaml'
    cfg.from_dict(merge_config(load_training_env_config(recipe), case_overlay(args.case)))
    if args.case in ('contacts_yibo', 'contacts_thickness_16mm', 'historical_physics'):
        install_contact_only_profile(args.seed)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    source = read_policy_cfg(args.policy_config, args.device)
    cfg.action.from_dict(source['env']['action'])
    cfg.obs.keypoint_mode = args.mode or cfg.obs.keypoint_mode
    mode = cfg.obs.keypoint_mode
    if args.pinned:
        disable_randomization(cfg)
    # A resolved JSON is written before construction, so startup failures retain
    # the exact requested configuration as well.
    args.out.with_suffix('.config.json').write_text(json.dumps(cfg.to_dict(), default=str, indent=2))
    env = None
    start = time.monotonic()
    try:
        with capture_fd_output() as captured:
            env = ClothEnv(cfg)
            player = PretrainedPlayer(args.policy_config, args.checkpoint, num_envs=args.num_envs,
                                      device=args.device, num_observations=134,
                                      num_actions=27, expl_coef=args.expl_coef)
            n, dev = env.num_envs, env.device
            completed = torch.zeros(n, dtype=torch.bool, device=dev)
            ever_folded = completed.clone()
            ever_released = completed.clone()
            best_error = torch.full((n,), float('inf'), device=dev)
            best_footprint = best_error.clone()
            lengths = torch.zeros(n, dtype=torch.long, device=dev)
            successes = lengths.clone()
            first_fold_step = torch.full_like(lengths, -1)
            terminal_error, terminal_footprint = best_error.clone(), best_error.clone()
            max_speed, max_joint_speed, max_height, max_palm_motion, max_xy_drift = [torch.zeros(n, device=dev) for _ in range(5)]
            min_height = torch.full((n,), float('inf'), device=dev)
            min_clearance = min_height.clone()
            trajectory = []
            initial = {}
            reasons = {}
            original_evaluate = env.task.evaluate

            def evaluate(state, episode_length, max_episode_length, touching=None, *, robot_clearance=None):
                result = original_evaluate(state, episode_length, max_episode_length, touching,
                                           robot_clearance=robot_clearance)
                view = env.task.derive(state)
                live = ~completed
                best_error[live] = torch.minimum(best_error[live], view.fold_error[live])
                best_footprint[live] = torch.minimum(best_footprint[live], view.footprint[live])
                ever_folded[live] |= env.task.is_folded(state, view)[live]
                ever_released[live] |= env.task._released[live]
                lengths[live] += 1
                first = live & env.task.is_folded(state, view) & (first_fold_step < 0)
                first_fold_step[first] = lengths[first]
                speed = state.points_vel_w.norm(dim=-1).amax(dim=1)
                heights = state.points_w[..., 2] - state.table_top_z[:, None]
                for target, value in ((max_speed, speed), (max_joint_speed, state.joint_vel.abs().amax(dim=1)),
                        (max_height, heights.amax(dim=1)),
                        (max_palm_motion, (state.palm_pos_w - initial['palm']).norm(dim=-1)),
                        (max_xy_drift, (view.pos_w[:, :2] - initial['center'][:, :2]).norm(dim=-1))):
                    target[live] = torch.maximum(target[live], value[live])
                min_height[live] = torch.minimum(min_height[live], heights.amin(dim=1)[live])
                if robot_clearance is not None:
                    min_clearance[live] = torch.minimum(min_clearance[live], robot_clearance[live])
                ended = live & (result.terminated | result.truncated)
                successes[ended] = env.task._successes[ended]
                terminal_error[ended], terminal_footprint[ended] = view.fold_error[ended], view.footprint[ended]
                for name, value in env.task._termination_reasons.items():
                    if name not in reasons:
                        reasons[name] = torch.zeros(n, dtype=torch.bool, device=dev)
                    reasons[name][ended] = value[ended]
                completed.logical_or_(ended)
                return result

            env.task.evaluate = evaluate
            # The policy loader seeds from its saved config. Explicitly reseed
            # AFTER model construction so --seed controls the evaluated resets.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            obs, _ = env.reset()
            player.reset_rnn()
            initial_state, initial_view = env._now()
            initial = {'palm': initial_state.palm_pos_w.clone(), 'center': initial_view.pos_w.clone()}
            initial_record = {'joint_pos': initial_state.joint_pos.tolist(),
                              'joint_vel': initial_state.joint_vel.tolist(),
                              'points': (initial_state.points_w - env.scene.env_origins[:, None]).tolist(),
                              'dimensions': env.backend._dims.tolist(),
                              'density': env.backend._dr_density.tolist()}
            # Actual PhysX tensor readbacks after reset: resolve sparse cloth rows
            # through each group's global-ID mapping, rather than assuming env order.
            readback = {'robot_preset': cfg.robot_preset,
                        'material_profile': cfg.rigid_cloth.randomization.profile,
                        'cloth': []}
            for group in env.backend.groups:
                view = group.articulation.root_physx_view
                count = min(2, len(group.env_ids))
                entry = {'design': group.design.name, 'env_ids': group.env_ids[:count].tolist()}
                for name, values in (('stiffness', view.get_dof_stiffnesses()),
                                     ('damping', view.get_dof_dampings()),
                                     ('armature', view.get_dof_armatures()),
                                     ('rest_offsets', view.get_rest_offsets()),
                                     ('material', view.get_material_properties())):
                    entry[name] = values[:count].cpu().tolist()
                entry['mass_kg'] = view.get_masses()[:count].sum(dim=1).cpu().tolist()
                readback['cloth'].append(entry)
            for name, asset in (('robot', env.robot), ('table', env.table)):
                readback[name] = asset.root_physx_view.get_material_properties()[:2].cpu().tolist()
            args.out.with_suffix('.readback.json').write_text(json.dumps(readback, indent=2))
            print(f'[readback] preset={cfg.robot_preset} material={readback["material_profile"]} '
                  f'first_cloth={readback["cloth"][0]}', flush=True)
            step = 0
            with torch.inference_mode():
                while step < args.max_steps and not bool(completed.all()):
                    action = player.get_action(obs['policy'], deterministic=not args.stochastic)
                    obs, reward, terminated, truncated, extras = env.step(action)
                    ended = terminated | truncated
                    if bool(ended.any()):
                        player.reset_rnn(ended.nonzero(as_tuple=True)[0])
                    step += 1
                    if step % 25 == 0:
                        trajectory.append({'step': step, 'best_error': best_error.tolist(),
                                           'folded': ever_folded.tolist(), 'released': ever_released.tolist()})
                    if step % 100 == 0:
                        print(f'[eval] {args.case} step={step} completed={int(completed.sum())}/{n} '
                              f'folds={int(ever_folded.sum())} releases={int(ever_released.sum())} '
                              f'best_error={float(best_error.min()):.6f}', flush=True)
                        raise_if_overflowed(captured(), n, f'rollout step {step}')
            raise_if_overflowed(captured(), n, 'final rollout')
            captured()

        def summary(mask):
            count = int(mask.sum())
            return {'n': count, 'completed': int((completed & mask).sum()),
                    'fold_entries': int((ever_folded & mask).sum()),
                    'full_releases': int((ever_released & mask).sum()),
                    'successes': int(successes[mask].sum()),
                    'min_fold_error_m': float(best_error[mask].min()) if count else None,
                    'median_best_fold_error_m': float(best_error[mask].median()) if count else None,
                    'mean_length': float(lengths[mask].float().mean()) if count else None,
                    'termination_reasons': {key: int((value & mask).sum()) for key, value in reasons.items()}}

        result = {'checkpoint': args.checkpoint, 'policy_config': args.policy_config,
                  'case': args.case, 'case_overlay': case_overlay(args.case),
                  'preset': preset, 'observation_mode': mode, 'expl_coef': args.expl_coef,
                  'stochastic': args.stochastic, 'pinned': args.pinned, 'seed': args.seed,
                  'steps_run': step, 'elapsed_seconds': time.monotonic()-start,
                  'protocol': 'first episode per env; pre-autoreset physical fold/release metrics; robot/actor/materials from resolved recipe',
                  'fold_tolerance_m': cfg.task.keypoint_tolerance,
                  'max_folded_footprint': cfg.task.max_folded_footprint,
                  'startup_physics_readback': readback,
                  'initial': initial_record,
                  'trajectory': trajectory,
                  'all': summary(torch.ones(n, dtype=torch.bool, device=dev)),
                  'by_design': {group.design.name: summary(torch.isin(torch.arange(n, device=dev), group.env_ids))
                                for group in env.backend.groups},
                  'per_env': {'completed': completed.tolist(), 'folded': ever_folded.tolist(),
                              'released': ever_released.tolist(), 'best_error': best_error.tolist(),
                              'best_footprint': best_footprint.tolist(), 'length': lengths.tolist(),
                              'first_fold_step': first_fold_step.tolist(),
                              'terminal_error': terminal_error.tolist(), 'terminal_footprint': terminal_footprint.tolist(),
                              'max_cloth_speed_mps': max_speed.tolist(), 'max_joint_speed_radps': max_joint_speed.tolist(),
                              'max_cloth_height_above_table_m': max_height.tolist(),
                              'min_cloth_height_above_table_m': min_height.tolist(),
                              'max_palm_motion_m': max_palm_motion.tolist(), 'max_cloth_xy_drift_m': max_xy_drift.tolist(),
                              'min_robot_clearance_m': min_clearance.tolist()}}
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result[key] for key in ['observation_mode', 'all', 'by_design']}, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
