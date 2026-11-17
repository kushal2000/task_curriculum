"""Replay identical native VBD stress probes while changing numerical solver settings.

First create one shared fixture with --prepare-fixture --fixture PATH.npz. Each
candidate then loads that immutable initial state and action tape in a fresh
process. No policy inference, reward calculation or automatic reset occurs while
stepping. This is a numerical benchmark, not a folding-success evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROBES = ('settle', 'scripted', 'fold_release')
NUMERICAL_FIELDS = ('vbd_iterations', 'substeps', 'coupler_iterations')
VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('vbd_stability.yaml'))
    parser.add_argument('--candidate', default='baseline')
    parser.add_argument('--prepare-fixture', action='store_true')
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).with_name('results')/'stability')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--material', default='base')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    matrix = yaml.safe_load(args.config.read_text())
    args.steps = args.steps if args.steps is not None else int(matrix.get('steps', 300))
    args.num_envs = args.num_envs if args.num_envs is not None else int(matrix.get('num_envs', 12))
    if args.steps < 1 or args.num_envs < 3 or args.num_envs % 3:
        parser.error('Require steps>=1 and num-envs>=3 divisible by3')
    if args.candidate == 'baseline' or args.prepare_fixture:
        args.candidate = matrix.get('reference_candidate', 'vbd12')
    if args.candidate not in matrix['candidates']:
        parser.error(f'Unknown candidate {args.candidate!r}')
    selected = matrix['candidates'][args.candidate]
    if set(selected) != set(NUMERICAL_FIELDS):
        parser.error(f'Each candidate must contain only {NUMERICAL_FIELDS}')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in selected.values()):
        parser.error('Candidate numerical settings must be positive integers')
    return args, matrix


def canonical(value):
    """Stable JSON form; callable reprs otherwise contain process-specific addresses."""
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        return f'{value.__module__}.{value.__qualname__}'
    return str(value)


def json_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def configuration(args, matrix):
    from experiments.nov16_yibo_our_dr.evaluate_vbd import parser, build_config, check_cache

    recipe = Path(matrix['base_recipe'])
    if not recipe.is_absolute():
        recipe = args.config.resolve().parent/recipe
    source = parser().parse_args(['--material', args.material, '--num-envs', str(args.num_envs),
                                 '--steps', str(args.steps), '--seed', str(args.seed), '--device', args.device,
                                 '--experiment', str(recipe)])
    cfg = build_config(source, args.material)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    cfg.reset.reset_dof_pos_random_interval_arm = 0.
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.
    cfg.reset.reset_dof_vel_random_interval = 0.
    dr = cfg.domain_randomization
    dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = dr.joint_pos_obs_noise_std = dr.action_noise_std = 0.
    dr.object_state_xyz_noise_std = dr.object_state_rotation_noise_degrees = 0.
    for key, value in matrix['candidates'][args.candidate].items():
        setattr(cfg.deformable_cloth, key, value)
    protocol = canonical(cfg.to_dict())
    for key in NUMERICAL_FIELDS:
        protocol['deformable_cloth'].pop(key)
    protocol.pop('seed', None)
    protocol.pop('episode_length_s', None)  # No task termination is evaluated; horizon hashed separately.
    protocol['sim'].pop('device', None)
    check_cache(cfg)
    return cfg, protocol



def native_solver_settings(expected):
    """Assert the installed solver actually received the requested numerical settings."""
    from isaaclab_newton.physics import NewtonManager

    solver = NewtonManager._solver
    actual = dict(vbd_iterations=int(solver.solver('cloth').iterations),
                  substeps=int(solver._entries['cloth'].substeps),
                  coupler_iterations=int(solver._coupling.iterations))
    if actual != expected:
        raise RuntimeError(f'Native solver settings differ: requested={expected}, actual={actual}')
    actual['manager_decimation'] = int(NewtonManager._decimation)
    if actual['manager_decimation'] != 1:
        raise RuntimeError('Manual boundary sampling requires Newton manager decimation=1')
    actual['newton_substeps'] = int(NewtonManager._num_substeps)
    actual['cloth_substep_dt_s'] = float(NewtonManager._solver_dt)/actual['substeps']
    return actual


def snapshot(env):
    from isaacsimenvs.tasks.cloth.backends.base import as_torch

    def copy(value):
        return as_torch(value).detach().cpu().numpy().copy()
    state = env.backend.read_state()
    return dict(joint_q=copy(state.joint_pos), joint_dq=copy(state.joint_vel),
                table_pose=np.concatenate((copy(env.backend.table.data.root_pos_w),
                                           copy(env.backend.table.data.root_quat_w)), axis=1),
                table_z=copy(env.backend._table_z), cloth_points=copy(state.points_w),
                cloth_velocity=copy(state.points_vel_w), controller_previous=copy(env.controller._prev_targets),
                controller_targets=copy(env.controller._cur_targets), origins=copy(env.scene.env_origins),
                joint_names=np.asarray(list(env.backend.robot.data.joint_names)))


def make_fixture(env, args, protocol):
    import torch
    from experiments.nov16_yibo_our_dr.evaluate_vbd import write_json

    if args.fixture.exists() or args.fixture.with_suffix('.json').exists():
        raise FileExistsError(f'Fixture already exists; use a new path to preserve active comparisons: {args.fixture}')
    env.reset(seed=args.seed)
    group_size = args.num_envs//3
    ids = torch.arange(args.num_envs, device=env.device)
    variant = ids % group_size
    yaw = variant.float()*(2*torch.pi/group_size)
    choices = torch.tensor([[0.,0.],[.025,-.02],[-.025,.02],[.02,.02]], device=env.device)
    xy = choices[variant % len(choices)]
    group = ids//group_size
    # A recorded, independently audited physical pose places all five pads close
    # to the tabletop. The previous home pose never contacted the falling sheet.
    checkpoint = ROOT/'pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth'
    source_row = 18447
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)[0]['obs']['obs'][source_row,:134].to(env.device)
    perm = env.controller._perm_lab_to_canon
    lower,upper = env.controller._joint_lower_canon,env.controller._joint_upper_canon
    q = env.backend.read_joint_pos().clone()
    qcanon = (saved[:27]+1)*.5*(upper-lower)+lower
    selected = (group == 1).nonzero().flatten()
    q[selected[:,None],perm[None,:]] = qcanon
    env.backend.robot.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q))
    env.controller._prev_targets.copy_(q)
    env.controller._cur_targets.copy_(q)
    # Saved Play palm coordinates use the historical base position (0,.8,0).
    from isaacsimenvs.tasks.cloth.backends.robot import PLAY_ROBOT_ROOT_POS
    shift = torch.tensor(env.cfg.robot_root_pos,device=env.device)-torch.tensor(PLAY_ROBOT_ROOT_POS,device=env.device)
    pads = saved[81:84]+shift+saved[92:107].reshape(5,3)
    xy[selected] = pads[:,:2].mean(0)-torch.tensor(env.cfg.reset.table_reset_xy,device=env.device)
    source = dict(checkpoint=str(checkpoint.relative_to(ROOT)),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  observation_row=source_row,saved_fingertips_env_m=pads.cpu().tolist(),
                  scripted_worlds=selected.cpu().tolist())
    angle = torch.where(group == 2, torch.pi, 0.)
    env.backend._write_cloth(ids, xy, yaw, angle)
    env.backend._randomize(ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    data = snapshot(env)
    data['probe_group'] = group.cpu().numpy()
    q = env.backend.read_joint_pos()
    perm = env.controller._perm_lab_to_canon
    lower, upper = env.controller._joint_lower_canon, env.controller._joint_upper_canon
    hold = (2*(q[:,perm]-lower)/(upper-lower)-1).clamp(-1,1)
    hold[:,:7] = 0.
    tape = hold.unsqueeze(0).expand(args.steps,-1,-1).clone()
    # The same smooth canonical action sequence is replayed for every solver.
    # One second of settling, then gentle arm motion and hand close/open cycles.
    t = (torch.arange(args.steps, device=env.device)*env.step_dt-1.).clamp_min(0)
    wave = torch.sin(2*torch.pi*.4*t)
    active = t > 0
    tape[:,group == 1,1] = (.18*wave*active)[:,None]
    tape[:,group == 1,3] = (-.12*wave*active)[:,None]
    tape[:,group == 1,5] = (.08*wave*active)[:,None]
    close = (.5*(1-torch.cos(2*torch.pi*.4*t)))[:,None,None]
    tape[:,group == 1,7:] = (1-close)*hold[None,group == 1,7:] + close*.5
    data['actions'] = tape.cpu().numpy()
    initial = {key:value for key,value in data.items() if key != 'actions'}
    meta = dict(version=VERSION, seed=args.seed, material=args.material, num_envs=args.num_envs,
                steps=args.steps, candidate_id=args.candidate, probes=list(PROBES),
                initial_state_sha256=array_hash(initial), action_trace_sha256=array_hash({'actions':data['actions']}),
                physical_config_sha256=json_hash(protocol), physical_config=protocol,
                scripted_pose_source=source,
                action_description='Hold pose; scripted group starts gentle arm oscillation and smooth hand close/open after1s.')
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.fixture.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(args.fixture)
    write_json(args.fixture.with_suffix('.json'), meta)
    print(f'[stability] fixture ready: {args.fixture}', flush=True)


def restore_fixture(env, args, protocol):
    import torch

    meta = json.loads(args.fixture.with_suffix('.json').read_text())
    with np.load(args.fixture, allow_pickle=False) as archive:
        data = {name:archive[name].copy() for name in archive.files}
    if meta['version'] != VERSION or meta['num_envs'] != args.num_envs or meta['material'] != args.material:
        raise ValueError('Fixture version, environment count or material differs from requested probe')
    if meta['seed'] != args.seed or args.steps > len(data['actions']):
        raise ValueError('Fixture seed differs or requested horizon exceeds the shared tape')
    if meta['physical_config_sha256'] != json_hash(protocol):
        raise ValueError('Physical configuration differs from the fixture; only numerical candidate settings may vary')
    if array_hash({k:v for k,v in data.items() if k != 'actions'}) != meta['initial_state_sha256']:
        raise ValueError('Initial-state fixture checksum failed')
    if array_hash({'actions':data['actions']}) != meta['action_trace_sha256']:
        raise ValueError('Action-tape fixture checksum failed')
    env.reset(seed=args.seed)
    if list(env.backend.robot.data.joint_names) != data['joint_names'].tolist():
        raise ValueError('Fixture joint names/order differ from the live articulation')
    if not np.array_equal(env.scene.env_origins.cpu().numpy(),data['origins']):
        raise ValueError('Fixture environment origins differ from this scene')
    tensor = lambda key: torch.as_tensor(data[key], device=env.device)
    ids = torch.arange(args.num_envs, device=env.device)
    env.backend.robot.write_joint_state_to_sim_index(position=tensor('joint_q'), velocity=tensor('joint_dq'))
    env.backend.table.write_root_pose_to_sim(tensor('table_pose'))
    env.backend._table_z.copy_(tensor('table_z'))
    env.backend._cloth.write_nodal_pos_to_sim_index(tensor('cloth_points'), ids)
    env.backend._cloth.write_nodal_velocity_to_sim_index(tensor('cloth_velocity'), ids)
    env.backend._randomize(ids)
    env.controller._prev_targets.copy_(tensor('controller_previous'))
    env.controller._cur_targets.copy_(tensor('controller_targets'))
    env.scene.write_data_to_sim()
    env.sim.forward()
    restored = snapshot(env)
    errors = {}
    for key in restored:
        if key == 'joint_names':
            continue
        errors[key] = float(np.max(np.abs(restored[key]-data[key])))
    if max(errors.values()) > 2e-6:
        raise RuntimeError(f'Initial state did not restore accurately: {errors}')
    return data, meta, errors


class Diagnostics:
    def __init__(self, env, fixture):
        import torch
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
        from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_triangles
        from experiments.nov16_yibo_our_dr.vbd_contacts import VbdContactDiagnostics

        self.env = env
        self.contacts = VbdContactDiagnostics(env)
        grid = env.backend.cloth_grid
        triangles = torch.tensor(grid_triangles(grid.resolution),device=env.device).reshape(-1,3)
        edges = torch.cat((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
        self.edges = torch.unique(edges.sort(dim=1).values,dim=0)
        self.rest_lengths = (grid.rest[:,self.edges[:,1]]-grid.rest[:,self.edges[:,0]]).norm(dim=-1)
        self.mass = wp.to_torch(NewtonManager.get_model().particle_mass)[self.contacts.particle_rows]
        self.initial_q = torch.as_tensor(fixture['joint_q'],device=env.device)
        initial = env.backend.read_state()
        self.initial_palm = initial.palm_pos_w.clone()
        self.previous_velocity = initial.points_vel_w.clone()
        self.peaks = {}
        self.samples = 0

    def read(self, *, late=False):
        import torch

        env = self.env
        state = env.backend.read_state()
        for name in ('joint_pos','joint_vel','palm_pos_w','points_w','points_vel_w'):
            if not bool(torch.isfinite(getattr(state,name)).all()):
                raise RuntimeError(f'Nonfinite raw physics state: {name}')
        points,velocity = state.points_w,state.points_vel_w
        stretch = (points[:,self.edges[:,1]]-points[:,self.edges[:,0]]).norm(dim=-1)/self.rest_lengths
        arm = env.controller._arm_ids
        metrics = self.contacts.read()
        metrics.update(
            particle_speed_m_s=velocity.norm(dim=-1).amax(1),
            particle_acceleration_m_s2=((velocity-self.previous_velocity)/env.cfg.sim.dt).norm(dim=-1).amax(1),
            cloth_kinetic_energy_j=(.5*self.mass*velocity.square().sum(-1)).sum(1),
            edge_stretch_ratio=stretch.amax(1),p95_edge_stretch_ratio=torch.quantile(stretch,.95,dim=1),
            edge_compression_fraction=(1-stretch).clamp_min(0).amax(1),
            moving_height_above_table_m=(points[:,env.backend.cloth_grid.moving_idx,2]-state.table_top_z[:,None]).amax(1),
            centroid_height_above_table_m=points[...,2].mean(1)-state.table_top_z,
            joint_displacement_rad=(state.joint_pos-self.initial_q).abs().amax(1),
            palm_displacement_m=(state.palm_pos_w-self.initial_palm).norm(dim=-1),
            arm_target_tracking_rmse_rad=(state.joint_pos[:,arm]-env.controller._cur_targets[:,arm]).square().mean(1).sqrt(),
            near_ground_node_count=((points-env.scene.env_origins[:,None])[...,2]<.1).float().sum(1))
        metrics['late_particle_speed_m_s'] = (metrics['particle_speed_m_s'] if late
                                              else torch.zeros_like(metrics['particle_speed_m_s']))
        for name,value in metrics.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f'Nonfinite diagnostic: {name}')
            self.peaks[name] = value.clone() if name not in self.peaks else torch.maximum(self.peaks[name],value)
        self.previous_velocity.copy_(velocity)
        self.samples += 1
        return metrics


def run(args, matrix):
    import torch
    from experiments.nov16_yibo_our_dr.evaluate_vbd import write_json
    from isaacsimenvs.scripts.evaluation.contact_guard import capture_fd_output, raise_if_overflowed

    cfg, protocol = configuration(args,matrix)
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
    started = time.perf_counter()
    env = diagnostics = data = meta = native_settings = None
    status,error = 'failed',None
    finite_ok = overflow_ok = coverage_ok = None

    def check_overflow(output, context):
        nonlocal overflow_ok
        try:
            raise_if_overflowed(output,args.num_envs,context)
        except RuntimeError:
            overflow_ok = False
            raise
        overflow_ok = True
    steps_run = physics_samples = 0
    physics_seconds = diagnostics_seconds = rollout_seconds = simulated_seconds = 0.
    startup_seconds = None
    records = []
    batch_id = f'{args.candidate}:{os.environ.get("SLURM_JOB_ID","local")}:{time.time_ns()}'
    try:
        with capture_fd_output() as captured:
            try:
                env = ClothEnv(cfg)
                # DirectRLEnv configures Newton to advance a whole control interval.
                # Here the outer loop owns decimation so each sim.step is ONE physics tick.
                env.sim.physics_manager.set_decimation(1)
                native_settings = native_solver_settings(matrix['candidates'][args.candidate])
                if args.prepare_fixture:
                    make_fixture(env,args,protocol)
                    check_overflow(captured(),'fixture initialization')
                    return
                data,meta,restore_errors = restore_fixture(env,args,protocol)
                diagnostics = Diagnostics(env,data)
                actions = torch.as_tensor(data['actions'][:args.steps],device=env.device)
                if not bool(torch.isfinite(actions).all()):
                    raise ValueError('Action tape contains nonfinite values')
                startup_seconds = time.perf_counter()-started
                rollout_start = time.perf_counter()
                for step in range(args.steps):
                    with torch.no_grad():
                        env.controller.process(actions[step],torch.full((args.num_envs,),step,device=env.device,dtype=torch.long))
                        for _ in range(cfg.decimation):
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                            env.backend.apply_joint_commands(env.controller.commands(env.backend.read_joint_pos()))
                            env.scene.write_data_to_sim()
                            sim_time_before = env.sim.physics_manager.get_simulation_time()
                            env.sim.step(render=False)
                            elapsed = env.sim.physics_manager.get_simulation_time()-sim_time_before
                            if not np.isclose(elapsed, cfg.sim.dt, rtol=1e-6, atol=1e-10):
                                raise RuntimeError(f'One sim.step advanced {elapsed}s; expected {cfg.sim.dt}s')
                            simulated_seconds += elapsed
                            env.scene.update(dt=cfg.sim.dt)
                            torch.cuda.synchronize()
                            physics_seconds += time.perf_counter()-tick
                            tick = time.perf_counter()
                            metrics = diagnostics.read(late=(step+1 >= 2*args.steps//3+1))
                            finite_ok = True
                            torch.cuda.synchronize()
                            diagnostics_seconds += time.perf_counter()-tick
                            physics_samples += 1
                    steps_run = step+1
                    if steps_run%20 == 0 or steps_run == args.steps:
                        records.append(dict(step=steps_run,**{key:value.cpu().tolist() for key,value in metrics.items()}))
                        check_overflow(captured(),f'{args.candidate},step{steps_run}')
                    if steps_run%60 == 0:
                        print(f'[stability] {args.candidate} {steps_run}/{args.steps}',flush=True)
                rollout_seconds = time.perf_counter()-rollout_start
                check_overflow(captured(),'completed stability probe')
                counts = diagnostics.peaks['robot_record_count'][args.num_envs//3:2*args.num_envs//3]
                coverage_ok = bool((counts > 0).all())
                if not coverage_ok:
                    raise RuntimeError(f'Invalid scripted contact probe: peak robot records {counts.tolist()}')
                status = 'complete'
            finally:
                captured()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if 'nonfinite' in str(exc).lower():
            finite_ok = False
        failure_traceback = traceback.format_exc()
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if not args.prepare_fixture:
                group_size = args.num_envs//3
                if data is not None:
                    trace_hash = array_hash({'actions':data['actions'][:args.steps]})
                else:
                    trace_hash = None
                for group,probe in enumerate(PROBES):
                    sl = slice(group*group_size,(group+1)*group_size)
                    result = dict(status=status,error=error,probe=probe,material=args.material,seed=args.seed,
                        candidate_id=args.candidate,candidate=matrix['candidates'][args.candidate],
                        native_solver_settings=native_settings,
                        version=VERSION,fixture=str(args.fixture.resolve()),batch_id=batch_id,
                        initial_state_sha256=meta['initial_state_sha256'] if meta else None,
                        action_trace_sha256=trace_hash,
                        physical_protocol_sha256=json_hash(dict(version=VERSION,config=protocol,horizon_steps=args.steps)),
                        physical_config_sha256=json_hash(protocol),
                        num_envs=group_size,steps_run=steps_run,requested_steps=args.steps,
                        batch_num_envs=args.num_envs,batch_steps_run=steps_run,
                        completed_episodes=group_size if status=='complete' else 0,
                        censored=0 if status=='complete' else group_size,
                        completion_definition='Completed fixed-duration probe trajectories, NOT RL episodes or successful folds.',
                        rollout_seconds=rollout_seconds,batch_rollout_seconds=rollout_seconds,
                        batch_physics_seconds=physics_seconds,batch_diagnostics_seconds=diagnostics_seconds,
                        startup_seconds=startup_seconds,total_seconds=time.perf_counter()-started,
                        physics_boundary_samples=physics_samples,simulated_seconds=simulated_seconds,
                        physics_step_duration_checked=physics_samples>0,late_window_start_step=2*args.steps//3+1,
                        finite_checks_passed=finite_ok,overflow_checks_passed=overflow_ok,
                        scripted_contact_coverage_passed=coverage_ok,
                        quality_metrics=matrix.get('quality_metrics',{}).get(probe,[]),
                        per_env_peak_metrics={key:value[sl].cpu().tolist() for key,value in diagnostics.peaks.items()} if diagnostics else {},
                        diagnostic_samples=[dict(step=row['step'],**{key:value[sl] for key,value in row.items() if key!='step'}) for row in records],
                        resolved_config=canonical(cfg.to_dict()),
                        gpu_name=torch.cuda.get_device_name(args.device),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
                    if meta is not None:
                        result['initial_restore_max_errors'] = restore_errors
                        result['scripted_pose_source'] = meta.get('scripted_pose_source')
                    if error:
                        result['traceback'] = failure_traceback
                    write_json(args.out_dir/f'{probe}.json',result)


if __name__ == '__main__':
    arguments, matrix = parse_args()
    run(arguments,matrix)
