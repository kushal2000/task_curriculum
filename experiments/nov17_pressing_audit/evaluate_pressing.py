"""Record trained cloth rollouts and normal contact loads without changing rewards.

Uses the saved training environment settings, the environment's normal controller,
and one first episode per world. Sample geometry and robot/table contacts at120Hz.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@contextmanager
def capture_audit_output():
    from isaacsimenvs.scripts.evaluation.contact_guard import capture_fd_output

    with capture_fd_output() as read:
        try:
            yield read
        finally:
            read()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--design', help='Optional single topology; otherwise retain all training designs.')
    parser.add_argument('--out-dir', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.run, args.out_dir = args.run.resolve(), args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        choices = [p for p in (args.run/'nn').glob('periodic_*.pth') if time.time()-p.stat().st_mtime > 20]
        args.checkpoint = max(choices, key=lambda p:p.stat().st_mtime)
    app = AppLauncher(args).app

    import torch
    import yaml
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
    from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
    from isaacsimenvs.tasks.cloth.backends.rigid.designs import extrusion_mesh
    from isaacsimenvs.tasks.cloth.mdp.geometry import rotations
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer
    from isaacsimenvs.scripts.evaluation.contact_guard import raise_if_overflowed
    from experiments.nov17_pressing_audit.contacts import enable_contact_reporting, RigidClothContacts

    saved = yaml.safe_load((args.run/'.hydra/config.yaml').read_text())['env']
    cfg = ClothCfg.for_backend('rigid')
    for key in ('robot_preset','robot_root_pos','robot_self_collision','assets','action','obs','reward',
                'reset','termination','domain_randomization','task','rigid_cloth'):
        cfg.from_dict({key:saved[key]})
    cfg.sim.dt, cfg.decimation = saved['sim']['dt'], saved['decimation']
    cfg.from_dict({'sim': {'physx': saved['sim']['physx']}})
    cfg.episode_length_s = saved['episode_length_s']
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    if args.design:
        cfg.rigid_cloth.designs = [args.design]
    cfg.task.check()

    class InstrumentedEnv(ClothEnv):
        def _setup_scene(self):
            super()._setup_scene()
            enable_contact_reporting(self.backend)

    env = contacts = None
    frames, records = [], []
    metadata = dict(run=str(args.run), checkpoint=str(args.checkpoint), seed=args.seed,
                    num_envs=args.num_envs, requested_steps=args.steps, config=cfg.to_dict(),
                    contact_components='normal only; no friction force reconstructed', status='running')
    started = time.perf_counter()
    try:
        with capture_audit_output() as captured:
            env = InstrumentedEnv(cfg)
            contacts = RigidClothContacts(env.backend)
            player = PretrainedPlayer(ROOT/'pretrained_policy/flexiv_wuji/config.yaml',args.checkpoint,
                                      args.num_envs,device=args.device,num_observations=134,num_actions=27,expl_coef=0.)
            env.seed(args.seed)
            obs,_ = env.reset()
            player.reset_rnn()
            n,dev = env.num_envs,env.device
            origins = env.scene.env_origins
            active = torch.ones(n,dtype=torch.bool,device=dev)
            max_plates = max(len(g.body_ids) for g in env.backend.groups)
            max_joints = max(len(g.joint_ids) for g in env.backend.groups)
            geometry = []
            design_ids = ['']*n
            masses = torch.empty(n,device=dev)
            half_by_plate = torch.zeros(n,max_plates,dtype=torch.long,device=dev)
            for group in env.backend.groups:
                masses[group.env_ids] = group.articulation.root_physx_view.get_masses().sum(1).to(dev)
                for e in group.env_ids.tolist():
                    design_ids[e] = group.design.name
                    spec = group.design.resolve(*env.backend._dimensions_cpu[e])
                    plates = []
                    for plate_id,p in enumerate(spec.plates):
                        vertices,faces = extrusion_mesh(p)
                        plates.append(dict(vertices=vertices.tolist(),triangles=faces.tolist()))
                        half_by_plate[e,plate_id] = int(p.center[0] > 0)
                    geometry.append(dict(env_id=e,design=group.design.name,plates=plates))
            metadata.update(designs=design_ids,dimensions=env.backend._dimensions_cpu.tolist(),
                            masses_kg=masses.cpu().tolist(),geometry=geometry,
                            robot_joint_names=list(env.backend.robot.data.joint_names),
                            robot_urdf=cfg.assets.robot_urdf,table_size=urdf_box_size(cfg.assets.table_urdf),
                            physics_dt=cfg.sim.dt,force_capacity=contacts.capacity,
                            force_half_assignment='canonical plate material half')
            (args.out_dir/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))

            recording = False
            last_counter = -1
            old_update = env.scene.update
            def update(dt):
                nonlocal last_counter
                old_update(dt)
                if not recording or dt <= 0 or env._sim_step_counter == last_counter:
                    return
                last_counter = env._sim_step_counter
                state = env.backend.read_state()
                view = env.task.derive(state)
                contact = contacts.sample()
                record = {key:value.detach().cpu().numpy().copy() for key,value in contact.items()
                          if key not in ('robot_force_by_sensor','table_force_by_sensor')}
                record['position'] -= origins[contact['env_id']].cpu().numpy()
                record['frame_idx'] = np.full(len(record['env_id']),len(frames),dtype=np.int32)
                records.append(record)
                # Shared-crease contact coordinates can cross zero under solver error.
                # Each plate belongs to one material half; its identity is unambiguous.
                half = half_by_plate[contact['env_id'],contact['cloth_body']]
                bucket = contact['env_id']*4+contact['partner_type']*2+half
                vector = torch.zeros(n*4,3,device=dev).index_add_(0,bucket,contact['force_vector']).view(n,2,2,3)
                magnitude = torch.zeros(n*4,device=dev).index_add_(0,bucket,contact['force_magnitude']).view(n,2,2)
                downward = torch.zeros(n*4,device=dev).index_add_(0,bucket,(-contact['force_vector'][:,2]).clamp_min(0)).view(n,2,2)
                panel_pose = torch.zeros(n,max_plates,7,device=dev)
                panel_pose[...,3] = 1.
                hinge_q = torch.full((n,max_joints),float('nan'),device=dev)
                for group in env.backend.groups:
                    panel_pose[group.env_ids,:len(group.body_ids),:3] = group.articulation.data.body_link_pos_w[:,group.body_ids]-origins[group.env_ids,None]
                    panel_pose[group.env_ids,:len(group.body_ids),3:] = group.articulation.data.body_link_quat_w[:,group.body_ids]
                    hinge_q[group.env_ids,:len(group.joint_ids)] = group.articulation.data.joint_pos[:,group.joint_ids]
                offsets = env.task._fingertip_offsets.expand_as(state.fingertip_pos_w)
                pads = state.fingertip_pos_w+rotations.apply(state.fingertip_quat_w,offsets)
                height = state.points_w[...,2]-state.table_top_z[:,None]
                tensors = dict(robot_q=state.joint_pos,robot_root_pose=torch.cat((env.backend.robot.data.root_pos_w-origins,env.backend.robot.data.root_quat_w),-1),
                    cloth_points=state.points_w-origins[:,None],panel_pose=panel_pose,hinge_q=hinge_q,
                    fingerpad_positions=pads-origins[:,None],table_center=env.backend.table.data.root_pos_w-origins,
                    table_top=state.table_top_z-origins[:,2],active=active.clone(),force_vector_by_half=vector,
                    force_magnitude_by_half=magnitude,downward_force_by_half=downward,
                    fold_error=view.fold_error,footprint=view.footprint,stationary_normal_up=view.frame_r[:,2,2],
                    stationary_mean_height=height[:,env.backend.cloth_grid.stationary_idx].mean(1),
                    stationary_height_span=height[:,env.backend.cloth_grid.stationary_idx].amax(1)-height[:,env.backend.cloth_grid.stationary_idx].amin(1),
                    moving_mean_height=height[:,env.backend.cloth_grid.moving_idx].mean(1),
                    moving_max_height=height[:,env.backend.cloth_grid.moving_idx].amax(1))
                frames.append({key:value.detach().cpu().numpy().copy() for key,value in tensors.items()})
            env.scene.update = update
            terminal = []
            for step in range(args.steps):
                with torch.no_grad():
                    action = player.get_action(obs['policy'],deterministic=True).clamp(-1,1)
                    recording = True
                    obs,reward,terminated,truncated,extra = env.step(action)
                    recording = False
                done = terminated|truncated
                first = active & done
                for e in first.nonzero().flatten().tolist():
                    terminal.append(dict(env_id=e,step=step+1,reasons={k:bool(v[e]) for k,v in env.task._termination_reasons.items()}))
                active &= ~done
                if bool(done.any()):
                    player.reset_rnn(done.nonzero().flatten())
                if (step+1)%100 == 0:
                    raise_if_overflowed(captured(),n,f'pressing audit step{step+1}')
                    print(f'[press-audit] step={step+1} first_episodes_complete={int((~active).sum())}/{n}',flush=True)
                if not bool(active.any()):
                    break
            raise_if_overflowed(captured(),n,'pressing audit complete')
            metadata.update(status='complete',steps=step+1,physics_frames=len(frames),terminals=terminal,
                            contact_metadata=contacts.metadata,completed_first_episodes=int((~active).sum()))
    except BaseException as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        metadata['elapsed_seconds'] = time.perf_counter()-started
        if frames:
            np.savez_compressed(args.out_dir/'trajectory.npz',**{key:np.stack([f[key] for f in frames]) for key in frames[0]})
        if records:
            np.savez_compressed(args.out_dir/'contacts.npz',**{key:np.concatenate([r[key] for r in records]) for key in records[0]})
        (args.out_dir/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
        contacts = None
        if env is not None:
            env.close()
        if metadata['status'] == 'failed':
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        app.close()


if __name__ == '__main__':
    main()
