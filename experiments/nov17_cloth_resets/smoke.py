"""Exercise live episode checks and measure their cost at the training batch size."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--steps', type=int, default=360)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--force-outside', action='store_true')
    p.add_argument('--force-curriculum', action='store_true')
    p.add_argument('--experiment', default='experiments/nov17_cloth_resets/scratch.yaml')
    p.add_argument('--out', type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args();app = AppLauncher(args).app
    import torch
    import yaml
    from isaacsimenvs.scripts.configs.loader import load_experiment
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
    from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer

    for name in ('prior','scratch','prior_no_curriculum','scratch_no_curriculum'):
        recipe = load_experiment(f'experiments/nov17_cloth_resets/{name}.yaml')
        rigid, agent = recipe['env']['rigid_cloth'], recipe['agent']['params']
        assert rigid['thickness_curriculum']['enabled'] == ('no_curriculum' not in name)
        assert rigid['episode_checks']['outside_table'] and rigid['episode_checks']['press_driven_fold']
        assert agent['load_checkpoint'] == name.startswith('prior')
        assert agent['checkpoint_load_mode'] == 'weights'

    cfg = ClothCfg.for_backend('rigid')
    cfg.from_dict(yaml.safe_load((ROOT/'isaacsimenvs/scripts/configs/task/RigidBodyCloth.yaml').read_text()))
    cfg.from_dict(load_experiment(args.experiment)['env'])
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 17
    if args.force_curriculum:
        cfg.sim.enable_scene_query_support = True
    result = dict(status='running', num_envs=args.num_envs, config=cfg.to_dict())
    env = None
    try:
        start = time.perf_counter();env = ClothEnv(cfg);obs,_ = env.reset()
        result['startup_s'] = time.perf_counter()-start
        for group in env.backend.groups:
            view=group.articulation.root_physx_view
            contact,rest=view.get_contact_offsets(),view.get_rest_offsets()
            assert torch.allclose(contact,torch.full_like(contact,cfg.rigid_cloth.contact_offset))
            expected=getattr(env.backend,'_dr_rest_offset',None)
            expected=(torch.full_like(rest,cfg.rigid_cloth.rest_offset) if expected is None else
                      expected[group.env_ids].cpu()[:,None].expand_as(rest))
            assert torch.allclose(rest,expected)
        result['cloth_offsets_verified']=dict(contact=cfg.rigid_cloth.contact_offset,rest=cfg.rigid_cloth.rest_offset)
        player = (PretrainedPlayer(ROOT/'pretrained_policy/flexiv_wuji/config.yaml',args.checkpoint,args.num_envs,
                  device=args.device,num_observations=134,num_actions=27,expl_coef=0.) if args.checkpoint else None)
        counts = dict(outside_table=0,press_driven_fold=0,nonfinite=0,released=0)
        torch.cuda.synchronize();start = time.perf_counter()
        for step in range(args.steps):
            with torch.no_grad():
                action = (player.get_action(obs['policy'],deterministic=True).clamp(-1,1) if player else
                          torch.zeros(args.num_envs,27,device=env.device))
                obs,reward,done,timeout,extra = env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            final = extra['episode_final']
            for name in counts:counts[name] += int(final.get('done_'+name,torch.zeros(1)).sum())
            invalid = (final['done_outside_table']+final['done_press_driven_fold'])>0
            assert not bool((reward[invalid] != 0).any()), 'Invalid episode received reward'
            if player is not None and bool((done|timeout).any()):player.reset_rnn((done|timeout).nonzero().flatten())
        torch.cuda.synchronize()
        assert env._episode_guard.contacts._raw_samples == args.steps*cfg.decimation, 'Missed or duplicated physics observations'
        result.update(step_wall_s=time.perf_counter()-start,steps=args.steps,counts=counts,
                      ready_fraction=float(env._episode_guard.success_allowed.float().mean()),
                      torch_peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                      contact_capacity=env._episode_guard.contacts.capacity)
        if args.force_curriculum:
            from isaacsimenvs.tasks.cloth.backends.rigid.randomization import rigid_check
            curriculum = env.thickness_curriculum
            assert abs(float(env.backend._dims[:, 2].mean())-.020) < 1e-7
            curriculum.recent.extend([True]*curriculum.cfg.window_episodes)
            obs,reward,done,timeout,extra = env.step(torch.zeros(args.num_envs,27,device=env.device))
            assert bool(done.all()) and not bool(timeout.any())
            assert curriculum.advances == 1
            assert torch.allclose(env.backend._dims[:, 2],torch.full((args.num_envs,),.018,device=env.device))
            assert torch.allclose(env.backend.cloth_grid.ply_gap,env.backend._dims[:,2])
            assert torch.allclose(env.task._object_scales[:,2],env.backend._dims[:,2]/cfg.reward.object_base_size)
            assert len(curriculum.recent) == 0
            rigid_check(env.backend,torch.arange(args.num_envs,device=env.device))
            import numpy as np
            from isaacsimenvs.tasks.cloth.backends.rigid.designs import polygon_mass_properties
            for group in env.backend.groups:
                e = int(group.env_ids[0])
                spec = group.design.resolve(*env.backend._dimensions_cpu[e])
                for plate, body in zip(spec.plates,group.body_ids):
                    mass,_,inertia,_ = polygon_mass_properties(plate,cfg.rigid_cloth.density)
                    assert np.isclose(float(group.mass_base[0,body]),mass,rtol=1e-5)
                    actual = np.linalg.eigvalsh(group.inertia_base[0,body].reshape(3,3).numpy())
                    assert np.allclose(actual,np.sort(inertia),rtol=1e-4,atol=1e-12),(actual,inertia)
            # Verify native collision surfaces and joint anchors, not just cached
            # dimensions/observations. Use the two-plate topology, away from robot.
            from omni.physx import get_physx_scene_query_interface
            b = env.backend
            group = next(g for g in b.groups if len(g.design.plates)==2)
            ids = group.env_ids
            b._write_cloth(ids,torch.zeros(len(ids),2,device=env.device),torch.zeros(len(ids),device=env.device),
                           torch.zeros(len(ids),device=env.device))
            # Exercise normal physics hooks after writing the diagnostic pose.
            env.step(torch.zeros(args.num_envs,27,device=env.device))
            from isaaclab.sim.utils import get_current_stage
            from pxr import Usd, UsdGeom, Gf
            # GPU tensor pose writes do not synchronize the CPU query scene's
            # poses. Query its authored rest pose to measure shape thickness.
            e = int(ids[0])
            plate = group.design.resolve(*b._dimensions_cpu[e]).plates[0]
            prim = get_current_stage().GetPrimAtPath(f'/World/envs/env_{e}/RigidCloth_{group.index}/{plate.name}')
            matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
            com = matrix.Transform(Gf.Vec3d(*plate.center))
            hit = get_physx_scene_query_interface().raycast_closest((com[0],com[1],com[2]+.04),(0.,0.,-1.),.08)
            assert hit['hit'] and 'RigidCloth_' in hit['rigidBody'], hit
            assert abs(float(hit['position'][2])-com[2]-.009) < 2e-5, (hit,com)
            b._write_cloth(ids,torch.zeros(len(ids),2,device=env.device),torch.zeros(len(ids),device=env.device),
                           torch.full((len(ids),),torch.pi,device=env.device))
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
            p = group.articulation.data.body_link_pos_w[0,group.body_ids]
            assert abs(float((p[1,2]-p[0,2]).abs())-.018) < 2e-5, p
            obs,_ = env.reset()
            for _ in range(8):
                obs,reward,done,timeout,extra = env.step(torch.zeros(args.num_envs,27,device=env.device))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            result['curriculum_rebuild_verified']=True
            saved = env.get_env_state()
            saved['thickness_curriculum'].update(thickness=.002,recent=[False,True],advances=22)
            env.set_env_state(saved)
            assert env.get_env_state() == saved
            assert torch.allclose(env.backend._dims[:,2],torch.full((args.num_envs,),.002,device=env.device))
            for _ in range(8):
                obs,reward,done,timeout,extra = env.step(torch.zeros(args.num_envs,27,device=env.device))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            result['curriculum_checkpoint_restore_verified']=True
        if args.force_outside:
            group=next(g for g in env.backend.groups if 0 in g.env_ids.tolist())
            local=int(group.local_from_global[0]);pose=group.articulation.data.root_pose_w[local:local+1].clone()
            pose[:,0] += 1.
            group.articulation.write_root_pose_to_sim(pose,env_ids=torch.tensor([local],device=env.device))
            _,reward,done,_,extra=env.step(torch.zeros(args.num_envs,27,device=env.device))
            assert bool(done[0]) and bool(extra['episode_final']['done_outside_table'][0])
            assert float(reward[0])==0
            assert not bool(env._episode_guard.history.ready[0].any())
            result['forced_outside_reset_verified']=True
        result['status']='complete'
        print(json.dumps({k:v for k,v in result.items() if k!='config'}),flush=True)
    except BaseException:
        result.update(status='failed',traceback=traceback.format_exc());traceback.print_exc()
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,default=str)+'\n')
        if env is not None:env.close()
        if result['status']=='failed':sys.stdout.flush();sys.stderr.flush();os._exit(1)
        app.close()


if __name__=='__main__':main()
