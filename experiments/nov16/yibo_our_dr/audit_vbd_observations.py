"""Real Newton observation audit against frozen Yibo code and independent URDF FK.

Runs 12 environments to include multi-digit environment indices. No production
code is changed; controlled states are written directly, without a policy rollout.
"""
from __future__ import annotations
import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.nov16.yibo_our_dr.evaluate_vbd import parser,build_config
REF='2b82faf1c73f5782068f67eac9cd6085aa1e27e7'
OUT=ROOT/'debug_outputs/deformable_observation_audit/2026-11-16'

def old_source(path):
    return subprocess.check_output(['git','show',f'{REF}:{path}'],cwd=ROOT,text=True)

def frozen_module(name,path,drop_imports=False,namespace=None):
    mod=types.ModuleType(name);sys.modules[name]=mod
    if namespace: mod.__dict__.update(namespace)
    tree=ast.parse(old_source(path))
    if drop_imports: tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(tree,f'{REF}:{path}','exec'),mod.__dict__)
    return mod

def load_oracles():
    robots=frozen_module('audit_old_robots','isaacsimenvs/tasks/play/robots.py')
    adapter=frozen_module('audit_old_cloth','isaacsimenvs/tasks/cloth/utils/cloth_adapter.py')
    math_path=ROOT/'.venv/isaacsim/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/utils/math.py'
    spec=importlib.util.spec_from_file_location('audit_lab2_math',math_path)
    math2=importlib.util.module_from_spec(spec);sys.modules[spec.name]=math2;spec.loader.exec_module(math2)
    ns={'torch':torch,'math':math,'KUKA_SHARPA':robots.KUKA_SHARPA}
    ns.update({k:getattr(math2,k) for k in ('convert_quat','quat_apply','quat_from_angle_axis','quat_mul')})
    observer=frozen_module('audit_old_obs','isaacsimenvs/tasks/play/utils/obs_utils.py',True,ns)
    # Extract the two goal-frame functions verbatim, avoiding the old simulator class imports.
    src=old_source('isaacsimenvs/tasks/cloth/fold_pool_task.py');tree=ast.parse(src)
    funcs={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    goal={'torch':torch}
    for name in ('_stationary_frame','_folded_w'):
        node=funcs[name];node.decorator_list=[]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),REF,'exec'),goal)
    return robots.FLEXIV_WUJI_RIGHT_HUMAN_LIKE,adapter,observer,goal

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    args=parser().parse_args(['--material','base','--num-envs','12','--steps','120'])
    cfg=build_config(args,'base')
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
    from isaacsimenvs.tasks.cloth.backends.base import as_torch
    from isaacsimenvs.scripts.evaluation.utils import PretrainedPlayer
    from isaaclab_newton.physics import NewtonManager
    from deployment.utils import robot_model
    oldspec,adapter,oldobs,goal=load_oracles()
    result={'oracle_commit':REF,'num_envs':12,'cases':[]}
    env=None
    try:
        env=ClothEnv(cfg);env.reset()
        n=env.num_envs;device=env.device;origins=env.scene.env_origins
        model=NewtonManager.get_model();raw=NewtonManager.get_state_0()
        names=list(env.backend.robot.data.joint_names)
        bodies=list(env.backend.robot.data.body_names)
        pworld=model.particle_world.numpy();tris=model.tri_indices.numpy()
        imported_rest=model.particle_q.numpy()
        result['particle_offsets']=env.backend._cloth.data._particle_offsets.numpy().tolist()
        result['particle_world_sequence']=pworld.tolist()
        result['body_labels']=list(model.body_label)
        result['env_origins']=origins.cpu().tolist()
        result['robot_body_names']=bodies
        result['robot_joint_names']=names
        result['deformable_data_attrs']=[k for k in vars(env.backend._cloth.data) if any(t in k for t in ('index','offset','count','view'))]
        # Independent row-major rest mesh, not the production grid helper.
        rest=torch.tensor([(i*.025-.075,j*.025-.075,0.) for i in range(7) for j in range(7)],device=device)
        expected_tris=[]
        for i in range(6):
            for j in range(6):
                a=i*7+j;b=a+1;c=a+7;d=c+1
                expected_tris.extend([[a,c,d],[a,d,b]] if (i+j)%2 else [[a,c,b],[b,c,d]])
        expected_set={tuple(t) for t in expected_tris}
        result['table_geometry']=[]
        bodyq=raw.body_q.numpy();labels=list(model.body_label)
        shape_body=model.shape_body.numpy();shape_scale=model.shape_scale.numpy();shape_transform=model.shape_transform.numpy()
        for e in range(n):
            tid=labels.index(f'/World/envs/env_{e}/Table/box')
            rid=labels.index(f'/World/envs/env_{e}/Robot/base_link')
            origin=origins[e].cpu().numpy()
            result['table_geometry'].append(dict(env=e,
                table_center_env=bodyq[tid,:3].__sub__(origin).tolist(),
                table_quaternion_xyzw=bodyq[tid,3:7].tolist(),
                robot_base_env=bodyq[rid,:3].__sub__(origin).tolist(),
                shape_half_extents=shape_scale[shape_body==tid].tolist(),
                shape_transforms=shape_transform[shape_body==tid].tolist(),
                readout_table_top_env_z=float(env.backend.read_state().table_top_z[e]-origins[e,2])))
        result['topology']=[]
        rawp=raw.particle_q.numpy();readp=env.backend.cloth_points()[0].cpu().numpy()
        for e in range(n):
            ids=np.flatnonzero(pworld==e);inv={int(v):i for i,v in enumerate(ids)}
            ts=[t for t in tris if int(t[0]) in inv]
            actual={tuple(inv[int(v)] for v in t) for t in ts}
            result['topology'].append(dict(env=e,global_particle_ids=ids.tolist(),triangles=len(ts),
                triangle_connectivity_equal=actual==expected_set,
                imported_rest_coordinate_error_m=float(np.max(np.abs((imported_rest[ids]-imported_rest[ids].mean(0))-rest.cpu().numpy()))),
                raw_world_to_read_row_error=float(np.max(np.abs(rawp[ids]-readp[e])))))
        fkmodel=robot_model.load_robot_model(dict(urdf_path=str(ROOT/cfg.assets.robot_urdf),joint_names=names,
            robot_root_pos_env=np.array(cfg.robot_root_pos),robot_root_quat_wxyz=np.array([1.,0.,0.,0.])))
        selected=[oldspec.palm_body_name,*oldspec.fingertip_link_names]
        perm=torch.tensor([names.index(k) for k in oldspec.joint_names_canonical],device=device)
        limits=as_torch(env.backend.robot.data.joint_pos_limits)
        oldcloth=adapter.ClothAsRigidObject(env.backend._cloth,num_envs=n,rest_local=rest.cpu().tolist(),
            keypoint_idx=[28,34,42,48],corner_idx=[28,34,42,48],corner_rest=rest[[28,34,42,48]].cpu().tolist(),
            spawn_z=.58,mass=.045,device=device)
        stat=torch.tensor(list(range(21)),device=device);st=rest[stat].expand(n,-1,-1)
        owner=NS(_stationary_idx=stat,_st_rest_centred=st-st.mean(1,keepdim=True),_st_rest_centroid=st.mean(1),
                 _particles_w=lambda: oldcloth._particles())
        owner._stationary_frame=lambda p:goal['_stationary_frame'](owner,p)
        player=PretrainedPlayer(args.policy_config,args.checkpoint,n,device=device,num_observations=134,num_actions=27,expl_coef=0.)
        qdefault=as_torch(env.backend.robot.data.default_joint_pos).clone()
        for case,(angle,deform) in enumerate([(0,False),(90,False),(180,False),(65,True)]):
            q=qdefault.clone();q[:,0]+=torch.linspace(-.12,.12,n,device=device)
            q[:,7:]+=.08*case
            q=torch.maximum(torch.minimum(q,limits[...,1]),limits[...,0])
            qvel=torch.arange(n*27,device=device,dtype=torch.float32).reshape(n,27)*.001
            env.backend.robot.write_joint_state_to_sim_index(position=q,velocity=qvel)
            env.controller._prev_targets.copy_(q+.01)
            yaw=torch.tensor([0.,90.,180.,37.]*3,device=device)*math.pi/180
            points=rest.expand(n,-1,-1).clone();positive=rest[:,0]>1e-6
            points[:,positive,0]*=math.cos(math.radians(angle))
            points[:,positive,2]=rest[positive,0]*math.sin(math.radians(angle))
            if deform:points[:,48]+=points.new_tensor([.012,-.007,.018])
            x,y=points[...,0].clone(),points[...,1].clone()
            points[...,0]=yaw.cos()[:,None]*x-yaw.sin()[:,None]*y
            points[...,1]=yaw.sin()[:,None]*x+yaw.cos()[:,None]*y
            points+=origins[:,None]+points.new_tensor([.03,-.04,.58])
            vels=torch.arange(n*49*3,device=device,dtype=torch.float32).reshape(n,49,3)*1e-5
            ids=torch.arange(n,device=device)
            env.backend._cloth.write_nodal_pos_to_sim_index(points,ids)
            env.backend._cloth.write_nodal_velocity_to_sim_index(vels,ids)
            state=env.backend.read_state();view=env.task.derive(state)
            actual=env.observations.build(state,view,env.episode_length_buf)['policy']
            rawbody=as_torch(env.backend.robot.data.body_state_w).clone()
            rawlink=as_torch(env.backend.robot.data.body_link_pose_w)
            selectedids=[bodies.index(k) for k in selected]
            fkerrors=[];qerrors=[]
            for e in range(n):
                for k,bid in zip(selected,selectedids):
                    p,r=robot_model.link_pose(fkmodel,k,q[e].cpu().numpy())
                    p+=origins[e].cpu().numpy()
                    actualp=rawbody[e,bid,:3].cpu().numpy();actualq=rawbody[e,bid,3:7].cpu().numpy()[[3,0,1,2]]
                    fkerrors.append(np.max(np.abs(p-actualp)))
                    qerrors.append(min(np.max(np.abs(r-actualq)),np.max(np.abs(r+actualq))))
            # Original observation builder receives original robot LINK state in wxyz.
            rawbody[...,3:7]=rawbody[...,3:7][...,[3,0,1,2]]
            r,_=owner._stationary_frame(oldcloth._particles())
            folded=rest[[28,34,42,48]].expand(n,-1,-1).clone();folded[...,0]*=-1
            goalpos=goal['_folded_w'](owner,folded).mean(1)
            # Preserve old Lab3 xyzw goal write read as wxyz, including the component relabel.
            goalq=adapter._mat_to_wxyz(r@torch.diag(rest.new_tensor([-1.,1.,-1.])))[:,[1,2,3,0]]
            dims=rest.new_tensor([.15,.15,.016]).expand(n,-1)
            fake=NS(cfg=cfg,robot_spec=oldspec,num_envs=n,device=device,scene=NS(env_origins=origins),
                robot=NS(data=NS(joint_pos=as_torch(env.backend.robot.data.joint_pos),joint_vel=as_torch(env.backend.robot.data.joint_vel),body_state_w=rawbody)),
                object=oldcloth,goal_viz=NS(data=NS(root_pos_w=goalpos,root_quat_w=goalq)),
                _perm_lab_to_canon=perm,_joint_lower_canon=limits[0,perm,0],_joint_upper_canon=limits[0,perm,1],
                _prev_targets=env.controller._prev_targets,_palm_body_id=selectedids[0],_fingertip_body_ids=selectedids[1:],
                _keypoint_offsets=(rest[[28,34,42,48]]-rest[[28,34,42,48]].mean(0)).expand(n,-1,-1),
                _object_scale_multiplier=torch.ones(n,3,device=device),_object_scale_per_env=dims/.04,
                _object_bias_pos=torch.zeros(n,3,device=device),_object_bias_quat=rest.new_tensor([1.,0.,0.,0.]).expand(n,-1),
                _closest_keypoint_max_dist=env.task._closest_fold_err,_closest_fingertip_dist=env.task._closest_fingertip_dist,
                _lifted_object=torch.zeros(n,device=device),_successes=env.task._successes,
                episode_length_buf=env.episode_length_buf,reward_buf=env.task.reward_buf)
            cfg.domain_randomization.point_dr_mode='off'
            reference=oldobs.build_observations(fake)['policy']
            field_errors={};start=0
            for field in cfg.obs.obs_list:
                width=27 if field in ('joint_pos','joint_vel','prev_action_targets') else oldobs.OBS_FIELD_SIZES[field]
                field_errors[field]=float((actual[:,start:start+width]-reference[:,start:start+width]).abs().max());start+=width
            player.reset_rnn();a=player.get_action(actual,deterministic=True)
            player.reset_rnn();b=player.get_action(reference,deterministic=True)
            # Compare readout directly to global simulation buffers grouped by raw world tags.
            raw=NewtonManager.get_state_0();rawp=raw.particle_q.numpy();readp=state.points_w.cpu().numpy()
            bodyq=raw.body_q.numpy();labels=list(model.body_label);bodyworld=model.body_world.numpy()
            raw_body_error=0.;world_tags=[]
            for e in range(n):
                for link,bid in zip(selected,selectedids):
                    global_id=labels.index(f'/World/envs/env_{e}/Robot/{link}')
                    world_tags.append(int(bodyworld[global_id]))
                    expected=as_torch(env.backend.robot.data.body_link_pose_w)[e,bid].cpu().numpy()
                    raw_body_error=max(raw_body_error,float(np.max(np.abs(bodyq[global_id]-expected))))
            assert world_tags==[e for e in range(n) for _ in selected]

            worlderr=max(float(np.max(np.abs(rawp[np.flatnonzero(pworld==e)]-readp[e]))) for e in range(n))
            row=dict(joint_velocity_write_read_error=float((state.joint_vel-qvel).abs().max()),raw_body_label_pose_error=raw_body_error,fold_degrees=angle,deformed=deform,point_write_read_error=float((state.points_w-points).abs().max()),
                velocity_write_read_error=float((state.points_vel_w-vels).abs().max()),raw_world_row_error=worlderr,
                fk_link_position_error_m=float(max(fkerrors)),fk_link_quaternion_error_sign_invariant=float(max(qerrors)),
                body_state_vs_link_pose_error=float((as_torch(env.backend.robot.data.body_state_w)[...,:7]-rawlink).abs().max()),
                actor_field_errors=field_errors,actor_max_error=float((actual-reference).abs().max()),
                checkpoint_action_max_error=float((a-b).abs().max()))
            result['cases'].append(row);print('AUDIT',json.dumps(row),flush=True)
        snapshot=torch.load(args.checkpoint,map_location='cpu',weights_only=False)[0]['obs']['obs']
        source_rows=[0,1,2,9,10,11,128,1024,4096,8192,16384,24575]
        saved=snapshot[source_rows,:134].to(device)
        qcanon=(saved[:,:27]+1)*.5*(limits[0,perm,1]-limits[0,perm,0])+limits[0,perm,0]
        q=torch.empty_like(qcanon);q[:,perm]=qcanon
        env.backend.robot.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q))
        state=env.backend.read_state()
        actual=env.observations.build(state,env.task.derive(state),env.episode_length_buf)['policy']
        entries=[]
        for i,source in enumerate(source_rows):
            entries.append(dict(source_row=source,
                joint_normalized_error=float((actual[i,:27]-saved[i,:27]).abs().max()),
                palm_position_error_m=float((actual[i,81:84]-saved[i,81:84]).abs().max()),
                palm_quaternion_signed_error=float((actual[i,84:88]-saved[i,84:88]).abs().max()),
                palm_quaternion_dot=float((actual[i,84:88]*saved[i,84:88]).sum()),
                fingertip_relative_position_error_m=float((actual[i,92:107]-saved[i,92:107]).abs().max()),
                actual_palm_quaternion_xyzw=actual[i,84:88].cpu().tolist(),
                saved_palm_quaternion_xyzw=saved[i,84:88].cpu().tolist()))
        result['saved_physx_robot_checks']=entries
        print('SAVED_PHYSX',json.dumps(entries),flush=True)
        assert all(t['triangle_connectivity_equal'] and t['raw_world_to_read_row_error']<1e-6
                   and t['imported_rest_coordinate_error_m']<1e-5 for t in result['topology'])
        assert all(c['actor_max_error']<2e-6 and c['checkpoint_action_max_error']<1e-4
                   and c['fk_link_position_error_m']<2e-6 and c['raw_body_label_pose_error']<1e-6
                   and c['raw_world_row_error']<1e-6 and c['point_write_read_error']<1e-6
                   and c['velocity_write_read_error']<1e-6 and c['joint_velocity_write_read_error']<1e-6
                   for c in result['cases'])
        assert all(e['palm_quaternion_signed_error']<2e-6 and e['palm_position_error_m']<5e-5
                   and e['fingertip_relative_position_error_m']<5e-5 for e in entries)
        result['status']='complete'
    except BaseException as exc:
        import traceback
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        if env is not None:env.close()

if __name__=='__main__':main()
