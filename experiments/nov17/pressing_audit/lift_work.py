"""Symmetric tests of whether the robot directly drives the half that rises.

Work is a finite-step estimate, not an exact energy-conservation certificate.
Normal and patch-friction forces are kept separate. The crease-work quantity
projects the hand's moment onto the rising core plate's actual crease rotation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_pressing import geometry_metrics, load_dataset, plate_properties, rotation
    from .validate_resets import first_crossing, opening_events
except ImportError:
    from analyze_pressing import geometry_metrics, load_dataset, plate_properties, rotation
    from validate_resets import first_crossing, opening_events


def record_work(record, force, poses, previous, owners, anchors, axes, angular_step,
                com_pose=None, previous_com=None, velocity=None, previous_velocity=None, dt=None):
    """Contact work on cloth and hand moment work about the central crease."""
    t, n, p, _ = poses.shape
    k, e, b = (record[x].astype(int) for x in ('frame_idx','env_id','cloth_body'))
    partner = record['partner_type'].astype(int)
    h = owners[e,b]
    current_r = rotation(poses[k,e,b,3:])
    previous_r = rotation(previous[k,e,b,3:])
    rest = np.einsum('rji,rj->ri',current_r,record['position']-poses[k,e,b,:3])
    previous_point = np.einsum('rij,rj->ri',previous_r,rest)+previous[k,e,b,:3]
    delta = record['position']-previous_point
    displacement_work = np.einsum('ri,ri->r',force,delta)
    if velocity is not None:
        old_v = previous_velocity[k,e,b,:3]+np.cross(previous_velocity[k,e,b,3:],previous_point-previous_com[k,e,b,:3])
        new_v = velocity[k,e,b,:3]+np.cross(velocity[k,e,b,3:],record['position']-com_pose[k,e,b,:3])
        work = np.einsum('ri,ri->r',force,.5*(old_v+new_v))*dt
    else:
        work = displacement_work
    moment = np.cross(record['position']-anchors[k,e],force)
    torque = np.einsum('ri,ri->r',moment,axes[k,e])
    crease = torque*angular_step[k,e,h]
    shape=(t,n,2,2)
    output={}
    for name,value in [('work',work),('displacement_work',displacement_work),('positive_work',np.maximum(work,0)),('crease_work',crease),('opening_torque_abs',np.abs(torque))]:
        array=np.zeros(shape)
        np.add.at(array,(k,e,partner,h),value)
        output[name]=array
    return output


def robot_lifting_work(record,robot_pose,robot_com,robot_velocity,owners,anchors,axes,opening_sign,dt):
    """Input work at the ACTUATOR side of contact, qualified by opening moment.

    A stationary hand receives zero credit even when contact correction moves
    the cloth. Downward work that closes the rising half earns no lifting credit.
    """
    t,n,_,_=robot_pose.shape
    keep=(record['partner_type']==0)&(record['frame_idx']>0)
    k,e,b,j=(record[x][keep].astype(int) for x in ('frame_idx','env_id','cloth_body','partner_index'))
    x,F=record['position'][keep],record['force_vector'][keep]
    half=owners[e,b]
    rest=np.einsum('rji,rj->ri',rotation(robot_pose[k,e,j,3:]),x-robot_pose[k,e,j,:3])
    old_point=np.einsum('rij,rj->ri',rotation(robot_pose[k-1,e,j,3:]),rest)+robot_pose[k-1,e,j,:3]
    current=robot_velocity[k,e,j,:3]+np.cross(robot_velocity[k,e,j,3:],x-robot_com[k,e,j,:3])
    previous=robot_velocity[k-1,e,j,:3]+np.cross(robot_velocity[k-1,e,j,3:],old_point-robot_com[k-1,e,j,:3])
    energy=np.einsum('ri,ri->r',F,.5*(current+previous))*dt
    moment=np.einsum('ri,ri->r',np.cross(x-anchors[k,e],F),axes[k,e])*opening_sign[e,half]
    all_work=np.zeros((t,n,2));lift_work=np.zeros_like(all_work)
    np.add.at(all_work,(k,e,half),energy)
    eligible=moment>0
    np.add.at(lift_work,(k[eligible],e[eligible],half[eligible]),energy[eligible])
    return all_work,lift_work


def compute(directory):
    m,d,c,g=load_dataset(directory)
    if m['status']!='complete':raise ValueError('Incomplete work capture')
    for field in ('panel_com_velocity','robot_com_velocity'):
        if field in d and not np.isfinite(d[field][d['active'].astype(bool)]).all():
            raise ValueError(f'Nonfinite {field} in an active first episode')
    fp=Path(directory)/'friction.npz'
    if not fp.exists():raise ValueError('This test needs separate friction records')
    with np.load(fp) as f:fr={k:f[k] for k in f.files}
    pose=d['panel_pose'].astype(float);t,n,p,_=pose.shape
    previous=np.concatenate([pose[:1].copy(),pose[:-1]],axis=0)
    velocity=d['panel_com_velocity'].astype(float)
    previous_velocity=np.concatenate([velocity[:1].copy(),velocity[:-1]],axis=0)
    com_pose=d['panel_com_pose'].astype(float)
    previous_com=np.concatenate([com_pose[:1].copy(),com_pose[:-1]],axis=0)
    owners=np.zeros((n,p),int);masses=np.zeros((n,p));coms=np.zeros((n,p,3));core=np.zeros((n,2),int)
    anchors=np.zeros((t,n,3));axes=np.zeros_like(anchors);angular=np.zeros((t,n,2))
    initial_height=np.zeros((n,2));height=np.zeros((t,n,2));weight=np.zeros((n,2));opening_sign=np.zeros((n,2))
    for e,s in g.items():
        count=len(s['plates']);previous[0,e,:count]=s['initial_panel_pose']
        masses[e,:count]=s['material']['plate_masses']
        coms[e,:count]=np.asarray(s['material']['plate_coms'])[:,:3]
        if 'initial_com_velocity' in s:
            previous_velocity[0,e,:count]=s['initial_com_velocity']
        previous_com[0,e,:count,:3]=np.einsum('bij,bj->bi',rotation(previous[0,e,:count,3:]),coms[e,:count])+previous[0,e,:count,:3]
        for b,plate in enumerate(s['plates']):owners[e,b]=plate_properties(plate)[2]
        hs=[h for h in s['hinges'] if owners[e,h['parent']]!=owners[e,h['child']]]
        if len(hs)!=1:raise ValueError('Need exactly one central crease between material halves')
        hinge=hs[0];a,b=hinge['parent'],hinge['child'];r=rotation(pose[:,e,a,3:])
        anchors[:,e]=np.einsum('tij,j->ti',r,hinge['anchor'])+pose[:,e,a,:3]
        axes[:,e]=np.einsum('tij,j->ti',r,hinge['axis'])
        for plate in (a,b):core[e,owners[e,plate]]=plate
        for half in (0,1):
            b=core[e,half]
            midpoint_omega=.5*(velocity[:,e,b,3:]+previous_velocity[:,e,b,3:])
            angular[:,e,half]=np.einsum('ti,ti->t',midpoint_omega,axes[:,e])*m['physics_dt']
            selected=np.flatnonzero(owners[e,:count]==half);w=masses[e,selected];weight[e,half]=w.sum()*9.81
            current_com=np.einsum('tbij,bj->tbi',rotation(pose[:,e,selected,3:].reshape(-1,4)).reshape(t,len(selected),3,3),coms[e,selected])+pose[:,e,selected,:3]
            first_com=np.einsum('bij,bj->bi',rotation(previous[0,e,selected,3:]),coms[e,selected])+previous[0,e,selected,:3]
            height[:,e,half]=(current_com[:,:,2]*w[None]).sum(1)/w.sum()
            initial_height[e,half]=(first_com[:,2]*w).sum()/w.sum()
            initial_center=(first_com*w[:,None]).sum(0)/w.sum()
            initial_axis=rotation(previous[0,e,a,3:])@np.asarray(hinge['axis'])
            initial_anchor=rotation(previous[0,e,a,3:])@np.asarray(hinge['anchor'])+previous[0,e,a,:3]
            opening_sign[e,half]=np.sign(np.cross(initial_axis,initial_center-initial_anchor)[2])
            if opening_sign[e,half]==0:raise ValueError('Central-crease opening direction is degenerate')
    gravity=(height-initial_height[None])*weight[None]
    args=(pose,previous,owners,anchors,axes,angular,com_pose,previous_com,velocity,previous_velocity,m['physics_dt'])
    normal=record_work(c,c['force_vector'],*args)
    friction=record_work(fr,fr['force_vector'],*args)
    sums={key:(normal[key]+friction[key]).cumsum(0) for key in ('work','positive_work','crease_work','displacement_work')}
    for key in tuple(sums):
        sums['normal_'+key]=normal[key].cumsum(0)
        sums['friction_'+key]=friction[key].cumsum(0)
    if 'robot_body_pose' in d:
        source_normal,lift_normal=robot_lifting_work(c,d['robot_body_pose'],d['robot_com_pose'],d['robot_com_velocity'],owners,anchors,axes,opening_sign,m['physics_dt'])
        source_friction,lift_friction=robot_lifting_work(fr,d['robot_body_pose'],d['robot_com_pose'],d['robot_com_velocity'],owners,anchors,axes,opening_sign,m['physics_dt'])
    else:
        if bool((c['force_magnitude'][c['partner_type']==0]>1e-5).any()) or bool((np.linalg.norm(fr['force_vector'][fr['partner_type']==0],axis=1)>1e-5).any()):
            raise ValueError('Robot-side work requires actual robot poses and velocities')
        source_normal=lift_normal=source_friction=lift_friction=np.zeros((t,n,2))
    sums['robot_source_work_j']=(source_normal+source_friction).cumsum(0)
    sums['robot_lifting_work_j']=(lift_normal+lift_friction).cumsum(0)
    return m,d,c,fr,g,dict(**sums,gravity_j=gravity,half_weight_n=weight,half_height_m=height,owners=owners,
                           normal_torque_abs=normal['opening_torque_abs'],friction_torque_abs=friction['opening_torque_abs'])


def detect_without_lift_work(geom,active,inside,dims,gravity,weight,credit,press,angle=30.,fraction=.1,
                             current_press=False, return_halves=False,ready=None):
    """Task rule: reject pressed-base ascent without credited direct opening work.

    Inputs are symmetric material-half arrays. Work credit persists through
    release; it is not a recent-contact flag or an exact energy balance.
    """
    rejected=np.zeros((*active.shape,2),dtype=bool)
    for half in (0,1):
        other=1-half
        supported=inside&(geom['tilt_deg'][:,:,other]<20)&(geom['mean_height'][:,:,other]<dims[None,:,2]/2+.015)
        pressure=(np.where(supported,press[:,:,other],0)>.001 if current_press else
                  np.maximum.accumulate(np.where(supported,press[:,:,other],0),axis=0)>.01)
        rise=active&inside&(geom['tilt_deg'][:,:,half]>angle)&(geom['tilt_deg'][:,:,half]<90)&(geom['tilt_deg'][:,:,other]<20)
        gravity_gain=np.diff(gravity[:,:,half],axis=0,prepend=np.zeros((1,active.shape[1])))
        rise&=gravity_gain>1e-8
        rise&=geom['mean_height'][:,:,other]<dims[None,:,2]/2+.015
        rise&=gravity[:,:,half] > weight[None,:,half]*np.maximum(.005,.05*dims[None,:,0])
        if ready is not None:rise&=ready[:,:,half]
        rejected[:,:,half]=rise&pressure&(credit[:,:,half] < fraction*gravity[:,:,half])
    return rejected if return_halves else rejected.any(2)


def evaluate(directory, angle=30., fraction=.1):
    m,d,c,fr,g,work=compute(directory)
    geom=geometry_metrics(m,d,g);inside=geom['outside_vertex_fraction']==0;active=d['active'].astype(bool)
    # Coupled compression is an interaction observation, not a force budget.
    count=d['panel_pose'].shape[2];sh=(*active.shape,count);down=np.zeros(sh);up=np.zeros(sh);robot_z=np.zeros(sh)
    for source in (c,fr):
        k,e,b=(source[x].astype(int) for x in ('frame_idx','env_id','cloth_body'))
        robot=source['partner_type']==0
        np.add.at(robot_z,(k[robot],e[robot],b[robot]),source['force_vector'][robot,2])
        np.add.at(down,(k[robot],e[robot],b[robot]),np.maximum(-source['force_vector'][robot,2],0))
        np.add.at(up,(k[~robot],e[~robot],b[~robot]),np.maximum(source['force_vector'][~robot,2],0))
    paired=np.minimum(down,up)
    press=np.stack([(paired*(work['owners']==h)[None]).sum(2) for h in (0,1)],2)
    half_press=np.stack([np.minimum((down*(work['owners']==h)[None]).sum(2),
                                     (up*(work['owners']==h)[None]).sum(2)) for h in (0,1)],2)
    net_press=np.stack([net_supported_load((robot_z*(work['owners']==h)[None]).sum(2),
                                            (up*(work['owners']==h)[None]).sum(2)) for h in (0,1)],2)
    dims=np.asarray(m['dimensions'])
    variants={}
    for quantity in ('work','positive_work','crease_work'):
        rejected=detect_without_lift_work(geom,active,inside,dims,work['gravity_j'],work['half_weight_n'],
                                          work[quantity][:,:,0],press,angle,fraction)
        variants[quantity]=dict(first_trigger=first_crossing(rejected).tolist(),trigger_mask=rejected)
    rejected=detect_without_lift_work(geom,active,inside,dims,work['gravity_j'],work['half_weight_n'],
                                      work['work'][:,:,0],half_press,angle,fraction,current_press=True)
    variants['current_supported_work']=dict(first_trigger=first_crossing(rejected).tolist(),trigger_mask=rejected)
    # Start a new work account only after that whole material half is flat,
    # table-supported and nearly stationary. Earlier contact losses are not debt
    # against a later lift; credit survives release until the half settles again.
    pose=d['panel_pose'];t,n,p,_=pose.shape
    normal_up=rotation(pose[...,3:].reshape(-1,4)).reshape(t,n,p,3,3)[...,2,2]
    v=d['panel_com_velocity']
    phase_work=np.zeros_like(work['gravity_j']);phase_gravity=np.zeros_like(phase_work);phase_source=np.zeros_like(phase_work);ready=np.zeros_like(phase_work,dtype=bool)
    for h in (0,1):
        selected=(work['owners']==h)
        real=np.zeros_like(selected)
        for e,spec in g.items():real[e,:len(spec['plates'])]=True
        selected&=real
        flat=np.where(selected[None],normal_up,1).min(2)>np.cos(np.deg2rad(5))
        flat&=np.where(selected[None],np.linalg.norm(v[...,:3],axis=-1),0).max(2)<.02
        flat&=np.where(selected[None],np.linalg.norm(v[...,3:],axis=-1),0).max(2)<.5
        flat&=work['half_height_m'][:,:,h]-d['table_top']<dims[None,:,2]/2+.003
        flat&=(up*selected[None]).sum(2)>1e-5
        flat&=active&inside
        ready[:,:,h]=np.logical_or.accumulate(flat,axis=0)
        phase_work[:,:,h],phase_gravity[:,:,h]=restart_work_account(
            work['work'][:,:,0,h],work['gravity_j'][:,:,h],flat)
        phase_source[:,:,h],_=restart_work_account(work['robot_lifting_work_j'][:,:,h],work['gravity_j'][:,:,h],flat)
    work['phase_work_j']=phase_work;work['phase_gravity_j']=phase_gravity
    rejected=detect_without_lift_work(geom,active,inside,dims,phase_gravity,work['half_weight_n'],
                                      phase_work,half_press,angle,fraction,current_press=True)
    variants['phase_supported_work']=dict(first_trigger=first_crossing(rejected).tolist(),trigger_mask=rejected)
    rejected=detect_without_lift_work(geom,active,inside,dims,phase_gravity,work['half_weight_n'],
                                      phase_work,net_press,angle,fraction,current_press=True)
    variants['net_phase_supported_work']=dict(first_trigger=first_crossing(rejected).tolist(),trigger_mask=rejected)
    work['net_supported_load_n']=net_press
    work['phase_source_lifting_work_j']=phase_source
    work['phase_source_ready']=ready
    by_half=detect_without_lift_work(geom,active,inside,dims,phase_gravity,work['half_weight_n'],
                                      phase_source,net_press,angle,fraction,current_press=True,ready=ready,return_halves=True)
    rejected=by_half.any(2);first=first_crossing(rejected)
    detail=[]
    for e,k in enumerate(first):
        if k==t:continue
        h=int(np.flatnonzero(by_half[k,e])[0])
        detail.append(dict(env_id=e,frame=int(k),time_s=float((k+1)*m['physics_dt']),rising_half=h,
                           tilt_degrees=geom['tilt_deg'][k,e].tolist(),
                           robot_lifting_work_j=float(phase_source[k,e,h]),gravity_j=float(phase_gravity[k,e,h]),
                           other_half_net_supported_load_n=float(net_press[k,e,1-h])))
    variants['robot_source_lifting_work']=dict(first_trigger=first.tolist(),trigger_mask=rejected,first_triggers=detail)
    return m,d,g,work,variants


def restart_work_account(work,gravity,flat):
    last=np.maximum.accumulate(np.where(flat,np.arange(len(flat))[:,None],0),axis=0)
    observed=np.logical_or.accumulate(flat,axis=0)
    env=np.arange(flat.shape[1])[None]
    base_work=np.where(observed,work[last,env],0)
    base_gravity=np.where(observed,gravity[last,env],0)
    return work-base_work,gravity-base_gravity


def net_supported_load(robot_z,table_z):
    """A net upward pinch is not downward pressing, even if one finger pushes down."""
    return np.minimum(np.maximum(-robot_z,0),np.maximum(table_z,0))


def summarize(directory,events,angle=30.,fraction=.1):
    m,d,g,w,variants=evaluate(directory,angle,fraction)
    result=dict(dataset=str(directory),checkpoint=m['checkpoint'],angle_deg=angle,minimum_direct_work_fraction=fraction,
                num_envs=m['num_envs'],events=[],variants={})
    for quantity,v in variants.items():
        first=np.asarray(v['first_trigger']);entry=dict(worlds_triggered=int((first<len(d['active'])).sum()))
        if quantity=='robot_source_lifting_work':entry['first_triggers']=v['first_triggers']
        for cohort in ('reverse_half','intended_half'):
            selected=[e for e in events if e['cohort']==cohort]
            hit=[e['env_id'] for e in selected if first[e['env_id']]<=e['frame']]
            entry[cohort]=dict(count=len(selected),intercepted=len(hit),env_ids=hit)
            if quantity=='robot_source_lifting_work':
                entry[cohort]['ready_at_lift']=sum(bool(w['phase_source_ready'][e['frame'],e['env_id'],int(cohort=='intended_half')]) for e in selected)
        result['variants'][quantity]=entry
    for e in events:
        k,n=e['frame'],e['env_id'];h=int(e['cohort']=='intended_half')
        entry=dict(e)
        entry['robot_work_j']=float(w['work'][k,n,0,h]);entry['robot_positive_work_j']=float(w['positive_work'][k,n,0,h])
        entry['robot_crease_work_j']=float(w['crease_work'][k,n,0,h]);entry['gravity_j']=float(w['gravity_j'][k,n,h])
        entry['normal_crease_work_j']=float(w['normal_crease_work'][k,n,0,h]);entry['friction_crease_work_j']=float(w['friction_crease_work'][k,n,0,h])
        entry['phase_work_j']=float(w['phase_work_j'][k,n,h]);entry['phase_gravity_j']=float(w['phase_gravity_j'][k,n,h])
        entry['phase_robot_lifting_work_j']=float(w['phase_source_lifting_work_j'][k,n,h])
        entry['source_work_ready']=bool(w['phase_source_ready'][k,n,h])
        result['events'].append(entry)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets',nargs='+',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fraction',type=float,default=.1);parser.add_argument('--angle',type=float,default=30.)
    args=parser.parse_args();reports=[]
    for directory in args.datasets:
        m,d,c,g=load_dataset(directory);events,_=opening_events(m,d,g)
        result=summarize(directory,events,args.angle,args.fraction);reports.append(result)
        print(directory.name,json.dumps(result['variants']),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
