"""Render independent hand/cloth storyboards for work-detector disagreements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yourdfpy

try:
    from .analyze_pressing import load_dataset, plate_properties, rotation
    from .render_rollouts import storyboard
except ImportError:
    from analyze_pressing import load_dataset, plate_properties, rotation
    from render_rollouts import storyboard

ROOT=Path(__file__).resolve().parents[3]


def make_scene(m,d,g,world):
    vertices=[];indices=[[],[]];offset=0
    for p,plate in enumerate(g[world]['plates']):
        pose=d['panel_pose'][:,world,p]
        v=np.einsum('tij,vj->tvi',rotation(pose[:,3:]),np.asarray(plate['vertices']))+pose[:,None,:3]
        vertices.append(v);half=plate_properties(plate)[2]
        indices[half].extend((np.asarray(plate['triangles'])+offset).flatten().tolist());offset+=v.shape[1]
    vertices=np.concatenate(vertices,1)
    root=d['robot_root_pose'][:,world]
    traj=dict(joint_names=m['robot_joint_names'],positions=d['robot_q'][:,world].tolist(),
              timestamps=((np.arange(len(vertices))+1)*m['physics_dt']).tolist(),
              base_trajectory=dict(positions=root[:,:3].tolist(),quats=root[:,[4,5,6,3]].tolist()),
              object_trajectories=dict(table=dict(positions=d['table_center'][:,world].tolist(),quats=np.tile([0,0,0,1],(len(vertices),1)).tolist())),
              deformables={})
    for half,name in enumerate(('cloth_stationary','cloth_moving')):
        traj['deformables'][name]=dict(indices=indices[half])
    metrics=dict(vertices=vertices,table_size=np.asarray(m['table_size']),table_offset=np.zeros(3))
    for half,name in enumerate(('stationary','moving')):
        selected=np.unique(indices[half]);metrics[name+'_height']=vertices[:,selected,2].mean(1)
    return dict(trajectory=traj),metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worlds',type=int,nargs='*')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reports=json.loads(args.summary.read_text())
    report=next((r for r in reports if Path(r['dataset']).resolve()==args.dataset.resolve()),None)
    if report is None:raise ValueError('Summary does not contain the selected dataset')
    m,d,c,g=load_dataset(args.dataset)
    hit=report['variants'].get('robot_source_lifting_work',report['variants']['work'])
    selected=args.worlds
    if selected is None:
        selected=hit['intended_half']['env_ids']+[e['env_id'] for e in report['events'] if e['cohort']=='reverse_half' and e['env_id'] not in hit['reverse_half']['env_ids']]
    robot=yourdfpy.URDF.load(str(ROOT/m['robot_urdf']),build_scene_graph=True,load_meshes=True)
    # Display-only hulls keep dense fingertip visuals from dominating rendering.
    # Contact/work tests use the recorded physics data, not these hulls.
    for name,mesh in list(robot.scene.geometry.items()):
        robot.scene.geometry[name]=mesh.convex_hull
    for world in selected:
        event=next(e for e in report['events'] if e['env_id']==world)
        end=event['frame'];touch=np.flatnonzero((d['force_magnitude_by_half'][:end+1,world,0].sum(1)>.01))
        first=int(touch[0]) if len(touch) else 0
        frames=np.unique(np.linspace(max(0,first-6),min(end+18,len(d['active'])-1),5).astype(int)).tolist()
        scene,metrics=make_scene(m,d,g,world)
        path=args.output/f'world_{world}_{event["cohort"]}.png'
        storyboard(scene,metrics,frames,path,f'{args.dataset.name} · world {world} · {event["design"]} · {event["cohort"]} · hand visual hulls',robot)
        print(world,frames,str(path),flush=True)


if __name__=='__main__':main()
