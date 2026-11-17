"""Decode saved training HTMLs and render measured cloth/robot pose storyboards.

Run with the kitless isaaclab3 Python. Captures contain geometry and joint poses,
not contacts: proximity and cloth movement cannot establish contact force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
JOBS = ("710887", "710889", "711271", "711273")


def read_capture(path):
    match = re.search(r'<script id="scene-json" type="application/json">(.*?)</script>',
                      Path(path).read_text(), re.S)
    if match is None:
        raise ValueError(f"No scene JSON in {path}")
    return json.loads(match.group(1))


def measure(scene):
    traj = scene["trajectory"]
    cloth = traj["deformables"]
    vertices = np.asarray(cloth["cloth_moving"]["vertices"])
    mi = np.unique(cloth["cloth_moving"]["indices"])
    si = np.unique(cloth["cloth_stationary"]["indices"])
    moving, stationary = vertices[:, mi], vertices[:, si]
    table = next(r for r in scene["robots"] if r["name"] == "table")
    urdf = ET.fromstring(table["urdf_text"])
    visual = urdf.find(".//visual")
    size = np.fromstring(visual.find("geometry/box").attrib["size"], sep=" ")
    origin = visual.find("origin")
    offset = np.zeros(3) if origin is None else np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    tp = traj["object_trajectories"]["table"]
    tr = Rotation.from_quat(tp["quats"]).as_matrix()
    # Table-local positions support arbitrary captured table yaw/translation.
    local = np.einsum("tji,tvj->tvi", tr, vertices-np.asarray(tp["positions"])[:, None]) - offset
    top = size[2]/2
    center = vertices.mean(1)
    reset = np.r_[False, np.linalg.norm(np.diff(center, axis=0), axis=1) > .07]
    return dict(vertices=vertices, moving=moving, stationary=stationary,
                moving_height=moving[:,:,2].mean(1), stationary_height=stationary[:,:,2].mean(1),
                moving_span=np.ptp(moving[:,:,2],axis=1), stationary_span=np.ptp(stationary[:,:,2],axis=1),
                edge_fraction=(np.abs(local[:,:,:2]) > size[None,None,:2]/2+.002).any(2).mean(1),
                lowest_above_table=local[:,:,2].min(1)-top,
                table_size=size, table_offset=offset, reset=reset)


def transform(pose):
    out = np.eye(4)
    out[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix()
    out[:3,3] = pose[:3]
    return out


def stationary_support(scene, frame, tolerance=.002):
    """Exact planar area near the table for a single-hinge rectangular capture.

    Support here means bottom-face geometry within tolerance of the tabletop,
    clipped to the tabletop footprint; it is not a measured contact area.
    """
    m = measure(scene)
    stationary = m["stationary"]
    if stationary.shape[1] != 8:
        raise ValueError("This witness calculation requires one rectangular stationary plate")
    bottom, top = stationary[frame,:4], stationary[frame,4:]
    mid = (bottom+top)/2
    normal = (top-bottom).mean(0)
    thickness = np.linalg.norm(normal)
    normal /= thickness
    t = scene["trajectory"]["object_trajectories"]["table"]
    rotation = Rotation.from_quat(t["quats"][frame]).as_matrix()
    table_up = rotation[:,2]
    center = np.asarray(t["positions"][frame]) + rotation @ m["table_offset"]
    local = (bottom-center) @ rotation
    size = m["table_size"]
    table_top = size[2]/2
    polygon = local.copy()
    for axis, limit, sense in ((2,table_top+tolerance,1),(2,table_top-tolerance,-1),
                               (0,size[0]/2,1),(0,-size[0]/2,-1),
                               (1,size[1]/2,1),(1,-size[1]/2,-1)):
        clipped=[]
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            da,db=sense*(a[axis]-limit),sense*(b[axis]-limit)
            if da <= 0: clipped.append(a)
            if (da <= 0) != (db <= 0): clipped.append(a+(b-a)*da/(da-db))
        polygon=np.asarray(clipped).reshape(-1,3)
        if not len(polygon): break
    def area(poly):
        if len(poly)<3: return 0.
        return sum(np.linalg.norm(np.cross(poly[i]-poly[0],poly[i+1]-poly[0]))/2 for i in range(1,len(poly)-1))
    mid0=(stationary[0,:4]+stationary[0,4:])/2
    toward = m["moving"][0].mean(0)-mid0.mean(0)
    far = np.argsort(mid0 @ toward)[:2]
    mid_local=(mid-center) @ rotation
    cosine=float(normal @ table_up)
    return dict(frame=int(frame),timestamp_s=scene["trajectory"]["timestamps"][frame],
                normal_up_cosine=cosine,tilt_degrees=float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
                thickness_m=float(thickness),centerline_minus_tabletop_minus_half_thickness_m=float(mid_local[:,2].mean()-table_top-thickness/2),
                far_edge_midplane_above_table_m=float(mid_local[far,2].mean()-table_top),
                support_area_fraction=float(area(polygon)/area(local)),support_height_tolerance_m=tolerance,
                vertices_outside_table_fraction=float(m["edge_fraction"][frame]))


def storyboard(scene, metrics, frames, path, title, robot):
    traj = scene["trajectory"]
    fig = plt.figure(figsize=(16, 7.7), facecolor="white")
    cloth = traj["deformables"]
    allv = metrics["vertices"][frames]
    center = allv.mean((0,1))
    radius = max(.12, np.ptp(allv,axis=(0,1)).max()/2+.055)
    for j, frame in enumerate(frames):
        ax=fig.add_subplot(2,len(frames),j+1,projection="3d")
        ax2=fig.add_subplot(2,len(frames),len(frames)+j+1,projection="3d")
        robot.update_cfg(dict(zip(traj["joint_names"],traj["positions"][frame])))
        base=traj["base_trajectory"]
        world=transform(base["positions"][frame]+base["quats"][frame])
        meshes=[]
        for node in robot.scene.graph.nodes_geometry:
            matrix, name=robot.scene.graph[node]
            mesh=robot.scene.geometry[name]
            v=(world@matrix@np.c_[mesh.vertices,np.ones(len(mesh.vertices))].T).T[:,:3]
            # Render geometry near the cloth; exclude the arm/base outside this view.
            faces=mesh.faces
            centroids=v[faces].mean(1)
            keep=np.all(np.abs(centroids-center)<[radius*1.25,radius*1.25,radius*1.8],axis=1)
            if keep.any():
                meshes.append(v[faces[keep]])
        tp=traj["object_trajectories"]["table"]
        tablemat=transform(tp["positions"][frame]+tp["quats"][frame])
        sx,sy,sz=metrics["table_size"]/2
        corners=np.array([[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])+metrics["table_offset"]
        table=(tablemat@np.c_[corners,np.ones(4)].T).T[:,:3]
        for a,elev,azim in ((ax,27,-60),(ax2,7,-90)):
            a.add_collection3d(Poly3DCollection([table],facecolors="#dededb",edgecolors="#999999",alpha=.25))
            for name,color in (("cloth_stationary","#4b99d3"),("cloth_moving","#ed983d")):
                tris=np.array(cloth[name]["indices"]).reshape(-1,3)
                a.add_collection3d(Poly3DCollection(metrics["vertices"][frame][tris],facecolors=color,edgecolors="#303030",linewidths=.35))
            for mesh in meshes:
                a.add_collection3d(Poly3DCollection(mesh,facecolors="#676b70",edgecolors="none",alpha=.83))
            a.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                  zlim=(min(center[2]-.06,table[:,2].min()-.04),center[2]+radius*1.4))
            a.set_box_aspect((1,1,.8));a.view_init(elev,azim)
            a.set_xlabel("x (m)",fontsize=7);a.set_ylabel("y (m)",fontsize=7);a.set_zlabel("z (m)",fontsize=7)
            a.tick_params(labelsize=6)
        ax.set_title(f"Frame {frame} · {traj['timestamps'][frame]:.2f} s\n"
                     f"mean z: orange {metrics['moving_height'][frame]:.3f}, blue {metrics['stationary_height'][frame]:.3f} m",fontsize=9)
    fig.suptitle(title+"\nActual captured plate geometry + robot URDF forward kinematics",fontsize=15)
    fig.text(.5,.015,"Orange: designated moving half · Blue: designated stationary half · Gray: robot hand / table. Poses do not measure contact forces.",ha="center",fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,bottom=.05,top=.86,wspace=.05,hspace=.02)
    fig.savefig(path,dpi=125);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"debug_outputs/pressing_audit_20261117")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--jobs",nargs="+",default=JOBS)
    parser.add_argument("--support-capture",type=Path,help="Measure one single-hinge HTML's stationary support instead of scanning runs")
    parser.add_argument("--support-frame",type=int,default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.support_capture is not None:
        print(json.dumps(stationary_support(read_capture(args.support_capture),args.support_frame),indent=2))
        return
    robot=None
    if args.render:
        import yourdfpy
        robot=yourdfpy.URDF.load(str(ROOT/"assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"),build_scene_graph=True,load_meshes=True)
        if len(robot.scene.geometry) < 25:
            raise ValueError("Robot meshes did not load; refusing to render a misleading hand-free storyboard")
    results=[]
    for job in args.jobs:
        run=next((ROOT/"debug_outputs/train_logs/nov").glob("*_"+job))
        for folder in sorted((run/"interactive_viewer").iterdir()):
            paths=sorted(folder.glob("*.html"))
            if not paths: continue
            path=paths[-1];scene=read_capture(path);m=measure(scene)
            # Lift geometry alone nominates frames; no claimed pressing classification.
            score=np.maximum(m["moving_span"],m["stationary_span"])
            eligible=np.ones(len(score),bool)
            for reset in np.flatnonzero(m["reset"]): eligible[max(0,reset-2):reset+3]=False
            peak=int(np.argmax(np.where(eligible,score,-1)))
            start=max([0]+[int(x) for x in np.flatnonzero(m["reset"][:peak+1])])
            frames=sorted(set([max(start,peak-60),max(start,peak-30),peak,min(len(score)-1,peak+30)]))
            record=dict(job=job,design=folder.name,capture=str(path.relative_to(ROOT)),peak_frame=peak,storyboard_frames=frames,
                        reset_candidates=np.flatnonzero(m["reset"]).tolist(),
                        moving_z_span_max_m=float(m["moving_span"].max()),stationary_z_span_max_m=float(m["stationary_span"].max()),
                        maximum_outside_table_fraction=float(m["edge_fraction"].max()),
                        minimum_vertex_above_table_m=float(m["lowest_above_table"].min()),
                        mean_z_at_peak_m=dict(moving=float(m["moving_height"][peak]),stationary=float(m["stationary_height"][peak])))
            print(json.dumps(record),flush=True);results.append(record)
            if args.render:
                storyboard(scene,m,frames,args.output/f"{job}_{folder.name}.png",f"Job {job} · {folder.name} · latest saved capture",robot)
    (args.output/"capture_metrics.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    main()
