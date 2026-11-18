"""Diagnose recorded cloth motion and normal contacts; never change the experiment.

Usage: python analyze_pressing.py RESULTS_DIRECTORY [RESULTS_DIRECTORY ...]
Writes per-world summaries, two illustrative geometry/force plots per dataset,
and a comparison JSON. Only the first episode's active physics frames are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
COLORS = ("#4795c7", "#ed983d")


def rotation(quaternion):
    """Recorded poses use wxyz; scipy uses xyzw."""
    return Rotation.from_quat(np.asarray(quaternion)[..., [1, 2, 3, 0]]).as_matrix()


def polygon_area(vertices):
    return sum(np.linalg.norm(np.cross(vertices[i]-vertices[0], vertices[i+1]-vertices[0])) / 2
               for i in range(1, len(vertices)-1))


def plate_properties(plate):
    vertices = np.asarray(plate["vertices"])
    polygon = vertices[:len(vertices)//2].copy()
    polygon[:, 2] = 0.
    areas = np.array([polygon_area(polygon[[0, i, i+1]]) for i in range(1, len(polygon)-1)])
    centers = np.array([polygon[[0, i, i+1]].mean(0) for i in range(1, len(polygon)-1)])
    center = np.average(centers, weights=areas, axis=0)
    return center, float(areas.sum()), int(center[0] > 0)


def load_dataset(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    if metadata.get('status', 'complete') != 'complete':
        raise ValueError(f'Capture is not complete; its NPZ files may still be writing: {directory}')
    with np.load(directory/"trajectory.npz") as archive:
        trajectory = {key: archive[key] for key in archive.files}
    with np.load(directory/"contacts.npz") as archive:
        contacts = {key: archive[key] for key in archive.files}
    geometry = {int(g["env_id"]): g for g in metadata["geometry"]}
    # A contact near a crease may lie slightly across x=0 due to solver tolerance.
    # Material identity follows its rigid plate, never that noisy contact coordinate.
    frame_count, world_count = trajectory["active"].shape
    owners = np.full((world_count, trajectory["panel_pose"].shape[2]), -1, dtype=int)
    for world, design in geometry.items():
        for plate_id, plate in enumerate(design["plates"]):
            rest_x = np.asarray(plate["vertices"])[:, 0]
            if rest_x.min() < -1e-8 and rest_x.max() > 1e-8:
                raise ValueError("A plate crosses the material split; body ownership cannot classify its contacts")
            owners[world, plate_id] = plate_properties(plate)[2]
    half = owners[contacts["env_id"], contacts["cloth_body"]]
    if np.any(half < 0):
        raise ValueError("Contact references an unrecognized cloth plate")
    bucket = ((contacts["frame_idx"]*world_count+contacts["env_id"])*2+contacts["partner_type"])*2+half
    shape = (frame_count, world_count, 2, 2)
    slots = int(np.prod(shape))
    vectors = contacts["force_vector"]
    trajectory["force_vector_by_half"] = np.stack([
        np.bincount(bucket, weights=vectors[:, axis], minlength=slots).reshape(shape) for axis in range(3)
    ], axis=-1)
    trajectory["force_magnitude_by_half"] = np.bincount(bucket, weights=contacts["force_magnitude"], minlength=slots).reshape(shape)
    trajectory["downward_force_by_half"] = np.bincount(bucket, weights=np.maximum(-vectors[:, 2], 0), minlength=slots).reshape(shape)
    return metadata, trajectory, contacts, geometry


def geometry_metrics(metadata, trajectory, geometry):
    """Area-weighted material normals retain sign: inversion is 180°, not zero."""
    count, worlds = trajectory["active"].shape
    normal = np.zeros((count, worlds, 2))
    height = np.zeros_like(normal)
    outside = np.zeros((count, worlds))
    size = np.asarray(metadata["table_size"])
    for world, design in geometry.items():
        area_sum = np.zeros(2)
        outside_count = np.zeros(count)
        vertex_count = 0
        for plate_id, plate in enumerate(design["plates"]):
            center, area, half = plate_properties(plate)
            pose = trajectory["panel_pose"][:, world, plate_id]
            matrix = rotation(pose[:, 3:])
            transformed_center = np.einsum("tij,j->ti", matrix, center)+pose[:, :3]
            normal[:, world, half] += area*matrix[:, 2, 2]
            height[:, world, half] += area*(transformed_center[:, 2]-trajectory["table_top"][:, world])
            area_sum[half] += area
            vertices = np.einsum("tij,vj->tvi", matrix, np.asarray(plate["vertices"]))+pose[:, None, :3]
            local = vertices-trajectory["table_center"][:, world, None]
            outside_count += (np.abs(local[:, :, :2]) > size[None, None, :2]/2+.002).any(2).sum(1)
            vertex_count += vertices.shape[1]
        normal[:, world] /= area_sum
        height[:, world] /= area_sum
        outside[:, world] = outside_count/vertex_count
    return dict(normal_up=normal, tilt_deg=np.degrees(np.arccos(np.clip(normal, -1, 1))),
                mean_height=height, outside_vertex_fraction=outside)


def force_summary(metadata, trajectory, metrics, world):
    valid = trajectory["active"][:, world].astype(bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return {"env_id": world, "active_frames": 0}
    force = trajectory["force_magnitude_by_half"][valid, world]
    downward = trajectory["downward_force_by_half"][valid, world]
    vector = trajectory["force_vector_by_half"][valid, world]
    tilt = metrics["tilt_deg"][valid, world]
    height = metrics["mean_height"][valid, world]
    weight = metadata["masses_kg"][world]*9.81
    # These nominate geometry witnesses, not successes or causal classifications.
    reverse = (tilt[:, 0] > 45) & (height[:, 0] > .02) & (height[:, 1] < .035)
    intended = (tilt[:, 0] < 15) & (height[:, 0] < .025) & (height[:, 1] > .025)
    return dict(env_id=world, design=metadata["designs"][world], active_frames=int(valid.sum()),
                duration_s=float(valid.sum()*metadata["physics_dt"]), weight_n=float(weight),
                robot_normal_peak_n=force[:, 0].max(0).tolist(),
                robot_downward_peak_n=downward[:, 0].max(0).tolist(),
                robot_normal_impulse_ns=(force[:, 0].sum(0)*metadata["physics_dt"]).tolist(),
                table_upward_peak_n=vector[:, 1, :, 2].max(0).tolist(),
                stationary_tilt_max_deg=float(tilt[:, 0].max()),
                stationary_normal_up_min=float(metrics["normal_up"][valid, world, 0].min()),
                stationary_center_max_above_table_m=float(height[:, 0].max()),
                reverse_pose_frame_fraction=float(reverse.mean()),
                intended_lift_frame_fraction=float(intended.mean()),
                outside_table_frame_fraction=float((metrics["outside_vertex_fraction"][valid, world] > 0).mean()),
                reverse_pose_frames=indices[reverse].tolist(), intended_lift_frames=indices[intended].tolist())


def select_witnesses(metadata, trajectory, metrics, summaries):
    selected = {}
    dt = metadata["physics_dt"]
    for label, field in (("reverse", "reverse_pose_frames"), ("intended", "intended_lift_frames")):
        candidates = []
        for summary in summaries:
            world = summary["env_id"]
            for frame in summary.get(field, []):
                if metrics["outside_vertex_fraction"][frame, world] > 0:
                    continue
                start = max(0, frame-int(.25/dt))
                press = float(trajectory["downward_force_by_half"][start:frame+1, world, 0].sum(axis=1).max())
                current_load = float(trajectory["force_magnitude_by_half"][frame, world, 0].sum())
                if label == "reverse":
                    score = 90-abs(metrics["tilt_deg"][frame, world, 0]-90)+min(press, 10.)
                else:
                    score = 1000*metrics["mean_height"][frame, world, 1]-metrics["tilt_deg"][frame, world, 0]
                score += 200*(current_load > .01)
                candidates.append((score, world, frame))
        if candidates:
            _, world, frame = max(candidates)
            selected[label] = dict(env_id=int(world), frame=int(frame),
                                   time_s=float((frame+1)*dt), design=metadata["designs"][world],
                                   stationary_tilt_deg=float(metrics["tilt_deg"][frame, world, 0]),
                                   stationary_normal_up=float(metrics["normal_up"][frame, world, 0]),
                                   mean_height_m=metrics["mean_height"][frame, world].tolist(),
                                   outside_vertex_fraction=float(metrics["outside_vertex_fraction"][frame, world]),
                                   robot_downward_force_n=trajectory["downward_force_by_half"][frame, world, 0].tolist(),
                                   table_upward_force_n=trajectory["force_vector_by_half"][frame, world, 1, :, 2].tolist())
    return selected


def force_distributions(directory, metadata, trajectory, metrics, summaries):
    """Pose-conditioned physics-frame distributions, excluding any table overhang.

    A preceding 100 ms window retains the pressing impulse that initiates motion.
    Frames are temporally correlated; these are diagnostics, not independent trials.
    """
    buckets = {name: {key: [] for key in ("downward_n", "normal_n", "downward_weight_multiple", "normal_weight_multiple", "worlds")}
               for name in ("reverse", "intended")}
    history = max(1, round(.1/metadata["physics_dt"]))
    for summary in summaries:
        world = summary["env_id"]
        weight = metadata["masses_kg"][world]*9.81
        down = trajectory["downward_force_by_half"][:, world, 0].sum(1)
        normal = trajectory["force_magnitude_by_half"][:, world, 0].sum(1)
        def preceding_max(values):
            return np.maximum.reduce([np.r_[np.zeros(k), values[:len(values)-k]] for k in range(min(history, len(values)))])
        recent_down, recent_normal = preceding_max(down), preceding_max(normal)
        for name, field in (("reverse", "reverse_pose_frames"), ("intended", "intended_lift_frames")):
            ids = np.asarray(summary.get(field, []), dtype=int)
            # Require the entire short force window to remain over the table.
            edge = preceding_max(metrics["outside_vertex_fraction"][:, world])
            ids = ids[(edge[ids] == 0) & (recent_normal[ids] > .01)]
            buckets[name]["downward_n"].extend(recent_down[ids].tolist())
            buckets[name]["normal_n"].extend(recent_normal[ids].tolist())
            buckets[name]["downward_weight_multiple"].extend((recent_down[ids]/weight).tolist())
            buckets[name]["normal_weight_multiple"].extend((recent_normal[ids]/weight).tolist())
            if len(ids): buckets[name]["worlds"].append(world)
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, data in buckets.items():
        entry = dict(physics_frames=len(data["downward_n"]), worlds=data["worlds"])
        for key in ("downward_n", "normal_n", "downward_weight_multiple", "normal_weight_multiple"):
            values = np.asarray(data[key])
            entry[key] = dict(zip(("min", "p50", "p90", "p99", "max"), np.quantile(values, [0, .5, .9, .99, 1]).tolist())) if len(values) else None
        result[name] = entry
        for ax, key in zip(axes, ("downward_n", "downward_weight_multiple")):
            values = np.sort(data[key])
            if len(values): ax.plot(values, np.arange(1, len(values)+1)/len(values), label=f"{name}: {len(data['worlds'])} worlds")
    for ax, units in zip(axes, ("Robot downward normal force (N)", "Robot downward force / sheet weight")):
        ax.set_xlabel(units); ax.set_ylabel("Cumulative fraction of selected physics frames")
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle(f"{directory.name} · load distributions during geometric pose classes\n"
                 "Peak in preceding 100 ms; entire window within tabletop; robot normal load >0.01 N", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .88)); fig.savefig(directory/"force_distributions.png", dpi=140); plt.close(fig)
    return dict(window_s=.1, minimum_robot_normal_load_n=.01, excludes_table_overhang=True,
                sampling="Temporally correlated physics frames within first episodes; descriptive only.", classes=result)


def draw_geometry(directory, metadata, trajectory, contacts, geometry, witness, label):
    world, frame = witness["env_id"], witness["frame"]
    dt = metadata["physics_dt"]
    last = int(np.flatnonzero(trajectory["active"][:, world])[-1])
    frames = np.unique(np.clip([frame-int(.4/dt), frame-int(.15/dt), frame, frame+int(.15/dt)], 0, last))
    all_vertices = []
    panels = []
    for f in frames:
        scene = []
        for p, plate in enumerate(geometry[world]["plates"]):
            pose = trajectory["panel_pose"][f, world, p]
            vertices = np.asarray(plate["vertices"]) @ rotation(pose[3:]).T+pose[:3]
            scene.append((vertices, np.asarray(plate["triangles"]), plate_properties(plate)[2]))
            all_vertices.append(vertices)
        panels.append(scene)
    bounds = np.concatenate(all_vertices)
    center = (bounds.max(0)+bounds.min(0))/2
    radius = max(.12, np.ptp(bounds, axis=0).max()/2+.04)
    fig = plt.figure(figsize=(4.8*len(frames), 5.5))
    for i, (f, scene) in enumerate(zip(frames, panels)):
        ax = fig.add_subplot(1, len(frames), i+1, projection="3d")
        ax.computed_zorder = False
        tc = trajectory["table_center"][f, world]
        sx, sy = np.asarray(metadata["table_size"])[:2]/2
        tz = trajectory["table_top"][f, world]
        table = np.array([[tc[0]-sx, tc[1]-sy, tz], [tc[0]+sx, tc[1]-sy, tz],
                          [tc[0]+sx, tc[1]+sy, tz], [tc[0]-sx, tc[1]+sy, tz]])
        ax.add_collection3d(Poly3DCollection([table], facecolors="#cccccc", alpha=.25, edgecolors="#777777", zorder=0))
        for vertices, faces, half in scene:
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors=COLORS[half], edgecolors="#333333", linewidths=.3, zorder=1))
        fingers = trajectory["fingerpad_positions"][f, world]
        ax.scatter(*fingers.T, color="#333333", s=25, marker="o", depthshade=False, zorder=3)
        mask = (contacts["frame_idx"] == f) & (contacts["env_id"] == world) & (contacts["force_magnitude"] > 1e-5)
        p = contacts["position"][mask]
        force = contacts["force_vector"][mask]
        robot = contacts["partner_type"][mask] == 0
        for is_robot, color in ((True, "#d62f8b"), (False, "#278f4d")):
            keep = robot == is_robot
            if not keep.any():
                continue
            ax.scatter(*p[keep].T, color=color, s=15, depthshade=False, zorder=5)
            # Fixed 2 cm direction arrows; numeric force magnitudes are in the traces.
            norm = np.linalg.norm(force[keep], axis=1)
            arrows = force[keep]/np.maximum(norm[:, None], 1e-12)*.02
            ax.quiver(*p[keep].T, *arrows.T, color=color, arrow_length_ratio=.25, zorder=6)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(min(tz-.015, bounds[:, 2].min()-.01), max(tz+.16, bounds[:, 2].max()+.04)))
        ax.set_box_aspect((1, 1, .8)); ax.view_init(24, -65)
        ax.set_title(f"Frame {f} · {(f+1)*dt:.3f} s", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{directory.name} · {label} pose witness · world {world} · {witness['design']}", fontsize=14)
    fig.text(.5, .025, "Blue: stationary material · Orange: moving material · Gray dots: finger pads · Pink: robot contact · Green: table contact\n"
             "Contact markers overlay surfaces for visibility. Arrows: normal force ON cloth, fixed length (not magnitude).", ha="center", fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.15, wspace=.02)
    fig.savefig(directory/f"{label}_geometry.png", dpi=140); plt.close(fig)


def draw_traces(directory, metadata, trajectory, metrics, witness, label):
    world, frame = witness["env_id"], witness["frame"]
    valid = trajectory["active"][:, world].astype(bool)
    times = (np.flatnonzero(valid)+1)*metadata["physics_dt"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for half, name in enumerate(("Stationary", "Moving")):
        color = COLORS[half]
        axes[0].plot(times, trajectory["force_magnitude_by_half"][valid, world, 0, half], color=color, label=name+" robot normal load")
        axes[0].plot(times, trajectory["downward_force_by_half"][valid, world, 0, half], color=color, linestyle="--", label=name+" downward component")
        axes[1].plot(times, trajectory["force_vector_by_half"][valid, world, 1, half, 2], color=color, label=name+" table force z")
        axes[2].plot(times, metrics["tilt_deg"][valid, world, half], color=color, label=name+" material tilt")
        axes[3].plot(times, 1000*metrics["mean_height"][valid, world, half], color=color, label=name+" midplane height")
    axes[1].axhline(metadata["masses_kg"][world]*9.81, color="black", linestyle=":", label="Whole-sheet weight")
    axes[2].set_ylim(-3, 183)
    for ax, unit in zip(axes, ("Robot load (N)", "Table load (N)", "Tilt from up (°)", "Above table (mm)")):
        ax.axvline((frame+1)*metadata["physics_dt"], color="#a53333", linewidth=1, alpha=.8)
        ax.set_ylabel(unit); ax.grid(alpha=.2); ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[3].set_xlabel("Time in first episode (s)")
    fig.suptitle(f"{directory.name} · world {world} · {witness['design']} · {label} witness\n"
                 "Normal contact loads only; friction is not included. Tilt preserves material orientation through 180°.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(directory/f"{label}_traces.png", dpi=140); plt.close(fig)


def analyze(directory):
    directory = Path(directory)
    metadata, trajectory, contacts, geometry = load_dataset(directory)
    metrics = geometry_metrics(metadata, trajectory, geometry)
    summaries = [force_summary(metadata, trajectory, metrics, world) for world in range(metadata["num_envs"])]
    witnesses = select_witnesses(metadata, trajectory, metrics, summaries)
    distributions = force_distributions(directory, metadata, trajectory, metrics, summaries)
    report = dict(dataset=str(directory), checkpoint=metadata["checkpoint"], status=metadata["status"],
                  half_order=["stationary (rest x <= 0)", "moving (rest x > 0)"],
                  table_bounds="Axis aligned; table rotations are fixed identity in these runs. 2 mm footprint margin.",
                  material_half_assignment="Forces rebuilt from raw contacts using canonical cloth_body plate ownership; rest-x contact signs are not used.",
                  scope="First active episode only; normal contact forces only. Witness thresholds nominate poses, not reward success or causality.",
                  worlds=summaries, witnesses=witnesses, force_distributions=distributions)
    (directory/"analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    for label in ("reverse", "intended"):
        if label not in witnesses:
            for suffix in ("geometry", "traces"):
                (directory/f"{label}_{suffix}.png").unlink(missing_ok=True)
    for label, witness in witnesses.items():
        draw_geometry(directory, metadata, trajectory, contacts, geometry, witness, label)
        draw_traces(directory, metadata, trajectory, metrics, witness, label)
    print(json.dumps(dict(dataset=str(directory), witnesses=witnesses), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--comparison", type=Path, help="Optional combined JSON for multiple datasets")
    args = parser.parse_args()
    reports = [analyze(path) for path in args.datasets]
    if args.comparison:
        args.comparison.parent.mkdir(parents=True, exist_ok=True)
        compact = [{**report, "worlds": [
            {key: value for key, value in world.items() if key not in ("reverse_pose_frames", "intended_lift_frames")}
            for world in report["worlds"]]} for report in reports]
        args.comparison.write_text(json.dumps(compact, indent=2)+"\n")


if __name__ == "__main__":
    main()
