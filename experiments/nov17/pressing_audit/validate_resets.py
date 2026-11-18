"""Replay proposed reset predicates on untouched, fully recorded first episodes.

Reports interception before a first clear half lift. Motion labels are geometry
cohorts, not human-certified success/failure or proof of a numerical artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_pressing import geometry_metrics, load_dataset, rotation
except ImportError:
    from analyze_pressing import geometry_metrics, load_dataset, rotation


def physical_metrics(metadata, trajectory, contacts, geometry):
    shape = trajectory['active'].shape
    frames, worlds = shape
    plate_count = trajectory['panel_pose'].shape[2]
    indices = (contacts['frame_idx'], contacts['env_id'], contacts['cloth_body'])
    downward = np.zeros((*shape, plate_count))
    upward = np.zeros_like(downward)
    robot = contacts['partner_type'] == 0
    table = ~robot
    force = contacts['force_vector'][:, 2]
    np.add.at(downward, tuple(x[robot] for x in indices), np.maximum(-force[robot], 0))
    np.add.at(upward, tuple(x[table] for x in indices), np.maximum(force[table], 0))
    # Opposed normal loads on the SAME physical plate, irrespective of half label.
    compression = np.minimum(downward, upward).sum(2)
    penetration = np.zeros(shape)
    anchor_error = np.zeros(shape)
    axis_error = np.zeros(shape)
    limit_error = np.zeros(shape)
    robot_gap = np.zeros(shape)
    loaded = robot & (contacts['force_magnitude'] > 1e-5)
    np.maximum.at(robot_gap, (indices[0][loaded], indices[1][loaded]),
                  np.maximum(-contacts['separation'][loaded], 0))
    for world, design in geometry.items():
        matrices = []
        for plate, geometry_plate in enumerate(design['plates']):
            pose = trajectory['panel_pose'][:, world, plate]
            matrix = rotation(pose[:, 3:])
            matrices.append(matrix)
            vertices = np.einsum('tij,vj->tvi', matrix, np.asarray(geometry_plate['vertices']))+pose[:, None, :3]
            penetration[:, world] = np.maximum(penetration[:, world],
                np.maximum(trajectory['table_top'][:, world]-vertices[:, :, 2].min(1), 0))
        if 'hinges' not in design:
            # Archived reference captures can still validate load and depth checks.
            # Do not reconstruct unrecorded joint settings or silently report zero.
            for quantity in (anchor_error, axis_error, limit_error):
                quantity[:, world] = np.nan
            continue
        for index, hinge in enumerate(design['hinges']):
            a, b = hinge['parent'], hinge['child']
            pa = trajectory['panel_pose'][:, world, a, :3]
            pb = trajectory['panel_pose'][:, world, b, :3]
            anchor = np.asarray(hinge['anchor'])
            wa = np.einsum('tij,j->ti', matrices[a], anchor)+pa
            wb = np.einsum('tij,j->ti', matrices[b], anchor)+pb
            anchor_error[:, world] = np.maximum(anchor_error[:, world], np.linalg.norm(wa-wb, axis=1))
            axis = np.asarray(hinge['axis'])
            aa = np.einsum('tij,j->ti', matrices[a], axis)
            ab = np.einsum('tij,j->ti', matrices[b], axis)
            angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(aa, ab), axis=1), (aa*ab).sum(1)))
            axis_error[:, world] = np.maximum(axis_error[:, world], angle)
            q = np.degrees(trajectory['hinge_q'][:, world, index])
            lo, hi = hinge['limits_deg']
            limit_error[:, world] = np.maximum(limit_error[:, world], np.maximum(np.maximum(lo-q, q-hi), 0))
    return dict(compression_n=compression, table_depth_m=penetration,
                anchor_error_m=anchor_error, axis_error_deg=axis_error,
                limit_error_deg=limit_error, robot_reported_depth_m=robot_gap)


def causal_mean(values, window):
    """Past and current physics steps only; zero before rollout start."""
    if window == 1:
        return values.copy()
    padded = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)], axis=0)
    ends = np.arange(1, len(values)+1)
    return (padded[ends]-padded[np.maximum(ends-window, 0)])/window


def first_crossing(mask):
    hit = mask.any(0)
    return np.where(hit, mask.argmax(0), len(mask))


def opening_events(metadata, trajectory, geometry):
    """Earliest 60deg half lift while the other stays approximately table-flat.

    All vertices must remain over the tabletop (2mm margin) up through the event.
    This separates edge maneuvers. Material signs distinguish comparison cohorts,
    without using force, penetration, constraint error or tested thresholds.
    """
    metrics = geometry_metrics(metadata, trajectory, geometry)
    active = trajectory['active'].astype(bool)
    inside = metrics['outside_vertex_fraction'] == 0
    prefix_inside = np.logical_and.accumulate(inside, axis=0)
    dims = np.asarray(metadata['dimensions'])
    times = np.arange(len(active))*metadata['physics_dt']
    first = []
    for half in (0, 1):
        other = 1-half
        candidate = active & prefix_inside & (times[:, None] >= .1)
        candidate &= metrics['tilt_deg'][:, :, half] >= 60
        candidate &= metrics['tilt_deg'][:, :, other] <= 20
        candidate &= metrics['mean_height'][:, :, other] <= dims[None, :, 2]/2+.015
        candidate &= metrics['mean_height'][:, :, half] >= dims[None, :, 2]/2+.02
        first.append(first_crossing(candidate))
    events = []
    for world in range(metadata['num_envs']):
        half = int(first[1][world] < first[0][world])
        frame = int(first[half][world])
        if frame == len(active):
            continue
        events.append(dict(env_id=world, frame=frame, time_s=(frame+1)*metadata['physics_dt'],
                           cohort='reverse_half' if half == 0 else 'intended_half',
                           design=metadata['designs'][world]))
    return events, active & inside


def evaluate(metadata, trajectory, physical, events, valid):
    predicates = {}
    for seconds in (0., .025, .05, .1):
        load = causal_mean(physical['compression_n'], max(1, round(seconds/metadata['physics_dt'])))
        for force in (.25, .5, 1., 2., 5., 10.):
            predicates[f'compression_{force:g}N_{seconds*1000:g}ms'] = load > force
    average_load = causal_mean(physical['compression_n'], max(1, round(.05/metadata['physics_dt'])))
    weights = np.asarray(metadata['masses_kg'])*9.81
    for multiple in (10, 25, 50, 100, 250):
        predicates[f'compression_{multiple}weights_50ms'] = average_load > multiple*weights[None]
    thickness = np.asarray(metadata['dimensions'])[:, 2]
    for fraction in (.05, .1, .25, .5):
        permitted_depth = np.maximum(.0001, fraction*thickness)
        predicates[f'table_{fraction:g}thickness_floor0.1mm'] = physical['table_depth_m'] > permitted_depth[None]
    for mm in (.1, .25, .5, 1., 2., 5.):
        predicates[f'table_{mm:g}mm'] = physical['table_depth_m'] > mm/1000
        if np.isfinite(physical['anchor_error_m']).all():
            predicates[f'anchor_{mm:g}mm'] = physical['anchor_error_m'] > mm/1000
        predicates[f'robot_reported_{mm:g}mm'] = physical['robot_reported_depth_m'] > mm/1000
    for degrees in (.25, .5, 1., 2., 5., 10.):
        if np.isfinite(physical['axis_error_deg']).all():
            predicates[f'axis_{degrees:g}deg'] = physical['axis_error_deg'] > degrees
            predicates[f'limit_{degrees:g}deg'] = physical['limit_error_deg'] > degrees
    for n in (.5, 1., 2., 5.):
        for mm in (.25, .5, 1., 2.):
            predicates[f'union_{n:g}N50ms_{mm:g}mm'] = (
                causal_mean(physical['compression_n'], max(1, round(.05/metadata['physics_dt']))) > n
            ) | (physical['table_depth_m'] > mm/1000)
    rows = []
    released = [t['env_id'] for t in metadata['terminals']
                if t['reasons'].get('released')
                and not any(t['reasons'].get(k) for k in ('fall','nonfinite','hand_far'))]
    for name, mask in predicates.items():
        first = first_crossing(mask & valid)
        row = dict(condition=name, worlds_triggered=int((first < len(valid)).sum()))
        row['natural_releases'] = dict(count=len(released),
            would_reset=sum(int(first[e] < len(valid)) for e in released))
        for cohort in ('reverse_half', 'intended_half'):
            subset = [e for e in events if e['cohort'] == cohort]
            hits = [e for e in subset if first[e['env_id']] <= e['frame']]
            row[cohort] = dict(count=len(subset), reset_before_lift=len(hits),
                              hit_envs=[e['env_id'] for e in hits],
                              reset_anytime=sum(int(first[e['env_id']] < len(valid)) for e in subset))
        rows.append(row)
    summaries = {}
    for name, values in physical.items():
        available = values[valid & np.isfinite(values)]
        summaries[name] = dict(available=bool(len(available)),
            all_valid_percentiles=np.percentile(available, [50, 90, 99, 100]).tolist() if len(available) else [])
    for event in events:
        world, frame = event['env_id'], event['frame']
        event['prefix_maxima'] = {k:(float(v[:frame+1, world].max()) if np.isfinite(v[:frame+1, world]).all() else None)
                                  for k,v in physical.items()}
        event['compression_50ms_prefix_max'] = float(causal_mean(physical['compression_n'][:, world:world+1], max(1, round(.05/metadata['physics_dt'])))[:frame+1].max())
    return rows, summaries


def analyze(directory):
    metadata, trajectory, contacts, geometry = load_dataset(directory)
    if metadata['status'] != 'complete':
        raise ValueError(f'Incomplete capture: {directory}')
    physical = physical_metrics(metadata, trajectory, contacts, geometry)
    events, valid = opening_events(metadata, trajectory, geometry)
    rows, summaries = evaluate(metadata, trajectory, physical, events, valid)
    return dict(dataset=str(directory), checkpoint=metadata['checkpoint'], num_envs=metadata['num_envs'],
                scope='Shadow reset predicates; unchanged first episodes. Cohorts are first clear material-half lifts, not certified good/bad manipulation. Table-plane depth valid only over tabletop; edge frames excluded. Robot depth is PhysX reported loaded-contact separation, not independent exact mesh intersection.',
                events=events, physical_summaries=summaries, conditions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    reports = [analyze(p) for p in args.datasets]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2)+'\n')
    for report in reports:
        print(Path(report['dataset']).name, report['checkpoint'], 'events',len(report['events']), flush=True)
        for row in report['conditions']:
            if row['condition'] in ('compression_1N_50ms','table_0.5mm','anchor_0.5mm','union_1N50ms_0.5mm','limit_1deg','axis_1deg','robot_reported_0.5mm'):
                print(row['condition'], row['reverse_half'], row['intended_half'], flush=True)


if __name__ == '__main__':
    main()
