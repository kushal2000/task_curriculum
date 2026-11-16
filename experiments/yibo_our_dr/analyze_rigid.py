"""Summarize paired first-episode rigid ablations using physical pre-reset metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(directory):
    cases = {}
    for path in sorted(directory.glob('*_seed*.json')):
        if path.name.endswith(('.config.json', '.readback.json')):
            continue
        result = json.loads(path.read_text())
        if 'all' in result:
            cases[result['case']] = result
    baseline = cases.get('hybrid')
    report = []
    for name, data in cases.items():
        row = {'case': name, **data['all'], 'seed': data['seed'], 'by_design': data['by_design']}
        per = data['per_env']
        row['motion_medians'] = {key: statistics.median(per[key]) for key in
            ('max_cloth_speed_mps', 'max_joint_speed_radps', 'max_palm_motion_m', 'max_cloth_xy_drift_m')}
        row['high_speed_episodes_gt_2mps'] = sum(value > 2 for value in per['max_cloth_speed_mps'])
        row['large_drift_episodes_gt_10cm'] = sum(value > .1 for value in per['max_cloth_xy_drift_m'])
        at_fold = [error < data['fold_tolerance_m'] and footprint < data['max_folded_footprint']
                   for error, footprint in zip(per['terminal_error'], per['terminal_footprint'])]
        row['failure_classes'] = {
            'never_entered_fold': sum(not value for value in per['folded']),
            'fold_entered_then_lost_without_release': sum(folded and not released and not valid
                for folded, released, valid in zip(per['folded'], per['released'], at_fold)),
            'terminal_fold_valid_but_no_release': sum(folded and not released and valid
                for folded, released, valid in zip(per['folded'], per['released'], at_fold)),
        }

        def cohort(indices):
            return {'n': len(indices), 'fold_entries': sum(per['folded'][i] for i in indices),
                    'full_releases': sum(per['released'][i] for i in indices),
                    'median_best_error_m': statistics.median(per['best_error'][i] for i in indices) if indices else None,
                    'median_max_speed_mps': statistics.median(per['max_cloth_speed_mps'][i] for i in indices) if indices else None}

        sides = [max(dimensions[:2]) for dimensions in data['initial']['dimensions']]
        density = data['initial']['density']
        row['by_size'] = {label: cohort([i for i, value in enumerate(sides) if low < value <= high])
                         for label, low, high in [('max_side_10_15cm', .09, .15),
                            ('max_side_15_20cm', .15, .20), ('max_side_20_25cm', .20, .25),
                            ('max_side_over_25cm', .25, float('inf'))]}
        row['by_density'] = {label: cohort([i for i, value in enumerate(density) if low < value <= high])
                            for label, low, high in [('up_to_100g_m2', 0., .1),
                                ('100_to_500g_m2', .1, .5), ('over_500g_m2', .5, float('inf'))]}
        if baseline and data['seed'] == baseline['seed']:
            a, b = baseline['per_env']['folded'], per['folded']
            rescued = sum(not x and y for x, y in zip(a, b))
            lost = sum(x and not y for x, y in zip(a, b))
            discordant = rescued + lost
            p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(rescued, lost)+1)) / 2**discordant) if discordant else 1.
            row.update(rescued_folds=rescued, lost_folds=lost, paired_exact_p_unadjusted=p)
            for key in ('joint_pos', 'joint_vel', 'points', 'dimensions', 'density'):
                def flatten(value):
                    return [item for child in value for item in flatten(child)] if isinstance(value, list) else [value]
                one, two = flatten(baseline['initial'][key]), flatten(data['initial'][key])
                row[f'initial_max_difference_{key}'] = max(abs(x-y) for x, y in zip(one, two))
        report.append(row)
    summary = {'cases': report, 'limitations': [
        '64 first episodes per case, one seed: estimates are preliminary.',
        'Exact paired p-values are exploratory and unadjusted for multiple comparisons.',
        'Cases restore one parameter group on hybrid; yibo restores many and is only a reference.',
        'Explicit seeds are reset after policy construction; these baselines supersede earlier seed42-reset diagnostics.',
        'contacts_yibo keeps modern mass/gain DR and changes only authored contact mixing, friction, rest/contact offsets.']}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Controlled rigid-cloth ablations', '',
        'Physical fold criterion: corner error <4 cm and footprint <0.65; full success also requires 10 cm whole-robot release.', '',
        '| Case | Fold entries | Full release | Falls | Median best error | Rescued / lost vs hybrid |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report:
        lines.append(f"| {row['case']} | {row['fold_entries']}/{row['n']} | {row['full_releases']}/{row['n']} | "
                     f"{row['termination_reasons'].get('fall',0)} | {100*row['median_best_fold_error_m']:.2f} cm | "
                     f"{row.get('rescued_folds','—')} / {row.get('lost_folds','—')} |")
    lines.extend(['', *summary['limitations'], ''])
    lines.extend(['', '## Motion and failure diagnostics', '',
        '| Case | Median peak cloth speed | Median maximum XY drift | Never folded | Fold lost | Fold valid at end, no release |',
        '|---|---:|---:|---:|---:|---:|'])
    for row in report:
        motion, failure = row['motion_medians'], row['failure_classes']
        lines.append(f"| {row['case']} | {motion['max_cloth_speed_mps']:.2f} m/s | "
                     f"{100*motion['max_cloth_xy_drift_m']:.1f} cm | {failure['never_entered_fold']} | "
                     f"{failure['fold_entered_then_lost_without_release']} | {failure['terminal_fold_valid_but_no_release']} |")
    lines.extend(['', 'Size/density cohorts and per-design results are in summary.json. '
                  'These within-case cohorts are descriptive; the one-factor cases provide the controlled comparisons.', ''])
    (directory/'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
