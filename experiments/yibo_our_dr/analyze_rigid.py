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
    (directory/'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
