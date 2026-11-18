"""Audit VBD stability probes and report local diagnostic/runtime Pareto frontiers.

No weighted quality score or globally optimal solver is selected. Frontiers require
completed, numerically verified runs, complete declared diagnostics, matched input
hashes and comparable protocols. A contact-record plane is only a proxy: missing
records are unknown observations, never zero penetration. Particle speed and
one-sided stretch peaks are stability diagnostics, not physical accuracy oracles.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


QUALITY_METRICS = (
    'robot_record_contact_plane_crossing_m', 'table_box_center_depth_m',
    'particle_speed_m_s', 'edge_stretch_ratio', 'p95_edge_stretch_ratio',
)
KNOWN_METRICS = QUALITY_METRICS + ('late_particle_speed_m_s',)
DEFAULT_PROBES = ('settle', 'scripted', 'fold_release')
SHA_FIELDS = ('initial_state_sha256', 'action_trace_sha256', 'physical_protocol_sha256')
NUMERICAL_ERROR = re.compile(r'non[ -]?finite|\bnan\b|\binfinite\b|overflow|exceeds.*capacity|invalid reward/state|divergen', re.I)


def metadata(run, name, default=None):
    for source in (run, run.get('benchmark', {}), run.get('protocol', {})):
        if isinstance(source, dict) and name in source:
            return source[name]
    return default


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def _values(value, n):
    values = list(value) if isinstance(value, list) else []
    return [(values[i] if i < len(values) else None) for i in range(n)], len(values)


def metric_summary(run, name, n):
    peaks = run.get('per_env_peak_metrics', {})
    values, length = _values(peaks.get(name), n)
    counts, count_length = _values(peaks.get('robot_record_count'), n)
    is_robot = name.startswith('robot_record_')
    known = [finite_number(v) and (not is_robot or (finite_number(c) and c > 0))
             for v, c in zip(values, counts)]
    observed = [float(value) for value, valid in zip(values, known) if valid]
    invalid = [i for i, value in enumerate(values)
               if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value)]
    output = {'expected_episodes': n, 'present_values': length, 'observed_episodes': len(observed),
              'missing_or_unobserved_episodes': n-len(observed),
              'observed_fraction': len(observed)/n if n else None,
              'median': statistics.median(observed) if observed else None,
              'p95': quantile(observed, .95), 'max': max(observed) if observed else None,
              'nonfinite_episode_ids': invalid,
              'complete': n > 0 and length == n and len(observed) == n}
    if is_robot:
        covered = [i for i, c in enumerate(counts) if finite_number(c) and c > 0]
        output.update(record_covered_episodes=len(covered), record_counts_present=count_length,
                      record_covered_episode_ids=covered,
                      interpretation='Cached proxy contact-plane crossing only; not exact robot-mesh penetration.')
    elif name == 'table_box_center_depth_m':
        output['interpretation'] = 'Exact node-to-native-table-box depth at sampled frames; not triangle-interior or inter-frame penetration.'
    return output


def limit_map(config, probe, metrics):
    config = config.get('quality_limits', config) if isinstance(config, dict) else {}
    global_limits = config.get('global', {k: v for k, v in config.items() if k != 'probes'})
    specific = config.get('probes', {}).get(probe, {})
    result = {**global_limits, **specific}
    for name, bound in result.items():
        if name not in KNOWN_METRICS or not finite_number(bound) or bound < 0:
            raise ValueError(f'Invalid quality limit {name!r}: {bound!r}')
    return {name: float(value) for name, value in result.items() if name in metrics}


def normalize_run(run, path, *, limits=None, statistic='max', metrics=None):
    n = int(run.get('num_envs') or 0)
    probe = str(metadata(run, 'probe', 'policy'))
    selected = list(metrics or metadata(run, 'quality_metrics', QUALITY_METRICS))
    if not selected or len(selected) != len(set(selected)) or set(selected)-set(KNOWN_METRICS):
        raise ValueError(f'{path}: quality_metrics must be a nonempty unique subset of {KNOWN_METRICS}')
    diagnostic = {name: metric_summary(run, name, max(n, 0)) for name in KNOWN_METRICS}
    error = str(run.get('error') or '')
    reasons = run.get('termination_reasons') or {}
    numerical_failure = (run.get('finite_checks_passed') is False or run.get('overflow_checks_passed') is False
        or bool(NUMERICAL_ERROR.search(error)) or any(d['nonfinite_episode_ids'] for d in diagnostic.values())
        or any(value > 0 for key, value in reasons.items()
               if ('nonfinite' in key or 'overflow' in key) and finite_number(value)))
    raw_status = str(run.get('status', 'unknown'))
    completed, censored = run.get('completed_episodes'), run.get('censored')
    partial = raw_status in ('running', 'partial', 'interrupted') or (
        raw_status in ('complete', 'completed') and ((finite_number(censored) and censored > 0)
        or (finite_number(completed) and n > 0 and completed < n)))
    if numerical_failure:
        status = 'invalid_numerical'
    elif partial:
        status = 'partial'
    elif raw_status in ('complete', 'completed'):
        status = 'complete'
    elif raw_status == 'failed':
        status = 'failed'
    else:
        status = 'unknown'
    verified = run.get('finite_checks_passed') is True and run.get('overflow_checks_passed') is True
    elapsed = run.get('rollout_seconds', run.get('batch_rollout_seconds'))
    steps = run.get('steps_run')
    active = metadata(run, 'active_transitions', metadata(run, 'active_policy_transitions'))
    if finite_number(active) and active >= 0:
        transitions, basis = active, 'active_transitions'
    elif n > 0 and finite_number(steps) and steps >= 0:
        transitions, basis = n*steps, 'num_envs_times_steps'
    else:
        transitions, basis = None, 'missing'
    throughput = transitions/elapsed if finite_number(elapsed) and elapsed > 0 and transitions is not None else None
    shared_batch = metadata(run, 'batchid', metadata(run, 'batch_id')) is not None or 'batch_rollout_seconds' in run
    runtime = {'policy_sps': throughput, 'counted_transitions': transitions, 'count_basis': basis,
               'rollout_seconds': elapsed if finite_number(elapsed) else None,
               'scope': 'shared_batch_probe_attribution' if shared_batch else 'standalone_run',
               'batchid': metadata(run, 'batchid', metadata(run, 'batch_id')),
               'batch_rollout_seconds': metadata(run, 'batch_rollout_seconds'),
               'batch_num_envs': metadata(run, 'batch_num_envs'),
               'batch_steps_run': metadata(run, 'batch_steps_run'),
               'physics_seconds': run.get('physics_seconds', run.get('batch_physics_seconds'))}
    hashes = {name: metadata(run, name) for name in SHA_FIELDS}
    missing_hashes = [name for name, value in hashes.items()
                      if not isinstance(value, str) or not re.fullmatch('[0-9a-fA-F]{64}', value)]
    cfg = run.get('config') or run.get('resolved_config') or {}
    dt = metadata(run, 'policy_dt')
    if dt is None:
        sim_dt, decimation = cfg.get('sim', {}).get('dt'), cfg.get('decimation')
        if finite_number(sim_dt) and finite_number(decimation):
            dt = sim_dt*decimation
    clock_required = probe in DEFAULT_PROBES
    simulated = metadata(run, 'simulated_seconds')
    expected_seconds = dt*steps if finite_number(dt) and finite_number(steps) else None
    clock_errors = []
    if clock_required:
        if metadata(run, 'physics_step_duration_checked') is not True:
            clock_errors.append('native_physics_step_duration_not_verified')
        if not finite_number(simulated) or simulated <= 0:
            clock_errors.append('missing_or_invalid_simulated_seconds')
        elif expected_seconds is None or not math.isclose(simulated, expected_seconds, rel_tol=1e-6, abs_tol=1e-7):
            clock_errors.append('simulated_duration_does_not_match_policy_clock')
    context = {'material': run.get('material'), 'probe': probe, 'seed': run.get('seed'),
               'num_envs': n, 'max_policy_steps': run.get('max_policy_steps', steps),
               'policy_dt': dt, 'checkpoint': metadata(run, 'checkpoint_sha256', run.get('checkpoint')),
               'actor_encoding': run.get('actor_encoding'), 'robot_preset': run.get('robot_preset'),
               'deterministic': run.get('deterministic'), 'gpu_name': run.get('gpu_name'),
               'runtime_scope': runtime['scope'], 'runtime_count_basis': basis,
               'batch_num_envs': runtime['batch_num_envs'], 'batch_steps_run': runtime['batch_steps_run'],
               'diagnostic_hz': metadata(run, 'diagnostic_hz', metadata(run, 'diagnostics_hz')),
               'diagnostic_sample_every': metadata(run, 'diagnostic_sample_every'),
               'simulated_seconds': simulated if finite_number(simulated) else None,
               'quality_metrics': selected, 'statistic': statistic, **hashes}
    applied_limits = limit_map(limits or {}, probe, selected)
    incomplete = [name for name in selected if not diagnostic[name]['complete']]
    violations = [name for name, bound in applied_limits.items()
                  if diagnostic[name][statistic] is not None and diagnostic[name][statistic] > bound]
    objective = {name: diagnostic[name][statistic] for name in selected}
    eligible = (status == 'complete' and verified and n > 0 and not incomplete and not missing_hashes and not clock_errors
                and throughput is not None and throughput > 0 and not violations)
    candidate = str(metadata(run, 'candidate_id', metadata(run, 'candidate', Path(path).parent.name)))
    return {'path': str(path), 'candidate_id': candidate, 'material': run.get('material'), 'probe': probe,
            'seed': run.get('seed'), 'raw_status': raw_status, 'status': status,
            'num_envs': n, 'completed_episodes': completed, 'censored': censored,
            'numerical_failure': numerical_failure, 'numerical_checks_verified': verified,
            'error': error or None, 'behavioral_falls': reasons.get('done_fall', reasons.get('fall')),
            'physical_folds': run.get('physical_folds'), 'metrics': diagnostic, 'runtime': runtime,
            'required_quality_metrics': selected, 'quality_objective': objective,
            'applied_quality_limits': applied_limits, 'quality_limit_violations': violations,
            'missing_quality_metrics': incomplete, 'missing_input_audit_hashes': missing_hashes,
            'physics_clock_audit': {'required': clock_required, 'verified': not clock_errors,
                'simulated_seconds': simulated if finite_number(simulated) else None,
                'expected_seconds_from_steps': expected_seconds, 'errors': clock_errors},
            'input_audit_verified': not missing_hashes and not clock_errors, 'comparison_protocol': context,
            'pareto_eligible': eligible,
            'acceptance_scope': 'Numerical completion, observed diagnostics and supplied provisional limits; not validated physical accuracy.'}


def dominates(a, b, metrics):
    one = [a['quality'][name] for name in metrics] + [-a['policy_sps']]
    two = [b['quality'][name] for name in metrics] + [-b['policy_sps']]
    return all(x <= y for x, y in zip(one, two)) and any(x < y for x, y in zip(one, two))


def rank_group(runs):
    by_candidate = defaultdict(list)
    for run in runs:
        by_candidate[run['candidate_id']].append(run)
    candidates = []
    for name, replicates in sorted(by_candidate.items()):
        eligible = all(run['pareto_eligible'] for run in replicates)
        metrics = replicates[0]['required_quality_metrics']
        candidates.append({'candidate_id': name, 'run_paths': [r['path'] for r in replicates],
            'replicates': len(replicates), 'accepted': eligible, 'pareto_rank': None,
            'quality': {metric: max(r['quality_objective'][metric] for r in replicates)
                        for metric in metrics} if eligible else None,
            'policy_sps': statistics.median(r['runtime']['policy_sps'] for r in replicates) if eligible else None})
    accepted = [c for c in candidates if c['accepted']]
    if len(accepted) < 2:
        conclusion = 'Insufficient fully observed, matched candidates for a comparative frontier.'
    else:
        metrics = runs[0]['required_quality_metrics']
        remaining, layer = accepted[:], 1
        while remaining:
            front = [a for a in remaining if not any(dominates(b, a, metrics) for b in remaining)]
            for candidate in front:
                candidate['pareto_rank'] = layer
            remaining = [a for a in remaining if a not in front]
            layer += 1
        quality_vectors = {tuple(c['quality'][metric] for metric in metrics) for c in accepted}
        conclusion = ('All measured quality diagnostics tie; frontier distinguishes runtime only, not physical accuracy.'
                      if len(quality_vectors) == 1 else
                      'Local diagnostic/runtime Pareto layers; no weighted score or global optimum is inferred.')
    return {'candidates': candidates, 'frontier_candidate_ids': [c['candidate_id'] for c in candidates if c['pareto_rank'] == 1],
            'selected_candidate': None, 'conclusion': conclusion}


def analyze_runs(inputs, *, limits=None, statistic='max', metrics=None):
    runs = [normalize_run(run, path, limits=limits, statistic=statistic, metrics=metrics) for path, run in inputs]
    cohorts = defaultdict(list)
    for run in runs:
        cohorts[(run['material'], run['probe'], run['seed'])].append(run)
    output = []
    for (material, probe, seed), rows in sorted(cohorts.items(), key=lambda item: str(item[0])):
        groups = defaultdict(list)
        for row in rows:
            # Unknown hashes never match merely because both values are missing.
            key = json.dumps(row['comparison_protocol'], sort_keys=True)
            if not row['input_audit_verified']:
                key += row['path']
            groups[key].append(row)
        protocols = []
        for key, group in groups.items():
            protocols.append({'protocol_group_id': hashlib.sha256(key.encode()).hexdigest()[:16],
                'input_audit_verified': all(r['input_audit_verified'] for r in group),
                'run_paths': [r['path'] for r in group], 'comparison_protocol': group[0]['comparison_protocol'],
                **rank_group(group)})
        output.append({'material': material, 'probe': probe, 'seed': seed,
            'run_status_counts': dict(Counter(r['status'] for r in rows)),
            'protocol_group_count': len(protocols), 'protocol_groups': protocols,
            'comparability_note': 'Separate groups must not be ranked against each other.' if len(protocols)>1 else None})
    return {'schema_version': 1, 'global_optimum_established': False, 'selected_candidate': None,
            'pareto_statistic': statistic, 'limits': limits, 'runs': runs, 'cohorts': output,
            'all_probe_acceptance': all_probe_acceptance(runs, output, limits or {}),
            'limitations': [
                'Robot contact-plane crossings are cached-record proxies, not exact robot mesh penetration.',
                'Zero records mean unobserved robot crossing; missing diagnostics are not filled with zero.',
                'Table-box depths inspect nodes at sampled frames, not triangle interiors or every substep.',
                'Particle speed and one-sided stretch peaks are stability proxies, not physical accuracy guarantees.',
                'Falls are behavioral outcomes; numerical failures and overflows invalidate a run.',
                'Shared-batch probe SPS attributes a shared wall time; it is not independently measured probe runtime.',
                'No globally optimal solver setting is selected from these probes.']}


def all_probe_acceptance(runs, cohorts, matrix):
    """Fastest threshold-acceptable setting for a complete, matched local matrix.

    Materials/seeds remain separate rather than introducing weights across tasks.
    Shared batch timings are deduplicated across probe records. Until every
    requested candidate has an attempted run and all required probes are present,
    no fastest setting is selected.
    """
    declared = matrix.get('quality_metrics')
    required = list(declared) if isinstance(declared, dict) else list(DEFAULT_PROBES)
    expected = set(matrix.get('candidates', {}))
    horizon = matrix.get('steps')
    results = []
    pairs = sorted({(r['material'], r['seed']) for r in runs}, key=str)
    for material, seed in pairs:
        local = [r for r in runs if r['material'] == material and r['seed'] == seed
                 and (horizon is None or r['comparison_protocol']['max_policy_steps'] == horizon)]
        seen = {r['candidate_id'] for r in local}
        pending = sorted(expected-seen)
        missing, unverified, ambiguous, accepted = [], [], [], None
        for probe in required:
            matches = [c for c in cohorts if (c['material'], c['seed'], c['probe']) == (material, seed, probe)]
            scoped = [g for c in matches for g in c['protocol_groups']
                      if horizon is None or g['comparison_protocol']['max_policy_steps'] == horizon]
            groups = [g for g in scoped if g['input_audit_verified']]
            if not groups:
                (unverified if scoped else missing).append(probe)
                continue
            if len(groups) != 1:
                ambiguous.append(probe)
                continue
            valid = {c['candidate_id'] for c in groups[0]['candidates'] if c['accepted']}
            accepted = valid if accepted is None else accepted & valid
        complete = not missing and not unverified and not ambiguous and not pending
        accepted = sorted(accepted or []) if complete else []
        elapsed, timing_errors = {}, []
        for candidate in accepted:
            batches = {}
            for run in local:
                if run['candidate_id'] != candidate or run['probe'] not in required:
                    continue
                runtime = run['runtime']
                key = runtime['batchid'] or run['path']
                seconds = runtime['batch_rollout_seconds'] or runtime['rollout_seconds']
                if not finite_number(seconds) or seconds <= 0:
                    timing_errors.append(candidate)
                    continue
                if key in batches and not math.isclose(batches[key], seconds, rel_tol=1e-8):
                    timing_errors.append(candidate)
                batches[key] = seconds
            if candidate not in timing_errors and batches:
                elapsed[candidate] = statistics.median(batches.values())
        gates_supplied = (bool(matrix.get('quality_limits', matrix.get('global', {})))
                          or bool(matrix.get('probes')) or any(k in matrix for k in KNOWN_METRICS))
        shared_timing = all(r['runtime']['scope'] == 'shared_batch_probe_attribution'
                            for r in local if r['candidate_id'] in accepted)
        fastest = []
        if complete and gates_supplied and elapsed and shared_timing:
            best = min(elapsed.values())
            fastest = [name for name, value in elapsed.items() if value == best]
        results.append({'material': material, 'seed': seed, 'required_probes': required,
            'complete_comparable_matrix': complete, 'missing_probes': missing,
            'unverified_input_or_clock_probes': unverified,
            'incomparable_protocol_probes': ambiguous, 'pending_candidate_ids': pending,
            'acceptable_candidate_ids': accepted, 'median_unique_batch_wall_seconds': elapsed,
            'timing_errors_candidate_ids': sorted(set(timing_errors)),
            'fastest_acceptable_candidate_ids': fastest,
            'selection_scope': 'Fastest among this complete matrix meeting explicitly supplied provisional probe limits; not a global optimum.',
            'selection_withheld_reason': ('matrix_incomplete_or_incomparable' if not complete else
                'no_explicit_quality_limits' if not gates_supplied else
                'no_fully_observed_candidate_passes_all_probe_limits' if not elapsed else
                'overall_selection_requires_shared_batch_timing' if not shared_timing else None)})
    return results


def load_inputs(patterns):
    paths, errors = set(), []
    for pattern in patterns:
        found = glob.glob(str(pattern), recursive=True)
        if not found:
            errors.append({'path': str(pattern), 'error': 'Input path or glob did not match'})
        for item in found:
            path = Path(item)
            paths.update(path.rglob('*.json') if path.is_dir() else [path])
    inputs, seen = [], set()
    for path in sorted(paths, key=lambda p: (p.name == 'summary.json', str(p))):
        try:
            payload = json.loads(path.read_text())
            values = [(str(path), payload)] if isinstance(payload, dict) and 'status' in payload else (
                [(f'{path}#{name}', value) for name, value in payload.items()
                 if isinstance(value, dict) and 'status' in value] if isinstance(payload, dict) else [])
            for label, value in values:
                if 'material' not in value:
                    continue
                digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                if digest not in seen:
                    inputs.append((label, value))
                    seen.add(digest)
        except (OSError, ValueError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    return inputs, errors


def markdown(report):
    lines = ['# VBD stability probe audit', '',
        'Frontiers compare measured diagnostic peaks and runtime within matched material/probe/seed/input protocols. No global optimum is selected.', '',
        '| Candidate | Material / probe / seed | Status | Falls | SPS | Missing diagnostics | Input/clock audit |',
        '|---|---|---|---:|---:|---|---|']
    for run in report['runs']:
        sps = run['runtime']['policy_sps']
        rate = f'{sps:.3f}' if sps is not None else 'unknown'
        falls = run['behavioral_falls'] if run['behavioral_falls'] is not None else 'unknown'
        audit = 'verified' if run['input_audit_verified'] else ', '.join(
            run['missing_input_audit_hashes'] + run['physics_clock_audit']['errors'])
        lines.append(f"| {run['candidate_id']} | {run['material']} / {run['probe']} / {run['seed']} | "
                     f"{run['status']} | {falls} | {rate} | {', '.join(run['missing_quality_metrics'])} | {audit} |")
    display = list(QUALITY_METRICS)
    show_late = any('late_particle_speed_m_s' in r['required_quality_metrics'] for r in report['runs'])
    if show_late:
        display.append('late_particle_speed_m_s')
    columns = ['Candidate / probe', 'Robot plane proxy max (coverage)', 'Table node depth max',
               'Particle speed max', 'Edge ratio max', 'P95-edge ratio peak max']
    if show_late:
        columns.append('Late particle speed max')
    lines += ['', '## Measured episode-peak diagnostics', '',
              '| '+' | '.join(columns)+' |', '|---|'+'---:|'*(len(columns)-1)]
    for run in report['runs']:
        values = []
        for metric in display:
            d = run['metrics'][metric]
            value = f"{d['max']:.6g}" if d['max'] is not None else 'unknown'
            if metric.startswith('robot_'):
                value += f" ({d['observed_episodes']}/{d['expected_episodes']})"
            values.append(value)
        lines.append(f"| {run['candidate_id']} / {run['probe']} | "+' | '.join(values)+' |')
    for cohort in report['cohorts']:
        lines += ['', f"## {cohort['material']} / {cohort['probe']} / seed {cohort['seed']}", '',
                  f"Run statuses: {cohort['run_status_counts']}. Comparable protocol groups: {cohort['protocol_group_count']}."]
        for group in cohort['protocol_groups']:
            frontier = ', '.join(group['frontier_candidate_ids']) or 'none'
            lines += ['', f"Protocol {group['protocol_group_id']}: frontier **{frontier}**. {group['conclusion']}"]
    lines += ['', '## All-probe acceptance', '']
    for group in report['all_probe_acceptance']:
        choice = ', '.join(group['fastest_acceptable_candidate_ids']) or 'none'
        lines += [f"- {group['material']} / seed {group['seed']}: fastest acceptable within this matrix **{choice}**. "
                  f"Accepted across required probes: {group['acceptable_candidate_ids']}. "
                  f"Withheld reason: {group['selection_withheld_reason'] or 'not withheld'}. "
                  f"{group['selection_scope']}"]
    lines += ['', '## Measurement limits', '', *['- '+line for line in report['limitations']]]
    if report.get('input_errors'):
        lines += ['', 'Input errors: '+json.dumps(report['input_errors'])]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limits', type=Path)
    parser.add_argument('--statistic', choices=('max','p95'), default='max')
    parser.add_argument('--metrics', help='Explicit comma-separated scope; otherwise use each probe quality_metrics')
    args = parser.parse_args()
    limits = None
    if args.limits:
        text = args.limits.read_text()
        if args.limits.suffix.lower() == '.json':
            limits = json.loads(text)
        else:
            import yaml
            limits = yaml.safe_load(text)
    inputs, errors = load_inputs(args.inputs)
    report = analyze_runs(inputs, limits=limits, statistic=args.statistic,
                          metrics=[name.strip() for name in args.metrics.split(',') if name.strip()] if args.metrics else None)
    report['input_errors'] = errors
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'stability.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.out_dir/'stability.md').write_text(markdown(report))
    print(f"Audited {len(inputs)} runs across {len(report['cohorts'])} cohorts; wrote {args.out_dir}")


if __name__ == '__main__':
    main()
