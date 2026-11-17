"""Synthetic evidence guards for missing-data, protocol and Pareto stability analysis."""
import hashlib
import json

import pytest

from experiments.nov16_yibo_our_dr.analyze_vbd_stability import (
    QUALITY_METRICS, analyze_runs, load_inputs, markdown, normalize_run,
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def run(candidate='candidate', probe='scripted', quality=1., seconds=10.):
    peaks = {'robot_record_count': [1., 2.],
        'robot_record_contact_plane_crossing_m': [quality*.001]*2,
        'table_box_center_depth_m': [quality*.001]*2,
        'particle_speed_m_s': [quality]*2,
        'late_particle_speed_m_s': [quality*.1]*2,
        'edge_stretch_ratio': [1+quality*.1]*2,
        'p95_edge_stretch_ratio': [1+quality*.05]*2}
    metrics = list(QUALITY_METRICS)
    if probe != 'scripted':
        metrics.remove('robot_record_contact_plane_crossing_m')
    return {'status': 'complete', 'candidate_id': candidate, 'material': 'base', 'probe': probe,
        'seed': 0, 'num_envs': 2, 'max_policy_steps': 10, 'steps_run': 10,
        'completed_episodes': 2, 'censored': 0, 'physical_folds': 0,
        'finite_checks_passed': True, 'overflow_checks_passed': True,
        'rollout_seconds': seconds, 'per_env_peak_metrics': peaks,
        'quality_metrics': metrics, 'termination_reasons': {'done_fall': 2},
        'initial_state_sha256': sha('initial '+probe), 'action_trace_sha256': sha('action '+probe),
        'physical_protocol_sha256': sha('physics '+probe), 'gpu_name': 'test_gpu',
        'actor_encoding': 'yibo', 'robot_preset': 'yibo', 'deterministic': True,
        'physics_step_duration_checked': True, 'simulated_seconds': 10/60,
        'config': {'sim': {'dt': 1/120}, 'decimation': 2}}


def inputs(*runs):
    return [(f'{i}.json', value) for i, value in enumerate(runs)]


def test_falls_are_behavioral_while_nonfinite_and_overflow_invalidate():
    good = normalize_run(run(), 'good.json')
    assert good['status'] == 'complete' and good['behavioral_falls'] == 2
    assert good['pareto_eligible']
    for field in ('finite_checks_passed', 'overflow_checks_passed'):
        bad = run()
        bad[field] = False
        result = normalize_run(bad, 'bad.json')
        assert result['status'] == 'invalid_numerical' and not result['pareto_eligible']
    bad = run()
    bad.update(status='failed', error='Contact count exceeds buffer capacity')
    assert normalize_run(bad, 'overflow.json')['numerical_failure']


def test_failed_and_partial_runs_remain_visible_and_cannot_enter_frontier():
    failed, partial = run('failed'), run('partial')
    failed.update(status='failed', error='Missing asset cache')
    partial.update(status='running', completed_episodes=1, censored=1)
    report = analyze_runs(inputs(run('good'), failed, partial))
    assert [r['status'] for r in report['runs']] == ['complete', 'failed', 'partial']
    assert not report['cohorts'][0]['protocol_groups'][0]['frontier_candidate_ids']


def test_missing_contact_probe_is_not_reported_as_numerical_blowup():
    data = run()
    data.update(status='failed', error='Invalid scripted contact probe: no robot records',
                scripted_contact_coverage_passed=False)
    data['per_env_peak_metrics']['robot_record_count'] = [0., 0.]
    result = normalize_run(data, 'no-contact.json')
    assert result['status'] == 'failed'
    assert not result['numerical_failure'] and not result['pareto_eligible']


def test_missing_diagnostics_and_zero_records_are_not_zero_penetration():
    missing = run()
    del missing['per_env_peak_metrics']['table_box_center_depth_m']
    result = normalize_run(missing, 'missing.json')
    metric = result['metrics']['table_box_center_depth_m']
    assert metric['max'] is None and metric['missing_or_unobserved_episodes'] == 2
    zero = run()
    zero['per_env_peak_metrics']['robot_record_contact_plane_crossing_m'] = [0., 0.]
    zero['per_env_peak_metrics']['robot_record_count'] = [0., 0.]
    result = normalize_run(zero, 'zero.json')
    metric = result['metrics']['robot_record_contact_plane_crossing_m']
    assert metric['max'] is None and metric['record_covered_episodes'] == 0
    assert not result['pareto_eligible']
    zero['per_env_peak_metrics']['robot_record_count'] = [1., 1.]
    result = normalize_run(zero, 'measured_zero.json')
    assert result['metrics']['robot_record_contact_plane_crossing_m']['max'] == 0.
    assert result['pareto_eligible']


def test_nonfinite_diagnostic_is_invalid_and_output_stays_json_serializable():
    bad = run()
    bad['per_env_peak_metrics']['particle_speed_m_s'][0] = float('nan')
    report = analyze_runs(inputs(bad))
    assert report['runs'][0]['status'] == 'invalid_numerical'
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('field,value', [('action_trace_sha256', sha('other action')),
    ('initial_state_sha256', sha('other initial')), ('physical_protocol_sha256', sha('other physics')),
    ('max_policy_steps', 20), ('gpu_name', 'other_gpu')])
def test_incomparable_protocols_are_not_ranked_together(field, value):
    a, b = run('a'), run('b')
    b[field] = value
    report = analyze_runs(inputs(a, b))
    cohort = report['cohorts'][0]
    assert cohort['protocol_group_count'] == 2
    assert all(not g['frontier_candidate_ids'] for g in cohort['protocol_groups'])
    assert report['selected_candidate'] is None and not report['global_optimum_established']


def test_missing_hashes_do_not_match_each_other():
    a, b = run('a'), run('b')
    for value in (a, b):
        del value['action_trace_sha256']
    report = analyze_runs(inputs(a, b))
    assert report['cohorts'][0]['protocol_group_count'] == 2
    assert all(not r['pareto_eligible'] for r in report['runs'])


@pytest.mark.parametrize('flag', [None, False])
def test_benchmark_requires_actual_native_step_duration_check(flag):
    value = run()
    if flag is None:
        del value['physics_step_duration_checked']
    else:
        value['physics_step_duration_checked'] = flag
    result = normalize_run(value, 'wrong_clock.json')
    assert not result['input_audit_verified'] and not result['pareto_eligible']
    assert 'native_physics_step_duration_not_verified' in result['physics_clock_audit']['errors']


def test_actual_simulated_duration_mismatch_cannot_enter_timing_comparison():
    value = run()
    value['simulated_seconds'] *= 2  # The detected manager+manual-loop double-decimation bug.
    result = normalize_run(value, 'double_tick.json')
    assert not result['pareto_eligible']
    assert 'simulated_duration_does_not_match_policy_clock' in result['physics_clock_audit']['errors']


def test_unverified_clock_is_not_mislabeled_as_absent_probe():
    data = run(probe='settle')
    del data['physics_step_duration_checked']
    group = analyze_runs(inputs(data))['all_probe_acceptance'][0]
    assert 'settle' in group['unverified_input_or_clock_probes']
    assert 'settle' not in group['missing_probes']


def test_misspelled_explicit_quality_limit_is_rejected():
    with pytest.raises(ValueError, match='Invalid quality limit'):
        normalize_run(run(), 'typo.json', limits={'particle_speeed_m_s': 5.})


def test_failed_replicate_cannot_be_hidden_by_a_successful_replicate():
    good, failed = run('unstable'), run('unstable')
    failed.update(status='failed', error='Nonfinite cloth state')
    report = analyze_runs(inputs(good, failed, run('stable')))
    candidates = report['cohorts'][0]['protocol_groups'][0]['candidates']
    assert not next(c for c in candidates if c['candidate_id'] == 'unstable')['accepted']


def test_zero_record_fast_candidate_cannot_outrank_measured_zero_crossing():
    unknown, known, slower = run('unobserved', quality=0., seconds=.1), run('known', quality=0.), run('slower', quality=0., seconds=20.)
    unknown['per_env_peak_metrics']['robot_record_count'] = [0., 0.]
    report = analyze_runs(inputs(unknown, known, slower))
    group = report['cohorts'][0]['protocol_groups'][0]
    assert group['frontier_candidate_ids'] == ['known']
    assert not next(c for c in group['candidates'] if c['candidate_id'] == 'unobserved')['accepted']
    assert 'runtime only' in group['conclusion'] and not report['global_optimum_established']


def test_quality_runtime_tradeoff_is_pareto_incomparable_and_dominated_run_ranks_lower():
    report = analyze_runs(inputs(run('accurate', quality=1., seconds=10.),
                                run('fast', quality=2., seconds=5.),
                                run('dominated', quality=3., seconds=15.)))
    group = report['cohorts'][0]['protocol_groups'][0]
    assert set(group['frontier_candidate_ids']) == {'accurate', 'fast'}
    assert next(c for c in group['candidates'] if c['candidate_id'] == 'dominated')['pareto_rank'] == 2
    assert group['selected_candidate'] is None


def test_equal_quality_is_reported_as_runtime_only_tie_without_optimality_claim():
    report = analyze_runs(inputs(run('slow', seconds=10.), run('fast', seconds=5.)))
    group = report['cohorts'][0]['protocol_groups'][0]
    assert group['frontier_candidate_ids'] == ['fast']
    assert 'runtime only' in group['conclusion'] and 'tie' in group['conclusion']
    assert not report['global_optimum_established'] and report['selected_candidate'] is None


def test_active_transition_runtime_and_shared_batch_scope_are_explicit():
    data = run()
    data.update(active_transitions=7, batchid='batch1', batch_num_envs=6, batch_steps_run=10,
                batch_rollout_seconds=10.)
    result = normalize_run(data, 'batch.json')
    assert result['runtime']['policy_sps'] == .7
    assert result['runtime']['scope'] == 'shared_batch_probe_attribution'
    assert result['runtime']['count_basis'] == 'active_transitions'


def test_probe_declared_scope_can_exclude_unobserved_robot_and_support_late_speed():
    data = run(probe='settle')
    data['quality_metrics'].append('late_particle_speed_m_s')
    data['per_env_peak_metrics']['robot_record_count'] = [0., 0.]
    result = normalize_run(data, 'settle.json', limits={'global': {
        'robot_record_contact_plane_crossing_m': 0., 'table_box_center_depth_m': .002}})
    assert result['pareto_eligible']
    assert result['quality_objective']['late_particle_speed_m_s'] == .1
    assert 'robot_record_contact_plane_crossing_m' not in result['applied_quality_limits']


def matrix():
    return {'steps': 10, 'candidates': {'fast': {}, 'slow': {}},
            'quality_metrics': {p: [] for p in ('settle', 'scripted', 'fold_release')},
            'quality_limits': {'global': {'table_box_center_depth_m': .01, 'particle_speed_m_s': 10.,
                'edge_stretch_ratio': 2., 'p95_edge_stretch_ratio': 2.},
                'probes': {'scripted': {'robot_record_contact_plane_crossing_m': .002}}}}


def batch_runs():
    result = []
    for candidate, seconds in (('fast', 2.), ('slow', 4.)):
        for probe in ('settle', 'scripted', 'fold_release'):
            value = run(candidate, probe=probe, seconds=seconds)
            value.update(batchid=candidate+'batch', batch_num_envs=6, batch_steps_run=10,
                         batch_rollout_seconds=seconds)
            result.append(value)
    return result


def test_all_probe_intersection_rejects_fast_candidate_that_fails_manipulation():
    data = batch_runs()
    data[1]['per_env_peak_metrics']['robot_record_contact_plane_crossing_m'] = [.003, .003]
    report = analyze_runs(inputs(*data), limits=matrix())
    group = report['all_probe_acceptance'][0]
    assert group['complete_comparable_matrix']
    assert group['acceptable_candidate_ids'] == ['slow']
    assert group['fastest_acceptable_candidate_ids'] == ['slow']
    assert group['median_unique_batch_wall_seconds']['slow'] == 4.  # Not 3 × shared batch cost.
    assert not report['global_optimum_established']


def test_overall_selection_waits_for_required_probes_and_requested_candidates():
    data = batch_runs()
    report = analyze_runs(inputs(*data[:1]), limits=matrix())
    group = report['all_probe_acceptance'][0]
    assert set(group['missing_probes']) == {'scripted', 'fold_release'}
    assert group['pending_candidate_ids'] == ['slow']
    assert group['fastest_acceptable_candidate_ids'] == []
    spec = matrix()
    spec['candidates']['pending'] = {}
    group = analyze_runs(inputs(*data), limits=spec)['all_probe_acceptance'][0]
    assert not group['complete_comparable_matrix'] and group['pending_candidate_ids'] == ['pending']


def test_no_limits_means_pareto_only_and_no_overall_fastest_selection():
    group = analyze_runs(inputs(*batch_runs()))['all_probe_acceptance'][0]
    assert group['acceptable_candidate_ids'] == ['fast', 'slow']
    assert group['fastest_acceptable_candidate_ids'] == []
    assert group['selection_withheld_reason'] == 'no_explicit_quality_limits'


def test_short_valid_smokes_do_not_count_as_completed_full_matrix_protocol():
    spec = matrix()
    spec['steps'] = 300
    report = analyze_runs(inputs(*batch_runs()), limits=spec)
    assert all(r['physics_clock_audit']['verified'] for r in report['runs'])
    group = report['all_probe_acceptance'][0]
    assert set(group['missing_probes']) == {'settle', 'scripted', 'fold_release'}
    assert not group['unverified_input_or_clock_probes']
    assert not group['fastest_acceptable_candidate_ids']


def test_duplicate_material_summary_does_not_double_count_and_bad_json_is_reported(tmp_path):
    value = run()
    (tmp_path/'base.json').write_text(json.dumps(value))
    (tmp_path/'summary.json').write_text(json.dumps({'base': value}))
    (tmp_path/'broken.json').write_text('{')
    data, errors = load_inputs([str(tmp_path)])
    assert len(data) == 1 and len(errors) == 1
    report = analyze_runs(data)
    report['input_errors'] = errors
    assert 'unknown' not in markdown(report)  # All fields in this complete fixture are observed.
