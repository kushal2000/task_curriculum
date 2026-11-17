"""Kit-free regressions for ragged contact buffers, pressure, sign and body transforms."""
import pytest
import torch

from experiments.nov17_pressing_audit.contacts import (
    contact_rest_positions,
    unpack_contacts,
)


def _buffers(sign=1.):
    force = torch.zeros(16, 1)
    force[:4, 0] = torch.tensor([2., 2., 3., 5.])
    points = torch.zeros(16, 3)
    points[:4] = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.], [2., 3., 4.]])
    normals = torch.zeros(16, 3)
    normals[:4] = sign*torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.], [0., 0., -1.]])
    counts = torch.tensor([[2, 1], [1, 0]])
    # An absent pair's offset is unspecified (often UINT_MAX), and must be ignored.
    starts = torch.tensor([[0, 2], [3, 2**32-1]])
    matrix = torch.tensor([[[0., 0., 0.], [0., 0., 3.]], [[0., 0., -5.], [0., 0., 0.]]])
    return (force, points, normals, torch.zeros(16, 1), counts, starts), matrix


@pytest.mark.parametrize('sign', [1., -1.])
def test_contact_normals_are_verified_against_on_sensor_matrix_without_pressure_cancellation(sign):
    data, matrix = _buffers(sign)
    result = unpack_contacts(data, matrix, capacity=16)
    assert result['sensor_index'].tolist() == [0, 0, 0, 1]
    assert result['partner_index'].tolist() == [0, 0, 1, 0]
    assert result['force_magnitude'].tolist() == [2., 2., 3., 5.]
    # Two opposing robot normal contacts exert 4 N scalar load, despite zero net vector.
    assert result['force_magnitude'][:2].sum() == 4.
    torch.testing.assert_close(result['force_vector'][:2].sum(0), torch.zeros(3))
    assert result['force_vector'][2, 2] == 3.  # table support points UP on cloth
    assert result['normal_buffer_sign'] == sign


def test_empty_buffers_do_not_index_unspecified_offsets():
    data, matrix = _buffers()
    data[4].zero_()
    result = unpack_contacts(data, torch.zeros_like(matrix), capacity=16)
    assert result['position'].shape == (0, 3)


@pytest.mark.parametrize('failure', ['capacity', 'span', 'mismatch'])
def test_incomplete_contact_data_fails_loudly(failure):
    data, matrix = _buffers()
    capacity = 16
    if failure == 'capacity':
        capacity = 4
    elif failure == 'span':
        data[5][0, 0] = 15
    else:
        matrix[1, 0, 2] = -7.
    with pytest.raises(RuntimeError):
        unpack_contacts(data, matrix, capacity=capacity)


def test_contact_rest_coordinates_follow_rotated_translated_link_and_sparse_sensor_order():
    # Sensor0 is rotated90° aboutY; sensor1 translated with identity orientation.
    root = torch.tensor([[10., 20., 30.], [-3., 2., 1.]])
    q = torch.tensor([[2**-.5, 0., 2**-.5, 0.], [1., 0., 0., 0.]])
    ids = torch.tensor([1, 0, 0])
    world = torch.tensor([[-2.8, 1.9, 1.], [10., 20., 29.8], [10., 20., 30.2]])
    rest = contact_rest_positions(world, ids, root, q)
    torch.testing.assert_close(rest, torch.tensor([[.2, -.1, 0.], [.2, 0., 0.], [-.2, 0., 0.]]), atol=2e-6, rtol=1e-5)


def test_initialized_view_uses_actual_sparse_paths_and_per_sensor_env_filters(monkeypatch):
    import sys
    from types import SimpleNamespace as NS

    from experiments.nov17_pressing_audit.contacts import RigidClothContacts

    def path(env, name):
        return f'/World/envs/env_{env}/{name}'
    links = [[path(i, 'RigidCloth_0/b'), path(i, 'RigidCloth_0/a')] for i in (1, 0)]
    robot_links = [[path(i, 'Robot/r0'), path(i, 'Robot/r1')] for i in (1, 0)]
    requested = [row[j] for row in links for j in (1, 0)]
    native_order = list(reversed(requested))
    positions = torch.tensor([[[10., 0., 0.], [20., 0., 0.]], [[30., 0., 0.], [40., 0., 0.]]])
    asset = NS(root_physx_view=NS(link_paths=links), data=NS(body_link_pos_w=positions,
               body_link_quat_w=torch.tensor([1., 0., 0., 0.]).expand(2, 2, 4)))
    group = NS(articulation=asset, env_ids=torch.tensor([1, 0]), body_ids=[1, 0])
    backend = NS(cfg=NS(sim=NS(dt=1/120)), device='cpu', groups=[group],
        _press_audit_report_paths=requested,
        robot=NS(root_physx_view=NS(link_paths=robot_links), body_names=['r0', 'r1']),
        table=NS(root_physx_view=NS(prim_paths=[path(i, 'Table/box') for i in (1, 0)])))
    matrix = torch.zeros(4, 3, 3)
    matrix[:, 2, 2] = 2.
    points = torch.zeros(16, 3)
    points[:4, 0] = torch.tensor([30., 40., 10., 20.])+.1
    normals = torch.zeros(16, 3)
    normals[:4, 2] = 1.
    forces = torch.zeros(16, 1)
    forces[:4] = 2.
    counts = torch.zeros(4, 3, dtype=torch.long)
    counts[:, 2] = 1
    starts = torch.zeros_like(counts)
    starts[:, 2] = torch.arange(4)
    data = (forces, points, normals, torch.zeros(16, 1), counts, starts)
    view = NS(sensor_paths=native_order, filter_count=3,
              get_contact_data=lambda dt: data, get_contact_force_matrix=lambda dt: matrix)
    def create(sensors, *, filter_patterns, max_contact_data_count):
        assert sensors == requested
        assert max_contact_data_count == 16
        for sensor, filters in zip(sensors, filter_patterns, strict=True):
            env = int(sensor.split('/env_')[1].split('/')[0])
            assert filters == [path(env, 'Robot/r0'), path(env, 'Robot/r1'), path(env, 'Table/box')]
        return view
    manager = NS(get_physics_sim_view=lambda: NS(create_rigid_contact_view=create))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.simulation_manager', NS(SimulationManager=manager))
    contacts = RigidClothContacts(backend, capacity=16)
    result = contacts.sample()
    assert result['env_id'].tolist() == [0, 0, 1, 1]
    assert result['cloth_body'].tolist() == [1, 0, 1, 0]
    assert result['partner_type'].tolist() == [1, 1, 1, 1]
    torch.testing.assert_close(result['cloth_rest_position'], torch.tensor([[.1, 0., 0.]]).expand(4, 3), atol=2e-6, rtol=1e-5)
    assert all(isinstance(value, torch.Tensor) for value in result.values())
    assert contacts.metadata['force_matrix_verified'] is True


def test_analyzer_rebuilds_moving_plate_contact_despite_negative_rest_x_at_crease(tmp_path):
    import json

    import numpy as np

    from experiments.nov17_pressing_audit.analyze_pressing import load_dataset

    def plate(x0, x1):
        return {'vertices': [[x, y, z] for z in (-.001, .001)
                             for x, y in ((x0, -.05), (x1, -.05), (x1, .05), (x0, .05))]}

    metadata = {'geometry': [{'env_id': 0, 'plates': [plate(-.1, 0.), plate(0., .1)]}]}
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    # Deliberately stale aggregates must be replaced from the raw contact/body identity.
    stale = np.zeros((1, 1, 2, 2))
    stale[0, 0, 1, 0] = 3.
    np.savez(tmp_path/'trajectory.npz', active=np.ones((1, 1), dtype=bool),
             panel_pose=np.zeros((1, 1, 2, 7)), force_magnitude_by_half=stale)
    np.savez(tmp_path/'contacts.npz', frame_idx=np.array([0]), env_id=np.array([0]),
             cloth_body=np.array([1]), partner_type=np.array([1]),
             force_magnitude=np.array([3.]), force_vector=np.array([[0., 0., 3.]]),
             cloth_rest_position=np.array([[-.00028, 0., -.001]]))
    _, trajectory, _, _ = load_dataset(tmp_path)
    np.testing.assert_array_equal(trajectory['force_magnitude_by_half'][0, 0, 1], [0., 3.])
    np.testing.assert_array_equal(trajectory['force_vector_by_half'][0, 0, 1], [[0., 0., 0.], [0., 0., 3.]])
    np.testing.assert_array_equal(trajectory['downward_force_by_half'], np.zeros((1, 1, 2, 2)))


def test_analysis_refuses_running_capture_before_opening_partially_written_npz(tmp_path):
    import json
    from experiments.nov17_pressing_audit.analyze_pressing import load_dataset
    (tmp_path/'metadata.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='not complete'):
        load_dataset(tmp_path)
