"""Check the geometry/force quantities used by the shadow reset sweep."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.nov17.pressing_audit.validate_resets import causal_mean, first_crossing, physical_metrics


def fixture():
    half = .001
    anchor = np.array([0., 0., half])
    angle = Rotation.from_rotvec(np.array([0., -1., 0.])*np.pi/3)
    pose = np.zeros((1, 1, 2, 7))
    pose[..., 3] = 1
    pose[0, 0, :, 2] = .531
    pose[0, 0, 1, :3] += anchor-angle.apply(anchor)
    pose[0, 0, 1, 3:] = angle.as_quat()[[3, 0, 1, 2]]
    plates = []
    for lo, hi in ((-.1, 0.), (0., .1)):
        plates.append(dict(vertices=[[x, y, z] for z in (-half, half) for x, y in ((lo, -.1), (hi, -.1), (hi, .1), (lo, .1))]))
    geometry = {0: dict(plates=plates, hinges=[dict(parent=0, child=1, anchor=anchor, axis=[0., -1., 0.], limits_deg=[-90., 180.])])}
    trajectory = dict(active=np.ones((1, 1), bool), panel_pose=pose,
                      table_top=np.array([[.53]]), hinge_q=np.array([[[np.pi/3]]]))
    contacts = dict(frame_idx=np.array([0, 0]), env_id=np.array([0, 0]), cloth_body=np.array([0, 0]),
                    partner_type=np.array([0, 1]), force_vector=np.array([[0., 0., -2.], [0., 0., 2.1]]),
                    force_magnitude=np.array([2., 2.1]), separation=np.array([-.0007, 0.]))
    return trajectory, contacts, geometry


def test_rest_and_folded_anchor_transforms_agree_without_false_penetration():
    d, c, g = fixture()
    result = physical_metrics({}, d, c, g)
    assert result['anchor_error_m'][0, 0] < 1e-12
    assert result['axis_error_deg'][0, 0] < 1e-10
    assert result['table_depth_m'][0, 0] < 1e-12
    assert result['limit_error_deg'][0, 0] == 0
    assert result['compression_n'][0, 0] == 2
    assert result['robot_reported_depth_m'][0, 0] == .0007


def test_compression_requires_opposed_loads_on_the_same_plate():
    d, c, g = fixture()
    c['cloth_body'][1] = 1
    assert physical_metrics({}, d, c, g)['compression_n'][0, 0] == 0


def test_known_anchor_table_and_limit_errors_are_measured():
    d, c, g = fixture()
    d['panel_pose'][0, 0, 1, 0] += .0004
    d['panel_pose'][0, 0, 0, 2] -= .0003
    d['hinge_q'][0, 0, 0] = np.deg2rad(182)
    result = physical_metrics({}, d, c, g)
    assert result['anchor_error_m'][0, 0] == pytest.approx(.0005)
    assert result['table_depth_m'][0, 0] == pytest.approx(.0003)
    assert result['limit_error_deg'][0, 0] == pytest.approx(2)


def test_causal_average_does_not_anticipate_future_pressure():
    values = np.array([[0.], [0.], [6.], [0.]])
    np.testing.assert_allclose(causal_mean(values, 3).flatten(), [0., 0., 2., 2.])
    np.testing.assert_array_equal(first_crossing(np.array([[False, False], [True, False]])), [1, 2])


def test_missing_archived_hinge_metadata_is_unknown_not_zero():
    d, c, g = fixture()
    del g[0]['hinges']
    result = physical_metrics({}, d, c, g)
    assert np.isnan(result['anchor_error_m']).all()
    assert np.isnan(result['axis_error_deg']).all()
    assert np.isnan(result['limit_error_deg']).all()
    assert result['compression_n'][0, 0] == 2
