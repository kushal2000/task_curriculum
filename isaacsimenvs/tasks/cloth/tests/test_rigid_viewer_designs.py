"""Viewer adapters must preserve polygon offsets and the chosen central fold."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from isaacsimenvs.scripts.visualization.viser_rigidCloth_viewer import Physics, Settings
from isaacsimenvs.tasks.cloth.backends.rigid import designs


def test_render_poses_use_links_not_offset_or_rotated_inertial_frames():
    physics = Physics.__new__(Physics)
    physics.body_ids = [1, 0]
    links = torch.tensor([[[1., 2., 3., 1., 0., 0., 0.], [4., 5., 6., 0., 0., 0., 1.]]])
    physics.cloth = NS(data=NS(body_link_pose_w=links, body_com_pose_w=links + 10))
    positions, orientations = physics.read_poses()
    np.testing.assert_array_equal(positions, [[4., 5., 6.], [1., 2., 3.]])
    np.testing.assert_array_equal(orientations, [[0., 0., 0., 1.], [1., 0., 0., 0.]])
    links.zero_()
    np.testing.assert_array_equal(positions[0], [4., 5., 6.])


@pytest.mark.parametrize("name", ["single_hinge", "strip_4", "strip_8", "hinge_band", "flexible_corners"])
def test_reset_folds_only_target_joint_with_reordered_simulator_storage(name):
    physics = Physics.__new__(Physics)
    settings = Settings(design=name, bend_deg=45.)
    physics.spec = designs.load_design(name).resolve(settings.length, settings.width, settings.thickness)
    names = list(reversed(physics.spec.joint_names))
    physics.joint_ids = [names.index(joint) for joint in physics.spec.joint_names]
    physics.table_top, physics.dt = .53, 1 / 120
    physics.dimensions = (settings.length, settings.width, settings.thickness)
    physics.table_size = np.array([.6, .4, .3])
    physics.table_position = np.array([0., 0., .38])
    physics.update_material = lambda _: None
    state = {}
    cloth = physics.cloth = NS(
        device="cpu",
        data=NS(default_root_state=torch.zeros(1, 13), default_joint_pos=torch.zeros(1, len(names))),
    )
    cloth.write_root_state_to_sim = lambda root: state.update(root=root)
    cloth.write_joint_state_to_sim = lambda position, velocity: state.update(position=position, velocity=velocity)
    cloth.set_joint_position_target = lambda value: state.update(position_target=value)
    cloth.set_joint_velocity_target = lambda value: state.update(velocity_target=value)
    cloth.reset = cloth.write_data_to_sim = lambda: None
    cloth.update = lambda _: None
    physics.sim = NS(forward=lambda: None)
    physics.reset(settings)

    expected = torch.zeros(1, len(names))
    fold_name = physics.spec.joint_names[physics.spec.fold_hinge]
    expected[0, names.index(fold_name)] = np.pi / 4
    torch.testing.assert_close(state["position"], expected)
    for key in ("velocity", "position_target", "velocity_target"):
        assert torch.count_nonzero(state[key]) == 0
    torch.testing.assert_close(state["root"][0, :3], torch.tensor([0., 0., .53 + settings.drop_height + settings.thickness / 2]))
    assert physics.time == 0.
