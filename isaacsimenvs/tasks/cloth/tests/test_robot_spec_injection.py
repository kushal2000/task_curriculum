"""A supplied robot profile controls rigid scene links and shared policy semantics."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT
from isaacsimenvs.tasks.cloth.mdp.actions import JointController
from isaacsimenvs.tasks.cloth.mdp.geometry import grid
from isaacsimenvs.tasks.cloth.mdp.observations import ObservationBuilder
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask


@pytest.fixture
def robot_spec():
    return replace(
        ROBOT,
        name="test_profile",
        joint_names_canonical=tuple(reversed(ROBOT.joint_names_canonical)),
        palm_body_name="test_palm",
        fingertip_link_names=tuple(f"test_tip_{i}" for i in range(5)),
        palm_center_offset=(0.1, 0.2, 0.3),
        fingertip_offsets=((0.0, 0.0, 0.1),) * 5,
        hand_spring_stiffness={ROBOT.hand_joint_names[0]: 2.0},
        hand_spring_ref={ROBOT.hand_joint_names[0]: 0.25},
    )


def test_injected_joint_order_springs_and_contact_offsets(robot_spec):
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg

    cfg = ClothCfg()
    names = list(ROBOT.joint_names_canonical)
    pos = torch.zeros(1, len(names))
    limits = torch.stack((pos - 2, pos + 2), dim=-1)
    controller = JointController(cfg, names, pos, limits, 1 / 60, robot_spec=robot_spec)
    assert controller._perm_lab_to_canon.tolist() == list(reversed(range(len(names))))
    effort = controller.commands(pos).effort
    assert effort[0, 7] == 0.5
    assert torch.count_nonzero(effort) == 1

    cloth = grid.make_grid(torch.tensor([[0.2, 0.15, 0.01]]), 7, torch.tensor([0.01]))
    task = ClothTask(cfg, cloth, torch.zeros(1, 3), controller._arm_ids, controller._hand_ids,
                     robot_spec=robot_spec)
    observer = ObservationBuilder(cfg, controller, task, robot_spec=robot_spec)
    torch.testing.assert_close(task._fingertip_offsets, torch.tensor(robot_spec.fingertip_offsets))
    torch.testing.assert_close(observer._fingertip_offsets, task._fingertip_offsets)
    state = NS(palm_pos_w=torch.zeros(1, 3), palm_quat_w=torch.tensor([[1., 0., 0., 0.]]))
    torch.testing.assert_close(observer.palm_center_w(state), torch.tensor([[0.1, 0.2, 0.3]]))


def test_injected_body_names_control_readout_selection(robot_spec):
    cfg = NS(task=NS(release=NS(contact_bodies="")), reset=NS(table_reset_z=.38))
    backend = ClothBackend(cfg, robot_spec=robot_spec)
    backend.device, backend.num_envs = "cpu", 1
    backend._dims = torch.tensor([[.2, .15, .01]])
    backend._point_resolution = lambda: 7
    names = [*reversed(robot_spec.fingertip_link_names), robot_spec.palm_body_name]
    backend.robot = NS(find_bodies=lambda requested, preserve_order: ([names.index(v) for v in requested], requested))
    backend.initialize()
    assert backend._palm_id == 5
    assert backend._fingertip_ids == [4, 3, 2, 1, 0]
    assert backend._hand_body_ids == [4, 3, 2, 1, 0, 5]


def test_vbd_rejects_custom_robot_before_simulation_start(robot_spec):
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv

    cfg = NS(task=NS(check=lambda: None, release=NS(enabled=False, touch="origin")), backend="vbd")
    env = ClothEnv.__new__(ClothEnv)
    env._is_closed = True  # No simulator exists when constructor validation fails.
    with pytest.raises(ValueError, match="only by the rigid"):
        env.__init__(cfg, robot_spec=robot_spec)


def test_environment_passes_profile_and_sizes_spaces_before_scene_start(monkeypatch, robot_spec):
    from isaacsimenvs.tasks.cloth.backends.rigid import backend as rigid
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv

    class StopBeforeSimulation(Exception):
        pass

    captured = {}

    class Backend:
        def __init__(self, cfg, designs, *, robot_spec):
            captured.update(cfg=cfg, designs=designs, robot_spec=robot_spec)

        def startup(self):
            raise StopBeforeSimulation

    monkeypatch.setattr(rigid, "RigidBackend", Backend)
    # Smaller profile makes hard-coded default joint dimensions detectable.
    robot_spec = replace(robot_spec, joint_names_canonical=("a", "b"))
    cfg = NS(task=NS(check=lambda: None, release=NS(enabled=False, touch="origin")), backend="rigid", rigid_cloth=NS(designs=["single_hinge.yaml"]),
             obs=NS(obs_list=["joint_pos", "palm_pos"], state_list=["joint_vel"]))
    env = ClothEnv.__new__(ClothEnv)
    env._is_closed = True
    with pytest.raises(StopBeforeSimulation):
        env.__init__(cfg, robot_spec=robot_spec)
    assert captured["robot_spec"] is robot_spec
    assert captured["designs"] == cfg.rigid_cloth.designs
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (2, 5, 2)
