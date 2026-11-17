"""The shared control loop owns history independently of the physics representation."""

from dataclasses import fields
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT
from isaacsimenvs.tasks.cloth.mdp.actions import JointController
from isaacsimenvs.tasks.cloth.mdp.geometry import grid, rotations
from isaacsimenvs.tasks.cloth.mdp.observations import ObservationBuilder
from isaacsimenvs.tasks.cloth.mdp.state import SimState
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask


@pytest.fixture
def components():
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg

    cfg = ClothCfg()
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0
    n, joints = 3, ROBOT.num_joints
    limits = torch.stack((-2 * torch.ones(n, joints), 2 * torch.ones(n, joints)), dim=-1)
    controller = JointController(cfg, list(ROBOT.joint_names_canonical), torch.zeros(n, joints), limits, 1 / 60)
    cloth = grid.make_grid(torch.tensor([[0.2, 0.15, 0.01]] * n), 7, torch.full((n,), 0.01))
    task = ClothTask(cfg, cloth, torch.zeros(n, 3), controller._arm_ids, controller._hand_ids)
    observer = ObservationBuilder(cfg, controller, task)
    points = grid.folded_start(cloth, torch.full((n,), torch.pi)) + torch.tensor([0, 0, 0.5])
    state = SimState(
        joint_pos=torch.zeros(n, joints),
        joint_vel=torch.zeros(n, joints),
        palm_pos_w=torch.tensor([[0, 0, 0.7]] * n),
        palm_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n),
        palm_vel_w=torch.zeros(n, 6),
        fingertip_pos_w=torch.zeros(n, 5, 3),
        fingertip_quat_w=torch.tensor([1.0, 0, 0, 0]).expand(n, 5, 4),
        hand_pos_w=torch.zeros(n, 6, 3),
        points_w=points,
        points_vel_w=torch.zeros_like(points),
        table_top_z=torch.full((n,), 0.495),
    )
    return controller, task, observer, state


def test_actions_keep_velocity_arm_and_absolute_hand_semantics(components):
    controller, _, _, _ = components
    controller.process(torch.ones(3, 27), torch.ones(3, dtype=torch.long))
    targets = controller.commands(torch.zeros(3, 27)).position
    torch.testing.assert_close(targets[:, :7], torch.full((3, 7), 0.0025))
    torch.testing.assert_close(targets[:, 7:], torch.full((3, 20), 0.2))


def test_partial_reset_preserves_other_episode_histories(components):
    controller, task, observer, state = components
    task.evaluate(state, torch.ones(3, dtype=torch.long), 600)
    controller._action_queue.fill_(4)
    observer._obs_queue.fill_(5)
    observer._object_state_queue.fill_(6)
    ids = torch.tensor([1])
    initial = task.sample_reset(ids, *controller.sample_reset(ids))
    task.reset(ids)
    observer.reset(ids)
    assert task._successes.tolist() == [1, 0, 1]
    assert task._prev_episode_successes.tolist() == [0, 1, 0]
    assert torch.count_nonzero(initial.joint_pos) == 0
    for queue in (controller._action_queue, observer._obs_queue, observer._object_state_queue):
        assert torch.count_nonzero(queue[1]) == 0
        assert torch.count_nonzero(queue[[0, 2]]) == queue[[0, 2]].numel()


def test_queries_do_not_advance_task_history(components):
    _, task, observer, state = components
    lengths = torch.ones(3, dtype=torch.long)
    transition = task.evaluate(state, lengths, 600)
    for _ in range(3):
        view = task.derive(state)
        assert task.is_folded(state, view).all()
        obs = observer.build(state, view, lengths)
        assert obs["policy"].shape == (3, 134)
    assert task._successes.tolist() == [1, 1, 1]
    torch.testing.assert_close(task.reward_buf, transition.reward)


def test_bad_observation_is_terminated_on_the_next_step(components):
    _, task, observer, state = components
    lengths = torch.ones(3, dtype=torch.long)
    state.points_w[1, 0, 0] = float("nan")
    obs = observer.build(state, task.derive(state), lengths)
    observer.guard_finite(obs, state, lengths)
    assert task._nonfinite_envs.tolist() == [False, True, False]
    assert all(torch.isfinite(value).all() for value in obs.values())
    task.evaluate(state, lengths, 600)
    assert task._termination_reasons["nonfinite"].tolist() == [False, True, False]


def test_invalid_frame_resets_without_relying_on_policy_observations(components):
    """Yibo omits angular velocity, so termination must inspect the task fit itself."""
    _, task, _, state = components
    view = task.derive(state)
    view.ang_vel_w[1] = float("nan")
    task.episode_length_buf = torch.ones(3, dtype=torch.long)
    task.max_episode_length = 600
    terminated, _ = task.compute_terminations(state, view)
    assert terminated[1]
    assert task._termination_reasons["nonfinite"].tolist() == [False, True, False]
    assert task._successes.tolist() == [1, 0, 1]
    assert task._bonus_mask.tolist() == [True, False, True]


def test_rigid_and_vbd_robot_readouts_share_one_convention():
    """Tensor/wxyz and ProxyArray/xyzw asset data produce identical public state."""
    torch.manual_seed(0)
    body_state = torch.randn(2, 6, 13)
    body_state[..., 3:7] = torch.nn.functional.normalize(body_state[..., 3:7], dim=-1)
    joint_pos, joint_vel = torch.randn(2, 27), torch.randn(2, 27)
    points, velocities = torch.randn(2, 49, 3), torch.randn(2, 49, 3)

    def backend(order):
        result = ClothBackend(NS())
        result.quaternion_order = order
        bodies = body_state.clone()
        if order == "xyzw":
            bodies[..., 3:7] = rotations.to_xyzw(bodies[..., 3:7])
        wrap = (lambda value: NS(torch=value)) if order == "xyzw" else (lambda value: value)
        result.robot = NS(data=NS(body_state_w=wrap(bodies), joint_pos=wrap(joint_pos), joint_vel=wrap(joint_vel)))
        result._palm_id, result._fingertip_ids, result._hand_body_ids = 0, [1, 2, 3, 4, 5], list(range(6))
        result.scene = NS(env_origins=torch.tensor([[0.0, 0, 0], [1.0, 2.0, 3.0]]))
        result._table_z, result._table_half_height = torch.tensor([0.4, 0.5]), 0.01
        result.cloth_points = lambda: (points, velocities)
        return result.read_state()

    rigid, vbd = backend("wxyz"), backend("xyzw")
    for field in fields(SimState):
        torch.testing.assert_close(getattr(rigid, field.name), getattr(vbd, field.name), rtol=0, atol=0)
    torch.testing.assert_close(rigid.table_top_z, torch.tensor([0.41, 3.51]))
