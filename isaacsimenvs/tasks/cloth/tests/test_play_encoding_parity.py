"""Native cloth geometry must encode the same manipulated object as the Play prior."""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs import REPO_ROOT
from isaacsimenvs.tasks.cloth.mdp.actions import JointController
from isaacsimenvs.tasks.cloth.mdp.geometry import grid
from isaacsimenvs.tasks.cloth.mdp.observations import FIELD_SIZES, ObservationBuilder
from isaacsimenvs.tasks.cloth.mdp.state import SimState
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask
from isaacsimenvs.tasks.play.robots import FLEXIV_WUJI_RIGHT_HUMAN_LIKE as ROBOT


@pytest.fixture(scope="module")
def physx_math():
    """Load official Lab 2 wxyz math, independently of cloth's quaternion helpers."""
    paths = sorted((REPO_ROOT / '.venv/isaacsim/lib').glob(
        'python*/site-packages/isaaclab/source/isaaclab/isaaclab/utils/math.py'))
    if not paths:
        pytest.skip('The independent PhysX reference requires installed Isaac Lab 2 math source')
    spec = importlib.util.spec_from_file_location('_play_parity_physx_math', paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def play_builder(monkeypatch, physx_math):
    # Lab 3 is kitless but its default quaternion math is xyzw; Play PhysX uses
    # wxyz. Temporarily bind its actual training-runtime functions, then restore.
    play = importlib.import_module('isaacsimenvs.tasks.play.utils.obs_utils')
    for name in ('quat_apply', 'quat_mul', 'quat_from_angle_axis', 'convert_quat'):
        monkeypatch.setattr(play, name, getattr(physx_math, name))
    return play


def _world(points, orientation, translation, math2):
    n, k, _ = points.shape
    q = orientation[:, None].expand(n, k, 4).reshape(-1, 4)
    return math2.quat_apply(q, points.reshape(-1, 3)).reshape(n, k, 3) + translation[:, None]


def _case(angle_deg, robot_root, explicit, math2):
    from isaacsimenvs.scripts.evaluation.utils import disable_randomization
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg

    torch.manual_seed(871)
    n = 3
    cfg = ClothCfg()
    cfg.obs.keypoint_scale = 1.5
    cfg.robot_root_pos = robot_root
    disable_randomization(cfg)
    # Play's modern no-noise path also checks these pose/point-DR settings.
    cfg.domain_randomization.point_dr_mode = 'off'
    origins = torch.tensor([[0., 0., 0.], [5., -2., 1.], [-3., 4., -.5]])
    dims = torch.tensor([[.2, .16, .004], [.3, .12, .012], [.12, .24, .008]])
    cloth = grid.make_grid(dims, 9, dims[:, 2])
    physical_rest = dims[:, None] * torch.tensor([[[0., -.5, 0.], [0., .5, 0.], [.5, -.5, 0.], [.5, .5, 0.]]])
    cloth.keypoint_rest = physical_rest if explicit else None
    yaw = torch.tensor([.7, -1.1, 2.2])
    yaw_q = torch.stack((torch.cos(yaw/2), torch.zeros(n), torch.zeros(n), torch.sin(yaw/2)), dim=1)
    angle = torch.full((n,), math.radians(angle_deg))
    fold_q = torch.stack((torch.cos(angle/2), torch.zeros(n), -torch.sin(angle/2), torch.zeros(n)), dim=1)
    moving_q = math2.quat_mul(yaw_q, fold_q)
    # Keep the native continuous -Y fold branch, including its w≈0 endpoint.
    goal_q = math2.quat_mul(yaw_q, torch.tensor([0., 0., 1., 0.]).expand(n, 4))
    translation = origins + torch.tensor([[.05, .2, .45], [-.1, .15, .5], [.12, -.1, .6]])
    pivot = torch.zeros(n, 3)
    pivot[:, 2] = dims[:, 2]/2
    body_offset = pivot - math2.quat_apply(fold_q, pivot)
    local_moving = _world(cloth.rest, fold_q, body_offset, math2)
    is_moving = cloth.rest[..., 0] > 0
    local_points = torch.where(is_moving[..., None], local_moving, cloth.rest)
    points = _world(local_points, yaw_q, translation, math2)
    exact_corners = _world(_world(physical_rest, fold_q, body_offset, math2), yaw_q, translation, math2)
    center_rest = torch.zeros(n, 1, 3)
    center_rest[:, 0, 0] = dims[:, 0]/4
    object_center = _world(_world(center_rest, fold_q, body_offset, math2), yaw_q, translation, math2)[:, 0]
    goal_rest = center_rest.clone()
    goal_rest[..., 0] *= -1
    goal_rest[..., 2] = dims[:, 2:3]
    goal_center = _world(goal_rest, yaw_q, translation, math2)[:, 0]
    angular = math2.quat_apply(yaw_q, torch.tensor([[0., -.2, 0.], [0., .3, 0.], [0., -.4, 0.]]))
    linear_base = torch.tensor([[.02, -.01, 0.], [-.03, .01, .02], [0., .03, -.01]])
    pivot_world = math2.quat_apply(yaw_q, pivot) + translation
    velocities = torch.where(is_moving[..., None],
        torch.cross(angular[:, None].expand_as(points), points-pivot_world[:, None], dim=-1), torch.zeros_like(points))
    velocities += linear_base[:, None]
    object_linear = linear_base + torch.cross(angular, object_center-pivot_world, dim=-1)

    names = list(reversed(ROBOT.arm_joint_names)) + list(reversed(ROBOT.hand_joint_names))
    lower = torch.linspace(-1.9, -.1, 27)
    upper = torch.linspace(.5, 2.4, 27)
    limits = torch.stack((lower, upper), dim=-1).expand(n, -1, -1)
    joints = lower + torch.rand(n, 27)*(upper-lower)
    joint_vel = torch.randn(n, 27)*.2
    controller = JointController(cfg, names, joints, limits, 1/60, robot_spec=ROBOT)
    controller._prev_targets = joints + .03*torch.randn_like(joints)
    bodies = torch.randn(n, 6, 13)*.1
    bodies[..., :3] += origins[:, None] + torch.tensor([0., .3, .8])
    bodies[..., 3:7] = torch.nn.functional.normalize(torch.randn(n, 6, 4), dim=-1)
    state = SimState(joints, joint_vel, bodies[:, 0, :3], bodies[:, 0, 3:7], bodies[:, 0, 7:13],
        bodies[:, 1:, :3], bodies[:, 1:, 3:7], bodies[..., :3], points, velocities,
        origins[:, 2]+.38, exact_corners if explicit else None)
    task = ClothTask(cfg, cloth, origins, list(range(7)), list(range(7, 27)), robot_spec=ROBOT)
    task._unfold_env[:] = False
    steps = torch.tensor([0, 17, 299])
    return NS(cfg=cfg, state=state, task=task, controller=controller, steps=steps, bodies=bodies,
              object_center=object_center, object_quat=moving_q, object_linear=object_linear, angular=angular,
              goal_center=goal_center, goal_quat=goal_q, dims=dims, origins=origins)


@pytest.mark.parametrize('angle_deg', [0, 45, 90, 180])
@pytest.mark.parametrize('robot_root', [(0., .8, 0.), (0., .51, .20)])
@pytest.mark.parametrize('explicit', [True, False], ids=['rigid-landmarks', 'vbd-grid'])
def test_native_cloth_actor_encoding_matches_actual_play_builder(angle_deg, robot_root, explicit, physx_math, play_builder):
    case = _case(angle_deg, robot_root, explicit, physx_math)
    cfg, task, controller, state = case.cfg, case.task, case.controller, case.state
    view = task.derive(state)
    observer = ObservationBuilder(cfg, controller, task, robot_spec=ROBOT)
    half_dims = case.dims.clone()
    half_dims[:, 0] /= 2
    # Independent reference uses the literal four points from original Play,
    # generated from the analytic moving-body/goal poses, not cloth's features.
    corners = torch.tensor(((1., 1., 1.), (1., 1., -1.), (-1., -1., 1.), (-1., -1., -1.)))
    offsets = corners[None]*half_dims[:, None]*(1.5/2)
    play_env = NS(cfg=cfg, robot_spec=ROBOT, num_envs=3, device='cpu', scene=NS(env_origins=case.origins),
        robot=NS(data=NS(joint_pos=state.joint_pos, joint_vel=state.joint_vel, body_state_w=case.bodies)),
        object=NS(data=NS(root_pos_w=case.object_center, root_quat_w=case.object_quat,
                         root_lin_vel_w=case.object_linear, root_ang_vel_w=case.angular)),
        goal_viz=NS(data=NS(root_pos_w=case.goal_center, root_quat_w=case.goal_quat)),
        _perm_lab_to_canon=controller._perm_lab_to_canon, _joint_lower_canon=controller._joint_lower_canon,
        _joint_upper_canon=controller._joint_upper_canon, _prev_targets=controller._prev_targets,
        _palm_body_id=0, _fingertip_body_ids=list(range(1, 6)), _keypoint_offsets=offsets,
        _object_scale_multiplier=torch.ones(3, 3), _object_scale_per_env=half_dims/.04,
        _object_bias_pos=torch.zeros(3, 3), _object_bias_quat=torch.tensor([1., 0., 0., 0.]).expand(3, -1),
        _closest_keypoint_max_dist=task._closest_fold_err, _closest_fingertip_dist=task._closest_fingertip_dist,
        _lifted_object=torch.zeros(3, dtype=torch.bool), _successes=task._successes,
        episode_length_buf=case.steps, reward_buf=task.reward_buf)
    expected = play_builder.build_observations(play_env)['policy']
    actual = observer.build(state, view, case.steps)['policy']
    assert actual.shape == (3, 134)
    start = 0
    for field in cfg.obs.obs_list:
        size = 27 if field in ('joint_pos', 'joint_vel', 'prev_action_targets') else FIELD_SIZES[field]
        torch.testing.assert_close(actual[:, start:start+size], expected[:, start:start+size],
                                   atol=5e-6, rtol=0, msg=lambda msg: f'{field}: {msg}')
        start += size
    features = observer.object_features(state, view)
    # Verify physical rotation separately; the actor assertion above also checks
    # the raw quaternion sign, without aligning the reference to the result.
    torch.testing.assert_close(physx_math.matrix_from_quat(features.quat_w),
                               physx_math.matrix_from_quat(case.object_quat), atol=3e-6, rtol=0)
    torch.testing.assert_close(features.pos_w, case.object_center, atol=2e-6, rtol=0)
    torch.testing.assert_close(features.lin_vel_w, case.object_linear, atol=2e-6, rtol=0)
    torch.testing.assert_close(features.ang_vel_w, case.angular, atol=3e-6, rtol=0)
    torch.testing.assert_close(features.keypoints_w, _world(offsets, case.object_quat, case.object_center, physx_math),
                               atol=2e-6, rtol=0)
    torch.testing.assert_close(features.targets_w, _world(offsets, case.goal_quat, case.goal_center, physx_math),
                               atol=3e-6, rtol=0)
    if angle_deg == 180:
        torch.testing.assert_close(actual[:, 119:131], torch.zeros(3, 12), atol=3e-6, rtol=0)


@pytest.mark.parametrize('explicit', [True, False], ids=['rigid-landmarks', 'vbd-grid'])
def test_object_quaternion_stays_on_native_branch_through_half_turn(explicit, physx_math):
    initial = _case(0, (0., .8, 0.), explicit, physx_math)
    observer = ObservationBuilder(initial.cfg, initial.controller, initial.task, robot_spec=ROBOT)
    previous = None
    for angle in (0., 179., 179.99, 180., 179.99):
        case = _case(angle, (0., .8, 0.), explicit, physx_math)
        view = initial.task.derive(case.state)
        features = observer.object_features(case.state, view)
        # Raw quaternion equality matters: q and -q are not identical policy inputs.
        torch.testing.assert_close(features.quat_w, case.object_quat, atol=3e-6, rtol=0)
        if previous is not None:
            assert torch.all((previous*features.quat_w).sum(dim=-1) > 0)
        repeated = observer.object_features(case.state, view)
        torch.testing.assert_close(repeated.quat_w, features.quat_w, atol=0, rtol=0)
        previous = features.quat_w.clone()


def test_partial_reset_clears_only_selected_quaternion_history_and_handles_folded_start(physx_math):
    case = _case(180, (0., .8, 0.), True, physx_math)
    case.task._unfold_env[:] = True
    observer = ObservationBuilder(case.cfg, case.controller, case.task, robot_spec=ROBOT)
    view = case.task.derive(case.state)
    first = observer.object_features(case.state, view)
    torch.testing.assert_close(first.quat_w, case.object_quat, atol=3e-6, rtol=0)
    previous = observer._object_quat_previous.clone()
    observer.reset(torch.tensor([1]))
    assert observer._object_quat_valid.tolist() == [True, False, True]
    torch.testing.assert_close(observer._object_quat_previous[[0, 2]], previous[[0, 2]], atol=0, rtol=0)
    restarted = observer.object_features(case.state, view)
    assert observer._object_quat_valid.all()
    torch.testing.assert_close(restarted.quat_w, case.object_quat, atol=3e-6, rtol=0)
