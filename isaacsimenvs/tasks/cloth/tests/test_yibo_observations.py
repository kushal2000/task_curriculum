"""Frozen branch functions verify the finetuned actor's observation contract.

Reference functions below are extracted verbatim from commit
2b82faf1c73f5782068f67eac9cd6085aa1e27e7: cloth_adapter.py,
fold_pool_task.py and play/utils/obs_utils.py. Keep this independent oracle frozen.
"""
import ast
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs import REPO_ROOT
from isaacsimenvs.tasks.cloth.mdp.geometry import grid
from isaacsimenvs.tasks.cloth.mdp.geometry.yibo import yibo_object_features
from isaacsimenvs.tasks.cloth.mdp.observations import ObservationBuilder
from isaacsimenvs.tasks.cloth.tests.test_cloth_utils import _obs_env


def quat_apply(q, v):
    # Independent standard wxyz rotation used by the old observation builder.
    t = 2 * torch.cross(q[:, 1:], v, dim=-1)
    return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)


def _mat_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` to wxyz quaternions, via the trace formulation.

    Uses the largest-diagonal branch rather than the naive trace form, which loses precision (and
    can divide by ~0) when the rotation approaches 180 degrees -- exactly the case a completed fold
    produces.
    """
    n = r.shape[0]
    q = torch.zeros((n, 4), device=r.device, dtype=r.dtype)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]

    big = trace > 0
    if bool(big.any()):
        sq = torch.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * sq
        q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / sq
        q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / sq
        q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / sq
    rest = ~big
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = rest & (r[:, i, i] >= r[:, j, j]) & (r[:, i, i] >= r[:, k, k])
        if not bool(sel.any()):
            continue
        sq = torch.sqrt(1.0 + r[sel, i, i] - r[sel, j, j] - r[sel, k, k]) * 2.0
        q[sel, 0] = (r[sel, k, j] - r[sel, j, k]) / sq
        q[sel, 1 + i] = 0.25 * sq
        q[sel, 1 + j] = (r[sel, j, i] + r[sel, i, j]) / sq
        q[sel, 1 + k] = (r[sel, k, i] + r[sel, i, k]) / sq
    # eps guards a zero-norm quaternion, which a degenerate fit can produce.
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)

def fit_rotation_wxyz(self) -> torch.Tensor:
        """Least-squares rotation from rest corners to current corners, as wxyz. ``(N, 4)``."""
        cur = self._particles()[:, self._corner_idx, :]                 # (N, 4, 3)
        q = cur - cur.mean(dim=1, keepdim=True)                         # centred current
        p = self._corner_rest - self._corner_rest.mean(dim=0, keepdim=True)  # centred rest (4, 3)

        # H = P^T Q, then R = V diag(1, 1, det(VU^T)) U^T. The determinant term forbids a
        # reflection: without it a badly deformed patch can fit an improper "rotation", which is a
        # mirror -- and a mirrored frame fed to a policy is worse than a wrong one.
        h = torch.einsum("ki,nkj->nij", p, q)
        # A crumpled patch can drive the corners toward collinear or coincident, leaving H rank
        # deficient. The SVD then returns an arbitrary rotation and the quaternion can normalise
        # to NaN, which reaches the policy as a non-finite observation and surfaces far away as
        # "normal expects all elements of std >= 0.0" from the action sampler. Ridge the diagonal
        # so a degenerate patch degrades to identity instead of to NaN.
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        det = torch.linalg.det(v @ u.transpose(-2, -1))
        d = torch.ones((cur.shape[0], 3), device=cur.device)
        d[:, 2] = det
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)               # (N, 3, 3)
        quat = _mat_to_wxyz(r)

        # Last line of defence: never hand a non-finite orientation to the policy. An identity
        # fallback is wrong but bounded; a NaN poisons the whole observation vector and kills the
        # run several layers away from the cause.
        bad = ~torch.isfinite(quat).all(dim=-1)
        if bool(bad.any()):
            quat = quat.clone()
            quat[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype)
        return quat

def _stationary_frame(self, parts: torch.Tensor):
        cur = parts[:, self._stationary_idx, :]
        centroid = cur.mean(dim=1)
        q = cur - centroid.unsqueeze(1)
        h = torch.einsum("nki,nkj->nij", self._st_rest_centred, q)
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        d = torch.ones((cur.shape[0], 3), device=cur.device, dtype=cur.dtype)
        d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
        bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
        if bool(bad.any()):
            r = r.clone()
            r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
        return r, centroid

def _folded_w(self, folded_rest: torch.Tensor) -> torch.Tensor:
        r, centroid = self._stationary_frame(self._particles_w())
        local = folded_rest - self._st_rest_centroid.unsqueeze(1)
        return torch.einsum("nij,nmj->nmi", r, local) + centroid.unsqueeze(1)

def _keypoints_world(
    center_pos: torch.Tensor,    # (N, 3)
    center_rot: torch.Tensor,    # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, K, 3)
) -> torch.Tensor:
    """Rotate + translate object-frame keypoints."""
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)

# End of frozen source. Native test states below do not use the production pose fit.
def _case(resolution=7, deform=False):
    env, state, old_observer = _obs_env(9, False)
    dims = torch.tensor([[.12 + .02*i, .25 - .01*i, .002] for i in range(9)])
    cloth = grid.make_grid(dims, resolution, torch.full((9,), .002))
    env._cloth_grid = cloth
    yaw = torch.deg2rad(torch.tensor([0., 90., 180.]).repeat_interleave(3))
    angle = torch.deg2rad(torch.tensor([0., 90., 180.]).repeat(3))
    points = cloth.rest.clone()
    moving = points[..., 0] > 1e-9
    x = points[..., 0].clone()
    points[..., 0] = torch.where(moving, x*angle.cos()[:, None], x)
    points[..., 2] = torch.where(moving, x*angle.sin()[:, None], 0.)
    x, y = points[..., 0].clone(), points[..., 1].clone()
    points[..., 0] = yaw.cos()[:, None]*x - yaw.sin()[:, None]*y
    points[..., 1] = yaw.sin()[:, None]*x + yaw.cos()[:, None]*y
    points += torch.tensor([.03, -.04, .55])
    if deform:
        points[:, cloth.corner_idx[-1]] += torch.tensor([.012, -.007, .018])
    state.points_w = points
    state.points_vel_w = torch.arange(points.numel()).reshape_as(points)*.0001
    # Exact crease landmarks are intentionally irrelevant to this historical mode.
    state.keypoints_w = torch.full((9, 4, 3), 12.)
    env.cfg.obs.keypoint_mode = 'yibo'
    env.cfg.obs.keypoint_scale = 1.5
    env.cfg.obs.yibo_contact_thickness = .016
    env.cfg.obs.yibo_goal_lift = 0.
    observer = ObservationBuilder(env.cfg, old_observer.controller, env)
    return env, state, observer


def _reference(cloth, state, lift=0.):
    idx = grid.corner_indices(cloth.resolution)
    corners = state.points_w[:, idx]
    rest = cloth.rest[:, idx]
    quats = []
    for i in range(len(rest)):
        old = NS(_particles=lambda i=i: state.points_w[i:i+1],
                 _corner_idx=idx, _corner_rest=rest[i])
        quats.append(fit_rotation_wxyz(old))
    quat = torch.cat(quats)
    offsets = rest - rest.mean(1, keepdim=True)
    pos = corners.mean(1)
    old = NS(_stationary_idx=cloth.stationary_idx, _st_rest_centred=cloth.st_rest_centred,
             _st_rest_centroid=cloth.st_rest_centroid, _particles_w=lambda: state.points_w)
    old._stationary_frame = lambda parts: _stationary_frame(old, parts)
    folded = rest.clone()
    folded[..., 0] *= -1
    folded[..., 2] += lift
    goal_pos = _folded_w(old, folded).mean(1)
    r, _ = old._stationary_frame(state.points_w)
    # Lab 3's goal-viz write used xyzw, subsequently interpreted as wxyz.
    goal_rot = r @ torch.diag(torch.tensor([-1., 1., -1.]))
    goal_q = _mat_to_wxyz(goal_rot)[:, [1, 2, 3, 0]]
    return (pos, quat, state.points_vel_w[:, idx].mean(1), torch.zeros_like(pos),
            _keypoints_world(pos, quat, offsets), _keypoints_world(goal_pos, goal_q, offsets))


@pytest.mark.parametrize('resolution', [6, 7, 9])
@pytest.mark.parametrize('deform', [False, True])
def test_yibo_geometry_matches_frozen_branch_across_yaw_and_fold(resolution, deform):
    env, state, observer = _case(resolution, deform)
    reference = _reference(env._cloth_grid, state)
    actual = yibo_object_features(env._cloth_grid, state)
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    # The sign is observable to the actor: never compare quaternion absolute values.
    observer._object_quat_previous[:] = -reference[1]
    observer._object_quat_valid[:] = True
    features = observer.object_features(state, env.derive(state))
    torch.testing.assert_close(features.quat_w, reference[1], atol=2e-6, rtol=2e-6)
    if deform:
        assert not torch.allclose(features.keypoints_w, state.points_w[:, env._cloth_grid.corner_idx], atol=1e-4)


def test_yibo_actor_uses_inset_planar_markers_full_dimensions_and_fixed_felt_thickness():
    env, state, observer = _case()
    view = env.derive(state)
    physical_targets = view.targets_w.clone()
    features = observer.object_features(state, view)
    # First env is flat at zero yaw: original inset x=L/6; center is L/3.
    local = features.keypoints_w[0] - torch.tensor([.03, -.04, .55])
    length, width = env._cloth_grid.dims[0, :2]
    expected = torch.tensor([[length/6, -width/2, 0.], [length/6, width/2, 0.],
                             [length/2, -width/2, 0.], [length/2, width/2, 0.]])
    torch.testing.assert_close(local, expected)
    policy = observer.build(state, view, env.episode_length_buf)['policy']
    scales = env._cloth_grid.dims.clone()
    scales[:, 2] = .016
    torch.testing.assert_close(policy[:, -3:], scales/.04)
    torch.testing.assert_close(policy[:, 88:92], features.quat_w[:, [1, 2, 3, 0]])
    palm = observer.palm_center_w(state)
    torch.testing.assert_close(policy[:, 107:119], (features.keypoints_w-palm[:, None]).flatten(1))
    torch.testing.assert_close(policy[:, 119:131], (features.keypoints_w-features.targets_w).flatten(1))
    observer.keypoint_scale = 3.
    torch.testing.assert_close(observer.keypoints_w(view, state)[0], features.keypoints_w)
    env.cfg.obs.yibo_goal_lift = .014
    lifted = observer.object_features(state, view)
    torch.testing.assert_close(lifted.targets_w[..., 2]-features.targets_w[..., 2], torch.full((9, 4), .014))
    torch.testing.assert_close(env.derive(state).targets_w, physical_targets)


@pytest.mark.parametrize('field,value', [('yibo_contact_thickness', 0.), ('yibo_contact_thickness', float('nan')),
                                         ('yibo_goal_lift', float('inf'))])
def test_yibo_rejects_invalid_observation_dimensions(field, value):
    env, _, observer = _case()
    setattr(env.cfg.obs, field, value)
    with pytest.raises(ValueError, match='yibo'):
        ObservationBuilder(env.cfg, observer.controller, env)


def test_yibo_goal_relabelling_matches_actual_lab3_math_source():
    # The historical goal writer imported this xyzw routine unpatched. Read only
    # its two pure functions so the oracle also runs under the Lab 2 test runtime.
    source = (REPO_ROOT / 'third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py').read_text()
    nodes = ast.parse(source)
    functions = {n.name: n for n in nodes.body if isinstance(n, ast.FunctionDef)}
    namespace = {'torch': torch}
    for name in ('_sqrt_positive_part', 'quat_from_matrix'):
        exec(ast.get_source_segment(source, functions[name]), namespace)  # noqa: S102 -- trusted local math source
    env, state, _ = _case()
    cloth = env._cloth_grid
    owner = NS(_stationary_idx=cloth.stationary_idx, _st_rest_centred=cloth.st_rest_centred)
    stationary_r, stationary_c = _stationary_frame(owner, state.points_w)
    offsets = cloth.rest[:, cloth.corner_idx]
    offsets = offsets-offsets.mean(1, keepdim=True)
    goal_q = namespace['quat_from_matrix'](stationary_r @ torch.diag(torch.tensor([-1., 1., -1.])))
    goal = yibo_object_features(cloth, state)[-1]
    expected_offsets = _keypoints_world(torch.zeros_like(stationary_c), goal_q, offsets)
    torch.testing.assert_close(goal-goal.mean(1, keepdim=True), expected_offsets, atol=2e-6, rtol=2e-6)


def test_yibo_without_thickness_override_reports_actual_geometry_per_env():
    env, state, previous = _case()
    env.cfg.obs.yibo_contact_thickness = None
    env._cloth_grid.dims[:, 2] = torch.linspace(.002, .010, 9)
    observer = ObservationBuilder(env.cfg, previous.controller, env)
    torch.testing.assert_close(observer.object_scales, env._cloth_grid.dims/.04)
    policy = observer.build(state, env.derive(state), env.episode_length_buf)['policy']
    torch.testing.assert_close(policy[:, -1], env._cloth_grid.dims[:, 2]/.04)
    assert policy[0, -1].item() == pytest.approx(.05)
