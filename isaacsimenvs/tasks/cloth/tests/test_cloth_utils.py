"""cloth/utils without a simulator: geometry, quaternions, actions, obs layout, reward, termination."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.utils import (
    action_utils, geometry, obs_utils, quat, reward_utils, robot, state, termination_utils,
)

POLICY = ("joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "object_rot",
          "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales")
CRITIC = ("joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "palm_vel",
          "object_rot", "object_vel", "fingertip_pos_rel_palm", "keypoints_rel_palm",
          "keypoints_rel_goal", "object_scales", "closest_keypoint_max_dist",
          "closest_fingertip_dist", "lifted_object", "progress", "successes", "reward")


def random_quat(n: int) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)


def sheet(dims=((0.24, 0.12, 0.01),), resolution=7, gap=None) -> geometry.Sheet:
    d = torch.tensor(dims)
    return geometry.make_sheet(d, resolution, d[:, 2] if gap is None else torch.tensor(gap))


def place(points: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return quat.apply(q.unsqueeze(1), points) + t.unsqueeze(1)


# ------------------------------------------------------------------ layout


def test_obs_layout_is_plays_for_this_robot():
    assert obs_utils.obs_dim(POLICY, robot.ROBOT.num_joints) == 134
    assert obs_utils.obs_dim(CRITIC, robot.ROBOT.num_joints) == 156


def test_robot_canonical_order_and_fingertip_order_are_the_priors():
    spec = robot.ROBOT
    assert spec.joint_names_canonical[:7] == tuple(f"joint{i}" for i in range(1, 8))
    assert spec.joint_names_canonical[7:11] == ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip")
    assert spec.fingertip_link_names == ("r_index_finger_distal", "r_middle_finger_distal", "r_pinky_distal",
                                         "r_ring_finger_distal", "r_thumb_distal")
    assert len(spec.hand_spring_stiffness) == 20 and spec.num_joints == 27


# ------------------------------------------------------------------ quaternions


def test_quat_apply_matches_its_matrix_and_round_trips():
    torch.manual_seed(0)
    q = random_quat(256)
    v = torch.randn(256, 3)
    assert torch.allclose(quat.apply(q, v), torch.einsum("nij,nj->ni", quat.to_matrix(q), v), atol=1e-5)
    back = quat.from_matrix(quat.to_matrix(q))
    assert torch.allclose(back.abs(), q.abs(), atol=1e-4)  # equal up to sign
    assert torch.allclose(quat.to_wxyz(quat.to_xyzw(q)), q)


def test_quat_from_matrix_is_exact_at_180_deg():
    """A completed fold is a half turn: the plain trace formula divides by ~0 there."""
    for axis in torch.eye(3):
        q = quat.from_angle_axis(torch.tensor([math.pi]), axis.unsqueeze(0))
        assert torch.allclose(quat.to_matrix(quat.from_matrix(quat.to_matrix(q))), quat.to_matrix(q), atol=1e-6)


def test_quat_mul_composes_rotations():
    torch.manual_seed(1)
    a, b, v = random_quat(64), random_quat(64), torch.randn(64, 3)
    assert torch.allclose(quat.apply(quat.mul(a, b), v), quat.apply(a, quat.apply(b, v)), atol=1e-5)


# ------------------------------------------------------------------ geometry


@pytest.mark.parametrize("dims", [(0.10, 0.30, 0.002), (0.30, 0.10, 0.020), (0.17, 0.23, 0.0092)])
def test_a_perfect_fold_scores_zero_and_halves_the_footprint(dims):
    s = sheet((dims,))
    folded = geometry.folded_start(s, torch.tensor([math.pi]))
    r, c = geometry.stationary_frame(s, folded)
    kp = folded[:, s.kp_idx]
    targets = geometry.keypoint_targets(s, r, c, torch.tensor([False]))
    assert float(geometry.fold_error(kp, targets)) < 1e-6
    fp = float(geometry.footprint_ratio(s, folded, r))
    assert 0.5 <= fp < 0.65


def test_fold_targets_follow_the_sheet_through_rotation_and_translation():
    torch.manual_seed(2)
    s = sheet(((0.2, 0.15, 0.01),) * 32)
    folded = geometry.folded_start(s, torch.full((32,), math.pi))
    q, t = random_quat(32), torch.randn(32, 3)
    moved = place(folded, q, t)
    r, c = geometry.stationary_frame(s, moved)
    assert torch.allclose(r, quat.to_matrix(q), atol=1e-4)
    err = geometry.fold_error(moved[:, s.kp_idx], geometry.keypoint_targets(s, r, c, torch.zeros(32, dtype=torch.bool)))
    assert float(err.max()) < 1e-5
    assert torch.allclose(geometry.footprint_ratio(s, moved, r), geometry.footprint_ratio(s, folded, torch.eye(3).expand(32, 3, 3)), atol=1e-5)


def test_a_slid_flat_sheet_is_not_folded():
    """Negative control: pushing the flat sheet half its length puts the moving half where a
    WORLD-fixed target would be; the sheet-relative target is not fooled."""
    s = sheet()
    flat = s.rest + torch.tensor([-0.12, 0.0, 0.0])
    r, c = geometry.stationary_frame(s, flat)
    err = geometry.fold_error(flat[:, s.kp_idx], geometry.keypoint_targets(s, r, c, torch.tensor([False])))
    assert float(err) > 0.1
    assert float(geometry.footprint_ratio(s, flat, r)) == pytest.approx(1.0)


def test_unfold_targets_are_the_flat_sheet():
    s = sheet()
    r, c = geometry.stationary_frame(s, s.rest)
    err = geometry.fold_error(s.rest[:, s.kp_idx], geometry.keypoint_targets(s, r, c, torch.tensor([True])))
    assert float(err) < 1e-6


def test_folded_start_lifts_by_the_ply_gap_without_overlap():
    s = sheet(gap=[0.004])
    folded = geometry.folded_start(s, torch.tensor([math.pi]))[0]
    mv = folded[s.moving_idx]
    assert torch.allclose(mv[:, 2], torch.full_like(mv[:, 2], 0.004), atol=1e-7)
    assert torch.allclose(mv[:, 0], -s.rest[0, s.moving_idx, 0], atol=1e-7)


def test_rectangles_differ_per_env():
    s = sheet(((0.10, 0.30, 0.002), (0.30, 0.10, 0.02)))
    ext = s.rest.amax(dim=1) - s.rest.amin(dim=1)
    assert torch.allclose(ext[:, :2], torch.tensor([[0.10, 0.30], [0.30, 0.10]]))


def test_sheet_frame_recovers_a_rigid_spin():
    s = sheet(((0.2, 0.2, 0.01),) * 4)
    omega = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.3, -0.4, 0.5], [0.0, 0.0, 0.0]])
    pts = s.rest + torch.tensor([0.1, 0.2, 0.3])
    vel = torch.cross(omega.unsqueeze(1).expand_as(pts), pts - pts.mean(dim=1, keepdim=True), dim=-1) + 0.5
    r, _ = geometry.stationary_frame(s, pts)
    _, _, lin, ang = geometry.sheet_frame(pts, vel, r)
    assert torch.allclose(ang, omega, atol=1e-4)
    assert torch.allclose(lin, torch.full_like(lin, 0.5), atol=1e-6)


def test_grid_halves_exclude_the_crease_and_corners_are_on_the_moving_half():
    for res in (6, 7, 9):
        mv, st = geometry.half_indices(res, True), geometry.half_indices(res, False)
        assert not set(mv) & set(st)
        assert set(geometry.corner_indices(res)) <= set(mv)
        assert len(geometry.grid_triangles(res)) == 6 * (res - 1) ** 2


# ------------------------------------------------------------------ actions


def test_joint_maps_round_trip_and_reject_a_foreign_robot():
    canon = list(robot.ROBOT.joint_names_canonical)
    lab = canon[:7] + sorted(canon[7:])
    m = action_utils.joint_maps(lab, robot.ROBOT)
    x = torch.arange(27.0)
    assert torch.equal(x[m["canon_to_lab"]][m["lab_to_canon"]], x)
    assert [lab[k] for k in m["lab_to_canon"]] == canon
    with pytest.raises(ValueError):
        action_utils.joint_maps(lab[:-1] + ["l_pinky_dip"], robot.ROBOT)


def test_delay_flushes_at_episode_start_and_reads_a_recent_value():
    torch.manual_seed(3)
    q = torch.randn(8, 3, 2)
    v = torch.ones(8, 2)
    q, out = action_utils.delayed(q, v, torch.ones(8, dtype=torch.bool))
    assert torch.equal(out, v) and torch.equal(q, torch.ones(8, 3, 2))
    q, out = action_utils.delayed(q, 2 * v, torch.zeros(8, dtype=torch.bool))
    assert bool(((out == 1) | (out == 2)).all())


# ------------------------------------------------------------------ reward / termination


def test_ratchet_pays_only_improvement_on_the_best():
    best = torch.tensor([-1.0, 0.5, 0.5])
    gain, best = reward_utils.ratchet(best, torch.tensor([0.4, 0.3, 0.6]), 100.0)
    assert torch.allclose(gain, torch.tensor([0.0, 0.2, 0.0]))
    assert torch.allclose(best, torch.tensor([0.4, 0.3, 0.5]))


def test_reward_guard_zeros_huge_finite_rewards_not_just_nan():
    env = NS(num_envs=4, extras={}, _nonfinite_envs=torch.zeros(4, dtype=torch.bool), _insane_count=0)
    out = reward_utils.guard(env, torch.tensor([1.0, -1e6, float("nan"), 1000.0]))
    assert torch.equal(out, torch.tensor([1.0, 0.0, 0.0, 1000.0]))
    assert env._nonfinite_envs.tolist() == [False, True, True, False]


def _task_env(n=2, task="fold", release=False):
    cfg = NS(
        fold=NS(task=task, nan_policy="reset", keypoint_tolerance=0.04, max_folded_footprint=0.65, min_unfolded_footprint=0.9,
                unfold_on_table=False, on_table_max_spread=0.015, on_table_max_lift=0.015,
                release=NS(enabled=release, clearance=0.03, latch_steps=3, fold_tolerance=0.0, touch="origin",
                           release_dist_rew_scale=3000.0, persist_consecutive=True, persist_window_steps=4,
                           persist_fail_terminates=True)),
        termination=NS(success_steps=1, held_steps=10, fall_height=0.1, hand_far_distance=1.5),
        domain_randomization=NS(action_delay_max=3, obs_delay_max=3, object_state_delay_max=10),
        reset=NS(table_reset_z=0.38), observation_space=134,
    )
    env = NS(cfg=cfg, num_envs=n, device="cpu", _spec=robot.ROBOT, extras={},
             episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=600,
             scene=NS(env_origins=torch.zeros(n, 3)))
    state.allocate(env)
    return env


def _view(err, footprint, z=0.5):
    n = len(err)
    return NS(fold_error=torch.tensor(err), footprint=torch.tensor(footprint),
              pos_w=torch.tensor([[0.0, 0.0, z]] * n), fingertip_dist=torch.full((n, 5), 0.2))


def test_success_needs_both_keypoints_and_footprint():
    env = _task_env(4)
    v = _view([0.01, 0.01, 0.2, 0.2], [0.55, 0.95, 0.55, 0.95])
    term, _ = termination_utils.compute(env, None, v)
    assert env._termination_reasons["fold"].tolist() == [True, False, False, False]
    assert term.tolist() == [True, False, False, False]
    assert env._bonus_mask.tolist() == [True, False, False, False]


def test_unfold_success_wants_the_full_footprint():
    env = _task_env(2, task="unfold")
    v = _view([0.01, 0.01], [0.95, 0.55])
    flat = NS(points_w=torch.zeros(2, 4, 3), table_top_z=torch.zeros(2))
    termination_utils.compute(env, flat, v)
    assert env._termination_reasons["fold"].tolist() == [True, False]
    assert env._termination_reasons["on_table"].tolist() == [True, False]


def test_a_fallen_sheet_ends_the_episode():
    env = _task_env(1)
    term, _ = termination_utils.compute(env, None, _view([0.2], [1.0], z=0.05))
    assert bool(term[0]) and bool(env._termination_reasons["fall"][0])


def test_release_latches_after_the_fold_and_persists():
    env = _task_env(1, release=True)
    pts = torch.zeros(1, 4, 3)
    near = NS(hand_pos_w=torch.zeros(1, 2, 3), points_w=pts)
    far = NS(hand_pos_w=torch.full((1, 2, 3), 1.0), points_w=pts)
    v = _view([0.01], [0.55])
    termination_utils.compute(env, near, v)            # folded, hand on it: entry bonus only
    assert bool(env._bonus_mask[0]) and not bool(env._released[0])
    for _ in range(3):                                  # three clear steps latch the release
        termination_utils.compute(env, far, v)
    assert bool(env._released[0]) and not bool(env._bonus_mask[0])
    for _ in range(3):
        term, _ = termination_utils.compute(env, far, v)
    assert bool(env._persist_done[0]) and bool(term[0])


def test_release_counts_nothing_without_the_fold():
    env = _task_env(1, release=True)
    far = NS(hand_pos_w=torch.full((1, 2, 3), 1.0), points_w=torch.zeros(1, 4, 3))
    for _ in range(10):
        termination_utils.compute(env, far, _view([0.2], [1.0]))
    assert not bool(env._released[0]) and not bool(env._persist_done[0])


# ------------------------------------------------------------------ observations end to end


def _obs_env(n: int, dr_on: bool):
    torch.manual_seed(4)
    env = _task_env(n)
    s_geom = sheet(((0.2, 0.15, 0.01),) * n)
    canon = list(robot.ROBOT.joint_names_canonical)
    m = action_utils.joint_maps(canon, robot.ROBOT)
    env.cfg.domain_randomization = NS(
        use_object_state_delay_noise=dr_on, object_state_xyz_noise_std=0.01, object_state_rotation_noise_degrees=5.0,
        joint_velocity_obs_noise_std=0.1 if dr_on else 0.0, joint_pos_obs_noise_std=0.0,
        use_obs_delay=dr_on, obs_delay_max=3, action_delay_max=3, object_state_delay_max=10)
    env.cfg.obs = NS(obs_list=POLICY, state_list=CRITIC, clamp_abs_observations=10.0)
    env.cfg.reward = NS(fingertip_distance_from_pad=True)
    state.allocate(env)
    env.__dict__.update(
        _sheet_geom=s_geom, _perm_lab_to_canon=torch.tensor(m["lab_to_canon"]),
        _joint_lower_canon=-torch.ones(27), _joint_upper_canon=torch.ones(27),
        _palm_offset=torch.tensor(robot.ROBOT.palm_center_offset),
        _palm_obs_shift=torch.tensor(robot.ROOT_POS) - torch.tensor(robot.PLAY_ROBOT_ROOT_POS),
        _fingertip_offsets=torch.tensor(robot.ROBOT.fingertip_offsets),
        _object_scales=s_geom.dims / 0.04, reward_buf=torch.zeros(n))
    folded = geometry.folded_start(s_geom, torch.full((n,), math.pi)) + torch.tensor([0.0, 0.0, 0.5])
    s = state.ClothState(
        joint_pos=torch.zeros(n, 27), joint_vel=torch.zeros(n, 27),
        palm_pos_w=torch.tensor([[0.0, 0.3, 0.8]] * n), palm_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n),
        palm_vel_w=torch.zeros(n, 6), fingertip_pos_w=torch.zeros(n, 5, 3),
        fingertip_quat_w=torch.tensor([1.0, 0, 0, 0]).expand(n, 5, 4), hand_pos_w=torch.zeros(n, 6, 3),
        points_w=folded, points_vel_w=torch.zeros_like(folded), table_top_z=torch.full((n,), 0.495))
    return env, s


@pytest.mark.parametrize("dr_on", [False, True])
def test_observations_have_plays_shape_and_a_perfect_fold_reads_zero_goal_offset(dr_on):
    env, s = _obs_env(8, dr_on)
    v = state.derive(env, s)
    obs = obs_utils.build(env, s, v)
    assert obs["policy"].shape == (8, 134) and obs["critic"].shape == (8, 156)
    assert bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(obs["critic"]).all())
    # critic is clean: keypoints_rel_goal (columns 27*3 + 3+4+6+4+6+15+12 = ...) is zero at a perfect fold
    start = 81 + 3 + 4 + 6 + 4 + 6 + 15 + 12
    assert float(obs["critic"][:, start:start + 12].abs().max()) < 1e-5
    # object_rot (critic, after joints and palm pos/rot/vel) is the stationary frame = identity (xyzw)
    rot = obs["critic"][:, 81 + 13:81 + 17]
    assert torch.allclose(rot.abs(), torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(8, 4), atol=1e-5)


def test_nonfinite_sheet_is_zeroed_and_flagged():
    env, s = _obs_env(4, False)
    s.points_w[2, 0, 0] = float("nan")
    v = state.derive(env, s)
    obs = obs_utils.build(env, s, v)
    obs_utils.guard_finite(env, obs, s)
    assert env._nonfinite_envs.tolist() == [False, False, True, False]
    assert bool(torch.isfinite(obs["policy"]).all())
