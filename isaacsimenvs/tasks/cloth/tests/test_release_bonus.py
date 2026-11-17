"""Fold entry and complete robot release pay once, with no hold timer."""
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.mdp.geometry import grid
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask


def task(n=1):
    release = NS(enabled=True, touch='robot_clearance', clearance=.1, latch_steps=999,
                 fold_tolerance=0., fold_entry_bonus=1000., release_bonus=1000.,
                 release_dist_rew_scale=3000., persist_rew_per_step=2., persist_bonus=1000.,
                 persist_window_steps=999, persist_fail_penalty=-300., persist_consecutive=True,
                 persist_fail_terminates=True)
    cfg = NS(task=NS(mode='fold', release=release, keypoint_tolerance=.015,
                     max_folded_footprint=.65, min_unfolded_footprint=.9, unfold_on_table=False),
             termination=NS(success_steps=1, held_steps=10, fall_height=.1, hand_far_distance=1.5),
             reward=NS(object_base_size=.04, keypoint_rew_scale=0., distance_delta_rew_scale=0.,
                       kuka_actions_penalty_scale=0., hand_actions_penalty_scale=0.))
    dims = torch.tensor([[.2, .2, .004]]*n)
    obj = ClothTask(cfg, grid.make_grid(dims, 7, dims[:, 2]), torch.zeros(n, 3), list(range(7)), list(range(7, 27)))
    obj.episode_length_buf, obj.max_episode_length = torch.zeros(n, dtype=torch.long), 600
    return obj


def step(obj, folded=True, clearance=.01):
    n = obj.num_envs
    state = NS(joint_vel=torch.zeros(n, 27))
    folded = torch.as_tensor(folded).expand(n)
    view = NS(fold_error=torch.where(folded, .001, .2), footprint=torch.where(folded, .5, 1.),
              pos_w=torch.tensor([[0., 0., .5]]*n), fingertip_dist=torch.full((n, 5), .2),
              lin_vel_w=torch.zeros(n, 3), ang_vel_w=torch.zeros(n, 3))
    obj.robot_clearance = torch.as_tensor(clearance).expand(n)
    done, _ = obj.compute_terminations(state, view)
    reward = obj.compute_reward(state, view)
    return reward, done


def test_fold_then_release_pays_1000_each_once_without_latch_or_persistence_timer():
    obj = task()
    reward, done = step(obj, clearance=.01)
    assert reward.item() == 1000 and not done.item()
    reward, done = step(obj, clearance=.099)
    assert reward.item() == 0 and not done.item()
    reward, done = step(obj, clearance=.1)
    assert reward.item() == 1000 and done.item()
    assert obj._successes.item() == 1
    reward, done = step(obj, clearance=.3)
    assert reward.item() == 0 and done.item()
    assert obj._successes.item() == 1
    assert obj._persist_steps.item() == 0 and not obj._persist_done.item()


def test_fold_and_release_can_complete_same_frame_without_double_paying_later():
    obj = task()
    reward, done = step(obj, clearance=.2)
    assert reward.item() == 2000 and done.item()
    assert step(obj, clearance=.2)[0].item() == 0


def test_clearance_is_not_release_unless_fold_is_still_valid_and_nonfinite_fails_closed():
    obj = task()
    step(obj, clearance=.01)  # Paid fold entry.
    reward, done = step(obj, folded=False, clearance=.5)
    assert reward.item() == 0 and not done.item()
    reward, done = step(obj, clearance=float('nan'))
    assert reward.item() == 0 and not done.item()
    reward, done = step(obj, clearance=.2)
    assert reward.item() == 1000 and done.item()


def test_selected_resets_rearm_both_bonuses_without_affecting_other_envs():
    obj = task(2)
    step(obj, clearance=.2)
    obj.reset(torch.tensor([1]))
    reward, _ = step(obj, clearance=.2)
    torch.testing.assert_close(reward, torch.tensor([0., 2000.]))


def test_origin_proxy_cannot_silently_replace_missing_robot_geometry():
    obj = task()
    obj.robot_clearance = None
    with pytest.raises(ValueError, match='whole-robot'):
        obj.update_release(NS(), NS(fold_error=torch.zeros(1)), torch.ones(1, dtype=torch.bool),
                           torch.ones(1, dtype=torch.bool))
