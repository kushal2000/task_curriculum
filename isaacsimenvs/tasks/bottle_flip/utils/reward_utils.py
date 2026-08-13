"""Staged reward for bottle flipping.

The first two stages — approach the bottle, then lift it — are play's own reward
functions, reused unchanged (`distance_delta_reward`, `lifting_reward`,
`action_penalty`). That reuse is deliberate: it is the same behaviour the play2perfect
policy was pretrained on, so a finetuning run starts with those stages already solved and
only has to learn what comes after.

The new stages are release → rotate → land upright → hold still, and each is one term
here. All of them route through `curriculum.reward_scale`, so
`cfg.curriculum.reward_schedule` can anneal the dense shaping away as difficulty rises
without editing this file.
"""

from __future__ import annotations

import torch

from isaacsimenvs.curriculum import reward_scale
from isaacsimenvs.tasks.play.utils.reward_utils import (
    action_penalty,
    distance_delta_reward,
    lifting_reward,
)

__all__ = ["compute_rewards"]


def compute_rewards(env) -> torch.Tensor:
    rew_cfg = env.cfg.reward
    origins = env.scene.env_origins
    obj_pos = env.object.data.root_pos_w - origins

    # --- Stage 1-2: reach and lift (inherited from play). ---
    lift_rew, lift_bonus_rew, new_lifted = lifting_reward(
        object_z=obj_pos[:, 2],
        object_init_z=env._object_init_z,
        prev_lifted=env._lifted_object,
        lifting_bonus_threshold=rew_cfg.lifting_bonus_threshold,
        lifting_bonus=rew_cfg.lifting_bonus,
        lifting_rew_scale=rew_cfg.lifting_rew_scale,
    )
    env._lifted_object = new_lifted

    ft_rew, new_closest_ft = distance_delta_reward(
        curr_fingertip_dist=env._curr_fingertip_distances,
        closest_fingertip_dist=env._closest_fingertip_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.distance_delta_rew_scale,
    )
    env._closest_fingertip_dist = new_closest_ft

    kuka_pen, hand_pen = action_penalty(
        joint_vel=env.robot.data.joint_vel,
        arm_ids=env._arm_joint_ids,
        hand_ids=env._hand_joint_ids,
        kuka_scale=rew_cfg.kuka_actions_penalty_scale,
        hand_scale=rew_cfg.hand_actions_penalty_scale,
    )

    # --- Stage 3: rotate. Paid on saturating progress toward `required_turns`. ---
    spin_rew = rew_cfg.spin_rew_scale * reward_scale(env, "spin") * env._bf_turn_delta

    # --- Stage 4: land upright. Dense uprightness only once the bottle is in flight,
    #     so it cannot be collected by simply holding the bottle vertical in the hand. ---
    upright_rew = (
        rew_cfg.upright_rew_scale
        * reward_scale(env, "upright")
        * env._bf_upright_cos
        * env._bf_airborne.float()
    )

    # --- One-shot bonuses. Latched so a long settle does not pay repeatedly. ---
    upright_ok = env._bf_upright_cos >= env._bf_upright_cos_tol
    turns_ok = env._bf_flip_turns >= env._bf_required_turns
    good = env._bf_landed & upright_ok & turns_ok

    just_landed = good & ~env._bf_land_bonus_paid
    env._bf_land_bonus_paid = env._bf_land_bonus_paid | good
    land_rew = rew_cfg.landing_bonus * reward_scale(env, "landing_bonus") * just_landed.float()

    just_succeeded = env._bf_is_success & ~env._bf_success_paid
    env._bf_success_paid = env._bf_success_paid | env._bf_is_success
    success_rew = (
        rew_cfg.success_bonus * reward_scale(env, "success_bonus") * just_succeeded.float()
    )

    terms = {
        "fingertip_delta_rew": ft_rew,
        "lifting_rew": lift_rew,
        "lift_bonus_rew": lift_bonus_rew,
        "spin_rew": spin_rew,
        "upright_rew": upright_rew,
        "landing_bonus_rew": land_rew,
        "success_bonus_rew": success_rew,
        "kuka_actions_penalty": kuka_pen,
        "hand_actions_penalty": hand_pen,
    }
    reward = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    terms["total_reward"] = reward
    env._reward_terms = terms
    return reward
