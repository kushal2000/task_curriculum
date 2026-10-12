# The finetune optimises the action penalties, not the fold

Observed in BOTH runs, monotonically over ten deciles, at ~2 h of 24 h.

## The trend

| | 197715 (yaw-fixed, 1:51) | 199350 (yaw-random, 1:32) |
|---|---|---|
| `episode_final/done_fold` | 0.0209 -> 0.0058 | 0.0915 -> 0.0096 |
| `keypoint_rew` (dense fold progress) | 2.66 -> 0.50 (down 5.3x) | 2.67 -> 0.64 (down 4.2x) |
| `hand_actions_penalty` | -47.1 -> -17.2 (63% reclaimed) | -31.6 -> -21.8 (31% reclaimed) |
| `kuka_actions_penalty` | -17.3 -> -13.3 | -13.3 -> -13.6 |
| `fold_err_mean` | flat 0.1002 | flat, slightly worse 0.1006 |
| `lifting_rew`, `lift_bonus_rew` | 0.0 throughout | 0.0 throughout |

The lift masking works exactly as designed. Everything else says the policy is drifting AWAY from
the folds the pretrained prior produced by accident, while total reward holds up or rises because
the penalty savings more than cover the lost fold reward.

## Why: the dense fold signal is structurally too small

`keypoint_rew` is `delta(best-so-far fold error) * keypoint_rew_scale`, ratcheted per episode. The
fold error of an unfolded sheet is ~0.1007 m, so the term is BOUNDED by how much error exists to
remove:

    perfect fold      (0.1007 -> 0)     20.1 reward units
    fold to tolerance (0.1007 -> 0.04)  12.1 reward units

Against that, the action-penalty budget the policy can reclaim by simply moving less:

    197715: hand 47.1 -> 17.2 and kuka 17.3 -> 13.3, i.e. ~34 units ALREADY BANKED
    theoretical maximum (zero actions): ~64 units

So going limp is worth ~2.8x the entire dense fold signal at what the policy has already achieved,
and ~5x at the limit. The sparse +1000 bonus does not rescue it: at the observed 0.6-2% success
rate its expected value is 6-20 units, with enormous variance, against a certain ~34.

Gradient descent is behaving correctly. The reward is misspecified.

## Options

1. **Raise `keypoint_rew_scale`** (currently 200). Parity with the observed penalty budget needs
   ~560; clear dominance wants 1000-2000. Least invasive -- it does not touch the regularisation
   the prior was trained under. Risk: the term is a ratcheted difference, so scaling it also scales
   its noise.
2. **Cut the action penalties** (`kuka_actions_penalty_scale` 0.03, `hand_actions_penalty_scale`
   0.003) by ~3x. Directly removes the competing incentive, but these are play2perfect's `r_smooth`
   and what the pretrained policy was regularised with, so this risks degrading the prior's
   smoothness -- the thing that makes it transferable.
3. **Re-point `keypoints_rel_goal` at `fold_targets_w()`.** Independent of the imbalance and
   compatible with either fix. The plan called for it and it was never implemented, so the policy
   currently OBSERVES the rigid `goal_viz` approximation while being REWARDED on true fold error.

(1) plus (3) is the combination I would try first.

## Caveats

Two hours of twenty-four, and SAPG explores broadly early. The decline is monotonic across ten
deciles in two independent runs, which is why this is recorded now rather than waited out -- but it
is a trend, not an outcome, and the arithmetic above uses observed penalty deltas as a proxy for
the available budget.
