# The robot NaN clusters immediately after reset

## The observation

Divergences are ~15x over-represented in the first ten steps of an episode, and it replicates
across two independent runs with different seeds, different nodes, and different env code:

| run | env | <= step 10 | uniform expectation | P(>= observed \| uniform) |
|---|---|---|---|---|
| 197715 | yaw-fixed | 6 / 25 (24.0%) | 1.67% | 2.9e-6 |
| 199350 | yaw-random | 3 / 7 (42.9%) | 1.67% | 1.5e-4 |

Both runs also show a long tail across the rest of the episode (medians 191 and 12, maxima 527 and
452), so this is not the only mode -- but roughly a quarter of all NaN events are reset-adjacent.

## Why it was invisible before

`_guard_finite` reported `episode_length_buf.max()` -- the max over ALL envs, not the offender's own
progress. That equals the offender's step only while the envs are still synchronised, i.e. during
the first episode; afterwards it saturates near `episode_length` and every report read
"at step 599". The whole distribution was hidden behind a constant. Fixed in d11b8c0, which is what
made this measurable.

## What it is NOT

Not one broken env being re-flagged: the offending env ids are almost all distinct (in the smoke,
15 events across 14 distinct envs). The reset does clear the condition; the problem is that the
reset sometimes CREATES it.

Not the cloth: `particles`, `velocities`, `quat` and `fold_targets` checks have never fired. Every
event names the MJWarp articulation -- `robot_joint_pos`, `robot_joint_vel`, `robot_body_pos_w`,
`robot_body_quat_w` -- and the obs and reward NaN are downstream of it, same env, same step.

## Leading hypothesis, untested

`reset_env_state` teleports the sheet to a new XY within +-0.1 m and simultaneously redraws the
arm and finger joint positions (interval 0.1) and joint velocities (interval 0.5). A sheet
teleported into the hand, or a hand redrawn into the sheet or table, would resolve as a very large
first-step contact impulse, which is exactly what makes an articulation diverge within a handful of
steps.

Ways to discriminate, none run yet:
  * log the minimum distance between the fingertips and the nearest cloth particle at reset, and
    compare the distribution for envs that diverge against those that do not;
  * zero `reset_dof_vel_random_interval` alone and see whether the <= step 10 cluster survives;
  * step the sim a few times after reset before handing the observation to the policy, and see
    whether the divergence is merely delayed or actually removed.

## Why it is not urgent

The overall rate is ~1e-6 per env-step, the guard sanitises the observation and resets the env
before anything non-finite reaches the optimiser, and `nonfinite_reward` has stayed at 0.0 in
training. Over 24 h this is ~0.065% of episodes. It costs a little sample efficiency and nothing
else -- but it is a real defect with a specific, testable cause, and it is the single largest
identifiable slice of the NaN budget.
