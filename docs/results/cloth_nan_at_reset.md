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

## CORRECTION: it is not only the robot

An earlier version of this note claimed the cloth-side checks had "never fired" and that every
event named the MJWarp articulation. That was generalised from an early sample and is WRONG. Full
counts over both 24 h runs:

| check | 197715 | 199350 |
|---|---|---|
| `particles` | 10 | 13 |
| `velocities` | 10 | 13 |
| `fold_targets` | 6 | 5 |
| `robot_joint_pos` / `_vel` / `body_pos_w` / `body_quat_w` | 89 | 149 |

The robot still dominates by ~10x, but the cloth diverges too.

## Two distinct modes, not one

**Robot-led.** All four robot quantities fire together, `fold_targets` with them, and obs and
reward follow -- same env, same step. Example: env 244 at env-step 273.

**Cloth-only, and this one is a bug in our own code.** `fold_targets` fires with NO robot quantity
at all, and the poisoned observation columns are 125-136, exactly `keypoints_rel_goal`. Example:
env 692 at env-step 2. The guard tests `particles` FIRST and it did not fire, so the particle
positions were finite and `_stationary_frame`'s Kabsch fit produced NaN from finite input.

`_stationary_frame` already carries a ridge term and an identity fallback for a non-finite `R`, but
`_folded_w` composes `R` with a centroid, so a finite-but-degenerate fit still yields non-finite
targets downstream of that fallback. Three of the observed cases are at env-steps 1, 2 and 4 --
immediately after reset, when the sheet is flat and the stationary half is very nearly planar,
which is exactly when the SVD is most degenerate.

Fix candidates: check the fit's residual and singular values rather than only `isfinite(R)`; or
guard `fold_targets_w()` itself and fall back to the rest-frame targets for envs whose fit is
degenerate.

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
