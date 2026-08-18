# Cloth-fold baseline for the SAPG finetune (job 197715)

Produced by `scripts/analysis/cloth_fold_summary.py`. Every set below is 5 seeds x 32 envs,
600 steps, `success_tolerance=0.01` (the inherited rigid-tool criterion pinned so it cannot fire
and become a second writer to `_successes`), fold tolerance 0.04, footprint 0.65.

| set | commit | entered | held | best_fold_err (across seeds) | falls |
|---|---|---|---|---|---|
| `cloth_foldsep_s*` | c8ac0dd | 1/160 | 1/160 | 0.0793 +- 0.0027 | 16 |
| **`cloth_nohold_s*`** | **d774ba2** | **3/160 (1.88%)** | **0/160** | **0.0824 +- 0.0021** | **4** |
| `cloth_iter5_s*` | d9a0238 | 0/160 | 0/160 | 0.0871 +- 0.0023 | 4 |

**`cloth_nohold_s*` is THE baseline** for job 197715:

* it is the most recent, and post-dates the fix that made the fold the sole `_successes` writer;
* its termination rule matches training -- `success_steps=1`, no hold, episode ends on first entry;
* the observation is unchanged since it was taken. f74d526 ("re-point the inherited reward at the
  fold") is 93 insertions and 0 deletions, touching only `_get_rewards`; `_keypoints_max_dist` and
  `keypoints_rel_goal` were left alone. So the pretrained policy sees the same 140 dims it saw here.

`iter5` is a probe of `vbd_iterations=5`, not a baseline -- that setting was rejected. The
foldsep/nohold falls gap (16 vs 4) is not a physics difference: requiring a 10-step hold keeps
episodes running past the first fold, so there is more opportunity to fall.

## Correction

Earlier reports of "3 folds / 160, best_fold_err 0.0793 +- 0.0012" conflated two sets: the 3 is
`nohold`, the 0.0793 is `foldsep`, and the quoted spread was tighter than either set's across-seed
sd (0.0021 / 0.0027). Use the table.

## Known deviation from the plan

The plan called for feeding `keypoints_rel_goal` from `cloth_keypoints_w() - fold_targets_w()` so
that observation, dense shaping and sparse bonus all key on one definition of the fold. That was
NOT implemented. The reward keys on `fold_error()`; the observation still describes the rigid
`goal_viz` approximation. `_drive_goal_marker` does drive `goal_viz` onto the fold target every
step, so the observation is an approximation of the fold error rather than something unrelated --
workable, but the mismatch is real and is a candidate explanation if the run fails to improve.

Changing it now would invalidate this baseline and require restarting, so job 197715 runs as-is.

## Judging the run

Improvement means moving `best_fold_err` off ~0.082 and lifting `entered` above 1.88%. The plateau
is tight across seeds (+-0.0021), so a real shift should be obvious. If `best_fold_err` stays at
~0.082 the dense shaping has saturated and the +1000 bonus will essentially never fire -- the
"prior does not transfer" outcome in the plan's risk list.
