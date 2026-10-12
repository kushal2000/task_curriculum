# Cloth-fold baseline for the SAPG finetune (job 197715)

> **SUPERSEDED as a like-for-like baseline.** The cloth physics in `Cloth.yaml` was replaced
> wholesale with SIM1's calibrated values (arXiv 2604.08544 Table 2): the sheet went from 9x9 to
> 7x7 at 16 mm thickness, in-plane stiffness dropped 100x, area density dropped 50x, self-contact
> was turned ON for the first time, and the cloth timestep went from 1/480 s to 1/1440 s. The
> numbers below still describe the OLD physics and remain the reference for what that physics
> achieved -- they are no longer a control for runs on the current config. A new baseline has to be
> re-measured before anything is compared against 3/160 or best_fold_err 0.082.


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

---

# Known risk: the frame-relative footprint can fail permissive

`footprint_ratio` was changed (7a58c82) from world-axis extent to extent along the sheet's own fold
axis, `r[:, :, ax]`, where `r` is the stationary half's Kabsch fit. That was required for init yaw
randomisation and it fixed a real fail-STRICT bug: the world-axis version inflated to as much as
sqrt(2) for a rotated but undeformed sheet, so a genuine fold of a turned sheet scored as unfolded.

The replacement has the opposite failure mode. Projecting onto a direction that tilts out of the
sheet's plane collapses the measured extent:

| tilt of the fitted axis out of the sheet plane | footprint of an UNFOLDED sheet |
|---|---|
| 0 deg (in plane, along an edge) | 1.0000 |
| 0 deg (in plane, diagonal) | 1.4142 |
| 50 deg | 0.6428  <- crosses max_folded_footprint 0.65 |
| 80 deg | 0.1736 |
| 90 deg (the sheet normal) | 0.0000 |

**This cannot happen for a rigid motion**: the sheet and the fitted axis rotate together, so the
projection stays 1.0 -- that is the point of measuring in the sheet's frame. It needs a MISMATCH,
i.e. the stationary half tilted or crumpled while the rest of the sheet lies flat, so the fit tilts
the axis away from the plane the sheet actually occupies.

Not yet observed firing. It is, however, the alternative explanation for the settled fold rate
being 3.12% in the yaw run (199350, new footprint) against 0.65% in the yaw-fixed run (197715, old
footprint), which is otherwise consistent with simply no longer discarding rotated folds. The two
cannot be separated from those numbers alone.

Deliberately NOT fixed mid-flight: this is the scoring criterion for two live experiments, and the
obvious repairs carry their own failure modes -- projecting onto the horizontal component instead
would read ~0 for a legitimately lifted sheet, which is fail-permissive in a different state.

Options if it needs addressing:
  * fail SAFE on a suspect fit -- compute the Kabsch residual and report a large footprint (not
    folded) when the stationary half is too deformed for its frame to mean anything;
  * measure extent in the plane fitted to ALL particles rather than along one basis vector;
  * keep both measures and require the fold criterion to satisfy the stricter of the two.
