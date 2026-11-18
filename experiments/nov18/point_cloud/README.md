# Top-face point cloud: simulator check (job 789400)

`check.py` on the heavy-prior run config (5 designs, 20 mm, 64 envs) with
`env.obs.num_points=64 env.task.mode=mixed env.task.release.enabled=false
env.rigid_cloth.episode_checks.press_driven_fold=false` (that check is fold-only).

Fold envs start flat and unfold envs start folded at 180 deg, so each env's start pose is the
other task's goal. Matched distance, samples -> goal of the start shape (positive control) and
-> the env's own task goal (negative control):

| step | fold envs: start-shape goal (median / max) | unfold envs: start-shape goal (median / max) | task goal (min / median) |
|---|---|---|---|
| 0 | 2.0 / 2.0 mm | 2.0 / 2.0 mm | 20-26 / 52-60 mm |
| 1 | 0.00 / 0.01 mm | 0.01 / 0.96 mm | 19-25 / 52-59 mm |
| 20 | 0.00 / 0.00 mm | 0.00 / 5.1 mm | 19-25 / 52-59 mm |

* Step 0 is exactly the 2 mm spawn clearance; the cloth lands on the table in one step.
* The unfold outliers (max 5 mm, one point 13 mm) are folded flaps springing back a little, the
  known stiff-hinge effect, not a sampling error.
* Plots: `heavy_prior/env_0_fold.png`, `heavy_prior/env_1_unfold.png`.
