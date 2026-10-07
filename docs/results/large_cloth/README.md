# Fold policies on larger VBD cloths: performance falls steadily with size

Newton VBD cloth (`Isaacsimenvs-Cloth-Direct-v0`), protocol of `bos14_px_eval_matrix.sh` (32 envs x 2
eval seeds, 900 steps, fold held 10 steps, SAPG coef 0). 0 failed tasks, 0 non-finite episodes.

## Cloth model per size

Calibrated material unchanged (16 mm particles, 2 kg/m^2, SIM1 stiffnesses). Grid = the spacing
closest to the 10 cm sheet's 16.67 mm that is still >= the 16 mm particle thickness (ClothCfg guard).
New `cloth.reset_offset_xy` shifts the reset centre toward the robot by (size - 0.10) * sqrt(2) / 2,
as the fold pool's `reach_shift`. `object_scales` follows `cloth.size` (5.0 at 20 cm, as the pool).

| size | hydra overrides | grid | spacing | mass |
|---|---|---|---|---|
| 10 cm | (defaults) | 7x7 | 16.67 mm | 20 g |
| 12.5 cm | `env.cloth.size=0.125 env.cloth.resolution=8 env.cloth.reset_offset_xy=[0.0,0.0177]` | 8x8 | 17.86 mm | 31 g |
| 15 cm | `env.cloth.size=0.15 env.cloth.resolution=10 env.cloth.reset_offset_xy=[0.0,0.0354]` | 10x10 | 16.67 mm | 45 g |
| 17.5 cm | `env.cloth.size=0.175 env.cloth.resolution=11 env.cloth.reset_offset_xy=[0.0,0.0530]` | 11x11 | 17.50 mm | 61 g |
| 20 cm | `env.cloth.size=0.2 env.cloth.resolution=13 env.cloth.reset_offset_xy=[0.0,0.0707]` | 13x13 | 16.67 mm | 80 g |

Stability (`bos14_large_cloth_probe.sh`, zero actions, 8 envs, 150 steps, nan_policy=raise): every
size finite, drops 15 cm and lands flat (height spread <= 0.02 mm) at the 10 cm sheet's rest height
0.538 m by step 20, then still; centre at the intended shift.

## Run 2 (flat pool), all three seeds, by size -- `size_sweep.md`

Held folds per seed (of 64), overall share; falls = whole sheet pulled off the table.

| size | 4 cm pinned | 4 cm rand | 1.5 cm pinned | 1.5 cm rand | falls |
|---|---|---|---|---|---|
| 10 cm | 56, 49, 50 (81%) | 37, 29, 42 (56%) | 32, 25, 32 (46%) | 13, 20, 23 (29%) | 20% |
| 12.5 cm | 47, 32, 29 (56%) | 22, 21, 33 (40%) | 2, 0, 2 (2%) | 3, 1, 0 (2%) | 35% |
| 15 cm | 28, 22, 36 (45%) | 8, 13, 15 (19%) | 4, 2, 0 (3%) | 1, 0, 1 (1%) | 48% |
| 17.5 cm | 23, 22, 13 (30%) | 11, 8, 16 (18%) | 8, 5, 3 (8%) | 1, 0, 2 (2%) | 48% |
| 20 cm | 4, 11, 5 (10%) | 2, 1, 1 (2%) | 1, 1, 0 (1%) | 0, 0, 0 (0%) | 57% |

Caveat: 12.5 and 15 cm need EVEN grids (8, 10), where the crease falls between particle rows
instead of on a hinge row; the 1.5 cm column there may partly measure grid geometry. 4 cm is the
robust trend.

## The viewer's four policies at 20 cm -- `menu_policies_s20.md`

| policy | 10 cm: 4cm pin / rand, 1.5cm pin / rand | 20 cm: same |
|---|---|---|
| run 2 flat pool s3 | 78 / 66, 50 / 36% | 8 / 8, 0 / 0% |
| run 1 pool + folded starts s1 | 56 / 50, 25 / 25% | 0 / 0, 0 / 0% |
| run 3 box3 s3 | 61 / 64, 20 / 23% | 0 / 2, 0 / 0% |
| previous best 157622 s1 | 88 / 62, 77 / 22% | 0 / 6, 0 / 0% |

## Why 20 cm fails: mostly the sheet is dragged off the table

60% of 20 cm episodes end with the sheet on the floor (centre < 0.1 m), mid-episode (steps
135-563) from a pinned start -- the policy drags it off; it does not slide off passively. Placement
diagnostic, run 2 s3 (held / falls of 64, 4 cm):

| variant | pinned | rand |
|---|---|---|
| shift 7.07 cm, cloth table 0.475 x 0.4 (as evaluated) | 5 / 28 | 5 / 38 |
| no shift | 1 / 30 | 4 / 23 |
| shift, pool table 0.6 x 0.4 (`env.assets.table_urdf=assets/urdf/table_fold_pool.urdf`) | 12 / 15 | 2 / 25 |
| no shift, pool table | 5 / 23 | 6 / 19 |

The wider table halves pinned falls but folds stay <= 19%: the policies, trained on rigid plates
up to 20 cm, do not handle a large draping VBD sheet.

Reproduce: `sbatch --array=0-23 scripts/cluster/bos14_cloth_size_sweep.sh`,
`sbatch --array=0-7 scripts/cluster/bos14_large_cloth_eval.sh`,
`python3 scripts/analysis/cloth_size_sweep_summary.py`, `python3 scripts/analysis/large_cloth_summary.py`.
