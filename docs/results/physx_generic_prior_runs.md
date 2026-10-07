# Three PhysX runs from the generic prior: the flat pool transfers best to the cloth

**Setup (2026-10-06).** All three from the generic SimToolReal prior (`pretrained_policy/model.pth`),
PhysX, 24576 envs, 950 epochs = 7.47e8 env-steps (the cloth reference's budget: 3800 epochs of
196,608), 3 seeds, hinge armature 1e-3.

| run | task | start | hinge damping | train job |
|---|---|---|---|---|
| 1 `one_initfold` | 5-shape pool, 10-20 cm | 80%: a random single-stage named fold, 15-120 deg; 20% flat | 0.02 | 158375 |
| 2 `one_flat` | 5-shape pool, 10-20 cm | flat | 0.05 | 158376 |
| 3 `box3` | box3-mid chain | flat | 0.05 | 158324 |

Runs 1-2 use `fold_pool.one_shape_per_env` (below). Evaluation: `bos14_px_eval_matrix.sh`, every
policy on every target, nominal physics (damping 0.05), 900 steps, fold held 10 steps, 2 eval seeds;
pool targets 320 envs (640 episodes per seed), the rest 32 (64 per seed). 360 result files, 0 failed
tasks, 0 non-finite episodes.

## VBD cloth (Newton) -- the number the rigid stages are for

Held folds per training seed, and the share over all three:

| run | 4 cm pinned | 4 cm rand | 1.5 cm pinned | 1.5 cm rand |
|---|---|---|---|---|
| **2 flat pool** | **56, 49, 50 (81%)** | **37, 29, 42 (56%)** | **32, 25, 32 (46%)** | **13, 20, 23 (29%)** |
| 1 pool + folded starts | 36, 13, 35 (44%) | 32, 33, 24 (46%) | 16, 11, 21 (25%) | 16, 13, 17 (24%) |
| 3 box3 chain | 31, 5, 39 (39%) | 21, 8, 41 (36%) | 1, 1, 13 (8%) | 5, 8, 15 (15%) |
| *earlier best: PhysX stage 1 -> stage 2 (157622, physx_port.md)* | *65%* | *47%* | *37%* | *21%* |

Run 2 is the best cloth policy so far on every column, straight from the generic prior with no
stage 1, and every seed is good (49-56 pinned). Run 3 at this budget transfers worse than the shorter
earlier box3 runs (12288 envs / 4e8: 63% / 44%); its seed 2 folds its own chain 100% and the cloth 8%.

## Every target

| run | PhysX pool 4cm pin / rand | PhysX box3 | Newton pool | Newton box3 |
|---|---|---|---|---|
| 1 | 99% / 91% | 100% / 92% | 94% / 64% | 97% / 63% |
| 2 | 100% / 94% | 100% / 92% | 99% / 78% | 98% / 77% |
| 3 | 39% / 38% | 100% / 95% | 20% / 24% | 94% / 82% |

(4 cm cutoff; 1.5 cm is within a few points for the rigid targets -- full tables in
`docs/results/px_matrix/summary.md`.) PhysX box3 randomised tops out at 61/64: the same 3 starts fail
for every policy within 13-43 steps. The pool policies fold box3 as well as the box3 policy; the box3
policy folds only ~40% of the pool.

## Training curves (share of episodes ending folded)

| run | 1e8 | 2e8 | 3e8 | 4e8 | 5e8 | 7.47e8 |
|---|---|---|---|---|---|---|
| 1 (all starts) | 14-23% | 20-34% | 33-50% | 48-71% | 68-78% | 81-85% |
| 2 | 1-3% | 13-24% | 48-56% | 78-81% | 89-93% | 94-95% |
| 3 | 19-35% | 35-58%* | | | | 98% |
| 5-shape-per-env pool (158323, superseded run 2) | 2-2.5% | 10-21% | 40-60% | 74-85% | 89-92% | 94-95% |

\* 1.5e8. Run 1 at the end: flat starts 93-96% (= run 2), folded starts 78-83%. The folded starts
were a head start early (flat-start success 3-7x run 2's at 1e8) but ended harder: a pre-folded corner
or diagonal moves a keypoint that the half fold alone cannot fix. On the cloth run 1 is worse than run
2, so at this budget the folded starts did not pay; damping 0.02 is a second difference between the
two (not separated).

## One shape per env (`fold_pool.one_shape_per_env`, PhysX)

Each env carries one pool shape, fixed (env i -> shape i % 5), instead of all five with four parked.
24576 envs, fps total: 5 shapes per env 62-66k, one shape per env 94-97k (**1.5x**; single-shape
benches 104-117k). Equivalence: the same trained policy (157622 s1) scores 94.2% / 93.1% pinned and
75.8% / 76.4% randomised on the two envs, and runs 2 and 158323 learn on identical curves.

## Partly folded starts (`fold_pool.init_fold_prob`, `init_fold_deg`)

The 19 single-stage named folds of the pool (`FoldShape.start_folds`), turned up by a uniform angle.
Startup proofs (fatal): each fold's up direction measured (all +1), and every fold lands at the
largest angle within 2 deg (measured 0.000). Two traps the proofs caught: a check held at spawn height
near the env origin meets the hand (the sheet spawns 15 cm above the table by design,
`cloth.start_height`), and assigning test folds as `env % F` aliases with one-shape `env % 5` (four of
box2x8's folds went untested and defaulted downward). Probe: `bos14_fold_start_probe.sh`.

## Reproduce

    TASK=pool ENVS=24576 EPOCHS=950 RUN_TAG=one_flat EXTRA_ARGS="env.fold_pool.one_shape_per_env=true" \
        sbatch --mem=120G --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
    TASK=pool ENVS=24576 EPOCHS=950 RUN_TAG=one_initfold EXTRA_ARGS="env.fold_pool.one_shape_per_env=true \
        env.fold_pool.init_fold_prob=0.8 env.rigid_cloth.joint_damping=0.02" \
        sbatch --mem=120G --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
    TASK=chain ENVS=24576 EPOCHS=950 RUN_TAG=box3 sbatch --mem=120G --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
    KIND=pool TRAIN_JOB=<id> RUN_TAG=<tag> sbatch --array=0-29 scripts/cluster/bos14_px_eval_matrix.sh
    python3 scripts/analysis/px_eval_matrix_summary.py one_flat one_initfold box3
    sbatch --array=0-3 scripts/cluster/bos14_one_shape_parity.sh        # one-shape vs 5-shape parity
