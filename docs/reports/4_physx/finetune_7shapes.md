# Fine-tuning the PhysX pool policies on 7 shapes with non-target starts: result not yet measured

> **The evaluation below did NOT measure the fine-tuned policies.** `bos14_px_eval_matrix.sh`
> evaluates `nn/<name>.pth`, the *best-reward* checkpoint. In these runs that checkpoint was written
> once, ~10 epochs in, and never again. Each run saved "next best" a single time, about 25 min
> after start, while the final checkpoint was written ~5 h later. In the first epochs the
> episodes that end soonest are mostly quick successes, so the mean reward read high (1004-1058).
> It then dropped and climbed back to 900-975 without passing that early value. So the numbers
> below are for policies fine-tuned for ~10 epochs, and they cannot show whether the fine-tune
> helped. The trained policies are `nn/last_<name>_ep_1526_rew_*.pth` and have not been
> evaluated yet. The re-evaluation is step 1 of `docs/training_plan.md` (2026-10-11 draft).
>
> This affects only this round. In runs 1-3, 157622 and 157449 the best checkpoint was written
> within minutes of the end of training.

**Setup (2026-10-07).** Continues runs 1 and 2 of `physx_generic_prior_runs.md`. Each seed is
fine-tuned from the same seed of its source run.

| run | RUN_TAG | train job | prior |
|---|---|---|---|
| fine-tune of run 2 (flat pool) | `ft7lr_flat` | 158820 | `0_pxpool_one_flat_s<k>_158376` |
| fine-tune of run 1 (pool + folded starts) | `ft7lr_initfold` | 158821 | `0_pxpool_one_initfold_s<k>_158375` |

Both use the same settings: PhysX, 24576 envs, one shape per env, 1526 epochs = 1.2e9 env-steps
(~5.2 h per seed). The pool has **7 shapes**: the original 5 plus `box4-mid` and `star8-mid`.
`init_fold_starts=other` with `init_fold_prob=0.8` and `init_fold_deg=[15,180]` means 80% of
episodes start with non-target folds already made. The learning rate is **8.8e-6**, the source
runs' final lr. At the default 1e-4, a converged policy is wiped in a few updates (KL ~100, fold
rate 94% -> 0). The first attempt, 158800/158801, was cancelled for this reason.

## What was evaluated: the ~epoch-10 checkpoints

`bos14_px_eval_matrix.sh` (158822 / 158823), the same protocol as the source runs. The "pool"
targets are the default 5-shape pool. VBD cloth, held folds per training seed (of 64) and share:

| checkpoint | 4 cm pinned | 4 cm rand | 1.5 cm pinned | 1.5 cm rand |
|---|---|---|---|---|
| run 2 flat pool (source) | 56, 49, 50 (81%) | 37, 29, 42 (56%) | 32, 25, 32 (46%) | 13, 20, 23 (29%) |
| ft7lr_flat, ~epoch 10 | 57, 49, 50 (81%) | 34, 28, 41 (54%) | 49, 32, 12 (48%) | 8, 12, 21 (21%) |
| run 1 pool + folded starts (source) | 36, 13, 35 (44%) | 32, 33, 24 (46%) | 16, 11, 21 (25%) | 16, 13, 17 (24%) |
| ft7lr_initfold, ~epoch 10 | 34, 22, 36 (48%) | 34, 32, 31 (51%) | 15, 4, 11 (16%) | 21, 11, 17 (26%) |

The only thing this shows is that the low-lr start did not damage the source policies. The full
matrix for these checkpoints is kept in the raw files `docs/results/px_matrix/ft7lr_*`.
`python3 scripts/analysis/px_eval_matrix_summary.py ft7lr_flat ft7lr_initfold` prints it.
