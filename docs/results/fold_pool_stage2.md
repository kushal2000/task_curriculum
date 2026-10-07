# Stage 2: one policy, five fold shapes, 10-20 cm sheets

**Task.** `Isaacsimenvs-FoldPool-Direct-v0`: the half fold (right over left), with the sheet drawn
per episode from the fold-shape pool -- `box2-mid`, `box2diag4-mid`, `box2x8-mid`, `cross4-mid`,
`corners8-mid` -- and a sheet side drawn once per env from 10-20 cm. Fixed 1.5 cm keypoint
tolerance at every size; contact-shell randomisation on. 1536 envs, 2035 epochs (1e8 env-steps),
3 seeds per arm, single RTX PRO 6000.

**Result.** Starting from the stage-1 `box3-mid` rigid policy, the pool policy folds **98% pinned
and 82% randomised**, every shape and every size within a few points. Starting from the generic
tool-use prior, the same budget yields 0.5% / 5%.

## Evaluation

`bos14_fold_pool_eval.sh`: 320 envs, one episode each, two evaluation seeds per run; success = the
fold HELD 10 steps at 1.5 cm. *Pinned* = reset noise and contact-shell DR off (shape and size still
vary -- they are the task); *rand* = the training regime. Tabulated by
`scripts/analysis/fold_pool_eval_summary.py --train_job <id>`; the last column is the median best
fold error.

### From the `box3-mid` policy (train 156746, eval 156747)

Prior: `0_rc2_box3-mid_fixdr_s2_155349` (round 2 of the rigid-cloth comparison: own chain ~100%,
best zero-shot cloth transfer of the six `box3-mid` runs).

| | pinned | rand |
|---|---|---|
| **all** | **1882/1920 (98.0%)**, 3.2 mm | **1572/1920 (81.9%)**, 4.5 mm |
| `box2-mid` | 379/387 (97.9%), 0.4 mm | 291/330 (88.2%), 0.6 mm |
| `box2diag4-mid` | 346/366 (94.5%), 3.1 mm | 274/360 (76.1%), 5.4 mm |
| `box2x8-mid` | 353/354 (99.7%), 4.5 mm | 323/417 (77.5%), 7.0 mm |
| `corners8-mid` | 437/444 (98.4%), 7.2 mm | 336/417 (80.6%), 7.4 mm |
| `cross4-mid` | 367/369 (99.5%), 2.1 mm | 348/396 (87.9%), 2.6 mm |
| 10.0-12.5 cm | 403/420 (96.0%) | 355/420 (84.5%) |
| 12.5-15.0 cm | 420/426 (98.6%) | 376/426 (88.3%) |
| 15.0-17.5 cm | 471/474 (99.4%) | 386/474 (81.4%) |
| 17.5-20.0 cm | 588/600 (98.0%) | 455/600 (75.8%) |
| seeds s1 / s2 / s3 | 98.8 / 97.2 / 98.1% | 80.5 / 79.7 / 85.5% |

Randomised failures over all seeds: 199 falls (10%), 148 timeouts (8%), 1 hand-far. Every run
exited cleanly with no contact-buffer overflow and nothing censored.

### From the generic prior (train 156744, eval 156745)

`pretrained_policy/model.pth`, the tool-use prior stage 1 also started from.

| | pinned | rand |
|---|---|---|
| all | 9/1920 (0.5%), 156 mm | 98/1920 (5.1%), 146 mm |
| seeds s1 / s2 / s3 | 1.4 / 0.0 / 0.0% | 14.8 / 0.2 / 0.3% |

Only s1 began to fold within the budget (7% at its end, in training). A median best error of ~150 mm
is the flat sheet: these policies mostly never start a fold.

## Reading it

- **The prior is the whole difference.** Same env, reward, budget and seeds. The `box3-mid` policy
  already folds the pool zero-shot (73/320 pinned, 95/320 rand, almost all on sheets under 15 cm --
  measured before training, job 156735), so stage 2 starts from a policy that knows the motion and
  only adapts it to new shapes and sizes. The generic prior has to discover the fold first, and at
  1e8 steps mostly does not.
- **Size is solved.** Zero-shot the `box3-mid` policy folded essentially nothing above 15 cm; after
  training, pinned is flat at 96-99% across sizes. Randomised drops at the largest sizes (76% for
  17.5-20 cm against 88% at 12.5-15 cm), the only size effect left.
- **Precision is not the ceiling here** the way it was for the 17-slat chains in stage 1: the median
  best error is 0.4-7.4 mm against a 15 mm cutoff. The randomised misses are falls and timeouts.
- `box2diag4-mid` and `box2x8-mid` are the hardest randomised (76-78%), `box2-mid` and `cross4-mid`
  the easiest (88%).

## Invalidated runs: the wide table pinned the arm

The first arrays (generic prior 156603 / eval 156608; `box3-mid` prior 156656, stopped half-way)
ran on a 0.6 x 0.7 m table and folded nothing -- 0/1920 in eval, every episode a timeout. The table
reached y = +0.35 m, into the arm's elbow, which in the home pose rests below the table top
(`iiwa14_link_4` at y 0.38, z 0.36; table top z 0.53). Every reset knocked the arm away and it
stayed pinned at the table edge. Diagnosis (job 156715, the `box3-mid` prior, everything else
equal): narrow table 123/128 folds, wide table 0/128. A 0.6 x 0.5 m candidate (edge at y = +0.25)
still cost folds -- 33/70 against 58/70 on 10-12.5 cm sheets -- so the table is now 0.6 x 0.4 m,
widened in x only, its robot-side edge where the narrow table's is (commit 2bf302e). The price: in
~18% of resets a corner of a large, yawed sheet overhangs the robot-side edge (by > 5 cm in ~5%).

`FoldPoolEnv._verify_arm_clear` now refuses to start if any arm joint drifts more than 2 deg while
holding the home pose for 30 steps with the table in place (0.00 deg on this table, 25.9 deg on the
old one). The void results are kept out of `docs/results`.

## Zero-shot on the VBD cloth

`bos14_fold_pool_eval_cloth.sh` (array 156964): the stage-1 cloth protocol -- 32 envs, two eval
seeds, held fold, SAPG block 5 -- so the rows sit next to the stage-1 chains. Cells: held folds,
falls, median best fold error.

| policy | 4 cm pinned | 4 cm rand | 1.5 cm pinned | 1.5 cm rand |
|---|---|---|---|---|
| **pool, `box3-mid` prior (3 seeds)** | 51/192 (27%), falls 45%, 53 mm | 63/192 (33%), 32%, 42 mm | 8/192 (4%), 58%, 54 mm | 12/192 (6%), 52%, 52 mm |
| pool, generic prior (3 seeds) | 2/192 (1%) | 4/192 (2%) | 0/192 | 0/192 |
| stage-1 `box3-mid` chain, all 6 runs | 204/384 (53%), 32%, 33 mm | 157/384 (41%), 31%, 39 mm | 68/384 (18%), 53%, 32 mm | 42/384 (11%), 40%, 38 mm |
| stage-1 `box3-mid` fixdr s2 (the prior) | 50/64 (78%), 8%, 23 mm | 21/64 (33%), 34%, 48 mm | 31/64 (48%), 20%, 14 mm | 4/64 (6%), 44%, 63 mm |
| cloth-trained reference | 32/32 (100%) | 31/32 (97%) | 54/64 (84%) | 30/64 (47%) |

Stage 2 made cloth transfer WORSE: from its own prior's 78% pinned to 19-33%, and from the six
stage-1 runs' 53% to 27%. Falls are the failure (45% pinned against 32%). Training on five rigid
shapes at 10-20 cm moved the policy away from the cloth, it did not generalise toward it.
