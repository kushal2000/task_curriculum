# Cloth folding: our 1e8-step finetune vs the reference checkpoint

**The reference checkpoint folds every episode; our best seed folds 93% of them, less precisely and
three times slower.** The gap is a budget gap, not a method gap: the two start from the same prior,
differ in almost nothing else, and the reference had 7.5x the environment steps.

## What `/home/yiboc/mit/model.pth` is

It carries no config, so it was identified from the weights (`scripts/analysis/` was not used; the
checks are reproduced below in prose):

| | reference | ours (`0_pretrained_s1_146678`, epoch 509) |
|---|---|---|
| SAPG groups x envs | 8 x 768 = 6144 | 4 x 1536 = 6144 |
| `epoch` / `frame` | 3800 / **7.47e8** | 509 / **1.00e8** |
| `last_mean_rewards` | 1012 | 715 |
| architecture | asymmetric LSTM-1024, obs 140 (+1 coef), act 29, 6 expl blocks | identical |
| `||W - W_prior|| / ||W_prior||` (actor+critic) | 0.039 | 0.014 |

* Same architecture and the same 140/163 obs/state widths as `pretrained_policy/model.pth`, so the
  same env family.
* Its input normalizer inherits the prior's sample count (1.004e11) and adds ~8.7e8 samples, i.e.
  it is a **finetune of the same SimToolReal prior**, not an independent run.
* The direction that normalizer moved from the prior agrees with our cloth finetune:
  `cos(d_ref, d_ours_s1) = 0.937`, against `cos(d_ref, d_scratch) = 0.53`. It was finetuned on
  **this task**, and it moved ~8x as far from the prior as our run did (`||d|| 0.0275` vs `0.0034`).

Unknown, and worth stating: its reward config (our `keypoint_rew_scale: 1500` may not be what it
trained under) and which env commit it trained against. That does not affect the evaluation below,
which runs it in *our* current env and scores it with *our* criterion.

## Evaluation (identical protocol for every row)

`scripts/cluster/bos14_cloth_eval_ckpt.sh` / `bos14_cloth_eval_pvs.sh`: 5 seeds x 32 envs x 900
steps, `success_tolerance 0.01` (the inherited rigid criterion pinned so it cannot fire),
`env.termination.success_steps=10` so a fold must hold 10 steps, `--sapg_expl_coef 0`.

| checkpoint | held folds | best_fold_err (m) | best_footprint | ep. length | falls | timeouts |
|---|---|---|---|---|---|---|
| **reference (7.5e8 steps)** | **160 / 160 (100%)** | **0.0083 +- 0.0003** | **0.498 +- 0.001** | **60** | **0** | **0** |
| ours pretrained s1 | 149 / 160 (93.1%) | 0.0274 +- 0.0008 | 0.458 +- 0.014 | 187 | 8 | 3 |
| ours pretrained s2 | 28 / 160 (17.5%) | 0.0705 +- 0.0012 | 0.641 +- 0.024 | 417 | 72 | 59 |
| ours pretrained s3 | 21 / 160 (13.1%) | 0.0692 +- 0.0058 | 0.592 +- 0.041 | 336 | 117 | 21 |
| ours scratch s1/s2/s3 | 0 / 480 | 0.087 - 0.096 | 0.81 - 0.95 | 443 - 600 | 116 | 351 |
| untrained prior | 0 / 160 | 0.1000 | 1.000 | 600 | 0 | 160 |

Spreads are across eval seeds. `folds (first entry)` equals `folds (held)` everywhere, so no row is
counting a flap that springs back.

Three differences matter more than the headline rate:

* **Precision.** 0.0083 m is a fifth of the 0.04 m tolerance; our s1's 0.0274 m is two thirds of it.
* **Footprint 0.498, sd 0.001** is the sheet halved, repeatedly, to the same place. Our s1 averages
  0.458 with sd 0.014 -- it crumples the sheet somewhat past a clean halving, which the render
  confirms (a wad under the fingers rather than a flat lay).
* **Speed and safety.** 60 steps to a held fold against 187, with 0 falls against 8. Seeds s2/s3
  knock the sheet off the table in 45% / 73% of episodes; the reference never does.

## Which SAPG block

The reference folds in **every** block -- a probe at coefficients 50/40/30/20/10/0 (1 seed, 32 envs,
job 150896) gives 32/32 held folds at all six, `best_fold_err` 0.0069-0.0117. That is what a
converged SAPG run looks like, and it is the sharpest contrast with our runs, where only block 5
learned (0.55-0.64 successes/episode against ~0.001 in blocks 0-4, see
`cloth_prior_vs_scratch.md`). At 1e8 steps the exploration-bonus blocks have not yet been pulled
onto the task; at 7.5e8 they have.

## Renders (matched framing)

`scripts/cluster/bos14_render_cloth.sh` with identical `SEED=1 NUM_ENVS=8 STEPS=600 STRIDE=2
CAM_OFFSET="0.45 -0.45 0.78" LOOK_AT="0.0 0.0 0.55"`, so world *w* starts from the same sheet pose
in both, at `--sapg_expl_coef 0`. Jobs 150903 (reference) / 150904 (ours).

* `videos/10_ref_vs_ours/ref/w0.mp4`, `w1.mp4`
* `videos/10_ref_vs_ours/ours_s1/w0.mp4`, `w1.mp4`

In 600 steps the filmed reference env folds 4 times (w0: steps 122/263/374/543) against our 3
(w0: 225/356/548); across all 8 envs, 62 fold events against 32. Reference w0 at step 118 reads
`fold err 0.010 / footprint 0.510` with the vermilion moving half lying flat on the gold stationary
half; ours at step 224 reads `0.035 / 0.313`, the sheet bunched under the fingers. Both are genuine
folds -- a slide leaves footprint at 1.0 -- but only one is neat.

## What this says about the 1e8-step comparison

`cloth_prior_vs_scratch.md` claims "the prior is what makes the fold learnable at 1e8 steps". This
adds the ceiling: the same prior, same task and ~7.5x the steps reaches 100% at a fifth of the
tolerance, so our prior-arm numbers (93% / 18% / 13%) are an early slice of that curve, and the
seed variance is early-training variance rather than a property of the method. It does **not**
show scratch would get there with 7.5e8 steps; nothing here tests that.
