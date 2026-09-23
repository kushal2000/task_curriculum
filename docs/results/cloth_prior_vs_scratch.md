# Cloth folding: SimToolReal prior vs training from scratch

**The prior is what makes the fold learnable.** With identical config and 1e8 env-steps, all three
finetuned runs learned to fold (held-fold rate 93%, 18%, 13%), and none of the three from-scratch
runs produced a single held fold in 480 evaluation episodes.

## Setup

Commit `5d3a5b7`, bos14, launched by `scripts/cluster/submit_prior_vs_scratch.sh`.

| | |
|---|---|
| arms | `pretrained` (`--checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights`) vs `scratch` (no checkpoint). **Nothing else differs.** |
| seeds | 1, 2, 3 per arm (`agent.params.seed`) |
| scale | 4 nodes x 1 RTX PRO 6000, 1536 envs/rank, horizon 32, minibatch 24576/rank (98304 per optimizer step) |
| budget | 509 epochs = 1.0e8 env-steps, ~2.5 h per run |
| reward | `Cloth.yaml` at `keypoint_rew_scale: 1500` (d4c2158), `success_steps: 1` |
| jobs | pretrained 146678 / 146680 / 146682, scratch 146679 / 146681 / 146683 |

## Evaluation (held folds)

`scripts/cluster/bos14_cloth_eval_pvs.sh`: final checkpoint (epoch 509), 5 seeds x 32 envs x 900
steps, `success_tolerance 0.01`, **`env.termination.success_steps=10`** so a fold must persist 10
steps to count, **`--sapg_expl_coef 0`** (see below). Raw: `cloth_pvs_*_c0_s*.json`.

| checkpoint | held folds | best_fold_err (m, across eval seeds) | falls |
|---|---|---|---|
| untrained SimToolReal | 0 / 160 | 0.1000 +- 0.0000 | 0 |
| **pretrained s1** | **149 / 160 (93.1%)** | 0.0274 +- 0.0008 | 8 |
| **pretrained s2** | **28 / 160 (17.5%)** | 0.0705 +- 0.0012 | 72 |
| **pretrained s3** | **21 / 160 (13.1%)** | 0.0692 +- 0.0058 | 117 |
| scratch s1 | 0 / 160 | 0.0956 +- 0.0010 | 1 |
| scratch s2 | 0 / 160 | 0.0923 +- 0.0022 | 9 |
| scratch s3 | 0 / 160 | 0.0874 +- 0.0015 | 106 |

Per arm: **pretrained 198 / 480 held folds, scratch 0 / 480.** `folds (first entry)` equals
`folds (held)` for every checkpoint, so these are settled folds, not a flap that touches the target
and springs back.

## Training curves (all blocks, mean +- sd over 3 seeds)

| | pretrained, first 10% | pretrained, last 10% | scratch, first 10% | scratch, last 10% |
|---|---|---|---|---|
| `episode_final/done_fold` | 0.119 +- 0.041 | **0.591 +- 0.039** | 0.013 +- 0.008 | 0.003 +- 0.000 |
| `episode_cumulative/keypoint_rew` | 20.7 | **75.7** | 4.7 | 9.2 |
| `episode_cumulative/hand_actions_penalty` | -26.6 | -20.2 | -34.3 | -21.8 |
| `episode_final/done_timeout` | 0.70 | 0.24 | 0.94 | 0.79 |
| `episode_cumulative/total_reward` | 154 | 696 | 14 | 36 |

Unlike the earlier finetunes (`cloth_finetune_reward_imbalance.md`), fold progress rises
monotonically instead of being traded for lower action penalties -- the 200 -> 1500 change worked.

`done_fold` is a fraction of *completed* episodes, and at `success_steps=1` a fold ends the episode
immediately, so folding envs contribute disproportionately many episodes. It is not a per-env rate.

## SAPG block: evaluate at coefficient 0, not 50

Every fold in training came from **block 5** (coefficient 0.0, task reward only):

| last 10% successes/episode | block 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| pretrained s1 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | **0.644** |
| pretrained s2 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | **0.554** |
| pretrained s3 | 0.001 | 0.002 | 0.001 | 0.002 | 0.002 | **0.575** |

The eval player hard-coded 50.0 -- block 0, the released SimToolReal checkpoint's value. The first
evaluation pass (job 146704, `cloth_pvs_*_c50_s*.json`) therefore measured block 0 and gave
133 / 0 / 4 held folds for pretrained s1/s2/s3: right that s1 folds, badly wrong about s2 and s3.
`--sapg_expl_coef` now exists on `episodes.py` and `render_newton.py` (default 50.0, unchanged).
**Any finetuned SAPG checkpoint should be evaluated at the block that learned.**

## Caveats

* **Seed variance in the prior arm is large** (93% vs 18% vs 13%). s2 and s3 also knock the sheet
  off the table in 45% and 73% of eval episodes, against ~17% during training -- the deterministic
  mean action is much more destructive than the stochastic training policy for those two. Three
  seeds establish "prior >> scratch", not a reliable fold rate for the prior.
* **1e8 steps is short for a from-scratch dexterous policy.** The claim is "at equal budget", not
  "scratch can never learn this". Scratch s2/s3 do move best_fold_err a little off the flat 0.100.
* **Not visually verified.** The footprint half of the criterion has a documented fail-permissive
  mode (`cloth_fold_baseline.md`). best_fold_err 0.027 m for s1 is well inside tolerance, which
  makes a criterion artefact unlikely there, but a render of s1 at `--sapg_expl_coef 0` is the check.
* Eval is deterministic; training is stochastic. Training `done_fold` and eval held-fold rates are
  not the same quantity and should not be compared directly.

## Throughput on bos14 (for sizing future runs)

1536 envs/rank: 1 GPU 3,516 env-steps/s; 2 nodes 6,520 (1.85x); 4 nodes 11,450 (3.26x). More than
~1820 envs on one GPU fails at solver build (per-body contact buffer x bodies exceeds Warp's int32
array limit), and doubling 768 -> 1536 envs only gained 18%: the GPU, not memory, is the bottleneck.
