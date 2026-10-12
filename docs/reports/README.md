# Cloth folding: index of reports

The human-readable write-ups live here. The raw data they quote (one JSON per eval task, `.pt`
tensors, eval manifests) is in `docs/results/` and is read and written by the summary scripts; there
is normally no need to open it.

## Status in one paragraph (2026-10-07)

**The best cloth-folding policy so far is PhysX "run 2, flat pool" (job 158376).** It starts from the
generic SimToolReal prior and trains for 7.47e8 env-steps on PhysX, on 5 rigid hinged-plate shapes,
without ever seeing cloth. Zero-shot on the Newton VBD cloth it folds 81% pinned / 56% randomised at
the 4 cm cutoff and 46% / 29% at 1.5 cm. For reference, the policy trained directly on the cloth
scores 100% / 97% (4 cm) and 84% / 47% (1.5 cm). The follow-up fine-tune of run 2 on 7 shapes
(ft7, 1.2e9 env-steps) was evaluated on the wrong checkpoint, one from about epoch 10. **Whether that
fine-tune helped is not yet known.** See `4_physx/finetune_7shapes.md` and `docs/training_plan.md`.

## Glossary

| term | meaning |
|---|---|
| **VBD cloth** | Newton's particle cloth (`Isaacsimenvs-Cloth-Direct-v0`): the real target. Slow to simulate. |
| **rigid chain** | The sheet approximated as rigid plates joined by hinges (`RigidCloth`), 2-3x faster or more. `box3-mid` = two plates plus a 2 mm crease bar. |
| **fold pool** | Stage 2: one policy folds 5 hinged-plate shapes (`box2-mid`, `box2diag4-mid`, `box2x8-mid`, `cross4-mid`, `corners8-mid`), 10-20 cm on a side. |
| **stage 1 / stage 2** | Stage 1 = a single rigid chain (box3-mid); stage 2 = the shape pool. |
| **Newton / PhysX** | The two physics backends. Newton (MJWarp) is the original one; PhysX (Isaac Sim 5.1) is 10-20x faster. All rigid training since 10-05 runs on PhysX. |
| **generic prior** | `pretrained_policy/model.pth`, the general tool-use policy that every run starts from. |
| **reference** | **Two different checkpoints.** In `cloth_vs_reference.md` it is the external `/home/yiboc/mit/model.pth`. In every later report it is our own cloth-trained policy resumed to 7.47e8 steps, `0_pretrained_s1_resume_150911`. Both fold the cloth 100%. |
| **held fold** | Success = the fold reached and held for 10 steps. |
| **4 cm / 1.5 cm** | Keypoint tolerance. 4 cm is loose (a sloppy fold passes); 1.5 cm means a neat fold. |
| **pinned / rand** | At eval the sheet starts fixed and centred / from the training reset distribution. |
| **coef 0** | SAPG block 5 (task reward only). Every eval uses it. |

## The line of work and its reports

### 1. Training directly on the VBD cloth (Aug - Sep): `1_vbd_cloth/`

Conclusion: **learnable, but it needs the prior, many steps, and a slow simulator.** Fine-tuned from
the generic prior for 7.47e8 steps it reaches 100%; from scratch it does not fold once in 1e8 steps.

| report | date | what it covers | status |
|---|---|---|---|
| `cloth_fold_baseline.md` | 08-18 | early baseline (old physics), 3/160 | **superseded** (not comparable after the SIM1 physics change) |
| `cloth_finetune_reward_imbalance.md` | 08-18 | the policy saved action penalties instead of folding: `keypoint_rew_scale` 200 was too small, raised to 1500 | valid, fixed |
| `cloth_nan_at_reset.md` | 08-18 | robot NaNs cluster in the first steps after reset | not fixed; small effect (~1e-6 per step) |
| `cloth_sim1_physics_and_fold_criterion.md` | 08-26 | SIM1 physics adopted; bugs in the fold criterion (target lifted by the wrong offset, eval-statistics artefacts) | valid |
| `cloth_prior_vs_scratch.md` | 09-23 | **prior vs scratch** at 1e8 steps: prior learns on 3/3 seeds (93% / 18% / 13%), scratch 0/480. Jobs 146678-146683 | valid |
| `cloth_vs_reference.md` | 09-23 -> 09-29 | our s1 resumed to 7.47e8 steps (job 150911) also reaches 100%, but at 16 mm precision against the reference's 8 mm | valid; 150911 is the later "reference" |

### 2. Approximating the cloth with rigid chains (late Sep - 10-03): `2_rigid_chain/`

Conclusion: **the chains train fast and well, but transfer poorly to the real cloth.** With the
start pose, contact surface and friction fixed, box3-mid's cloth transfer rose from ~0% to ~50% at
4 cm, but only 11-21% at 1.5 cm. Physics randomisation (DR) showed no benefit in three rounds.
More slats transfer worse: the coarsest chain, box3-mid, is the useful one.

| report | what it covers |
|---|---|
| `rigid_cloth_approximation.md` | how the 5 chains are built and how closely their geometry and physics match the cloth (no training). Figures in the same folder |
| `rigid_cloth_rl_comparison.md` | 5 chains + the cloth, 3 seeds each: speed, learning, transfer. Round 1 (155046), round 2 with fixes (155349), round 3 many-slat chains (155712) |

### 3. Stage 2 shape pool on Newton (10-04 - 10-05): `3_fold_pool_newton/`

Conclusion: **started from the box3-mid policy, it folds every shape and size (98% pinned / 82%
rand), but cloth transfer gets worse** (78% for its prior, 27% after). From the generic prior it does
not learn in 1e8 steps (0.5%).

| report | what it covers |
|---|---|
| `fold_pool_stage2.md` | runs 156746 (box3 prior) / 156744 (generic prior), results per shape and size, the voided runs (a wide table pinned the arm), cloth transfer |

### 4. After the move to PhysX (10-05 - 10-07): `4_physx/`

Conclusion: **PhysX is 10-20x faster, and rigid policies trained on PhysX transfer to the cloth
better.** Best so far: the shape pool trained straight from the generic prior (run 2, 81% / 56%).

| report | what it covers |
|---|---|
| `physx_port.md` | how the PhysX version is built and what matches Newton; the hinge-armature trap (1e-5 makes the other plate jump; now 1e-3); the PhysX stage 1 -> stage 2 pipeline (157449 -> 157622, 65% / 47%) |
| `physx_generic_prior_runs.md` | **three 7.47e8-step runs**: run 1 with partly folded starts (158375), **run 2 with flat starts (158376, best so far)**, run 3 box3 chain only (158324) |
| `eval_matrix.md` | full results of those three runs on every target (PhysX pool / box3, Newton pool / box3, VBD cloth); generated by `px_eval_matrix_summary.py` |
| `finetune_7shapes.md` | runs 1 / 2 fine-tuned another 1.2e9 steps on 7 shapes with non-target starts (158820 / 158821). **The eval measured the ~epoch-10 best checkpoint, not the trained policy; to be re-measured** |
| `large_cloth.md` (+ `large_cloth_size_sweep.md`, `large_cloth_menu_policies_s20.md`) | cloth enlarged to 12.5-20 cm: success falls steadily with size, 10% at 20 cm, mostly because the sheet is dragged off the table |
| `cloth_finger_pierce.md` | fingers press through the VBD cloth and the sheet hangs on them; no solver setting fixes it (VBD has particle-vs-body contact only) |
| `contact_stiffness.md` | fingers sinking into plates and plates into the table in the Newton pool are soft contact; stiffer contacts have no downside and fold more precisely |

## Training-run registry

Cloth score = zero-shot on the Newton VBD cloth, 3 training seeds pooled, in the order 4 cm pinned /
4 cm rand / 1.5 cm pinned / 1.5 cm rand. "—" = not measured. Checkpoints are under
`outputs/<date>/<time>/<name>/nn/`.

| run (name prefix) | job | sim / task | start | steps | cloth score | report |
|---|---|---|---|---|---|---|
| `0_pretrained_s{1,2,3}` | 146678/80/82 | VBD cloth | generic | 1e8 | s1 93%, s2 18%, s3 13% (4 cm pinned) | prior_vs_scratch |
| `0_scratch_s*` | 146679/81/83 | VBD cloth | scratch | 1e8 | 0% | prior_vs_scratch |
| **`0_pretrained_s1_resume_150911`** ("reference") | 150911 | VBD cloth | 146678 s1 | 7.47e8 | **100 / 97 / 84 / 47%** | cloth_vs_reference |
| `0_rc*` (round 1, 5 chains) | 155046 | Newton chain | generic | 1e8 | box3-mid 4% (pinned) | rl_comparison |
| `0_rc2_box3-mid_fix(dr)_s*` | 155349 | Newton chain (fixed) | generic | 1e8 | 52 / 43 / 14 / 11% (fix) | rl_comparison |
| `0_rc3_*` (many-slat chains) | 155712 | Newton chain | generic | 1e8 | 8-34% / 26-35% / 1-7% / 4-6% | rl_comparison |
| `0_fp_box3_s*` | 156746 | Newton pool | rc2 box3-mid fixdr s2 | 1e8 | 27 / 33 / 4 / 6% | fold_pool_stage2 |
| `0_fp_pool_s*` | 156744 | Newton pool | generic | 1e8 | 1 / 2 / 0 / 0% | fold_pool_stage2 |
| `0_pxchain_arm1e3_s*` | 157449 | PhysX chain | generic | 1e8 | 55 / 45 / — / — | physx_port |
| `0_pxchain_arm1e3_e12288_4e8_s*` | 157494 | PhysX chain | generic | 4e8 | 63 / 44 / — / — | physx_port |
| `0_pxpool_pxs1_arm1e3_e6144_s*` | 157622 | PhysX pool | 157449 s1 | 1e8 | 65 / 47 / 37 / 21% | physx_port |
| `0_pxchain_box3_s*` (run 3) | 158324 | PhysX chain | generic | 7.47e8 | 39 / 36 / 8 / 15% | generic_prior_runs |
| `0_pxpool_one_initfold_s*` (run 1) | 158375 | PhysX pool + partly folded starts | generic | 7.47e8 | 44 / 46 / 25 / 24% | generic_prior_runs |
| **`0_pxpool_one_flat_s*` (run 2)** | **158376** | **PhysX pool** | **generic** | **7.47e8** | **81 / 56 / 46 / 29%** | generic_prior_runs |
| `0_pxpool_ft7lr_flat_s*` | 158820 | PhysX 7-shape pool | run 2 | +1.2e9 | **not measured** (the ep_1526 checkpoints are unevaluated; 81 / 54 / 48 / 21% is the ~epoch-10 policy) | finetune_7shapes |
| `0_pxpool_ft7lr_initfold_s*` | 158821 | PhysX 7-shape pool | run 1 | +1.2e9 | **not measured** (48 / 51 / 16 / 26% is the ~epoch-10 policy) | finetune_7shapes |

The four policies in the live viewer's menu (`isaacsimenvs/eval/live_viser.py`): run 2 s3, run 1 s1,
run 3 s3, 157622 s1.

## Smaller findings without their own report

- **Cloth CUDA graph.** `cloth.use_cuda_graph=true` raises the VBD cloth from ~4 to ~18-20 policy
  steps/s, the same at 4 and 32 envs because the cost is kernel launches. Results are unchanged
  (reference 64/64 with the graph, 32/32 without). The viewer turns it on; training still defaults to
  off and has not been measured at 1536 envs. Raw data: `docs/results/cloth_graph_*.json`.
- **When fine-tuning a converged policy, use the source run's final learning rate** (here 8.8e-6). At
  the default 1e-4 the policy is destroyed within a few updates (`finetune_7shapes.md`).
- **The eval scripts evaluate `nn/<name>.pth` by default, which is the best-reward checkpoint, not the
  last one.** Normally the two are minutes apart, but in a fine-tune the best checkpoint can stay at
  the very start, as it did for ft7. To evaluate a fine-tune, use `nn/last_*_ep_<N>_*.pth`.

## Planned work

`docs/training_plan.md` (drafted 2026-10-11, awaiting confirmation): re-evaluate the ft7 ep_1526
checkpoints, then continue training ft-flat and ft-initfold, with a learning-rate comparison.

## Earlier documents

`docs/phase1_baseline.md`, `docs/phase3_newton_transfer.md` and `docs/phase4_cable_env.md` cover the
phases before cloth folding (Play task baseline, transfer to Newton, the cable task).
`docs/isaacsim_installation.md` covers environment setup.
