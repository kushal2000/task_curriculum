# Cloth resets and thickness curriculum — November 17, 2026

Four fresh SAPG runs share `base.yaml`: `prior.yaml` initializes network weights
from Yibo's `flat7RW_fricition.pth`; `scratch.yaml` starts with random weights.
None resumes yesterday's optimizer or epoch. The matching `*_no_curriculum.yaml`
controls disable adaptation and retain the original 2–20 mm thickness range, with
both reset checks enabled. In the two curriculum runs, all five symmetric rigid designs
use one current thickness, initially **20 mm**. Length/width variation remains.

## Thickness adaptation

- At least **90% success over the most recent 4,096 completed episodes** advances
  the curriculum. Success is the task's final fold-and-release success. Every
  guard failure and timeout counts in the denominator.
- Multiply current thickness by **0.9**, clamped to **2 mm**. There is no prebuilt
  grid of thickness levels and no thickness randomization within a stage.
- Clear the success window after each advance. Unfinished episodes interrupted by
  geometry reconstruction do not contribute evidence to the next window.
- PhysX cannot update articulation hinge anchors while running. At an advance,
  stop and reconstruct native collision shapes, inertia, and hinge anchors; rebind
  simulator views; restart all episodes. The trainer, network, optimizer, epoch,
  and frame count continue in the same process. PPO receives terminal boundaries
  for interrupted trajectories, so recurrent state cannot cross the reset.
- Recreate observations, physical targets, work-history checks and robot clearance
  from the actual new geometry. Preserve areal-density mass; update thickness
  contributions to inertia. Interactive viewer captures use the actual new prisms.
- Checkpoints store current thickness and the fresh-window history. A training
  resume restores those plus optimizer state, while restarting physical episodes
  and discarding stale observations/recurrent rollout state.

## Reset checks

Both checks run at physics frequency (120 Hz, including intermediate substeps):

1. **Outside table:** any physical plate vertex projects outside the actual table
   surface in the table's XY frame. A 10 micrometre tolerance handles numerical
   roundoff; lifting above the table is allowed.
2. **Press-driven fold:** the audited symmetric robot-source lifting-work check.
   A rising half at 30–90 degrees gains gravitational energy while the opposite
   half stays near flat and carries opposed robot/table loading, but direct
   opening work from actual moving robot contact points supplies less than 10%
   of that gravitational rise. Normal and friction contacts both contribute.
   Static robot contacts earn no work credit. History rearms on settled flat
   halves; rewards are disabled until the whole cloth has a valid flat baseline.

A failing step resets without paying fold, release, or shaping rewards. No new
negative reward or arbitrary compression-force cap is added. The mechanical
rationale and held-out rollout results are in
`../nov17_pressing_audit/analysis/lift_work.md`.

## Launch and monitoring

```bash
experiments/nov17_cloth_resets/submit_prior.sh
experiments/nov17_cloth_resets/submit_scratch.sh
experiments/nov17_cloth_resets/submit_prior_no_curriculum.sh
experiments/nov17_cloth_resets/submit_scratch_no_curriculum.sh
```

W&B project: `rigidClothDR`; group: `nov17_cloth_resets`. All four use 12,288
environments, the Yibo robot/table/perception interface, our material DR and delays,
and the existing 1,000 fold + 1,000 release bonuses. W&B logs current thickness in
millimetres, current success window, number of completed episodes and advances.
The 10% reduction and 4,096-episode window are configurable defaults; the requested
90% threshold is explicit in `base.yaml`.

## Validation

CPU tests cover threshold equality, failure accounting, fresh windows, rolling
replacement, target clamping, checkpoint history, physical table containment,
physics-substep latching, work credit and invalid-success vetoes. Native GPU smoke
tests exercise all five designs, geometry reconstruction, actual collision
surfaces and central-hinge folded height, inertia readback, continued stepping,
and forced off-table reset with zero reward. Full-batch contact/reset checks
passed at 12,288 environments without nonfinite state or a buffer overflow.

Raw smoke outputs are local ignored files in `results/`; training outputs and
checkpoints remain in `debug_outputs/train_logs/nov/`.

## Submitted jobs

| Variant | Initialization | Job | W&B |
| --- | --- | --- | --- |
| 20 → 2 mm curriculum | prior | 742942 | [run](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_prior_20to2mm_curriculum_resets_20261117_111751) |
| 20 → 2 mm curriculum | scratch | 742943 | [run](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_20to2mm_curriculum_resets_20261117_111818) |
| 2–20 mm, no curriculum | prior | 742944 | [run](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_prior_2to20mm_no_curriculum_resets_20261117_112219) |
| 2–20 mm, no curriculum | scratch | 742945 | [run](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_2to20mm_no_curriculum_resets_20261117_112245) |

Yesterday's jobs 710887, 710889, 711271 and 711273 were cancelled at the user's request; their artifacts were retained. These runs use training commit `1eae373`.
