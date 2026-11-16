# nov15_fabric_eval: the 2-GPU rigid-cloth policies on 7 measured fabrics

Zero-shot sim-to-sim: the two `nov15_rigid_cloth_init` 2-GPU runs (trained on the PhysX rigid-plate
pool, fold + unfold mixed) evaluated on the deformable VBD sheet (Newton,
`Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0`) set up as each measured fabric.

| tag | run | init |
|---|---|---|
| `prior` | 659538 `rigid_cloth_prior_2gpu` | pretrained Play policy |
| `scratch` | 659539 `rigid_cloth_scratch_2gpu` | from scratch |

Both runs are still training; `submit.sh` evaluates each run's newest `snap_*_ep_<N>.pth` at submit
time (the epoch is in the job log and in each JSON's `checkpoint`).

Matrix: 2 policies x {fold, unfold} x 7 fabrics (`isaacsimenvs/cfg/fabric/<name>_measured.yaml`:
big_napkin, chiffon, cotton, denim, napkin, silk, towel) = 28 array tasks, 14 per policy.

Protocol (same as the 2026-11 measured-fabric eval, `engaging_fabric_eval.sh`):
- the preset's cloth parameters **with its plasticity** (`fabrics.py --plastic`);
- no DR, deterministic actions, SAPG block coef 0 (the last block, task reward only);
- 1.5 cm keypoint cutoff, success = the target held 10 steps (`fold_held`), 900 steps;
- 32 envs x eval seeds 3-8 = 192 episodes per cell.

```
experiments/nov15_fabric_eval/submit.sh             # two array jobs, 0-13 each
python experiments/nov15_fabric_eval/summarize.py   # success table from results/
```

Results: `results/<tag>__<task>__<fabric>_s<seed>.json`, plus each task's Slurm log and GPU log.

Videos (`render.sh`, local GPU): `debug_outputs/videos/<month>/<date>_fabric_eval/<tag>__{rigid,cotton}__{fold,unfold}.mp4`,
one scored episode each (`.json` next to each clip: outcome). Rigid clips film the box2-mid sheet.

## Jobs

Submitted 2026-11-15 20:27 from 7a1161c + uncommitted changes (full calls in `submissions.log`):

| array job | policy | checkpoint |
|---|---|---|
| 672247 (0-13) | prior | `snap_0_rigid_cloth_prior_2gpu_ep_1300.pth` |
| 672248 (0-13) | scratch | `snap_0_rigid_cloth_scratch_2gpu_ep_1100.pth` |

Array index = task * 7 + fabric (0-6 fold, 7-13 unfold; fabrics in the order above).

## Results

(to fill in)
