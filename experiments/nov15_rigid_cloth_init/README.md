# nov15_rigid_cloth_init: does the Play prior help rigid-cloth training?

Two runs of `rigid_cloth_flexiv_wuji_full_dr` (fold + unfold, all 7 pool shapes, identified hand,
SimToolReal DR, no epoch limit), identical except for the initialisation:

| Script | Init | wandb |
|---|---|---|
| `submit_prior.sh` | the right-Wuji Play prior, `pretrained_policy/flexiv_wuji/model.pth` (weights only) | `flexiv_wuji_cleanup / rigid_cloth_prior` |
| `submit_scratch.sh` | random | `flexiv_wuji_cleanup / rigid_cloth_scratch` |

The viewer logs one env per pool shape (`interactive_viewer/<shape>`, 7 per capture).

## Runs

| Run | Job | Run folder |
|---|---|---|
| prior | | |
| scratch | | |
