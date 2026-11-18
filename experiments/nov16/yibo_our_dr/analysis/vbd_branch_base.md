# VBD base with Yibo branch geometry and resets

**Restoring the branch geometry/reset values did not recover folding: 0/32
physical folds**, matching the earlier base evaluation's 0/32. Job `711188`
completed all episodes with finite states and no solver-buffer overflow.

| Measurement | Previous base | Branch geometry/reset values |
|---|---:|---:|
| Sheet size | 15 × 15 cm | 10 × 10 cm |
| Drop above tabletop | 5 cm | 15 cm |
| XY reset noise | ±5 cm | ±10 cm |
| Table-height variation | Fixed | ±1 cm |
| Physical folds / 32 | 0 | 0 |
| Falls | 5 | 12 |
| Timeouts | 27 | 20 |
| Median best fold error | 14.94 cm | 9.95 cm |
| Median post-settle peak flap height | 0.99 cm | 1.06 cm |

The smaller absolute fold error follows the smaller sheet: both medians remain
approximately one full side length, consistent with little folding. The extra
falls are descriptive results from one seed, not an isolated attribution to any
single reset parameter. Several settings change together. Post-settle flap-height
statistics use active samples after step 60; the higher initial drop must not be
mistaken for robot-induced lifting.

The original `flat7RW_fricition.pth` checkpoint, Flexiv-Wuji real-limit URDF,
Yibo actor encoding, table placement, base material, 7×7 mesh, 16 mm collision
thickness and solver settings were retained. Actions were deterministic with
exploration coefficient zero and observation/action noise disabled. Physical
fold scoring uses maximum corner error below 4 cm and footprint below 0.65;
the same absolute tolerance is relatively more permissive for the smaller sheet.
Release was disabled. Each environment contributed its first 10-second episode.

Initial simulation readbacks confirmed table centers between 0.37039 and
0.38920 m, tabletop height equal to table center plus 0.15 m within 36 nm, and
cloth centers 0.15 m above the tabletop within 0.16 μm. Sampled cloth XY centers
stayed within the requested ±0.10 m bounds. Both analytic scoring controls passed.

The new run used an RTX 6000 Ada and the previous base run an RTX A6000, so this
comparison does not establish a timing advantage. It also does not reproduce the
checkpoint's original rigid-articulation training environment. The successful
[perception audit](vbd_perception.md) checks observation compatibility; contact
and grasp dynamics remain unisolated.

Recipe: [vbd_base_branch.yaml](../vbd_base_branch.yaml). Reproduction command:

```bash
experiments/submit.sh --job-name=yibo_vbd_branch_base \
  experiments/nov16/yibo_our_dr/submit_vbd_eval.sh \
  --material base --experiment experiments/nov16/yibo_our_dr/vbd_base_branch.yaml
```

Raw result: `../results/vbd_711188/base.json`. Compact comparison:
[vbd_branch_base.json](vbd_branch_base.json). Evaluation source commit: `fe81cd7`.
