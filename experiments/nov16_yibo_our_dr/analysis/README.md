# Yibo checkpoint transfer analysis

The main folding gap is a **combined physics mismatch**, rather than the thickness feature or delay/noise alone. The checkpoint performs well in the Yibo-compatible rigid setup, partially recovers when historical physics is restored to our setup, and transfers poorly to the evaluated VBD fabrics.

## Rigid cloth: what helped and what did not

These are controlled seed-0 evaluations of the unchanged checkpoint, with 64 first episodes per case:

| Setup | Fold criterion reached | Fold + release |
| --- | ---: | ---: |
| Our DR | 15/64 | 1/64 |
| Yibo-compatible reference | 58/64 | 4/64 |
| Our DR + historical contact, density and hinge settings together | 37/64 | 4/64 |
| Our DR + historical contacts alone | 24/64 | 5/64 |
| Our DR + 10–20 cm size/placement procedure | 24/64 | 4/64 |
| Yibo reference + our sensor/action DR | 55/64 | 1/64 |
| Yibo reference + actual 2 mm thickness feature | 56/64 | 8/64 |

Restoring density alone reduced violent motion and falls but barely changed folding. Restoring the historical hinge settings alone reduced folding. Their combined restoration with historical contacts worked much better, so there is no single parameter to blame. The smaller-size result also changes placement conditioning and sampled material realizations; it is not a strict geometry-only intervention.

The inverse tests are useful: the otherwise good Yibo setup still folded in 55–56 episodes with our sensor/action DR or the actual 2 mm thickness feature. These are not the main folding-collapse mechanisms in this screen. Release counts are too small for a strong attribution, and the original checkpoint was not trained for whole-robot withdrawal.

The checkpoint interface was separately checked against historical code and saved checkpoint observations. A [live VBD perception audit](vbd_perception.md) also checked 12 real environments, 48 controlled cloth states, signed robot quaternions, particle ordering and actual table geometry. `yibo` mode intentionally preserves its incorrect goal-quaternion convention for compatibility. Physical success uses the correct cloth target. The tests here hold that convention fixed.

See [rigid.md](rigid.md) for all interventions, motion diagnostics, controls and limitations, and [rigid.png](rigid.png) for the comparison plot.

## VBD fabrics: where transfer failed

All eight presets completed 32 first episodes each. **Only chiffon reached the fold criterion, once; the other seven presets each reached zero times.**

| Material | Folds / 32 | Falls |
| --- | ---: | ---: |
| Base | 0 | 5 |
| Big napkin | 0 | 16 |
| Chiffon | 1 | 14 |
| Cotton | 0 | 11 |
| Denim | 0 | 14 |
| Napkin | 0 | 17 |
| Silk | 0 | 15 |
| Towel | 0 | 7 |

The strongest measured failure patterns were sheets staying nearly flat or misaligned, and sheets being transported off the table. Median best fold errors remained around 14.7–14.9 cm for 15 cm sheets. After settling, median peak moving-half particle heights were only 1.0–1.8 cm above the tabletop. Napkin and silk centroids left the tabletop in 19/32 episodes each.

A separate actuator check confirmed nonzero actions and substantial robot movement: median peak palm displacement was 16 cm over two seconds. This rules out a globally frozen robot, but does not prove successful grasping or identical PhysX/Newton contact dynamics. All accepted material runs stayed finite and within solver buffer capacities. The earlier overflowing attempt was rejected and rerun with larger buffers.

VBD also changes the robot/contact solver to Newton/MuJoCo-Warp, so this is a transfer test of the complete VBD setup, not a clean isolation of cloth deformation or individual material coefficients. Whole-robot release is unsupported on this backend and was not evaluated. See [fabrics.md](fabrics.md) and [fabrics.png](fabrics.png) for the full diagnostics and protocol.

## What this suggests next

Keep the target DR grounded in the intended cloth materials rather than restoring heavy sheets, large numerical armature or inflated contact shells merely to improve this checkpoint's score. The submitted fine-tune and scratch experiments test learning under the target dynamics. For VBD, the next useful controlled test is grasp/contact and robot-tracking parity before assigning the entire gap to fabric parameters.

All fold counts here use maximum physical corner error below 4 cm and footprint below 0.65. Rigid release additionally uses conservative all-robot clearance bounds at 10 cm. These are current-task criteria, not the original Yibo 1.5 cm benchmark. Results are preliminary, with one seed per condition.

Older 8/64 and 60/64 diagnostics used the policy loader's implicit reset seed 42. The rigid analysis explicitly reseeds after loading, and its baselines were reproduced exactly after the material-allocation fix.
