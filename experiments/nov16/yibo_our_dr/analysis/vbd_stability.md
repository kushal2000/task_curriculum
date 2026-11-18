# VBD solver stability search

**All seven candidates completed. None met all declared probe limits.** Raising
VBD iterations alone did not substantially improve the measured robot
contact-plane crossing in this fixed-input test. This is not an exact hand-mesh
penetration measurement and does not rule out contact-convergence problems.

The table below shows worst values across the four prescribed-manipulation worlds.
All candidates also completed flat-drop and folded-release probes. Every run
stayed finite, had no buffer overflow, and reported zero cloth-node penetration
into the native table box at the sampled physics boundaries.

| VBD iterations / cloth substeps / coupling passes | Robot plane crossing | Edge stretch | Late peak speed | Batch wall time |
|---|---:|---:|---:|---:|
| 6 / 6 / 1 | 13.95 mm | 1.64× | 0.907 m/s | 89 s |
| 12 / 6 / 1 | 13.97 mm | 1.64× | 0.907 m/s | 145 s |
| 40 / 6 / 1 | 13.97 mm | 1.64× | 0.907 m/s | 410 s |
| 80 / 6 / 1 | 13.97 mm | 1.64× | 0.907 m/s | 787 s |
| 24 / 3 / 1 | 13.32 mm | 1.53× | 0.042 m/s | 143 s |
| 6 / 12 / 1 | 8.52 mm | 2.17× | 0.00077 m/s | 150 s |
| 12 / 6 / 2 | 13.61 mm | 1.64× | 0.668 m/s | 276 s |

The 12-cloth-substep case illustrates why one metric cannot select the solver:
its lower contact-plane crossing and late speed coincided with all 49 nodes in
each manipulation world reaching near-ground height. It also produced the largest
stretch. A quiet sheet resting on the ground is not evidence of successful
hand-cloth contact. Falls remain behavioral observations rather than automatic
claims of numerical divergence.

The three-cloth-substep case reduced late motion, but still exceeded the
provisional contact-plane and stretch limits. Doubling coupling passes did not
recover an acceptable result either. Stretch and motion can reflect material
compliance and the prescribed forcing; these metrics alone do not prove an
integration error. There is no supported default change from this screen.

## Controls and scope

- One base material, one seed, twelve native worlds: four each of flat-drop/hold,
  prescribed manipulation and folded release, simulated for five seconds.
- Every candidate restored the same immutable initial state and action tape;
  physical-config, initial-state and action hashes matched. No policy feedback or
  automatic resets changed the inputs.
- Native numerical settings were read back. All 600 physics boundaries advanced
  exactly 1/120 s; diagnostics were sampled at those boundaries. Timing includes
  diagnostics and is one shared batch per candidate, not repeated performance trials.
- All four manipulation worlds registered robot contact. Their initial robot pose
  came from checkpoint observation row 18447; its source SHA is stored in the fixture.
- Contact-plane crossings use cached proxy normals and points at current poses,
  not an exact signed query of the current robot mesh. Independent table-box
  depths cover cloth nodes, not triangle interiors or between-boundary motion.
- Acceptance targets in [vbd_stability.yaml](../vbd_stability.yaml) are explicit
  provisional test requirements, not calibrated real-cloth truth. The analyzer
  correctly reported no acceptable candidate rather than selecting a false optimum.

Full generated report: `../results/vbd_stability_contact_20261116_native/report/stability.md`.
Compact tracked results: [vbd_stability_summary.json](vbd_stability_summary.json).
Benchmark array `722887`; automatic report job `722923`.
