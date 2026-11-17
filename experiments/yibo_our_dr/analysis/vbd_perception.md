# Real VBD observation audit

**The audited Yibo actor pathway is correctly wired.** No particle-order, environment-order, robot-frame, joint-order, fingertip-offset, quaternion-convention, or actor-field assembly error was found. This is a direct GPU audit of the real Newton backend, not an inference from the robot moving.

## Independent checks

The audit creates **12 actual VBD environments**, including multi-digit IDs 10 and 11, and injects controlled particle/joint states without a policy rollout. Four cloth states (flat, 90° fold, 180° fold, and a deformed 65° fold) each use yaw 0°, 90°, 180°, and 37° across multiple environment origins: 48 controlled physical states. Joint positions differ by environment; joint and particle velocity sentinels are nonzero.

- Raw Newton `particle_world` tags map numerically to environments 0 through 11, with 49 particles per world. Each world's 72 actual mesh triangles match the independently generated checkerboard connectivity. Imported rest coordinates agree within **1.68e-7 m**. This is checked before controlled particle writes, so write/read symmetry cannot hide an import permutation.
- Raw simulation particle buffers grouped by world tags exactly match the cloth observation rows. Controlled point/velocity writes read back exactly. Raw model body labels and world tags exactly match robot link readouts. No lexicographic `0,1,10,11,2` environment permutation was observed.
- Independent URDF forward kinematics, using the decoded live joint angles, matches palm and fingertip LINK positions within **4.42e-7 m**. Newton `body_state_w` pose exactly matches its link-pose property; there is no COM-position substitution. The state property uses link pose plus COM velocity, consistent with its documented legacy contract.
- Functions loaded from frozen historical commit `2b82faf1c73f5782068f67eac9cd6085aa1e27e7` independently reconstruct the old cloth object, goal and full actor vector. All **134 actor fields** agree within **3.58e-7**. The unchanged checkpoint's deterministic actions agree within **6.44e-6**, after identical recurrent-state resets. The oracle does not call the current observation builder or current cloth-pose fit.

## Stronger saved-PhysX robot check

Twelve rows spanning indices 0 through 24575 of the original checkpoint's saved observations were decoded using the actual URDF joint limits, then injected into the live Newton robot. The resulting robot observations match the saved PhysX observations:

| Quantity | Maximum absolute discrepancy |
|---|---:|
| Normalized joint position | 1.20e-7 |
| Palm position in the policy's Play base frame | 12.9 μm |
| Signed palm quaternion components | 7.16e-7 |
| Fingertip positions relative to palm | 16.7 μm |

The signed quaternion dot product is approximately **+1** in every sampled row. This explicitly checks the actor-visible quaternion sign; matching rotation matrices alone would not suffice. The original fingertip ordering, pad offsets and robot-base observation shift also agree.

## Actual table placement

Native Newton model readback in all 12 environments confirms:

- Robot base: `(0, 0.51, 0.20)` m relative to its environment origin.
- Table collision body: center `(0, 0, 0.38)` m, identity orientation.
- Actual collision-box half-extents: `(0.30, 0.20, 0.15)` m, identity local transform.
- Tabletop: `z=0.53` m, **0.33 m above the robot base**.
- Table near/far Y edges: `+0.20/-0.20` m, respectively **0.31/0.71 m forward of the robot base**.

This verifies the simulated geometry, not only the configured values. Evaluation configuration is assembled by `experiments/yibo_our_dr/evaluate_vbd.py::build_config`: typed VBD defaults, `isaacsimenvs/scripts/configs/task/Cloth.yaml`, the selected material YAML, checkpoint action/observation settings, and explicit evaluation overrides. Every raw material result stores the fully resolved configuration.

## What this does not prove

Correct transport into the actor does **not** mean the historical representation is a physically complete description of deformable cloth. It deliberately fits four inset moving-half corners to a rigid pose, reconstructs planar markers, reports full-sheet scales `(0.15, 0.15, 0.016)/0.04`, and preserves the historical goal-quaternion component relabelling. Those conventions match the checkpoint; physical fold scoring remains separate. In particular, a passing compatibility audit does not make the historical goal encoding geometrically correct.

This audit covers the clean Yibo evaluation pathway with observation noise/delay disabled and controlled states. It does not validate every randomization mode, critic field, long-horizon contact force, solver dynamic parity, or arbitrary deformation. The separate actuation audit establishes nonzero robot motion; the material sweep still shows poor transfer. No production perception fix was made because these checks did not identify a perception wiring error.

A separate HTML viewer quaternion-conversion issue was identified by the parent audit. That visualization does not feed the actor and was not used to compute the material evaluation scores.

## Reproduce

```bash
OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=2 \
  .venv/isaaclab3/bin/python experiments/yibo_our_dr/audit_vbd_observations.py
```

Full per-case results and raw model world/body labels: `debug_outputs/deformable_observation_audit/2026-11-16/audit.json`. The audit raises on failed tolerances and changes no production source.
