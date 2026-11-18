# nov15_setup_time: where the rigid-cloth env spends its setup time

A 24576-env `rigid_cloth_flexiv_wuji_full_dr` run takes ~40 min before its first epoch. Goal: find the
expensive phases, then test whether authoring the pool's prims directly into each env is faster than
referencing one USD file per sheet size.

Historical instrumentation (removed during the compact-layout cleanup):
`isaacsimenvs/utils/setup_timer.py` (`stamp()`), called at the phase boundaries of
`physx_envs.py` (`_setup_scene`, `_pool_spawn`) and after `gym.make` in `isaacsimenvs/scripts/training/train.py`.
Log lines: `[setup_t] +<since start>s (+<since previous>s) <phase>`.

Runs: local, one RTX 6000 Ada (portal-compute-02, interactive job), `train.py ... max_epochs=1`, the
rigid-cloth config, no viewer, no wandb. Logs in the session scratchpad (`setup_*.log`).

## Baseline: reference authoring (today's code)

| Phase | 512 envs | 2048 | 4096 | scales with envs |
|---|---|---|---|---|
| base scene (robot, table, object) | 8.6 s | 15.1 | 24.9 | yes |
| pool preamble (first shape's interval minus its USD files) | ~4.5 | ~17 | ~44 | yes |
| sheet-size USD files, 7 shapes (<= 100 sizes each) | ~15 | ~30 | ~32 | no (capped) |
| per-env prim specs, 7 shapes | 0.07 | 0.14 | 0.28 | negligible |
| layer insertion, 7 shapes | 15 | 55 | **101** | yes, ~25 ms/env |
| contact materials | 2.3 | 7.2 | 15.3 | yes |
| simulator start / PhysX parse | 16.4 | 57 | **108.5** | yes, ~27 ms/env |
| **gym.make total** | **63 s** | **185 s** | **329 s** | ~77 ms/env -> ~32 min at 24576 |

Writing the specs is free; what costs is Kit taking the per-env prims onto the stage (the layer
insertion) and PhysX parsing them at simulator start.

## Test 1: one layer for all 7 shapes instead of one per shape (2048 envs)

`ISAACSIMENVS_POOL_ONE_LAYER=1` prototype (reverted): every shape's specs in one anonymous layer,
inserted once before the articulations are built.

| at 2048 envs | 7 layers (baseline) | 1 layer |
|---|---|---|
| pool phase (base done -> all shapes on stage) | 105 s | 101 s |
| of which layer insertion | ~55 s | 8 s |
| PhysX parse | 57 s | 56 s |
| gym.make total | 185 s | 182 s |

**No gain.** The insertion time moved into the next shape's USD-file phase instead of disappearing:
the cost is per env prim landing on the stage (~50 ms/env for the pool), not the number of stage
edits.

## Test 2: direct (inline) authoring (2048 envs)

`ISAACSIMENVS_POOL_INLINE=1` prototype (reverted): each env's prim gets a full copy of its sheet-size
USD's prims (`Sdf.CopySpec` into the bulk layer) instead of a reference to the file. Same content,
no external references. A check confirmed every copied joint's body0/body1 point into its own env.

| at 2048 envs | reference (baseline) | inline |
|---|---|---|
| per-env authoring, 7 shapes | 0.14 s | ~2.5 s |
| layer insertion, 7 shapes | ~55 s | ~56 s |
| PhysX parse | 57 s | 56 s |
| gym.make total | 185 s | 186.5 s |

**No gain.** Resolving references is not the cost: Kit spends ~27 ms per env tree taking the prims
onto the stage however they are authored, and PhysX then parses every env separately.

## Test 3: author into the ROOT layer inside one Sdf.ChangeBlock (now the default)

The gen_mechanics `assembly.py` pattern (24576 hands referenced in 32.5 s): no sublayer at all. The
per-env reference prims are written straight into the stage's root layer, each shape's specs inside
one `Sdf.ChangeBlock`, so Kit gets one batched change notice instead of a full-stage
recompose on a sublayer insertion. `physx_envs.py:_pool_spawn`.

| | sublayer (baseline) | root layer + ChangeBlock |
|---|---|---|
| gym.make, 512 envs | 62.5 s | 48.3 s |
| gym.make, 2048 envs | 185 s | **133 s (-28%)** |

**Gain.** Equivalence: the composed pool prims of the first 64 envs (1141 prims, every attribute and
relationship) hash identically on both paths (sha256 `c1ca73b6...`). Tests: 214 passed / 34 skipped
(`.venv/isaacsim`), 289 / 1 (`.venv/isaaclab3`). At 24576 envs this should save ~10 min of setup.

Negatives tried alongside (reverted): pausing USD change tracking during the pool build (180 s at
2048) and hiding the plates until the build ends (185.5 s): neither changes the cost.

## What is left (after Test 3)

The per-env cost is in Kit and PhysX, not in how the USD is authored:
- Kit's per-prim handling as the layer lands (USD-notice listeners, Fabric population). Candidates:
  suspending them during the pool build. (The material-binding fix in `utils/usd_bulk.py` found
  disabling notice listeners made no difference there.)
- The PhysX parse (~27 ms/env). PhysX replication (`replicate_physics`) clones one env's physics
  instead of parsing each, but needs identical envs; the pool's per-env sheet sizes rule it out
  unless envs share a small set of sizes per shape.
