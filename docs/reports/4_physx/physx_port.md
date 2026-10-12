# The rigid fold tasks on PhysX: 10-20x faster, and both stages now transfer to the cloth

**What.** `Isaacsimenvs-RigidClothPhysx-Direct-v0` (stage 1, one chain) and
`Isaacsimenvs-FoldPoolPhysx-Direct-v0` (stage 2, the shape pool) run the rigid fold tasks on PhysX
(Isaac Sim 5.1, `.venv_isaacsim`) instead of Newton/MJWarp. The task -- reward, fold criterion,
observations, the chain and pool adapters, every startup proof -- is the same code as the Newton
envs (`cloth_task.py`, `rigid_cloth_task.py`, `fold_pool_task.py`); `physx_envs.py` only expresses
the manipuland's physics in PhysX terms. Policies are still evaluated on Newton's VBD cloth.

**Result (2026-10-06).**

* **Stage 1 trains on PhysX and transfers like Newton-trained stage 1**: from the generic prior,
  box3-mid, 1536 envs, 1e8 steps, it folds the VBD cloth 55% pinned / 45% randomised (Newton:
  52% / 43%). What blocked it was a PhysX articulation artefact, not contact softness: with the
  hinge armature at 1e-5 a fingertip pressing one plate makes the OTHER plate jump 5-6 cm, and
  policies learned to fold with that jump (3-6% on the cloth). `physx_chain.hinge_armature = 1e-3`
  (now the PhysX default) removes it.
* **Every rigid stage on PhysX beats the Newton pipeline on the cloth.** Stage 2 on PhysX from the
  PhysX stage-1 policy folds the VBD cloth 65% pinned / 47% randomised at the 4 cm cutoff and 37% /
  21% at 1.5 cm; the Newton pipeline (Newton stage 1 -> Newton stage 2): 27% / 33% and 4% / 6%. Stage 2
  takes 45 min per seed instead of ~5.5 h; stage 1 ~90 min at 12288 envs (4e8 steps, 63% / 44% on
  the cloth) or ~50 min at 1536 (1e8, 55% / 45%), instead of 3-5 h.
* **Speed.** One RTX PRO 6000: chain 41k env-steps/s end to end at 1536 envs, ~98k at 12288, ~106k
  at 24576 (simulation alone 53k / 179k / 222k); pool 30k / 62k / 97k at 1536 / 6144 / 12288
  (Newton: ~5k, saturated). 24576 envs build and run now, but buy only ~8% end to end at the
  24576 minibatch -- the PPO update grows with them.

## Stage 1 on the cloth

Generic prior, box3-mid, 1e8 env-steps unless noted, `rigid_cloth.randomization` off ("fix"), 3
seeds; protocol of `bos14_rigid_cloth_eval_rc2.sh`. Held folds out of 64 per seed.

| arm | hinge armature | own Newton chain pinned / rand | VBD cloth pinned / rand |
|---|---|---|---|
| Newton rc2 fix, 1536 envs | 1e-5 | 64, 64, 64 / 59, 59, 59 | 31, 29, 40 (52%) / 36, 19, 27 (43%) |
| **PhysX, 1536 envs (157449)** | **1e-3** | 64, 62, 64 / 54, 48, 55 | **42, 23, 40 (55%) / 31, 30, 25 (45%)** |
| PhysX, 1536 envs (157648) | 3e-3 | 64, 5, 64 / 55, 33, 55 | 13, 40, 4 (30%) / 27, 33, 25 (44%) |
| PhysX, 1536 envs (157451) | 3e-4 | 62, 22, 58 / 51, 36, 44 | 25, 26, 12 (33%) / 20, 29, 10 (31%) |
| PhysX, 1536 envs (157222) | 1e-5 | 2, 0, 4 / 1, 3, 1 | 3, 8, 1 (6%) / 2, 4, 1 (4%) |
| PhysX, 12288 envs, 1e8 (157450) | 1e-3 | 60, 14, 8 / 14, 19, 18 | 7, 0, 32 (20%) / 14, 9, 10 (17%) |
| **PhysX, 12288 envs, 4e8 (157494)** | **1e-3** | 64, 64, 64 / 48, 54, 49 | **55, 29, 37 (63%) / 24, 22, 39 (44%)** |
| PhysX, 12288 envs, 1e8 (157223) | 1e-5 | 0, 0, 0 / 1, 1, 0 | 1, 3, 2 (3%) / 1, 0, 0 (1%) |

Cloth transfer peaks at 1e-3: 3e-4 still leaks (33% / 31%), 3e-3 over-weights the hinge (30% pinned).

Training curves agree with this: at 1e-3 the share of episodes ending folded rises 32-62% (2.5e7
steps) -> 94% (5e7) -> 97% (1e8), as on Newton (32% -> 87% -> 95%); at 1e-5 it was already 96% at
2.5e7 -- the jump made folding the rigid chain easy. At 12288 envs 1e8 steps are only 255 policy
updates, and the run was still at ~50%: more envs need more steps.

## The stage-1 artefact: a pressed plate kicks the other one up

Same open-loop hand motion (recorded from the 1e-5 PhysX policy) replayed in three simulators
(`bos14_contact_fingerprint.sh`). Around step 27 its thumb presses the -x plate near its outer edge,
z 26 -> 21 mm; no finger touches the +x plate. Row heights above the table [mm], +x -> -x:

| step | PhysX chain, armature 1e-5 | Newton chain | VBD cloth |
|---|---|---|---|
| 26 | 8 8 8 8 8 8 8 | 8 8 8 8 8 8 8 | flat |
| 28 | **36 27 18** 10 9 8 8 | 13 11 10 9 8 7 6 | flat |
| 30 | **59 43 26** 11 9 9 8 | 16 13 11 8 5 3 0 | flat |

The pressed plate stays flat; the untouched one swings up about the hinge. What removes it
(`flick_probe_summary.py`; height of the untouched plate's outer row at steps 26 / 28 / 30 / 34):

| PhysX setting | mm |
|---|---|
| base (armature 1e-5) | 8 / 36 / 58 / 67 |
| hinge armature 1e-4 / 3e-4 / 1e-3 | 8 / 15 / 18 / 11 -- 8 / 13 / 13 / 8 -- 8 / 12 / 11 / 8 |
| slat max depenetration velocity 0.5 m/s | 8 / 16 / 22 / 18 |
| 32 position iterations | 12 / 57 / 87 / 86 |
| 4 velocity iterations | 8 / 34 / 56 / 56 |
| hinge damping x10 | 9 / 36 / 59 / 70 |

1e-5 kg m^2 is about the plate's own inertia about its hinge (20 g, 5 cm: ~8e-6); the squeeze between
fingertip and table leaks into that nearly massless hinge DOF. Passive physics at 1e-3 is unchanged
(rest 8.0 mm, mu 0.38, same release and half fold). Screened before training with three fixed policies
on the PhysX chain (`bos14_physx_chain_screen.sh`; randomised start, held / 64): the 1e-5 PhysX policy
drops 62 -> 0 (its trick is gone), the Newton chain policy rises 43 -> 51, the cloth-trained reference
24 vs 32. Compliant (spring) slat contacts, tried first, made every policy worse.

## Stage 2 on the cloth

Pool, 10-20 cm sheets, 1e8 env-steps, 3 seeds; VBD cloth, 32 envs x 2 eval seeds per cell
(`bos14_physx_eval_on_newton.sh KIND=pool`, protocol of `bos14_fold_pool_eval_cloth.sh`). Held folds
out of 64 per seed.

| arm | prior | wall / seed | 4 cm pinned | 4 cm rand | 1.5 cm pinned | 1.5 cm rand |
|---|---|---|---|---|---|---|
| Newton, 1536 envs (156746) | Newton box3 | ~5.5 h | 12, 21, 18 (27%) | 22, 19, 22 (33%) | 2, 3, 3 (4%) | 1, 6, 5 (6%) |
| PhysX, 6144, arm. 1e-5 (157274) | Newton box3 | 45 min | 12, 39, 25 (40%) | 11, 17, 11 (20%) | 8, 7, 4 (10%) | 1, 2, 1 (2%) |
| PhysX, 1536, arm. 1e-5 (157273) | Newton box3 | ~1.2 h | 15, 18, 32 (34%) | 8, 12, 12 (17%) | 3, 7, 6 (8%) | 4, 1, 3 (4%) |
| PhysX, 6144, arm. 1e-3 (157461/157626) | Newton box3 | 45 min | 16, 0, 34 (26%) | 23, 2, 22 (24%) | 4, 0, 4 (4%) | 4, 0, 10 (7%) |
| **PhysX, 6144, arm. 1e-3 (157622)** | **PhysX stage 1 (157449 s1)** | 45 min | **56, 53, 15 (65%)** | **40, 21, 30 (47%)** | **49, 21, 1 (37%)** | **14, 15, 12 (21%)** |

The PhysX prior is seed 1 of the 1536-env armature-1e-3 stage-1 run, picked as the first seed (not by
its cloth score). Its stage-2 training learns on every seed (episodes ending folded 79-83% at 1e8,
Newton stage 2: 56% -> 82%), while the Newton prior under armature 1e-3 starts near 0 and one seed
never learns (1%): a prior trained in the simulator it continues in is worth more than the simulator
it was trained in. The 1.5 cm column is where the pipelines differ most (37% vs 4% pinned); seed 3
of the PhysX pipeline is the weak one (15 / 1).

## What matches and what does not

**Matches** (all measured):

* Observation: the 140-dim policy input at the pinned start agrees to ~1e-3 in every block, chain
  and pool (`cloth_obs_parity.py`); the two Newton convention quirks every policy was trained under
  (reset-pose reorder, relabelled goal quaternion) are reproduced. Pool `object_scales` differs by
  <=0.009 from the 1 mm size quantisation.
* Passive physics (`rigid_physics_compare.py`): rest height 7.9 vs 8.0 mm above the table; slide
  friction 0.37 vs 0.38 (0.43-0.49 on PhysX for the 8-plate shapes); hinge release identical;
  driven half folds reach <0.3 mm with seams closed to 0.008 mm on both.
* Contact friction per pair (`physx_contact_probe.py`): slat-table 0.354, slat-fingertip 0.612,
  fingertip-table 0.612, as Newton's priority rule.

**Differs:** contact stiffness. Uniform downward force on the sheet (0.5 / 2 / 8 N): Newton sinks
0.02 / 0.12 / 0.38 mm (~20 N/mm, MJWarp's soft contacts), PhysX 0.00 / 0.01 / 0.05 mm (rigid). Under
a fingertip Newton tilts the whole sheet (pressed edge into the table, far edge up, table above);
neither rigid engine reproduces the cloth's local lift. Stage-1 transfer is equal anyway.

## Scale: 24576 envs and startup

Three separate problems made 24576 envs fail; all fixed or handled:

* **Scene construction was O(N^2)**: per-env Usd/Sdf edits on Kit's root layer (material binding,
  tool neutralising, goal marker, the pool's per-env prims) cost more per edit as the layer grew --
  binding took 13 s at 3072 envs, 46-55 s at 6144 (plain usd-core: 0.2 / 0.5 s); 24576 never finished
  building, or took 19-35 min. Kit's USD listeners are NOT the cause (A/B, 3 runs each). The edits are
  now written into a detached anonymous layer and inserted once as a sublayer
  (`utils/usd_bulk.bulk_edit_layer`; the loops take 1.6 s at 12288). Chain 24576: 446-462 s to build.
* **One pool collider fell back to CPU**: box2-mid at exactly 0.200 m (a 200 x 100 x 2 mm plate) fails
  PhysX GPU convex cooking ("failed to cook GPU-compatible mesh"; 0.199 m and 0.150 m cook). Rounded
  sizes are now kept at <= 0.199 m (134 of 24576 envs move by 1 mm).
* **PhysX's GPU pipeline sometimes dies at simulator start** ("Fetching GPU Narrowphase failed! 700",
  with "PxgCudaMemoryAllocator" warnings) at modest memory (2-8 GB GPU) and then HANGS until the time
  limit. Intermittent: 1 of 6 startups at 6144 envs, 2 of 4 at 24576, never tied to a node. The
  launcher now watches each attempt's log for those messages, kills it and retries (`PX_TRIES`, 3).

**Throughput saturates around 20k envs** (chain, fps step / fps total): 12288 179k / 98k, 18432
210k / 101-107k, 21504 220k / 104-108k, 24576 222k / 106-111k. The pool at 24576 steps at 90k, no
more than at 12288 (89-97k, GPU 97% busy).

**Checkpointing dominated fast runs.** rl_games saves the "best" checkpoint every time the reward sets a
new high -- on most epochs of a run that is still improving -- at ~4 s each on the shared filesystem,
GPU idle meanwhile: the 1536-env chain took 5 s per epoch for 1.2 s of training, the 12288-env one
7.5 s for 3.6 s. `best_save_min_interval_s` (ClothSAPG.yaml, 0 = off; the PhysX launcher sets 60)
caps that; the evaluated checkpoint is then at most 60 s stale.

## Port notes (traps hit, all fixed)

* A PhysX tensor-API material write (`set_material_properties`) creates materials with the default
  `average` combine mode -- a `min` slat rewritten to its own value went 0.354 -> 0.427. Friction is
  USD-authored (slat `min`, fingertip `max`); the Play task's tensor material pass is off; friction
  DR is refused.
* PhysX caps a scene at 64K materials, and a material inside the referenced asset is copied per env
  (12288 pool envs crossed it; the friction read-back caught it). One scene-level material each,
  bound per env.
* PhysX GPU contact buffers are sized for the Play task's one tool; the pool overflowed the patch
  buffer at 12288 envs (CUDA illegal access). Buffers now grow with `num_envs` (`physx_chain.gpu_*`).
* Isaac Sim 5.1's URDF importer instances collision geometry even with `make_instanceable=False`;
  de-instanced before editing.
* Rest offset (= Newton's margin, +7.000 mm measured); contact offset fixed above the margin-DR
  range because PhysX rejects rest >= contact.
* Newton checkpoints pickle `numpy._core` (numpy 2); `.venv_isaacsim` has numpy 1.26 -- aliased in
  `rl_games/algos_torch/torch_ext.safe_load`.
* `OMNI_KIT_ACCEPT_EULA=YES` is required in batch jobs.
* Hydra rejects an int for a float field (`...stiffness=100000` -> pass `100000.0`).

## Reproduce

    # stage 1 on PhysX (hinge armature 1e-3 is the default now)
    TASK=chain ENVS=1536 RUN_TAG=<tag> sbatch --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
    KIND=chain TRAIN_JOB=<id> RUN_TAG=<tag> sbatch --array=0-11 scripts/cluster/bos14_physx_eval_on_newton.sh
    # stage 2 on PhysX from a stage-1 policy
    TASK=pool ENVS=6144 PRIOR=<stage-1>.pth RUN_TAG=<tag> sbatch --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
    KIND=pool TRAIN_JOB=<id> RUN_TAG=<tag> sbatch --array=0-11 scripts/cluster/bos14_physx_eval_on_newton.sh
    python3 scripts/analysis/physx_eval_summary.py px_<tag> pxpool_<tag>
    # physics screening without training
    TAG=<name> EXTRA_ARGS="env.physx_chain.hinge_armature=0.001" sbatch --array=0-5 scripts/cluster/bos14_physx_chain_screen.sh
    python3 scripts/analysis/physx_chain_screen_summary.py
    OUT_TAG=_<name> STAGE=replay EXTRA_ARGS="..." sbatch --array=5 scripts/cluster/bos14_contact_fingerprint.sh
    .venv_isaaclab3/bin/python scripts/analysis/flick_probe_summary.py docs/results/fingerprint_physx_on_physx_*.pt
    # validation, scene-build profile
    sbatch --array=0-11 scripts/cluster/bos14_physx_cloth_validate.sh
    ENVS=6144 sbatch scripts/cluster/bos14_physx_build_profile.sh
