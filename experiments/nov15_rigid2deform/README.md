# nov15_rigid2deform: the single-hinge rigid sheet env (Rigid2Deform, step 2)

Plan: `.claude/plans/pasted-content-id-e162-in-this-vectorized-wombat.md`. Goal of the method: train
folding on rigid-body approximations of cloth, transfer zero-shot to VBD cloth over many sizes and
materials. Generation 0 is one hinge at the horizontal fold line.

New env `Isaacsimenvs-RigidBodyCloth-Direct-v0` (`isaacsimenvs/tasks/cloth/rigid_body_env.py`,
`rigid_body_env_cfg.py`, `cfg/task/RigidBodyCloth.yaml`, experiment `experiments/configs/rigid_body_cloth.yaml`):

- **Designs** (`utils/sheet_designs.py`): 16 lengths x 16 widths (10-30 cm) x 16 thicknesses (2-20 mm)
  = 4096; env i -> design `perm[i % 4096]`, so each 4096-env SAPG block holds every design once.
- **Sheet**: two boxes + one revolute hinge on the top surface at the crease (a 180 deg fold stacks
  the + plate one thickness on the - plate = the fold target). Authored per env into the root layer
  (template in env 0, `Sdf.CopySpec` to the rest, dims set per env), native box colliders, one
  articulation over all envs. Play's offsets (contact 2 mm, rest 0); max depenetration 1 m/s.
- **Task**: fold only, `ClothTaskMixin` unchanged (targets scaled per env to length x width, lifted by
  the thickness; `object_scales` = [length, width, thickness] / 0.04).
- **Reset**: yaw uniform; centre uniform over every position that keeps the yawed sheet's bounding box
  inside the table region x [-0.29, 0.29], y [-0.18, 0.19] m (centred if it does not fit; centred
  everywhere under the eval protocol's zero position noise).
- **DR every reset**: hinge damping 0.05 x logU[0.5, 3], stiffness U[0, 5e-4]; friction x[0.7, 1.5]
  plates (1.0), x[0.7, 1.3] robot (0.5), x[0.7, 1.3] table (1.0), 16 levels each; Play's obs/action DR.

## Startup proofs (`probe_env.py`, 256 envs, local RTX 6000 Ada)

All pass in 3 runs (0-6 of 256 envs skipped for a fingertip within 3 cm of the sheet):
authored collider dims exact; sheets at rest on their tables within 0.06 mm, hinges within 0.05 deg;
+90 deg lifts the far corners; hinge at 180 deg: fold_error <= 0.1 mm and `is_folded` in every env,
still after 30 physics steps; per-reset DR changes the reset envs' damping, stiffness and the three
frictions and leaves the others bit-identical.

Found on the way:
- Play's +-10 cm reset noise plus the pool's reach shift put large sheets off the 0.6 x 0.4 m table
  (fell, or hung with the hinge at 15-27 deg) -> size-aware uniform placement.
- A plate spawned under a fingertip (Play's random reset arm pose rests it ~1.4 cm above the table)
  was shot ~1 m into the air by PhysX depenetration at Play's 1000 m/s cap -> 1 m/s.

## Smoke training (local, 1 GPU, 6144 envs, 3 epochs, viewer every 100 steps)

`train.py --experiment rigid_body_cloth env.scene.num_envs=6144 agent.params.config.expl_coef_block_size=1024
... max_epochs=3`: trains (8.6 / 10.5 / 6.5 s per epoch, 33-75k env steps/s), checkpoint written. The
viewer HTML (`debug_outputs/train_logs/nov/2026-11-15_smoke_rigid_body_local_626022/interactive_viewer/`)
draws env 0's two plates (8 vertices per frame, the + plate as the moving part) on the table.

Setup at 6144 envs: base scene 36 s, Play-tool neutralising + goal marker reshape 78 s (inherited from
the pool, the next thing to cut), sheet authoring 3.2 s, simulator start / PhysX parse 154 s
(~25 ms/env, as before).

## Overnight runs (2 GPUs x 12288 envs each)

`submit_prior_2gpu.sh` (Play prior, weights) and `submit_scratch_2gpu.sh`; wandb
`flexiv_wuji_cleanup` / `nov15_rigid2deform`. Submitted 2026-11-15 ~22:00 from aa78615: 675228
(`rigid_body_prior_2gpu`), 675229 (`rigid_body_scratch_2gpu`).
