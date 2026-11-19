# Friction prior driven by points (PointNet adapter)

The friction prior (`flat7RW_fricition.pth`) sees 134 inputs: 103 robot values and 31 cloth values
(object_rot, keypoints_rel_palm, keypoints_rel_goal, object_scales). The adapter replaces the 31
cloth values with a PointNet encoder over the 64 top-face points, their 64 goals and the palm;
every other prior weight is unchanged (`scripts/training/pointnet.py`, `pointnet_adapter_actor_critic`).

1. `collect_train.py` (job 789718): 1.2 M samples from the prior's own rollouts (4096 envs,
   stochastic, 15x15 cm single_hinge, 2 mm, nominal). Encoder trained against the prior's LSTM input
   projection, ||W (f_hat - f)||^2, held out by env:
   - val normalised MSE 0.012, input-embedding error 8 % (median)
   - keypoints 17 mm median, object_rot 3.5 deg, scales 1.4e-3 -- about the prior's own observation
     noise (1 cm per axis, 5 deg, up to 10 steps of delay), which the targets carry and no encoder
     can predict; the encoder outputs the clean values.
2. `assemble.py`: prior + encoder -> `encoder_15cm_2mm/adapter_prior.pth` (not committed; rebuild
   with the command in its docstring). Critic: prior's, with zero-weight columns for the two point
   distances.
3. Same recipe, 1024 episodes each, deterministic:

| policy | cloth input | fold success |
|---|---|---|
| A prior | true state, with its training noise (as trained) | 50.9 +/- 1.6 % |
| **B prior + adapter** | **64 points + goals only** | **53.6 +/- 1.6 %** |
| C prior | true state, noise off | 60.3 +/- 1.5 % |

Release success is 0 % for all three: the prior never learned to release.

The adapter keeps the prior's behaviour (B within 2 sigma of A, slightly above it, as its inputs carry
no injected noise). The gap to C is the encoder's remaining error. Fine-tuning recipe:
`adapter_15cm_2mm.yaml` (same env and point reward as the from-scratch run 789626).
