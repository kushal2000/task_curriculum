# Evaluating the Play prior on rigid cloth

The [rigid cloth viewer](../viser_rigidCloth_viewer.py) can run the bundled
[Flexiv/Wuji checkpoint](../../../../pretrained_policy/flexiv_wuji/model.pth) with its
[training configuration](../../../../pretrained_policy/flexiv_wuji/config.yaml):

```bash
.venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py \
  --policy --design single_hinge --robot_setup training
```

`--policy` enables inference; without it the viewer remains a cloth physics inspector.

### Yibo presets

```bash
python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py --preset yibo
python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py --preset yibo_our_dr
```

Both presets enable policy control and select
`pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth`, historical
HAND2_DRIVE dynamics with the real-limit URDF, and the table at **33 cm above the
robot base**. Table height stays fixed. They restore Yibo's inset planar marker
order, centroid and quaternion fitting, full-sheet scale features, and historical
goal-quaternion conversion. The latter deliberately reproduces a training bug:
at some sheet headings, the observed goal is vertical. Physical scoring always
uses the actual fold target.

| Preset | Cloth physics and DR | Observed thickness | Policy delay/noise DR |
| --- | --- | --- | --- |
| `yibo` | Historical gains, density, contact shells and friction | Historical 16 mm felt | Off |
| `yibo_our_dr` | Our 10–30 cm sizes, 2 mm plates, density/B/C/friction DR, zero armature | Actual geometry (2 mm) | On |

The recipes are [yibo.yaml](../../configs/runs/yibo.yaml) and
[yibo_our_dr.yaml](../../configs/runs/yibo_our_dr.yaml); the latter inherits
our canonical rigid-cloth recipe. Geometry sliders start at each range's midpoint.
**Sample cloth DR on reset** and **Policy delay and noise DR** are enabled according
to the preset; changing either requires **Apply geometry and restart**. With cloth
sampling enabled, each reset replaces fixed slider values with a new material draw.
Yibo friction is sampled at startup, while its contact shells are redrawn on reset.
Yibo's fixed per-hinge gains disable the B/C sliders.

These presets run our current design catalog and fold/release objective. They do
not recreate Yibo's old seven-design simulator, reset distribution or hinge layout.
The two training variants live under
[experiments/nov16/yibo_our_dr](../../../../experiments/nov16/yibo_our_dr/README.md).

Policy mode starts paused with a flat cloth; click **Run simulation** or **Single step**.
The default DR recipe is `scripts/configs/runs/rigid_body_cloth.yaml`; override it with
`--training_config PATH`. Geometry, density, hinge and friction controls take their
defaults/bounds from that recipe. Sliders choose fixed physics values for inspection;
the **Reset placement** selector controls manual versus sampled initial poses.
The robot and table remain static in preview mode. **Reset cloth** in policy mode resets
the robot, cloth, episode state and policy memory together. Coral points show the current policy keypoints; green points show their policy
targets, generated from the moving-half and goal poses using the checkpoint's
`reward.keypoint_scale` (1.5). Yellow lines connect each current point to its corresponding target.
Geometry/setup changes require **Apply geometry and restart**; gains and friction update live.

Static preview and policy mode both default to `--robot_setup training`: the
47.5 × 40 × 4 cm table centered at `(0,0.29,0.26)` m and the human-like-limit robot
based at `(0,0.8,0)` m. The setup selector applies to both modes. The setup dropdown selects **Training setup**
or **Real hand / cloth setup**. Use `--robot_setup real` to evaluate transfer to the identified
hand. The viewer uses nominal placements with randomization off; the ranges below describe
the saved training distribution. `--checkpoint`, `--policy_config`, and `--sapg_expl_coef` override the checkpoint, its
network configuration, and the SAPG conditioning coefficient. Different coefficients select
different learned behavior; the generic player's coefficient is not evidence of the best
coefficient for this checkpoint on cloth.

This is a **Play manipulation prior, evaluated zero-shot on cloth**, not a policy trained to
fold. Its saved configuration describes manipulation of rigid tool-shaped objects. Matching
the network interface makes inference valid; it does not establish folding competence.

## Robot and table differences

The bundled configuration identifies `play_scratch_flexiv_right_dr_cont_41213`, November 6,
2026, epoch 239585, exported November 15. It specifies
`flexiv_wuji_right_human_like` and the human-like-limit URDF. The cloth environment's default
is the identified real hand. The profiles differ in more than joint limits:

| Property | Training setup | Real hand / cloth setup |
| --- | --- | --- |
| Robot URDF | [Human-like limits](../../../../assets/urdf/flexiv_wuji/rizon4s_right_wuji_human_like_joint_limits.urdf) | [Real limits](../../../../assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf) |
| Arm limits | Original limits | Each lower/upper bound moved inward by 5° |
| Four non-thumb PIP joints | Approximately 120° maximum | 100° maximum |
| Hand dynamics | HAND2_DRIVE; zero joint friction; no passive springs | Identified armature, friction, passive damping and springs |
| Hand effort limits | 0.21–1.06 N m across joints | 0.1002–0.6611 N m across joints, from 1 A × identified torque constant |
| Table URDF | [table_narrow_flexiv](../../../../assets/urdf/table_narrow_flexiv.urdf) | [table_fold_pool](../../../../assets/urdf/table_fold_pool.urdf) |
| Table dimensions | 0.475 × 0.4 × 0.04 m | 0.6 × 0.4 × 0.3 m |
| Nominal tabletop above robot base | 0.28 m | 0.33 m |
| Nearest/farthest table edge ahead of base | 0.31 / 0.71 m | 0.31 / 0.71 m |

Joint names/order, fingertip order and offsets, palm offset, arm/hand PD gains, and home pose
agree between the [Play robot specification](../../../tasks/play/robots.py) and
[cloth robot specification](../../../tasks/cloth/backends/robot.py). Changing only the URDF does
not reproduce the training dynamics.

The saved training table center is `(0, 0.29, 0.26)` m, with uniform height noise ±0.01 m.
Its top is therefore 0.28 ±0.01 m. XY/yaw randomization is zero, scaling is fixed at one,
and there is one table variant. Historical source at `cdf26eb`,
`isaacsimenvs/tasks/play/utils/scene_utils.py`, places the robot base at `(0, 0.8, 0)`;
the table center relative to that base is `(0, -0.51, 0.26)`.

The current [rigid cloth configuration](../../../tasks/cloth/backends/rigid/config.py) overrides
the shared table asset with `table_fold_pool.urdf`. Its nominal center is `(0, 0, 0.38)`
and the robot base is `(0, 0.51, 0.20)`. The tabletop is 5 cm higher relative to the base
than in training. The saved checkpoint configuration does not contain `robot_root_pos`
or frozen URDF hashes: the historical base position is corroborated by local source,
not encoded in the network weights. Its old robot asset path was subsequently reorganized.

## Observation and action contract

Inspecting the checkpoint tensors confirms **134 actor observations and 27 actions**:
normalization statistics have width 134, the output layer has 27 rows, and the LSTM input
width is 166 = 134 observations + 32 SAPG embedding features. The saved YAML's nominal
`observation_space: 140` is stale; Play recomputes it from its fields and robot joint count.
Do not use the generic player's 140-observation/29-action defaults for this checkpoint.

The [cloth observation builder](../../../tasks/cloth/mdp/observations.py) preserves Play's field
layout, canonical joint order and xyzw quaternion convention. The object fields describe the moving-half pose and Play's ordered 3D keypoints,
with a fold/unfold target pose. Cloth dynamics still differ from rigid tool manipulation.

The [controller](../../../tasks/cloth/mdp/actions.py) runs the policy at 60 Hz over 120 Hz physics:
arm actions increment joint targets at scale 1.5, hand actions map to absolute joint targets,
and both use exponential smoothing with coefficient 0.1. Joint positions are normalized by
joint limits; previous targets remain radians. Limits therefore affect both inputs and outputs:
a PIP action of zero maps to roughly 60° with training limits but 50° with real limits.
A measured 90° normalizes to roughly 0.5 versus 0.8. The real setup measures transfer under
that changed interface and dynamics; it is not a reproduction of the training environment.

## Shared policy object representation

Every cloth policy observes the moving half as one object: its fitted center,
orientation and velocity, its dimensions `[length/2,width,thickness]`, and four
ordered keypoints selected by `obs.keypoint_mode`. The same definition is used during training and
inference, across rigid and deformable cloth.

The local keypoint pattern is `(+++),(++−),(−−+),(−−−)`: two diagonal XY
locations, each with a top/bottom thickness pair. Offsets are
`pattern * moving_half_dimensions * obs.keypoint_scale / 2`, transformed by the
moving-half pose. Target markers use the corresponding fold/unfold goal pose.
The folded goal includes a physical half turn, with the normal pointing down.
For a single rigid hinge the fitted pose is exact; for bending or deformable
halves, one frame is a least-squares approximation.

In `play` mode, `obs.keypoint_scale` defaults to 1.5. It expands virtual offsets
without changing physical geometry or object-size observations. The pretrained
viewer reads the value from its checkpoint. `reward.object_base_size=0.04` gives
`object_scales = [length/2,width,thickness]/0.04`. For a full `[0.2,0.2,0.004]` m
sheet, the policy sees `[2.5,5,0.1]`.

Choose policy inputs with `--keypoint_mode play` (default) or
`--keypoint_mode planar`. The **Robot → Policy keypoints** dropdown also changes
this setting; click **Apply geometry and restart** to apply it. Planar mode sends
the four actual moving-half landmarks in near-left, near-right, far-left,
far-right order, with physical fold/unfold targets and no keypoint expansion.
Rigid landmarks include the crease endpoints; VBD uses its existing grid
landmarks. The object pose and moving-half size observations stay the same.
This option alone does not reproduce Yibo's historical full-sheet scales,
inset grid landmarks, or quaternion relabelling bug.

The viewer's **Keypoint display** selector offers **Policy input** and
**Physical corners**. Coral is current, green is target; yellow lines link pairs.
Policy markers are the exact pre-noise features supplied to the network. Physical
corners remain the fold-scoring landmarks. Rigid physical corners attach directly
to moving plates, including the crease endpoints, independently of grid resolution.
Particle cloth retains its surface samples for physical task scoring.

The original Play control and numerical transport audit are recorded in the
[audit report](../../../../debug_outputs/policy_piping_audit/2026-11-16/README.md).
That audit identified the former planar-corner/stationary-orientation mismatch;
the shared object representation described here replaces that encoding. Policies
trained on the former encoding need retraining or their original environment to
reproduce their behavior.

## Reset placement and DR controls

The run recipe lists active training parameters under `env.rigid_cloth`,
`env.rigid_cloth.randomization`, `env.reset` and `env.domain_randomization`.
The viewer reads the same nominal values and bounds; its sliders select fixed
values and its reset controls can sample the training placement distribution.

All five symmetric designs are enabled in the training recipe. Collision thickness
is fixed at 2 mm and armature randomization is disabled. Areal density is
log-uniform over 0.02–0.5 kg/m². Shared sheet bending rigidity B [N m] and bending
viscosity C [N m s] are converted to each hinge using its crease length divided
by the transverse distance between neighboring plate centroids. B and C ranges
are provisional calibration bounds, not measured fabric coefficients.

Effective static contact friction is 0.1–0.8 for cloth/table and 0.3–1.2 for
cloth/robot. Dynamic friction is 0.6–1.0 times static. Both sides explicitly use
PhysX multiply combine: cloth coefficients are one, so the table and robot
coefficients are the actual interface values. Non-neighbor cloth/cloth friction
remains one; the default joint-adjacent collision exclusions are unchanged.

Fold entry pays a one-time 1,000 bonus. A further one-time 1,000 is paid when the
fold remains valid and every robot collider is at least 10 cm clear; that completes
the episode immediately. There is no hold timer. Clearance uses conservative
collision-geometry bounds across the base, arm, palm and fingers; it can delay a
release for rotated shapes, but does not award it while those bounds are too close.
The viewer displays this clearance. Imported guide-purpose colliders are included.

The reset controls expose manual X/Y/yaw and editable X/Y placement bounds plus a
yaw interval. Sampled mode applies them on manual and automatic resets. Placement
coordinates are relative to the active table center. The sampler accounts for the
whole sheet's yawed footprint and the active table, including the narrower Play
training table. It rejects impossible placements rather than intentionally starting
with unsupported overhang.

Default requested placement bounds are X `[-0.20,0.20]` m and Y `[-0.17,0.17]` m
relative to table center, with yaw `[0,360]` degrees. These bound the whole sheet,
not its center. They leave margin inside the training table; sampling further
accounts for size and yaw. Large sheets use headings conditioned on fitting;
smaller sheets retain the full angular range. The recipe and viewer use the same
values.
