# G1 + Wuji Hand 2: trajectory replay

Replay joint trajectories on the Unitree G1 humanoid with two Wuji Hand 2 hands, in MuJoCo or
Isaac Sim. The robot models come from
[wuji-hand-teleop](https://github.com/MSC-Wuji-Teleop/wuji-hand-teleop)
(`src/g1_wuji2_description`).

## Setup

```bash
scripts/g1_wuji/fetch_assets.sh
```

This fetches the model package at a pinned upstream commit into `assets/g1_wuji2_description/`
(gitignored, ~35 MB): MJCF with a fixed or floating base, a URDF mirror, and all meshes, for the
29-DoF G1 (the default) and the 23-DoF G1. To move to a newer upstream, change `SHA` in the
script and rerun it.

Nothing new needs installing. MuJoCo replay runs in `.venv_isaaclab3` (mujoco 3.11). Isaac Sim
replay runs in `.venv_isaacsim` (Isaac Sim 5.1 + Isaac Lab 2.3).

## Trajectory format

A `.npz` file. `g1_wuji.save_trajectory` writes it:

| key | shape | |
|---|---|---|
| `joint_names` | `(J,)` str | any subset of the model's joints, in any order |
| `q` | `(T, J)` float | joint positions, radians |
| `fps` | scalar | frame rate |
| `root_pos` | `(T, 3)` | optional: pelvis world position. Giving it selects the floating-base model |
| `root_quat` | `(T, 4)` | optional: pelvis world orientation, **w, x, y, z** |

```python
import sys

sys.path.insert(0, "scripts/g1_wuji")
import g1_wuji

g1_wuji.save_trajectory("traj.npz", joint_names, q, fps=50)
```

Joints the file leaves out are held at the model's `stand` keyframe: shoulder pitch 0.2,
shoulder roll ±0.2, elbows 1.28, everything else 0. `g1_wuji.mjcf_joint_names(29)` lists all
69 joint names: legs 12, waist 3, arms 7 + 7, then 20 per hand, named
`{left,right}_wuji_{l,r}_<finger>_<joint>`. Upstream's
[wujihand_urdf README](https://github.com/MSC-Wuji-Teleop/wuji-hand-teleop/blob/main/src/wujihand_urdf/README.md)
documents the hand joint order and conventions.

A wuji-hand-teleop clip directory (`clip.json` + `arm_q.npz` + `hand_q20.npz`, as in upstream's
`clips/`) also loads directly.

## MuJoCo

```bash
# interactive viewer (needs a display)
.venv_isaaclab3/bin/python scripts/g1_wuji/replay_mujoco.py traj.npz

# headless mp4: MUJOCO_GL=egl on a GPU node, MUJOCO_GL=osmesa on the login node (slower)
MUJOCO_GL=egl .venv_isaaclab3/bin/python scripts/g1_wuji/replay_mujoco.py traj.npz --video out.mp4
```

By default the replay is **kinematic**: each frame writes `qpos` directly, so you see exactly
the trajectory, including any penetration or out-of-limit poses.

`--physics` instead sends each frame to the model's position servos and simulates. The servos
use the upstream gains: kp 500 on the G1, the vendor's gains on the hands. The run then prints
the tracking RMSE and the worst-tracked joints. Use it to check whether a trajectory is
physically feasible: a finger pressing into the other hand, for example, shows up as a large
error on that finger. `--physics` only works with a fixed base, because the G1's balance
controller is not modelled.

Every run ends with a contact report, split into hand-hand, hand-body, body-body and
within-hand contacts; feet on the floor are left out. Each hand has 27 collision meshes, and
all of them can collide with the other hand, with the body, and with the hand's own
non-adjacent links. What the report measures depends on the mode:

- **Kinematic:** contacts are detected but not resolved, so the hands pass through each other.
  The report gives the deepest penetration, which is how far the trajectory itself drives
  geometry into geometry.
- **`--physics`:** the solver pushes the geometry apart. The report gives the residual
  penetration and the peak contact force.

Adjacent fingertips touch by a few millimetres even in normal poses, so small within-hand
numbers are expected.

```
contact hand-hand: 69/260 frames, deepest 32.9 mm (left_wuji_l_ring_finger_proximal_abd / right_wuji_r_wrist)
contact hand-body: none
```

Other flags: `--variant 23`, `--speed 0.5`, `--loop`, `--width/--height`.

## Isaac Sim

```bash
# GUI (needs a display)
.venv_isaacsim/bin/python scripts/g1_wuji/replay_isaacsim.py traj.npz

# headless mp4, on a GPU node
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/g1_wuji/replay_isaacsim.py traj.npz \
    --headless --video out.mp4
```

On the first run the URDF is converted to USD, cached under
`assets/g1_wuji2_description/usd/`, and reused after that. The replay is kinematic: joint
positions and the root pose are written to PhysX and rendered, and physics is never stepped.
Collisions therefore play no part: the hands pass through each other, and nothing reports it.
Self-collision is also switched off in the conversion. For physics-level tracking and for
contact checks, use MuJoCo.

The bos14 login node has no GPU, so submit the job through SLURM from the repo root:

```bash
sbatch scripts/cluster/bos14_g1_wuji_replay_isaacsim.sh traj.npz --video outputs/out.mp4
```

The compute nodes do not have `libGLU.so.1`. Without it, Isaac Sim's renderer fails to start
and the video comes out empty. The launcher works around this by using a copy of the library,
which you make once from the login node:

```bash
mkdir -p .venv_isaacsim/compat-libs
cp /usr/lib/x86_64-linux-gnu/libGLU.so.1.3.1 .venv_isaacsim/compat-libs/libGLU.so.1
```

Inputs and outputs must be on the shared filesystem (for example under `outputs/`), because
`/tmp` is not shared with the compute nodes.
