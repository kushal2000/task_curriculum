# Calibrating the left Wuji hand (torque constants and firmware gain law)

`robot.py` turns the policy's hand gains (sim, N·m/rad) into the firmware's (A/rad) per joint:

```
kp_A = kp_sim / (hand_kp_scale × Kt)        kd_A = kd_sim / (hand_kd_scale × Kt)
```

- **Kt:** the joint's torque constant (N·m/A), set in `robot.hand_kt`.
- **`hand_kp_scale`, `hand_kd_scale`:** the firmware applies about 1.95× its documented PD law.

The values in `config.yaml` come from the right hand, `WH2KA01260825022` (firmware 2.10.0, fingers-up and palm-down
runs, `--fit-gear`). The left hand uses them until it is measured, and `robot.py` prints a note saying so. A Kt
that is off by x % makes that joint x % stiffer or softer than in training.

This calibration measures all three on the left hand: about 15 minutes of hand motion plus the fit.

**aiowuji's own documentation** (submodule `third_party/aiowuji`, commit f05afcc):

- `README.md`:
  - "Recording" (data collection);
  - "Hand model";
  - "Simulation and system identification" (the fit, the firmware law, why Kt needs two mounts).
- `scripts/sysid_ref_collect.py`: its docstring describes the trajectories, the safety margins and the output files.
- `scripts/sysid_fit.py`: its docstring describes the model, what is fitted, the mounts and `params.json`.
- `--help` on either script lists every option.

## 0. Before you start

- **Arm:** the arm is not needed. Only the hand is used, on USB.
- **Mount:** the hand must be held rigidly in two different orientations, one per recording.
  - It can stay on the arm's flange if the arm is posed for each orientation. In that case keep the arm enabled
    and in a fixed pose, or switched off with its brakes holding, and keep the e-stop in reach.
  - Otherwise use a fixture.
- **Mounts the fit knows:**
  - `fingers-up` / `fingers-down`: fingers pointing up / down;
  - `palm-up` / `palm-down`: palm facing up / down, fingers horizontal.
  - Any other orientation can be given as `--gravity X Y Z`, the gravity vector in the hand's wrist frame (m/s²).
- **Which two mounts:** the right hand used `fingers-up` and `palm-down`. Use the same two, so the left and right
  values are directly comparable.
- **Clearance:** nothing may touch the fingers. The fit assumes no contact. The trajectories keep 5 mm (`--margin`)
  between the fingers and the palm; the other fingers spread away from the one being moved, and the thumb stays home.
- **Effort limit:** the default is 0.5 A (`--effort-limit`, max 1.5 A). Keep the default.
- **Firmware:** the gain law (≈1.95×) was measured on 2.10.0. Note the left hand's firmware version; the fit
  re-measures the law in any case.

```bash
cd third_party/aiowuji
uv pip install --python ../../.venv_deploy/bin/python -e ".[sysid,fit]"   # MuJoCo + tqdm; mjbatch + matplotlib
PY=../../.venv_deploy/bin/python
../../.venv_deploy/bin/aiowuji scan              # the serial number
../../.venv_deploy/bin/aiowuji status <LEFT_SN>  # Side: must say left; note Firmware
H=<LEFT_SN>
```

- **mjbatch's MuJoCo:** `mjbatch` pins its own MuJoCo version. If that conflicts with the deploy venv, put the fit
  (step 3) in its own venv: `uv venv .venv_fit && uv pip install --python .venv_fit/bin/python -e ".[fit]"`. The
  fit doesn't need the hand.

## 1. Plan, without moving the hand

```bash
$PY scripts/sysid_ref_collect.py $H --handedness left --plan
```

- **`--handedness left` is required.** The default is `right`, and then every finger's collision-free path is planned
  against the right-hand model.
- **What `--plan` prints:** each finger's trials, its amplitudes after the collision limits, and the total time
  (about 6 minutes).
- **Gain settings:** the defaults are kp:kd = 1.5:0.03, 3.0:0.05 and 6.0:0.1 (A/rad).

## 2. Record both mounts

Mount 1 (fingers up):

```bash
$PY scripts/sysid_ref_collect.py $H --handedness left        # -> logs/sysid-<UTC time 1>/
```

Then remount the hand in mount 2 (palm down) and run the same command again:

```bash
$PY scripts/sysid_ref_collect.py $H --handedness left        # -> logs/sysid-<UTC time 2>/
```

- **What each run does:** for each finger and each gain setting, the finger's four joints follow one 21.6 s trajectory
  together: steps, then slow sines, then a multisine (at most 2 rad/s). Before every trial the hand goes home and
  then to the trial's start pose with stiff gains.
- **Stopped part-way:** `--resume` continues the newest run (or `--resume logs/sysid-...`). It first checks that
  the recorded trials match the plan.
- **What each run writes:** `logs/sysid-<time>/` holds a `trial_NNNN.npz` per trial and a `manifest.json` (the hand's
  serial number, firmware, settings, trials).
- **Write down which folder is which mount.** The folder name doesn't record it.

## 3. Fit

```bash
$PY scripts/sysid_fit.py logs/sysid-<time 1>:fingers-up logs/sysid-<time 2>:palm-down --fit-gear
```

- **`--fit-gear` is what measures Kt** (one scale per finger).
  - From one mount, Kt can't be identified: a spring in the joint and gravity look the same.
  - Without `--fit-gear`, or with one mount, the output simply repeats the vendor's default Kt.
- **Optional check:** add `--holdout 3.0:0.05` to leave that gain setting out of the fit and report its error separately.
- **Output:** the fit writes to `logs/sysid-<time 1>/fit/` (`--out` changes this):
  - `params.json`:
    - `firmware_law`: `kp_scale`, `kd_scale`, `lag_steps`;
    - `joints.<joint>.gear`: Kt in N·m/A, per aiowuji joint (`thumb_S1` … `pinky_S4`);
    - `fit`: each trial's error per joint in mrad, per window and for the whole trial;
    - `handedness` and `hand_sn`, so check they say `left` and your serial number.
  - `plots/<finger>.png`: recorded and simulated joint angles for every trial.
  - `wuji_hand2_left.xml`: a fitted MuJoCo model of the hand (not used by deploy).

## 4. Check the fit

- **Plots:** the simulated angles should follow the recorded ones. The largest errors are expected at kp 1.5, from
  stick-slip, because MuJoCo has Coulomb friction only.
- **Errors:** in `params.json`, `rollout_rms_mrad` per trial and joint. Compare it with the right hand's fit
  (`reference/mujoco_fit_params.json` in `wuji_hand2_sysid_data_2026-10-14.zip`).
- **Firmware law:** `kp_scale` and `kd_scale` should be about 1.95 and 1.9, as on the right hand (1.9662, 1.9438). A
  very different value suggests different firmware.
- **Left vs right Kt:** print the ratio per joint (step 5).
  - If the ratios are within a few percent, the left hand was already close to the right values.
  - If some are far off, those joints have been running stiffer or softer than in training, so update the config.

## 5. Use the values in deploy

Print them as one line of config overrides, after a comment line with the left/right Kt ratio per joint:

```bash
cd ../..   # repo root
.venv_deploy/bin/python - third_party/aiowuji/logs/sysid-<time 1>/fit/params.json <<'EOF'
import json, sys
from omegaconf import OmegaConf
p = json.load(open(sys.argv[1]))
assert p["handedness"] == "left", p["handedness"]
right = OmegaConf.load("deploy/simtoolreal/config.yaml").robot
law = p["firmware_law"]
print(f"# hand {p['hand_sn']}, firmware {p['firmware']}, mounts {[r['mount'] for r in p['runs']]}")
print("# Kt left/right: " + ", ".join(f"{n} {j['gear'] / right.hand_kt[n]:.2f}" for n, j in p["joints"].items()))
print(f"robot.hand_kp_scale={law['kp_scale']:.4f} robot.hand_kd_scale={law['kd_scale']:.4f} "
      + " ".join(f"robot.hand_kt.{n}={j['gear']:.4f}" for n, j in p["joints"].items()))
EOF
```

- **Where to put them:** add them to the left-hand commands (4c in `HARDWARE_CHECKS.md`) as overrides.
- **Recommended:** keep the fit's folder, and record the hand's serial number, the firmware version and the date
  next to the values, as the right hand's comment in `config.yaml` does.
- **Don't edit `config.yaml`'s `hand_kt` in place.** Those are the right hand's values and the right hand still uses
  them. A per-side `hand_kt` block, picked by `robot.side`, would make this automatic.

## 6. After the values change

- **Dry run first:** step 4c with `robot.hand=dry_run`.
- **Watch the start of the first real run:** if a joint tracks much worse than before, a gain is off. Ctrl-C stops it.
