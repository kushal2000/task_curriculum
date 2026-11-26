"""The open-loop action sequences of the free-space probe: (T, NUM_ENVS, 27) in the policy's action
space and canonical joint order (7 arm joint-velocity commands, then 20 absolute hand positions in
[-1, 1] over each joint's range). Deterministic; built from the env's own limits / default pose, so
both backends get the same sequence as long as their URDF limits agree (checked by the analysis).
"""

from __future__ import annotations

import numpy as np

NUM_ENVS = 8
POLICY_HZ = 60.0
STEP_JOINTS = ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_index_finger_mcp_flex", "r_index_finger_pip")
CHIRP_JOINTS = ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_index_finger_mcp_flex", "r_index_finger_pip",
                "r_middle_finger_mcp_flex")
SIGNAL_NAMES = tuple(f"step_{j}" for j in STEP_JOINTS) + ("chirp", "multisine", "arm_open", "arm_closed")
STEP_D = 0.5        # action units (= 25 % of the joint range)
ARM_A = 0.10        # arm velocity command amplitude (x dof_speed_scale 1.5 = 0.15 rad/s peak)
ARM_F = (0.20, 0.25, 0.30, 0.22, 0.35, 0.28, 0.40)  # Hz per arm joint


def build(names, default, lower, upper, steps: int) -> np.ndarray:
    names = list(names)
    n_arm = 7
    hand = slice(n_arm, len(names))
    hold = np.zeros(len(names))
    hold[hand] = np.clip(2.0 * (default[hand] - lower[hand]) / (upper[hand] - lower[hand]) - 1.0, -1, 1)
    inward = np.where(hold > 0.0, -1.0, 1.0)  # toward the middle of the range
    t = np.arange(steps) / POLICY_HZ
    a = np.tile(hold, (steps, NUM_ENVS, 1))

    def j(name):
        return names.index(name)

    # 0-3: steps on one joint each: +d at 1 s, back at 3.5 s, +2d at 6 s, back at 8 s.
    for e, name in enumerate(STEP_JOINTS):
        k = j(name)
        lvl = np.zeros(steps)
        lvl[(t >= 1.0) & (t < 3.5)] = 1.0
        lvl[(t >= 6.0) & (t < 8.0)] = 2.0
        a[:, e, k] = hold[k] + inward[k] * STEP_D * lvl

    # 4: linear chirp 0.5 -> 5 Hz from 1 s to the end, amplitude 0.3 about a centre 0.4 inward.
    f0, f1, t0, t1 = 0.5, 5.0, 1.0, t[-1]
    tau = np.clip(t - t0, 0.0, None)
    phase = 2 * np.pi * (f0 * tau + 0.5 * (f1 - f0) / (t1 - t0) * tau**2)
    ramp = np.clip(t / 1.0, 0, 1)
    for name in CHIRP_JOINTS:
        k = j(name)
        a[:, 4, k] = hold[k] + inward[k] * (0.4 * ramp + 0.3 * np.sin(phase) * (t >= t0))

    # 5: multisine on every hand joint, 0.5-3 Hz, amplitude 0.25 about a centre 0.35 inward.
    freqs = np.linspace(0.5, 3.0, len(names) - n_arm)
    for i, k in enumerate(range(n_arm, len(names))):
        a[:, 5, k] = hold[k] + inward[k] * (0.35 * ramp + 0.25 * np.sin(2 * np.pi * freqs[i] * np.clip(t - 1.0, 0, None)))

    # 6 / 7: slow arm sinusoids (velocity commands, zero-mean position), hand open / closed.
    for e in (6, 7):
        for i in range(n_arm):
            a[:, e, i] = ARM_A * np.sin(2 * np.pi * ARM_F[i] * np.clip(t - 1.0, 0, None)) * (t >= 1.0)
    closed = [k for k in range(n_arm, len(names)) if not names[k].endswith(("abd",))]
    for k in closed:
        a[:, 7, k] = hold[k] + inward[k] * 1.0 * ramp
    return np.clip(a, -1.0, 1.0).astype(np.float32)
