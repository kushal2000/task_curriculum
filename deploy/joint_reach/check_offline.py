"""Check deploy/joint_reach/core.py against a rollout recorded in sim, with no robot.

real_test_sequence.pt (isaacsimenvs/eval/joint_reach_eval.py) holds env 0's no-DR rollout of the real
test sequence: per policy step the observation, the action, and after the step the joint targets,
joint positions and velocities. Three checks, each must match to float32 precision:

  obs       build_obs from the recorded state, previous target and goal schedule = recorded obs
  pipeline  ActionPipeline fed the recorded actions = recorded targets
  policy    Policy fed the recorded observations, in order = recorded actions, to 5e-2: the sim ran
            the network on the GPU with 500 envs a batch, whose kernels (cuDNN LSTM in TF32) round
            differently (step 0 already differs by ~5e-4; a batch-500 TF32 GPU replay gets to 1.5e-3).
            The arm target moves 0.025 rad per unit action per step, so 1e-2 is ~2e-4 rad.

    python deploy/joint_reach/check_offline.py --checkpoint <ckpt>.pth --sequence <dir>/real_test_sequence.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.joint_reach.core import DEFAULT_SPEC, ActionPipeline, GoalSequence, Policy, Spec, build_obs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import torch

    spec = Spec.load(args.spec)
    rec = torch.load(args.sequence, map_location="cpu", weights_only=False)
    seq = GoalSequence.load(args.sequence)
    tr = {k: v.double().numpy() for k, v in rec["sim_trajectory"].items()}
    n_steps = len(tr["obs"])
    goal_at = seq.goal_at()
    assert len(goal_at) == n_steps, (len(goal_at), n_steps)
    # The eval writes the next goal after env.step has built the observation, so in the recording
    # the policy sees each new goal one step late.
    goal_seen = np.vstack([goal_at[:1], goal_at[:-1]])
    ok = True

    # Observations: obs[t] sees the state after step t - 1 (the start pose, at rest, before step 0).
    q_prev = np.vstack([seq.start, tr["q"][:-1]])
    qd_prev = np.vstack([np.zeros_like(seq.start), tr["qd"][:-1]])
    tgt_prev = np.vstack([seq.start, tr["target"][:-1]])
    obs = np.stack([build_obs(spec, q_prev[t], qd_prev[t], tgt_prev[t], goal_seen[t]) for t in range(n_steps)])
    err = np.abs(obs - tr["obs"]).max(axis=0)
    names = ["pos", "vel", "prev_target", "goal"]
    n = spec.num_joints
    worst = {nm: float(err[i * n:(i + 1) * n].max()) for i, nm in enumerate(names)}
    ok &= max(worst.values()) < 1e-4
    print(f"[check] obs       max |diff| per block {worst}")

    # Action pipeline.
    pipe = ActionPipeline(spec, seq.start)
    tgt = np.stack([pipe.step(a) for a in tr["action"]])
    d = float(np.abs(tgt - tr["target"]).max())
    ok &= d < 1e-5
    print(f"[check] pipeline  max |target diff| {d:.2e} rad")
    arm_rate = np.abs(np.diff(tgt[:, :spec.n_arm], axis=0)).max() / spec.dt
    print(f"[check] pipeline  fastest arm target {arm_rate:.3f} rad/s "
          f"(limit {spec.dof_speed_scale * spec.arm_ma:.3f})")

    # Policy, recurrent: the recorded observations in order.
    pol = Policy(args.checkpoint, num_obs=4 * n, num_actions=n, device=args.device)
    t0 = time.perf_counter()
    act = np.stack([pol.act(o) for o in tr["obs"]])
    per_step_ms = (time.perf_counter() - t0) / n_steps * 1e3
    d = np.abs(act - tr["action"])
    ok &= float(d.max()) < 5e-2
    for lo_t in range(0, n_steps, 500):
        print(f"[check] policy    steps {lo_t}-{lo_t + 499}: max |action diff| {d[lo_t:lo_t + 500].max():.1e}")
    print(f"[check] policy    max |action diff| {d.max():.2e} (median {np.median(d):.1e}), "
          f"{per_step_ms:.2f} ms/step on {args.device}")
    print("[check] PASS" if ok else "[check] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
