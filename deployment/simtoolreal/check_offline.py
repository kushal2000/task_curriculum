"""Check deployment/simtoolreal/core.py against a rollout recorded in the training env.

record_sim_rollout.py saved, every step, the state the real loop will measure and what the sim made
of it. This replays that state through core.py and reports, field by field, how far apart they are:

  fk        palm and fingertip body poses (positions, and quaternions including their sign)
  obs       the 134-dim observation, per field
  action    joint targets from the recorded action (the action pipeline)
  success   keypoint distance and goal-reached steps
  goals     every new goal against the training sampler's rules (step size, rotation, volume, table)
  policy    CPU policy actions on the recorded observations vs the sim's (GPU) actions

    .venv_deploy/bin/python deployment/simtoolreal/check_offline.py --rollout <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.simtoolreal.core import (  # noqa: E402
    DEFAULT_CKPT, DEFAULT_SPEC, ActionPipeline, GoalSampler, ObsInputs, RobotFK, Spec, SuccessTracker,
    build_obs, load_policy, quat_mul,
)


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle between two wxyz quaternions, in degrees (sign-free)."""
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rollout", type=Path, required=True, help="record_sim_rollout.py --out directory")
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--policy_envs", type=int, default=3, help="envs whose actions are re-run on CPU")
    args = p.parse_args()

    r = dict(np.load(args.rollout / "rollout.npz"))
    meta = json.loads((args.rollout / "rollout_meta.json").read_text())
    spec = Spec.load(args.spec)
    assert meta["joint_names"] == spec.joint_names, "joint order differs from the spec"
    # Use the fingertip order the sim really observed, whatever the spec says.
    if "fingertip_obs_bodies" not in spec.d:
        spec.d["fingertip_obs_bodies"] = meta["fingertip_bodies_obs_order"]
        spec.d["fingertip_obs_offsets"] = meta["fingertip_offsets_by_position"]
    print("fingertips, obs order:", meta["fingertip_bodies_obs_order"])
    print("spec fingertip_links: ", spec["fingertip_links"])
    fk = RobotFK(spec)
    T, N = r["q"].shape[:2]
    done = r["done"].astype(bool)
    print(f"{T} steps x {N} envs, {int(r['is_success'].sum())} goals reached, {int(done.sum())} episodes ended, "
          f"success tolerance {float(r['success_tol'][-1]):.3f} m")
    report: dict[str, float] = {}

    def worst(name: str, value: float) -> None:
        report[name] = max(report.get(name, 0.0), float(value))

    # --------------------------------------------------------------------------- fk + obs
    field_names = list(spec["obs_list"])
    sign_flips = 0
    for t in range(T):
        for e in range(N):
            q = r["q"][t, e].astype(np.float64)
            pp, pq = fk.link_pose(spec["palm_body"], q)
            worst("fk palm pos [m]", np.abs(pp - r["palm_body_pos"][t, e]).max())
            worst("fk palm quat (signed)", np.abs(pq - r["palm_body_quat"][t, e]).max())
            sign_flips += int(np.dot(pq, r["palm_body_quat"][t, e]) < 0)
            for k, (body, _) in enumerate(spec.fingertip_obs):
                fp, fq = fk.link_pose(body, q)
                worst("fk fingertip pos [m]", np.abs(fp - r["ft_body_pos"][t, e, k]).max())
                worst("fk fingertip quat (sign-free)", min(np.abs(fq - r["ft_body_quat"][t, e, k]).max(),
                                                           np.abs(fq + r["ft_body_quat"][t, e, k]).max()))
            x = ObsInputs(q=q, qd=r["qd"][t, e], prev_targets=r["prev_targets"][t, e],
                          object_pos=r["object_pos"][t, e], object_quat=r["object_quat"][t, e],
                          goal_pos=r["goal_pos"][t, e], goal_quat=r["goal_quat"][t, e],
                          object_scales=r["object_scales"][e])
            obs, _ = build_obs(spec, fk, x, parts=True)
            err = np.abs(obs.astype(np.float64) - r["obs"][t, e])
            i = 0
            for f in field_names:
                width = {"joint_pos": 27, "joint_vel": 27, "prev_action_targets": 27, "palm_pos": 3,
                         "palm_rot": 4, "object_rot": 4, "fingertip_pos_rel_palm": 15,
                         "keypoints_rel_palm": 12, "keypoints_rel_goal": 12, "object_scales": 3}[f]
                worst(f"obs {f}", err[i:i + width].max())
                i += width
    report["fk palm quat sign flips"] = sign_flips

    # --------------------------------------------------------------------------- action pipeline
    for e in range(N):
        for t in range(T):
            if done[t, e]:
                continue
            pipe = ActionPipeline(spec, r["prev_targets"][t, e])
            tgt = pipe.step(r["action"][t, e])
            worst("action targets [rad]", np.abs(tgt - r["targets"][t, e]).max())

    # --------------------------------------------------------------------------- success test
    tracker_mismatch = 0
    tol = float(r["success_tol"][-1])
    for e in range(N):
        tr = SuccessTracker(spec, tolerance=tol)
        for t in range(T):
            d = tr.max_dist(r["object_pos_after"][t, e], r["object_quat_after"][t, e],
                            r["goal_pos"][t, e], r["goal_quat"][t, e])
            if not done[t, e]:
                worst("success keypoint dist [m]", abs(d - r["kp_max_dist"][t, e]))
            hit = tr.update(r["object_pos_after"][t, e], r["object_quat_after"][t, e],
                            r["goal_pos"][t, e], r["goal_quat"][t, e])
            tracker_mismatch += int(hit != bool(r["is_success"][t, e]))
            if done[t, e]:
                tr.near_steps = 0
    report["success steps that disagree"] = tracker_mismatch

    # --------------------------------------------------------------------------- goal rules
    d_max = spec["delta_goal_distance"]
    ang_max = spec["delta_rotation_degrees"]
    lo, hi = np.asarray(spec["target_volume_mins"]), np.asarray(spec["target_volume_maxs"])
    viol = {"step": 0, "rotation": 0, "volume_xy": 0, "table": 0}
    n_goals = 0
    for e in range(N):
        sampler = GoalSampler(spec, r["object_corners"][e], table_top_z=float(r["table_top_z"][e]))
        for t in range(T - 1):
            if not r["is_success"][t, e] or done[t, e]:
                continue
            g0, q0 = r["goal_pos"][t, e], r["goal_quat"][t, e]
            g1, q1 = r["goal_pos"][t + 1, e], r["goal_quat"][t + 1, e]
            n_goals += 1
            step = np.abs(g1 - g0)
            raised = g1[2] > np.clip(g0[2] + d_max, lo[2], hi[2]) + 1e-5
            viol["step"] += int(step[:2].max() > d_max + 1e-5 or (step[2] > d_max + 1e-5 and not raised))
            viol["rotation"] += int(quat_angle(q0, q1) > ang_max + 1e-3)
            viol["volume_xy"] += int(np.any(g1[:2] < lo[:2] - 1e-5) or np.any(g1[:2] > hi[:2] + 1e-5))
            viol["table"] += int(g1[2] < sampler.min_z(q1) - 1e-4)
    report.update({f"goal rule broken: {k}": v for k, v in viol.items()})
    # and our own sampler under the same rules
    own = {"step": 0, "rotation": 0, "table": 0}
    sampler = GoalSampler(spec, r["object_corners"][0], seed=1, table_top_z=float(r["table_top_z"][0]))
    g, q = sampler.first()
    for _ in range(2000):
        g1, q1 = sampler.next()
        raised = g1[2] > np.clip(g[2] + d_max, lo[2], hi[2]) + 1e-9
        own["step"] += int(np.abs(g1 - g)[:2].max() > d_max + 1e-9 or (abs(g1[2] - g[2]) > d_max + 1e-9 and not raised))
        own["rotation"] += int(quat_angle(q, q1) > ang_max + 1e-6)
        own["table"] += int(g1[2] < sampler.min_z(q1) - 1e-9)
        g, q = g1, q1
    report.update({f"own sampler rule broken: {k}": v for k, v in own.items()})

    # --------------------------------------------------------------------------- policy
    if args.policy_envs > 0:
        pol = load_policy(args.checkpoint, expl_coef=float(meta["expl_coef"]))
        for e in range(min(args.policy_envs, N)):
            pol.reset()
            for t in range(T):
                a = pol.act(r["obs"][t, e])
                worst("policy action, CPU vs sim GPU", np.abs(a - r["action"][t, e]).max())
                if done[t, e]:
                    pol.reset()

    print(f"\n{n_goals} goal switches checked\n")
    for k, v in report.items():
        print(f"  {k:42s} {v:.3g}" if isinstance(v, float) else f"  {k:42s} {v}")
    (args.rollout / "check_offline.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
