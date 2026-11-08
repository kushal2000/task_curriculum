"""Check the deploy code against a rollout recorded in the training env (checks/record_sim_rollout.py):
the same inputs must give the same numbers.

    .venv_deploy/bin/python deploy/simtoolreal/checks/check_rec1.py [config overrides]
    # on this machine the checkpoint is elsewhere:
    .venv_deploy/bin/python deploy/simtoolreal/checks/check_rec1.py policy.checkpoint=nov6_dr.txt

1. FK, observation, action -> targets, success distance: every EVERY-th step of all 24 envs.
2. Network: every step of the first NET_ENVS envs, the LSTM run in order (memory reset at each episode).
3. Our own goal sampler against training's rules (no recorded numbers needed).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # deploy/simtoolreal
from policy import action_to_targets, load_network                                         # noqa: E402
from utils.config import load_config, repo_path                                            # noqa: E402
from utils.goals import _volume, first_goal, keypoint_distance, next_goal, object_corners  # noqa: E402
from utils.observation import build_obs, obs_fields                                        # noqa: E402
from utils.quat import quat_apply, quat_conj, quat_mul                                     # noqa: E402
from utils.robot_model import link_pose, load_robot_model                                  # noqa: E402
from utils.spec import load_spec                                                           # noqa: E402

REC = "docs/results/simtoolreal/rec1/rollout.npz"
EVERY = 10
NET_ENVS = 4


def quat_err(a: np.ndarray, b: np.ndarray) -> float:
    """q and -q are the same rotation."""
    return min(np.abs(a - b).max(), np.abs(a + b).max())


def main() -> None:
    cfg = load_config()
    spec = load_spec(repo_path(cfg.spec_path))
    model = load_robot_model(spec)
    with np.load(repo_path(REC)) as f:
        rec = {k: f[k] for k in f.files}   # load once: indexing the npz re-reads the whole array
    n_steps, n_envs = rec["q"].shape[:2]
    err = {}

    def track(name: str, value: float) -> None:
        err[name] = max(err.get(name, 0.0), float(value))

    def inputs(s: int, e: int) -> tuple:
        return (rec["q"][s, e], rec["qd"][s, e], rec["prev_targets"][s, e], rec["object_pos"][s, e],
                rec["object_quat"][s, e], rec["goal_pos"][s, e], rec["goal_quat"][s, e], rec["object_scales"][e])

    # the observation's fields, as slices of the recorded 134 numbers
    slices, start = {}, 0
    sizes = {k: np.size(v) for k, v in obs_fields(spec, model, *inputs(0, 0)).items()}
    for name in spec["obs_list"]:
        slices[name] = slice(start, start + sizes[name])
        start += sizes[name]

    # 1. per step
    for s in range(0, n_steps, EVERY):
        for e in range(n_envs):
            q = rec["q"][s, e]
            p, r = link_pose(model, spec["palm_body"], q)
            track("fk palm body pos [m]", np.abs(p - rec["palm_body_pos"][s, e]).max())
            track("fk palm body quat", quat_err(r, rec["palm_body_quat"][s, e]))
            for k, link in enumerate(spec["fingertip_obs_bodies"]):
                p, r = link_pose(model, link, q)
                track("fk fingertip body pos [m]", np.abs(p - rec["ft_body_pos"][s, e, k]).max())
                track("fk fingertip body quat", quat_err(r, rec["ft_body_quat"][s, e, k]))

            diff = np.abs(build_obs(spec, model, *inputs(s, e)) - rec["obs"][s, e])
            track("obs (all 134)", diff.max())
            for name, sl in slices.items():
                track(f"obs {name}", diff[sl].max())

            if rec["done"][s, e]:   # the episode reset in this step: targets and object were reset too
                continue
            targets = action_to_targets(spec, rec["prev_targets"][s, e], rec["action"][s, e])
            track("targets [rad]", np.abs(targets - rec["targets"][s, e]).max())

            d = keypoint_distance(spec, rec["object_pos_after"][s, e], rec["object_quat_after"][s, e],
                                  rec["goal_pos"][s, e], rec["goal_quat"][s, e])
            track("success keypoint distance [m]", abs(d - rec["kp_max_dist"][s, e]))

    # 2. network, in order
    net = load_network(repo_path(cfg.policy.checkpoint), repo_path(cfg.policy.config), cfg.policy.expl_coef,
                       num_actions=len(spec["joint_names"]))
    net_err = []
    for e in range(NET_ENVS):
        net.reset_rnn()
        for s in range(n_steps):
            if s > 0 and rec["done"][s - 1, e]:
                net.reset_rnn()
            with torch.no_grad():
                action = net.get_action(torch.from_numpy(rec["obs"][s, e])[None], deterministic=True)[0].numpy()
            net_err.append(np.abs(action - rec["action"][s, e]).max())
    net_err = np.array(net_err)

    # 3. our goal sampler vs training's rules
    rng = np.random.default_rng(0)
    corners = object_corners(repo_path(cfg.objects[cfg.object.name].mesh))
    lo, hi = _volume(spec)
    broken = {"outside volume (xy)": 0, "step > delta": 0, "turn > delta": 0, "below table clearance": 0}
    pos, quat = first_goal(spec, rng, corners)
    for _ in range(2000):
        new_pos, new_quat = next_goal(spec, rng, corners, pos, quat)
        broken["outside volume (xy)"] += bool(np.any(new_pos[:2] < lo[:2] - 1e-9) or np.any(new_pos[:2] > hi[:2] + 1e-9))
        broken["step > delta"] += bool(np.any(np.abs(new_pos[:2] - pos[:2]) > spec["delta_goal_distance"] + 1e-9))
        turn = 2 * np.degrees(np.arccos(min(1.0, abs(quat_mul(new_quat, quat_conj(quat))[0]))))
        broken["turn > delta"] += bool(turn > spec["delta_rotation_degrees"] + 1e-6)
        rotated = new_pos + quat_apply(np.broadcast_to(new_quat, (len(corners), 4)), corners)
        broken["below table clearance"] += bool(
            rotated[:, 2].min() < spec["table_top_z_env"] + spec["goal_table_clearance_m"] - 1e-9)
        pos, quat = new_pos, new_quat

    print(f"{REC}: 1. {len(range(0, n_steps, EVERY))} steps x {n_envs} envs")
    for name, v in err.items():
        print(f"  {name:32s} {v:.2e}")
    print(f"2. network, {n_steps} steps x {NET_ENVS} envs: |action - recorded| "
          f"max {net_err.max():.2e}, mean {net_err.mean():.2e}, 99th pct {np.percentile(net_err, 99):.2e}")
    print("3. own goal sampler, 2000 goals, rule broken:", broken)


if __name__ == "__main__":
    main()
