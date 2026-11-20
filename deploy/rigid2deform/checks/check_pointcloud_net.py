"""Check the point-cloud policy (utils/pointcloud_net.py + pc_obs.py) against its golden file, on the CPU.

The golden file (pc_deploy_handoff/make_golden.py, run on the cluster with the TRAINING code) holds a
40-step input sequence and the training network's actions on it. Here:

  1. load    policy.checkpoint through policy_net.load_policy_net: a pointcloud net, its pc_args = the
             golden's, a cloud pc_obs.py builds
  2. points  pc_obs.featurize on the golden's raw inputs (live_base; from ref_step on, mode label:
             ref_base + ref_moving, mode goal: the goal cloud make_golden.py made of them, GOLDEN_GOAL)
             = the golden's points and mask (the training code's featurisation)
  3. actions net.act on those points, step after step (the LSTM carried) = the golden's actions (< 1e-4)
  4. proprio observation.build_proprio = build_keypoints_obs's 134 without the object's fields, on random joint
             states (the golden's proprio is random numbers, so step 3 does not cover it)
  5. time    one net.act per step at 1 thread (as policy.py runs it), median / p99 over 300 steps

Exits non-zero on any mismatch. With no arguments: every checkpoint in GOLDENS that is in policy/pc/.

    .venv_deploy/bin/python deploy/rigid2deform/checks/check_pointcloud_net.py \
        [--checkpoint deploy/rigid2deform/policy/pc/<name>.pth --golden deploy/rigid2deform/policy/pc/golden_<x>.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform import pc_obs                                               # noqa: E402
from deploy.utils.config import load_config, repo_path                                  # noqa: E402
from deploy.utils.observation import OBJECT_FIELDS, build_keypoints_obs, build_proprio, keypoints_obs_fields   # noqa: E402
from deploy.utils.policy_net import load_policy_net                                     # noqa: E402
from deploy.utils.robot_model import load_robot_model                                   # noqa: E402
from deploy.utils.spec import load_spec                                                 # noqa: E402

CONFIG = HERE.parent / "config.yaml"
PC = "deploy/rigid2deform/policy/pc"
ACTION_TOL = 1e-4
POINT_TOL = 1e-6
# checkpoint -> its golden file (pc_deploy_handoff; the checkpoints' weights = the handoff's)
GOLDENS = {"rigidcloth_pc_label_noiseDR_dagger.pth": "golden_sc_B-noise.npz",
           "rigidcloth_pc_goal_dagger.pth": "golden_gpE_D_scene.npz"}


def golden_goal(ref_base: np.ndarray, ref_moving: np.ndarray) -> np.ndarray:
    """The goal cloud of a mode goal golden file: make_golden.py stores only ref_base / ref_moving and made
    the goal of them as goal_b = [-|x|, y, z + 0.016 on the moving half] (made-up test data, not how
    training makes a goal)."""
    return np.c_[-np.abs(ref_base[:, 0]), ref_base[:, 1], ref_base[:, 2] + np.where(ref_moving, 0.016, 0.0)]


def check_points(g, pc: dict) -> None:
    """2. featurize on the golden's raw inputs = its points and mask."""
    ref_step = int(g["ref_step"])
    worst = 0.0
    for t in range(len(g["actions"])):
        live = g["live_base"][t]
        live = live[np.isfinite(live).all(-1)]                       # NaN rows = padding
        if pc_obs.mode_of(pc) == "label":
            ref = (g["ref_base"][t], g["ref_moving"][t]) if t >= ref_step else (None, None)
            points, mask = pc_obs.featurize(live, pc, *ref)
        else:
            goal = golden_goal(g["ref_base"][t], g["ref_moving"][t]) if t >= ref_step else None
            points, mask = pc_obs.featurize(live, pc, goal_base=goal)
        if not np.array_equal(mask, g["mask"][t]):
            raise SystemExit(f"[points] step {t}: mask differs in {int((mask != g['mask'][t]).sum())} slots")
        worst = max(worst, float(np.abs(points - g["points"][t]).max()))
    if worst > POINT_TOL:
        raise SystemExit(f"[points] max point error {worst:.2e} > {POINT_TOL:.0e}")
    print(f"[points]  ok: {len(g['actions'])} steps, max point error {worst:.1e}, "
          f"{'reference' if pc_obs.mode_of(pc) == 'label' else 'goal'} from step {ref_step}, "
          f"M = {g['points'].shape[1]}, D = {g['points'].shape[2]}")


def check_actions(g, net) -> None:
    """3. the network on the golden's points = its actions."""
    net.reset()
    worst = 0.0
    for t in range(len(g["actions"])):
        a = net.act((g["proprio"][t], g["points"][t], g["mask"][t]))
        worst = max(worst, float(np.abs(a - g["actions"][t]).max()))
    if worst > ACTION_TOL:
        raise SystemExit(f"[actions] max action error {worst:.2e} > {ACTION_TOL:.0e}")
    print(f"[actions] ok: max action error {worst:.1e} (tolerance {ACTION_TOL:.0e}), "
          f"|a| max {np.abs(g['actions']).max():.2f}")


def check_proprio(cfg, n_proprio: int) -> None:
    """4. build_proprio = build_keypoints_obs's 134 without OBJECT_FIELDS."""
    spec = load_spec(repo_path(cfg.spec_path))
    model = load_robot_model(spec)
    rng = np.random.default_rng(0)
    lo, hi = spec["joint_lower"], spec["joint_upper"]
    for _ in range(20):
        q, dq, prev = rng.uniform(lo, hi), rng.normal(0.0, 2.0, len(lo)), rng.uniform(lo, hi)
        quats = rng.normal(size=(2, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        obj = (rng.normal(0, 0.1, 3), quats[0], rng.normal(0, 0.1, 3), quats[1], np.ones(3))
        full = build_keypoints_obs(spec, model, q, dq, prev, *obj)
        sizes = {k: np.size(v) for k, v in keypoints_obs_fields(spec, model, q, dq, prev, *obj).items()}
        ends = np.cumsum([sizes[k] for k in spec["obs_list"]])
        keep = np.concatenate([np.arange(e - sizes[k], e) for k, e in zip(spec["obs_list"], ends)
                               if k not in OBJECT_FIELDS])
        mine = build_proprio(spec, model, q, dq, prev)
        if mine.shape != (n_proprio,):
            raise SystemExit(f"[proprio] {mine.shape}, the network takes ({n_proprio},)")
        if not np.array_equal(mine, full[keep]):
            raise SystemExit(f"[proprio] differs from build_keypoints_obs by {np.abs(mine - full[keep]).max():.2e}")
    print(f"[proprio] ok: {n_proprio} = build_keypoints_obs without {', '.join(OBJECT_FIELDS)} (20 random states)")


def check_time(g, net, steps: int = 300) -> None:
    """5. one act per step on the fullest golden cloud, 1 thread."""
    t = int(np.argmax(g["mask"].sum(-1)))
    obs = (g["proprio"][t], g["points"][t], g["mask"][t])
    net.reset()
    dts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        net.act(obs)
        dts.append(time.perf_counter() - t0)
    dts = 1000 * np.array(dts[20:])                                    # past the warm-up
    print(f"[time]    act: median {np.median(dts):.1f} ms, p99 {np.percentile(dts, 99):.1f} ms "
          f"({torch.get_num_threads()} thread, {int(g['mask'][t].sum())} valid points; a step is 16.7 ms)")


def check_one(checkpoint: str, golden: str) -> None:
    cfg = load_config([f"policy.checkpoint={checkpoint}"], path=CONFIG)
    net = load_policy_net(cfg, num_actions=27, kinds=("pointcloud",))
    g = np.load(repo_path(golden))
    golden_pc = json.loads(str(g["pc_args"]))
    if golden_pc != net.pc_args:
        raise SystemExit(f"[load] checkpoint pc_args {net.pc_args} != the golden's {golden_pc}")
    pc_obs.check_supported(net.pc_args, net.point_dim)
    print(f"[load]    ok: {Path(checkpoint).name} ({Path(golden).name}): {net.kind}, mode {pc_obs.mode_of(net.pc_args)}, "
          f"labels {pc_obs.labels_of(net.pc_args)}, proprio {net.n_proprio}, point dim {net.point_dim}")

    check_points(g, net.pc_args)
    check_actions(g, net)
    check_proprio(cfg, net.n_proprio)
    check_time(g, net)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", help="default: every checkpoint of GOLDENS in policy/pc/")
    ap.add_argument("--golden")
    args = ap.parse_args()
    if (args.checkpoint is None) != (args.golden is None):
        raise SystemExit("give both --checkpoint and --golden, or neither")
    pairs = ([(args.checkpoint, args.golden)] if args.checkpoint else
             [(f"{PC}/{c}", f"{PC}/{g}") for c, g in GOLDENS.items() if repo_path(f"{PC}/{c}").is_file()])
    if not pairs:
        raise SystemExit(f"none of {list(GOLDENS)} in {PC}")
    for checkpoint, golden in pairs:
        check_one(checkpoint, golden)
    print(f"all checks passed ({len(pairs)} checkpoint{'s' * (len(pairs) > 1)})")


if __name__ == "__main__":
    main()
