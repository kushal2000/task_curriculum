"""Startup proofs for the rewritten Isaacsimenvs-Cloth-Direct-v0 (VBD on Newton); fails loudly.

    .venv/isaaclab3/bin/python experiments/nov16/cloth_from_scratch/probe_vbd.py --num_envs 8 [hydra overrides]

1. builds, resets, obs 134 / critic 156, finite;
2. zero actions for 120 steps: every cloth lands flat on its table (lowest particle within 2 cm of
   the table top, height spread < 1.5 cm) and is NOT folded (the negative control);
3. a teleported perfect fold (``grid.folded_start`` at 180 deg) left to settle 360 physics steps keeps
   ``fold_error`` small (measured ~2 mm before the rewrite) and passes ``is_folded``;
   envs a fingertip came within 3 cm of are skipped and counted; then one task step on it
   terminates with ``done_fold``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat  # noqa: E402

compat.install_isaaclab3_compat()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.cloth.mdp.geometry import fold, grid  # noqa: E402
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml  # noqa: E402

TASK = "Isaacsimenvs-Cloth-Direct-v0"
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"[probe] {'PASS' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


@hydra_task_config_with_yaml(TASK, "rl_games_sapg_cfg_entry_point")
def run(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(TASK, cfg=env_cfg)
    e = env.unwrapped
    n, dev = e.num_envs, e.device
    obs, _ = env.reset()
    check(obs["policy"].shape == (n, 134) and obs["critic"].shape == (n, 156), f"obs {tuple(obs['policy'].shape)} / {tuple(obs['critic'].shape)}")
    check(bool(torch.isfinite(obs["policy"]).all()), "reset obs finite")

    zero = torch.zeros((n, e.cfg.action_space), device=dev)
    near = torch.full((n,), 9.0, device=dev)

    def hand_gap():
        s = e.backend.read_state()
        return torch.cdist(s.fingertip_pos_w, s.points_w).amin(dim=(1, 2))

    for _ in range(120):
        obs, *_ = env.step(zero)
        near = torch.minimum(near, hand_gap())
    s = e.backend.read_state()
    spread, lift = fold.flat_on_table(s.points_w, s.table_top_z)
    free = near > 0.03
    print(f"[probe] {int((~free).sum())}/{n} envs had a fingertip within 3 cm of the cloth: skipped", flush=True)
    print(f"[probe] flat: spread {spread.min():.4f}-{spread.max():.4f} m, lift {lift.min():.4f}-{lift.max():.4f} m", flush=True)
    check(bool((lift[free].abs() < 0.02).all()), f"cloths on their tables (lowest particle {float(lift[free].abs().max()) * 1e3:.1f} mm from the top)")
    check(bool((spread[free] < 0.015).all()), f"cloths flat (spread max {float(spread[free].max()) * 1e3:.1f} mm)")
    check(not bool(e.is_folded()[free].any()), "no flat cloth counts as folded")
    check(bool(torch.isfinite(obs["policy"]).all()), "obs finite after 120 steps")

    # 3. teleport a perfect fold onto each cloth where it lies, and let it settle
    s = e.backend.read_state()
    ids = torch.arange(n, device=dev)
    cloth = e.backend.cloth_grid
    r, c = fold.stationary_frame(cloth, s.points_w)
    folded = fold.carry(cloth, grid.folded_start(cloth, torch.full((n,), math.pi, device=dev)), r, c)
    e.backend._cloth.write_nodal_pos_to_sim_index(folded, ids)
    e.backend._cloth.write_nodal_velocity_to_sim_index(torch.zeros_like(folded), ids)
    err0 = e.fold_error()
    print(f"[probe] teleported fold: fold_error {float(err0.max()) * 1e3:.2f} mm (before settling)", flush=True)
    # Settle on raw physics steps: a task step would end the episode on the fold (success) and reset.
    near = torch.full((n,), 9.0, device=dev)
    for _ in range(360):  # 180 policy steps x decimation 2
        e.scene.write_data_to_sim()
        e.sim.step(render=False)
        e.scene.update(e.physics_dt)
        near = torch.minimum(near, hand_gap())
    free = near > 0.03
    err = e.fold_error()[free]
    folded_ok = e.is_folded()[free]
    print(f"[probe] settled fold_error {err.min() * 1e3:.2f}-{err.max() * 1e3:.2f} mm over {int(free.sum())} envs, "
          f"footprint {e.footprint_ratio()[free].min():.3f}-{e.footprint_ratio()[free].max():.3f}", flush=True)
    check(float(err.max()) < 0.006, f"perfect fold settles: fold_error max {float(err.max()) * 1e3:.2f} mm")
    check(bool(folded_ok.all()), f"perfect fold is_folded for {int(folded_ok.sum())}/{int(free.sum())}")
    # ... and one task step on it ends the episode as a fold
    _, _, term, _, extras = env.step(zero)
    done_fold = extras["episode_final"]["done_fold"].bool()
    check(bool(done_fold[free].all() and term[free].all()), f"a task step on the fold terminates with done_fold ({int(done_fold.sum())}/{n})")
    print(f"[probe] {'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}", flush=True)
    env.close()


run()
sys.stdout.flush()
os._exit(0 if not FAILS else 1)
