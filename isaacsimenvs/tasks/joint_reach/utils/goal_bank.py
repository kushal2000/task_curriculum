"""Offline goal generation: a collision-checked goal bank and random-walk goal chains through it.

Both are built once with the live sim (zero joint gains, the table at its highest) and cached to disk;
later runs load the cache when its joint order matches the robot's.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .scene_utils import check_points, park_object, robot_points, table_top_and_halfsize

__all__ = [
    "build_goal_bank",
    "build_goal_chains",
    "chain_starts",
    "default_goal",
    "load_goals",
]


def arm_limits(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampling limits, canonical order: arm limits shrunk by the margin, hand limits as they are."""
    lo, hi = env._lo.clone(), env._hi.clone()
    margin = math.radians(env.cfg.goal.arm_limit_margin_deg)
    lo[:7] += margin
    hi[:7] -= margin
    return lo, hi


def begin_validation(env) -> None:
    """Ready the scene for valid: the highest table the reset noise can place, the object away,
    zero joint gains. end_validation restores the gains."""
    g = env.cfg.goal
    top, _, _ = table_top_and_halfsize(env)
    table_pose = torch.zeros(env.num_envs, 7, device=env.device)
    table_pose[:, :3] = env.scene.env_origins
    table_pose[:, 2] += top - 0.5 * g.table_box_size[2]
    table_pose[:, 3] = 1.0
    env.table.write_root_pose_to_sim(table_pose)
    park_object(env, torch.arange(env.num_envs, device=env.device))
    zeros = torch.zeros(env.num_envs, env.robot_spec.num_joints, device=env.device)
    env.robot.write_joint_stiffness_to_sim(zeros)
    env.robot.write_joint_damping_to_sim(zeros)


def end_validation(env) -> None:
    env.robot.write_joint_stiffness_to_sim(env._base_k_lab)
    env.robot.write_joint_damping_to_sim(env._base_d_lab)


def valid(env, q: torch.Tensor) -> torch.Tensor:
    """Which of num_envs candidate configurations (canonical order) are goals: written with zero
    joint gains and stepped, a configuration in contact moves (gravity is off on the robot); one that
    stays put must also pass the workspace test."""
    g = env.cfg.goal
    q_lab = q[:, env._perm_canon_to_lab]
    env.robot.write_joint_state_to_sim(q_lab, torch.zeros_like(q_lab))
    env.robot.set_joint_position_target(q_lab)
    for _ in range(g.collision_check_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    still = (env.robot.data.joint_pos - q_lab).abs().max(dim=-1).values < g.collision_tol_rad
    palm, points = robot_points(env)
    return still & check_points(env, palm, points)


def build_goal_bank(env, size: int, seed: int) -> torch.Tensor:
    """Collision-free configurations spread over the workspace, canonical order, (size, 27): every
    joint uniform in its sampling limits, one candidate per env per round, rejection by valid."""
    g = env.cfg.goal
    gen = torch.Generator(device=env.device).manual_seed(int(seed))
    lo, hi = arm_limits(env)
    begin_validation(env)
    kept, n_kept, n_tried, rounds = [], 0, 0, 0
    while n_kept < size:
        q = lo + (hi - lo) * torch.rand(env.num_envs, lo.numel(), device=env.device, generator=gen)
        ok = valid(env, q)
        kept.append(q[ok])
        n_kept += int(ok.sum())
        n_tried += env.num_envs
        rounds += 1
        if rounds % 100 == 0 or n_kept >= size:
            print(
                f"[joint_reach] goal bank round {rounds}: kept {n_kept}/{n_tried} "
                f"({100.0 * n_kept / n_tried:.2f}%)", flush=True,
            )
        if rounds >= g.max_bank_rounds:
            raise RuntimeError(f"goal bank: {n_kept} of {size} after {rounds} rounds; loosen the workspace")
    end_validation(env)
    return torch.cat(kept)[:size].contiguous()


def build_goal_chains(env, starts: torch.Tensor, length: int, seed: int) -> torch.Tensor:
    """Random walks of `length` goals from `starts` (num_envs, 27), canonical order, (num_envs,
    length, 27). Each step moves every arm joint by up to chain_arm_step_rad x a scale drawn per
    step and redraws the hand; a proposal valid rejects is redrawn next round."""
    g = env.cfg.goal
    dev = env.device
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    lo, hi = arm_limits(env)
    n_env = env.num_envs
    rows = torch.arange(n_env, device=dev)
    chains = starts.unsqueeze(1).repeat(1, length, 1)
    filled = torch.ones(n_env, dtype=torch.long, device=dev)
    begin_validation(env)
    rounds, tried = 0, 0
    while bool((filled < length).any()) and rounds < g.max_chain_rounds:
        cur = chains[rows, filled - 1]
        s_lo, s_hi = g.chain_step_scale_range
        scale = s_lo + (s_hi - s_lo) * torch.rand(n_env, 1, device=dev, generator=gen)
        step = (2.0 * torch.rand(n_env, 7, device=dev, generator=gen) - 1.0) * g.chain_arm_step_rad * scale
        q = lo + (hi - lo) * torch.rand(n_env, lo.numel(), device=dev, generator=gen)
        q[:, :7] = torch.minimum(torch.maximum(cur[:, :7] + step, lo[:7]), hi[:7])
        open_ = filled < length
        ok = valid(env, q) & open_
        tried += int(open_.sum())
        ids = ok.nonzero(as_tuple=False).squeeze(-1)
        chains[ids, filled[ids]] = q[ids]
        filled[ids] += 1
        rounds += 1
        if rounds % 50 == 0:
            print(
                f"[joint_reach] chains round {rounds}: {int(filled.sum())}/{n_env * length} goals, "
                f"acceptance {100.0 * (int(filled.sum()) - n_env) / max(tried, 1):.1f}%", flush=True,
            )
    end_validation(env)
    # A chain that could not be extended repeats its last goal.
    short = filled < length
    if bool(short.any()):
        print(f"[joint_reach] {int(short.sum())} chains stopped short after {rounds} rounds", flush=True)
        idx = torch.arange(length, device=dev).unsqueeze(0).expand(n_env, -1)
        last = torch.minimum(idx, (filled - 1).unsqueeze(1))
        chains = chains[rows.unsqueeze(1), last]
    return chains


def default_goal(env) -> torch.Tensor:
    """The default pose, canonical order (27,)."""
    return env.robot.data.default_joint_pos[0, env._perm_lab_to_canon].clone()


def chain_starts(env, bank: torch.Tensor, default_frac: float, seed: int) -> torch.Tensor:
    """num_envs chain starts: the first default_frac at the default pose, the rest from `bank`."""
    gen = torch.Generator(device=env.device).manual_seed(int(seed) + 1)
    n_def = int(round(default_frac * env.num_envs))
    starts = bank[torch.randint(0, bank.shape[0], (env.num_envs,), device=env.device, generator=gen)]
    starts[:n_def] = default_goal(env)
    return starts


def load_or_build(env, path_str: str, build, what: str) -> torch.Tensor:
    path = Path(path_str)
    if path.is_file():
        payload = torch.load(path, map_location=env.device)
        if list(payload["joint_names"]) != list(env.robot_spec.joint_names_canonical):
            raise ValueError(f"{path}: joint order differs from {env.robot_spec.name}")
        return payload["goals"].to(env.device)
    goals = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"goals": goals.cpu(), "joint_names": list(env.robot_spec.joint_names_canonical),
         "cfg": env.cfg.goal.to_dict()},
        tmp,
    )
    tmp.rename(path)
    print(f"[joint_reach] wrote {what} {path}", flush=True)
    return goals


def load_goals(env) -> None:
    """Populate env._goal_bank, env._chains and env._n_default_chains from the cache or the sim."""
    g = env.cfg.goal
    env._goal_bank = load_or_build(
        env, g.bank_file, lambda: build_goal_bank(env, g.bank_size, g.bank_seed), "goal bank"
    )
    env._chains = load_or_build(
        env,
        g.chain_file,
        lambda: build_goal_chains(
            env, chain_starts(env, env._goal_bank, g.chain_default_frac, g.chain_seed), g.chain_length,
            g.chain_seed,
        ),
        "goal chains",
    )
    env._n_default_chains = int(round(g.chain_default_frac * env._chains.shape[0]))
    print(
        f"[joint_reach] goal bank {tuple(env._goal_bank.shape)}, chains {tuple(env._chains.shape)} "
        f"({env._n_default_chains} from the default pose)", flush=True,
    )
