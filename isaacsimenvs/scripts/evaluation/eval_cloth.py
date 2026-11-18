"""Evaluate a cloth checkpoint under the exact settings in a training config file.

    .venv/isaacsim/bin/python isaacsimenvs/scripts/evaluation/eval_cloth.py --headless \\
        --config <run>/.hydra/config.yaml --checkpoint <run>/nn/<name>.pth --num_envs 1024 \\
        [env.rigid_cloth.designs=[strip_4.yaml] env.rigid_cloth.randomization.enabled=false ...]

Everything comes from ``--config`` (a training run's ``.hydra/config.yaml``: ``env:`` holds the
env settings, ``agent:`` the policy settings), so domain randomization, cloth designs and sizes
are whatever that file says. Trailing ``key=value`` overrides edit that file's values before
anything is built; every key must already exist in it.

The only things changed for evaluation:
* ``--num_envs``: one scored episode per env, the first. (Taking the first N finished
  episodes overall would over-count failures, which finish soonest.)
* The rigid thickness curriculum is frozen at the thickness saved in the checkpoint, so the
  geometry cannot change mid-evaluation. To pick a thickness yourself, pass
  ``env.rigid_cloth.thickness_curriculum.enabled=false env.rigid_cloth.thickness_range=[t,t]``.

Success is the env's own episode-final ``successes > 0`` -- the same signal the thickness
curriculum and training metrics use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from isaacsimenvs.scripts.evaluation.utils import capture_fd_output, raise_if_overflowed

# Serialized as JSON strings in the saved config; the registered config builds them itself.
_SPACES = ("observation_space", "state_space", "action_space")


def add_common_args(parser: argparse.ArgumentParser, num_envs: int) -> None:
    """Arguments shared with ``viser_eval_cloth.py``."""
    parser.add_argument("--config", type=Path, required=True, help="training run's .hydra/config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=num_envs)
    parser.add_argument("--expl_coef", type=float, default=0.0,
                        help="SAPG block coefficient; 0.0 is the last block, trained on task reward only")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    parser.add_argument("overrides", nargs="*", help="key=value edits to --config, e.g. env.rigid_cloth.density=0.2")


def launch(args):
    """Start Kit (PhysX rigid cloth needs it; Newton VBD runs without). Returns the app or None."""
    from isaaclab.app import AppLauncher

    try:
        return AppLauncher(args).app
    except ImportError as exc:
        print(f"[eval_cloth] running kit-less: {exc}", flush=True)
        return None


def _task_id(config: Path, saved: dict) -> str:
    hydra = config.parent / "hydra.yaml"
    if hydra.exists():
        import yaml
        return yaml.safe_load(hydra.read_text())["hydra"]["job"]["config_name"]
    backend = saved["env"].get("backend")
    return {"rigid": "Isaacsimenvs-RigidBodyCloth-Direct-v0", "vbd": "Isaacsimenvs-Cloth-Direct-v0"}[backend]


def build(args, device: str = "cuda:0") -> SimpleNamespace:
    """Env and policy exactly as configured; call after :func:`launch`."""
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.evaluation.utils import PretrainedPlayer, read_policy_cfg

    saved = read_policy_cfg(args.config, device, args.overrides)
    task = _task_id(args.config, saved)
    agent = saved["agent"]["params"]

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.from_dict({k: v for k, v in saved["env"].items() if k not in _SPACES})
    cfg.scene.num_envs = args.num_envs
    if cfg.seed is None:
        cfg.seed = int(agent["seed"])  # training seeds the env from the agent seed

    thickness = None
    if getattr(cfg, "backend", None) == "rigid":
        curriculum = cfg.rigid_cloth.thickness_curriculum
        if curriculum.enabled:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state = (ckpt[0] if 0 in ckpt else ckpt).get("env_state") or {}
            thickness = float(state.get("thickness_curriculum", {}).get("thickness", curriculum.start))
            curriculum.enabled = False
            cfg.rigid_cloth.thickness_range = [thickness, thickness]
            print(f"[eval_cloth] thickness frozen at {thickness * 1000:.2f} mm (from checkpoint)", flush=True)
        else:
            thickness = list(cfg.rigid_cloth.thickness_range)

    env = gym.make(task, cfg=cfg)
    inner = env.unwrapped
    for name in ("observation_space", "action_space"):
        want = json.loads(saved["env"][name])["value"]
        if int(getattr(inner.cfg, name)) != want:
            raise ValueError(f"{name}: env has {getattr(inner.cfg, name)}, training config has {want}")

    player = PretrainedPlayer(args.config, args.checkpoint, num_envs=inner.num_envs, device=device,
                              num_observations=int(inner.cfg.observation_space),
                              num_actions=int(inner.cfg.action_space), expl_coef=args.expl_coef,
                              overrides=args.overrides)
    # The player reseeds torch from the agent config while it is built; reseed for the rollout.
    torch.manual_seed(cfg.seed)
    clip_obs = float(agent["env"].get("clip_observations", math.inf))  # as RlGamesVecEnvWrapper
    clip_act = float(agent["env"].get("clip_actions", math.inf))

    def act(obs):
        action = player.get_action(obs["policy"].clamp(-clip_obs, clip_obs), deterministic=not args.stochastic)
        return action.clamp(-clip_act, clip_act)

    designs = ["?"] * inner.num_envs
    for g in getattr(inner.backend, "groups", []):
        for e in g.env_ids.cpu().tolist():
            designs[e] = g.design.name
    dims = inner.backend._dims.cpu().tolist() if hasattr(inner.backend, "_dims") else [None] * inner.num_envs
    return SimpleNamespace(task=task, cfg=cfg, env=env, inner=inner, player=player, act=act,
                           thickness=thickness, designs=designs, dims=dims)


def _rate(hits: int, n: int) -> dict:
    p = hits / n if n else float("nan")
    return {"n": n, "successes": hits, "success_rate": round(p, 4),
            "sem": round(math.sqrt(p * (1 - p) / n), 4) if n else None}


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser, num_envs=1024)
    parser.add_argument("--out", type=Path, help="JSON results path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = launch(args)

    import torch

    b = build(args)
    env, inner, n, dev = b.env, b.inner, b.inner.num_envs, b.inner.device

    done = torch.zeros(n, dtype=torch.bool, device=dev)
    success = torch.zeros_like(done)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    reasons: dict[str, torch.Tensor] = {}
    max_steps = 2 * int(inner.max_episode_length)

    obs, _ = env.reset()
    b.player.reset_rnn()
    step = 0
    with capture_fd_output() as captured, torch.inference_mode():
        while step < max_steps and not bool(done.all()):
            obs, _, terminated, truncated, extras = env.step(b.act(obs))
            step += 1
            length += (~done).long()
            ended = terminated | truncated
            first = ended & ~done
            if bool(first.any()):
                final = extras["episode_final"]
                success[first] = final["successes"][first] > 0
                for key, value in final.items():
                    if key.startswith("done_"):
                        reasons.setdefault(key[5:], torch.zeros_like(done))[first] = value[first].bool()
                done |= first
            if bool(ended.any()):
                b.player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if step % 50 == 0:
                raise_if_overflowed(captured(), n, f"rollout step {step}")
                print(f"[eval_cloth] step {step}: {int(done.sum())}/{n} done, "
                      f"{int(success.sum())} successes", flush=True)
        raise_if_overflowed(captured(), n, "the rollout")

    scored, succ, ln = done.cpu(), success.cpu(), length.cpu()
    by_design = {}
    for name in dict.fromkeys(b.designs):
        mask = torch.tensor([d == name for d in b.designs]) & scored
        by_design[name] = _rate(int(succ[mask].sum()), int(mask.sum()))

    result = {
        "task": b.task, "config": str(args.config), "checkpoint": str(args.checkpoint),
        "overrides": list(args.overrides), "num_envs": n, "expl_coef": args.expl_coef,
        "stochastic": args.stochastic, "seed": b.cfg.seed, "thickness_m": b.thickness, "steps_run": step,
        "censored": int((~scored).sum()),
        "all": _rate(int(succ[scored].sum()), int(scored.sum())),
        "by_design": by_design,
        "termination_reasons": {k: int(v.cpu()[scored].sum()) for k, v in sorted(reasons.items())},
        "mean_episode_length": round(float(ln[scored].float().mean()), 1) if scored.any() else None,
        "per_env": [{"design": b.designs[e], "dims_m": b.dims[e], "scored": bool(scored[e]),
                     "success": bool(succ[e]), "length": int(ln[e])} for e in range(n)],
    }

    print("\n" + "=" * 72)
    print(f"{b.task} | {args.checkpoint.name} | {n} episodes (first per env)")
    if args.overrides:
        print(f"overrides    = {' '.join(args.overrides)}")
    a = result["all"]
    print(f"success rate = {a['success_rate']} +/- {a['sem']}  (n={a['n']}, censored={result['censored']})")
    for name, r in by_design.items():
        print(f"  {name:20s} {r['success_rate']} +/- {r['sem']}  (n={r['n']})")
    print(f"termination  = {result['termination_reasons']}")
    print("=" * 72 + "\n", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"[eval_cloth] wrote {args.out}", flush=True)

    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
