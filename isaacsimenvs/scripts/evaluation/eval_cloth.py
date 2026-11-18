"""Evaluate a cloth checkpoint under the exact settings in a training config file.

    .venv/isaacsim/bin/python isaacsimenvs/scripts/evaluation/eval_cloth.py --headless \\
        --config <run>/.hydra/config.yaml --checkpoint <run>/nn/<name>.pth --num_envs 1024 \\
        [env.rigid_cloth.designs=[strip_4.yaml] env.rigid_cloth.randomization.enabled=false ...]

Everything comes from ``--config`` (a training run's ``.hydra/config.yaml``: ``env:`` holds the
env settings, ``agent:`` the policy settings), so domain randomization, cloth designs and sizes
are whatever that file says. Trailing ``key=value`` overrides edit that file's values before
anything is built; each key (or its parent section) must exist in it.

The only thing changed for evaluation is ``--num_envs``: one scored episode per env, the first.
(Taking the first N finished episodes overall would over-count failures, which finish soonest.)
Configs saved before the thickness curriculum and the size grid were removed are translated on
load (``backends/rigid/legacy.py``): a curriculum run is evaluated at the thickness saved in its
checkpoint. To pick a thickness yourself, pass ``env.rigid_cloth.thickness=<m>``.

Two success rates per scored episode:
* fold success: the task's ``fold`` flag fired on at least one step -- the cloth met ``is_folded``
  (on an eligible step); the same event that pays the fold-entry bonus in training;
* release success: the env's episode-final ``successes > 0`` -- fold plus release (robot clear of the
  cloth while folded); the signal the training metrics use.
Penetration (``PenetrationMeter``): per scored episode, the deepest robot-cloth contact seen (PhysX
separation, the last physics substep of each policy step) and the fraction of contact steps deeper
than 5 mm.
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


def load_env_cfg(config: Path, overrides, num_envs: int, checkpoint: Path | None = None,
                 device: str = "cuda:0") -> SimpleNamespace:
    """The env config a training run used, plus overrides; call after :func:`launch`.

    Old rigid-cloth configs are translated first (``legacy.upgrade_rigid_cloth``); a thickness
    curriculum run gets the thickness saved in ``checkpoint`` when one is given.
    """
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.evaluation.utils import read_policy_cfg

    from isaacsimenvs.tasks.cloth.backends.rigid.legacy import upgrade_rigid_cloth

    saved = read_policy_cfg(config, device, overrides)
    task = _task_id(config, saved)
    rc = saved["env"].get("rigid_cloth")
    if isinstance(rc, dict):
        checkpoint_thickness = None
        if checkpoint is not None and (rc.get("thickness_curriculum") or {}).get("enabled"):
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = (ckpt[0] if 0 in ckpt else ckpt).get("env_state") or {}
            checkpoint_thickness = (state.get("thickness_curriculum") or {}).get("thickness")
        for note in upgrade_rigid_cloth(rc, checkpoint_thickness):
            print(f"[eval_cloth] old config: {note}", flush=True)
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.from_dict({k: v for k, v in saved["env"].items() if k not in _SPACES})
    cfg.scene.num_envs = num_envs
    if cfg.seed is None:
        cfg.seed = int(saved["agent"]["params"]["seed"])  # training seeds the env from the agent seed

    thickness = float(cfg.rigid_cloth.thickness) if getattr(cfg, "backend", None) == "rigid" else None
    return SimpleNamespace(task=task, cfg=cfg, saved=saved, thickness=thickness)


def build(args, device: str = "cuda:0") -> SimpleNamespace:
    """Env and policy exactly as configured; call after :func:`launch`."""
    import gymnasium as gym
    import torch

    from isaacsimenvs.scripts.evaluation.utils import PretrainedPlayer

    loaded = load_env_cfg(args.config, args.overrides, args.num_envs, args.checkpoint, device)
    task, cfg, saved, thickness = loaded.task, loaded.cfg, loaded.saved, loaded.thickness
    agent = saved["agent"]["params"]

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


class PenetrationMeter:
    """How deep the robot is inside the cloth, from PhysX's own robot-cloth contacts.

    Reads the contact view the rigid cloth's press-driven-fold check already creates (the last
    physics substep of each policy step): ``current()`` is, per env, the deepest robot contact
    (-separation, metres, 0 when not touching). ``update(live)`` folds it into per-episode
    statistics for the envs in ``live``; ``reset(ids)`` starts their next episode.
    ``available`` is False when the config does not enable contact reporting.
    """

    def __init__(self, inner, deep_m: float = 0.005):
        import torch

        guard = getattr(inner, "_episode_guard", None)
        self.view = getattr(guard, "contacts", None) if guard is not None else None
        self.available = self.view is not None
        n, dev = inner.num_envs, inner.device
        self.n, self.deep_m = n, deep_m
        self.now = torch.zeros(n, device=dev)
        self.episode_max = torch.zeros(n, device=dev)
        self.contact_steps = torch.zeros(n, device=dev)
        self.deep_steps = torch.zeros(n, device=dev)

    def current(self):
        import torch

        from isaacsimenvs.tasks.cloth.backends.rigid.contacts import contact_indices

        out = torch.zeros(self.n, device=self.now.device)
        if not self.available:
            return out
        cv = self.view
        _, _, _, separation, counts, starts = cv.view.get_contact_data(dt=cv.dt)
        pair, index = contact_indices(counts, starts, cv.capacity)
        robot = (pair % cv.view.filter_count) < cv.num_robot_links
        env = cv.env_ids[pair // cv.view.filter_count][robot]
        depth = (-separation.flatten()[index][robot]).clamp_min(0)
        out.scatter_reduce_(0, env, depth, reduce="amax")
        # Envs whose robot touches without penetrating still count as in contact.
        touching = torch.zeros(self.n, dtype=torch.bool, device=out.device)
        touching[env] = True
        self._touching = touching
        return out

    def update(self, live):
        import torch

        self.now = self.current()
        if not self.available:
            return
        self.episode_max = torch.where(live, self.episode_max.maximum(self.now), self.episode_max)
        self.contact_steps += (live & self._touching).float()
        self.deep_steps += (live & (self.now > self.deep_m)).float()

    def reset(self, ids):
        self.episode_max[ids] = 0.0
        self.contact_steps[ids] = 0.0
        self.deep_steps[ids] = 0.0


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
    folded = torch.zeros_like(done)  # the fold flag fired at least once in the scored episode
    pen = PenetrationMeter(inner)
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
            final = extras["episode_final"]
            # Published every step (pre-reset on an env's last step), so this covers the whole episode.
            folded |= (final["done_fold"] > 0) & ~done
            pen.update(~done)  # the scored (first) episode only; stats freeze once an env is done
            if bool(first.any()):
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
                      f"{int((folded & done).sum())} folds, {int(success.sum())} releases", flush=True)
        raise_if_overflowed(captured(), n, "the rollout")

    scored, succ, fold_ok, ln = done.cpu(), success.cpu(), folded.cpu(), length.cpu()
    pen_max = pen.episode_max.cpu() * 1000
    deep_frac = (pen.deep_steps / pen.contact_steps.clamp_min(1)).cpu()

    def penetration(mask):
        if not pen.available or not bool(mask.any()):
            return None
        x = pen_max[mask]
        return {"median_max_mm": round(float(x.median()), 2), "p90_max_mm": round(float(x.quantile(0.9)), 2),
                "max_mm": round(float(x.max()), 2), "episodes_over_10mm": round(float((x > 10).float().mean()), 4),
                "deep_contact_step_fraction": round(float(deep_frac[mask].mean()), 4)}

    def rates(mask):
        m = int(mask.sum())
        return {"fold": _rate(int(fold_ok[mask].sum()), m), "release": _rate(int(succ[mask].sum()), m),
                "penetration": penetration(mask)}

    by_design = {name: rates(torch.tensor([d == name for d in b.designs]) & scored) for name in dict.fromkeys(b.designs)}

    result = {
        "task": b.task, "config": str(args.config), "checkpoint": str(args.checkpoint),
        "overrides": list(args.overrides), "num_envs": n, "expl_coef": args.expl_coef,
        "stochastic": args.stochastic, "seed": b.cfg.seed, "thickness_m": b.thickness, "steps_run": step,
        "censored": int((~scored).sum()),
        "all": rates(scored),
        "by_design": by_design,
        "termination_reasons": {k: int(v.cpu()[scored].sum()) for k, v in sorted(reasons.items())},
        "mean_episode_length": round(float(ln[scored].float().mean()), 1) if scored.any() else None,
        "per_env": [{"design": b.designs[e], "dims_m": b.dims[e], "scored": bool(scored[e]),
                     "fold_success": bool(fold_ok[e]), "release_success": bool(succ[e]),
                     "max_penetration_mm": round(float(pen_max[e]), 2) if pen.available else None,
                     "deep_contact_step_fraction": round(float(deep_frac[e]), 4) if pen.available else None,
                     "length": int(ln[e])} for e in range(n)],
    }

    print("\n" + "=" * 72)
    print(f"{b.task} | {args.checkpoint.name} | {n} episodes (first per env)")
    if args.overrides:
        print(f"overrides    = {' '.join(args.overrides)}")
    def line(label, r):
        f, rel = r["fold"], r["release"]
        p = r["penetration"]
        pen_txt = ("penetration n/a" if p is None else
                   f"penetration max/episode median {p['median_max_mm']} p90 {p['p90_max_mm']} max {p['max_mm']} mm, "
                   f">10 mm {p['episodes_over_10mm']:.0%}")
        return (f"{label:20s} fold {f['success_rate']:.4f} +/- {f['sem']}   "
                f"release {rel['success_rate']:.4f} +/- {rel['sem']}   (n={f['n']})\n{'':22s}{pen_txt}")
    print(line("all", result["all"]) + f"  censored={result['censored']}")
    for name, r in by_design.items():
        print("  " + line(name, r))
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
