"""Command line and the env, built as ``scripts/evaluation/eval.py`` builds it for the target."""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import os
import sys
from pathlib import Path

from . import targets
from .policies import DEFAULT_MENU, load_menu, student_checkpoint

ROOT = Path(__file__).resolve().parents[4]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Watch a fold policy live in a browser (viser).")
    p.add_argument("target", choices=list(targets.TARGETS), help="what to run, as experiments/eval.sh's targets")
    p.add_argument("--policies", default=str(DEFAULT_MENU), help="the policy menu (YAML; see policies.yaml)")
    p.add_argument("--policy", default=None,
                   help="the policy at start: a menu name, or a checkpoint (added to the menu); default the menu's first")
    p.add_argument("--experiment", default=None,
                   help="the env's run recipe; default for the rigid targets: the starting policy's training "
                        "recipe (its menu entry's, else run.json / args.json beside it), none for vbd")
    p.add_argument("--start", choices=("pinned", "rand"), default=None,
                   help="rand: all domain randomisation on; pinned: off and the start pinned (default targets.START)")
    p.add_argument("--num_envs", type=int, default=6, help="rigid sheet: the sizes run evenly over the envs")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    return p


def _recipe(args, checkpoint: str, menu_recipe: str | None) -> tuple[str | None, list[str]]:
    """``(recipe, the training's env overrides)`` of the env, as eval.sh picks them."""
    from isaacsimenvs.scripts.evaluation.training_run import training_run

    if args.experiment is not None:
        return args.experiment, []
    if args.target == "vbd":
        return None, []
    if menu_recipe is not None:
        return menu_recipe, []
    run = training_run(checkpoint)
    if run is None:
        sys.exit(f"[live] no run.json / args.json beside {checkpoint}: name its recipe (--experiment, or "
                 "`recipe:` in its menu entry)")
    return run["experiment"], run["overrides"]


def _reset_starts(env_cfg, randomize: bool) -> dict[str, dict]:
    """``{start pose name: cfg.reset field values}``, the env's own first: the GUI swaps between them
    (the env reads ``cfg.reset`` at every reset)."""
    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization

    pinned_cfg = copy.deepcopy(env_cfg)
    disable_randomization(pinned_cfg)

    def values(cfg):
        return {f.name: copy.deepcopy(getattr(cfg.reset, f.name)) for f in dataclasses.fields(cfg.reset)}

    starts = {"pinned": values(pinned_cfg), "training reset": values(env_cfg)}
    return dict(reversed(starts.items())) if randomize else starts


def _student_cloud(env_cfg, checkpoint: str, recipe: str | None, randomize: bool) -> None:
    """If the starting policy is a student, its point cloud as ``eval.py`` sets it: with a recipe, the
    recipe's (checked against the student's); without one, the student's own, its DR only at rand.
    Students in the menu with another cloud cannot be loaded."""
    from isaacsimenvs.scripts.distillation.student import apply_student_env
    from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud

    student = student_checkpoint(checkpoint)
    if student is None:
        return
    if recipe is None:
        apply_student_env(env_cfg, student["env"], with_dr=randomize)
    else:
        _check_student_cloud(env_cfg, student["env"])


def main() -> None:
    from isaaclab.app import AppLauncher

    from isaacsimenvs.scripts.evaluation.training_run import without_keys

    parser = _parser()
    AppLauncher.add_app_launcher_args(parser)
    args, extra = parser.parse_known_args()
    menu = load_menu(args.policies)
    policy = args.policy or next(iter(menu))
    if policy not in menu:
        if not Path(policy).is_file():
            parser.error(f"--policy {policy!r} is neither in {args.policies} nor a file")
        menu[policy] = (str(Path(policy).resolve()), None)
    os.chdir(ROOT)  # the menu's checkpoints are relative to the repo
    checkpoint = menu[policy][0]
    start = args.start or targets.START[args.target]
    recipe, trained = _recipe(args, checkpoint, menu[policy][1])
    task, overrides = targets.overrides(args.target, extra)
    overrides = without_keys(trained, overrides) + overrides  # the training's, unless the target sets the key
    sys.argv = [sys.argv[0], *overrides]
    rigid = task == targets.RIGID_TASK
    if rigid:
        args.headless = True
    if not str(getattr(args, "device", "") or "").startswith("cuda"):
        args.device = "cuda:0"  # AppLauncher owns --device
    try:
        AppLauncher(args)
    except ImportError as exc:  # the kit-less Newton backend, as in eval.py
        print(f"[live] running kit-less: {exc}", flush=True)

    import gymnasium as gym
    import viser

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    from .fold_pool import even_env_sizes
    from .policies import Players
    from .viewer import LiveViewer

    print(f"[live] {args.target} ({start}), recipe {recipe}: {task} {' '.join(overrides)}", flush=True)
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    status = server.gui.add_markdown(f"**Building {args.target} with {args.num_envs} envs...** (~2-3 min)")

    # A recipe carries an agent block too, so it needs the agent cfg loaded (unused here).
    @hydra_task_config_with_yaml(task, "rl_games_sapg_cfg_entry_point" if recipe else "", experiment=recipe)
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        if not rigid and not any(o.startswith("env.cloth.use_cuda_graph=") for o in overrides):
            # VBD is kernel-launch bound at a handful of envs: the coupled step captured as a CUDA graph
            # is ~5x faster, with the same folds. Pass the override to opt out.
            env_cfg.cloth.use_cuda_graph = True
        randomize = start == "rand"
        reset_starts = _reset_starts(env_cfg, randomize)
        if not randomize:
            from isaacsimenvs.scripts.evaluation.protocol import disable_randomization

            disable_randomization(env_cfg)
        _student_cloud(env_cfg, checkpoint, recipe, randomize)
        with even_env_sizes() if rigid else contextlib.nullcontext():
            env = gym.make(task, cfg=env_cfg)
        status.remove()
        players = Players(menu, env.unwrapped, args.policy_config)
        LiveViewer(server, env, target=f"{args.target} ({start})", players=players, policy=policy,
                   reset_starts=reset_starts).run()

    run()
