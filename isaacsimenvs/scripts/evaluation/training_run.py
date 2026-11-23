"""What a checkpoint was trained with: its run recipe and env overrides (stdlib only).

    python -m isaacsimenvs.scripts.evaluation.training_run <checkpoint .pth>

``train.py`` writes ``run.json`` into its Hydra run directory, two levels above ``nn/<file>.pth``;
since 2026-11-23 it also holds the env config the training resolved to (``env``), which an
evaluation applies instead of composing the recipe again (``resolved_recipe``): the world the policy
was trained in, whatever task.yaml has become since. ``distill_pc.py`` writes ``args.json`` next to
its student checkpoints. Checkpoints trained before 2026-11-22 have no ``run.json``: their recipe has
to be named (``experiments/eval.sh`` LIST column).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Env overrides of the training command that are not part of the policy's world.
_NOT_CARRIED = ("env.scene.num_envs",)


def _key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def training_run(checkpoint: str | Path) -> dict | None:
    """``{"experiment": <recipe or None>, "overrides": [env.* hydra overrides], "env": <the resolved
    env config, or None>, "source": <the file>}``, or ``None`` if nothing next to the checkpoint
    records it."""
    for folder in list(Path(checkpoint).resolve().parents)[:3]:
        run = folder / "run.json"
        if run.is_file():
            meta = json.loads(run.read_text())
            overrides = [o for o in meta.get("overrides", [])
                         if _key(o).startswith("env.") and _key(o) not in _NOT_CARRIED]
            return {"experiment": meta.get("experiment"), "overrides": overrides, "env": meta.get("env"),
                    "task": meta.get("task"), "source": str(run)}
        args = folder / "args.json"
        if args.is_file():
            meta = json.loads(args.read_text())
            if "experiment" in meta:
                return {"experiment": meta["experiment"], "overrides": [], "env": None, "task": None,
                        "source": str(args)}
    return None


def resolved_recipe(run: dict) -> dict | None:
    """The training's resolved env as a composed recipe (``tasks/loader.py`` takes it as its
    ``experiment``), or ``None`` if ``run`` has none (run.json before 2026-11-23, args.json)."""
    if not run.get("env"):
        return None
    out = {"env": run["env"], "agent": {}, "dr": [], "files": [run["source"] + " (resolved env)"]}
    if run.get("task"):
        out["task"] = run["task"]
    return out


def without_keys(overrides: list[str], given: list[str]) -> list[str]:
    """``overrides`` minus those whose key is also in ``given`` (Hydra rejects a key set twice)."""
    keys = {_key(o) for o in given}
    return [o for o in overrides if _key(o) not in keys]


if __name__ == "__main__":
    print(json.dumps(training_run(sys.argv[1])))
