"""Find a training run's checkpoint by its run name (stdlib only).

    python -m isaacsimenvs.scripts.evaluation.checkpoints <run name> [best|last|ep:<N>] [--outputs outputs]

prints the path ("" if there is none). Used by experiments/eval.sh.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_checkpoint(name: str, outputs: Path, select: str = "best") -> Path | None:
    """A checkpoint of the run called ``name``.

    ``train.py`` co-locates rl_games artifacts with the Hydra run directory, so the path contains a
    date and a time a caller cannot know. Searching by name is the only stable handle.

    ``select="best"`` prefers ``nn/<name>.pth`` -- rl_games' BEST-reward checkpoint, NOT the last
    one -- and falls back to the highest-epoch ``last_<name>_ep_<n>_rew_<r>.pth``. In a long run the
    best checkpoint is usually written near the end, but not always: the 2026-10-07 ft7lr finetunes
    wrote it once at epoch ~10 (inflated early reward, before any episode could time out) and never
    again. ``select="last"`` skips ``nn/<name>.pth`` and returns the highest-epoch ``last_*_ep_*``
    file, i.e. the policy at the end of training. Use it after a finetune. ``select="ep:<N>"`` returns
    epoch N's ``last_`` / ``periodic_`` (full) / ``snap_`` (weights only) checkpoint, in that order.

    Deliberately does NOT fall back to ``last/model.pth``: that file is written by a different code
    path and has been observed to lag the ``nn/`` directory by several hundred epochs, which would
    silently evaluate a younger policy.
    """
    epoch = None
    if select.startswith("ep:"):
        epoch = int(select[3:])
    elif select not in ("best", "last"):
        raise ValueError(f"select must be 'best', 'last' or 'ep:<N>', got {select!r}")
    for run_dir in sorted(outputs.rglob(name), reverse=True):
        nn = run_dir / "nn"
        if not nn.is_dir():
            continue
        if epoch is not None:
            for pattern in (f"last_{name}_ep_{epoch}_rew_*.pth", f"periodic_{name}_ep_{epoch}.pth",
                            f"snap_{name}_ep_{epoch}.pth"):
                hits = sorted(nn.glob(pattern))
                if hits:
                    return hits[0]
            continue
        direct = nn / f"{name}.pth"
        if select == "best" and direct.is_file():
            return direct
        eps = []
        for p in nn.glob(f"last_{name}_ep_*.pth"):
            try:
                eps.append((int(p.name.split("_ep_")[1].split("_")[0]), p))
            except (IndexError, ValueError):
                continue
        if eps:
            return max(eps)[1]
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name")
    ap.add_argument("select", nargs="?", default="best")
    ap.add_argument("--outputs", default="outputs")
    a = ap.parse_args()
    print(find_checkpoint(a.name, Path(a.outputs), a.select) or "")
