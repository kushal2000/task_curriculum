"""VBD cloth presets for real fabrics: ``isaacsimenvs/cfg/fabric/<name>.yaml``.

    .venv/isaaclab3/bin/python -m isaacsimenvs.tasks.cloth.utils.fabrics cotton_measured [--plastic]

prints the preset's hydra overrides (``env.cloth.<key>=<value>``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

FABRIC_DIR = Path(__file__).resolve().parents[3] / "cfg" / "fabric"

#: Display order: the env default first, then roughly lightest to heaviest.
ORDER = ("base", "chiffon", "silk", "cotton", "napkin", "towel", "denim")


def names() -> list[str]:
    """Every preset, ``ORDER`` first, then any other file alphabetically."""
    found = {p.stem for p in FABRIC_DIR.glob("*.yaml")}
    return [n for n in ORDER if n in found] + sorted(found - set(ORDER))


def load(name: str) -> dict:
    """The preset's whole file."""
    path = FABRIC_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"no fabric preset {name!r} in {FABRIC_DIR} (have {names()})")
    with open(path) as f:
        return yaml.safe_load(f)


def cloth_params(name: str, plastic: bool = False) -> dict[str, float]:
    """``{DeformableClothCfg field: value}``; with ``plastic`` also ``plastic_yield_curvature``."""
    d = load(name)
    out = {k: float(v) for k, v in d["cloth"].items()}
    if plastic:
        out["plastic_yield_curvature"] = float(d.get("plastic_yield_curvature", 0.0))
        if float(d.get("plastic_set_curvature", 0.0)) > 0.0:  # two-stage presets only
            out["plastic_set_curvature"] = float(d["plastic_set_curvature"])
    return out


def overrides(name: str, plastic: bool = False) -> list[str]:
    """Hydra overrides ``env.cloth.<key>=<value>`` for the preset."""
    return [f"env.cloth.{k}={v!r}" for k, v in cloth_params(name, plastic).items()]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plastic"]
    if len(args) != 1:
        sys.exit(f"usage: python -m isaacsimenvs.tasks.cloth.utils.fabrics NAME [--plastic]   (NAME in {names()})")
    print(" ".join(overrides(args[0], plastic="--plastic" in sys.argv[1:])))
