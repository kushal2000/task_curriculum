"""VBD cloth presets for real fabrics: ``isaacsimenvs/cfg/fabric/<name>.yaml``.

The one source of the fabric parameters for the eval launcher (``scripts/cluster/bos14_fabric_eval.sh``)
and the analysis scripts (cantilever, droop, plasticity probes). ``base`` is the env's default sheet.

Each file has
  * ``cloth:`` -- ``ClothCfg`` fields, applied as ``env.cloth.<key>=<value>``:
      - ``density``: areal mass [kg/m^2], literature (2026-10-28 table);
      - ``tri_ke = tri_ka``: 56 x KES shear G; the env default 100 where there is no source;
      - ``edge_ke``: tuned (2026-10-30, ``scripts/analysis/fabric_cantilever.py``) so the 5 cm flap of
        the env's 10 cm 7x7 sheet droops like the exact elastica for the literature bending length
        ``(B / (rho g))^(1/3)`` at 12 VBD iterations; lands at 127-150 x B;
      - ``edge_kd``: 1e-3 x density / 2.0 -- the default 1e-3 at 12 iterations under-converges on
        these light sheets and adds 6-32 deg of extra sag;
      - ``soft_contact_mu``: mu_skin^2 / 1.5; the env default 0.25 where there is no source;
      - ``self_contact_radius`` (and the towel's ``self_contact_margin``): the env default except
        towel (thick pile).
  * ``plastic_yield_curvature`` [1/m]: bending plasticity (``utils/plasticity.py``), applied only on
    request: the gentlest threshold that holds a pressed fold in the scripted fold-press-release test
    (``scripts/analysis/cloth_plasticity_probe.py``, job 259470).
  * ``literature:`` -- inputs the scripts read and never apply to the env: ``bending_rigidity`` B
    [N m] and ``tri_range``, the tri_ke = tri_ka range from tensile data (2026-10-31).

Standalone (yaml only), so scripts load it by file path and the shell can call it::

    python isaacsimenvs/tasks/cloth/utils/fabrics.py towel [--plastic]   # prints the hydra overrides
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
    """``{ClothCfg field: value}``; with ``plastic`` also ``plastic_yield_curvature``."""
    d = load(name)
    out = {k: float(v) for k, v in d["cloth"].items()}
    if plastic:
        out["plastic_yield_curvature"] = float(d.get("plastic_yield_curvature", 0.0))
    return out


def overrides(name: str, plastic: bool = False) -> list[str]:
    """Hydra overrides ``env.cloth.<key>=<value>`` for the preset."""
    return [f"env.cloth.{k}={v!r}" for k, v in cloth_params(name, plastic).items()]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plastic"]
    if len(args) != 1:
        sys.exit(f"usage: fabrics.py NAME [--plastic]   (NAME in {names()})")
    print(" ".join(overrides(args[0], plastic="--plastic" in sys.argv[1:])))
