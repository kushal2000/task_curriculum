"""Cloth materials covering a run's randomization: every corner of the DR box plus random draws.

Each sample is one value per DR parameter of ``rigid_cloth.randomization`` (material profile).
Corners take every combination of min/max over the parameters whose range is not a single value;
random draws follow the training distribution exactly (log-uniform for bending rigidity, bending
viscosity and density; a uniform choice of ``friction_buckets`` levels for the friction terms;
uniform armature scale). :func:`apply_materials` writes them through the training randomization
code (``randomization._apply``) and reads them back (``randomization.rigid_check``).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

PARAMS = ("bending_rigidity", "bending_viscosity", "density",
          "table_static_friction", "hand_static_friction", "dynamic_friction_ratio", "armature_scale")
LOG_UNIFORM = {"bending_rigidity", "bending_viscosity", "density"}


def _range(rd, name):
    lo, hi = (float(v) for v in getattr(rd, "hinge_armature_scale" if name == "armature_scale" else name))
    return lo, hi


def material_samples(rd, n_random: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """``(M, len(PARAMS))`` values and a 'corner'/'random' label per sample."""
    if getattr(rd, "profile", "material") != "material":
        raise ValueError("physics tests cover the 'material' randomization profile only")
    ranges = {p: _range(rd, p) for p in PARAMS}
    varying = [p for p in PARAMS if ranges[p][1] > ranges[p][0]]
    rows, labels = [], []
    for combo in itertools.product((0, 1), repeat=len(varying)):
        pick = dict(zip(varying, combo))
        rows.append([ranges[p][pick.get(p, 0)] for p in PARAMS])
        labels.append("corner")
    rng = np.random.default_rng(seed)
    buckets = int(getattr(rd, "friction_buckets", 16))
    for _ in range(n_random):
        row = []
        for p in PARAMS:
            lo, hi = ranges[p]
            if p in LOG_UNIFORM:
                row.append(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            elif p == "armature_scale":
                row.append(rng.uniform(lo, hi))
            else:
                row.append(float(rng.choice(np.linspace(lo, hi, buckets))))
        rows.append(row)
        labels.append("random")
    return np.asarray(rows, dtype=np.float64), labels


def apply_materials(backend, ids, values) -> None:
    """Write per-env materials through the training path and verify them in the simulator."""
    import torch

    from isaacsimenvs.tasks.cloth.backends.rigid import randomization

    values = torch.as_tensor(values, dtype=torch.float32, device=backend.device)
    for i, p in enumerate(PARAMS):
        getattr(backend, f"_dr_{p}")[ids] = values[:, i]
    randomization._apply(backend, ids)
    randomization.rigid_check(backend, ids)


def print_failing_ranges(ranges: dict) -> None:
    """One line per varying parameter: its tested range, and the range among failing envs."""
    for p, r in ranges.items():
        failing = "none" if r["failing"] is None else f"{r['failing'][0]:.4g}..{r['failing'][1]:.4g}"
        print(f"  {p:24s} tested {r['all'][0]:.4g}..{r['all'][1]:.4g}   failing {failing}")


def describe_failures(per_env: list[dict], failed_key: str) -> dict:
    """For each DR parameter: its range over all samples vs over failing ones, to show what fails."""
    out = {}
    bad = [e for e in per_env if e[failed_key]]
    for p in [*PARAMS, "length", "width"]:
        values = [e["material"][p] if p in e["material"] else e[p] for e in per_env]
        if max(values) == min(values):
            continue
        bvals = [e["material"][p] if p in e["material"] else e[p] for e in bad]
        out[p] = {"all": [min(values), max(values)],
                  "failing": [min(bvals), max(bvals)] if bvals else None}
    return out
