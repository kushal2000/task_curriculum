"""Shared training-recipe loading and cloth placement for both viewer simulators."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from isaacsimenvs import CONFIG_DIR
from isaacsimenvs.tasks.cloth.backends.base import place_in_region
from isaacsimenvs.tasks.cloth.mdp.geometry.placement import placement_bounds, constrain_yaw


def load_training_env_config(path=None) -> dict:
    """Read a run recipe's env settings, including its relative base recipe chain."""
    path = Path(path) if path is not None else CONFIG_DIR / "runs" / "rigid_body_cloth.yaml"

    def merge(base, overlay):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = (merge(result[key], value) if isinstance(value, dict)
                           and isinstance(result.get(key), dict) else value)
        return result

    def read(source, ancestors):
        source = source.resolve()
        if source in ancestors:
            raise ValueError(f"Training recipe inheritance cycle at {source}")
        recipe = yaml.safe_load(source.read_text()) or {}
        env = recipe.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"{source}: env must be a mapping")
        if recipe.get("base"):
            base = Path(recipe["base"])
            if not base.suffix:
                base = base.with_suffix(".yaml")
            return merge(read(source.parent / base, ancestors | {source}), env)
        return env

    return read(path, set())


def validate_reset_settings(settings, dimensions=None, table_size=None) -> None:
    """Reject invalid editor values before changing an active simulation."""
    for name in ("x", "y", "yaw_deg", "drop_height", "bend_deg"):
        if not math.isfinite(float(getattr(settings, name))):
            raise ValueError(f"{name} must be finite")
    if settings.drop_height < 0:
        raise ValueError("drop_height must be nonnegative")
    for name in ("x_range", "y_range", "yaw_range_deg"):
        bounds = getattr(settings, name)
        if len(bounds) != 2 or not all(math.isfinite(float(value)) for value in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f"{name} must contain two finite ordered bounds")
    if settings.yaw_range_deg[1] - settings.yaw_range_deg[0] > 360:
        raise ValueError("yaw_range_deg must span at most 360 degrees")

    if dimensions is not None and settings.sample_reset:
        lo, hi = effective_placement_bounds(settings, table_size)
        # Fork RNG so validation cannot alter subsequent sampled reset sequences.
        devices = [dimensions.device] if dimensions.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            constrain_yaw(dimensions, dimensions.new_full((len(dimensions),), float("inf")),
                          dimensions.new_tensor(lo), dimensions.new_tensor(hi), settings.yaw_range_deg)


def effective_placement_bounds(settings, table_size=None):
    return placement_bounds(settings.x_range, settings.y_range, table_size)


def sample_reset_pose(settings, dimensions: torch.Tensor, *, table_size=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Table-relative center/yaw; sampled whole sheets fit the actual tabletop.

    Requested ranges are intersected with the tabletop minus a 5 mm margin.
    Sampled headings are uniform conditional on fitting; manual poses are kept.
    """
    validate_reset_settings(settings)
    if dimensions.ndim != 2 or dimensions.shape[1] != 3 or not dimensions.is_floating_point():
        raise ValueError("dimensions must be a floating (N, 3) tensor")
    if not bool(torch.isfinite(dimensions).all()) or bool((dimensions <= 0).any()):
        raise ValueError("dimensions must be finite and positive")
    count = len(dimensions)
    if not settings.sample_reset:
        xy = dimensions.new_tensor([settings.x, settings.y]).expand(count, -1).clone()
        yaw = dimensions.new_full((count,), math.radians(settings.yaw_deg))
        return xy, yaw
    low, high = map(math.radians, settings.yaw_range_deg)
    yaw = dimensions.new_empty(count).uniform_(low, high)
    lo, hi = effective_placement_bounds(settings, table_size)
    lower, upper = dimensions.new_tensor(lo), dimensions.new_tensor(hi)
    yaw = constrain_yaw(dimensions, yaw, lower, upper, settings.yaw_range_deg)
    return place_in_region(dimensions, yaw, lower, upper, pinned=False), yaw
