"""Cloth.yaml (the VBD task, on Newton) keeps every section of Play.yaml it was copied from.

``hydra_task_config_with_yaml`` does a flat ``yaml.safe_load`` of one file: no ``defaults:``
merge, no inheritance. So the cloth task YAML is a full copy of Play's, and a section dropped from
it silently falls back to the configclass default rather than to Play's value. The task VALUES are
the cloth task's own (rewards, terminations, DR); what is checked here is the shape: the same
sections, the Newton-only additions allow-listed, and the one Newton cloning trap. RigidClothPhysx.yaml
is in turn diffed against Cloth.yaml by test_physx_cloth_conventions.py.

Kit-free by construction -- it reads YAML and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CFG = Path(__file__).resolve().parents[1] / "isaacsimenvs" / "cfg" / "task"

#: Top-level keys Cloth.yaml is allowed to add, with why.
ALLOWED_EXTRA = {
    "newton",        # MJWarp solver knobs; no PhysX counterpart
    "use_gravcomp",  # per-body gravity route, Newton-only
    "meshify",       # convex-mesh colliders, Newton-only
    "cloth",         # the manipuland
}


#: Dotted paths Cloth.yaml is allowed to omit.
#:
#: `sim.physx` must be absent: Isaac Lab 3.0 renamed the field to `physics`, and the overlay goes
#: through `configclass.from_dict`, which raises `KeyError: Key not found under namespace:
#: /sim/physx` on an unknown key. (compat.py shims the *constructor* keyword, not dict overlays.)
#: It would be inert anyway -- `tasks/cloth/newton_base_env.py` replaces `cfg.sim.physics` with a NewtonCfg.
ALLOWED_MISSING_PREFIXES = ("sim.physx",)


def _flatten(node, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


@pytest.fixture(scope="module")
def cfgs() -> tuple[dict, dict]:
    play = yaml.safe_load((CFG / "Play.yaml").read_text())
    newton = yaml.safe_load((CFG / "Cloth.yaml").read_text())
    return play, newton


def test_cloth_adds_only_allowlisted_top_level_keys(cfgs) -> None:
    play, newton = cfgs
    extra = set(newton) - set(play)
    assert extra <= ALLOWED_EXTRA, (
        f"Cloth.yaml adds unexpected top-level keys: {sorted(extra - ALLOWED_EXTRA)}. "
        "If the addition is intentional, add it to ALLOWED_EXTRA with a reason."
    )


def test_cloth_drops_no_keys(cfgs) -> None:
    play, newton = cfgs
    missing = set(play) - set(newton)
    assert not missing, (
        f"Cloth.yaml is missing top-level keys present in Play.yaml: {sorted(missing)}. "
        "The loader does no inheritance, so a missing section silently falls back to the "
        "configclass default rather than to Play's value."
    )


def test_cloth_omits_no_play_setting(cfgs) -> None:
    """Every leaf Play.yaml sets, Cloth.yaml sets too (to its own value), bar ``sim.physx``."""
    play, cloth = cfgs
    flat_play, flat_cloth = _flatten(play), _flatten(cloth)
    dropped = sorted(
        key for key in flat_play.keys() - flat_cloth.keys() if not key.startswith(ALLOWED_MISSING_PREFIXES)
    )
    assert not dropped, (
        f"Cloth.yaml silently omits settings Play.yaml specifies: {dropped}. The loader does no "
        "inheritance, so an omitted key falls back to the configclass default rather than to Play's "
        "value -- a difference that would not show up as a diff between runs."
    )


def test_replicate_physics_still_matches_play(cfgs) -> None:
    """A specific trap, asserted by name.

    Under Isaac Lab 2.x ``replicate_physics=False`` meant "parse each environment independently
    because its contents differ", and it is tempting to flip it to True for Newton on the theory
    that Newton needs replication for its world partitioning. It does -- but ``install_cloning``
    performs that replication itself with ``replicate_physics=True`` and leaves the scene cfg
    alone. In 3.0 this flag only tells ``isaaclab.cloner.replicate`` to drop physics contexts, so
    flipping it here would disable the replication rather than enable it.
    """
    play, newton = cfgs
    assert newton["scene"]["replicate_physics"] == play["scene"]["replicate_physics"] is False
