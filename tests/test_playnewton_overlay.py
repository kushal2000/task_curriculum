"""PlayNewton.yaml must not drift from Play.yaml except where it means to.

``hydra_task_config_with_yaml`` does a flat ``yaml.safe_load`` of one file: no ``defaults:``
merge, no inheritance. So the Newton task YAML is a full copy of the PhysX one, and nothing stops
someone tuning a reward scale or a termination threshold in one and not the other.

That failure would be invisible and expensive. The two envs exist to be compared; if their task
definitions differ, a phase-3 gap stops being about physics and there is no signal in the diff
pointing at the cause. This test turns it into a failing assertion instead.

Kit-free by construction -- it reads YAML and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CFG = Path(__file__).resolve().parents[1] / "isaacsimenvs" / "cfg" / "task"

#: Top-level keys PlayNewton is allowed to add, with why.
ALLOWED_EXTRA = {
    "newton",        # MJWarp solver knobs; no PhysX counterpart
    "use_gravcomp",  # per-body gravity route, Newton-only
    "meshify",       # convex-mesh colliders, Newton-only
}

#: Dotted paths PlayNewton is allowed to omit.
#:
#: `sim.physx` must be absent: Isaac Lab 3.0 renamed the field to `physics`, and the overlay goes
#: through `configclass.from_dict`, which raises `KeyError: Key not found under namespace:
#: /sim/physx` on an unknown key. (compat.py shims the *constructor* keyword, not dict overlays.)
#: It would be inert anyway -- `play_newton_env.py` replaces `cfg.sim.physics` with a NewtonCfg.
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
    newton = yaml.safe_load((CFG / "PlayNewton.yaml").read_text())
    return play, newton


def test_playnewton_adds_only_allowlisted_top_level_keys(cfgs) -> None:
    play, newton = cfgs
    extra = set(newton) - set(play)
    assert extra <= ALLOWED_EXTRA, (
        f"PlayNewton.yaml adds unexpected top-level keys: {sorted(extra - ALLOWED_EXTRA)}. "
        "If the addition is intentional, add it to ALLOWED_EXTRA with a reason."
    )


def test_playnewton_drops_no_keys(cfgs) -> None:
    play, newton = cfgs
    missing = set(play) - set(newton)
    assert not missing, (
        f"PlayNewton.yaml is missing top-level keys present in Play.yaml: {sorted(missing)}. "
        "The loader does no inheritance, so a missing section silently falls back to the "
        "configclass default rather than to Play's value."
    )


def test_shared_task_definition_is_identical(cfgs) -> None:
    """Every shared leaf must match: rewards, obs/state lists, terminations, reset, DR, assets.

    This is the assertion that makes a cross-backend comparison meaningful.
    """
    play, newton = cfgs
    flat_play = _flatten({k: v for k, v in play.items()})
    flat_newton = _flatten({k: v for k, v in newton.items() if k not in ALLOWED_EXTRA})

    differing = {
        key: (flat_play[key], flat_newton[key])
        for key in flat_play.keys() & flat_newton.keys()
        if flat_play[key] != flat_newton[key]
    }
    assert not differing, (
        "Play.yaml and PlayNewton.yaml disagree on shared task settings, so the two backends "
        f"would no longer be running the same task: {differing}"
    )

    dropped = sorted(
        key
        for key in flat_play.keys() - flat_newton.keys()
        if not key.startswith(ALLOWED_MISSING_PREFIXES)
    )
    assert not dropped, (
        f"PlayNewton.yaml silently omits settings Play.yaml specifies: {dropped}. The loader "
        "does no inheritance, so an omitted key falls back to the configclass default rather "
        "than to Play's value -- a difference that would not show up as a diff between runs."
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
