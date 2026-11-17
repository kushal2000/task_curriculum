from types import SimpleNamespace

import pytest

from isaacsimenvs.tasks.cloth.mdp.thickness_curriculum import ThicknessCurriculum


def curriculum(**overrides):
    values = dict(start=.020, target=.002, reduction=.9, success_threshold=.9, window_episodes=100)
    values.update(overrides)
    return ThicknessCurriculum(SimpleNamespace(**values))


def test_exactly_90_percent_advances_and_discards_old_geometry_evidence():
    c = curriculum()
    assert not c.observe([True]*89 + [False]*10)  # insufficient sample count
    assert c.observe([True])
    assert c.thickness == pytest.approx(.018)
    assert c.last_advance_success_rate == .9
    assert not c.recent
    assert not c.observe([True]*99)
    assert c.observe([False])
    assert c.thickness == pytest.approx(.0162)


def test_failures_count_and_recent_window_replaces_stale_outcomes():
    c = curriculum()
    assert not c.observe([False]*100)
    assert not c.observe([True]*89)
    assert c.observe([True])


def test_large_vector_batch_can_advance_only_once_and_never_below_target():
    c = curriculum(window_episodes=10)
    assert c.observe([True]*10000)
    assert c.thickness == pytest.approx(.018)
    for _ in range(100):
        c.observe([True]*10)
    assert c.thickness == .002
    assert not c.observe([True]*10)


def test_checkpoint_preserves_current_thickness_and_fresh_window():
    c = curriculum()
    c.observe([True]*100)
    c.observe([True]*10+[False]*3)
    other = curriculum()
    other.load_state_dict(c.state_dict())
    assert other.state_dict() == c.state_dict()


@pytest.mark.parametrize('overrides', [dict(target=0), dict(reduction=1), dict(success_threshold=0),
                                      dict(window_episodes=0), dict(start=.001), dict(start=float('nan'))])
def test_invalid_configuration_fails_early(overrides):
    with pytest.raises(ValueError):
        curriculum(**overrides)
