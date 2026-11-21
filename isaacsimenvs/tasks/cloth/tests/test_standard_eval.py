import torch

from isaacsimenvs.scripts.evaluation.cloth_standard import FoldScore


def advance(score, count=1, *, error=0.01, touching=False, failure=None):
    off = torch.tensor([False])
    reasons = {name: torch.tensor([name == failure]) for name in ("timeout", "fall", "nonfinite", "hand_far")}
    for _ in range(count):
        score.update(torch.tensor([error]), torch.tensor([0.5]), torch.tensor([touching]), reasons, off)


def test_acquisition_and_release_boundaries_and_both_tolerances():
    score = FoldScore(1, "cpu", 1 / 60)
    advance(score, 9, error=0.03)
    assert score.acquired.tolist() == [[0, 0]]
    advance(score, error=0.03)
    assert score.acquired.tolist() == [[0, 10]]
    advance(score, 179, error=0.03)
    assert score.persisted.tolist() == [[0, 0]]
    advance(score, error=0.03, failure="timeout")
    assert score.persisted.tolist() == [[0, 190]]


def test_touch_and_fold_loss_restart_persistence():
    score = FoldScore(1, "cpu", 1 / 60)
    advance(score, 100)
    advance(score, touching=True)
    advance(score, 100)
    advance(score, error=0.1)
    advance(score, 179)
    assert not score.persisted.any()
    assert score.lost.all()
    advance(score)
    assert score.persisted.all()


def test_late_acquisition_and_terminal_failure():
    for outcome in ("timeout", "fall", "nonfinite"):
        score = FoldScore(1, "cpu", 1 / 60)
        advance(score, 10, touching=True)
        advance(score, failure=outcome)
        advance(score, 180)
        row, = score.results([42])
        assert row["outcome"] == outcome
        assert row["start_id"] == 42
        assert row["scores"]["0.015"]["acquisition_s"] is not None
        assert row["scores"]["0.015"]["persistence_s"] is None
