"""Run recipe composition (tasks/recipes.py): base, DR blocks, own overrides, lookup, errors."""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from isaacsimenvs.tasks import recipes  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A repo root with runs/, dr/ and experiments/x/; returns a writer."""
    runs, dr, exp = tmp_path / "runs", tmp_path / "dr", tmp_path / "experiments" / "x"
    for d in (runs, dr, exp):
        d.mkdir(parents=True)
    monkeypatch.setattr(recipes, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(recipes, "RUNS_DIR", runs)
    monkeypatch.setattr(recipes, "DR_DIR", dr)

    def write(rel: str, data: dict):
        path = tmp_path / rel
        path.write_text(yaml.safe_dump(data))
        return path

    write("runs/base.yaml", {"task": "T", "env": {"a": 1, "dr": {"x": 0, "y": 0}}, "agent": {"lr": 1}})
    write("dr/one.yaml", {"dr": {"x": 1}})
    write("dr/two.yaml", {"dr": {"x": 2, "y": 2}})
    return write


def test_base_then_blocks_in_order_then_own_values(tree):
    tree("experiments/x/run.yaml", {"base": "base", "dr": ["two", "one"], "env": {"dr": {"y": 3}}})
    out = recipes.load_recipe("experiments/x/run.yaml")
    assert out["task"] == "T"
    assert out["env"] == {"a": 1, "dr": {"x": 1, "y": 3}}   # one after two; own y wins
    assert out["agent"] == {"lr": 1}
    assert out["dr"] == ["two", "one"]
    assert out["files"] == ["runs/base.yaml", "experiments/x/run.yaml"]


def test_base_is_found_next_to_the_recipe_before_runs(tree):
    tree("runs/mid.yaml", {"base": "base", "env": {"a": 3}})
    tree("experiments/x/mid.yaml", {"base": "base", "env": {"a": 2}})   # shadows runs/mid.yaml here
    tree("experiments/x/run.yaml", {"base": "mid"})
    out = recipes.load_recipe("experiments/x/run.yaml")
    assert out["env"]["a"] == 2
    assert out["files"] == ["runs/base.yaml", "experiments/x/mid.yaml", "experiments/x/run.yaml"]


def test_a_recipe_named_like_its_base_reads_the_one_in_runs(tree):
    tree("experiments/x/base.yaml", {"base": "base", "env": {"a": 2}})
    out = recipes.load_recipe("experiments/x/base.yaml")
    assert out["env"]["a"] == 2
    assert out["files"] == ["runs/base.yaml", "experiments/x/base.yaml"]


def test_blocks_of_the_base_are_kept(tree):
    tree("runs/mid.yaml", {"base": "base", "dr": ["one"]})
    tree("experiments/x/run.yaml", {"base": "mid", "dr": ["two"]})
    out = recipes.load_recipe("experiments/x/run.yaml")
    assert out["dr"] == ["one", "two"]
    assert out["env"]["dr"] == {"x": 2, "y": 2}


def test_name_in_runs(tree):
    assert recipes.load_recipe("base")["env"]["a"] == 1


def test_errors(tree):
    tree("runs/cycle_a.yaml", {"base": "cycle_b"})
    tree("runs/cycle_b.yaml", {"base": "cycle_a"})
    with pytest.raises(ValueError, match="cycle"):
        recipes.load_recipe("cycle_a")
    tree("runs/typo.yaml", {"base": "base", "envs": {}})
    with pytest.raises(ValueError, match="unknown top-level keys"):
        recipes.load_recipe("typo")
    tree("runs/badblock.yaml", {"base": "base", "dr": ["nope"]})
    with pytest.raises(FileNotFoundError, match="unknown DR block 'nope'"):
        recipes.load_recipe("badblock")
    with pytest.raises(FileNotFoundError, match="not found"):
        recipes.load_recipe("missing")
