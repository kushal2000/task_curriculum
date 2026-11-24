"""``mdp/surface_match``: the four half folds of a square sheet match its goal at any placement;
an unfolded sheet, a diagonal fold and a third fold do not; the in-place distance sees the shift."""

import math

import pytest
import torch

from isaacsimenvs.tasks.cloth.mdp import surface_match as sm

R = 7


def _sheet(kind: str, a: torch.Tensor, r: int = R) -> torch.Tensor:
    """(B, r*r, 3) a square sheet of side ``a`` (B,), flat or folded (mid-plane hinge: ply gap 0)."""
    t = torch.linspace(0, 1, r)
    u, v = torch.meshgrid(t, t, indexing="ij")
    u, v = u.reshape(1, -1).expand(a.shape[0], -1).clone(), v.reshape(1, -1).expand(a.shape[0], -1).clone()
    if kind in ("lr", "rl", "tb", "bt"):
        s = u if kind in ("lr", "rl") else v
        s = torch.where(s < 0.5, 1 - s, s) if kind in ("lr", "tb") else torch.where(s > 0.5, 1 - s, s)
        u, v = (s, v) if kind in ("lr", "rl") else (u, s)
    elif kind == "third":
        u = torch.where(u < 1 / 3, 2 / 3 - u, u)
    elif kind == "diag":
        top = u + v < 1
        u, v = torch.where(top, 1 - v, u), torch.where(top, 1 - u, v)
    return torch.stack((u * a[:, None], v * a[:, None], torch.zeros_like(u)), dim=-1)


def _place(p, yaw, t):
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    x, y = p[..., 0], p[..., 1]
    return torch.stack((c * x - s * y + t[:, None, 0], s * x + c * y + t[:, None, 1], p[..., 2] + t[:, None, 2]), dim=-1)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    b = 256
    a = 0.10 + 0.10 * torch.rand(b)
    yaw = 2 * math.pi * torch.rand(b)
    t = torch.cat(((torch.rand(b, 2) - 0.5) * 0.3, torch.full((b, 1), 0.6)), dim=1)
    goal = _place(_sheet("lr", a), torch.zeros(b), torch.tensor([0.0, 0.0, 0.6]).expand(b, 3))
    return a, yaw, t, goal


@pytest.mark.parametrize("kind", ["lr", "rl", "tb", "bt"])
def test_every_half_fold_matches_at_any_placement(setup, kind):
    a, yaw, t, goal = setup
    d = sm.aligned_distance(_place(_sheet(kind, a), yaw, t), goal, R)
    assert float(d["max"].max()) < 1e-3


@pytest.mark.parametrize("kind", ["flat", "third", "diag"])
def test_wrong_shapes_stay_above_a_tenth_of_the_side(setup, kind):
    a, yaw, t, goal = setup
    d = sm.aligned_distance(_place(_sheet(kind, a), yaw, t), goal, R)
    assert bool((d["max"] > 0.1 * a).all())


def test_lifted_sheet_does_not_match(setup):
    a, yaw, t, goal = setup
    t = t.clone()
    t[:, 2] += 0.05
    d = sm.aligned_distance(_place(_sheet("lr", a), yaw, t), goal, R)
    assert bool((d["max"] > 0.045).all())


def test_in_place_distance_sees_the_shift(setup):
    a, _yaw, _t, goal = setup
    shift = torch.tensor([0.03, 0.0, 0.0])
    assert float(sm.in_place_distance(goal, goal, R).max()) < 1e-6
    d = sm.in_place_distance(goal + shift, goal, R)
    assert torch.allclose(d, torch.full_like(d, 0.03), atol=1e-4)


def test_dense_grid_keeps_the_corners_and_interpolates():
    p = torch.randn(3, R * R, 3)
    g = sm.dense_grid(p, R, 13).view(3, 13, 13, 3)
    q = p.view(3, R, R, 3)
    assert torch.allclose(g[:, ::2, ::2], q, atol=1e-6)
    assert torch.allclose(g[:, 1, 0], 0.5 * (q[:, 0, 0] + q[:, 1, 0]), atol=1e-6)
