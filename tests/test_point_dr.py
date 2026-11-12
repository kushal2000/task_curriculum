"""Point-level observation DR (isaacsimenvs/tasks/play/utils/point_dr.py): Kit-free, CPU."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import load_module

pdr = load_module("isaacsimenvs/tasks/play/utils/point_dr.py", "point_dr")


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def _random_rot(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen, dtype=torch.float64), dim=-1)
    return _quat_to_mat(q)


@pytest.mark.parametrize("planar", [False, True])
def test_refit_recovers_rigid_transform(planar):
    gen = torch.Generator().manual_seed(0)
    n = 256
    # Play's keypoints span a volume; the cloth's corners are planar -- both must fit exactly.
    rest = torch.tensor([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=torch.float64) * 0.03
    if planar:
        rest = torch.tensor([[0.01, -0.05, 0], [0.01, 0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0]],
                            dtype=torch.float64)
        rest = rest - rest.mean(dim=0)
    rot = _random_rot(n, gen)
    # Include exact 180-degree flips about x and y: a completed fold.
    rot[0] = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    rot[1] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64))
    t = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    pts = t.unsqueeze(1) + torch.einsum("nij,kj->nki", rot, rest)

    kp, quat = pdr.refit_points(rest, pts)
    # 1e-6: the fit's 1e-8 ridge on H, which is ~1e-3 for 3-5 cm offsets.
    assert torch.allclose(kp, pts, atol=1e-6)
    assert torch.allclose(_quat_to_mat(quat), rot, atol=1e-4)  # planar: ridge-limited
    assert torch.allclose(quat.norm(dim=-1), torch.ones(n, dtype=torch.float64))


def test_refit_degenerate_is_finite():
    rest = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    pts = torch.zeros(3, 4, 3)  # all points coincident
    kp, quat = pdr.refit_points(rest, pts)
    assert torch.isfinite(kp).all() and torch.isfinite(quat).all()


def test_push_and_read_lag_and_flush():
    q = torch.zeros(2, 4, 1)
    start = torch.tensor([True, True])
    for step in range(6):
        vals = torch.full((2, 1), float(step))
        q, out = pdr.push_and_read(q, vals, torch.tensor([0, 3]), start)
        start = torch.tensor([False, False])
        # lag 0 reads now; lag 3 reads 3 steps ago, or the first value while the queue is young.
        assert out[0, 0] == step
        assert out[1, 0] == max(step - 3, 0)
    q, out = pdr.push_and_read(q, torch.full((2, 1), 9.0), torch.tensor([3, 3]), torch.tensor([True, False]))
    assert out[0, 0] == 9.0  # flushed: a new episode never reads the old one
    assert out[1, 0] == 3.0


def test_simple_lags_cover_inclusive_range():
    torch.manual_seed(0)
    lags = pdr.simple_lags(20000, (2, 10), "cpu")
    assert int(lags.min()) == 2 and int(lags.max()) == 10


def _run_realistic(n=512, steps=200, base=(2, 5), jitter=2, max_lag=10, ks=(1, 2, 3), episode=50):
    torch.manual_seed(1)
    st = pdr.RealisticLags(n, "cpu")
    lags, frames, starts = [], [], []
    for t in range(steps):
        start = torch.full((n,), t % episode == 0)
        if bool(start[0]):
            st.draw_episode(torch.arange(n), base, ks)
        lag = st.step(start, jitter, max_lag)
        lags.append(lag.clone())
        frames.append((st.t - lag).clone())
        starts.append(start.clone())
    return st, torch.stack(lags), torch.stack(frames), torch.stack(starts)


def test_realistic_bounds_and_no_step_back():
    st, lags, frames, starts = _run_realistic()
    assert int(lags.min()) >= 0 and int(lags.max()) <= 10
    df = frames[1:] - frames[:-1]
    assert bool((df[~starts[1:]] >= 0).all()), "the read frame stepped back in time"


def test_realistic_reads_only_camera_frames():
    # max_lag is never binding here (base 5 + jitter 2 + k-1 2 = 9 < 10), so every read frame is a
    # capture time phase + m k (negative ones read the flushed first observation).
    st, lags, frames, starts = _run_realistic(episode=10_000)
    assert bool((((frames - st.phase) % st.k) == 0).all())


def test_realistic_without_jitter_holds_k_steps():
    # No jitter: a new frame lands exactly every k steps and the lag cycles base .. base + k - 1.
    st, lags, frames, starts = _run_realistic(base=(3, 3), jitter=0, episode=10_000)
    for e in range(64):
        k = int(st.k[e])
        change = torch.nonzero(frames[1:, e] != frames[:-1, e]).flatten()
        assert bool(((change[1:] - change[:-1]) == k).all())
        assert set(lags[20:, e].tolist()) == set(range(3, 3 + k))


def test_realistic_mean_lag_tracks_base():
    torch.manual_seed(2)
    n = 4096
    st = pdr.RealisticLags(n, "cpu")
    st.draw_episode(torch.arange(n), (3, 3), (1,))
    lag = None
    total = torch.zeros(n)
    for t in range(400):
        lag = st.step(torch.full((n,), t == 0), 2, 10)
        if t >= 50:
            total += lag
    mean = float((total / 350).mean())
    # k = 1, base 3, jitter +-2: in-order delivery holds a late frame's successors back, so the
    # mean sits a little above the base.
    assert 2.5 < mean < 4.5, mean


def test_realistic_k_and_base_draws():
    torch.manual_seed(3)
    st = pdr.RealisticLags(30000, "cpu")
    st.draw_episode(torch.arange(30000), (2, 5), (1, 2, 3))
    assert set(st.k.unique().tolist()) == {1, 2, 3}
    assert set(st.base.unique().tolist()) == {2, 3, 4, 5}
    assert bool((st.phase < st.k).all()) and bool((st.phase >= 0).all())
    assert math.isclose(float((st.k == 2).float().mean()), 1 / 3, abs_tol=0.02)
