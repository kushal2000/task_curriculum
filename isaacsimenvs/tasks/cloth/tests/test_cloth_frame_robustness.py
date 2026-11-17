"""A degenerate point cloud must never abort the other environments' velocity fits."""
import pytest
import torch

from isaacsimenvs.tasks.cloth.mdp.geometry.fold import cloth_frame


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def _old_fit(points, velocity):
    points, velocity = points.double(), velocity.double()
    rel, dv = points-points.mean(1, keepdim=True), velocity-velocity.mean(1, keepdim=True)
    eye = torch.eye(3, dtype=torch.float64, device=points.device)
    inertia = rel.square().sum((1, 2))[:, None, None]*eye-torch.einsum('npi,npj->nij', rel, rel)
    momentum = torch.cross(rel, dv, dim=-1).sum(1)
    return torch.linalg.solve(inertia+1e-9*eye, momentum[..., None])[..., 0]


def test_normal_cloud_matches_original_regularized_least_squares(device):
    torch.manual_seed(837)
    points = torch.randn(128, 49, 3, device=device)*torch.tensor([.08, .12, .002], device=device)
    points += torch.rand(128, 1, 3, device=device)*200
    velocity = torch.randn_like(points)
    r = torch.eye(3, device=device).expand(128, 3, 3)
    pos, _, lin, angular = cloth_frame(points, velocity, r)
    torch.testing.assert_close(pos, points.mean(1))
    torch.testing.assert_close(lin, velocity.mean(1))
    torch.testing.assert_close(angular.double(), _old_fit(points, velocity), atol=4e-5, rtol=3e-5)


def test_mixed_normal_collapsed_collinear_huge_and_nonfinite_batch(device):
    torch.manual_seed(27)
    points = torch.randn(8, 49, 3, device=device)*.1
    points[1] = 3.  # no observable angular velocity
    line = torch.linspace(-1., 1., 49, device=device)
    points[2] = line[:, None]*torch.tensor([1., 2., 3.], device=device)
    points[3] = points[2]*1e12  # old absolute ridge rounds away in singular inertia
    points[4] *= 1e20  # old products overflow despite finite inputs
    omega = torch.tensor([.3, -.7, 1.2], device=device).expand(8, 3)
    velocity = torch.cross(omega[:, None].expand_as(points), points-points.mean(1, keepdim=True), dim=-1)+.25
    points[5, 0, 0] = float('nan')
    velocity[6, 0, 1] = float('inf')
    points[7] = torch.finfo(torch.float32).max  # finite raw input, unrepresentable centroid reduction
    _, _, lin, angular = cloth_frame(points, velocity, torch.eye(3, device=device).expand(8, 3, 3))
    assert torch.isfinite(angular[:5]).all()
    torch.testing.assert_close(angular[0], omega[0], atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(angular[1], torch.zeros(3, device=device))
    torch.testing.assert_close(angular[4], omega[4], atol=4e-6, rtol=4e-6)
    # An axial spin is unobservable for a line: compare reconstructed velocities,
    # not the arbitrary angular component parallel to that line.
    for i in (2, 3):
        rel = (points[i]-points[i].mean(0))/points[i].abs().max()
        expected = torch.cross(omega[i].expand_as(rel), rel, dim=-1)
        actual = torch.cross(angular[i].expand_as(rel), rel, dim=-1)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    assert torch.isnan(angular[5:]).all()
    assert torch.isfinite(lin[:6]).all()


def test_failed_solver_row_becomes_nan_without_affecting_neighbors(monkeypatch):
    original = torch.linalg.solve_ex
    def failing_row(a, b, **kwargs):
        solution, info = original(a, b, **kwargs)
        info[1] = 1
        return solution, info
    monkeypatch.setattr(torch.linalg, 'solve_ex', failing_row)
    points = torch.randn(3, 49, 3)
    velocity = torch.randn_like(points)
    _, _, _, angular = cloth_frame(points, velocity, torch.eye(3).expand(3, 3, 3))
    assert torch.isfinite(angular[[0, 2]]).all()
    assert torch.isnan(angular[1]).all()
