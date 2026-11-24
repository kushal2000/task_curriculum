"""The compiled PointNet trains like the eager one under fp16 autocast (torch 2.7's compiled amax
backward did not: non-finite gradients, every update skipped)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast fp16 on the GPU")


def test_compiled_gradients_finite_and_equal():
    from isaacsimenvs.scripts.distillation.student import PointNet

    torch.manual_seed(0)
    x = torch.randn(4096, 64, 3, device="cuda")
    m = torch.rand(4096, 64, device="cuda") > 0.1
    eager = PointNet(3, widths=(32, 64, 128)).cuda()
    comp = PointNet(3, widths=(32, 64, 128), compile=True).cuda()
    comp.load_state_dict(eager.state_dict())
    grads = []
    for net in (eager, comp):
        with torch.autocast("cuda", dtype=torch.float16):
            f = net(x, m)
        (f.float() ** 2).sum().backward()
        grads.append([p.grad for p in net.parameters()])
    for a, b in zip(*grads):
        assert torch.isfinite(b).all()
        assert ((a - b).norm() / (a.norm() + 1e-12)) < 1e-2
