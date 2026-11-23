"""``DelayLine.push_and_read``'s flush is masked (no ``.any()`` host sync): same result as indexing."""

import torch

from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine


def test_flush_masked_matches_indexed():
    torch.manual_seed(0)
    n, d = 8, 3
    a, b = DelayLine(n, 4, (d,), "cpu"), DelayLine(n, 4, (d,), "cpu")
    lag = torch.randint(0, 5, (n,))
    for t in range(12):
        v = torch.randn(n, d)
        flush = torch.rand(n) < 0.3
        out_a = a.push_and_read(v, flush, lag)
        # reference: the old indexed flush
        b.ptr = (b.ptr + 1) % b.size
        b.buffer[b.ptr] = v
        if bool(flush.any()):
            b.buffer[:, flush] = v[flush].unsqueeze(0)
        out_b = b.buffer[(b.ptr - lag.clamp(0, b.size - 1)) % b.size, b.env]
        assert torch.equal(out_a, out_b), t
