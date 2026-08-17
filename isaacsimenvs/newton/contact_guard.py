"""Fail loudly when a Newton contact buffer overflows.

Newton reports buffer overflows -- triangle pairs, rigid contacts, MJWarp's ``nefc`` -- by
printing from inside a kernel. That write goes to the process's *file descriptor 1*, not through
``sys.stdout``, so it is invisible to Python: no exception, no warning, no log record. The
simulation keeps running and silently drops the contacts that did not fit.

That failure mode cost this project a full day. Every sweep piped stdout through
``grep -E "goals/episode|lift fraction|..."`` to extract the result line, which discarded the
warnings, and 64-env runs were measured for hours with the coupler's proxy pipeline overflowing on
every step -- the proxy builds its own ``CollisionPipeline`` and defaults ``max_triangle_pairs`` to
a *global* 1e6, while the correctly sized budget went to a config object the proxy never reads.
Dropped hand-vs-cable contacts read exactly like "the policy cannot grasp a cable".

So this guard exists to convert a silent, greppable-away warning into an exception. Capturing the
fd is the only approach that works: the counters live in device arrays inside lazily built
pipelines that are not reachable from the solver object, and the message itself is the one signal
the engine reliably emits.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager

__all__ = ["capture_fd_output", "assert_no_buffer_overflow", "OVERFLOW_PATTERN"]


#: Matched against captured fd-1 text. Deliberately broad: an overflow that is not caught here is
#: worth far more than a false positive, which shows up immediately as a failed run.
OVERFLOW_PATTERN = re.compile(
    r"(buffer overflow|overflowed|exceeds? (?:the )?(?:maximum|capacity|limit)"
    r"|too many contacts|contact limit|nefc\s*>|njmax|increase\s+\w*max)",
    re.IGNORECASE,
)


@contextmanager
def capture_fd_output():
    """Capture writes to file descriptors 1 and 2, including those from native code.

    Yields a callable returning everything captured so far. ``sys.stdout`` is flushed first so
    Python-level output already queued does not land inside the capture window.
    """
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)

            def read_so_far() -> str:
                sys.stdout.flush()
                sys.stderr.flush()
                pos = tmp.tell()
                tmp.seek(0)
                data = tmp.read()
                tmp.seek(pos)
                return data.decode("utf-8", errors="replace")

            yield read_so_far
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def assert_no_buffer_overflow(env, steps: int = 20, action=None) -> None:
    """Step ``env`` briefly with fd output captured; raise if a buffer overflowed.

    Run this once after ``reset`` and before any measurement. Overflow depends on environment
    count and scene geometry, so it either happens in the first handful of steps or does not
    happen at all -- there is no need to police the whole rollout.

    Raises:
        RuntimeError: with the offending lines, the env count, and the usual cause.
    """
    import torch

    inner = env.unwrapped
    if action is None:
        action = torch.zeros(
            (inner.num_envs, int(inner.cfg.action_space)), device=inner.device
        )

    with capture_fd_output() as read_so_far:
        for _ in range(steps):
            env.step(action)
        captured = read_so_far()

    hits = [ln.strip() for ln in captured.splitlines() if OVERFLOW_PATTERN.search(ln)]
    if not hits:
        return

    unique = list(dict.fromkeys(hits))[:6]
    raise RuntimeError(
        f"Newton reported a contact-buffer overflow at num_envs={inner.num_envs} "
        f"({len(hits)} warnings in {steps} steps). Contacts are being dropped and every number "
        f"from this run would be measured on a scene where the hand cannot fully feel the object."
        f"\n\n  " + "\n  ".join(unique) + "\n\n"
        "Buffer budgets in Newton are GLOBAL, not per-env, so they must scale with num_envs. "
        "Check that the sized budget actually reaches the solver: a coupled scene builds a "
        "separate CollisionPipeline per proxy, and that one defaults to max_triangle_pairs=1e6 "
        "regardless of what NewtonCfg.collision_cfg says."
    )
