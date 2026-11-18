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

**Capturing has a failure mode of its own, and it bit hard.** While the capture is active, fd 1/2
point at a temp file, so warnings do not reach the terminal until it unwinds. A process that dies
abruptly -- a CUDA device-side assert, a signal -- never unwinds it, and the warnings die with the
temp file. Three cloth runs were diagnosed as "0 overflow warnings" on that basis; running with
``--allow_overflow`` (capture off) showed "Per-body rigid contact buffer overflowed 87 > 64" on
the very first attempt.

So the capture **tees**: everything written is copied straight through to the real stdout as well
as scanned. A diagnostic that can destroy the diagnostic it exists to preserve is worse than none.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager

__all__ = [
    "capture_fd_output",
    "assert_no_buffer_overflow",
    "raise_if_overflowed",
    "OVERFLOW_PATTERN",
]


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

            forwarded = [0]

            def read_so_far() -> str:
                sys.stdout.flush()
                sys.stderr.flush()
                pos = tmp.tell()
                tmp.seek(0)
                data = tmp.read()
                tmp.seek(pos)
                # Tee: copy anything new straight to the real stdout, so a hard crash cannot take
                # the warnings with it. Written to the saved fd, not `sys.stdout`, which is still
                # redirected at this point.
                if len(data) > forwarded[0]:
                    os.write(saved_out, data[forwarded[0]:])
                    forwarded[0] = len(data)
                return data.decode("utf-8", errors="replace")

            yield read_so_far
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def raise_if_overflowed(captured: str, num_envs: int, where: str) -> None:
    """Raise if ``captured`` contains an overflow warning."""
    hits = [ln.strip() for ln in captured.splitlines() if OVERFLOW_PATTERN.search(ln)]
    if not hits:
        return
    unique = list(dict.fromkeys(hits))[:6]
    raise RuntimeError(
        f"Newton reported a contact-buffer overflow at num_envs={num_envs} during {where} "
        f"({len(hits)} warnings). Contacts are being dropped and every number from this run would "
        f"be measured on a scene where the hand cannot fully feel the object."
        f"\n\n  " + "\n  ".join(unique) + "\n\n"
        "Buffer budgets in Newton are GLOBAL, not per-env, so they must scale with num_envs AND "
        "with scene geometry -- a 24-segment cable generates far more triangle pairs than a "
        "12-segment one at the same env count. Check that the sized budget reaches the solver: a "
        "coupled scene builds a separate CollisionPipeline per proxy, and that one defaults to "
        "max_triangle_pairs=1e6 regardless of NewtonCfg.collision_cfg."
    )


def assert_no_buffer_overflow(env, steps: int = 20, action=None) -> None:
    """Step ``env`` briefly with fd output captured; raise if a buffer overflowed.

    A warm-up check only. **It is not sufficient on its own**: it steps with zero actions, so the
    hand is not touching the object and the contact count is at its lowest. A 24-segment cable
    passed this check and then overflowed for hundreds of steps once the policy started
    manipulating. Use `capture_fd_output` around the whole rollout as well -- see
    `eval/episodes.py`.

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

    raise_if_overflowed(captured, inner.num_envs, f"a {steps}-step warm-up")
