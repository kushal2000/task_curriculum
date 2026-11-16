"""Wall-clock stamps for env setup: ``[setup_t] +<since first stamp>s (+<since last>s) <label>``.

For finding where scene construction spends its time (experiments/nov15_setup_time/). Printing only;
stdlib only.
"""

from __future__ import annotations

import time

_T0: float | None = None
_LAST: float | None = None


def stamp(label: str) -> float:
    """Print the time since the first and the previous stamp; return the time since the previous."""
    global _T0, _LAST
    now = time.perf_counter()
    if _T0 is None:
        _T0 = _LAST = now
    dt = now - _LAST
    print(f"[setup_t] +{now - _T0:8.2f}s (+{dt:7.2f}s) {label}", flush=True)
    _LAST = now
    return dt
