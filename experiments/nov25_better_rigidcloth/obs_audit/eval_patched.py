"""eval.py with the audit's monkeypatch (OBS_AUDIT_PATCH=bodyquat|none) installed on every env gym.make builds.

    OBS_AUDIT_PATCH=bodyquat python experiments/nov25_better_rigidcloth/obs_audit/eval_patched.py <eval.py args>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaacsimenvs.tasks.cloth as cloth_pkg  # noqa: E402
import patches_obs_audit as pa  # noqa: E402

PATCH = os.environ.get("OBS_AUDIT_PATCH", "none")
_make = cloth_pkg.make_env


def make_env_patched(cfg, render_mode=None, **kw):
    env = _make(cfg, render_mode, **kw)
    pa.apply_env(PATCH, env)
    return env


cloth_pkg.make_env = make_env_patched
print(f"[audit] eval with patch {PATCH}", flush=True)
from isaacsimenvs.scripts.evaluation.eval import main  # noqa: E402

main()
