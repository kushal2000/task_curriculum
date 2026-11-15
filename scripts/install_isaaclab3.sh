#!/usr/bin/env bash
# Build the Isaac Lab 3.0 / Newton venv (.venv/isaaclab3), alongside the existing
# .venv/isaacsim rather than replacing it.
#
# Isaac Lab 2.3 needs Python 3.11 and Isaac Lab 3.0 needs 3.12, so the two cannot share an
# interpreter. Both venvs editable-install this same source tree, so `isaacsimenvs.eval`,
# `isaacsimenvs.tasks.play.utils.*` and `rl_games` resolve to one copy of the code from either
# one -- which is what makes a phase-1 number and a phase-3 number comparable.
#
#   Python 3.12 | torch cu126 | Isaac Lab 3.0 from source (brings newton + warp)
#
# Isaac Sim is NOT installed: Newton runs kit-less and the eval is headless. Set
# WITH_ISAACSIM=1 to add it (needed only for Kit-hosted backends or rendering).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/${VENV_DIR:-.venv/isaaclab3}"
PY="${VENV}/bin/python"
LAB="${ROOT}/third_party/IsaacLab"

# Isaac Lab is vendored in the repo at the commit the Newton compat patches (isaacsimenvs/newton/)
# are written against -- see third_party/VENDORED.md; bump it with scripts/vendor_third_party.sh.
echo "==> Isaac Lab source: ${LAB} (VERSION $(cat "${LAB}/VERSION" 2>/dev/null || echo '?'))"
[[ -f "${LAB}/isaaclab.sh" ]] || { echo "missing ${LAB}: it is committed, check the clone" >&2; exit 1; }

echo "==> Creating ${VENV} (Python 3.12)"
# A Slurm requeue (preemption) reruns this from the top over a half-built venv, which uv refuses
# without --clear. Only then: a manual rerun keeps refusing rather than wiping a working venv.
CLEAR=()
[[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]] && CLEAR=(--clear)
uv venv "${VENV}" --python 3.12 --seed "${CLEAR[@]}"

# Installed mainly so the venv has a working torch before Isaac Lab's own installer runs.
# NOTE: `isaaclab.sh -i` below replaces this with its own pin -- the venv ends up on
# torch 2.11.0+cu128, not cu126. Verified after the fact rather than assumed. That is fine on
# this box (RTX 6000 Ada, sm_89, covered by both builds) and is also what a Blackwell card would
# need, so the override is left alone rather than fought.
echo "==> PyTorch (baseline; isaaclab.sh will re-pin it)"
uv pip install --python "${PY}" torch torchvision --index-url https://download.pytorch.org/whl/cu126

if [[ "${WITH_ISAACSIM:-0}" == "1" ]]; then
  echo "==> Isaac Sim (optional; Kit-hosted backends / rendering only)"
  uv pip install --python "${PY}" "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
else
  echo "==> Skipping Isaac Sim (kit-less Newton; set WITH_ISAACSIM=1 to include it)"
fi

# Isaac Lab 3.0 is source-only -- there is no `isaaclab==3.x` on PyPI, and no `isaaclab-newton`
# package at all. Its own installer builds every extension in the tree, including isaaclab_newton
# and isaaclab_contrib (the coupling/deformable modules the cable task needs).
#
# `isaaclab.sh -i` shells out to `sudo apt-get install cmake build-essential` unless cmake is
# already on PATH. It is here (3.28.3), so the apt step is skipped and no sudo is required.
echo "==> Isaac Lab 3.0 from source (brings newton + warp)"
command -v cmake >/dev/null || {
  echo "cmake not on PATH; isaaclab.sh will try to apt-get install it and needs sudo" >&2
  exit 1
}
(cd "${LAB}" && VIRTUAL_ENV="${VENV}" ./isaaclab.sh -i)

echo "==> Repo packages (same source tree as .venv/isaacsim)"
uv pip install --python "${PY}" -e "${ROOT}/rl_games" --no-deps
uv pip install --python "${PY}" -e "${ROOT}" --no-deps
# The Kit-free suite must pass under *both* interpreters -- that is what demonstrates the shared
# task modules are genuinely shared rather than accidentally 3.11-only.
uv pip install --python "${PY}" pytest
# viser's URDF loader; Isaac Lab brings viser but not yourdfpy. Used by scripts/flexiv_wuji/.
uv pip install --python "${PY}" yourdfpy

echo "==> Verifying"
"${PY}" - <<'PY'
import torch
print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available())
for mod in ("isaaclab", "isaaclab_newton", "isaaclab_contrib", "newton", "warp", "rl_games"):
    try:
        m = __import__(mod)
        print(f"{mod:17}", getattr(m, "__version__", "ok"))
    except Exception as exc:
        print(f"{mod:17} MISSING -- {type(exc).__name__}: {exc}")
PY
echo "==> ${VENV} ready"
