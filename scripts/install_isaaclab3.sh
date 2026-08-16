#!/usr/bin/env bash
# Build the Isaac Lab 3.0 / Newton venv (.venv_isaaclab3), alongside the existing
# .venv_isaacsim rather than replacing it.
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
VENV="${ROOT}/${VENV_DIR:-.venv_isaaclab3}"
PY="${VENV}/bin/python"
LAB="${ROOT}/third_party/IsaacLab"

# Pinned rather than tracking `develop`. This is the tree the SimToolReal->Newton port in
# github.com/kushal2000/isaac_newton was developed and measured against; its compat patches are
# written against these exact APIs, and Isaac Lab 3.0 is under active development.
LAB_COMMIT="a27ec0beb2345cb4618e31b8e42bc09c557ee193"
LAB_REF="develop"

echo "==> Isaac Lab source checkout: ${LAB}"
if [[ ! -d "${LAB}/.git" ]]; then
  mkdir -p "${ROOT}/third_party"
  # git-lfs is not installed on this machine and IsaacLab registers an LFS filter. The LFS
  # payloads are documentation media, not code, so the filter is disabled outright rather than
  # skipped -- with `git-lfs` absent, leaving it registered fails the checkout entirely
  # ("git-lfs filter-process: not found / the remote end hung up unexpectedly").
  git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
      clone --branch "${LAB_REF}" https://github.com/isaac-sim/IsaacLab.git "${LAB}"
fi
git -C "${LAB}" -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
    checkout -q "${LAB_COMMIT}"
echo "    pinned at $(git -C "${LAB}" rev-parse --short HEAD) (VERSION $(cat "${LAB}/VERSION" 2>/dev/null || echo '?'))"

echo "==> Creating ${VENV} (Python 3.12)"
uv venv "${VENV}" --python 3.12 --seed

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

echo "==> Repo packages (same source tree as .venv_isaacsim)"
uv pip install --python "${PY}" -e "${ROOT}/rl_games" --no-deps
uv pip install --python "${PY}" -e "${ROOT}" --no-deps
# The Kit-free suite must pass under *both* interpreters -- that is what demonstrates the shared
# task modules are genuinely shared rather than accidentally 3.11-only.
uv pip install --python "${PY}" pytest

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
