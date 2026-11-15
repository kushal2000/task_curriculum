#!/usr/bin/env bash
# Fetch the real-robot drivers deploy/ needs into third_party/ (gitignored), at pinned SHAs.
#
# Not submodules by choice, same as third_party/IsaacLab in install_isaaclab3.sh: the pins live
# here, so the checkout is reproducible without submodule machinery in the repo. Install them into
# the deploy venv afterwards, e.g.
#   uv pip install --python .venv_deploy/bin/python -e third_party/aioflexiv -e third_party/aiowuji
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/third_party"

# name  url  commit (the SHAs origin/deployment pinned as submodules)
DEPS=(
  "aioflexiv https://github.com/younghyopark/aioflexiv adce58442f945ec74c684b78f228c6aa1d4cb8f8"
  "aiowuji   https://github.com/Halo123188/aiowuji     f05afcc6fd536f4dfa7f3e6a57f928265e1d5aab"
)

for dep in "${DEPS[@]}"; do
  read -r name url commit <<<"${dep}"
  dir="${ROOT}/third_party/${name}"
  echo "==> ${name}: ${url} @ ${commit}"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone -q "${url}" "${dir}"
  fi
  git -C "${dir}" fetch -q origin
  git -C "${dir}" checkout -q "${commit}"
  echo "    pinned at $(git -C "${dir}" rev-parse --short HEAD)"
done
