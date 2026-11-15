#!/usr/bin/env bash
# Re-vendor third_party/ at the pinned commits below. third_party/ is COMMITTED, as plain files with
# no nested .git, so a clone of this repo is complete on its own; run this only to bump a pin (edit
# the SHA, run, commit the diff). It replaces each third_party/<name>/ with `git archive <sha>` of
# its upstream and rewrites third_party/VENDORED.md.
#
#   scripts/vendor_third_party.sh            # every dep
#   scripts/vendor_third_party.sh aiowuji    # just one
#
# Then reinstall whatever venv uses the dep: scripts/install_isaaclab3.sh for IsaacLab, or
#   uv pip install --python .venv_deploy/bin/python -e third_party/aioflexiv -e third_party/aiowuji
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# name  url  commit  what uses it
DEPS=(
  # The Isaac Lab 3.0 tree the Newton backend's compat patches (isaacsimenvs/newton/) are written
  # against; .venv/isaaclab3 editable-installs it (scripts/install_isaaclab3.sh).
  "IsaacLab  https://github.com/isaac-sim/IsaacLab.git     a27ec0beb2345cb4618e31b8e42bc09c557ee193  .venv/isaaclab3 (Newton / VBD)"
  # The real-robot drivers deploy/ talks to (the SHAs origin/deployment pinned as submodules).
  "aioflexiv https://github.com/younghyopark/aioflexiv   adce58442f945ec74c684b78f228c6aa1d4cb8f8  deploy/ (Flexiv arm)"
  "aiowuji   https://github.com/Halo123188/aiowuji       f05afcc6fd536f4dfa7f3e6a57f928265e1d5aab  deploy/ (Wuji hand)"
)

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

for dep in "${DEPS[@]}"; do
  read -r name url commit _ <<<"${dep}"
  [[ $# -eq 0 || " $* " == *" ${name} "* ]] || continue
  echo "==> ${name}: ${url} @ ${commit}"
  # IsaacLab registers a git-lfs filter and git-lfs is not installed here. Its LFS payloads are
  # documentation media, not code, so the filter is disabled and those files stay pointer stubs.
  git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
      clone -q --filter=blob:none --no-checkout "${url}" "${TMP}/${name}"
  git -C "${TMP}/${name}" cat-file -e "${commit}^{commit}" 2>/dev/null \
    || git -C "${TMP}/${name}" fetch -q origin "${commit}"
  rm -rf "${ROOT}/third_party/${name}"
  mkdir -p "${ROOT}/third_party/${name}"
  git -C "${TMP}/${name}" archive "${commit}" | tar -x -C "${ROOT}/third_party/${name}"
  echo "    $(find "${ROOT}/third_party/${name}" -type f | wc -l) files"
done

{
  echo "# Vendored third-party source"
  echo
  echo "Plain copies of upstream repos at pinned commits, committed so a clone of this repo is"
  echo "complete. Do not edit them in place: change the pin in \`scripts/vendor_third_party.sh\` and"
  echo "rerun it. Each directory keeps its upstream LICENSE."
  echo
  echo "| dir | upstream | commit | used by |"
  echo "|---|---|---|---|"
  for dep in "${DEPS[@]}"; do
    read -r name url commit used <<<"${dep}"
    echo "| \`${name}/\` | ${url%.git} | \`${commit}\` | ${used} |"
  done
} >"${ROOT}/third_party/VENDORED.md"
echo "==> wrote third_party/VENDORED.md"
