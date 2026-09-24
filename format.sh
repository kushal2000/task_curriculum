#!/bin/bash
# Format the codebase: Ruff (lint-fix imports + format) over the Python packages,
# and xmllint pretty-print over URDFs.
set -e

PKGS="isaacsimenvs tests"

for target in $PKGS; do
  echo "Ruff: $target"
  ruff check --extend-select I --fix "$target"
  ruff format "$target"
done

echo "xmllint: assets/**/*.urdf"
# g1_wuji2_description is fetched verbatim from upstream (scripts/g1_wuji/fetch_assets.sh); leave it be.
find assets -name "*.urdf" -type f -not -path "assets/g1_wuji2_description/*" \
  -exec xmllint --format "{}" --output "{}" \;

echo "✅ Formatting complete."
