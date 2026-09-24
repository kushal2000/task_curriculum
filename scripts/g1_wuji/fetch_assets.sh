#!/usr/bin/env bash
# Fetch the Unitree G1 + dual Wuji Hand 2 model package into assets/g1_wuji2_description/.
#
# Source is `src/g1_wuji2_description` of github.com/MSC-Wuji-Teleop/wuji-hand-teleop: MJCF
# (fixed and floating base), a URDF mirror, and every mesh they reference, for both the 29-DoF
# and 23-DoF G1. Upstream marks these files as generated, so they are fetched verbatim at a
# pinned commit rather than vendored and edited here. The destination is gitignored.
#
# Beyond the verbatim copy, one file per variant is added: scene_g1_<n>_wuji2_fixed.xml, the
# upstream floor-and-lighting scene wrapped around the fixed-base model instead of the floating
# one. It must sit next to the model because MuJoCo resolves mesh paths against the top file.
#
#   scripts/g1_wuji/fetch_assets.sh            # pinned SHA below
#   WUJI_SHA=<sha> scripts/g1_wuji/fetch_assets.sh
set -euo pipefail

REPO=https://github.com/MSC-Wuji-Teleop/wuji-hand-teleop.git
SHA=${WUJI_SHA:-59c6f67e204fbe8f513adb0d03b9e1e6f6e763ad}
SRC=src/g1_wuji2_description

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEST=$ROOT/assets/g1_wuji2_description

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git -C "$tmp" init -q
git -C "$tmp" remote add origin "$REPO"
git -C "$tmp" sparse-checkout set "$SRC"
# The upstream repo keeps prebuilt SDK binaries in LFS; none are under $SRC, so skip smudging.
GIT_LFS_SKIP_SMUDGE=1 git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$SHA"
GIT_LFS_SKIP_SMUDGE=1 git -C "$tmp" checkout -q FETCH_HEAD

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -r "$tmp/$SRC" "$DEST"

for n in 23 29; do
    sed "s|<include file=\"g1_${n}_wuji2.xml\"/>|<include file=\"g1_${n}_wuji2_fixed.xml\"/>|" \
        "$DEST/scene_g1_${n}_wuji2.xml" > "$DEST/scene_g1_${n}_wuji2_fixed.xml"
    grep -q "g1_${n}_wuji2_fixed.xml" "$DEST/scene_g1_${n}_wuji2_fixed.xml" \
        || { echo "scene include not rewritten for ${n}-DoF; upstream scene layout changed" >&2; exit 1; }
done

printf '%s\n%s\n' "$REPO" "$SHA" > "$DEST/UPSTREAM"
echo "G1 + Wuji Hand 2 models @ ${SHA:0:12} -> ${DEST#$ROOT/}"
