#!/usr/bin/env bash
# Fetch the Flexiv Rizon 4s arm and the Wuji Hand 2 into assets/flexiv_wuji/.
#
#   flexiv_description/  github.com/flexivrobotics/flexiv_description: the Rizon 4s xacro expanded
#                        to urdf/Rizon4s.urdf with upstream's own create_urdf.py, and its meshes.
#   wuji_hand2/          github.com/wuji-technology/wuji-description, hand2/hand2_beta2/body: the
#                        vendor URDFs (with and without the wrist mount), meshes, and MJCF (the
#                        source of the provisional hand gains in isaacsimenvs/tasks/play/robots.py,
#                        which tests/test_robot_specs.py checks against it). Beta 2
#                        at the commit wuji-hand-teleop's wujihand_urdf was copied from, so the
#                        hand matches assets/g1_wuji2_description/.
#
# Both are fetched verbatim at pinned commits. The left-hand robot's files are committed (the
# training-time pose viewer loads them from raw GitHub); the right hand is gitignored. Mesh paths in the
# generated Rizon4s.urdf are rewritten relative to the file, so the tree can move. Last, the two
# are joined into rizon4s_left_wuji.urdf (flexiv_wuji.py, at its MOUNT_*), the model the
# Isaacsimenvs-PlayFlexivWuji-Direct-v0 task loads.
#
#   scripts/flexiv_wuji/fetch_assets.sh
#   FLEXIV_SHA=<sha> WUJI_SHA=<sha> scripts/flexiv_wuji/fetch_assets.sh
set -euo pipefail

FLEXIV_REPO=https://github.com/flexivrobotics/flexiv_description.git
FLEXIV_SHA=${FLEXIV_SHA:-8b8452105c5faf088f456db18b0e2581b16fc630}
WUJI_REPO=https://github.com/wuji-technology/wuji-description.git
WUJI_SHA=${WUJI_SHA:-b13f7d52b23cb79e35357303c72b7f61f1d2fda2}
WUJI_SRC=hand2/hand2_beta2/body

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEST=$ROOT/assets/flexiv_wuji

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# sparse_fetch <repo> <sha> <dir> <paths...>
sparse_fetch() {
    local repo=$1 sha=$2 dir=$3
    shift 3
    git init -q "$dir"
    git -C "$dir" remote add origin "$repo"
    git -C "$dir" sparse-checkout set "$@"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$dir" checkout -q FETCH_HEAD
}

sparse_fetch "$FLEXIV_REPO" "$FLEXIV_SHA" "$tmp/flexiv_description" \
    config scripts urdf meshes/Rizon4s
sparse_fetch "$WUJI_REPO" "$WUJI_SHA" "$tmp/wuji" "$WUJI_SRC/urdf" "$WUJI_SRC/meshes" "$WUJI_SRC/mjcf"

# Upstream's xacro resolves $(find flexiv_description) through ROS 2's ament index. Stand in for
# it with a one-function module, so the expansion needs only `xacro` and `pyyaml` from PyPI.
mkdir -p "$tmp/shim/ament_index_python"
touch "$tmp/shim/ament_index_python/__init__.py"
cat > "$tmp/shim/ament_index_python/packages.py" <<'EOF'
import os


def get_package_share_directory(name):
    return os.environ["FLEXIV_DESCRIPTION_DIR"]
EOF
(
    cd "$tmp/flexiv_description"
    FLEXIV_DESCRIPTION_DIR=$PWD PYTHONPATH=$tmp/shim \
        uv run -q --no-project --with xacro --with pyyaml \
        python scripts/create_urdf.py --robot_type Rizon4s --output_path @FLEXIV@
)
urdf=$tmp/flexiv_description/urdf/Rizon4s.urdf
# create_urdf.py prints its errors and exits 0, so check for the output instead.
[[ -f $urdf ]] || { echo "xacro expansion did not write $urdf" >&2; exit 1; }
sed -i 's|@FLEXIV@/|../|g' "$urdf"
# Upstream's get_inertias macro (urdf/common/flexiv_common.xacro) emits <origin>, <mass> and
# <inertia> as direct children of <link>, without the <inertial> element URDF requires, so every
# parser ignores them: Isaac Sim falls back to default masses ("No mass specified for link link1")
# and yourdfpy reads mass None. Wrap them.
python3 - "$urdf" <<'EOF'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
tree = ET.parse(path)
wrapped = 0
for link in tree.getroot().iter("link"):
    loose = [c for c in link if c.tag in ("origin", "mass", "inertia")]
    if not loose:
        continue
    if link.find("inertial") is not None:
        sys.exit(f"{link.get('name')}: has both <inertial> and loose inertia elements")
    inertial = ET.Element("inertial")
    for child in loose:
        link.remove(child)
        inertial.append(child)
    link.insert(0, inertial)
    wrapped += 1
if wrapped == 0:
    sys.exit("no loose inertia elements found; upstream may have fixed get_inertias -- drop this step")
ET.indent(tree, space="  ")
tree.write(path, encoding="utf-8", xml_declaration=True)
print(f"wrapped loose inertia elements in <inertial> on {wrapped} links")
EOF

rm -rf "$DEST"
mkdir -p "$DEST/flexiv_description/urdf" "$DEST/wuji_hand2"
cp "$urdf" "$DEST/flexiv_description/urdf/"
cp -r "$tmp/flexiv_description/meshes" "$DEST/flexiv_description/"
cp -r "$tmp/wuji/$WUJI_SRC/urdf" "$tmp/wuji/$WUJI_SRC/meshes" "$tmp/wuji/$WUJI_SRC/mjcf" "$DEST/wuji_hand2/"

printf '%s %s\n%s %s\n' "$FLEXIV_REPO" "$FLEXIV_SHA" "$WUJI_REPO" "$WUJI_SHA" > "$DEST/UPSTREAM"
python3 "$ROOT/scripts/flexiv_wuji/flexiv_wuji.py"
echo "Rizon 4s @ ${FLEXIV_SHA:0:12} + Wuji Hand 2 @ ${WUJI_SHA:0:12} -> ${DEST#$ROOT/}"
