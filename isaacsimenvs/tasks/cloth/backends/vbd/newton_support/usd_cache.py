"""Content-addressed URDF -> USD cache, so the kit-less Newton stack can load baked assets.

``scene_utils._convert_urdf_to_usd`` drives ``isaaclab.sim.converters.UrdfConverter``, which needs
the Kit URDF-importer extension. Newton runs kit-less and has no Kit, so conversion cannot happen
there at all. The split is therefore: **bake once under ``.venv/isaacsim``, read under
``.venv/isaaclab3``.**

There is a second, independent reason to do it this way even where Kit *is* available. Both
backends then consume byte-identical geometry, converted by one importer. Without that, a
difference between the PhysX and Newton numbers is confounded with a difference in how their USDs
were produced, and is unattributable.

Populate (Isaac Sim venv)::

    OMNI_KIT_ACCEPT_EULA=YES .venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache \
        --populate --num_assets_per_type 4

Consume: the VBD scene calls :func:`lookup` for each robot and table asset.

The key is a hash of URDF *text* plus the conversion options, never a path -- the procedural tool
URDFs are written to a fresh ``tempfile.mkdtemp`` on every launch, so paths are useless as
identity while content is stable. Verified stable across the two interpreters: the 48-URDF pool
for ``num_assets_per_type=4`` hashes identically under numpy 1.26 (3.11) and numpy 2.5 (3.12),
because ``generate_objects`` seeds the legacy ``RandomState`` stream, which is compatibility-
guaranteed across numpy versions.

Adapted from ``simtoolreal_newton.upstream`` in github.com/kushal2000/isaac_newton @ beb9efb.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from isaacsimenvs import REPO_ROOT

DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "usd"

_MANIFEST_NAME = "manifest.json"

#: Conversion options that change the produced USD. Anything not listed here must not affect
#: output, or two different assets would collide on one key.
_KEYED_KWARGS = ("fix_base", "self_collision", "replace_cylinders_with_capsules")


def cache_key(urdf_text: str, kwargs: dict) -> str:
    """Content address for one conversion: URDF text plus the options that affect its output."""
    relevant = {k: kwargs.get(k) for k in _KEYED_KWARGS}
    # joint_drive is a config object; fold in its repr so differing drives key differently.
    relevant["joint_drive"] = repr(kwargs.get("joint_drive"))
    payload = json.dumps(relevant, sort_keys=True) + "\n" + urdf_text
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _entry_dir(cache_dir: Path, key: str) -> Path:
    return cache_dir / key


def lookup(asset_path, usd_work_dir, *, cache_dir: Path = DEFAULT_CACHE_DIR, **kwargs) -> str:
    """The cached conversion of ``asset_path`` under ``kwargs`` (the converter options), copied into
    ``usd_work_dir``; raises on a miss. The kit-less stand-in for converting the URDF."""
    if not cache_dir.is_dir():
        raise RuntimeError(f"USD cache missing at {cache_dir}; populate it under .venv/isaacsim "
                           "(python -m isaacsimenvs.scripts.assets.prepare_usd_cache --populate)")
    key = cache_key(Path(asset_path).read_text(), kwargs)
    entry = _entry_dir(cache_dir, key)
    manifest_path = entry / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"USD cache miss for {Path(asset_path).name} (key {key}).\n"
            "The bake and this run disagree on URDF content or conversion options. "
            "Re-populate with the same --num_assets_per_type and --handle_head_types: the "
            "object pool is shuffled over its full length, so a different count is a "
            "different set of objects, not a subset."
        )
    manifest = json.loads(manifest_path.read_text())

    # Copy out of the cache into this run's work dir, so `_bake_usd` can write alongside it
    # without mutating the shared cache.
    dest_dir = Path(usd_work_dir) / Path(asset_path).stem
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(entry, dest_dir)
    (dest_dir / _MANIFEST_NAME).unlink(missing_ok=True)
    return str(dest_dir / manifest["usd_name"])
