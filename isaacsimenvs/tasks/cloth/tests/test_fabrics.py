"""Fabric presets remain discoverable after package layout changes."""

from isaacsimenvs.tasks.cloth.backends.vbd import fabrics


def test_presets_are_loaded_from_the_package_config_directory():
    assert {"base", "cotton", "denim"} <= set(fabrics.names())
    params = fabrics.cloth_params("cotton")
    assert params
    assert all(isinstance(value, float) for value in params.values())
    assert fabrics.overrides("cotton") == [f"env.deformable_cloth.{key}={value!r}" for key, value in params.items()]
