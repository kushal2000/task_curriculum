"""Fabric presets remain discoverable after package layout changes."""

from isaacsimenvs.tasks.cloth.backends.vbd import fabrics


def test_presets_are_loaded_from_the_package_config_directory():
    assert set(fabrics.names()) == {
        "base", "big_napkin_measured", "chiffon_measured", "cotton_measured",
        "denim_measured", "napkin_measured", "silk_measured", "towel_measured",
    }
    params = fabrics.cloth_params("cotton_measured")
    assert params
    assert all(isinstance(value, float) for value in params.values())
    assert fabrics.overrides("cotton_measured") == [f"env.deformable_cloth.{key}={value!r}" for key, value in params.items()]
