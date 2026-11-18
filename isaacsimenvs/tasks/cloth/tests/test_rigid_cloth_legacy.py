"""Configs saved before the thickness curriculum and the size grid were removed still load."""
from isaacsimenvs.tasks.cloth.backends.rigid.legacy import upgrade_rigid_cloth


def test_curriculum_run_gets_the_checkpoint_thickness_and_the_grid_becomes_num_geometries():
    rc = {"design_levels": 16, "thickness_range": [0.02, 0.02],
          "thickness_curriculum": {"enabled": True, "start": 0.02, "target": 0.002}}
    notes = upgrade_rigid_cloth(rc, checkpoint_thickness=0.0162)
    assert rc == {"thickness": 0.0162, "num_geometries": 256} and len(notes) == 2


def test_curriculum_without_checkpoint_uses_its_start():
    rc = {"thickness_range": [0.02, 0.02], "thickness_curriculum": {"enabled": True, "start": 0.02}}
    upgrade_rigid_cloth(rc)
    assert rc == {"thickness": 0.02}


def test_fixed_range_becomes_thickness_and_a_disabled_curriculum_is_dropped():
    rc = {"thickness_range": [0.002, 0.002], "thickness_curriculum": {"enabled": False, "start": 0.02}}
    upgrade_rigid_cloth(rc)
    assert rc == {"thickness": 0.002}


def test_a_current_config_is_unchanged():
    rc = {"thickness": 0.003, "num_geometries": 64, "geometry_seed": 1}
    assert upgrade_rigid_cloth(rc) == [] and rc == {"thickness": 0.003, "num_geometries": 64, "geometry_seed": 1}
