"""The bottle's inertia model, which is the whole of its difficulty axis.

The non-obvious property, and the one the config comments depend on, is that centre of
mass is *not* monotonic in fill level: an empty bottle's CoM sits near its centre, a
little liquid drags it down, and more liquid pushes it back up. `fill_fraction_easy` must
therefore sit at or above the CoM minimum, or "easier" settings would be harder.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

HEIGHT, RADIUS, WALL = 0.22, 0.035, 0.002


def test_com_is_below_centre_and_inertia_positive(gen_bottle):
    for fill in (0.0, 0.25, 0.5, 0.95):
        mass, com_z, ixx, iyy, izz = gen_bottle.bottle_inertial_properties(
            HEIGHT, RADIUS, WALL, fill
        )
        assert mass > 0
        assert com_z < 0.0, "CoM must sit below the geometric centre for any real bottle"
        assert ixx > 0 and iyy > 0 and izz > 0
        assert ixx == iyy, "a cylinder is transversely symmetric"


def test_mass_increases_with_fill(gen_bottle):
    masses = [
        gen_bottle.bottle_inertial_properties(HEIGHT, RADIUS, WALL, f)[0]
        for f in (0.0, 0.3, 0.6, 0.9)
    ]
    assert all(b > a for a, b in zip(masses, masses[1:]))


def test_com_height_has_an_interior_minimum(gen_bottle):
    """The real bottle-flip result — around a third full is most stable — recovered from
    the mass distribution alone, with no sloshing model."""
    fills = [i / 100 for i in range(101)]
    coms = [gen_bottle.bottle_inertial_properties(HEIGHT, RADIUS, WALL, f)[1] for f in fills]
    best = fills[min(range(len(coms)), key=lambda i: coms[i])]
    assert 0.1 < best < 0.5, f"CoM minimum at fill={best}, expected an interior minimum"
    assert coms[0] > min(coms) and coms[-1] > min(coms)


def test_com_rises_monotonically_above_the_minimum(gen_bottle):
    """This is the range `apply_difficulty` interpolates over, so it must be monotonic —
    otherwise raising difficulty could make the bottle easier to land."""
    fills = [0.25 + 0.05 * i for i in range(15)]  # 0.25 .. 0.95
    coms = [gen_bottle.bottle_inertial_properties(HEIGHT, RADIUS, WALL, f)[1] for f in fills]
    assert all(b > a for a, b in zip(coms, coms[1:]))


def test_same_function_works_on_batched_tensors(gen_bottle):
    """`reset_utils.apply_fill_level` reuses this exact function on torch tensors, so the
    baked geometry and the runtime fill level can never disagree."""
    torch = pytest.importorskip("torch")
    fills = torch.tensor([0.25, 0.6, 0.95])
    mass, com_z, ixx, _, izz = gen_bottle.bottle_inertial_properties(
        torch.tensor(HEIGHT), torch.tensor(RADIUS), WALL, fills
    )
    assert mass.shape == fills.shape
    assert torch.all(com_z[1:] > com_z[:-1])

    scalar_mass, scalar_com = gen_bottle.bottle_inertial_properties(
        HEIGHT, RADIUS, WALL, 0.6
    )[:2]
    assert mass[1].item() == pytest.approx(scalar_mass, rel=1e-6)
    assert com_z[1].item() == pytest.approx(scalar_com, rel=1e-6)


def test_urdf_is_single_link_with_shared_root_name(tmp_path, gen_bottle):
    """RigidObject derives its view regex from env_0's structure, so every variant in the
    pool must expose the same link name."""
    path = gen_bottle.generate_bottle_urdf(
        tmp_path / "b.urdf", height=HEIGHT, radius=RADIUS, fill_fraction=0.35
    )
    root = ET.parse(path).getroot()
    links = root.findall("link")
    assert len(links) == 1
    assert links[0].get("name") == gen_bottle.BOTTLE_ROOT_LINK
    assert root.findall("joint") == []
    assert float(root.find(".//mass").get("value")) > 0


def test_pool_sweeps_shape_and_stays_reproducible(tmp_path, gen_bottle):
    paths, shapes = gen_bottle.generate_bottle_urdfs(
        tmp_path / "pool", num_variants=8,
        height_range=(0.18, 0.26), aspect_ratio_range=(2.5, 4.0),
    )
    assert len(paths) == len(shapes) == 8
    heights = [h for h, _ in shapes]
    aspects = [h / (2 * r) for h, r in shapes]
    assert all(b > a for a, b in zip(heights, heights[1:]))
    assert all(b > a for a, b in zip(aspects, aspects[1:]))

    _, again = gen_bottle.generate_bottle_urdfs(
        tmp_path / "pool2", num_variants=8,
        height_range=(0.18, 0.26), aspect_ratio_range=(2.5, 4.0),
    )
    assert shapes == again, "the pool must be reproducible across runs and arms"


def test_rejects_a_wall_thicker_than_the_bottle(tmp_path, gen_bottle):
    with pytest.raises(ValueError):
        gen_bottle.generate_bottle_urdfs(
            tmp_path / "bad", num_variants=2,
            height_range=(0.05, 0.05), aspect_ratio_range=(30.0, 30.0),
        )
