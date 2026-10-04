"""The wiring between a hinged-slat articulation and the cloth-folding task.

Everything here runs on a CPU without Isaac Sim, which is the point: the three things most likely to
be silently wrong when a rigid chain stands in for a VBD sheet are all pure arithmetic, and all three
fail in ways that look like a policy failing to learn rather than like an error.

  * **the grid mapping**. ``ClothEnv`` computes every index set it uses -- tracked keypoints, the two
    halves, the goal mesh -- as arithmetic on ``grid_mesh``'s row-major order. A cloud of the right
    shape in a different order scores a fold that never happened.
  * **the reset pose**. The articulation's root link is a slat, whose frame is not the sheet's centre.
    Writing the sheet centre straight into the root pose offsets the sheet by up to 25 mm, rotating
    with the yaw draw, on every reset.
  * **the Kabsch fit**. ``object_rot`` is observed by the policy, and the fit is only well-posed if
    the four tracked corners stay rigid through the fold.
"""

from __future__ import annotations

import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

RESOLUTION = 7
SIZE = 0.10


# --------------------------------------------------------------------------- the grid mapping


def test_the_grid_map_reproduces_the_flat_sheet_exactly(rigid_cloth, cloth_geometry):
    """Rest-pose FK against the analytic grid, for every variant.

    The cheapest possible check and the one that invalidates everything else if it fails: if the
    mapped cloud is not the flat sheet at rest, no later number about folding means anything.

    A surface-hinge chain's grid sits on its slabs' mid-plane, one half-thickness below the link
    frames, so the whole cloud is offset in z by that much -- see ``sheet_grid_slat_map``. The shape
    must still be exact, which is what ``offset`` isolates rather than hides.
    """
    verts, _ = cloth_geometry.grid_mesh(SIZE, RESOLUTION)
    for name in rigid_cloth.VARIANTS:
        spec = rigid_cloth.variant(name)
        offset = 0.0 if spec.hinge == "mid" else -0.5 * spec.thickness
        mapping = rigid_cloth.sheet_grid_slat_map(spec, RESOLUTION)
        assert len(mapping) == RESOLUTION**2
        for (slat, pt), want in zip(mapping, verts):
            got = rigid_cloth.chain_point_in_sheet_frame(spec, slat, pt, None)
            assert got == pytest.approx((want[0], want[1], want[2] + offset), abs=1e-12), name


def test_the_grid_map_is_row_major_in_x_then_y(rigid_cloth):
    """Order matters more than content: it is what every index set the task computes assumes."""
    spec = rigid_cloth.variant("box-mid-odd")
    mapping = rigid_cloth.sheet_grid_slat_map(spec, RESOLUTION)
    # Within one row of constant i, the slat must not change and y must increase.
    for i in range(RESOLUTION):
        row = mapping[i * RESOLUTION : (i + 1) * RESOLUTION]
        assert len({slat for slat, _ in row}) == 1, "a row of constant x spans one slat"
        ys = [pt[1] for _, pt in row]
        assert ys == sorted(ys)
    # Across rows, the slat index is non-decreasing -- x increases monotonically.
    firsts = [mapping[i * RESOLUTION][0] for i in range(RESOLUTION)]
    assert firsts == sorted(firsts)


def test_a_seam_vertex_resolves_to_the_minus_x_slat(rigid_cloth):
    """``box2-surface``'s only seam sits at x = 0, where an odd grid has a row.

    Pinned rather than left to float comparison. It does not reach the score -- ``half_indices``
    excludes the centre row from both halves -- but an unpinned tie makes the cloud depend on
    rounding, and a cloud that changes between runs is not a reproducible measurement.
    """
    spec = rigid_cloth.variant("box2-surface")
    mapping = rigid_cloth.sheet_grid_slat_map(spec, RESOLUTION)
    mid = RESOLUTION // 2  # the row at x = 0
    assert mapping[mid * RESOLUTION][0] == 0
    assert mapping[(mid + 1) * RESOLUTION][0] == 1


def test_the_grid_map_refuses_a_chain_that_is_not_square(rigid_cloth):
    """A short chain would silently pile every far vertex onto the outermost slat."""
    spec = rigid_cloth.uniform_chain(size=0.10, num_slats=8)
    short = rigid_cloth.ChainSpec(widths=spec.widths[:-1], size=spec.size, thickness=spec.thickness)
    with pytest.raises(ValueError, match="square"):
        rigid_cloth.sheet_grid_slat_map(short, RESOLUTION)


# --------------------------------------------------------------------------- the adapter


class _Data:
    """The articulation data surface, in the convention the REAL backend uses.

    ``quat`` arrives wxyz (it is built from an angle here) and is stored **xyzw**, because that is what
    Isaac Lab 3.0's articulation API hands out -- its own docstrings say "quaternion orientation in
    (x, y, z, w)", and on the live solver a resting slat reads ``(-0.0, 0.0, 0.0, 1.0)``.

    This double previously stored wxyz, which is precisely why the convention bug reached a training
    launch: every test passed against a fake that agreed with the adapter and disagreed with Isaac.
    A test double that does not mirror the real convention cannot catch a convention error.
    """

    def __init__(self, pos, quat):
        quat = torch.cat((quat[..., 1:4], quat[..., 0:1]), dim=-1)
        self.body_pos_w = pos
        self.body_quat_w = quat
        self.body_lin_vel_w = torch.zeros_like(pos)
        self.body_ang_vel_w = torch.zeros_like(pos)
        self.joint_pos = torch.zeros((pos.shape[0], max(1, pos.shape[1] - 1)))


class _Art:
    """A stand-in for ``isaaclab.assets.Articulation``, recording what was written to it."""

    def __init__(self, rigid_cloth, spec, num_envs, angles=None, body_order=None):
        self._rc = rigid_cloth
        self.spec = spec
        names = [rigid_cloth.slat_link_name(i) for i in range(spec.num_slats)]
        self.body_names = list(names if body_order is None else body_order)
        frames = rigid_cloth.chain_frames(spec, angles)
        pos = torch.zeros((num_envs, spec.num_slats, 3))
        quat = torch.zeros((num_envs, spec.num_slats, 4))
        for b, name in enumerate(self.body_names):
            i = names.index(name)
            fx, fz, a = frames[i]
            pos[:, b, 0] = fx
            pos[:, b, 2] = fz
            # `chain_frames` composes x' = x cos a + z sin a, z' = -x sin a + z cos a, which is a
            # rotation by -a about +y. Encoding it as +a here would fold the sheet the wrong way and
            # still look plausible in every aggregate number.
            quat[:, b, 0] = math.cos(-a / 2.0)
            quat[:, b, 2] = math.sin(-a / 2.0)
        self.data = _Data(pos, quat)
        self.num_joints = max(spec.num_slats - 1, 0)
        self.written_pose = None
        self.written_joint_pos = None
        self.written_root_vel = None

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
        """The CURRENT API, which is what the adapter must prefer.

        Its quaternion is **(x, y, z, w)**, as documented: measured on the live solver from slat
        positions alone by ``scripts/analysis/rigid_cloth_yaw_probe.py`` (an xyzw write turned the
        chain to exactly the requested 0 / 45 / 90 / -120 deg; a wxyz one rolled it about its own
        long axis instead).
        """
        self.written_pose = root_pose.clone()

    def write_root_pose_to_sim(self, pose, env_ids=None):
        raise AssertionError(
            "the adapter must use write_root_pose_to_sim_index; the deprecated alias drops the "
            "orientation on the Newton backend"
        )

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        self.written_root_vel = vel.clone()

    def write_joint_state_to_sim(self, pos, vel, env_ids=None):
        """The joint write that actually reaches the Newton solver.

        Deprecated by name, but it is the one the Newton articulation overrides;
        ``write_joint_position_to_sim_index`` is accepted and silently does nothing (hinges driven to
        90 deg through it stayed at 0.03 deg).
        """
        self.written_joint_pos = pos.clone()
        self.written_joint_vel = vel.clone()


def _mesh(adapter_mod, rigid_cloth, spec, *, num_envs=1, angles=None, body_order=None):
    art = _Art(rigid_cloth, spec, num_envs, angles, body_order)
    mesh = adapter_mod.ChainAsClothMesh(
        art, spec=spec, resolution=RESOLUTION, num_envs=num_envs, device=torch.device("cpu")
    )
    return art, mesh


def test_the_adapter_reproduces_the_flat_sheet(rigid_cloth, rigid_cloth_adapter, cloth_geometry):
    """The same check as above, but through the torch code path the env actually runs."""
    verts, _ = cloth_geometry.grid_mesh(SIZE, RESOLUTION)
    want = torch.tensor(verts, dtype=torch.float32)
    for name in rigid_cloth.VARIANTS:
        spec = rigid_cloth.variant(name)
        _art, mesh = _mesh(rigid_cloth_adapter, rigid_cloth, spec)
        got = mesh._positions()[0]
        # Compare the SHAPE: a surface-hinge grid rides one half-thickness lower, uniformly.
        assert torch.allclose(got - got.mean(0), want - want.mean(0), atol=1e-6), name


def test_the_adapter_resolves_slats_by_name_not_by_position(
    rigid_cloth, rigid_cloth_adapter, cloth_geometry
):
    """A reordered ``body_names`` must not change the cloud.

    The importer and Newton are both free to reorder bodies, and a positional mapping would attach
    the grid to the wrong slats -- a sheet that bends in the wrong places, which reads as strange
    physics rather than as a bug.
    """
    spec = rigid_cloth.variant("box-mid-odd")
    verts, _ = cloth_geometry.grid_mesh(SIZE, RESOLUTION)
    names = [rigid_cloth.slat_link_name(i) for i in range(spec.num_slats)]
    shuffled = names[::-1]
    _art, mesh = _mesh(rigid_cloth_adapter, rigid_cloth, spec, body_order=shuffled)
    assert torch.allclose(
        mesh._positions()[0], torch.tensor(verts, dtype=torch.float32), atol=1e-6
    ), "a reordered body list changed the cloud"


def test_the_adapter_falls_back_to_body_state_w(rigid_cloth, rigid_cloth_adapter, cloth_geometry):
    """``body_pos_w`` is not in ``patches.TORCH_ATTRS``; ``body_state_w`` is.

    So on an articulation the Newton patch layer never wrapped -- this one, held under
    ``scene.articulations["rigid_cloth"]`` rather than one of the five names the patch knows -- the
    two split reads may be missing while the packed state is present. Untestable on this machine
    against the real solver, so the fallback is pinned here instead.
    """
    spec = rigid_cloth.variant("box3-mid")
    verts, _ = cloth_geometry.grid_mesh(SIZE, RESOLUTION)
    art, mesh = _mesh(rigid_cloth_adapter, rigid_cloth, spec)
    want = mesh._positions()[0].clone()

    packed = torch.zeros((1, spec.num_slats, 13))
    packed[..., 0:3] = art.data.body_pos_w
    packed[..., 3:7] = art.data.body_quat_w
    del art.data.body_pos_w
    del art.data.body_quat_w
    art.data.body_state_w = packed
    assert torch.allclose(mesh._positions()[0], want, atol=1e-6)


def test_the_simulator_quaternion_convention_is_xyzw(rigid_cloth_adapter):
    """The identity must round-trip, and a 180 deg z-rotation must NOT be mistaken for it.

    Isaac Lab 3.0's articulation API is xyzw. Read as wxyz, its identity ``(0, 0, 0, 1)`` becomes
    ``w = 0, z = 1`` -- a half turn about z -- which mirrored every grid vertex inside its slat and
    reported a 47 mm cloud for a 100 mm sheet. It raised nothing, because a 47 mm cloud is a
    believable crumpled sheet.

    Pinned as its own test because the conversion is one line at a boundary, it is invisible in every
    aggregate number downstream, and the cost of getting it wrong was the whole debugging session.
    """
    to_wxyz = rigid_cloth_adapter._sim_quat_to_wxyz
    ident_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(to_wxyz(ident_xyzw), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # A real half turn about z, in xyzw, must survive as a half turn about z in wxyz.
    half_z_xyzw = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert torch.allclose(to_wxyz(half_z_xyzw), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    # And it must be a pure relabelling, batch dimensions included.
    batch = torch.randn(3, 5, 4)
    got = to_wxyz(batch)
    assert got.shape == batch.shape
    assert torch.allclose(got[..., 0], batch[..., 3])
    assert torch.allclose(got[..., 1:4], batch[..., 0:3])


def test_the_adapter_is_loud_about_a_missing_slat(rigid_cloth, rigid_cloth_adapter):
    """A slat the articulation does not have must raise, and it must raise on the FIRST READ.

    Not at construction: the chain is built inside ``pre_clone_scene_hook``, where an Isaac Lab
    ``Articulation`` has no ``_data`` yet and ``body_names`` itself raises. Body resolution is
    therefore deferred to first use, so this is the earliest point the check can fire -- and it still
    fires before any number reaches the task, which is what matters. Constructing must stay quiet.
    """
    spec = rigid_cloth.variant("box3-mid")
    art = _Art(rigid_cloth, spec, 1)
    art.body_names = art.body_names[:-1] + ["not_a_slat"]
    mesh = rigid_cloth_adapter.ChainAsClothMesh(
        art, spec=spec, resolution=RESOLUTION, num_envs=1, device=torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="missing"):
        mesh._positions()


def test_a_folded_chain_puts_the_moving_half_back_over_the_stationary_one(
    rigid_cloth, rigid_cloth_adapter, cloth_geometry
):
    """The fold, measured on the emulated cloud rather than on joint angles.

    The moving half's particles must end up at negative x (mirrored across the crease) and one ply
    gap above z = 0. Checking this on the cloud is what proves the mapping and the fold agree; the
    joint angles alone were already known to be right.
    """
    for name in rigid_cloth.VARIANTS:
        spec = rigid_cloth.variant(name)
        angles = rigid_cloth.chain_fold_angles(spec)
        _art, mesh = _mesh(rigid_cloth_adapter, rigid_cloth, spec, angles=angles)
        cloud = mesh._positions()[0]
        moving = sorted(set(cloth_geometry.half_indices(RESOLUTION, "x", positive=True)))
        mv = cloud[moving]
        assert float(mv[:, 0].max()) < 1e-6, f"{name}: moving half did not cross the crease"
        # One ply gap ABOVE the stationary half's own surface, which for a surface hinge is itself
        # one half-thickness below z = 0. Comparing against the stationary cloud rather than against
        # zero is what makes the two hinge modes answerable by one number.
        stationary = sorted(set(cloth_geometry.half_indices(RESOLUTION, "x", positive=False)))
        base = float(cloud[stationary][:, 2].mean())
        gap = rigid_cloth.chain_ply_gap(spec)
        assert float(mv[:, 2].min()) - base == pytest.approx(gap, abs=1e-6), name


def _adapter(adapter_mod, rigid_cloth, cloth_geometry, spec, *, num_envs=1, angles=None):
    verts, _ = cloth_geometry.grid_mesh(SIZE, RESOLUTION)
    kp = cloth_geometry.corner_indices(RESOLUTION, "x")
    art, mesh = _mesh(adapter_mod, rigid_cloth, spec, num_envs=num_envs, angles=angles)
    obj = adapter_mod.ChainAsRigidObject(
        mesh,
        articulation=art,
        spec=spec,
        num_envs=num_envs,
        rest_local=verts,
        keypoint_idx=kp,
        corner_idx=kp,
        corner_rest=[verts[i] for i in kp],
        spawn_z=0.5,
        mass=2.0 * SIZE * spec.span,
        device=torch.device("cpu"),
    )
    return art, mesh, obj


def test_the_tracked_corners_stay_rigid_through_the_fold(
    rigid_cloth, rigid_cloth_adapter, cloth_geometry
):
    """``object_rot`` must read a completed fold, i.e. 180 degrees.

    This is the exact defect ``corner_indices`` was rewritten for on the VBD sheet: keypoints that
    straddle the crease are not rigid under a fold, the Kabsch fit returned 162 degrees, and the
    policy could never observe a finished fold. A chain folds about a different axis (the hinge line,
    not the particle row), so the property has to be re-established here, not assumed.
    """
    for name in rigid_cloth.VARIANTS:
        spec = rigid_cloth.variant(name)
        angles = rigid_cloth.chain_fold_angles(spec)
        _art, _mesh, obj = _adapter(
            rigid_cloth_adapter, rigid_cloth, cloth_geometry, spec, angles=angles
        )
        quat = obj.fit_rotation_wxyz()[0]
        angle = 2.0 * math.degrees(math.acos(min(1.0, abs(float(quat[0])))))
        assert angle == pytest.approx(180.0, abs=2.0), f"{name}: object_rot reads {angle:.1f} deg"


def test_reset_offsets_the_root_pose_by_the_root_slats_own_position(
    rigid_cloth, rigid_cloth_adapter, cloth_geometry
):
    """The articulation root is a SLAT, so its frame is not the sheet's centre.

    ``box3-mid``'s root slat centre is 25 mm from the sheet centre; a mid-hinge odd chain's is a few
    mm. Writing the requested centre straight through would offset the whole sheet by that much on
    every reset, in a direction that rotates with the yaw draw -- so it would look like a random
    spawn perturbation rather than a systematic error.
    """
    for name in rigid_cloth.VARIANTS:
        spec = rigid_cloth.variant(name)
        _art, _mesh, obj = _adapter(rigid_cloth_adapter, rigid_cloth, cloth_geometry, spec)
        pose = torch.tensor([[0.3, -0.2, 9.9, 1.0, 0.0, 0.0, 0.0]])
        obj.write_root_pose_to_sim(pose)
        written = _art.written_pose[0]
        want_x = 0.3 + rigid_cloth.chain_frames(spec, None)[spec.root][0]
        assert float(written[0]) == pytest.approx(want_x, abs=1e-6), name
        assert float(written[1]) == pytest.approx(-0.2, abs=1e-6)
        # The requested z is DISCARDED for the configured spawn height, as for the cloth.
        assert float(written[2]) == pytest.approx(0.5, abs=1e-6)


def test_reset_keeps_only_the_yaw_of_the_requested_orientation(
    rigid_cloth, rigid_cloth_adapter, cloth_geometry
):
    """Roll or pitch would stand the sheet on edge; the task draws a Haar-uniform SO(3) quaternion.

    Same contract as the cloth's adapter, and the offset above has to be rotated by the SAME yaw or
    the sheet lands off-centre in a heading-dependent way.
    """
    spec = rigid_cloth.variant("box3-mid")
    _art, _mesh, obj = _adapter(rigid_cloth_adapter, rigid_cloth, cloth_geometry, spec)
    yaw = math.pi / 2.0
    # A 90 deg yaw composed with a 90 deg roll: only the yaw may survive.
    qz = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    qx = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    quat = [
        qz[0] * qx[0],
        qz[0] * qx[1],
        qz[3] * qx[1],
        qz[3] * qx[0],
    ]
    pose = torch.tensor([[0.0, 0.0, 0.0] + quat])
    obj.write_root_pose_to_sim(pose)
    written = _art.written_pose[0]
    # The pose on the wire is xyzw: (x, y, z, qx, qy, qz, qw).
    assert float(written[3]) == pytest.approx(0.0, abs=1e-6), "roll leaked into the reset"
    assert float(written[4]) == pytest.approx(0.0, abs=1e-6), "pitch leaked into the reset"
    assert float(written[5]) == pytest.approx(math.sin(yaw / 2), abs=1e-6), "z must be THIRD on the wire"
    assert float(written[6]) == pytest.approx(math.cos(yaw / 2), abs=1e-6), "w must be LAST on the wire"
    # The root offset is along +x in the sheet frame, so a 90 deg yaw sends it to +y.
    off = rigid_cloth.chain_frames(spec, None)[spec.root][0]
    assert float(written[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(written[1]) == pytest.approx(off, abs=1e-6)
    # And the yaw itself survived: 90 deg about +z is qz = sin45, qw = cos45.
    assert float(written[5]) == pytest.approx(math.sin(yaw / 2), abs=1e-6)
    assert float(written[6]) == pytest.approx(math.cos(yaw / 2), abs=1e-6)


def test_reset_flattens_every_joint_and_stops_the_chain(
    rigid_cloth, rigid_cloth_adapter, cloth_geometry
):
    """A reset that leaves the hinges bent or spinning is not a reset.

    The cloth's adapter gets this for free -- it rewrites every particle position and zeroes every
    particle velocity. A chain has joint state that survives a root-pose write, so it has to be
    written explicitly, and forgetting it would carry the previous episode's fold into the next one.
    """
    spec = rigid_cloth.variant("box-mid-odd")
    angles = rigid_cloth.chain_fold_angles(spec)
    art, _mesh, obj = _adapter(
        rigid_cloth_adapter, rigid_cloth, cloth_geometry, spec, angles=angles
    )
    obj.write_root_pose_to_sim(torch.zeros((1, 7)))
    assert art.written_joint_pos is not None, "joints were never written"
    assert float(art.written_joint_pos.abs().max()) == 0.0
    assert float(art.written_root_vel.abs().max()) == 0.0


def test_the_mesh_refuses_a_nodal_write(rigid_cloth, rigid_cloth_adapter):
    """The inherited cloth path must not be reachable.

    ``ClothAsRigidObject.write_root_pose_to_sim`` teleports a particle cloud. If a future edit stops
    overriding it, this fails loudly instead of the chain silently not resetting.
    """
    spec = rigid_cloth.variant("box3-mid")
    _art, mesh = _mesh(rigid_cloth_adapter, rigid_cloth, spec)
    with pytest.raises(NotImplementedError):
        mesh.write_nodal_pos_to_sim_index(torch.zeros((1, 49, 3)), torch.zeros(1, dtype=torch.long))


def test_velocity_includes_the_rotational_term(rigid_cloth, rigid_cloth_adapter, cloth_geometry):
    """A folding slat's far edge moves almost entirely by rotation.

    ``root_lin_vel_w`` is observed by the policy and is the mean over the tracked keypoints, which
    sit at the moving half's corners -- the farthest points from any hinge. Dropping ``omega x r``
    would report the fastest part of the sheet as stationary.
    """
    spec = rigid_cloth.variant("box3-mid")
    art, mesh = _mesh(rigid_cloth_adapter, rigid_cloth, spec)
    # Spin the outermost slat about +y at 1 rad/s, everything else still.
    far = spec.moving[-1]
    names = list(art.body_names)
    b = names.index(rigid_cloth.slat_link_name(far))
    art.data.body_ang_vel_w[:, b, 1] = 1.0
    vel = mesh._velocities()[0]
    assert float(vel.abs().max()) > 1e-3, "the rotational term was dropped"


# --------------------------------------------------------------------------- the config


def test_every_variant_is_a_valid_config_choice(rigid_cloth):
    """The env's variant guard has to name the same five the model does."""
    cfg_mod = pytest.importorskip(
        "isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg",
        reason="needs isaaclab.utils.configclass",
    )
    for name in rigid_cloth.VARIANTS:
        cfg = cfg_mod.RigidClothCfg(variant=name)
        # Compared FIELD BY FIELD. The `rigid_cloth` fixture is path-loaded while the cfg imports
        # the package member, so `ChainSpec` is two distinct classes and dataclass `__eq__` returns
        # False between them regardless of content -- `==` here would pass vacuously in reverse.
        want = rigid_cloth.variant(name)
        for attr in ("widths", "size", "thickness", "hinge", "shape"):
            assert getattr(cfg.spec, attr) == getattr(want, attr), f"{name}.{attr}"
        assert cfg.ply_gap == rigid_cloth.chain_ply_gap(cfg.spec)
    with pytest.raises(ValueError, match="unknown rigid_cloth.variant"):
        cfg_mod.RigidClothCfg(variant="no-such-chain")


def test_the_default_slat_friction_is_the_cloths_effective_value(rigid_cloth):
    """Not 0.25, and not 0.5.

    MJWarp mixes friction by ``max``, so writing the cloth's own 0.25 onto a slat would make it
    grippier than the cloth against everything it touches. The table is the contact a flat sheet
    spends its life in, so the default is what the cloth effectively feels there.
    """
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg")
    cfg = cfg_mod.RigidClothCfg()
    want = rigid_cloth.cloth_effective_friction(rigid_cloth.CLOTH_REFERENCE["shape_mu"]["table"])
    assert cfg.friction == pytest.approx(want)
    assert cfg.friction == pytest.approx(math.sqrt(0.25 * 0.5), abs=1e-6)
    assert cfg_mod.RigidClothCfg(slat_friction=0.9).friction == pytest.approx(0.9)


def test_the_task_yaml_and_the_config_class_agree(rigid_cloth):
    """A yaml key with no field is silently dropped by the loader, and vice versa.

    Both directions matter: an unknown key means a setting a run believes it has pinned and has not,
    and a field absent from the yaml means a value that changes when the dataclass default changes.
    """
    import dataclasses
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg")
    root = Path(__file__).resolve().parents[1]
    loaded = yaml.safe_load((root / "isaacsimenvs/cfg/task/RigidCloth.yaml").read_text())
    block = loaded["rigid_cloth"]
    fields = {f.name for f in dataclasses.fields(cfg_mod.RigidClothCfg)}
    assert set(block) == fields
    # And the yaml must not drift from Cloth.yaml on anything else -- "only the manipuland changed"
    # is the claim the whole comparison rests on.
    cloth = yaml.safe_load((root / "isaacsimenvs/cfg/task/Cloth.yaml").read_text())
    assert set(loaded) - set(cloth) == {"rigid_cloth"}
    # The one sanctioned difference: the contact buffer. A 16-17 slat chain with self-collision needs
    # ~119k triangle pairs per env against the sheet's ~27k, and under-sizing drops contacts silently.
    # It is a buffer size, not physics, so it does not break "only the manipuland changed".
    allowed = {("newton", "per_env_triangle_pairs")}
    for k in cloth:
        if isinstance(cloth[k], dict):
            for kk in cloth[k]:
                if (k, kk) not in allowed:
                    assert loaded[k].get(kk) == cloth[k][kk], f"RigidCloth.yaml drifted: {k}.{kk}"
            assert set(loaded[k]) == set(cloth[k]), f"RigidCloth.yaml drifted: {k} keys"
        else:
            assert loaded[k] == cloth[k], f"RigidCloth.yaml drifted from Cloth.yaml: {k}"


def test_the_conversion_options_are_the_ones_the_fold_needs(rigid_cloth):
    """``self_collision`` is what stops the plies passing through each other.

    With it off the chain folds to zero thickness, which presents as a policy that has learned a
    perfect fold. ``replace_cylinders_with_capsules`` must stay off or a slat cylinder's ends are
    rounded away, shortening the sheet's contact with the table by a radius at each edge.
    """
    assets = pytest.importorskip("isaacsimenvs.tasks.cloth.utils.rigid_cloth_assets")
    assert assets.CONVERT_KWARGS["self_collision"] is True
    assert assets.CONVERT_KWARGS["replace_cylinders_with_capsules"] is False
    assert assets.CONVERT_KWARGS["fix_base"] is False


def test_the_config_defaults_are_the_ones_the_bake_uses():
    """A run at defaults must hit the baked cache key, so the two must be ONE definition.

    The bake cannot import ``RigidClothCfg`` (it inherits ``PlayEnvCfg``, which needs Isaac Lab 3.0's
    ``SimulationCfg``, absent from the Isaac Sim venv), so the shared values live in
    ``rigid_cloth_assets.URDF_DEFAULTS`` and the config reads them. If someone re-inlines a literal
    on either side, every default run dies with a cache miss -- recoverable, but only after a job has
    been queued, scheduled and started. This fails in 0.1 s instead.
    """
    assets = pytest.importorskip("isaacsimenvs.tasks.cloth.utils.rigid_cloth_assets")
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg")
    cfg = cfg_mod.RigidClothCfg()
    # `joint_damping` is the one exception and is checked by the next test instead: the URDF's value
    # is pinned at 0.0 because the importer drops `<dynamics>`, while the config's field is the
    # actuator gain that actually damps. They are two different quantities that share a name.
    for field in ("density", "joint_limit"):
        want = assets.URDF_DEFAULTS[field]
        assert getattr(cfg, field) == want, f"{field}: cfg {getattr(cfg, field)!r} != bake {want!r}"
    assert set(assets.URDF_DEFAULTS) == {"density", "joint_damping", "joint_limit"}, (
        "a new URDF parameter must be added to this check, or a default run can miss the bake"
    )


def test_hinge_damping_is_not_claimed_in_the_urdf():
    """The URDF must not claim a damping the importer throws away.

    Measured on the real stack: whatever ``<dynamics damping>`` says, the Newton model reports
    ``joint_damping = 0.0`` on every slat hinge. So hinge damping is an Isaac Lab actuator gain
    (``RigidClothCfg.joint_damping``), and the URDF's value is pinned at zero -- otherwise the two
    read as agreeing when only one of them does anything.
    """
    assets = pytest.importorskip("isaacsimenvs.tasks.cloth.utils.rigid_cloth_assets")
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg")
    assert assets.URDF_DEFAULTS["joint_damping"] == 0.0
    assert cfg_mod.RigidClothCfg().joint_damping > 0.0, "the actuator must supply real damping"


def test_the_hinges_get_armature_at_every_stiffness():
    """Armature must reach the solver even when the chain is limp (``joint_stiffness = 0``).

    The actuator used to be created only for a stiff chain, which is how the default configuration
    ran with an inertia-free, undamped hinge: from a dead stop in free fall, joint velocity reached
    1.3e3-4.9e3 rad/s on the FIRST step and all five variants diverged within 2-7 steps. A unit test
    cannot build the articulation, but it can pin the two facts that made the bug possible -- the
    default is limp, and the default armature is non-zero -- so a future edit cannot quietly restore
    "actuator only when stiff" without a failure.
    """
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg")
    cfg = cfg_mod.RigidClothCfg()
    assert cfg.joint_stiffness == 0.0
    assert cfg.joint_armature > 0.0
    env_py = (
        pathlib.Path(__file__).resolve().parents[1]
        / "isaacsimenvs/tasks/cloth/rigid_cloth_env.py"
    )
    src = env_py.read_text()
    assert "armature=r.joint_armature" in src, "the actuator must pass armature through"
    assert "if r.joint_stiffness > 0.0:" not in src, "the actuator must not be gated on stiffness"


def test_the_urdf_text_is_deterministic(rigid_cloth, tmp_path):
    """It is the USD cache key, so a non-deterministic URDF is a guaranteed cache miss."""
    assets = pytest.importorskip("isaacsimenvs.tasks.cloth.utils.rigid_cloth_assets")
    texts = []
    for _ in range(2):
        path = assets.write_variant_urdf(
            "box3-mid", tmp_path, density=2.0, joint_damping=1.0e-5, joint_limit=None
        )
        texts.append(path.read_text())
    assert texts[0] == texts[1]
    other = assets.write_variant_urdf(
        "box-mid-odd", tmp_path, density=2.0, joint_damping=1.0e-5, joint_limit=None
    )
    assert other.read_text() != texts[0], "two variants must not share a cache key"
