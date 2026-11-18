"""Mixed rigid topologies retain global environment identity through sparse PhysX views."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.rigid import designs, randomization
from isaacsimenvs.tasks.cloth.backends.rigid.backend import RigidBackend, RigidGroup
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT


class MaterialView:
    def __init__(self, rows, shapes, joints=0, prim_paths=()):
        self.prim_paths = list(prim_paths)
        self.materials = torch.ones(rows, shapes, 3)
        self.damping = torch.zeros(rows, joints)
        self.stiffness = torch.zeros(rows, joints)
        self.armature = torch.zeros(rows, joints)
        self.masses = torch.ones(rows, shapes)
        self.inertias = torch.eye(3).reshape(1, 1, 9).repeat(rows, shapes, 1)

    def get_masses(self):
        return self.masses.clone()

    def set_masses(self, values, ids):
        self.masses[ids] = values[ids]

    def get_inertias(self):
        return self.inertias.clone()

    def set_inertias(self, values, ids):
        self.inertias[ids] = values[ids]

    def get_dof_armatures(self):
        return self.armature.clone()

    def get_material_properties(self):
        return self.materials.clone()

    def set_material_properties(self, values, ids):
        self.materials[ids] = values[ids]

    def get_dof_dampings(self):
        return self.damping.clone()

    def get_dof_stiffnesses(self):
        return self.stiffness.clone()


def _backend(names=("single_hinge", "strip_4", "strip_8"), num_envs=None, resolution=9):
    backend = RigidBackend.__new__(RigidBackend)
    backend.geometry_revision = 0
    backend.robot_spec = ROBOT
    backend.device, backend.num_envs = "cpu", num_envs or 2 * len(names)
    n = backend.num_envs
    backend._dims = torch.stack((torch.linspace(.12, .3, n), torch.linspace(.2, .16, n),
                                 torch.linspace(.004, .012, n)), dim=1)
    backend._dimensions_cpu = backend._dims.numpy()
    backend._num_sizes, backend._size_ids = n, np.arange(n)
    rd = NS(enabled=True, density=[.02, .5], hinge_armature_scale=[0., 0.],
            bending_rigidity=[1e-6, 2.5e-4], bending_viscosity=[1e-6, 1.5e-4],
            table_static_friction=[.1, .8], hand_static_friction=[.3, 1.2],
            dynamic_friction_ratio=[.6, 1.], friction_buckets=16)
    backend.cfg = NS(
        task=NS(release=NS(contact_bodies="")),
        reset=NS(table_reset_z=.38, table_reset_xy=(.02, -.01)),
        rigid_cloth=NS(point_resolution=resolution, spawn_clearance=.002, randomization=rd,
                       bending_rigidity=5e-5, bending_viscosity=5e-5, density=.1, hinge_armature=0.,
                       table_static_friction=.4, hand_static_friction=.8, dynamic_friction_ratio=.8),
    )
    robot_names = [ROBOT.palm_body_name, *ROBOT.fingertip_link_names]
    backend.robot = NS(
        find_bodies=lambda requested, preserve_order: ([robot_names.index(v) for v in requested], requested),
        root_physx_view=MaterialView(n, 3),
    )
    backend.table = NS(root_physx_view=MaterialView(n, 1))
    backend.groups = []
    for index, name in enumerate(names):
        assigned = torch.arange(index, n, len(names))
        if not len(assigned):
            continue
        actual = assigned.flip(0)  # PhysX stores sparse rows in a different order.
        design = designs.get_design(name)
        body_names = list(reversed([p.name for p in design.plates]))
        joint_names = list(design.joint_names[1:] + design.joint_names[:1])
        p, count = len(body_names), len(actual)
        positions = torch.zeros(count, p, 3)
        positions[..., 0] = actual[:, None] * 10 + torch.arange(p)
        positions[..., 1] = actual[:, None] * 2
        orientations = torch.zeros(count, p, 4)
        orientations[..., 0] = 1.
        linear = torch.zeros_like(positions)
        linear[..., 2] = torch.arange(p) * .1
        angular = torch.zeros_like(positions)
        angular[..., 2] = .5
        view = MaterialView(count, p, len(joint_names), [f"/World/envs/env_{i}/Cloth/panel" for i in actual.tolist()])
        for row, env_id in enumerate(actual.tolist()):
            spec = design.resolve(*backend._dimensions_cpu[env_id])
            for plate in spec.plates:
                mass, _, inertia, _ = designs.polygon_mass_properties(plate, backend.cfg.rigid_cloth.density)
                body = body_names.index(plate.name)
                view.masses[row, body] = mass
                view.inertias[row, body] = torch.diag(torch.as_tensor(inertia)).flatten()
        asset = NS(body_names=body_names, joint_names=joint_names, num_joints=len(joint_names), root_physx_view=view,
                   data=NS(body_link_pos_w=positions, body_link_quat_w=orientations,
                           body_link_lin_vel_w=linear, body_link_ang_vel_w=angular))
        asset.write_joint_damping_to_sim = lambda value, env_ids, view=view: view.damping.__setitem__(env_ids, value)
        asset.write_joint_stiffness_to_sim = lambda value, env_ids, view=view: view.stiffness.__setitem__(env_ids, value)
        asset.write_joint_armature_to_sim = lambda value, env_ids, view=view: view.armature.__setitem__(env_ids, value)
        group = RigidGroup(index, design, assigned, articulation=asset)
        group.scene_asset = NS(set_env_ids=lambda ids: None)
        backend.groups.append(group)
    backend.initialize()
    return backend


@pytest.mark.parametrize("names", [("single_hinge",), ("single_hinge", "strip_4", "strip_8"),
                                   ("single_hinge", "flexible_corners", "hinge_band", "strip_8")])
def test_readout_scatters_named_bodies_from_reordered_sparse_rows(names):
    backend = _backend(names)
    points, velocities = backend.cloth_points()
    for env_id in range(backend.num_envs):
        design = designs.get_design(names[env_id % len(names)])
        rest = backend.cloth_grid.rest[env_id].numpy()
        owners = design.point_owners(rest / backend._dims[env_id].numpy())
        body_ids = len(design.plates) - 1 - owners
        expected = rest + np.c_[10 * env_id + body_ids, np.full(len(rest), 2 * env_id), np.zeros(len(rest))]
        expected_velocity = np.c_[-.5 * rest[:, 1], .5 * rest[:, 0], body_ids * .1]
        np.testing.assert_allclose(points[env_id], expected, atol=4e-6)
        np.testing.assert_allclose(velocities[env_id], expected_velocity, atol=1e-7)


def _capture_reset_state(asset):
    rows = len(asset.root_physx_view.prim_paths)
    state = {"root_pose": torch.full((rows, 7), 17.), "root_velocity": torch.full((rows, 6), 17.)}
    for name in ("joint_pos", "joint_vel", "position_target", "velocity_target"):
        state[name] = torch.full((rows, asset.num_joints), 17.)

    def write(key, value, env_ids):
        state[key][env_ids] = value

    asset.write_root_pose_to_sim = lambda value, env_ids: write("root_pose", value, env_ids)
    asset.write_root_velocity_to_sim = lambda value, env_ids: write("root_velocity", value, env_ids)
    asset.set_joint_position_target = lambda value, env_ids: write("position_target", value, env_ids)
    asset.set_joint_velocity_target = lambda value, env_ids: write("velocity_target", value, env_ids)

    def write_joints(position, velocity, env_ids):
        write("joint_pos", position, env_ids)
        write("joint_vel", velocity, env_ids)

    asset.write_joint_state_to_sim = write_joints
    return state


def test_partial_reset_maps_unsorted_global_ids_and_only_folds_the_designated_hinge():
    backend = _backend(("single_hinge", "strip_8", "flexible_corners", "hinge_band"))
    states = [_capture_reset_state(group.articulation) for group in backend.groups]
    env_ids = torch.arange(backend.num_envs)
    backend.scene = NS(env_origins=env_ids[:, None] * torch.tensor([[2., 3., 4.]]))
    backend._table_half_height = .15
    backend._table_z = .38 + env_ids * .01
    selected = torch.tensor([7, 0, 2])
    angles = torch.tensor([1.1, .4, .9])
    backend._write_cloth(selected, torch.tensor([[.03, .04]]).repeat(3, 1), torch.zeros(3), angles)
    for group, state in zip(backend.groups, states):
        for local, global_id in enumerate(group.env_ids.tolist()):
            if global_id not in selected:
                assert all(torch.all(values[local] == 17.) for values in state.values())
                continue
            requested_row = selected.tolist().index(global_id)
            expected = torch.zeros(group.articulation.num_joints)
            fold_name = group.design.joint_names[group.design.fold_hinge]
            expected[group.articulation.joint_names.index(fold_name)] = angles[requested_row]
            torch.testing.assert_close(state["joint_pos"][local], expected)
            for name in ("joint_vel", "position_target", "velocity_target", "root_velocity"):
                assert torch.count_nonzero(state[name][local]) == 0
            z = 4 * global_id + backend._table_z[global_id] + .15 + backend._dims[global_id, 2] / 2 + .002
            expected_pose = torch.tensor([2 * global_id + .05, 3 * global_id + .03, z, 1., 0., 0., 0.])
            torch.testing.assert_close(state["root_pose"][local], expected_pose)


def test_randomization_preserves_global_draw_order_and_unselected_envs_across_topologies():
    results = []
    for names in (("single_hinge",), ("strip_8", "flexible_corners", "hinge_band")):
        backend = _backend(names, num_envs=6)
        torch.manual_seed(123)
        randomization.rigid_init(backend)
        before_damping = backend._dr_bending_viscosity.clone()
        before_materials = [group.plate_mats.clone() for group in backend.groups]
        selected = torch.tensor([5, 0, 2])
        randomization.rigid_randomize(backend, selected)
        randomization.rigid_check(backend, torch.arange(6))
        torch.testing.assert_close(backend._dr_bending_viscosity[[1, 3, 4]], before_damping[[1, 3, 4]])
        plate_friction = torch.empty(6)
        for group, before in zip(backend.groups, before_materials):
            plate_friction[group.env_ids] = group.plate_mats[:, 0, 0]
            untouched = ~torch.isin(group.env_ids, selected)
            torch.testing.assert_close(group.plate_mats[untouched], before[untouched])
        results.append((backend._dr_density, backend._dr_armature_scale,
                        backend._dr_bending_viscosity, backend._dr_bending_rigidity, plate_friction,
                        backend._robot_mats, backend._table_mats, torch.rand(4)))
    for one_group, mixed in zip(*results):
        torch.testing.assert_close(one_group, mixed, rtol=0, atol=0)


def test_viewer_uses_polygon_faces_and_the_selected_global_row():
    backend = _backend(("flexible_corners", "strip_8"))
    for env_id in range(backend.num_envs):
        group = backend.groups[int(backend._env_group[env_id])]
        spec = group.design.resolve(*backend._dimensions_cpu[env_id])
        _, vertices, triangles, moving = backend.viewer_plates(env_id)
        assert len(vertices) == sum(2 * len(plate.vertices) for plate in spec.plates)
        assert len(triangles) == sum(4 * len(plate.vertices) - 4 for plate in spec.plates)
        assert triangles.min() >= 0 and triangles.max() < len(vertices)
        assert moving.dtype == np.bool_ and len(moving) == len(triangles)
        assert vertices[:, 0].min() > env_id * 10 - .5
        assert vertices[:, 0].max() < env_id * 10 + len(spec.plates) + .5


@pytest.mark.parametrize("paths", [[], "single_hinge.yaml"])
def test_design_list_is_validated_before_scene_construction(paths):
    cfg = NS(task=NS(release=NS(touch="origin")))
    with pytest.raises(ValueError, match="nonempty list"):
        RigidBackend(cfg, paths)


def test_designs_cycle_through_global_envs_and_each_sees_every_geometry():
    cfg = NS(
        task=NS(release=NS(touch="origin")), scene=NS(num_envs=9),
        rigid_cloth=NS(num_geometries=3, geometry_seed=0, geometries=[], length_range=[.1, .3],
                       width_range=[.1, .2], thickness=.004),
    )
    mixed = RigidBackend(cfg, ["single_hinge.yaml", "flexible_corners.yaml", "strip_8.yaml"])
    assert [group.env_ids.tolist() for group in mixed.groups] == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    sizes = {tuple(r) for r in np.round(mixed._dimensions_cpu, 9)}
    assert len(sizes) == 3 and all(r[2] == .004 for r in sizes)
    for group in mixed.groups:  # every topology gets all three geometries
        assert {tuple(r) for r in np.round(mixed._dimensions_cpu[group.env_ids.numpy()], 9)} == sizes
    cfg.rigid_cloth.geometries = [[.25, .15]]  # an explicit list overrides sampling
    explicit = RigidBackend(cfg, ["single_hinge.yaml"])
    np.testing.assert_allclose(explicit._dimensions_cpu, [[.25, .15, .004]] * 9)
    cfg.scene.num_envs = 1
    sparse = RigidBackend(cfg, ["single_hinge.yaml", "flexible_corners.yaml", "strip_8.yaml"])
    assert len(sparse.groups) == 1 and sparse.groups[0].env_ids.tolist() == [0]


ALL_DESIGNS = ("single_hinge", "strip_4", "strip_8", "flexible_corners", "hinge_band")


def test_exact_keypoints_are_grid_independent_across_sizes_and_sparse_group_rows():
    readings = []
    for resolution in (5, 7, 9):
        backend = _backend(ALL_DESIGNS, resolution=resolution)
        keypoints = backend.cloth_keypoints()
        expected_rest = backend._dims[:, None] * torch.tensor(
            [[[0., -.5, 0.], [0., .5, 0.], [.5, -.5, 0.], [.5, .5, 0.]]]
        )
        torch.testing.assert_close(backend.cloth_grid.keypoint_rest, expected_rest)
        for group in backend.groups:
            assert all(group.design.plates[group.body_ids.index(body)].center[0] > 0
                       for body in group.keypoint_body_ids.tolist())
            for env_id in group.env_ids.tolist():
                bodies = group.keypoint_body_ids
                expected = expected_rest[env_id].clone()
                expected[:, 0] += 10 * env_id + bodies
                expected[:, 1] += 2 * env_id
                torch.testing.assert_close(keypoints[env_id], expected)
        readings.append(keypoints)
    for reading in readings[1:]:
        torch.testing.assert_close(reading, readings[0], rtol=0, atol=0)


@pytest.mark.parametrize("name", ALL_DESIGNS)
def test_exact_crease_keypoints_follow_moving_plate_during_center_fold(name):
    backend = _backend((name,), num_envs=1)
    group = backend.groups[0]
    data = group.articulation.data
    data.body_link_pos_w.zero_()
    data.body_link_quat_w.zero_()
    data.body_link_quat_w[..., 0] = 1.
    thickness = backend._dims[0, 2]
    # A 90 degree fold around -y at z=T/2 moves even x=0 midplane points.
    for plate, body in zip(group.design.plates, group.body_ids):
        if plate.center[0] > 0:
            data.body_link_pos_w[0, body] = torch.tensor([thickness / 2, 0., thickness / 2])
            data.body_link_quat_w[0, body] = torch.tensor([2**-.5, 0., -2**-.5, 0.])
    length, width, thickness = backend._dims[0]
    expected = torch.tensor([
        [thickness/2, -width/2, thickness/2], [thickness/2, width/2, thickness/2],
        [thickness/2, -width/2, length/2 + thickness/2],
        [thickness/2, width/2, length/2 + thickness/2],
    ])
    torch.testing.assert_close(backend.cloth_keypoints()[0], expected, atol=2e-8, rtol=1e-6)
    assert torch.all(backend.cloth_keypoints()[0, :2, 2] > 0)


def test_flexible_corner_keypoint_tracks_its_own_diagonal_hinge():
    backend = _backend(("flexible_corners",), num_envs=1)
    group = backend.groups[0]
    data = group.articulation.data
    data.body_link_pos_w.zero_()
    data.body_link_quat_w.zero_()
    data.body_link_quat_w[..., 0] = 1.
    body = int(group.keypoint_body_ids[2])
    spec = group.design.resolve(*backend._dimensions_cpu[0])
    plate = spec.plates[group.body_ids.index(body)]
    hinge = next(hinge for hinge in spec.hinges if hinge.child == plate.name)
    axis, anchor = np.array(hinge.axis), np.array(hinge.anchor)
    angle = np.pi / 3

    def rotate(point):
        return (point * np.cos(angle) + np.cross(axis, point) * np.sin(angle)
                + axis * np.dot(axis, point) * (1 - np.cos(angle)))

    data.body_link_pos_w[0, body] = torch.as_tensor(anchor - rotate(anchor), dtype=torch.float32)
    data.body_link_quat_w[0, body] = torch.as_tensor(
        np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)], dtype=torch.float32
    )
    rest = backend.cloth_grid.keypoint_rest[0]
    expected = rest.clone()
    expected[2] = torch.as_tensor(anchor + rotate(rest[2].numpy() - anchor), dtype=torch.float32)
    torch.testing.assert_close(backend.cloth_keypoints()[0], expected, atol=2e-8, rtol=1e-6)
    assert not torch.allclose(expected[2], rest[2])



def test_density_and_armature_scale_physical_mass_and_inertia_without_compounding():
    from isaacsimenvs.tasks.cloth.backends.rigid.inertia import hinge_inertias

    backend = _backend(ALL_DESIGNS)
    # Include nonzero off-diagonal entries: all nine tensor entries must scale.
    for group in backend.groups:
        group.articulation.root_physx_view.inertias[..., [1, 3]] = 1e-9
    torch.manual_seed(57)
    randomization.rigid_init(backend)
    assert bool(((backend._dr_density >= .02) & (backend._dr_density <= .5)).all())
    assert bool(((backend._dr_armature_scale >= 0) & (backend._dr_armature_scale <= .1)).all())
    before = [(g.articulation.root_physx_view.get_masses(), g.articulation.root_physx_view.get_inertias(),
               g.articulation.root_physx_view.get_dof_armatures()) for g in backend.groups]
    selected = torch.tensor([9, 0, 3])
    backend.cfg.rigid_cloth.randomization.density = [.25, .25]
    backend.cfg.rigid_cloth.randomization.hinge_armature_scale = [.075, .075]
    for _ in range(2):
        randomization.rigid_randomize(backend, selected)
        randomization.rigid_check(backend, torch.arange(backend.num_envs))
        for group, old in zip(backend.groups, before):
            view = group.articulation.root_physx_view
            for local, env_id in enumerate(group.env_ids.tolist()):
                if env_id not in selected:
                    for got, expected in zip((view.masses, view.inertias, view.armature), old):
                        torch.testing.assert_close(got[local], expected[local], rtol=0, atol=0)
                    continue
                torch.testing.assert_close(view.masses[local], group.mass_base[local] * 2.5)
                torch.testing.assert_close(view.inertias[local], group.inertia_base[local] * 2.5)
                dimensions = backend._dimensions_cpu[env_id]
                assert float(view.masses[local].sum()) == pytest.approx(.25 * dimensions[0] * dimensions[1])
                reference = hinge_inertias(group.design.resolve(*dimensions), .25)
                torch.testing.assert_close(view.armature[local, group.joint_ids],
                                           torch.tensor(reference * .075, dtype=torch.float32))
                assert torch.count_nonzero(view.inertias[local, :, [1, 3]]) == 2 * len(group.body_ids)


def test_disabled_density_and_armature_randomization_preserves_authored_physics():
    backend = _backend(ALL_DESIGNS)
    backend.cfg.rigid_cloth.randomization.enabled = False
    before = [(g.articulation.root_physx_view.get_masses(), g.articulation.root_physx_view.get_inertias())
              for g in backend.groups]
    randomization.rigid_init(backend)
    randomization.rigid_randomize(backend, torch.tensor([4, 0]))
    for group, (masses, inertias) in zip(backend.groups, before):
        torch.testing.assert_close(group.articulation.root_physx_view.get_masses(), masses, rtol=0, atol=0)
        torch.testing.assert_close(group.articulation.root_physx_view.get_inertias(), inertias, rtol=0, atol=0)
        assert torch.count_nonzero(group.articulation.root_physx_view.get_dof_armatures()) == 0
    torch.testing.assert_close(backend._dr_density, torch.full((backend.num_envs,), .1))
    assert torch.count_nonzero(backend._dr_armature_scale) == 0


def test_backend_reset_region_clips_table_and_replaces_only_invalid_yaws():
    backend = RigidBackend.__new__(RigidBackend)
    backend._dims = torch.tensor([[.3, .3, .004]]).repeat(256, 1)
    backend._table_size = (.475, .4, .04)
    backend.cfg = NS(rigid_cloth=NS(place_x=[-.29, .29], place_y=[-.18, .19]),
                     reset=NS(random_yaw=True, yaw_range_deg=[0., 360.],
                              reset_position_noise_x=.1, reset_position_noise_y=.1))
    yaw = torch.full((256,), torch.pi / 4)
    yaw[::2] = 0
    xy = backend._cloth_start_xy(torch.arange(256), yaw)
    assert torch.count_nonzero(yaw[::2]) == 0
    assert bool((yaw[1::2] != torch.pi / 4).all())
    c, s = yaw.cos().abs(), yaw.sin().abs()
    half = .15 * (c + s)
    assert bool((xy[:, 0].abs() + half <= .475 / 2 - .005 + 1e-7).all())
    assert bool((xy[:, 1] - half >= -.18 - 1e-7).all())
    assert bool((xy[:, 1] + half <= .19 + 1e-7).all())
    backend.cfg.reset.random_yaw = False
    with pytest.raises(ValueError, match="fixed reset yaw"):
        backend._cloth_start_xy(torch.tensor([0]), torch.tensor([torch.pi / 4]))



def test_material_parameters_map_to_named_hinge_geometry_and_effective_contact_pairs():
    from isaacsimenvs.tasks.cloth.backends.rigid.bending import hinge_geometry_factors

    backend = _backend(ALL_DESIGNS)
    randomization.rigid_init(backend)
    for group in backend.groups:
        view = group.articulation.root_physx_view
        for row, env_id in enumerate(group.env_ids.tolist()):
            spec = group.design.resolve(*backend._dimensions_cpu[env_id])
            geometry = torch.as_tensor(hinge_geometry_factors(spec), dtype=torch.float32)
            torch.testing.assert_close(view.stiffness[row, group.joint_ids], geometry * backend._dr_bending_rigidity[env_id])
            torch.testing.assert_close(view.damping[row, group.joint_ids], geometry * backend._dr_bending_viscosity[env_id])
            assert torch.count_nonzero(view.armature[row]) == 0
            assert torch.all(view.materials[row, :, :2] == 1)
    for view, coefficient in ((backend.robot.root_physx_view, backend._dr_hand_static_friction),
                              (backend.table.root_physx_view, backend._dr_table_static_friction)):
        torch.testing.assert_close(view.materials[..., 0], coefficient[:, None].expand_as(view.materials[..., 0]))
        expected_dynamic = coefficient * backend._dr_dynamic_friction_ratio
        torch.testing.assert_close(view.materials[..., 1], expected_dynamic[:, None].expand_as(view.materials[..., 1]))
        assert torch.all(view.materials[..., 1] <= view.materials[..., 0])
        assert torch.count_nonzero(view.materials[..., 2]) == 0


def test_nominal_material_gains_and_friction_apply_when_randomization_is_disabled():
    backend = _backend(ALL_DESIGNS)
    backend.cfg.rigid_cloth.randomization.enabled = False
    randomization.rigid_init(backend)
    randomization.rigid_check(backend, torch.arange(backend.num_envs))
    for group in backend.groups:
        view = group.articulation.root_physx_view
        torch.testing.assert_close(view.stiffness, group.hinge_geometry * 5e-5)
        torch.testing.assert_close(view.damping, group.hinge_geometry * 5e-5)
    torch.testing.assert_close(backend.robot.root_physx_view.materials,
                               torch.tensor([.8, .64, 0.]).expand_as(backend._robot_mats))
    torch.testing.assert_close(backend.table.root_physx_view.materials,
                               torch.tensor([.4, .32, 0.]).expand_as(backend._table_mats))


@pytest.mark.parametrize("randomized", [False, True])
def test_fixed_historical_hinge_gains_ignore_material_draws_and_topology(randomized):
    backend = _backend(ALL_DESIGNS)
    cfg = backend.cfg.rigid_cloth
    cfg.randomization.enabled = randomized
    cfg.hinge_stiffness_override, cfg.hinge_damping_override = 0., .05
    randomization.rigid_init(backend)
    randomization.rigid_randomize(backend, torch.tensor([5, 0, 2]))
    randomization.rigid_check(backend, torch.arange(backend.num_envs))
    for group in backend.groups:
        view = group.articulation.root_physx_view
        torch.testing.assert_close(view.stiffness, torch.zeros_like(view.stiffness))
        torch.testing.assert_close(view.damping, torch.full_like(view.damping, .05))
    # The checker must still catch a real physics write/read mismatch.
    backend.groups[0].articulation.root_physx_view.damping[0, 0] = .06
    with pytest.raises(RuntimeError, match="damping read-back mismatch"):
        randomization.rigid_check(backend, torch.arange(backend.num_envs))


def test_one_fixed_gain_preserves_the_other_material_gain():
    backend = _backend(ALL_DESIGNS)
    backend.cfg.rigid_cloth.hinge_damping_override = .05
    randomization.rigid_init(backend)
    for group in backend.groups:
        view = group.articulation.root_physx_view
        expected = group.hinge_geometry * backend._dr_bending_rigidity[group.env_ids, None]
        torch.testing.assert_close(view.stiffness, expected)
        torch.testing.assert_close(view.damping, torch.full_like(view.damping, .05))


@pytest.mark.parametrize("name,value", [("stiffness", -.1), ("damping", float('nan'))])
def test_invalid_historical_gain_override_is_rejected(name, value):
    backend = _backend()
    setattr(backend.cfg.rigid_cloth, f"hinge_{name}_override", value)
    with pytest.raises(ValueError, match=f"hinge_{name}_override must be finite and nonnegative"):
        randomization.rigid_init(backend)


def test_material_binding_authors_multiply_and_survives_per_env_template_copy():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
    from isaacsimenvs.tasks.cloth.backends.rigid.scene import bind_physics_material, pair_materials

    source = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source, "/Sheet")
    for name in ("one", "two"):
        collider = UsdGeom.Cube.Define(source, f"/Sheet/{name}")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    material = bind_physics_material(source, "/Sheet", pair_materials(.4, .8, .7)["cloth"])
    assert material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"
    target = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(target, "/World/env_3")
    assert Sdf.CopySpec(source.GetRootLayer(), "/Sheet", target.GetRootLayer(), "/World/env_3/Cloth")
    for name in ("one", "two"):
        prim = target.GetPrimAtPath(f"/World/env_3/Cloth/{name}")
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        assert str(bound.GetPath()) == "/World/env_3/Cloth/ClothContactMaterial"
        assert bound.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"
        api = UsdPhysics.MaterialAPI(bound.GetPrim())
        assert api.GetStaticFrictionAttr().Get() == api.GetDynamicFrictionAttr().Get() == 1.
    pairs = pair_materials(.4, .8, .7)
    assert pairs["cloth"][0] * pairs["table"][0] == .4
    assert pairs["cloth"][1] * pairs["table"][1] == pytest.approx(.28)
    assert pairs["cloth"][0] * pairs["robot"][0] == .8
    assert pairs["cloth"][1] * pairs["robot"][1] == pytest.approx(.56)
