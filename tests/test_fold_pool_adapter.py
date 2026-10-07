"""The pool adapter shows each env's ACTIVE shape as the cloth grid, and parks the others.

Same contract as the chain adapter (``test_rigid_cloth_env.py``), against fakes that report body
poses in the REAL backend's convention (xyzw quaternions) computed from ``fold_shapes``' own
kinematics -- so a wrong grid map, a wrong shape selection or a wrong park write shows up as a
wrong cloud, not as a passing test against a fake that agrees with the adapter.
"""

from __future__ import annotations

import math

import pytest
from conftest import load_module  # noqa: E402  pytest puts tests/ on sys.path

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

RES = 7
fs = load_module("isaacsimenvs/tasks/cloth/utils/fold_shapes.py", "fold_shapes")
SHAPES = [fs.POOL[n] for n in fs.DEFAULT_SHAPES]


@pytest.fixture(scope="module")
def pool_adapter():
    return pytest.importorskip("isaacsimenvs.tasks.cloth.utils.fold_pool_adapter")


def _rot_to_xyzw(r):
    """Rotation matrix -> (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    if w > 1e-6:
        return [
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
            w,
        ]
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return q


class _Data:
    def __init__(self, pos, quat_xyzw):
        self.body_pos_w = pos
        self.body_quat_w = quat_xyzw
        self.body_lin_vel_w = torch.zeros_like(pos)
        self.body_ang_vel_w = torch.zeros_like(pos)


class _Art:
    """A stand-in articulation for one shape, posed by ``fold_shapes.transforms`` per env."""

    def __init__(self, shape, num_envs, angles_per_env=None, offset=(0.0, 0.0, 0.0), shuffle=False):
        names = ["slat_" + p.name for p in shape.plates]
        self.body_names = list(reversed(names)) if shuffle else names
        pos = torch.zeros((num_envs, len(names), 3))
        quat = torch.zeros((num_envs, len(names), 4))
        for e in range(num_envs):
            ang = (angles_per_env or {}).get(e)
            tf = shape.transforms(ang)
            for b, nm in enumerate(self.body_names):
                r, t = tf[nm[len("slat_") :]]
                pos[e, b] = torch.tensor(t, dtype=torch.float32) + torch.tensor(offset)
                quat[e, b] = torch.tensor(_rot_to_xyzw(r), dtype=torch.float32)
        self.data = _Data(pos, quat)
        self.num_joints = len(shape.hinges)
        self.pose_writes = []
        self.vel_writes = []
        self.joint_writes = []

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
        self.pose_writes.append((root_pose.clone(), env_ids.clone()))

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        self.vel_writes.append((vel.clone(), env_ids))

    def write_joint_state_to_sim(self, pos, vel, env_ids=None):
        self.joint_writes.append((pos.clone(), env_ids))


def _pool(adapter, num_envs, *, angles=None, offsets=None, shuffle=False):
    arts = [
        _Art(sh, num_envs, (angles or {}).get(s), (offsets or {}).get(s, (0, 0, 0)), shuffle)
        for s, sh in enumerate(SHAPES)
    ]
    mesh = adapter.PoolAsClothMesh(
        arts,
        mappings=[fs.grid_plate_map(sh, RES) for sh in SHAPES],
        body_prefix="slat_",
        num_envs=num_envs,
        device=torch.device("cpu"),
    )
    return arts, mesh


def _grid():
    step = 0.1 / (RES - 1)
    return torch.tensor(
        [[i * step - 0.05, j * step - 0.05, 0.0] for i in range(RES) for j in range(RES)]
    )


def test_every_shape_reproduces_the_flat_sheet(pool_adapter):
    _arts, mesh = _pool(pool_adapter, len(SHAPES), shuffle=True)
    mesh.active[:] = torch.arange(len(SHAPES))
    cloud = mesh._positions()
    for s in range(len(SHAPES)):
        torch.testing.assert_close(cloud[s], _grid(), atol=1e-6, rtol=0)


def test_the_cloud_follows_each_envs_active_shape(pool_adapter):
    """Shapes posed at different offsets: env e must read shape active[e]'s offset, no other."""
    n = len(SHAPES)
    offsets = {s: (0.0, 0.0, 0.1 * (s + 1)) for s in range(n)}
    _arts, mesh = _pool(pool_adapter, n, offsets=offsets)
    mesh.active[:] = torch.tensor([(e + 2) % n for e in range(n)])
    z = mesh._positions()[:, :, 2].mean(dim=1)
    for e in range(n):
        assert abs(float(z[e]) - 0.1 * ((e + 2) % n + 1)) < 1e-6


def test_a_half_folded_shape_reads_as_a_half_folded_sheet(pool_adapter):
    """Every grid point of the moving half lands on its mirror image across x = 0."""
    n = len(SHAPES)
    angles = {s: {0: SHAPES[s].fold_angles(fs.HALF_FOLD)} for s in range(n)}
    _arts, mesh = _pool(pool_adapter, 1, angles=angles)
    rest = _grid()
    moving = rest[:, 0] > 1e-9
    for s in range(n):
        mesh.active[:] = s
        cloud = mesh._positions()[0]
        want = rest[moving].clone()
        want[:, 0] = -want[:, 0]
        torch.testing.assert_close(cloud[moving], want, atol=1e-6, rtol=0)
        torch.testing.assert_close(cloud[~moving], rest[~moving], atol=1e-6, rtol=0)


def _obj(adapter, mesh, arts, n):
    origins = torch.tensor([[10.0 * e, 0.0, 0.0] for e in range(n)])
    park = torch.tensor([[0.0, 3.0 + 0.3 * s, 2.0] for s in range(len(arts))])
    grid = _grid()
    kp = [42, 48, 35, 41]
    return (
        adapter.PoolAsRigidObject(
            mesh,
            articulations=arts,
            env_origins=origins,
            park_local=park,
            num_envs=n,
            rest_local=grid.tolist(),
            keypoint_idx=kp,
            corner_idx=kp,
            corner_rest=[grid[i].tolist() for i in kp],
            spawn_z=0.6,
            mass=0.02,
            device=torch.device("cpu"),
        ),
        origins,
        park,
    )


def test_reset_places_the_active_shape_and_parks_the_rest(pool_adapter):
    n = 4
    arts, mesh = _pool(pool_adapter, n)
    obj, origins, park = _obj(pool_adapter, mesh, arts, n)
    ids = torch.arange(n)
    obj.select(ids, torch.tensor([0, 2, 4, 2]))
    yaw = 0.7
    # The task's reset pose, in the layout `task_yaw` decodes (see `cloth_adapter.task_yaw`).
    req = torch.zeros((n, 7))
    req[:, :3] = origins + torch.tensor([0.01, -0.02, 0.9])
    req[:, 3] = math.cos(0.5 * yaw)
    req[:, 6] = math.sin(0.5 * yaw)
    obj.write_root_pose_to_sim(req, ids)
    want_yaw = float(pool_adapter.task_yaw(req)[0])
    for s, art in enumerate(arts):
        pose, env_ids = art.pose_writes[-1]
        assert env_ids.tolist() == ids.tolist()
        for e in range(n):
            if int(obj.active[e]) == s:
                assert torch.allclose(pose[e, :2], req[e, :2])
                assert abs(float(pose[e, 2]) - 0.6) < 1e-6
                # xyzw yaw quaternion about +z
                assert abs(float(pose[e, 6]) - math.cos(0.5 * want_yaw)) < 1e-6
                assert abs(float(pose[e, 5]) - math.sin(0.5 * want_yaw)) < 1e-6
            else:
                assert torch.allclose(pose[e, :3], origins[e] + park[s])
                assert torch.allclose(pose[e, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        # every shape of every reset env is flattened and stopped
        assert art.joint_writes[-1][0].abs().max() == 0.0


def test_park_writes_only_parked_envs(pool_adapter):
    n = 5
    arts, mesh = _pool(pool_adapter, n)
    obj, origins, park = _obj(pool_adapter, mesh, arts, n)
    obj.select(torch.arange(n), torch.tensor([1, 1, 3, 0, 1]))
    for a in arts:
        a.pose_writes.clear()
    obj.park()
    for s, art in enumerate(arts):
        parked = [e for e, a in enumerate([1, 1, 3, 0, 1]) if a != s]
        assert len(art.pose_writes) == 1
        pose, env_ids = art.pose_writes[0]
        assert env_ids.tolist() == parked
        assert torch.allclose(pose[:, :3], origins[parked] + park[s])


def test_start_angles_survive_both_reset_writes_on_the_active_shape_only(pool_adapter):
    n = 4
    arts, mesh = _pool(pool_adapter, n)
    obj, origins, _park = _obj(pool_adapter, mesh, arts, n)
    ids = torch.arange(n)
    obj.select(ids, torch.tensor([1, 1, 4, 0]))
    q1 = torch.full((2, arts[1].num_joints), 0.5)
    obj.set_start_joints(1, torch.tensor([0, 1]), q1)
    # Env 2 also gets start angles for shape 1 -- it is not its active shape, so they must not land.
    obj.set_start_joints(1, torch.tensor([2]), torch.full((1, arts[1].num_joints), 0.9))
    req = torch.zeros((n, 7))
    req[:, :3] = origins
    req[:, 3] = 1.0
    obj.write_root_pose_to_sim(req, ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6)), ids)  # the reset's second write
    pos, env_ids = arts[1].joint_writes[-1]
    assert env_ids is None or env_ids.tolist() == ids.tolist()
    assert torch.allclose(pos[:2], q1) and pos[2:].abs().max() == 0.0
    for s in (0, 2, 3, 4):
        assert arts[s].joint_writes[-1][0].abs().max() == 0.0
    # Back to flat once cleared.
    obj.set_start_joints(1, ids, torch.zeros((n, arts[1].num_joints)))
    obj.write_root_velocity_to_sim(torch.zeros((n, 6)), ids)
    assert arts[1].joint_writes[-1][0].abs().max() == 0.0


def test_start_folds_turn_up_and_keep_the_seams_closed():
    for sh in SHAPES:
        folds = sh.start_folds()
        assert fs.HALF_FOLD in folds
        for name, dirs in folds.items():
            assert all(d == 1.0 for d in dirs.values()), (sh.name, name)
            for deg in (15.0, 120.0):
                ang = {h.child: 0.0 for h in sh.hinges}
                ang.update({c: math.radians(deg) for c in dirs})
                assert sh.seam_error(ang) < 1e-9, (sh.name, name, deg)


class _View:
    def __init__(self, paths):
        self.prim_paths = paths


def _subset_pool(adapter, n, carriers, angles=None):
    """One shape per env: shape ``s`` exists only in ``carriers[s]`` (global env ids), its instances
    in that (deliberately shuffled) order, posed with ``angles[env]`` when given."""
    arts = []
    for s, sh in enumerate(SHAPES):
        envs = carriers[s]
        art = _Art(sh, len(envs), {i: (angles or {}).get(e) for i, e in enumerate(envs)})
        art.root_physx_view = _View([f"/World/envs/env_{e}/FoldPool{s}/slat_x" for e in envs])
        arts.append(art)
    mesh = adapter.PoolAsClothMesh(
        arts,
        mappings=[fs.grid_plate_map(sh, RES) for sh in SHAPES],
        body_prefix="slat_",
        num_envs=n,
        device=torch.device("cpu"),
    )
    carried = torch.zeros((len(SHAPES), n), dtype=torch.bool)
    for s, envs in enumerate(carriers):
        carried[s, envs] = True
    mesh.set_carried(carried)
    return arts, mesh


def test_one_shape_per_env_reads_each_envs_own_instance(pool_adapter):
    n = 10
    shape_of = [e % 5 for e in range(n)]
    carriers = [[e for e in reversed(range(n)) if shape_of[e] == s] for s in range(5)]
    # Env e folded by a distinct amount, so a row mix-up shows as the wrong fold.
    angles = {}
    for e in range(n):
        sh = SHAPES[shape_of[e]]
        angles[e] = {h.child: 0.0 for h in sh.hinges}
        angles[e][next(iter(sh.folds[fs.HALF_FOLD][0]))] = 0.1 * (e + 1)
    arts, mesh = _subset_pool(pool_adapter, n, carriers, angles)
    obj, _origins, _park = _obj(pool_adapter, mesh, arts, n)
    obj.select(torch.arange(n), torch.tensor(shape_of))
    got = mesh._positions()
    full_arts, full_mesh = _pool(pool_adapter, n, angles={s: angles for s in range(5)})
    full_mesh.active[:] = torch.tensor(shape_of)
    want = full_mesh._positions()
    assert torch.allclose(got, want, atol=1e-6)


def test_one_shape_per_env_writes_land_on_local_rows_only(pool_adapter):
    n = 10
    shape_of = [e % 5 for e in range(n)]
    carriers = [[e for e in reversed(range(n)) if shape_of[e] == s] for s in range(5)]
    arts, mesh = _subset_pool(pool_adapter, n, carriers)
    obj, origins, _park = _obj(pool_adapter, mesh, arts, n)
    obj.select(torch.arange(n), torch.tensor(shape_of))
    with pytest.raises(ValueError):
        obj.select(torch.tensor([0]), torch.tensor([1]))  # env 0 carries shape 0 only
    ids = torch.tensor([2, 7, 3])
    q = torch.full((1, arts[2].num_joints), 0.4)
    obj.set_start_joints(2, torch.tensor([2]), q)
    req = torch.zeros((3, 7))
    req[:, :3] = origins[ids]
    req[:, 3] = 1.0
    for a in arts:
        a.pose_writes.clear()
        a.joint_writes.clear()
    obj.write_root_pose_to_sim(req, ids)
    obj.write_root_velocity_to_sim(torch.zeros((3, 6)), ids)
    # env 2 and env 7 run shape 2 (instances in reversed env order: 7 -> 0, 2 -> 1); env 3 runs shape 3.
    pose, local = arts[2].pose_writes[-1]
    assert sorted(local.tolist()) == [0, 1] and local.tolist() == [1, 0]
    assert torch.allclose(pose[:, :2], req[:2, :2])
    # Both instances written -> a full write (env_ids None), rows sorted into instance order:
    # instance 0 = env 7 (flat), instance 1 = env 2 (start angles).
    pos, local = arts[2].joint_writes[-1]
    assert local is None
    assert pos[0].abs().max() == 0.0 and torch.allclose(pos[1], q[0])
    assert arts[3].pose_writes[-1][1].tolist() == [1]  # shape 3 instances: env 8, env 3
    for s in (0, 1, 4):
        assert not arts[s].pose_writes and not arts[s].joint_writes
    # Nothing is ever parked.
    for a in arts:
        a.pose_writes.clear()
    obj.park()
    assert all(not a.pose_writes for a in arts)


def test_a_subset_articulation_must_match_its_carriers(pool_adapter):
    n = 10
    carriers = [[e for e in range(n) if e % 5 == s] for s in range(5)]
    arts, mesh = _subset_pool(pool_adapter, n, carriers)
    arts[1].root_physx_view = _View([f"/World/envs/env_{e}/FoldPool1/x" for e in (1, 6, 7)][:2] + ["/World/envs/env_7/FoldPool1/x"])
    with pytest.raises(RuntimeError):
        mesh._positions()


def test_the_fold_pool_yaml_and_config_class_agree():
    """``fold_pool:`` keys are exactly ``FoldPoolCfg``'s fields, and the rest of the yaml is
    RigidCloth.yaml verbatim -- the pool changes the manipuland, nothing else."""
    import dataclasses
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.fold_pool_env_cfg")
    root = Path(__file__).resolve().parents[1]
    loaded = yaml.safe_load((root / "isaacsimenvs/cfg/task/FoldPool.yaml").read_text())
    assert set(loaded["fold_pool"]) == {f.name for f in dataclasses.fields(cfg_mod.FoldPoolCfg)}
    assert loaded["fold_pool"]["shapes"] == list(fs.DEFAULT_SHAPES)
    rigid = yaml.safe_load((root / "isaacsimenvs/cfg/task/RigidCloth.yaml").read_text())
    # The two sanctioned stage-2 differences (see the yaml header), and nothing else.
    assert loaded["assets"].pop("table_urdf") == "assets/urdf/table_fold_pool.urdf"
    rigid["assets"].pop("table_urdf")
    assert loaded["cloth"].pop("keypoint_tolerance") == 0.015
    # Contact-shell-only DR: switched on, every other term pinned neutral.
    rd = loaded["rigid_cloth"].pop("randomization")
    rigid["rigid_cloth"].pop("randomization")
    assert rd["enabled"] is True and rd["contact_margin_offset"] == [-0.003, 0.003]
    for key in (
        "slat_friction_scale",
        "fingertip_friction_scale",
        "hinge_damping_scale",
        "mass_scale",
    ):
        assert rd[key] == [1.0, 1.0], key
    assert rd["hinge_stiffness"] == [0.0, 0.0]
    assert {k: v for k, v in loaded.items() if k != "fold_pool"} == rigid


def test_per_env_size_scales_the_grid_and_reset_shifts_the_centre(pool_adapter):
    """Env e reads a sheet ``xy_scale[e]`` times the base grid, and resets at origin + offset."""
    n = 3
    arts = [_Art(sh, n) for sh in SHAPES]
    scale = torch.tensor([1.0, 1.5, 2.0])
    mesh = pool_adapter.PoolAsClothMesh(
        arts,
        mappings=[fs.grid_plate_map(sh, RES) for sh in SHAPES],
        body_prefix="slat_",
        num_envs=n,
        device=torch.device("cpu"),
        xy_scale=scale,
    )
    mesh.active[:] = torch.tensor([0, 3, 4])
    cloud = mesh._positions()
    base = _grid()
    for e in range(n):
        want = base.clone()
        want[:, :2] *= float(scale[e])
        torch.testing.assert_close(cloud[e], want, atol=1e-6, rtol=0)
    offset = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.07, 0.0]])
    grid = _grid()
    kp = [42, 48, 35, 41]
    obj = pool_adapter.PoolAsRigidObject(
        mesh,
        articulations=arts,
        env_origins=torch.zeros((n, 3)),
        park_local=torch.tensor([[0.0, 3.0 + 0.3 * s, 2.0] for s in range(len(arts))]),
        centre_offset=offset,
        num_envs=n,
        rest_local=grid.tolist(),
        keypoint_idx=kp,
        corner_idx=kp,
        corner_rest=[grid[i].tolist() for i in kp],
        spawn_z=0.6,
        mass=0.02,
        device=torch.device("cpu"),
    )
    obj.select(torch.arange(n), torch.tensor([0, 3, 4]))
    req = torch.zeros((n, 7))
    req[:, 3] = 1.0
    obj.write_root_pose_to_sim(req, torch.arange(n))
    for e, s in enumerate([0, 3, 4]):
        pose, _ = arts[s].pose_writes[-1]
        torch.testing.assert_close(pose[e, :2], offset[e, :2])
