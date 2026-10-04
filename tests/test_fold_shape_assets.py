"""The pool's USDs import into Newton as the articulations ``fold_shapes`` describes.

Runs Newton on the CPU (no GPU, no Kit): the importer is the one the env uses, so a wrong axis, a
flipped hinge sign, a seam that lands inside the articulation, or a missing loop constraint fails
here instead of inside a training launch.
"""

from __future__ import annotations

import math

import pytest
from conftest import load_module  # noqa: E402  pytest puts tests/ on sys.path

np = pytest.importorskip("numpy")
pytest.importorskip("pxr")
newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

fs = load_module("isaacsimenvs/tasks/cloth/utils/fold_shapes.py", "fold_shapes")
fsa = load_module("isaacsimenvs/tasks/cloth/utils/fold_shape_assets.py", "fold_shape_assets")


def _quat_apply_xyzw(q, v):
    x, y, z, w = q
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


@pytest.fixture(scope="module", params=list(fs.POOL))
def imported(request, tmp_path_factory):
    sh = fs.POOL[request.param]
    path = fsa.write_shape_usd(sh, tmp_path_factory.mktemp("usd") / f"{sh.name}.usda")
    b = newton.ModelBuilder()
    b.add_usd(str(path))
    return sh, b.finalize(device="cpu")


def test_bodies_joints_and_mass(imported):
    sh, m = imported
    labels = [lab.rsplit("/", 1)[-1] for lab in m.joint_label]
    hinges = sorted(lab for lab in labels if lab.startswith(fsa.JOINT_PREFIX))
    assert hinges == sorted(fsa.JOINT_PREFIX + h.child for h in sh.hinges)
    assert m.body_count == len(sh.plates)
    assert m.articulation_count == 1
    assert abs(float(m.body_mass.numpy().sum()) - 0.020) < 1e-9


def test_seams_are_loop_joints_outside_the_articulation(imported):
    sh, m = imported
    labels = [lab.rsplit("/", 1)[-1] for lab in m.joint_label]
    art = m.joint_articulation.numpy()
    seams = [j for j, lab in enumerate(labels) if lab.startswith(fsa.SEAM_PREFIX)]
    assert len(seams) == len(sh.seams)
    assert all(int(art[j]) == -1 for j in seams)


def test_positive_hinge_angle_folds_up_exactly_as_fold_shapes_says(imported):
    sh, m = imported
    labels = [lab.rsplit("/", 1)[-1] for lab in m.joint_label]
    blabels = [lab.rsplit("/", 1)[-1] for lab in m.body_label]
    qs = m.joint_q_start.numpy()
    state = m.state()
    for j, lab in enumerate(labels):
        if not lab.startswith(fsa.JOINT_PREFIX):
            continue
        child = lab[len(fsa.JOINT_PREFIX) :]
        q = m.joint_q.numpy().copy()
        q[qs[j]] = 0.5 * math.pi
        newton.eval_fk(m, wp.array(q, dtype=float, device="cpu"), m.joint_qd, state)
        x = state.body_q.numpy()[blabels.index(fsa.BODY_PREFIX + child)]
        c = sh.plate(child).centroid
        got = np.array(x[:3]) + _quat_apply_xyzw(x[3:7], np.array([c[0], c[1], 0.0]))
        np.testing.assert_allclose(got, sh.point(child, c, {child: 0.5 * math.pi}), atol=1e-6)


@pytest.mark.parametrize("s_xy,s_z", [(1.0, 1.0), (1.5, 1.0), (2.0, 2.0)])
def test_a_scaled_world_rests_flat_at_its_own_size(tmp_path, s_xy, s_z):
    """box2-mid scaled at build time: 20 g x s^2, flat, resting on its own scaled thickness.

    The regression this guards: scaling only `shape_scale` left mass/inertia/hinges at 10 cm and
    the sheet settled into a V with corners ~2 mm through the floor.
    """
    sh = fs.POOL["box2-mid"]
    path = fsa.write_shape_usd(sh, tmp_path / "box2.usda")
    b = newton.ModelBuilder()
    b.add_ground_plane()
    b.add_usd(str(path), xform=wp.transform((0.0, 0.0, 0.02), wp.quat_identity()))
    bodies = list(range(len(b.body_label)))
    joints = [j for j in range(len(b.joint_label)) if b.joint_type[j] != newton.JointType.FREE]
    shapes = [i for i in range(len(b.shape_body)) if b.shape_body[i] >= 0]
    fsa.scale_builder_world(b, bodies, joints, shapes, s_xy, s_z)
    m = b.finalize(device="cpu")
    assert abs(float(m.body_mass.numpy().sum()) - 0.020 * s_xy**2) < 1e-7  # float32 model
    solver = newton.solvers.SolverMuJoCo(m, use_mujoco_cpu=True, use_mujoco_contacts=False)
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1, ctl = m.state(), m.state(), m.control()
    for _ in range(480):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, 1.0 / 240.0)
        s0, s1 = s1, s0
    q = s0.body_q.numpy()
    hinge = float(s0.joint_q.numpy()[-1])
    assert abs(math.degrees(hinge)) < 0.5
    # Lowest bottom-face corner of either plate, within 0.3 mm of the floor.
    lows = []
    for bi, plate in enumerate(sh.plates):
        for x, y in plate.polygon:
            v = np.array([x * s_xy, y * s_xy, -0.5 * sh.thickness * s_z])
            lows.append(float((q[bi][:3] + _quat_apply_xyzw(q[bi][3:7], v))[2]))
    assert min(lows) > -3e-4
    assert abs(float(q[0][2]) - 0.5 * sh.thickness * s_z) < 3e-4
