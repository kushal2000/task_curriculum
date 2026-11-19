"""Startup proofs: each fails loudly on a defect that would otherwise train silently.

Per pool shape: the per-env sizes reached the simulator (plate mass), the emulated cloud is the flat
sheet at the requested place and yaw, parked shapes are parked, a partial joint write lands only where
it was sent. Then: folded starts point up and land where asked, and the arm holds its home pose.
Functions of the ``RigidBackend`` (``b``), run once from ``RigidBackend.start``.
"""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.backends.base import sheet_yaw
from isaacsimenvs.tasks.cloth.backends.rigid.pool import to_torch

__all__ = ["calibrate_start_signs", "verify_placement", "verify_start_folds"]


def _plate_mass(b) -> torch.Tensor:
    """``(N,)`` total plate mass per env, every shape it carries."""
    total = torch.zeros(b.num_envs, device=b.device)
    for s, a in enumerate(b.arts):
        total[b.pool.members(s)] += a.root_physx_view.get_masses().sum(dim=1).to(b.device)
    return total


def verify_placement(b, reset_table) -> None:
    """Placement / shape / yaw / joint-write proofs, once per pool shape.

    Each shape in turn is made active in every env that CARRIES it -- every env, unless one shape
    per env -- and checked there; the others are parked. All fatal. Two real defects motivated
    them, both of which produce a plausible crumpled sheet rather than an error: an xyzw/wxyz
    mix-up that mirrored every plate (a 100 mm sheet read as 47 mm), and a root offset that
    placed the sheet 51 mm off centre. The yawed checks catch a rotation-convention error on the
    write side, which a centroid check cannot see.
    """
    pool = b.pool
    n = b.num_envs
    dt = float(b.cfg.sim.dt)
    origins = b.scene.env_origins
    all_ids = torch.arange(n, device=b.device)
    want = torch.tensor(b.rest_local, device=b.device, dtype=torch.float32)
    # (N, P, 3): each env's own flat sheet, centred.
    ref = (want - want.mean(dim=0, keepdim=True)).unsqueeze(0) * b.scale3()
    target = origins + pool.centre_offset

    # The per-env sizes reached the SIMULATOR, in the right env: total plate mass per env is
    # every pool shape at density x size^2. A world/env mix-up in the build hook would show up
    # here and nowhere else -- the cloud above is built from the same scale table it would be
    # checked against.
    mass = _plate_mass(b)
    carried = b.pool.carried
    want_mass = carried.sum(dim=0).float() * float(b.cfg.rigid_cloth.density) * b.env_size**2
    mass_err = float(((mass - want_mass).abs() / want_mass).max())
    if mass_err > 1e-3:
        raise RuntimeError(
            f"[fold_pool] per-env plate mass is off by up to {mass_err * 100:.2f}% of the size "
            f"draw -- the sheets were not built at their envs' sizes"
        )

    def place(yaw: float) -> torch.Tensor:
        pose = torch.zeros((n, 7), device=b.device, dtype=torch.float32)
        pose[:, :3] = origins
        pose[:, 3] = math.cos(0.5 * yaw)
        pose[:, 6] = math.sin(0.5 * yaw)
        # The pose's own yaw (not a reset's: see `base.reset_yaw`).
        pool.place(all_ids, pose[:, :2], sheet_yaw(pose[:, 3:7]))
        b.sim.step(render=False)
        b.scene.update(dt)
        return b.read_points()[0]

    def rotated(yaw: float) -> torch.Tensor:
        return torch.stack(
            (
                ref[..., 0] * math.cos(yaw) - ref[..., 1] * math.sin(yaw),
                ref[..., 0] * math.sin(yaw) + ref[..., 1] * math.cos(yaw),
            ),
            dim=-1,
        )

    def centred(cloud: torch.Tensor) -> torch.Tensor:
        return cloud - cloud.mean(dim=1, keepdim=True)

    one_shape = b.env_shape_np is not None
    if one_shape:
        pool.select(all_ids, pool.first_carried())
    for s, sh in enumerate(b.shapes):
        ids = carried[s].nonzero(as_tuple=True)[0]  # the envs this shape is checked in
        if not one_shape:
            pool.select(all_ids, torch.full((n,), s, device=b.device, dtype=torch.long))
        cloud = place(0.0)
        if s == 0:
            # Env 0's OWN sheet: with size DR it is not the nominal grid.
            layout = b.pool.detect_quat_layout((want.unsqueeze(0) * b.scale3())[0])
            sc = b.pool._quat_layout_scores
            print(
                f"[fold_pool] quaternion layout = {layout} (flat-grid error: xyzw "
                f"{sc['xyzw'] * 1e3:.2f} mm, wxyz {sc['wxyz'] * 1e3:.2f} mm)",
                flush=True,
            )
        residual = None
        for _attempt in range(3):
            cloud = place(0.0)
            err = (cloud[ids].mean(dim=1) - target[ids])[:, :2]
            residual = float(err.norm(dim=-1).max())
            if residual < 1.0e-4:
                break
            pool.calibrate_root_offset(s, err[0])
        if residual is None or residual > 1.0e-4:
            raise RuntimeError(
                f"[fold_pool] {sh.name}: centroid {residual * 1e3:.2f} mm off after 3 passes"
            )

        art = b.arts[s]
        hinge = math.degrees(float(to_torch(art.data.joint_pos).abs().max()))
        if hinge > 1.0:
            raise RuntimeError(
                f"[fold_pool] {sh.name} is not flat at reset: hinge {hinge:.2f} deg"
            )

        cloud = place(0.0)[ids]
        shape_err = float((centred(cloud)[..., :2] - ref[ids][..., :2]).norm(dim=-1).max())
        yaw_err = 0.0
        cent_err = 0.0
        for yaw in (0.5 * math.pi, math.radians(-120.0)):
            cy = place(yaw)[ids]
            yaw_err = max(
                yaw_err, float((centred(cy)[..., :2] - rotated(yaw)[ids]).norm(dim=-1).max())
            )
            cent_err = max(
                cent_err, float((cy.mean(dim=1) - target[ids])[:, :2].norm(dim=-1).max())
            )
        if shape_err > 1.0e-3 or yaw_err > 1.0e-3 or cent_err > 1.0e-3:
            raise RuntimeError(
                f"[fold_pool] {sh.name}: cloud is not the placed flat sheet -- shape "
                f"{shape_err * 1e3:.2f} mm, yawed shape {yaw_err * 1e3:.2f} mm, yawed centroid "
                f"{cent_err * 1e3:.2f} mm (tolerance 1 mm); layout {b.pool.quat_layout}"
            )

        # The parked shapes really are parked: nowhere near this env's table. (With one shape per
        # env nothing is parked.)
        for k in range(len(b.shapes)):
            rows = (carried[k] & (pool.active != k)).nonzero(as_tuple=True)[0]
            if k == s or rows.numel() == 0:
                continue
            far = (
                (b.pool.shape_positions(k)[rows].mean(dim=1) - origins[rows]).norm(dim=-1).min()
            )
            if float(far) < 0.5 * b.cfg.fold_pool.park_distance:
                raise RuntimeError(
                    f"[fold_pool] {b.shapes[k].name} should be parked but is "
                    f"{float(far):.2f} m from its env origin"
                )

        _verify_pool_joint_writes(b, s)
        print(
            f"[fold_pool] {sh.name}: placed within {residual * 1e6:.0f} um, flat cloud within "
            f"{shape_err * 1e6:.0f} um, yawed within {yaw_err * 1e6:.0f} um, root offset "
            f"{[round(float(v) * 1e3, 2) for v in pool.root_offset[s]]} mm",
            flush=True,
        )
    _verify_arm_clear(b, reset_table)


def _verify_arm_clear(b, reset_table, steps: int = 30, tol_deg: float = 2.0) -> None:
    """The arm holds its home pose: nothing in the scene presses on it.

    A table extended toward the robot (0.6 x 0.7 m) once sat in the arm's elbow, which rests
    below the table top. Every reset then knocked the arm away, it stayed pinned at the table
    edge, and two training arrays ran to completion without one fold. Nothing else fails on
    that: the sheet is placed correctly and the episodes simply time out.
    """
    robot = b.robot
    dt = float(b.cfg.sim.dt)
    # The table is moved to its working height only by a reset; this runs before the first one.
    reset_table(torch.arange(b.num_envs, device=b.device))
    # The sheets out of the way first, as the start-fold checks do: the placement checks above
    # leave every env's sheet at its origin, at the hand's height, and one dropping onto the
    # fingers reads as a drifting arm (7-shape pool: 0.7 deg in every run, 6 and 13 deg twice).
    _reset_all_to(b, 0, torch.zeros((b.num_envs, int(b.arts[0].num_joints)), device=b.device), dy=-0.5)
    arm = [i for i, name in enumerate(robot.joint_names) if name in b.robot_spec.arm_joint_names]
    home = to_torch(robot.data.default_joint_pos).clone()
    robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    robot.set_joint_position_target(home)
    for _ in range(steps):
        robot.write_data_to_sim()
        b.sim.step(render=False)
        b.scene.update(dt)
    drift = math.degrees(float((to_torch(robot.data.joint_pos) - home)[:, arm].abs().max()))
    if drift > tol_deg:
        raise RuntimeError(
            f"[fold_pool] the arm cannot hold its home pose: a joint drifted {drift:.1f} deg in "
            f"{steps} steps (tolerance {tol_deg}) -- something in the scene, most likely the "
            f"table ({b.cfg.assets.table_urdf}), is pressing on it"
        )
    print(f"[fold_pool] arm holds its home pose ({drift:.2f} deg drift)", flush=True)


def _verify_pool_joint_writes(b, s: int) -> None:
    """A partial joint write reaches the solver and only the envs it names: a write that is
    silently dropped leaves a sheet that ended an episode folded starting the next one folded."""

    # In the articulation's own instance order: rows [0, m/2) are `half`, the rest are not.
    ids = b.pool.members(s)
    m = int(ids.numel())
    if m < 2:
        return
    art = b.arts[s]
    nj = int(art.num_joints)
    half = ids[: m // 2]
    test = 0.3
    q = torch.full((half.shape[0], nj), test, device=b.device, dtype=torch.float32)
    b.pool.write_joint_positions(s, q, env_ids=half)
    b.sim.step(render=False)
    b.scene.update(float(b.cfg.sim.dt))
    got = to_torch(art.data.joint_pos)
    moved = float(got[: m // 2].abs().max())
    untouched = float(got[m // 2 :].abs().max())
    b.pool.write_joint_positions(s, torch.zeros((m, nj), device=b.device), env_ids=ids)
    b.sim.step(render=False)
    b.scene.update(float(b.cfg.sim.dt))
    if moved < 0.5 * test:
        raise RuntimeError(
            f"[fold_pool] {b.shapes[s].name}: partial joint write did not reach the solver "
            f"({math.degrees(moved):.2f} deg of {math.degrees(test):.1f})"
        )
    if untouched > math.radians(5.0):
        raise RuntimeError(
            f"[fold_pool] {b.shapes[s].name}: partial joint write leaked "
            f"{math.degrees(untouched):.2f} deg into unnamed envs"
        )


def _fold_index(b, s: int, F: int) -> torch.Tensor:
    """``(N,)`` which of shape ``s``'s ``F`` start folds each env tests: its RANK among the envs
    carrying ``s``, mod ``F``. Not ``env % F``: with one shape per env (shape = ``env % S``) that
    aliases -- every box2x8 env (S=5, F=5) got the same fold and the other four went untested."""
    k = torch.zeros(b.num_envs, device=b.device, dtype=torch.long)
    ids = b.pool.members(s)
    k[ids] = torch.arange(ids.numel(), device=b.device) % F
    return k


def _reset_all_to(b, s: int, q: torch.Tensor, dy: float = 0.0) -> None:
    """Every env on shape ``s`` at hinge angles ``q``, flat yaw-0 at its origin shifted ``dy`` in
    y, via the reset path. ``dy < 0`` is away from the robot: the startup checks of the start
    folds use -0.5 m, off the table's far edge in free air, so that a tall flap at spawn height
    cannot touch the hand resting at its home pose."""
    pool, n = b.pool, b.num_envs
    all_ids = torch.arange(n, device=b.device)
    pose = torch.zeros((n, 7), device=b.device, dtype=torch.float32)
    pose[:, :3] = b.scene.env_origins
    pose[:, 1] += dy
    pose[:, 3] = 1.0
    if b.env_shape_np is None:
        pool.select(all_ids, torch.full((n,), s, device=b.device, dtype=torch.long))
    else:  # every env keeps its own shape; only the envs carrying `s` are measured
        pool.select(all_ids, pool.first_carried())
    pool.set_start_joints(s, all_ids, q)
    pool.place(all_ids, pose[:, :2], sheet_yaw(pose[:, 3:7]))


def calibrate_start_signs(b, probe_deg: float = 15.0) -> None:
    """Which joint-angle sign lifts each start fold UP, measured: ``fold_shapes`` says +, but the
    simulator's joint frame decides (PhysX turned +120 deg box2 starts DOWN through the table).

    Each fold is written at +``probe_deg`` (env ``i`` takes fold ``i % F``) and stepped once; the
    sheet's highest point rises for an up fold, its lowest point sinks for a down one. A fold whose
    envs disagree, or that moves neither way, is fatal.
    """
    n, dt = b.num_envs, float(b.cfg.sim.dt)
    all_ids = torch.arange(n, device=b.device)
    for s, (sh, table) in enumerate(zip(b.shapes, b.start_tables)):
        F = table.shape[0]
        if F == 0:
            continue
        _reset_all_to(b, s, torch.zeros((n, table.shape[1]), device=b.device), dy=-0.5)
        b.sim.step(render=False)
        b.scene.update(dt)
        z0 = b.read_points()[0][..., 2]
        kof = _fold_index(b, s, F)
        _reset_all_to(b, s, table[kof] * math.radians(probe_deg), dy=-0.5)
        b.sim.step(render=False)
        b.scene.update(dt)
        z = b.read_points()[0][..., 2]
        rise = z.amax(dim=1) - z0.amax(dim=1)
        sink = z0.amin(dim=1) - z.amin(dim=1)
        b.pool.set_start_joints(s, all_ids, torch.zeros((n, table.shape[1]), device=b.device))
        carried = b.pool.carried[s]
        signs = []
        for k in range(F):
            on = (kof == k) & carried
            if not bool(on.any()):
                raise RuntimeError(
                    f"[fold_pool] {sh.name}: start fold {k} has no env to measure its direction in "
                    f"({int(carried.sum())} envs carry the shape, {F} folds)"
                )
            up = float((rise[on] > sink[on]).float().mean())
            moved = float(torch.maximum(rise[on], sink[on]).median())
            if moved < 1.0e-3 or 0.05 < up < 0.95:
                raise RuntimeError(
                    f"[fold_pool] {sh.name}: start fold {k} at +{probe_deg:g} deg moved the sheet "
                    f"{moved * 1e3:.2f} mm, up in {up * 100:.0f}% of envs -- no consistent direction"
                )
            signs.append(1.0 if up >= 0.95 else -1.0)
        table *= torch.tensor(signs, device=b.device).unsqueeze(1)
        print(f"[fold_pool] {sh.name}: start-fold joint signs (up) {signs}", flush=True)
    _reset_all_to(b, 0, torch.zeros((n, b.start_tables[0].shape[1]), device=b.device))


def verify_start_folds(b, angle: float, tol_deg: float = 2.0, share: float = 0.95) -> None:
    """Every start fold, at the LARGEST start angle, through the reset's own write path: one
    physics step later the hinges are where they were asked to be (a wrong joint order or a
    write that does not reach the solver fails here; the direction is ``_calibrate_start_signs``).

    One step, and a share rather than every env, on purpose: the sheet spawns 15 cm above the
    table (``cloth.start_height``) at the robot's home height, and a tall flap held there for
    longer meets the hand in some envs -- which no real reset does, as the reset drops the sheet
    at once. Measured on PhysX through real resets: every start lands within 0.00 deg.
    """

    n, dt = b.num_envs, float(b.cfg.sim.dt)
    all_ids = torch.arange(n, device=b.device)
    for s, (sh, table) in enumerate(zip(b.shapes, b.start_tables)):
        if table.shape[0] == 0:
            continue
        art = b.arts[s]
        want = table[_fold_index(b, s, table.shape[0])] * angle
        _reset_all_to(b, s, want, dy=-0.5)
        b.sim.step(render=False)
        b.scene.update(dt)
        # Instance order: row i of the articulation is env members[i].
        rows = b.pool.members(s)
        d = to_torch(art.data.joint_pos) - want[rows]
        # Wrapped: a full 180 deg start may read back as -180.
        err = torch.rad2deg(torch.atan2(torch.sin(d), torch.cos(d)).abs().amax(dim=1))
        ok = float((err < tol_deg).float().mean())
        size = b.env_size[rows]
        corr = float(torch.corrcoef(torch.stack((err, size)))[0, 1]) if err.numel() > 2 else 0.0
        print(
            f"[fold_pool] {sh.name}: start folds at {math.degrees(angle):.0f} deg land within "
            f"{tol_deg:g} deg in {ok * 100:.1f}% of envs (error median {float(err.median()):.3f}, "
            f"p95 {float(err.quantile(0.95)):.3f}, max {float(err.max()):.3f} deg; corr with "
            f"sheet size {corr:+.2f})",
            flush=True,
        )
        if ok < share:
            raise RuntimeError(
                f"[fold_pool] {sh.name}: start folds land within {tol_deg:g} deg in only "
                f"{ok * 100:.1f}% of envs (need {share * 100:.0f}%)"
            )
    # Back to the startup state: every env on shape 0, flat, no start angles left anywhere. The
    # first trainer reset redraws.
    for s, table in enumerate(b.start_tables):
        b.pool.set_start_joints(s, all_ids, torch.zeros((n, table.shape[1]), device=b.device))
    _reset_all_to(b, 0, torch.zeros((n, b.start_tables[0].shape[1]), device=b.device))
    b.sim.step(render=False)
    b.scene.update(dt)
