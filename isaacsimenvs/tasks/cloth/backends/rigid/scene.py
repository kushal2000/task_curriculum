"""Building the pool: every fold shape as a PhysX articulation (one USD per shape and per-env sheet
size, referenced from each env that carries the shape) gathered in the backend's ``pool``
(``pool.PlatePool``), the per-episode shape draw, and the folded-start tables. Functions of the
``RigidBackend`` (``b``). With ``fold_pool.goal_task`` also each env's one-fold task (``mdp/goal_tasks.py``):
its draw, its start hinge angles, the sheet yaw that puts its fold's travel direction in a sector."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from isaacsimenvs.tasks.cloth.backends.rigid import assets as fsa
from isaacsimenvs.tasks.cloth.backends.rigid import checks
from isaacsimenvs.tasks.cloth.backends.rigid import contacts
from isaacsimenvs.tasks.cloth.backends.rigid.physx import author_articulation_props, prepare_collider_asset
from isaacsimenvs.tasks.cloth.backends.rigid.pool import PlatePool, to_torch
from isaacsimenvs.tasks.cloth.backends.rigid.usd import bulk_edit_layer
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.mdp.goal_tasks import GoalTasks

__all__ = [
    "build_goal_tasks", "draw_env_sizes", "draw_episode", "goal_task_yaw", "init_start_folds", "install_pool",
    "start_goal_tasks",
]


# ---------------------------------------------------------------------------- sizes


def draw_env_sizes_raw(cfg, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The unquantised per-env sheet size and slab thickness draw (``fold_pool.size_seed``)."""
    p = cfg.fold_pool
    rng = np.random.default_rng(int(p.size_seed))
    size = rng.uniform(float(p.size_range[0]), float(p.size_range[1]), n)
    slab = rng.uniform(float(p.thickness_range[0]), float(p.thickness_range[1]), n)
    return size, slab


def draw_env_sizes(cfg, n: int):
    """Per-env sheet size and slab thickness [m], drawn once and quantised: every env references the
    USD of its own (size, slab)."""
    size, slab = draw_env_sizes_raw(cfg, n)
    q = float(cfg.physx_chain.size_quantum)
    size = np.round(size / q) * q
    # Never the range top itself: a 0.200 m box2-mid half-plate (200 x 100 x 2 mm) fails PhysX's
    # GPU convex cooking ("failed to cook GPU-compatible mesh"; 0.199 m and 0.150 m cook) and
    # falls back to CPU collision inside the GPU pipeline. Rounding put ~1/200 of envs there.
    top = float(cfg.fold_pool.size_range[1])
    size = np.minimum(size, np.round((top - q) / q) * q) if top - q >= float(cfg.fold_pool.size_range[0]) else size
    return size, np.round(slab / 1.0e-4) * 1.0e-4


# ---------------------------------------------------------------------------- the articulations


def _articulation_cls(b):
    from isaaclab.assets import Articulation

    if b.env_shape_np is None:
        return Articulation

    class _SubsetArticulation(Articulation):
        """A pool shape's articulation when only some envs carry it.

        The scene resets assets with GLOBAL env ids, which do not index this articulation's
        instances. Its per-instance reset state is the actuators' (implicit, stateless) and the
        external wrench buffers (never used on the pool), so resetting every instance is exact.
        """

        def reset(self, env_ids=None):
            super().reset(None)

    return _SubsetArticulation


def _spawn_shape(b, s: int, shape, usd_dir: Path, park_local, layer):
    """Write one USD per distinct (size, slab) of shape ``s`` and point each env's prim at its own,
    in ``layer`` (``usd.bulk_edit_layer``; the prims are on the stage when that closes).

    Exact physics at every size: ``fold_shapes.sized(name, size, thickness)`` is the shape itself at
    that size, so mass, inertia, hinge and seam anchors all come out of the same construction as
    the nominal sheet. Returns ``None``: the prims exist, nothing is left for a spawner to do.
    """
    from pxr import Gf, Sdf

    r, pc = b.cfg.rigid_cloth, b.cfg.physx_chain
    m = contacts.contact_margin(b)
    envs = b.pool_envs(s)
    keys = sorted(set(zip(np.round(b.env_size_np[envs], 6), np.round(b.env_slab_np[envs], 6))))
    usd_of = {}
    for size, slab in keys:
        sized = fs.sized(shape.name, float(size), float(slab))
        path = usd_dir / f"{shape.name}_{size * 1e3:.2f}mm_{slab * 1e3:.2f}mm.usda"
        fsa.write_shape_usd(sized, path, density=r.density)
        prepare_collider_asset(
            str(path), rest_offset=m, contact_offset=contacts.contact_offset(b),
            max_depenetration_velocity=pc.slat_max_depenetration_velocity,
        )
        # What a spawner's `articulation_props` would set; no spawner runs here.
        if author_articulation_props(
            str(path),
            self_collisions=m <= 0.0,
            position_iterations=pc.solver_position_iterations,
            velocity_iterations=pc.solver_velocity_iterations,
        ) != 1:
            raise RuntimeError(f"[fold_pool] {path} does not have exactly one articulation root")
        usd_of[(size, slab)] = str(path)

    pos = Gf.Vec3d(*[float(v) for v in park_local])
    for i in envs:
        key = (round(float(b.env_size_np[i]), 6), round(float(b.env_slab_np[i]), 6))
        spec = Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{i}/FoldPool{s}")
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = "Xform"
        spec.referenceList.Prepend(Sdf.Reference(usd_of[key]))
        t = Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        t.default = pos
        o = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
        o.default = ["xformOp:translate"]
    print(f"[fold_pool] {shape.name}: in {len(envs)} envs, {len(keys)} sheet sizes, one USD each", flush=True)
    return None


def install_pool(b) -> None:
    """Spawn every pool shape as its own articulation, all parked, and the pool. Sets ``arts``,
    ``pool``, ``rest_local``, ``keypoint_idx``, ``xy_scale``, ``env_size``."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    c, r, p = b.cfg.cloth, b.cfg.rigid_cloth, b.cfg.fold_pool
    if c.fold_axis != "x":
        raise ValueError(
            f"the pool's half fold carries +x over onto -x; cloth.fold_axis must be 'x', got {c.fold_axis!r}"
        )
    for sh in b.shapes:
        if abs(sh.size - c.size) > 1e-9:
            raise ValueError(f"{sh.name} is {sh.size} m square, the task's sheet is {c.size} m")

    verts, _indices = grid_mesh(c.size, c.resolution)
    b.rest_local = verts
    b.keypoint_idx = corner_indices(c.resolution, c.fold_axis)

    # The hinge actuator is always present -- it is the only place damping and armature can be
    # set -- and built with a negligible stiffness when DR will draw one, so the position gain
    # exists to be written.
    stiffness = float(r.joint_stiffness)
    if float(b.cfg.domain_randomization.cloth.hinge_stiffness[1]) > 0.0 and stiffness == 0.0:
        stiffness = 1.0e-9

    park = torch.tensor(
        [[0.0, p.park_distance + s * p.park_spacing, p.park_height] for s in range(len(b.shapes))],
        dtype=torch.float32,
    )
    usd_dir = Path(tempfile.mkdtemp(prefix="fold_pool_"))
    # Every shape's prims in ONE layer: each layer inserted into the stage recomposes all of it (7
    # layers, one per shape, took ~100 s at 7680 envs).
    from isaaclab.sim.utils import get_current_stage

    with bulk_edit_layer(get_current_stage(), "pool") as layer:
        for s, sh in enumerate(b.shapes):
            _spawn_shape(b, s, sh, usd_dir, park[s], layer)
    arts = []
    for s, sh in enumerate(b.shapes):
        cfg = ArticulationCfg(
            prim_path=f"/World/envs/env_.*/FoldPool{s}",
            spawn=None,                 # the prims exist (_spawn_shape)
            # Every shape starts parked; the first reset puts the drawn one on the table.
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(float(v) for v in park[s])),
            actuators={
                "hinges": ImplicitActuatorCfg(
                    joint_names_expr=[f"{fsa.JOINT_PREFIX}.*"],
                    stiffness=stiffness,
                    damping=r.joint_damping,
                    armature=r.joint_armature,
                    effort_limit_sim=None,
                )
            },
        )
        art = _articulation_cls(b)(cfg)
        b.scene.articulations[f"fold_pool_{s}"] = art
        arts.append(art)
    b.arts = arts

    # Per-env geometry, as tensors: in-plane factor, size [m], and the reset-centre shift that
    # keeps a large sheet's far side within the arm's reach.
    b.xy_scale = torch.tensor(b.env_size_np / float(c.size), device=b.device, dtype=torch.float32)
    b.env_size = torch.tensor(b.env_size_np, device=b.device, dtype=torch.float32)
    reach = torch.tensor([*p.reach_shift, 0.0], device=b.device, dtype=torch.float32)
    growth = (b.env_size - float(c.size)).clamp(min=0.0) * (0.5 * math.sqrt(2.0))
    centre_offset = growth.unsqueeze(1) * reach.unsqueeze(0)

    carried = None
    if b.env_shape_np is not None:
        shape_of = torch.as_tensor(b.env_shape_np, device=b.device)
        carried = shape_of.unsqueeze(0) == torch.arange(len(b.shapes), device=b.device).unsqueeze(1)
    pool = PlatePool(
        arts,
        mappings=[fs.grid_plate_map(sh, c.resolution) for sh in b.shapes],
        body_prefix=fsa.BODY_PREFIX,
        num_envs=b.num_envs,
        device=b.device,
        xy_scale=b.xy_scale,
        park_local=park,
        centre_offset=centre_offset,
        spawn_z=float(b.cfg.reset.table_reset_z) + c.start_height,
        env_origins=lambda: b.scene.env_origins,
        carried=carried,
    )
    b.pool = pool
    _write_object_scale(b)
    print(
        "[fold_pool] pool: "
        + "; ".join(f"{sh.name} ({sh.describe()})" for sh in b.shapes)
        + f"\n[fold_pool] draw weights {b.shape_weights.tolist()}, parked at +y "
        f"{p.park_distance} m, z {p.park_height} m; USDs in {usd_dir}"
        + (
            "\n[fold_pool] ONE shape per env, fixed: "
            + ", ".join(f"{sh.name} {int((b.env_shape_np == k).sum())}" for k, sh in enumerate(b.shapes))
            + " envs"
            if b.env_shape_np is not None
            else "\n[fold_pool] every env carries every shape; one drawn per episode"
        )
        + f"\n[fold_pool] per-env sheet size {b.env_size_np.min() * 100:.1f}-"
        f"{b.env_size_np.max() * 100:.1f} cm, slab {b.env_slab_np.min() * 1e3:.2f}-"
        f"{b.env_slab_np.max() * 1e3:.2f} mm, max reach shift {float(growth.max()) * 100:.1f} cm",
        flush=True,
    )


def _write_object_scale(b) -> None:
    """``object_scales`` per env: this env's sheet, not the nominal one."""
    c = b.cfg.cloth
    base = float(b.cfg.reward.object_base_size)
    size = torch.tensor(b.env_size_np, device=b.device, dtype=torch.float32)
    # The thickness the hand meets is slab + 2 x contact margin -- by construction the cloth's
    # `thickness` at the nominal slab -- so a thicker slab ADDS its difference to it.
    slab = torch.tensor(b.env_slab_np, device=b.device, dtype=torch.float32)
    felt = float(c.thickness) + slab - float(b.shapes[0].thickness)
    b.object_scale[:] = torch.stack((size, size, felt), dim=-1) / base


# ---------------------------------------------------------------------------- per episode


def draw_episode(b, env_ids: torch.Tensor) -> None:
    """The episode's shape (drawn by ``fold_pool.weights``, or the env's own) and folded start."""
    if b.env_shape_np is not None:
        # One shape per env: the env's own, every episode.
        draw = b.pool.first_carried()[env_ids.to(b.pool.active.device)]
    else:
        draw = torch.multinomial(b.shape_weights, int(env_ids.numel()), replacement=True)
    b.pool.select(env_ids, draw.to(b.pool.active.device))
    if b.start_tables is not None:
        b.draw_start_folds(env_ids, draw.to(b.device))
    if b.goal_tasks is not None:
        draw_goal_tasks(b, env_ids, draw.to(b.device))


def init_start_folds(b) -> None:
    """Per shape, its start folds as ``(F, num_joints)`` hinge directions plus ``(F, num_joints)``
    part indices (which of the start's independently drawn angles each hinge takes), and the
    startup proof. ``init_fold_starts: single`` has one part per start; ``other`` uses
    ``FoldShape.start_poses`` and may leave a shape with no start (``F = 0``: always flat).

    Off (``start_tables = None``) unless ``fold_pool.init_fold_prob > 0``.
    """
    p = b.cfg.fold_pool
    b.start_folded = torch.zeros(b.num_envs, device=b.device, dtype=torch.bool)
    b.start_tables = None
    if float(p.init_fold_prob) <= 0.0:
        return
    prefix = fsa.JOINT_PREFIX
    tables, parts, names = [], [], []
    for s, sh in enumerate(b.shapes):
        joints = list(b.arts[s].joint_names)
        if p.init_fold_starts == "other":
            poses = sh.start_poses()
        else:
            poses = {n: (tuple(d),) for n, d in sh.start_folds().items()}
            if not poses:
                raise ValueError(f"[fold_pool] {sh.name} has no single-stage fold to start from")
        t = torch.zeros((len(poses), len(joints)), device=b.device)
        c = torch.zeros((len(poses), len(joints)), device=b.device, dtype=torch.long)
        for k, pose in enumerate(poses.values()):
            for i, part in enumerate(pose):
                for child in part:
                    t[k, joints.index(f"{prefix}{child}")] = 1.0
                    c[k, joints.index(f"{prefix}{child}")] = i
        tables.append(t)
        parts.append(c)
        names.append(list(poses))
    b.start_tables = tables
    b.start_parts = parts
    b.start_names = names
    b.start_num_parts = max([int(c.max()) + 1 for c, t in zip(parts, tables) if t.numel()] or [1])
    lo, hi = (float(v) for v in p.init_fold_deg)
    print(
        f"[fold_pool] partly folded starts: p={float(p.init_fold_prob):g}, {lo:g}-{hi:g} deg, "
        + "; ".join(f"{sh.name}: {n}" for sh, n in zip(b.shapes, names)),
        flush=True,
    )
    checks.calibrate_start_signs(b)
    checks.verify_start_folds(b, math.radians(hi))


def draw_start_folds(b, env_ids: torch.Tensor, draw: torch.Tensor) -> None:
    """Which of ``env_ids`` start partly folded, in which fold, at which angles (for shapes ``draw``)."""
    p = b.cfg.fold_pool
    n = int(env_ids.numel())
    folded = torch.rand(n, device=b.device) < float(p.init_fold_prob)
    lo, hi = (math.radians(float(v)) for v in p.init_fold_deg)
    ang = torch.empty((n, b.start_num_parts), device=b.device).uniform_(lo, hi)
    ang *= folded.unsqueeze(1)
    for s, (table, part) in enumerate(zip(b.start_tables, b.start_parts)):
        on = draw == s
        ids = env_ids[on]
        if ids.numel() == 0:
            continue
        if table.shape[0] == 0:  # no start but the target's: flat
            folded &= ~on
            b.pool.set_start_joints(s, ids, torch.zeros((ids.numel(), table.shape[1]), device=b.device))
            continue
        k = torch.randint(table.shape[0], (ids.numel(),), device=b.device)
        b.pool.set_start_joints(s, ids, table[k] * torch.gather(ang[on], 1, part[k]))
    b.start_folded[env_ids] = folded


# ---------------------------------------------------------------------------- goal tasks


def build_goal_tasks(b) -> GoalTasks | None:
    """The one-fold goal tables of the pool (``fold_pool.goal_task``), or ``None`` when off. No
    draws: each env's task is drawn at :func:`start_goal_tasks`."""
    g = b.cfg.fold_pool.goal_task
    if g.mode == "off":
        return None
    joints = [list(a.joint_names) for a in b.arts]
    return GoalTasks(
        b.shapes, int(b.cfg.cloth.resolution), g.mode, joints, fsa.JOINT_PREFIX, b.device,
        accept_lookalike=bool(g.accept_lookalike), read_hinge_abs=lambda: hinge_abs(b),
    )


def start_goal_tasks(b) -> None:
    """Each env's first task, on the shape the startup proofs left active."""
    g = b.cfg.fold_pool.goal_task
    all_ids = torch.arange(b.num_envs, device=b.device)
    draw_goal_tasks(b, all_ids, b.pool.active.to(b.device))
    print(
        f"[goal_task] accept_lookalike {b.goal_tasks.accept_lookalike}, travel sector {float(g.sector_deg):g} deg "
        f"about {float(g.sector_centre_deg):g} deg (margin {float(g.sector_margin_deg):g})",
        flush=True,
    )


def draw_goal_tasks(b, env_ids: torch.Tensor, draw: torch.Tensor) -> None:
    """The task of ``env_ids``, running shapes ``draw``; with ``any_fold`` also its start hinge
    angles (and whether it starts folded, for the logs)."""
    gt = b.goal_tasks
    gt.draw_envs(env_ids, draw)
    if not gt.any_fold:
        return
    task = gt.task[env_ids]
    shape = gt.task_shape[task]
    for s, art in enumerate(b.arts):
        on = shape == s
        if bool(on.any()):
            b.pool.set_start_joints(s, env_ids[on], gt.q_start[task[on], : art.num_joints])
    b.start_folded[env_ids] = gt.q_start[task].abs().amax(dim=1) > 1e-6


def goal_task_yaw(b, env_ids: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """The reset yaw of ``env_ids``: the sheet's own, or (``goal_task.sector_deg`` < 360) one that
    puts the designated fold's travel direction in the sector."""
    gt, g = b.goal_tasks, b.cfg.fold_pool.goal_task
    if gt is None or gt.task is None or float(g.sector_deg) >= 360.0:
        return yaw
    return gt.sector_yaw(gt.task[env_ids], g.sector_deg, g.sector_centre_deg, g.sector_margin_deg).to(yaw.dtype)


def hinge_abs(b) -> torch.Tensor:
    """``(N, J)`` |hinge angle| of each env's active shape, zero-padded to the widest shape."""
    gt = b.goal_tasks
    q = torch.zeros((b.num_envs, gt.m_q.shape[-1]), device=b.device)
    active = b.pool.active.to(b.device)
    for s, art in enumerate(b.arts):
        ids = b.pool.members(s)
        jp = to_torch(art.data.joint_pos)
        sel = active[ids] == s
        q[ids[sel], : jp.shape[1]] = jp[sel].abs().to(q.dtype)
    return q
