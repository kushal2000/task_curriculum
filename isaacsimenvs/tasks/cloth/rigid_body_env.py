"""The fold task on a rigid-body sheet: two plates and one hinge, one size design per env (PhysX).

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv/isaacsim``), on the Play task's ``PlayEnv``, with the fold
task of ``ClothTaskMixin`` -- the same targets, reward, success and observation the VBD cloth env
runs, so a policy trained here is evaluated there unchanged.

    OMNI_KIT_ACCEPT_EULA=YES .venv/isaacsim/bin/python scripts/training/train.py --headless \\
        --task Isaacsimenvs-RigidBodyCloth-Direct-v0 --experiment rigid_body_cloth

**The sheet** (``utils/sheet_designs.single_hinge``): two boxes ``length/2 x width x thickness``
joined by a revolute hinge on the top surface at the crease, so folding the ``+x`` plate 180 deg
lays it one thickness above the ``-x`` plate -- exactly the fold target (reflect across the crease,
lift by the ply gap). Env ``i`` carries design ``perm[i % n_designs]`` of the design grid (lengths
x widths x thicknesses; ``sheet_designs.design_of``), fixed for the run.

**Authoring.** One template sheet is written in env 0 with the USD physics schemas, copied into every
env with ``Sdf.CopySpec`` (which remaps the joint's body targets into each env) inside one
``Sdf.ChangeBlock`` in the stage's root layer, then each env's dimensions are set on the copies.
Native box colliders (a scaled ``Cube``), no USD file per design. Every env has the same prim
structure, so ONE articulation spans all envs.

**Contacts.** Play's offsets (contact 2 mm, rest 0). Frictions -- plates, robot, table -- are
written with the PhysX tensor API and combine by PhysX's default ``average``.

**Material DR, every reset** (``sheet.randomization``): hinge damping (log-uniform) and stiffness
(uniform), and the plate / robot / table frictions (multipliers quantised to ``friction_buckets``
levels: PhysX caps a scene at 64K materials).

**Conventions** (the VBD env's, reproduced so the observation matches; see
``tests/test_physx_cloth_conventions.py``): root-pose writes to ``env.object`` are reordered
wxyz -> xyzw before the yaw decode, and the goal quaternion is ``quat_from_matrix``'s components
relabelled (x, y, z, w) -> read as (w, x, y, z).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.fold_task import (
    ClothAsRigidObject,
    ClothTaskMixin,
    box_mass_properties,
    corner_indices,
    design_grid,
    design_of,
    grid_mesh,
    single_hinge,
    task_yaw,
)
from isaacsimenvs.tasks.cloth.rigid_body_env_cfg import RigidBodyClothEnvCfg
from isaacsimenvs.tasks.play.play_env import PlayEnv

__all__ = ["RigidBodyClothEnv"]

SHEET = "Sheet"
PLATES = ("minus", "plus")      # body names; "minus" is the articulation root
HINGE = "hinge"                 # joint name


# ---------------------------------------------------------------------------- helpers


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (..., 3) by wxyz ``q`` (..., 4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """``(n,)`` yaw -> ``(n, 4)`` wxyz about +z."""
    out = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    out[:, 0] = torch.cos(0.5 * yaw)
    out[:, 3] = torch.sin(0.5 * yaw)
    return out


def _reorder_object_pose_writes(obj) -> None:
    """Newton's ``patches.install_pose_write_conversion`` on ``env.object``: wxyz -> xyzw.

    The adapter decodes the reset yaw as wxyz from what it RECEIVES, and on Newton (the VBD eval) it
    receives the task's quaternion reordered; doing the same here keeps the decoded heading of a
    random draw and of the pinned eval start identical across the two simulators.
    """
    for method in ("write_root_pose_to_sim", "write_root_state_to_sim", "write_root_link_pose_to_sim"):
        original = getattr(obj, method, None)
        if original is None:
            continue

        def converted(value, *args, _orig=original, **kwargs):
            value = value.clone()
            value[..., 3:7] = torch.cat((value[..., 4:7], value[..., 3:4]), dim=-1)
            return _orig(value, *args, **kwargs)

        setattr(obj, method, converted)


def _apply_physx_api(prim, schema: str, attrs: dict) -> None:
    """Apply a PhysX API schema by name and author its attributes."""
    if schema not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(schema)
    for name, (vtype, value) in attrs.items():
        prim.CreateAttribute(name, vtype).Set(value)


def _data(art, *names):
    """The first of ``names`` the articulation data has (Isaac Lab renamed the link/COM fields)."""
    for n in names:
        v = getattr(art.data, n, None)
        if v is not None:
            return v
    raise AttributeError(f"articulation data has none of {names}")


class SheetMesh:
    """The sheet's grid points carried by the two plates, as the cloth mesh the task reads
    (``data.nodal_pos_w`` / ``nodal_vel_w``, flattened ``(N * P, 3)``).

    ``local`` ``(N, P, 3)``: each env's grid in the plate (= sheet, at rest) frame; points with
    ``x > 0`` ride the ``+x`` plate, the crease row and the rest the ``-x`` plate.
    """

    def __init__(self, art, local: torch.Tensor, on_plus: torch.Tensor) -> None:
        self._art = art
        self._local = local
        self._on_plus = on_plus
        self._idx = None
        self.data = _SheetMeshData(self)

    def body_index(self) -> torch.Tensor:
        """``(P,)`` body index of each grid point, by name on first use (bodies exist once played)."""
        if self._idx is None:
            names = list(self._art.body_names)
            missing = [p for p in PLATES if p not in names]
            if missing:
                raise RuntimeError(f"sheet articulation lacks bodies {missing}; has {names}")
            ids = torch.tensor([names.index(p) for p in PLATES], device=self._local.device)
            self._idx = ids[self._on_plus.long()]
        return self._idx

    def body_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, B, 3)`` link positions and ``(N, B, 4)`` wxyz orientations."""
        return _data(self._art, "body_link_pos_w", "body_pos_w"), _data(self._art, "body_link_quat_w", "body_quat_w")

    def positions(self) -> torch.Tensor:
        idx = self.body_index()
        pos, quat = self.body_pose()
        return pos[:, idx] + _quat_apply(quat[:, idx], self._local)

    def velocities(self) -> torch.Tensor:
        idx = self.body_index()
        _, quat = self.body_pose()
        lin = _data(self._art, "body_link_lin_vel_w", "body_lin_vel_w")[:, idx]
        ang = _data(self._art, "body_link_ang_vel_w", "body_ang_vel_w")[:, idx]
        return lin + torch.cross(ang, _quat_apply(quat[:, idx], self._local), dim=-1)


class _SheetMeshData:
    def __init__(self, owner: SheetMesh) -> None:
        self._owner = owner

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        return self._owner.positions().reshape(-1, 3)

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        return self._owner.velocities().reshape(-1, 3)


class SheetAsRigidObject(ClothAsRigidObject):
    """``env.object``: the Play task's rigid-object surface over the sheet (reads from
    ``ClothAsRigidObject``: keypoint centroid, Kabsch-fitted corner rotation). A reset write places
    the sheet flat at the requested XY and yaw, at rest, hinge at 0."""

    def __init__(self, mesh: SheetMesh, *, art, place, **kwargs) -> None:
        super().__init__(mesh, spawn_z=0.0, **kwargs)
        self._art = art
        self._place = place   # callable (env ids, yaw) -> (n, 3) world position of the sheet root

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Flat, at rest, at the requested YAW (``task_yaw``); position from the env's placement
        (the requested XY is not used: it does not know the sheet's size)."""
        ids = self._ids(env_ids)
        yaw = task_yaw(root_pose)
        pose = torch.cat((self._place(ids, yaw), _yaw_quat(yaw)), dim=-1)   # wxyz, Isaac Lab 2.3
        self._art.write_root_pose_to_sim(pose, env_ids=ids)
        self._stop(ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        self._stop(self._ids(env_ids))

    def _stop(self, ids: torch.Tensor) -> None:
        n = ids.shape[0]
        self._art.write_root_velocity_to_sim(torch.zeros((n, 6), device=self._device), env_ids=ids)
        zeros = torch.zeros((n, self._art.num_joints), device=self._device)
        self._art.write_joint_state_to_sim(zeros, zeros, env_ids=ids)
        self._art.set_joint_position_target(zeros, env_ids=ids)

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self._num_envs, device=self._device)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

    @property
    def root_physx_view(self):
        """The sheet's own view. Play's startup material pass writes ``env.object`` through it;
        ``RigidBodyClothEnv._init_sheet_physics`` then sets the plates' configured friction."""
        return self._art.root_physx_view


# ---------------------------------------------------------------------------- the env


class RigidBodyClothEnv(ClothTaskMixin, PlayEnv):
    """Fold the ``+x`` half of a rigid single-hinge sheet onto the ``-x`` half."""

    cfg: RigidBodyClothEnvCfg

    def __init__(self, cfg: RigidBodyClothEnvCfg, render_mode: str | None = None, **kwargs):
        c, s = cfg.cloth, cfg.sheet
        if c.task != "fold":
            raise ValueError(f"the rigid-body cloth env is fold only; cloth.task={c.task!r}")
        if c.fold_axis != "x":
            raise ValueError(f"the sheet folds +x onto -x; cloth.fold_axis must be 'x', got {c.fold_axis!r}")
        # Per-env designs, fixed for the run: set BEFORE super().__init__, whose scene hook builds them.
        grid = design_grid(int(s.design_levels), tuple(s.length_range), tuple(s.width_range), tuple(s.thickness_range))
        n = int(cfg.scene.num_envs)
        self._n_designs = len(grid)
        self._design_np = design_of(np.arange(n), self._n_designs)
        self._dims_np = grid[self._design_np]                       # (N, 3) length, width, thickness
        self._dr_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        self._init_sheet_physics()
        self._dr_ready = True

    # ------------------------------------------------------------------ construction

    def _setup_scene(self) -> None:
        from isaacsimenvs.utils.setup_timer import stamp

        stamp("scene: start")
        super()._setup_scene()
        stamp("scene: base (robot, table, object) done")
        self._install_sheet()
        stamp("scene: sheets installed")
        _reorder_object_pose_writes(self.object)

    def _install_sheet(self) -> None:
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        c, s = self.cfg.cloth, self.cfg.sheet
        verts, _ = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        self._cloth_kp_idx = corner_indices(c.resolution, c.fold_axis)
        self._neutralise_rigid_object()
        self._reshape_goal_marker()

        self._dims = torch.tensor(self._dims_np, device=self.device, dtype=torch.float32)
        self._author_sheets()

        rd = s.randomization
        stiffness = float(s.hinge_stiffness)
        # The position gain must exist to be written per reset: a negligible one when DR draws it.
        if rd.enabled and float(rd.hinge_stiffness[1]) > 0.0 and stiffness == 0.0:
            stiffness = 1.0e-9
        art = Articulation(ArticulationCfg(
            prim_path=f"/World/envs/env_.*/{SHEET}",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, float(self._spawn_local_z(0)))),
            actuators={HINGE: ImplicitActuatorCfg(
                joint_names_expr=[HINGE],
                stiffness=stiffness,
                damping=float(s.hinge_damping),
                armature=float(s.hinge_armature),
                effort_limit_sim=None,
            )},
        ))
        self.scene.articulations["sheet"] = art
        self._sheet = art

        rest = torch.tensor(verts, device=self.device, dtype=torch.float32)          # (P, 3)
        local = rest.unsqueeze(0) * self._scale3()                                    # (N, P, 3)
        mesh = SheetMesh(art, local, on_plus=rest[:, 0] > 1e-9)
        self._cloth = mesh

        self.object = SheetAsRigidObject(
            mesh,
            art=art,
            place=self._place_sheet,
            num_envs=self.num_envs,
            rest_local=verts,
            keypoint_idx=self._cloth_kp_idx,
            corner_idx=self._cloth_kp_idx,
            corner_rest=[verts[i] for i in self._cloth_kp_idx],
            mass=float(s.density) * float(c.size) ** 2,
            device=self.device,
        )
        self._write_object_scale()
        d = self._dims_np
        print(
            f"[rigid_body] {self.num_envs} envs, {self._n_designs} designs (env i -> design perm[i % "
            f"{self._n_designs}]): length {d[:, 0].min() * 100:.1f}-{d[:, 0].max() * 100:.1f} cm, width "
            f"{d[:, 1].min() * 100:.1f}-{d[:, 1].max() * 100:.1f} cm, thickness {d[:, 2].min() * 1e3:.1f}-"
            f"{d[:, 2].max() * 1e3:.1f} mm; placed inside table x {list(s.place_x)}, y {list(s.place_y)} m",
            flush=True,
        )

    def _spawn_local_z(self, i: int) -> float:
        """Nominal env-frame height of env ``i``'s sheet root at rest on the table (authoring)."""
        table_top = float(self.cfg.reset.table_reset_z) + float(self.cfg.cloth.table_half_thickness)
        return table_top + 0.5 * float(self._dims_np[i, 2]) + float(self.cfg.sheet.spawn_clearance)

    def _place_sheet(self, ids: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """``(n, 3)`` world position of the sheet root at a reset (``SheetCfg.place_x``/``place_y``).

        XY: uniform over every centre that keeps the yawed sheet's bounding box inside the placement
        region; centred where it does not fit, and everywhere when the reset position noise is zero
        (the eval protocol). Z: on THIS episode's table (its height is drawn per reset), the plates'
        underside ``spawn_clearance`` above it."""
        s, r = self.cfg.sheet, self.cfg.reset
        L, W, T = self._dims[ids, 0], self._dims[ids, 1], self._dims[ids, 2]
        c, sn = torch.cos(yaw).abs(), torch.sin(yaw).abs()
        half = torch.stack((0.5 * (L * c + W * sn), 0.5 * (L * sn + W * c)), dim=-1)      # (n, 2)
        lo = torch.tensor([float(s.place_x[0]), float(s.place_y[0])], device=self.device)
        hi = torch.tensor([float(s.place_x[1]), float(s.place_y[1])], device=self.device)
        room = (hi - lo - 2.0 * half).clamp(min=0.0)
        pinned = float(r.reset_position_noise_x) == 0.0 and float(r.reset_position_noise_y) == 0.0
        u = torch.full_like(half, 0.5) if pinned else torch.rand_like(half)
        fits = (hi - lo - 2.0 * half) >= 0.0
        xy = torch.where(fits, lo + half + u * room, 0.5 * (lo + hi))
        xy = xy + torch.tensor([float(v) for v in r.table_reset_xy], device=self.device)
        top = self._table_z_per_env[ids] + float(self.cfg.cloth.table_half_thickness)
        z = top + 0.5 * T + float(s.spawn_clearance)
        return torch.cat((xy, z.unsqueeze(1)), dim=-1) + self.scene.env_origins[ids]

    def _author_sheets(self) -> None:
        """One template sheet, written with the USD schema API into a private in-memory stage (Kit
        does not watch it: written into the scene's stage, the same ~60 edits took 78 s at 6144 envs,
        each recomposing the whole scene), then ``Sdf.CopySpec``-ed into every env of the scene's
        root layer and given that env's dimensions, all inside one ``Sdf.ChangeBlock``."""
        from isaaclab.sim.utils import get_current_stage
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

        from isaacsimenvs.utils.setup_timer import stamp

        s = self.cfg.sheet
        layer = get_current_stage().GetRootLayer()
        stage = Usd.Stage.CreateInMemory()          # the template's own stage
        tpl = f"/{SHEET}"
        dbl = UsdGeom.XformOp.PrecisionDouble

        sheet = UsdGeom.Xform.Define(stage, tpl)
        sheet.AddTranslateOp(precision=dbl)
        colors = {"minus": (0.29, 0.56, 0.76), "plus": (0.95, 0.68, 0.22)}
        for name in PLATES:
            body = UsdGeom.Xform.Define(stage, f"{tpl}/{name}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateMassAttr(1.0)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
            _apply_physx_api(body, "PhysxRigidBodyAPI", {
                "physxRigidBody:maxDepenetrationVelocity": (Sdf.ValueTypeNames.Float, float(s.max_depenetration_velocity)),
            })
            if name == PLATES[0]:
                UsdPhysics.ArticulationRootAPI.Apply(body)
                _apply_physx_api(body, "PhysxArticulationAPI", {
                    "physxArticulation:enabledSelfCollisions": (Sdf.ValueTypeNames.Bool, False),
                    "physxArticulation:solverPositionIterationCount": (Sdf.ValueTypeNames.Int, int(s.solver_position_iterations)),
                    "physxArticulation:solverVelocityIterationCount": (Sdf.ValueTypeNames.Int, int(s.solver_velocity_iterations)),
                })
            cube = UsdGeom.Cube.Define(stage, f"{tpl}/{name}/collider")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp(precision=dbl)
            cube.AddScaleOp(precision=dbl)
            cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colors[name])]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            _apply_physx_api(cube.GetPrim(), "PhysxCollisionAPI", {
                "physxCollision:contactOffset": (Sdf.ValueTypeNames.Float, float(s.contact_offset)),
                "physxCollision:restOffset": (Sdf.ValueTypeNames.Float, float(s.rest_offset)),
            })
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{tpl}/{HINGE}")
        joint.CreateBody0Rel().SetTargets([f"{tpl}/{PLATES[0]}"])
        joint.CreateBody1Rel().SetTargets([f"{tpl}/{PLATES[1]}"])
        # Joint X along the crease, pointing -y: a positive angle turns the + plate up and over
        # (sheet_designs.single_hinge's axis).
        joint.CreateAxisAttr(UsdPhysics.Tokens.x)
        rot = Gf.Quatf(math.cos(-math.pi / 4), Gf.Vec3f(0.0, 0.0, math.sin(-math.pi / 4)))
        joint.CreateLocalRot0Attr(rot)
        joint.CreateLocalRot1Attr(rot)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLowerLimitAttr(float(s.hinge_limits_deg[0]))
        joint.CreateUpperLimitAttr(float(s.hinge_limits_deg[1]))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
        stamp("sheet: template authored")

        src = stage.GetRootLayer()

        def put(path: str, value) -> None:
            spec = layer.GetAttributeAtPath(Sdf.Path(path))
            if spec is None:
                raise RuntimeError(f"no attribute spec at {path}")
            spec.default = value

        with Sdf.ChangeBlock():
            for i in range(self.num_envs):
                root = f"/World/envs/env_{i}/{SHEET}"
                Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{i}")
                if not Sdf.CopySpec(src, tpl, layer, root):
                    raise RuntimeError(f"Sdf.CopySpec {tpl} -> {root} failed")
                length, width, thick = (float(v) for v in self._dims_np[i])
                spec = single_hinge(length, width, thick)
                put(f"{root}.xformOp:translate", Gf.Vec3d(0.0, 0.0, self._spawn_local_z(i)))
                for box in spec.plates:
                    m, com, inertia = box_mass_properties(box, float(s.density))
                    put(f"{root}/{box.name}.physics:mass", float(m))
                    put(f"{root}/{box.name}.physics:centerOfMass", Gf.Vec3f(*map(float, com)))
                    put(f"{root}/{box.name}.physics:diagonalInertia", Gf.Vec3f(*map(float, inertia)))
                    put(f"{root}/{box.name}/collider.xformOp:translate", Gf.Vec3d(*box.center))
                    put(f"{root}/{box.name}/collider.xformOp:scale", Gf.Vec3d(*(2.0 * h for h in box.half_extents)))
                anchor = Gf.Vec3f(*map(float, spec.hinges[0].anchor))
                put(f"{root}/{HINGE}.physics:localPos0", anchor)
                put(f"{root}/{HINGE}.physics:localPos1", anchor)
        stamp(f"sheet: {self.num_envs} sheets authored")

    # ------------------------------------------------------------------ geometry overrides (per env)

    def _scale3(self) -> torch.Tensor:
        """``(N, 1, 3)``: the env's length and width over the nominal grid's ``cloth.size``; 1 on z."""
        size = float(self.cfg.cloth.size)
        d = self._dims
        return torch.stack((d[:, 0] / size, d[:, 1] / size, torch.ones_like(d[:, 0])), dim=-1).unsqueeze(1)

    def _write_object_scale(self) -> None:
        """``object_scales``: each env's ``[length, width, thickness]`` over ``reward.object_base_size``."""
        self._object_scale_per_env[:] = self._dims / float(self.cfg.reward.object_base_size)

    def _task_span(self) -> torch.Tensor:
        return self._dims[:, 0]

    def _init_fold_targets(self) -> None:
        """The shared targets on the nominal grid, then per env: scaled in-plane to the env's sheet
        and lifted by its ply gap (the plate thickness, as the hinge on the top surface lays it)."""
        self.cfg.cloth.self_contact_radius = 0.0       # the lift is per env, added below
        ClothTaskMixin._init_fold_targets(self)
        s3 = self._scale3()
        lift = self._dims[:, 2].view(-1, 1)
        self._corner_rest_offsets = self._corner_rest_offsets.unsqueeze(0) * s3
        self._mv_folded_rest = self._mv_folded_rest.unsqueeze(0) * s3
        self._kp_folded_rest = self._kp_folded_rest.unsqueeze(0) * s3
        self._mv_folded_rest[..., 2] += lift
        self._kp_folded_rest[..., 2] += lift
        self._mv_flat_rest = self._mv_flat_rest.unsqueeze(0) * s3
        self._kp_flat_rest = self._kp_flat_rest.unsqueeze(0) * s3
        self._st_rest_centroid = self._st_rest_centroid.unsqueeze(0) * s3[:, 0]
        self._st_rest_centred = self._st_rest_centred.unsqueeze(0) * s3

    def _stationary_frame(self, parts: torch.Tensor):
        cur = parts[:, self._stationary_idx, :]
        centroid = cur.mean(dim=1)
        q = cur - centroid.unsqueeze(1)
        h = torch.einsum("nki,nkj->nij", self._st_rest_centred, q)
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        d = torch.ones((cur.shape[0], 3), device=cur.device, dtype=cur.dtype)
        d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
        bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
        if bool(bad.any()):
            r = r.clone()
            r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
        return r, centroid

    def _folded_w(self, folded_rest: torch.Tensor) -> torch.Tensor:
        r, centroid = self._stationary_frame(self._particles_w())
        local = folded_rest - self._st_rest_centroid.unsqueeze(1)
        return torch.einsum("nij,nmj->nmi", r, local) + centroid.unsqueeze(1)

    def _sync_observed_keypoints(self) -> None:
        self._keypoint_offsets[:] = self._corner_rest_offsets
        self._keypoint_offsets_fixed[:] = self._corner_rest_offsets

    def footprint_ratio(self) -> torch.Tensor:
        """Extent along the sheet's own fold axis over its length."""
        parts = self._particles_w()
        r, _ = self._stationary_frame(parts)
        coord = torch.einsum("npi,ni->np", parts, r[:, :, 0])
        return (coord.amax(dim=1) - coord.amin(dim=1)) / self._dims[:, 0]

    def fold_fraction(self) -> torch.Tensor:
        """Progress in [0, 1]: a flat sheet's far corners are ~one length (plus the lift) from target."""
        span = torch.sqrt(self._dims[:, 0] ** 2 + self._dims[:, 2] ** 2)
        return (1.0 - self.fold_error() / span).clamp(0.0, 1.0)

    def _goal_quat_from_matrix(self, rot: torch.Tensor) -> torch.Tensor:
        # Isaac Lab 2.3's helper is wxyz; hand the task Newton's (x, y, z, w) relabelling.
        return quat_from_matrix(rot)[..., [1, 2, 3, 0]]

    # ------------------------------------------------------------------ material DR

    def _init_sheet_physics(self) -> None:
        """Base frictions (the plates', and the robot's / table's as the Play material pass left
        them), the first DR draw for every env, and a read-back of what it wrote."""
        s = self.cfg.sheet
        self._sheet_mats = self._sheet.root_physx_view.get_material_properties().clone()
        self._sheet_mats[..., 0:2] = float(s.friction)
        self._sheet_mats[..., 2] = 0.0
        self._robot_mats_base = self.robot.root_physx_view.get_material_properties().clone()
        self._table_mats_base = self.table.root_physx_view.get_material_properties().clone()
        self._robot_mats = self._robot_mats_base.clone()
        self._table_mats = self._table_mats_base.clone()
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._sheet.root_physx_view.set_material_properties(self._sheet_mats, all_ids.cpu())
        self._randomize(all_ids)
        self._check_dr_writes(all_ids[: min(64, self.num_envs)])
        rd = s.randomization
        print(
            f"[rigid_body] DR every reset: {'on' if rd.enabled else 'OFF'}; hinge damping "
            f"{s.hinge_damping:g} x logU{list(rd.hinge_damping_scale)}, stiffness U{list(rd.hinge_stiffness)}; "
            f"friction x{list(rd.sheet_friction_scale)} on plates ({s.friction:g}), "
            f"x{list(rd.robot_friction_scale)} on the robot, x{list(rd.table_friction_scale)} on the table "
            f"({rd.friction_buckets} levels each); offsets contact {s.contact_offset:g} / rest {s.rest_offset:g} m",
            flush=True,
        )

    def _randomize(self, env_ids: torch.Tensor) -> None:
        s = self.cfg.sheet
        rd = s.randomization
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        n = int(ids.numel())
        if n == 0 or not rd.enabled:
            return
        lo, hi = (math.log(float(v)) for v in rd.hinge_damping_scale)
        damping = float(s.hinge_damping) * torch.exp(torch.empty(n, device=self.device).uniform_(lo, hi))
        stiffness = torch.empty(n, device=self.device).uniform_(*(float(v) for v in rd.hinge_stiffness))
        self._sheet.write_joint_damping_to_sim(damping.view(-1, 1), env_ids=ids)
        self._sheet.write_joint_stiffness_to_sim(stiffness.view(-1, 1), env_ids=ids)

        cpu = ids.cpu()
        b = int(rd.friction_buckets)

        def bucket(scale) -> torch.Tensor:
            levels = torch.linspace(float(scale[0]), float(scale[1]), b)
            return levels[torch.randint(0, b, (n,))].view(-1, 1, 1)

        self._sheet_mats[cpu, :, 0:2] = float(s.friction) * bucket(rd.sheet_friction_scale)
        self._robot_mats[cpu, :, 0:2] = self._robot_mats_base[cpu, :, 0:2] * bucket(rd.robot_friction_scale)
        self._table_mats[cpu, :, 0:2] = self._table_mats_base[cpu, :, 0:2] * bucket(rd.table_friction_scale)
        self._sheet.root_physx_view.set_material_properties(self._sheet_mats, cpu)
        self.robot.root_physx_view.set_material_properties(self._robot_mats, cpu)
        self.table.root_physx_view.set_material_properties(self._table_mats, cpu)
        self._dr_damping, self._dr_stiffness = damping, stiffness   # for _check_dr_writes

    def _check_dr_writes(self, ids: torch.Tensor) -> None:
        """Read back what the last ``_randomize(ids)`` wrote; refuse to run on a mismatch."""
        if not self.cfg.sheet.randomization.enabled:
            return
        view = self._sheet.root_physx_view
        k = int(ids.numel())
        got_d = view.get_dof_dampings().reshape(self.num_envs, -1)[ids.cpu(), 0]
        got_s = view.get_dof_stiffnesses().reshape(self.num_envs, -1)[ids.cpu(), 0]
        want_d, want_s = self._dr_damping[:k].cpu(), self._dr_stiffness[:k].cpu()
        for label, got, want in (("damping", got_d, want_d), ("stiffness", got_s, want_s)):
            if not torch.allclose(got.float(), want.float(), rtol=1e-4, atol=1e-9):
                raise RuntimeError(f"hinge {label} read-back mismatch: wrote {want[:4].tolist()}, read {got[:4].tolist()}")
        for label, v, want in (("sheet", view, self._sheet_mats), ("robot", self.robot.root_physx_view, self._robot_mats),
                               ("table", self.table.root_physx_view, self._table_mats)):
            got = v.get_material_properties()[ids.cpu()]
            if not torch.allclose(got, want[ids.cpu()], atol=1e-6):
                raise RuntimeError(f"{label} friction read-back mismatch")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._dr_ready:
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            self._randomize(ids)

    # ------------------------------------------------------------------ viewer

    def viewer_plates(self, env_id: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """``(design, vertices (8, 3) world, triangles (4, 3), moving (4,))``: the two plates' mid-plane
        rectangles at their live poses, for the pose viewer."""
        L, W, _ = (float(v) for v in self._dims_np[env_id])
        pos, quat = self._cloth.body_pose()
        names = list(self._sheet.body_names)
        verts, tris, moving = [], [], []
        for k, (name, x0, x1) in enumerate((("minus", -L / 2, 0.0), ("plus", 0.0, L / 2))):
            b = names.index(name)
            rect = torch.tensor([(x0, -W / 2, 0.0), (x1, -W / 2, 0.0), (x1, W / 2, 0.0), (x0, W / 2, 0.0)],
                                device=self.device)
            world = pos[env_id, b].unsqueeze(0) + _quat_apply(quat[env_id, b].expand(4, 4), rect)
            verts.append(world)
            tris += [(4 * k, 4 * k + 1, 4 * k + 2), (4 * k, 4 * k + 2, 4 * k + 3)]
            moving += [name == "plus"] * 2
        v = torch.cat(verts).detach().cpu().numpy()
        return int(self._design_np[env_id]), v, np.asarray(tris, dtype=int), np.asarray(moving, dtype=bool)
