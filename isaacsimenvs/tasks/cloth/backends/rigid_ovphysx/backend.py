"""``OvPhysxRigidBackend``: the PhysX plate pool (``RigidBackend``) on kit-less Isaac Lab 3 + ovphysx.

Rung B of the sim-to-sim ladder: the SAME engine the policies trained on (PhysX, here the ovphysx
0.5.9 wheel instead of Isaac Sim 5.1's omni.physx), a new framework (Isaac Lab 3). The PhysX side
is ``RigidBackend`` unchanged -- USD contact materials with combine modes, rest / contact offsets,
articulation solver iterations, hinge armature, the startup proofs, the per-reset contact-shell DR.
What is added:

  framework  the Isaac Lab 3 shims the Newton port needs too and that are not Newton's
             (``vbd/newton_support``): 2.x import paths, the kit-less PhysxSchema stand-in, the
             URDF->USD cache (byte-identical robot / table USD to the 2.3 runs), wxyz->xyzw on
             config rotations / root-pose writes / data reads, actuator armature + friction write
             back, implicit actuators' bookkeeping torque zeroed. Not the Newton ones (collision
             push-down, USD drive gains, gravcomp, condim, cloning, materials, inertia floor).
  scene      Isaac Lab 2.3's PhysxCfg (``physx_scene``) authored as ``physxScene:*`` on the
             PhysicsScene -- OvPhysxCfg only carries the GPU buffers -- plus 2.3's default material
             (0.5 / 0.5, average) bound to the scene.
  stage      every env is authored (as 2.3 with ``replicate_physics: false``) and the wheel is asked
             for the FULL stage: by default it strips env_1..N and clones env_0, which would drop
             every per-env binding and size.
  views      ``root_physx_view`` (masses, inertias, rest / contact offsets, shape materials) as a
             thin adapter over ovphysx tensor bindings, so contacts.py / checks.py /
             randomization.py run as written; the pool's writes as Isaac Lab 3 ``*_index`` (xyzw).
"""

from __future__ import annotations

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat

compat.install_isaaclab3_compat()  # first: before any other Isaac Lab / task import

import re  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

from isaacsimenvs.tasks.cloth.backends.rigid import backend as rigid_backend_mod  # noqa: E402
from isaacsimenvs.tasks.cloth.backends.rigid import physx as rigid_physx  # noqa: E402
from isaacsimenvs.tasks.cloth.backends.rigid import scene as pool_scene  # noqa: E402
from isaacsimenvs.tasks.cloth.backends.rigid.backend import RigidBackend  # noqa: E402
from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import patches, physx_schema_shim, usd_cache  # noqa: E402

__all__ = ["OvPhysxRigidBackend"]


# ---------------------------------------------------------------------------- views


class _PhysxViewOnOv:
    """The slice of Isaac Lab 2.3's ``root_physx_view`` (omni.physics.tensors) the pool uses, over
    an ovphysx articulation's tensor bindings. Property tensors are CPU-resident in both."""

    _NAMES = {
        "masses": "articulation_body_mass",
        "inertias": "articulation_body_inertia",
        "rest_offsets": "articulation_rest_offset",
        "contact_offsets": "articulation_contact_offset",
        "material_properties": "articulation_shape_friction_and_restitution",
    }

    def __init__(self, art) -> None:
        self._art = art

    @property
    def _view(self):
        return self._art.root_view

    @property
    def prim_paths(self) -> list[str]:
        return list(self._view.prim_paths)

    def _get(self, key: str) -> torch.Tensor:
        import warp as wp

        name = self._NAMES[key]
        binding = self._view.binding_for(name)
        t = wp.to_torch(self._view.get_attribute(name)).cpu().clone()  # CPU, as omni.physics.tensors
        return t.reshape(tuple(int(s) for s in binding.shape))

    def _set(self, key: str, values: torch.Tensor, indices) -> None:
        import warp as wp

        name = self._NAMES[key]
        device = self._view._native_device(self._view._resolve(name))
        vals = values.detach().to(device=device, dtype=torch.float32).contiguous()
        idx = None
        if indices is not None:
            idx = wp.from_torch(torch.as_tensor(indices).to(device=device, dtype=torch.int32).contiguous())
        self._view.set_attribute(name, wp.from_torch(vals.reshape(-1)), indices=idx)

    def get_masses(self):
        return self._get("masses")

    def get_inertias(self):
        return self._get("inertias")

    def get_rest_offsets(self):
        return self._get("rest_offsets")

    def get_contact_offsets(self):
        return self._get("contact_offsets")

    def get_material_properties(self):
        return self._get("material_properties")

    def set_masses(self, v, indices):
        self._set("masses", v, indices)

    def set_inertias(self, v, indices):
        self._set("inertias", v, indices)

    def set_rest_offsets(self, v, indices):
        self._set("rest_offsets", v, indices)

    def set_contact_offsets(self, v, indices):
        self._set("contact_offsets", v, indices)

    def set_material_properties(self, v, indices):
        raise NotImplementedError("a tensor material write resets PhysX's combine modes (rigid/physx.py)")


def _xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat((q_wxyz[..., 1:4], q_wxyz[..., 0:1]), dim=-1)


class _OvPoolArticulation:
    """The pool's PhysX-style calls (wxyz root poses, positional ``env_ids``) as Isaac Lab 3
    ``*_index`` writes (xyzw), plus ``root_physx_view``. Everything else passes through. (The same
    adapter as ``rigid_newton._NewtonArticulation``, copied so this process never imports Newton.)"""

    def __init__(self, art) -> None:
        self._art = art

    def __getattr__(self, name):
        return getattr(self._art, name)

    @property
    def root_physx_view(self):
        return self._art.root_physx_view

    def _ids(self, env_ids):
        return torch.as_tensor(env_ids, dtype=torch.long)

    def write_root_pose_to_sim(self, pose_wxyz, env_ids):
        pose = torch.cat((pose_wxyz[..., :3], _xyzw(pose_wxyz[..., 3:7])), dim=-1)
        self._art.write_root_pose_to_sim_index(root_pose=pose, env_ids=self._ids(env_ids))

    def write_root_velocity_to_sim(self, vel, env_ids):
        self._art.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=self._ids(env_ids))

    def write_joint_state_to_sim(self, position, velocity, env_ids=None):
        self._art.write_joint_state_to_sim_index(position=position, velocity=velocity,
                                                 env_ids=None if env_ids is None else self._ids(env_ids))

    def set_joint_position_target(self, target, env_ids=None):
        self._art.set_joint_position_target_index(target=target,
                                                  env_ids=None if env_ids is None else self._ids(env_ids))


# ---------------------------------------------------------------------------- the PhysX scene


def _physx_scene_attrs(px, dt: float) -> dict:
    """Isaac Lab 2.3 ``PhysxCfg`` -> the ``physxScene:*`` values 2.3 / Isaac Sim 5.1 author (the
    GPU buffers and the two OvPhysxCfg flags go through OvPhysxCfg, which writes them itself)."""
    return {
        "physxScene:solverType": "TGS" if int(px.solver_type) == 1 else "PGS",
        "physxScene:minPositionIterationCount": int(px.min_position_iteration_count),
        "physxScene:maxPositionIterationCount": int(px.max_position_iteration_count),
        "physxScene:minVelocityIterationCount": int(px.min_velocity_iteration_count),
        "physxScene:maxVelocityIterationCount": int(px.max_velocity_iteration_count),
        "physxScene:bounceThreshold": float(px.bounce_threshold_velocity),
        "physxScene:frictionOffsetThreshold": float(px.friction_offset_threshold),
        "physxScene:frictionCorrelationDistance": float(px.friction_correlation_distance),
        "physxScene:enableStabilization": bool(px.enable_stabilization),
        "physxScene:enableCCD": bool(px.enable_ccd),
        "physxScene:solveArticulationContactLast": bool(px.solve_articulation_contact_last),
        "physxScene:gpuHeapCapacity": int(px.gpu_heap_capacity),
        "physxScene:gpuTempBufferCapacity": int(px.gpu_temp_buffer_capacity),
        "physxScene:gpuMaxNumPartitions": int(px.gpu_max_num_partitions),
        "physxScene:gpuMaxSoftBodyContacts": int(px.gpu_max_soft_body_contacts),
        "physxScene:gpuMaxParticleContacts": int(px.gpu_max_particle_contacts),
        "physxScene:timeStepsPerSecond": int(1.0 / dt),
    }


def _author(prim, name: str, value) -> None:
    from pxr import Sdf

    attr = prim.GetAttribute(name)
    if not attr or not attr.GetTypeName():
        vtype = {
            bool: Sdf.ValueTypeNames.Bool, float: Sdf.ValueTypeNames.Float, str: Sdf.ValueTypeNames.Token,
        }.get(type(value), Sdf.ValueTypeNames.UInt)
        attr = prim.CreateAttribute(name, vtype)
    attr.Set(value)


def build_ovphysx_cfg(px):
    from isaaclab_ov.physics import OvPhysxCfg

    return OvPhysxCfg(
        enable_enhanced_determinism=bool(px.enable_enhanced_determinism),
        enable_external_forces_every_iteration=bool(px.enable_external_forces_every_iteration),
        gpu_max_rigid_contact_count=int(px.gpu_max_rigid_contact_count),
        gpu_max_rigid_patch_count=int(px.gpu_max_rigid_patch_count),
        gpu_found_lost_pairs_capacity=int(px.gpu_found_lost_pairs_capacity),
        gpu_found_lost_aggregate_pairs_capacity=int(px.gpu_found_lost_aggregate_pairs_capacity),
        gpu_total_aggregate_pairs_capacity=int(px.gpu_total_aggregate_pairs_capacity),
        gpu_collision_stack_size=int(px.gpu_collision_stack_size),
    )


# ---------------------------------------------------------------------------- an isaaclab_ov fix


def install_joint_property_sync() -> bool:
    """Synchronize before ovphysx reads a joint property's pinned-host staging buffer.

    ``isaaclab_ov`` ``Articulation._push_joint_property`` (articulation.py:4897-4913) copies the
    GPU joint-property buffer into its pinned-host ``cpu_buffer`` with ``wp.copy`` -- asynchronous
    -- and hands that buffer to the wheel at once, with no ``wp.synchronize_stream``; every other
    staging path in ``articulation_data.py`` (1985, 2019) synchronizes. The wheel then reads stale
    host memory. Measured at construction: the arm actuator's stiffness / damping landed, the
    hand actuator's damping, armature, effort and velocity limits did not (solver: damping 0,
    armature 0, URDF effort / velocity limits instead of HAND2_DRIVE's), and the plate hinges kept
    damping 0 and armature 0 instead of 0.05 and 1e-3 -- while Isaac Lab's own buffers held the
    right values. The same method, with the synchronization."""
    import warp as wp
    from isaaclab_ov.assets.articulation.articulation import Articulation

    if getattr(Articulation, "_rigid_ovphysx_push_sync", False):
        return False

    def _push_joint_property(self, tensor_type, user_buffer, backend_buffer, *, cpu_buffer=None,
                             component_count=None, indices=None, mask=None):
        property_backend = self._get_backend_ordered_joint_buffer(
            user_buffer, backend_buffer, component_count=component_count,
        )
        if cpu_buffer is None:
            cpu_buffer = self._data._stage_to_pinned_cpu(tensor_type, "write", property_backend)
        else:
            source = property_backend
            if source.dtype != wp.float32:
                source = wp.array(ptr=source.ptr, shape=cpu_buffer.shape, dtype=wp.float32,
                                  device=str(source.device), copy=False)
            wp.copy(cpu_buffer, source)
            wp.synchronize_stream(source.device)  # the missing line
        if indices is not None:
            self._root_view.set_attribute(tensor_type, cpu_buffer, indices=indices)
        else:
            self._root_view.set_attribute(tensor_type, cpu_buffer, mask=mask)

    Articulation._push_joint_property = _push_joint_property
    Articulation._rigid_ovphysx_push_sync = True
    return True


# ---------------------------------------------------------------------------- the backend


class OvPhysxRigidBackend(RigidBackend):
    """The fold-shape pool on PhysX via kit-less Isaac Lab 3 + ovphysx."""

    def __init__(self, cfg, robot_spec) -> None:
        # --- the Isaac Lab 3 shims (framework, not Newton); before any asset is built.
        physx_schema_shim.install()
        usd_cache.install_reader()
        # Every env authored, as 2.3 with replicate_physics: false (no cloner runs here).
        patches.install_multi_asset(int(cfg.scene.num_envs), prototypes_only=False)
        patches.install_quat_convention()
        patches.install_quat_math()
        patches.install_actuator_broadcast()
        patches.install_armature_writeback()
        patches.install_implicit_effort_fix()
        install_joint_property_sync()

        p = cfg.fold_pool
        if p.one_shape_per_env and len(p.shapes) > 1:
            raise NotImplementedError("fold_pool.one_shape_per_env with several shapes on ovphysx (a subset "
                                      "articulation subclasses Isaac Lab 3's factory class)")
        # RigidBackend sizes the PhysX GPU buffers on cfg.sim.physx; Isaac Lab 3 keeps them in physx_scene.
        grow = rigid_physx.size_gpu_buffers

        def size_on_physx_scene(c):
            grow(SimpleNamespace(sim=SimpleNamespace(physx=c.physx_scene), scene=c.scene, physx_chain=c.physx_chain))

        with patch.object(rigid_backend_mod, "size_gpu_buffers", size_on_physx_scene):
            super().__init__(cfg, robot_spec)
        if self.env_shape_np is not None:
            # One shape: "one per env" and "every env carries every shape" are the same scene.
            print("[rigid_ovphysx] one_shape_per_env with one shape: built as every env carrying it", flush=True)
            self.env_shape_np = None
        cfg.sim.physics = build_ovphysx_cfg(cfg.physx_scene)
        cfg.sim.use_newton_actuators = False

    # ------------------------------------------------------------------ construction

    def build_scene(self, scene, sim) -> None:
        self._author_physics_scene(sim)
        # The wheel ingests every authored env (the startup flag is cleared by
        # SimulationContext.initialize, which has run; the stage is serialized at the first reset).
        from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

        OvPhysxManager.require_full_stage()
        super().build_scene(scene, sim)  # robot, table, pool, then the contact materials
        self._bind_fingertip_meshes()

    def _bind_fingertip_meshes(self) -> None:
        """The fingertip material on the fingertip colliders' MESHES too.

        The robot's colliders (Isaac Sim 5.1 URDF importer, the cached USD) are Xforms carrying
        CollisionAPI, with the geometry in a child ``mesh`` that has its own all-purpose (visual)
        ``material:binding``. Isaac Sim's omni.physx resolves the physics material on the collider
        Xform (``contacts.bind_contact_materials`` binds there: 10 shapes at 0.612 under 2.3); the
        ovphysx wheel resolves it on the child mesh, where the direct visual binding beats the
        inherited physics one, so every fingertip read back the default 0.5 (measured). A
        physics-purpose binding on the mesh itself is what both resolve to the same material."""
        from isaaclab.sim.utils import get_current_stage
        from pxr import Usd, UsdGeom

        stage = get_current_stage()
        tips = tuple(self.robot_spec.fingertip_link_names)
        robot = self.robot.cfg.prim_path.rsplit("/", 1)[-1]
        rels = rigid_physx.collider_paths(
            stage, f"/World/envs/env_0/{robot}",
            select=lambda prim: any(f"/{t}/" in str(prim.GetPath()) + "/" for t in tips),
        )
        base = f"/World/envs/env_0/{robot}"
        meshes = []
        for rel in rels:
            for p in Usd.PrimRange(stage.GetPrimAtPath(f"{base}/{rel}")):
                if p.IsA(UsdGeom.Mesh):
                    meshes.append(str(p.GetPath())[len(base) + 1:])
        if len(meshes) != len(rels):
            raise RuntimeError(f"[rigid_ovphysx] {len(rels)} fingertip colliders but {len(meshes)} meshes under them")
        tip = "/World/PhysicsMaterials/fingertip"
        n = rigid_physx.bind_materials(
            stage, ((f"/World/envs/env_{i}/{robot}/{m}", tip) for i in range(self.num_envs) for m in meshes),
            tag="bind_tip_meshes",
        )
        print(f"[rigid_ovphysx] fingertip material also bound on {n} collider meshes ({len(meshes)} per env)", flush=True)

    def _author_physics_scene(self, sim) -> None:
        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdShade

        stage = get_current_stage()
        path = sim.cfg.physics_prim_path
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"[rigid_ovphysx] no PhysicsScene at {path}")
        if "PhysxSceneAPI" not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema("PhysxSceneAPI")
        attrs = _physx_scene_attrs(self.cfg.physx_scene, float(sim.cfg.dt))
        for name, value in attrs.items():
            _author(prim, name, value)
        # Isaac Lab 2.3's default material (RigidBodyMaterialCfg: 0.5 / 0.5 / 0, average), bound to the
        # scene: every collider without its own material (robot links, table) uses it.
        mat = rigid_physx.define_scene_material(stage, f"{path}/defaultMaterial", mu=rigid_physx.DEFAULT_MU,
                                                combine="average")
        mat.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode",
                                      __import__("pxr").Sdf.ValueTypeNames.Token).Set("average")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        self._scene_attrs = attrs
        print("[rigid_ovphysx] PhysicsScene: " + ", ".join(f"{k.split(':')[1]}={v}" for k, v in attrs.items())
              + f"; OvPhysxCfg ext-forces-every-iteration={self.cfg.physx_scene.enable_external_forces_every_iteration}"
              f" determinism={self.cfg.physx_scene.enable_enhanced_determinism}; default material 0.5/0.5 average",
              flush=True)

    def install_sheet(self) -> None:
        # Kit-less, rigid/usd.bulk_edit_layer authors inside an Sdf.ChangeBlock, and the plate USDs are
        # written (Usd.Stage.CreateNew) inside it -- which USD refuses. The root layer directly instead.
        @contextmanager
        def plain_layer(stage, tag):
            yield stage.GetRootLayer()

        with patch.object(pool_scene, "bulk_edit_layer", plain_layer):
            pool_scene.install_pool(self)
        for art in self.arts:
            art.root_physx_view = _PhysxViewOnOv(art)
        self.pool._arts = [_OvPoolArticulation(a) for a in self.pool._arts]

    def finish_setup(self) -> None:
        self.robot.root_physx_view = _PhysxViewOnOv(self.robot)
        # Isaac Lab 3 asset data as torch with wxyz quaternions, and wxyz root-pose writes (the task's).
        patches.wrap_env_assets(self)
        patches.install_pose_write_conversion(self)
        self._check_instance_order()
        self._check_stage_handed_over()
        self._report_robot_shapes()

    def _report_robot_shapes(self) -> None:
        """The robot's and the table's shape offsets as the SOLVER has them, against the USD (2 mm
        contact, 0 rest on every robot / table collider)."""
        import warp as wp

        for name, asset, key in (("robot", self.robot, "articulation"), ("table", self.table, "rigid_body")):
            view = asset.root_view
            out = []
            for attr in ("contact_offset", "rest_offset"):
                t = wp.to_torch(view.get_attribute(f"{key}_{attr}")).float()
                out.append(f"{attr} {float(t.min()) * 1e3:.2f}-{float(t.max()) * 1e3:.2f} mm ({t.numel()} values)")
            print(f"[rigid_ovphysx] {name} shapes in the solver: " + ", ".join(out), flush=True)

    def _check_instance_order(self) -> None:
        """Every view lists its instances in env order (the task indexes data rows as env ids)."""
        for name, asset in [("robot", self.robot), ("table", self.table), ("goal_viz", self.goal_viz),
                            *[(f"fold_pool_{k}", a) for k, a in enumerate(self.arts)]]:
            paths = list(asset.root_view.prim_paths)
            ids = [int(re.search(r"/env_(\d+)/", p).group(1)) for p in paths]
            if ids != list(range(self.num_envs)):
                raise RuntimeError(f"[rigid_ovphysx] {name}: instances not in env order: {ids[:8]}...")
        print(f"[rigid_ovphysx] robot, table, goal marker, {len(self.arts)} pool articulation(s): "
              f"{self.num_envs} instances each, in env order", flush=True)

    def _check_stage_handed_over(self) -> None:
        """What the wheel was given: every env (not env_0 + clones) and the scene attributes."""
        from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

        usda = OvPhysxManager._stage_usda or ""
        import os

        if os.environ.get("RIGID_OVPHYSX_DUMP"):
            with open(os.environ["RIGID_OVPHYSX_DUMP"], "w") as f:
                f.write(usda)
        envs = len(set(re.findall(r'def Xform "env_(\d+)"', usda)))
        if envs != self.num_envs:
            raise RuntimeError(f"[rigid_ovphysx] the wheel got {envs} envs of {self.num_envs} (full stage not taken)")
        missing = [k for k in self._scene_attrs if k not in usda]
        if missing:
            raise RuntimeError(f"[rigid_ovphysx] PhysicsScene attributes missing from the serialized stage: {missing}")
        print(f"[rigid_ovphysx] serialized stage handed to ovphysx: {envs} envs, scene attributes present", flush=True)

    # ------------------------------------------------------------------ conventions

    def observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        # Isaac Lab 3's helper already returns the (x, y, z, w) relabelling the task was trained on.
        return quat_from_matrix(rot)

    # ------------------------------------------------------------------ diagnostics (start)

    def start(self, reset_table) -> None:
        super().start(reset_table)  # contacts read-back, placement proofs, physics DR, folded starts
        self._check_margin_dr()
        self._report_drives()
        self._report_rest_tracking()

    def _report_drives(self) -> None:
        """The robot's and the hinges' drive properties AS THE SOLVER HAS THEM, against the RobotSpec
        (``robots/flexiv_right.py``: Rizon gains, Wuji HAND2_DRIVE, vendor armature) and the pool cfg."""
        import warp as wp

        def dof(asset, attr):
            return wp.to_torch(asset.root_view.get_attribute(f"articulation_dof_{attr}")).float().cpu()

        spec, names = self.robot_spec, list(self.robot.joint_names)
        want = {"stiffness": {**spec.arm_stiffness, **spec.hand_stiffness},
                "damping": {**spec.arm_damping, **spec.hand_damping},
                "armature": {**{j: 0.0 for j in spec.arm_stiffness}, **spec.hand_armature},
                "max_force": dict(spec.hand_effort_limit),
                "max_velocity": {j: float(spec.hand_velocity_limit) for j in spec.hand_stiffness}}
        rows, worst = [], {}
        got = {k: dof(self.robot, k) for k in ("stiffness", "damping", "armature", "max_force", "max_velocity")}
        fric = dof(self.robot, "friction_properties")
        for key, table in want.items():
            g = got[key][0]
            err = max(abs(float(g[i]) - float(table[n])) / max(abs(float(table[n])), 1e-9) if table[n] else abs(float(g[i]))
                      for i, n in enumerate(names) if n in table)
            spread = float((got[key] - got[key][0:1]).abs().max())
            worst[key] = (err, spread)
        for i, n in enumerate(names):
            rows.append(f"{n} k={float(got['stiffness'][0, i]):.4g} d={float(got['damping'][0, i]):.4g} "
                        f"a={float(got['armature'][0, i]):.4g} fmax={float(got['max_force'][0, i]):.4g} "
                        f"vmax={float(got['max_velocity'][0, i]):.4g}")
        print("[rigid_ovphysx] robot drives in the solver (env 0): " + "; ".join(rows), flush=True)
        print("[rigid_ovphysx] robot drives vs RobotSpec: "
              + ", ".join(f"{k} max rel err {e:.2e} (spread across envs {s:.1e})" for k, (e, s) in worst.items())
              + f"; joint friction max {float(fric.abs().max()):.3g}", flush=True)
        bad = {k: e for k, (e, sp) in worst.items() if e > 1e-4 or sp > 1e-6}
        if bad:
            raise RuntimeError(f"[rigid_ovphysx] robot drives in the solver differ from the RobotSpec: {bad}")
        for k, art in enumerate(self.arts):
            r = self.cfg.rigid_cloth
            print(f"[rigid_ovphysx] pool {k} hinges in the solver: stiffness "
                  f"{float(dof(art, 'stiffness').max()):.3g} (cfg {r.joint_stiffness}), damping "
                  f"{float(dof(art, 'damping').min()):.4g}-{float(dof(art, 'damping').max()):.4g} (cfg {r.joint_damping}), "
                  f"armature {float(dof(art, 'armature').max()):.3g} (cfg {r.joint_armature})", flush=True)
            for attr, v in (("damping", r.joint_damping), ("armature", r.joint_armature)):
                t = dof(art, attr)
                if float((t - float(v)).abs().max()) > 1e-6 * max(1.0, float(v)) + 1e-9:
                    raise RuntimeError(f"[rigid_ovphysx] pool {k} hinge {attr} in the solver {t.flatten()[:4].tolist()} != {v}")

    def _report_rest_tracking(self, steps: int = 60) -> None:
        """Hold the robot's default pose for ``steps`` physics steps (the sheets parked away, as the
        arm check leaves them): how far each joint ends up from its target. A self-collision or a
        missing drive shows up as a finger that cannot hold its rest angle (ring / pinky MCP on Newton)."""
        import math

        from isaacsimenvs.tasks.cloth.backends.rigid.pool import to_torch

        robot, dt = self.robot, float(self.cfg.sim.dt)
        home = to_torch(robot.data.default_joint_pos).clone()
        robot.write_joint_state_to_sim(home, torch.zeros_like(home))
        robot.set_joint_position_target(home)
        for _ in range(steps):
            robot.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt)
        err = (to_torch(robot.data.joint_pos) - home).abs().max(dim=0).values
        names = list(robot.joint_names)
        order = torch.argsort(err, descending=True)[:6].tolist()
        print(f"[rigid_ovphysx] rest pose held {steps} steps: max |q - target| over envs, worst joints: "
              + ", ".join(f"{names[i]} {math.degrees(float(err[i])):.3f} deg" for i in order)
              + "; ring/pinky MCP flex "
              + ", ".join(f"{n} {math.degrees(float(err[names.index(n)])):.3f} deg"
                          for n in ("r_ring_finger_mcp_flex", "r_pinky_mcp_flex") if n in names), flush=True)

    def end_reset(self, env_ids: torch.Tensor) -> None:
        super().end_reset(env_ids)
        if self.physics_dr is not None and not getattr(self, "_margin_checked_at_reset", False):
            self._margin_checked_at_reset = True
            self._check_margin_dr(where="first reset")

    def _check_margin_dr(self, where: str = "start") -> None:
        """The contact-shell DR's per-env rest offsets reached the solver (written through the
        ``root_physx_view`` adapter, read back from the wheel)."""
        want = self.physics_dr.sample["contact_margin"].cpu()
        for k, art in enumerate(self.arts):
            got = art.root_physx_view.get_rest_offsets()
            err = float((got - want[self.pool.members(k).cpu()].unsqueeze(1)).abs().max())
            if err > 1e-6:
                raise RuntimeError(f"[rigid_ovphysx] pool {k}: rest offsets in the solver off the DR draw by {err * 1e3:.3f} mm")
        print(f"[rigid_ovphysx] contact-shell DR ({where}): rest offsets in the solver = the draw "
              f"({float(want.min()) * 1e3:.2f}-{float(want.max()) * 1e3:.2f} mm over {want.numel()} envs)", flush=True)
