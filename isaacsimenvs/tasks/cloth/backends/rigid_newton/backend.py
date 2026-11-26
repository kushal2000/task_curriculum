"""``NewtonRigidBackend``: the PhysX plate pool's sheet, simulated on Isaac Lab 3.0 + Newton with the
robot AND the plates on MJWarp (no VBD, no coupler).

For sim-to-sim: evaluate policies trained on the PhysX plates with the robot on MuJoCo, everything else
as close to the PhysX world as the solver allows. What is reused / replaced:

  reused     the pool (``rigid/pool.py``: grid points from plate poses, placing, parking, quaternion-layout
             detection), its scene construction (``rigid/scene.py``: one USD per shape and size, the
             articulations, goal tasks, folded starts) and the robot's Newton construction (the VBD
             backend's ``newton_setup`` patches)
  replaced   PhysX contact materials / rest offsets -> Newton shape margin (the 7 mm contact shell,
             summed per contact like PhysX's rest offsets), gap and friction (MuJoCo combines by max:
             every shape 0.354, fingertips 0.612 reproduces PhysX's priority pairs); the pool's
             PhysX-style writes -> Isaac Lab 3's ``*_index`` writes with xyzw quaternions; the hinge
             drive damping is authored into the USD (Newton builds MuJoCo actuators from it at import)
  not yet    physics DR (contact-shell +-3 mm, hinge/mass), finger penetration, hinge plasticity, a
             different sheet size per env (Newton replicates one prototype: size_range must be a point)
"""

from __future__ import annotations

from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import patches  # noqa: F401  (first: the Isaac Lab 3.0 shims)

import math
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend
from isaacsimenvs.tasks.cloth.backends.rigid import assets as fsa
from isaacsimenvs.tasks.cloth.backends.rigid import scene as pool_scene
from isaacsimenvs.tasks.cloth.backends.rigid.backend import RigidBackend
from isaacsimenvs.tasks.cloth.backends.vbd.newton_setup import (
    install_post_construction_patches,
    install_pre_construction_patches,
)
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs

__all__ = ["NewtonRigidBackend"]


def build_mjwarp_physics(newton_cfg, num_envs: int):
    """``NewtonCfg``: plain MJWarp for the robot and the plates, Newton's collision pipeline (the
    plates' contact shell is a shape margin, which MuJoCo's own contacts drop)."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

    if newton_cfg.use_mujoco_contacts:
        raise ValueError("newton.use_mujoco_contacts must be false: MuJoCo's contacts ignore the plates' margin")
    solver = MJWarpSolverCfg(
        solver=newton_cfg.solver,
        integrator=newton_cfg.integrator,
        njmax=newton_cfg.njmax,
        nconmax=newton_cfg.nconmax,
        impratio=newton_cfg.impratio,
        cone=newton_cfg.cone,
        iterations=newton_cfg.iterations,
        ls_iterations=newton_cfg.ls_iterations,
        use_mujoco_contacts=False,
        ccd_iterations=newton_cfg.ccd_iterations,
        disable_sensors=newton_cfg.disable_sensors,
    )
    return NewtonCfg(
        solver_cfg=solver,
        # Global (not per-env) budget: hand meshes against the plates overflowed the 1e6 default at 32 envs
        # (1.13e6 pairs once the hand works the sheet).
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=newton_cfg.rigid_contact_max,
                                                 max_triangle_pairs=int(max(1_000_000, 120_000 * num_envs))),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=newton_cfg.num_substeps,
        collision_decimation=newton_cfg.collision_decimation,
        use_cuda_graph=True,
    )


def _xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat((q_wxyz[..., 1:4], q_wxyz[..., 0:1]), dim=-1)


class _NewtonArticulation:
    """The pool's PhysX-style articulation calls (wxyz root poses, positional ``env_ids``) as Isaac
    Lab 3.0 ``*_index`` writes (xyzw). Everything else passes through."""

    def __init__(self, art) -> None:
        self._art = art

    def __getattr__(self, name):
        return getattr(self._art, name)

    def _ids(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        # A full, ordered index list is passed as None: an explicit full arange is a no-op on some
        # Newton write paths (the old Newton fold-pool adapter's finding).
        n = int(self._art.num_instances)
        if ids.numel() == n and bool((ids.cpu() == torch.arange(n)).all()):
            return None
        return ids

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


class NewtonRigidBackend(RigidBackend):
    """The fold-shape pool on Newton: robot and plates on MJWarp."""

    def __init__(self, cfg, robot_spec) -> None:  # noqa: D107 - RigidBackend's PhysX setup is skipped
        ClothBackend.__init__(self, cfg, robot_spec)
        p = cfg.fold_pool
        self.shapes = [fs.shape(n) for n in p.shapes]
        w = [1.0] * len(self.shapes) if p.weights is None else [float(x) for x in p.weights]
        self.shape_weights = torch.tensor(w, dtype=torch.float32)
        lo, hi = (float(x) for x in p.size_range)
        if abs(hi - lo) > 1e-9:
            raise ValueError(f"fold_pool.size_range {p.size_range}: Newton replicates one prototype per "
                             "asset, so every env must be one size -- set both ends equal")
        n = int(cfg.scene.num_envs)
        self.env_size_np, self.env_slab_np = pool_scene.draw_env_sizes(cfg, n)
        self.env_shape_np = None
        if p.one_shape_per_env and len(self.shapes) > 1:
            raise ValueError("fold_pool.one_shape_per_env with several shapes: Newton needs every env to carry "
                             "the same articulations -- use one_shape_per_env: false (each env carries all)")
        self.goal_tasks = None
        self.started = False
        self.physics_dr = None
        self.plasticity = None
        self.penetration = None
        self._finger_link_names = None
        if float(cfg.rigid_cloth.joint_yield_deg) > 0.0:
            raise NotImplementedError("rigid_cloth.joint_yield_deg (hinge plasticity) on Newton")
        if float(cfg.termination.max_finger_penetration) > 0.0:
            raise NotImplementedError("termination.max_finger_penetration on Newton")
        # PhysX's hinge armature (physx_chain.hinge_armature) is what the policies trained with.
        if float(cfg.physx_chain.hinge_armature) > 0.0:
            cfg.rigid_cloth.joint_armature = float(cfg.physx_chain.hinge_armature)
        install_pre_construction_patches(cfg, physics_cfg=build_mjwarp_physics(cfg.newton, n))

    # ------------------------------------------------------------------ construction

    #: Side-by-side finger proximal links whose MuJoCo collision hulls overlap at the rest pose (ring /
    #: pinky: 14 contacts, the fingers held 0.26-0.33 rad off target). PhysX's shapes for the same USD
    #: do not touch there, so training never saw them collide.
    ADJACENT_FINGER_PAIRS = (("index_finger", "middle_finger"), ("middle_finger", "ring_finger"), ("ring_finger", "pinky"))

    @contextmanager
    def construction_context(self):
        """Experiment hook (RN_FILTER_ADJ_FINGERS=1): keep robot self-collision on, as trained, but filter
        the adjacent fingers' proximal / proximal_abd pairs on Newton."""
        import os
        if os.environ.get("RN_FILTER_ADJ_FINGERS", "0") != "1":
            yield
            return
        from isaaclab_newton.physics import NewtonManager

        original = NewtonManager._prepare_builder_for_finalize
        pairs = self.ADJACENT_FINGER_PAIRS

        def prepare(cls_or_builder, builder=None):
            if builder is None:
                builder = cls_or_builder
                original(builder)
            else:
                original(cls_or_builder, builder)
            names = [str(x).rsplit("/", 1)[-1] for x in builder.body_label]
            shapes_of = {}
            for sh, b in enumerate(builder.shape_body):
                if b >= 0:
                    shapes_of.setdefault(b, []).append(sh)
            n = 0
            for b1, n1 in enumerate(names):
                for b2, n2 in enumerate(names):
                    if b2 <= b1 or not (n1.endswith(("_proximal", "_proximal_abd")) and n2.endswith(("_proximal", "_proximal_abd"))):
                        continue
                    if builder.body_world[b1] != builder.body_world[b2] if hasattr(builder, "body_world") else False:
                        continue
                    f1, f2 = n1[2:].rsplit("_proximal", 1)[0], n2[2:].rsplit("_proximal", 1)[0]
                    if (f1, f2) in pairs or (f2, f1) in pairs:
                        for a in shapes_of.get(b1, []):
                            for c in shapes_of.get(b2, []):
                                builder.add_shape_collision_filter_pair(a, c)
                                n += 1
            print(f"[rigid_newton] filtered {n} shape pairs between adjacent fingers' proximal links", flush=True)

        with patch.object(NewtonManager, "_prepare_builder_for_finalize", prepare):
            yield

    def build_scene(self, scene, sim) -> None:
        ClothBackend.build_scene(self, scene, sim)  # no PhysX materials / contact reports

    @contextmanager
    def _hinge_drives_in_usd(self):
        """Author the hinge damping into each plate USD's drives. Newton builds MuJoCo actuators from
        the USD at import (a zero-gain drive becomes no actuator, and the runtime gain then does
        nothing); its importer reads angular gains per DEGREE, so N m s/rad is written x pi/180."""
        from pxr import Usd

        damping = float(self.cfg.rigid_cloth.joint_damping) * math.pi / 180.0
        stiffness = float(self.cfg.rigid_cloth.joint_stiffness) * math.pi / 180.0
        write = fsa.write_shape_usd

        def write_with_drives(shape, out_path, **kw):
            path = write(shape, out_path, **kw)
            stage = Usd.Stage.Open(str(path))
            for prim in stage.Traverse():
                if prim.GetName().startswith(fsa.JOINT_PREFIX) and prim.GetAttribute("drive:angular:physics:damping"):
                    prim.GetAttribute("drive:angular:physics:damping").Set(damping)
                    prim.GetAttribute("drive:angular:physics:stiffness").Set(stiffness)
            stage.GetRootLayer().Save()
            return path

        with patch.object(fsa, "write_shape_usd", write_with_drives):
            yield

    def install_sheet(self) -> None:
        # Kit-less, rigid/usd.bulk_edit_layer authors inside an Sdf.ChangeBlock, and the plate USDs are
        # written (Usd.Stage.CreateNew + Define) inside it -- which USD refuses. Author on the root
        # layer directly instead; the env prims are few at evaluation sizes.
        @contextmanager
        def plain_layer(stage, tag):
            yield stage.GetRootLayer()

        with self._hinge_drives_in_usd(), patch.object(pool_scene, "bulk_edit_layer", plain_layer):
            pool_scene.install_pool(self)
        self.pool._arts = [_NewtonArticulation(a) for a in self.pool._arts]

    def finish_setup(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        install_post_construction_patches(self)  # robot / table materials, pose-write conversion, notify
        self._set_plate_contacts(NewtonManager._model, NewtonManager._solver)

    def _set_plate_contacts(self, model, solver) -> None:
        """The PhysX contact model in Newton terms: the plates' contact shell as their shape margin
        (rest offset), the contact offset's extra as the gap; friction 0.354 everywhere and the
        fingertips' 0.612 (MuJoCo's max combine = PhysX's min/max priority pairs)."""
        import newton

        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        labels = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
        full = [str(x) for x in model.body_label]
        shape_body = model.shape_body.numpy()
        plate = np.array([b >= 0 and labels[b].startswith(fsa.BODY_PREFIX) and "FoldPool" in full[b]
                          for b in shape_body])
        tips = set(self.robot_spec.fingertip_link_names)
        tip = np.array([b >= 0 and labels[b] in tips for b in shape_body])
        margin = model.shape_margin.numpy()
        gap = model.shape_gap.numpy()
        margin[:] = 0.0
        margin[plate] = float(r.contact_margin)
        gap[plate] = float(pc.contact_offset_margin)
        model.shape_margin.assign(margin)
        model.shape_gap.assign(gap)
        mu = model.shape_material_mu.numpy()
        mu[:] = float(r.slat_friction)
        mu[tip] = float(r.fingertip_slat_friction)
        model.shape_material_mu.assign(mu)
        solver.notify_model_changed(int(newton.ModelFlags.SHAPE_PROPERTIES))
        # Experiment hook: MuJoCo contact stiffness. Newton leaves geom_solref at MuJoCo's default
        # (time constant 0.02 s, damping ratio 1) -- far softer than PhysX's rigid contacts.
        import os
        tc = os.environ.get("RN_SOLREF_TC")
        if tc:
            mj = solver.mjw_model
            sr = mj.geom_solref.numpy()
            before = sorted({round(float(v), 5) for v in sr[..., 0].reshape(-1)})
            sr[..., 0] = float(tc)
            mj.geom_solref.assign(sr)
            print(f"[rigid_newton] geom_solref time constant {before} -> {float(tc)} s on every geom", flush=True)
        print(f"[rigid_newton] {int(plate.sum())} plate shapes: margin {r.contact_margin * 1e3:.1f} mm, gap "
              f"{pc.contact_offset_margin * 1e3:.1f} mm; friction {r.slat_friction:.3f} everywhere, "
              f"{r.fingertip_slat_friction:.3f} on {int(tip.sum())} fingertip shapes (max combine)", flush=True)

    def start(self, reset_table) -> None:
        """RigidBackend.start without the PhysX checks, physics DR, plasticity and penetration.

        The pool reads plate quaternions as wxyz unless told otherwise; RigidBackend measures the layout
        in its placement check (skipped here). Isaac Lab 3.0 reports xyzw, and reading xyzw as wxyz
        turns a flat plate 180 deg about z -- the two halves of the sheet swap in everything the task
        reads. Set it, then confirm it against the flat grid of env 0."""
        self.pool._quat_layout = "xyzw"
        reset_table(torch.arange(self.num_envs, device=self.device))
        self.pool.place(torch.arange(self.num_envs, device=self.device),
                        self.scene.env_origins[:, :2], torch.zeros(self.num_envs, device=self.device))
        self.sim.step(render=False)
        self.scene.update(dt=float(self.sim.get_physics_dt()))
        layout = self.pool.detect_quat_layout((torch.as_tensor(self.rest_local, device=self.device).unsqueeze(0) * self.scale3())[0])
        print(f"[rigid_newton] plate quaternion layout {layout} (grid error per layout, mm: "
              f"{ {k: round(v * 1e3, 2) for k, v in self.pool._quat_layout_scores.items()} })", flush=True)
        if layout != "xyzw":
            raise RuntimeError(f"[rigid_newton] measured plate quaternion layout {layout}, expected xyzw on Isaac Lab 3")
        self.shape_weights = self.shape_weights.to(self.device)
        pool_scene.init_start_folds(self)
        if self.goal_tasks is not None:
            pool_scene.start_goal_tasks(self)
        self.started = True

    def end_reset(self, env_ids: torch.Tensor) -> None:
        pass

    def pre_physics_step(self) -> None:
        self.pool.park()

    measures_finger_penetration = False

    def finger_penetration(self) -> torch.Tensor:
        raise NotImplementedError("finger penetration on Newton")

    def observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        # Isaac Lab 3.0's helper already returns the (x, y, z, w) relabelling the task was trained on.
        return quat_from_matrix(rot)

    def add_episode_final(self, final: dict, reasons: dict, episode_length_buf: torch.Tensor) -> None:
        entered = reasons["fold"].bool()
        flat_fold = entered & ~self.start_folded
        length = episode_length_buf.float()
        final["fold_steps"] = entered.float() * length
        final["fold_flat_start"] = flat_fold.float()
        final["fold_steps_flat_start"] = flat_fold.float() * length
