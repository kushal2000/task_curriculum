"""``RigidBackend``: the fold task's sheet as the fold-shape pool of hinged plates, on PhysX.

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv_isaacsim``). The backend owns the pool and its physics:

    scene.py           the pool's articulations, per-env sizes, shape draws, folded starts
    contacts.py        friction materials and contact offsets (PhysX USD, see physx.py)
    randomization.py   per-env physics DR at every reset
    checks.py          startup proofs
    pool.py            PlatePool: the active shape's grid points, placing and parking the shapes

Every env carries every pool shape (one active, the rest parked far away) unless
``fold_pool.one_shape_per_env``, where each env carries one. Each env's sheets are built at its own
size (``env_size``); the task scales its rest geometry by ``xy_scale``.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend, reset_yaw
from isaacsimenvs.tasks.cloth.backends.rigid import checks, contacts
from isaacsimenvs.tasks.cloth.backends.rigid import scene as pool_scene
from isaacsimenvs.tasks.cloth.backends.rigid.physx import DEFAULT_MU, size_gpu_buffers
from isaacsimenvs.tasks.cloth.backends.rigid.randomization import PlateRandomization
from isaacsimenvs.tasks.cloth.config.schema.rigid import CLOTH_FOLD_LIFT, RigidClothEnvCfg
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.state import SheetLayout

__all__ = ["RigidBackend"]


class RigidBackend(ClothBackend):
    """The fold-shape pool as the manipuland, on PhysX."""

    cfg: RigidClothEnvCfg

    def __init__(self, cfg: RigidClothEnvCfg, robot_spec) -> None:
        super().__init__(cfg, robot_spec)
        # --- the pool, before the scene is built.
        p = cfg.fold_pool
        self.shapes = [fs.shape(n) for n in p.shapes]
        w = [1.0] * len(self.shapes) if p.weights is None else [float(x) for x in p.weights]
        self.shape_weights = torch.tensor(w, dtype=torch.float32)
        thickness = {sh.thickness for sh in self.shapes}
        if len(thickness) != 1:
            raise ValueError(f"pool shapes disagree on thickness {sorted(thickness)}")
        # Per-env size and slab thickness, drawn ONCE (see FoldPoolCfg.size_range): each env's USD is
        # built at its size, so it cannot change afterwards.
        n = int(cfg.scene.num_envs)
        self.env_size_np, self.env_slab_np = pool_scene.draw_env_sizes(cfg, n)
        #: Each env's one shape (``one_shape_per_env``), or ``None``: every env carries every shape.
        self.env_shape_np = None
        if p.one_shape_per_env:
            S = len(self.shapes)
            if p.weights is None:
                self.env_shape_np = np.arange(n) % S
            else:
                w = np.asarray([float(x) for x in p.weights])
                self.env_shape_np = np.random.default_rng(int(p.size_seed) + 1).choice(S, n, p=w / w.sum())

        # --- PhysX contact setup that must be in the cfg before the scene is built.
        a = cfg.assets
        # Contacts are authored in USD (contacts.py): a tensor-API material write would reset the
        # combine modes (see physx.py). Robot links and the table keep PhysX's default material.
        model = str(cfg.physx_chain.friction_model)
        if model not in ("priority", "cloth_min"):
            raise ValueError(f"physx_chain.friction_model must be priority or cloth_min, got {model!r}")
        for name in ("robot_friction", "table_friction") if model == "priority" else ():
            if abs(float(getattr(a, name)) - DEFAULT_MU) > 1e-9:
                raise ValueError(
                    f"assets.{name}={getattr(a, name)}: the PhysX fold tasks leave robot links and the "
                    f"table on PhysX's default material ({DEFAULT_MU}); a different value needs a "
                    f"USD material, not the tensor write"
                )
        # A PhysX numerical requirement, not a task parameter: see PhysxChainCfg.hinge_armature. Written
        # into the task block so the actuator (and the logs) carry the value the hinges really get.
        if float(cfg.physx_chain.hinge_armature) > 0.0:
            cfg.rigid_cloth.joint_armature = float(cfg.physx_chain.hinge_armature)
        print(f"[physx] hinge armature {cfg.rigid_cloth.joint_armature:g} kg m^2", flush=True)
        size_gpu_buffers(cfg)
        rd = cfg.domain_randomization.cloth
        if model == "priority" and (
            tuple(rd.slat_friction_scale) != (1.0, 1.0) or tuple(rd.robot_friction_scale) != (1.0, 1.0)
            or tuple(rd.table_friction_scale) != (1.0, 1.0)
        ):
            raise NotImplementedError(
                "friction DR on PhysX: a tensor-API material write resets the combine modes that "
                "match the cloth's friction, so it would silently average instead. Set "
                "domain_randomization.cloth.{slat,robot,table}_friction_scale to [1.0, 1.0], or use "
                "physx_chain.friction_model: cloth_min."
            )
        #: One-fold goal tasks (``fold_pool.goal_task``): built with the sheet layout, drawn from start.
        self.goal_tasks = None
        #: Set by :meth:`start`: until then resets keep the shape the startup proofs left in place.
        self.started = False
        self.physics_dr: PlateRandomization | None = None

    # ------------------------------------------------------------------ construction

    def build_scene(self, scene, sim) -> None:
        super().build_scene(scene, sim)  # ends with install_sheet
        contacts.bind_contact_materials(self)

    def install_sheet(self) -> None:
        pool_scene.install_pool(self)

    def start(self, reset_table) -> None:
        """Startup proofs, the physics DR's first draw, the folded-start tables. Draws per episode
        start at the trainer's first reset; until then every env holds the shape the proofs left on
        the table -- placed, so the cloud and the scene agree."""
        contacts.check_contacts(self)
        # RC_SKIP_VERIFY=1 is for DIAGNOSING a failed verification only. Never for training or eval.
        if os.environ.get("RC_SKIP_VERIFY", "0") != "1":
            checks.verify_placement(self, reset_table)
        else:
            print("[rigid_cloth] WARNING placement verification SKIPPED (RC_SKIP_VERIFY=1)", flush=True)
        self.physics_dr = PlateRandomization(self)
        self.shape_weights = self.shape_weights.to(self.device)
        pool_scene.init_start_folds(self)
        if self.goal_tasks is not None:
            pool_scene.start_goal_tasks(self)
        self.started = True

    # ------------------------------------------------------------------ the sheet, for the task

    def sheet_layout(self) -> SheetLayout:
        """The pool's grid; each env's at its own size. The fold lift is the pool's ply gap -- 0 mm
        for mid-plane hinges."""
        c = self.cfg.cloth
        lift = CLOTH_FOLD_LIFT
        if self.cfg.fold_pool.fold_lift_from_ply_gap:
            gaps = {sh.ply_gap for sh in self.shapes}
            if len(gaps) != 1:
                raise ValueError(
                    f"pool shapes disagree on ply gap {sorted(gaps)}; the fold target is built once "
                    f"for every env, so a mixed pool needs a per-env lift"
                )
            gap = gaps.pop()
            print(
                f"[fold_pool] fold target lift {lift * 1e3:.2f} -> {gap * 1e3:.2f} mm (the pool's ply gap)",
                flush=True,
            )
            lift = gap
        if self.goal_tasks is None:
            self.goal_tasks = pool_scene.build_goal_tasks(self)
        return SheetLayout(
            rest_local=self.rest_local,
            keypoint_idx=list(self.keypoint_idx),
            resolution=int(c.resolution),
            fold_axis=c.fold_axis,
            size=float(c.size),
            fold_lift=lift,
            object_scale=self.object_scale,
            xy_scale=self.xy_scale,
            env_size=self.env_size,
            goal_tasks=self.goal_tasks,
        )

    def read_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pool.positions(), self.pool.velocities()

    def write_sheet(self, env_ids: torch.Tensor, pose_w: torch.Tensor) -> None:
        yaw = pool_scene.goal_task_yaw(self, env_ids, reset_yaw(pose_w[:, 3:7]))
        self.pool.place(env_ids, pose_w[:, :2], yaw)

    def observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        # Isaac Lab 2.3's helper is wxyz; hand the task the (x, y, z, w) relabelling Newton does.
        return quat_from_matrix(rot)[..., [1, 2, 3, 0]]

    def scale3(self) -> torch.Tensor:
        """``(N, 1, 3)``: each env's in-plane size factor on x and y, 1 on z."""
        s = self.xy_scale
        return torch.stack((s, s, torch.ones_like(s)), dim=-1).unsqueeze(1)

    # ------------------------------------------------------------------ per reset / per step

    def begin_reset(self, env_ids: torch.Tensor) -> None:
        if self.started:
            self.draw_episode(env_ids)

    def draw_episode(self, env_ids: torch.Tensor) -> None:
        """The episode's shape and folded start for ``env_ids`` (a viewer may override it)."""
        pool_scene.draw_episode(self, env_ids)

    def draw_start_folds(self, env_ids: torch.Tensor, shapes: torch.Tensor) -> None:
        """The folded start of ``env_ids``, running ``shapes``."""
        pool_scene.draw_start_folds(self, env_ids, shapes)

    def end_reset(self, env_ids: torch.Tensor) -> None:
        if self.physics_dr is not None:
            self.physics_dr.draw(env_ids)

    def pre_physics_step(self) -> None:
        self.pool.park()

    # ------------------------------------------------------------------ the pool's articulations

    def pool_envs(self, s: int) -> list[int]:
        """The envs pool shape ``s`` exists in (before the simulator starts)."""
        if self.env_shape_np is None:
            return list(range(self.num_envs))
        return np.nonzero(self.env_shape_np == s)[0].tolist()

    def pool_rows(self, s: int, ids: torch.Tensor):
        """``(keep, local)``: which of env ``ids`` carry shape ``s`` (``None`` = all) and their
        instance indices in its articulation."""
        return self.pool.local_rows(s, ids)

    # ------------------------------------------------------------------ logging

    def add_termination_reasons(self, reasons: dict[str, torch.Tensor]) -> None:
        entered = reasons["fold"]
        active = self.pool.active
        for s, sh in enumerate(self.shapes):
            on = active == s
            # Averaged over finished episodes by the logger: `shape_<n>` is the share of episodes on
            # that shape, `fold_<n>` the share that were that shape AND folded. Their ratio is the
            # per-shape fold rate.
            reasons[f"shape_{sh.name}"] = on
            reasons[f"fold_{sh.name}"] = entered & on
        if self.start_tables is not None:
            # As above: `start_folded` is the share of episodes that started partly folded,
            # `fold_start_folded` the share that did AND folded.
            reasons["start_folded"] = self.start_folded
            reasons["fold_start_folded"] = entered & self.start_folded

    def add_episode_final(self, final: dict, reasons: dict, episode_length_buf: torch.Tensor) -> None:
        # How many steps a fold takes. The logger averages `episode_final` over finished episodes,
        # so `fold_steps` (the episode length if it folded, else 0) divided by `done_fold` is the
        # mean length of a folded episode; `fold_steps_flat_start` / `fold_flat_start` the same for
        # episodes that started flat. Not `done_*`: the evaluation reads those as booleans.
        # `episode_length_buf` already counts this step (DirectRLEnv.step increments it before
        # `_get_dones`), so it equals the evaluation's episode length for an episode ending here.
        entered = reasons["fold"].bool()
        flat_fold = entered & ~self.start_folded
        length = episode_length_buf.float()
        final["fold_steps"] = entered.float() * length
        final["fold_flat_start"] = flat_fold.float()
        final["fold_steps_flat_start"] = flat_fold.float() * length
