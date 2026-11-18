"""``RigidClothEnv``: the fold task on the fold-shape pool of hinged plates, on PhysX.

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv_isaacsim``). The task (``cloth_env.ClothEnv`` and ``mdp/``) is
shared with the VBD backend; this class adds the pool and its physics:

    scene.py           the pool's articulations, per-env sizes, shape draws, folded starts
    contacts.py        friction materials and contact offsets (PhysX USD, see physx.py)
    randomization.py   per-env physics DR at every reset
    checks.py          startup proofs
    pool.py            the adapter that presents the active shape as the cloth's particle grid

Every env carries every pool shape (one active, the rest parked far away) unless
``fold_pool.one_shape_per_env``, where each env carries one. Each env's sheets are built at its own
size; every rest quantity of the fold is scaled per env below.
"""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix

from isaacsimenvs.tasks.cloth.backends.rigid.checks import ChecksMixin
from isaacsimenvs.tasks.cloth.backends.rigid.config import RigidClothEnvCfg
from isaacsimenvs.tasks.cloth.backends.rigid.contacts import ContactsMixin
from isaacsimenvs.tasks.cloth.backends.rigid.physx import DEFAULT_MU, reorder_object_pose_writes, size_gpu_buffers
from isaacsimenvs.tasks.cloth.backends.rigid.randomization import RandomizationMixin
from isaacsimenvs.tasks.cloth.backends.rigid.scene import SceneMixin
from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.task import ClothTaskMixin
from isaacsimenvs.tasks.cloth.mdp.tracker import TrackerMixin

__all__ = ["RigidClothEnv"]


class RigidClothEnv(
    SceneMixin, ContactsMixin, RandomizationMixin, ChecksMixin, TrackerMixin, ClothTaskMixin, ClothEnv
):
    """The fold task with the fold-shape pool as the manipuland, on PhysX."""

    cfg: RigidClothEnvCfg

    def __init__(self, cfg: RigidClothEnvCfg, render_mode: str | None = None, **kwargs):
        # --- the pool, before the scene is built (``_setup_scene`` and the first reset run inside
        # ``DirectRLEnv.__init__``).
        p = cfg.fold_pool
        self._pool = [fs.shape(n) for n in p.shapes]
        w = [1.0] * len(self._pool) if p.weights is None else [float(x) for x in p.weights]
        self._pool_weights = torch.tensor(w, dtype=torch.float32)
        thickness = {sh.thickness for sh in self._pool}
        if len(thickness) != 1:
            raise ValueError(f"pool shapes disagree on thickness {sorted(thickness)}")
        slab0 = thickness.pop()
        # The contact-margin sentinel resolves against the pool's own (nominal) plate thickness.
        r = cfg.rigid_cloth
        if r.contact_margin < 0.0:
            r.contact_margin = max(0.5 * float(cfg.cloth.thickness) - 0.5 * slab0, 0.0)
        # Per-env size and slab thickness, drawn ONCE (see FoldPoolCfg.size_range): each env's USD is
        # built at its size, so it cannot change afterwards.
        n = int(cfg.scene.num_envs)
        self._env_size_np, self._env_slab_np = self._draw_env_sizes(cfg, n)
        self._env_shape_np = None
        if p.one_shape_per_env:
            S = len(self._pool)
            if p.weights is None:
                self._env_shape_np = np.arange(n) % S
            else:
                w = np.asarray([float(x) for x in p.weights])
                self._env_shape_np = np.random.default_rng(int(p.size_seed) + 1).choice(S, n, p=w / w.sum())
        self._xy_np = self._env_size_np / float(cfg.cloth.size)
        self._z_np = self._env_slab_np / slab0

        # --- PhysX contact setup that must be in the cfg before the scene is built.
        a = cfg.assets
        # The scene's material pass writes every robot/table/object shape through the tensor API,
        # which resets combine modes (see physx.py). Contacts are authored in USD instead,
        # so it must not run; the colliders it would have set to 0.5 keep PhysX's default 0.5.
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
        a.modify_asset_frictions = False
        # A PhysX numerical requirement, not a task parameter: see PhysxChainCfg.hinge_armature. Written
        # into the task block so the actuator (and the logs) carry the value the hinges really get.
        if float(cfg.physx_chain.hinge_armature) > 0.0:
            cfg.rigid_cloth.joint_armature = float(cfg.physx_chain.hinge_armature)
        print(f"[physx] hinge armature {cfg.rigid_cloth.joint_armature:g} kg m^2", flush=True)
        size_gpu_buffers(cfg)
        rd = cfg.rigid_cloth.randomization
        if model == "priority" and rd.enabled and (
            tuple(rd.slat_friction_scale) != (1.0, 1.0) or tuple(rd.fingertip_friction_scale) != (1.0, 1.0)
            or tuple(rd.table_friction_scale) != (1.0, 1.0)
        ):
            raise NotImplementedError(
                "friction DR on PhysX: a tensor-API material write resets the combine modes that "
                "match the cloth's friction, so it would silently average instead. Set "
                "rigid_cloth.randomization.{slat,fingertip,table}_friction_scale to [1.0, 1.0], or use "
                "physx_chain.friction_model: cloth_min."
            )
        super().__init__(cfg, render_mode, **kwargs)
        # AFTER ClothEnv.__init__: its material pass would overwrite plate friction written earlier.
        self._check_contacts()
        # RC_SKIP_VERIFY=1 is for DIAGNOSING a failed verification only. Never for training or eval.
        if __import__("os").environ.get("RC_SKIP_VERIFY", "0") != "1":
            self._calibrate_and_verify_placement()
        else:
            print("[rigid_cloth] WARNING placement verification SKIPPED (RC_SKIP_VERIFY=1)", flush=True)
        self._init_physics_randomization()
        self._pool_weights = self._pool_weights.to(self.device)
        self._init_start_folds()
        # Draws start now, at the trainer's first reset. Until then every env holds the shape the
        # startup proofs left on the table -- placed, so the cloud and the scene agree.
        self._pool_ready = True

    # ------------------------------------------------------------------ Isaac Lab hooks

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._install_cloth()
        self._bind_contact_materials()
        reorder_object_pose_writes(self.object)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if getattr(self, "_pool_ready", False):
            self._draw(ids)
        super()._reset_idx(env_ids)
        if getattr(self, "_dr_ready", False):
            self._randomize_physics(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self.object.park()

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        entered = self._termination_reasons["fold"]
        active = self.object.active
        for s, sh in enumerate(self._pool):
            on = active == s
            # Averaged over finished episodes by the logger: `shape_<n>` is the share of episodes on
            # that shape, `fold_<n>` the share that were that shape AND folded. Their ratio is the
            # per-shape fold rate.
            self._termination_reasons[f"shape_{sh.name}"] = on
            self._termination_reasons[f"fold_{sh.name}"] = entered & on
        if getattr(self, "_start_tables", None) is not None:
            # As above: `start_folded` is the share of episodes that started partly folded,
            # `fold_start_folded` the share that did AND folded.
            self._termination_reasons["start_folded"] = self._start_folded
            self._termination_reasons["fold_start_folded"] = entered & self._start_folded
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        # How many steps a fold takes. The logger averages `episode_final` over finished episodes,
        # so `fold_steps` (the episode length if it folded, else 0) divided by `done_fold` is the
        # mean length of a folded episode; `fold_steps_flat_start` / `fold_flat_start` the same for
        # episodes that started flat. Not `done_*`: eval/episodes.py reads those as booleans.
        # `episode_length_buf` already counts this step (DirectRLEnv.step increments it before
        # `_get_dones`), so it equals eval's `per_env_length` for an episode ending here.
        entered = self._termination_reasons["fold"].bool()
        flat_fold = entered & ~self._start_folded
        length = self.episode_length_buf.float()
        final = self.extras["episode_final"]
        final["fold_steps"] = entered.float() * length
        final["fold_flat_start"] = flat_fold.float()
        final["fold_steps_flat_start"] = flat_fold.float() * length
        if getattr(self, "_tracker", None) is not None:
            # Cloth tracker DR: per-episode frozen share of camera frames, fresh share per half,
            # covered share by the hand and by the cloth itself.
            final.update(self._tracker.episode_stats())
        return reward

    # ------------------------------------------------------------------ the pool's articulations

    def _manipuland_arts(self) -> list:
        return list(self._pool_arts)

    def _manipuland_envs(self, k: int) -> list[int]:
        """The envs pool articulation ``k`` exists in (before the simulator starts)."""
        return self._pool_envs(k)

    def _manipuland_rows(self, k: int, ids: torch.Tensor):
        """``(keep, local)``: which of env ``ids`` carry shape ``k`` (``None`` = all) and their
        instance indices in its articulation."""
        return self._cloth.local_rows(k, ids)

    # ------------------------------------------------------------------ the fold, per-env size
    def _init_fold_targets(self) -> None:
        """The pool's ply gap as the target lift -- 0 mm for mid-plane hinges."""
        if self.cfg.fold_pool.fold_lift_from_ply_gap:
            gaps = {sh.ply_gap for sh in self._pool}
            if len(gaps) != 1:
                raise ValueError(
                    f"pool shapes disagree on ply gap {sorted(gaps)}; the fold target is built once "
                    f"for every env, so a mixed pool needs a per-env lift"
                )
            gap = gaps.pop()
            print(
                f"[fold_pool] fold target lift {self.cfg.cloth.self_contact_radius * 1e3:.2f} -> "
                f"{gap * 1e3:.2f} mm (the pool's ply gap)",
                flush=True,
            )
            self.cfg.cloth.self_contact_radius = gap
        ClothTaskMixin._init_fold_targets(self)
        # The task builds the rest geometry once, for a `cloth.size` sheet. Every quantity below is
        # that geometry scaled in-plane by the env's own factor (heights -- the fold lift -- are not
        # lengths in the sheet's plane and stay as they are).
        s3 = self._scale3()
        self._corner_rest_offsets = self._corner_rest_offsets.unsqueeze(0) * s3
        self._mv_folded_rest = self._mv_folded_rest.unsqueeze(0) * s3
        self._kp_folded_rest = self._kp_folded_rest.unsqueeze(0) * s3
        self._st_rest_centroid = self._st_rest_centroid.unsqueeze(0) * s3[:, 0]
        self._st_rest_centred = self._st_rest_centred.unsqueeze(0) * s3

    def _scale3(self) -> torch.Tensor:
        """``(N, 1, 3)``: in-plane factor on x and y, 1 on z."""
        s = self._xy_scale
        return torch.stack((s, s, torch.ones_like(s)), dim=-1).unsqueeze(1)

    # The fold-geometry methods of ``ClothTaskMixin`` that read the shared rest geometry, each the same
    # computation with a per-env rest.

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

    def footprint_ratio(self) -> torch.Tensor:
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1
        parts = self._particles_w()
        r, _ = self._stationary_frame(parts)
        coord = torch.einsum("npi,ni->np", parts, r[:, :, ax])
        return (coord.amax(dim=1) - coord.amin(dim=1)) / self._env_size

    def fold_fraction(self) -> torch.Tensor:
        rest = (
            torch.tensor(
                [self._cloth_rest_local[i] for i in self._cloth_kp_idx],
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            * self._scale3()
        )
        start = (
            rest
            + torch.tensor(
                [0.0, 0.0, float(self.cfg.reset.table_reset_z) + self.cfg.cloth.start_height],
                device=self.device,
            )
            + (self.scene.env_origins + self.object.centre_offset).unsqueeze(1)
        )
        span = (start - self.fold_targets_w()).norm(dim=-1).amax(dim=-1).clamp(min=1e-6)
        return (1.0 - self.fold_error() / span).clamp(0.0, 1.0)

    def _goal_quat_from_matrix(self, rot: torch.Tensor) -> torch.Tensor:
        # Isaac Lab 2.3's helper is wxyz; hand the task the (x, y, z, w) relabelling Newton does.
        return quat_from_matrix(rot)[..., [1, 2, 3, 0]]
