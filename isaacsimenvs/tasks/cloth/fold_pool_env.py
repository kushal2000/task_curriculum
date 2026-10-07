"""FoldPoolEnv -- stage 2: the half-fold task on a shape drawn per episode from the fold-shape pool.

Subclasses :class:`RigidClothEnv` and inherits everything that is about the TASK or the PHYSICS of a
rigid sheet -- fold targets, reward, success, observations (``ClothEnv``), slat friction, the 7 mm
contact margin, removal of plate-plate contact, physics DR (``RigidClothEnv``). Those all key on
model labels with the ``slat_`` prefixes, which the pool's USDs reuse, so they cover every pool
shape at once. What is replaced is what assumed ONE chain:

    ``_install_cloth``                    five articulations from ``fold_shape_assets`` USDs
    ``_calibrate_and_verify_placement``   the chain's startup proofs, run once per shape
    ``_reset_idx``                        draw each env's shape, then the inherited reset places it
    ``_pre_physics_step``                 re-park the unused shapes every step
    ``_init_fold_targets``                lift by the pool's ply gap (0 mm), and scale every rest
                                          quantity to the env's own sheet size
    ``_stationary_frame`` / ``_folded_w`` / ``_sync_observed_keypoints`` / ``footprint_ratio`` /
    ``fold_fraction``                     ClothEnv's versions, with a per-env rest geometry
    ``_write_object_scale``               each env's own size and felt thickness
    ``_get_dones``                        add per-shape fold rates to the episode logs

**Why five articulations per env.** Newton cannot vary an articulation's topology across worlds
(``ArticulationView``: "Articulations are not identical"), and ``SolverMuJoCo`` builds the MuJoCo
model from one template world. So every world holds all five shapes; the drawn one is on the table
and the rest hover ``fold_pool.park_distance`` away, untouchable -- worlds never collide. See
``utils/fold_pool_adapter.py``.

**Per-env size** (``fold_pool.size_range``) is the one geometric difference between worlds, and it is
set while the scene is REPLICATED (a per-world builder hook, ``_register_size_hook``): Newton ignores
a later change of a mesh's scale. So sizes are drawn once per env, not per episode, and a startup
check confirms each env's plate mass matches its draw.

    .venv_isaaclab3/bin/python isaacsimenvs/train.py \\
        --task Isaacsimenvs-FoldPool-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless

The policy is NOT told which shape it has: the observation layout is the cloth task's 140-dim one,
so a stage-1 checkpoint loads unchanged. Each shape differs only in where the sheet can bend.

The pool itself -- shapes, parking, draws, per-env size in the fold geometry, startup proofs -- is
``fold_pool_task.FoldPoolTaskMixin``, shared with the PhysX env. This class is the Newton side: the
per-world builder hook that sizes each env's sheets, and the mass read-back that checks it.
"""

from __future__ import annotations

import numpy as np
import torch

from isaacsimenvs.tasks.cloth.fold_pool_env_cfg import FoldPoolEnvCfg
from isaacsimenvs.tasks.cloth.fold_pool_task import FoldPoolTaskMixin
from isaacsimenvs.tasks.cloth.rigid_cloth_env import RigidClothEnv, wp_to_torch
from isaacsimenvs.tasks.cloth.utils import fold_shape_assets as fsa

__all__ = ["FoldPoolEnv"]


class FoldPoolEnv(FoldPoolTaskMixin, RigidClothEnv):
    cfg: FoldPoolEnvCfg

    def _pool_after_spawn(self) -> None:
        self._register_size_hook()

    def _pool_build_done(self) -> None:
        self._unregister_size_hook()

    def _check_pool_build(self) -> None:
        if self._size_hook is not None and self._scaled_worlds != self.num_envs:
            raise RuntimeError(
                f"[fold_pool] the size hook resized {self._scaled_worlds} worlds for "
                f"{self.num_envs} envs -- per-env sheet sizes did not reach the simulator"
            )

    def _pool_plate_mass(self) -> torch.Tensor:
        return wp_to_torch(self._model.body_mass)[self._slat_bodies].sum(dim=1)

    # ------------------------------------------------------------------ per-env size

    _size_hook = None

    def _register_size_hook(self) -> None:
        """Resize every world's pool sheets to its env's draw, as Newton replicates the scene.

        A per-world builder hook runs once per env, after that env's copy of the scene has been
        added and before the model is finalized -- the only point at which plate geometry can
        still differ between worlds. Skipped entirely when nothing is randomised, which keeps
        Isaac Lab's batched replication path.
        """
        if np.allclose(self._xy_np, 1.0) and np.allclose(self._z_np, 1.0):
            return
        import newton
        from isaaclab_newton.physics import NewtonManager

        xy, z = self._xy_np, self._z_np
        cursor = {"body": 0, "joint": 0, "shape": 0}
        free = newton.JointType.FREE

        def new_ids(labels, worlds, key, world):
            start, end = cursor[key], len(labels)
            cursor[key] = end
            return [
                i
                for i in range(start, end)
                if worlds[i] == world and "/FoldPool" in (labels[i] or "")
            ]

        def hook(builder, world, _pos, _quat) -> None:
            bodies = new_ids(builder.body_label, builder.body_world, "body", world)
            joints = [
                j
                for j in new_ids(builder.joint_label, builder.joint_world, "joint", world)
                if builder.joint_type[j] != free
            ]
            shapes = new_ids(builder.shape_label, builder.shape_world, "shape", world)
            if not bodies:
                return
            fsa.scale_builder_world(
                builder, bodies, joints, shapes, float(xy[world]), float(z[world])
            )
            self._scaled_worlds += 1

        NewtonManager._per_world_builder_hooks.append(hook)
        self._size_hook = hook

    def _unregister_size_hook(self) -> None:
        if self._size_hook is None:
            return
        from isaaclab_newton.physics import NewtonManager

        if self._size_hook in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.remove(self._size_hook)
