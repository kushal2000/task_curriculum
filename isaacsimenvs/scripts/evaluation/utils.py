"""Shared evaluation helpers: rollout protocol, contact-overflow guard, checkpoint policy.

Training, evaluation, visualization and deployment checks all import from here so that every
entry point measures under the same conditions.

Sections
--------
* **Protocol** -- :func:`disable_randomization`, :func:`use_single_object_variant`. Settings that
  define the comparison protocol, applied identically on every backend. They are not task
  configuration and not tuning: each one removes a source of variation that would otherwise make
  two backends' numbers incomparable, so they live in one place that both the PhysX and Newton
  runs import -- rather than being re-implemented per backend, which is how a protocol difference
  creeps in between two experiments.
* **Contact guard** -- :func:`capture_fd_output`, :func:`raise_if_overflowed`,
  :func:`assert_no_buffer_overflow`. Fail loudly when a Newton contact buffer overflows.
* **Policy player** -- :class:`PretrainedPlayer`, :func:`read_policy_cfg`. Load a checkpoint and
  drive it from a torch observation tensor.

Heavy dependencies (torch, numpy, omegaconf, yaml, rl_games) are imported inside the functions
that need them, so importing this module before the Isaac Sim app is launched stays cheap.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    # protocol
    "disable_randomization",
    "use_single_object_variant",
    "reapply_single_variant_if_requested",
    # contact guard
    "capture_fd_output",
    "assert_no_buffer_overflow",
    "raise_if_overflowed",
    "OVERFLOW_PATTERN",
    # policy player
    "PretrainedPlayer",
    "read_policy_cfg",
    "POLICY_OBS_DIM",
    "POLICY_ACTION_DIM",
    "SAPG_EXPL_COEF",
]


# ==============================================================================================
# Protocol
# ==============================================================================================

def disable_randomization(cfg) -> None:
    """Turn off every DR and reset-noise source, and pin the object's start pose.

    Domain randomisation is *on* by default in ``isaacsimenvs/scripts/configs/task/Play.yaml`` because that is the
    training regime. For evaluation it has to be off: obs/action delays, object-state noise and
    random wrench impulses all inject variance that has nothing to do with the backend under
    test, which is the comparison this measurement exists to support.
    """
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = 0.0
    dr.joint_pos_obs_noise_std = 0.0
    dr.action_noise_std = 0.0
    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    if hasattr(cfg, "task"):
        rs.random_yaw = False
        if hasattr(cfg, "rigid_cloth"):
            cfg.rigid_cloth.randomization.enabled = False
        return

    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.object_pose_bias_xyz = (0.0, 0.0, 0.0)
    dr.object_pose_bias_rot_degrees = 0.0
    dr.robot_stiffness_scale_range = (1.0, 1.0)
    dr.robot_damping_scale_range = (1.0, 1.0)
    dr.point_dr_mode = "off"
    dr.hand_friction_scale_range = (1.0, 1.0)
    dr.hand_effort_limit_scale_range = (1.0, 1.0)
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    rs.reset_position_noise_z = 0.0
    rs.fixed_start_pose = (
        rs.table_reset_xy[0],
        rs.table_reset_xy[1],
        rs.table_reset_z + rs.table_object_z_offset,
        1.0,
        0.0,
        0.0,
        0.0,
    )


#: Set when :func:`use_single_object_variant` has been requested, so a backend that patches
#: ``build_rigid_object_cfg`` during construction can re-apply the wrapper on top of its own.
_SINGLE_VARIANT_REQUESTED = False


def reapply_single_variant_if_requested() -> None:
    """Re-wrap ``build_rigid_object_cfg`` after a backend has installed its own patches.

    Ordering trap, and a silent one. :func:`use_single_object_variant` wraps whatever
    ``build_rigid_object_cfg`` is current, but the Newton backend's ``install_multi_asset``
    *replaces* that function during env construction -- which happens after the harness has
    already applied the wrapper. The wrapper is discarded, every env gets its own variant again,
    and the only symptom is ``SolverMuJoCo requires homogeneous worlds`` from deep inside the
    solver, which reads like a limitation rather than a lost patch.

    Calling this at the end of the backend's patch installation composes the two correctly.
    """
    if _SINGLE_VARIANT_REQUESTED:
        _wrap_build_rigid_object_cfg()


def use_single_object_variant() -> None:
    """Give every environment the same object, by truncating the spawn list to its first entry.

    Two distinct reasons to want this, and they are worth keeping separate:

    * **As a control.** It removes object diversity as a variable, so a difference between two
      stacks cannot be explained by them manipulating different tools.
    * **As a requirement.** ``SolverMuJoCo`` rejects worlds whose collision shapes differ in
      *type*, and the procedural tools mix boxes and capsules -- even a single ``handle_head``
      request emits both a handle-only variant (capsule) and a handle+head one (capsule + box).
      So without this, Newton cannot construct the scene at all::

          ValueError: SolverMuJoCo requires homogeneous worlds. Shape types mismatch at
          position 36: world 0 has type 4, but other worlds have types [7, 4, 7].

      The alternative -- ``meshify``, which makes every collider a convex mesh and so uniform in
      type -- does let Newton run the full pool, but it changes the geometry both backends see
      and therefore adds a confound to the very comparison being made.

    Wraps whichever ``build_rigid_object_cfg`` is currently installed, so it composes with the
    Newton multi-asset patch rather than replacing it. Idempotent.

    Applied to the *shared* ``scene_utils``, so both backends get an identical object. On a
    backend that re-patches ``build_rigid_object_cfg`` during construction, the wrapper is
    re-applied afterwards -- see :func:`reapply_single_variant_if_requested`.
    """
    global _SINGLE_VARIANT_REQUESTED
    _SINGLE_VARIANT_REQUESTED = True
    _wrap_build_rigid_object_cfg()


def _wrap_build_rigid_object_cfg() -> None:
    import importlib

    scene_utils = importlib.import_module("isaacsimenvs.tasks.play.utils.scene_utils")
    current = scene_utils.build_rigid_object_cfg
    if getattr(current, "_single_variant", False):
        return

    def one_variant(prim_path, usd_paths, _current=current):
        return _current(prim_path, list(usd_paths)[:1])

    one_variant._single_variant = True
    scene_utils.build_rigid_object_cfg = one_variant


# ==============================================================================================
# Contact guard
# ==============================================================================================

"""Fail loudly when a Newton contact buffer overflows.

Newton reports buffer overflows -- triangle pairs, rigid contacts, MJWarp's ``nefc`` -- by
printing from inside a kernel. That write goes to the process's *file descriptor 1*, not through
``sys.stdout``, so it is invisible to Python: no exception, no warning, no log record. The
simulation keeps running and silently drops the contacts that did not fit.

That failure mode cost this project a full day. Every sweep piped stdout through
``grep -E "goals/episode|lift fraction|..."`` to extract the result line, which discarded the
warnings, and 64-env runs were measured for hours with the coupler's proxy pipeline overflowing on
every step -- the proxy builds its own ``CollisionPipeline`` and defaults ``max_triangle_pairs`` to
a *global* 1e6, while the correctly sized budget went to a config object the proxy never reads.
Dropped hand-vs-cable contacts read exactly like "the policy cannot grasp a cable".

So this guard exists to convert a silent, greppable-away warning into an exception. Capturing the
fd is the only approach that works: the counters live in device arrays inside lazily built
pipelines that are not reachable from the solver object, and the message itself is the one signal
the engine reliably emits.

**Capturing has a failure mode of its own, and it bit hard.** While the capture is active, fd 1/2
point at a temp file, so warnings do not reach the terminal until it unwinds. A process that dies
abruptly -- a CUDA device-side assert, a signal -- never unwinds it, and the warnings die with the
temp file. Three cloth runs were diagnosed as "0 overflow warnings" on that basis; running with
``--allow_overflow`` (capture off) showed "Per-body rigid contact buffer overflowed 87 > 64" on
the very first attempt.

So the capture **tees**: everything written is copied straight through to the real stdout as well
as scanned. A diagnostic that can destroy the diagnostic it exists to preserve is worse than none.
"""

#: Matched against captured fd-1 text. Deliberately broad: an overflow that is not caught here is
#: worth far more than a false positive, which shows up immediately as a failed run.
OVERFLOW_PATTERN = re.compile(
    r"(buffer overflow|overflowed|exceeds? (?:the )?(?:maximum|capacity|limit)"
    r"|too many contacts|contact limit|nefc\s*>|njmax|increase\s+\w*max)",
    re.IGNORECASE,
)


@contextmanager
def capture_fd_output():
    """Capture writes to file descriptors 1 and 2, including those from native code.

    Yields a callable returning everything captured so far. ``sys.stdout`` is flushed first so
    Python-level output already queued does not land inside the capture window.
    """
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)

            forwarded = [0]

            def read_so_far() -> str:
                sys.stdout.flush()
                sys.stderr.flush()
                pos = tmp.tell()
                tmp.seek(0)
                data = tmp.read()
                tmp.seek(pos)
                # Tee: copy anything new straight to the real stdout, so a hard crash cannot take
                # the warnings with it. Written to the saved fd, not `sys.stdout`, which is still
                # redirected at this point.
                if len(data) > forwarded[0]:
                    os.write(saved_out, data[forwarded[0]:])
                    forwarded[0] = len(data)
                return data.decode("utf-8", errors="replace")

            yield read_so_far
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def raise_if_overflowed(captured: str, num_envs: int, where: str) -> None:
    """Raise if ``captured`` contains an overflow warning."""
    hits = [ln.strip() for ln in captured.splitlines() if OVERFLOW_PATTERN.search(ln)]
    if not hits:
        return
    unique = list(dict.fromkeys(hits))[:6]
    raise RuntimeError(
        f"Newton reported a contact-buffer overflow at num_envs={num_envs} during {where} "
        f"({len(hits)} warnings). Contacts are being dropped and every number from this run would "
        f"be measured on a scene where the hand cannot fully feel the object."
        f"\n\n  " + "\n  ".join(unique) + "\n\n"
        "Buffer budgets in Newton are GLOBAL, not per-env, so they must scale with num_envs AND "
        "with scene geometry -- a 24-segment cable generates far more triangle pairs than a "
        "12-segment one at the same env count. Check that the sized budget reaches the solver: a "
        "coupled scene builds a separate CollisionPipeline per proxy, and that one defaults to "
        "max_triangle_pairs=1e6 regardless of NewtonCfg.collision_cfg."
    )


def assert_no_buffer_overflow(env, steps: int = 20, action=None) -> None:
    """Step ``env`` briefly with fd output captured; raise if a buffer overflowed.

    A warm-up check only. **It is not sufficient on its own**: it steps with zero actions, so the
    hand is not touching the object and the contact count is at its lowest. A 24-segment cable
    passed this check and then overflowed for hundreds of steps once the policy started
    manipulating. Use `capture_fd_output` around the whole rollout as well -- see
    `isaacsimenvs/scripts/evaluation/eval.py`.

    Raises:
        RuntimeError: with the offending lines, the env count, and the usual cause.
    """
    import torch

    inner = env.unwrapped
    if action is None:
        action = torch.zeros(
            (inner.num_envs, int(inner.cfg.action_space)), device=inner.device
        )

    with capture_fd_output() as read_so_far:
        for _ in range(steps):
            env.step(action)
        captured = read_so_far()

    raise_if_overflowed(captured, inner.num_envs, f"a {steps}-step warm-up")


# ==============================================================================================
# Policy player
# ==============================================================================================

"""Load the released SimToolReal checkpoint and drive it from a torch observation tensor.

A repo-local port of ``deployment/rl_player.py`` from the upstream SimToolReal checkout. It is
ported rather than imported because upstream's version pulls ``isaacgymenvs.utils.reformat``,
which does not exist here.

The checkpoint is an asymmetric-actor-critic LSTM (``rnn.name: lstm``, 1024 units,
``before_mlp: true``, ``seq_length: 16``) trained under the Isaac Gym pipeline with SAPG, with
``normalize_input`` and ``normalize_value`` both on. Its running-mean/std statistics are baked
against the exact observation layout below, so the observation must be assembled by the env
rather than reconstructed here.
"""

# The observation the policy was trained on. `play`'s obs_list produces exactly this.
POLICY_OBS_DIM = 140
POLICY_ACTION_DIM = 29

# SAPG conditions the policy on an exploration coefficient appended as a final observation
# column, so the network takes 141 inputs for a 140-dim environment observation. Upstream's
# player calls this the "SAPG HACK" (deployment/rl_player.py:97-102) and uses 50.0, the value
# the released checkpoint was evaluated at.
#
# Omitting it does not crash: the first Linear would simply reject the shape -- but any code that
# pads or reshapes instead produces finite, plausible, and wrong actions. Asserted below rather
# than trusted.
#
# 50.0 is SAPG block 0 -- the block with the LARGEST entropy bonus (`a2c_common.py:342,351`:
# embedding linspace(50, 0, num_blocks), intrinsic coef linspace(0.5, 0)). Block num_blocks-1 is
# coefficient 0.0, trained on task reward alone. A finetuned checkpoint need not share the released
# one's best block: in the prior-vs-scratch cloth runs every fold came from block 5 (0.0) while
# blocks 0-4 folded ~0.1% of episodes, and evaluating at 50.0 measured the wrong policy. Pass
# `expl_coef` explicitly for anything other than the released checkpoint.
SAPG_EXPL_COEF = 50.0

# Registered by `isaacgymenvs/__init__.py` upstream. The released config interpolates through all
# of them (`${resolve_default:...}` on the run name, `${eval:"None"}` on several task fields), so
# resolution fails with UnsupportedInterpolationType without them.
_RESOLVERS = {
    "eq": lambda x, y: x.lower() == y.lower(),
    "contains": lambda x, y: x.lower() in y.lower(),
    "if": lambda pred, a, b: a if pred else b,
    "resolve_default": lambda default, arg: default if arg == "" else arg,
    "eval": lambda x: eval(x),  # noqa: S307 - upstream's resolver, used on its own config
}


def _register_resolvers() -> None:
    from omegaconf import OmegaConf

    for name, fn in _RESOLVERS.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


def read_policy_cfg(config_path: str | Path, device: str, overrides: list[str] | tuple = ()) -> dict:
    """Read the checkpoint's own ``config.yaml``, resolving upstream's custom interpolations.

    Accepts both layouts: upstream's released configs keep the rl_games settings under
    ``train:``, while this repo's training runs (``<run>/.hydra/config.yaml``) keep the same
    settings under ``agent:``. The latter is exposed as ``train`` as well.

    ``overrides`` are dotted ``key=value`` strings (``env.rigid_cloth.designs=[strip_4.yaml]``)
    merged onto the file before interpolations resolve. Each key, or its parent section, must
    already exist in the file, so a misspelled section is an error instead of a silently ignored setting.
    """
    import yaml
    from omegaconf import OmegaConf

    from isaacsimenvs.scripts.configs.formatting import omegaconf_to_dict

    _register_resolvers()
    raw = OmegaConf.create(yaml.safe_load(Path(config_path).read_text()))
    if overrides:
        # The key, or at least its parent section, must exist in the file: a setting added to the
        # code after the run was saved is allowed, a typo in a section name is not. (Unknown env
        # leaf keys are rejected again when the env configclass loads them.)
        missing = object()
        for item in overrides:
            key = item.split("=", 1)[0]
            parent = key.rsplit(".", 1)[0] if "." in key else None
            if "=" not in item or (OmegaConf.select(raw, key, default=missing) is missing
                                   and (parent is None or not isinstance(OmegaConf.select(raw, parent, default=None),
                                                                         type(raw)))):
                raise KeyError(f"override {item!r}: expected key=value whose key (or its parent section) "
                               f"exists in {config_path}")
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(list(overrides)))
    cfg = omegaconf_to_dict(raw)
    if "train" not in cfg and "agent" in cfg:
        cfg["train"] = cfg["agent"]
    train_cfg = cfg["train"]["params"]["config"]
    train_cfg["device"] = device
    train_cfg["device_name"] = device
    return cfg


class _SpacesOnlyEnv:
    """The minimum rl_games needs to build a player: the spaces, and nothing else.

    ``Runner.create_player`` resolves its env through ``env_configurations``, so a placeholder has
    to be registered even though the real stepping is driven by the caller.
    """

    def __init__(self, num_observations: int, num_actions: int, num_envs: int) -> None:
        import numpy as np
        from gym import spaces

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
        self.num_envs = num_envs

    def get_env_info(self) -> dict:
        return {"observation_space": self.observation_space, "action_space": self.action_space}

    def set_env_state(self, *args: Any, **kwargs: Any) -> None:
        return None


class PretrainedPlayer:
    """Checkpoint policy with configurable observation/action widths and SAPG conditioning."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        num_envs: int,
        device: str = "cuda:0",
        num_observations: int = POLICY_OBS_DIM,
        num_actions: int = POLICY_ACTION_DIM,
        expl_coef: float = SAPG_EXPL_COEF,
        overrides: list[str] | tuple = (),
    ) -> None:
        from rl_games.common import env_configurations
        from rl_games.torch_runner import Runner

        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.expl_coef = float(expl_coef)

        self.cfg = read_policy_cfg(config_path, device, overrides)

        # The space is the *environment's* 140, not the 141 columns handed to `get_action`.
        # rl_games derives both the input normalizer and the network input width from it:
        # `running_mean_std` is built over `obs_shape` (140, matching the checkpoint), while
        # `PpoPlayerContinuous.__init__` sets the network input to
        # `obs_shape + intr_reward_coef_embd.shape[1]` = 140 + 32 = 172, which is the LSTM's
        # `weight_ih_l0` width in the checkpoint. The trailing coefficient column is located by
        # `coef_id_idx = obs_shape[0]` and expanded into that 32-dim embedding internally
        # (`expl_type: mixed_expl_learn_param` selects the 'extra_param' model).
        #
        # Passing 141 here instead makes every one of those 140s a 141 and the restore fails on
        # a shape mismatch -- which is the good outcome. It is recorded because the failure names
        # `running_mean_std`, not the coefficient, and reads like a corrupt checkpoint.
        placeholder = _SpacesOnlyEnv(num_observations, num_actions, num_envs)
        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kw: placeholder, "vecenv_type": "RLGPU"}
        )

        train_cfg = dict(self.cfg["train"])
        train_cfg["load_path"] = str(checkpoint_path)

        runner = Runner()
        runner.load(train_cfg)
        self.player = runner.create_player()
        self.player.init_rnn()
        self.player.has_batch_dimension = True
        self.player.restore(str(checkpoint_path))

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Map an observation batch to normalized joint actions using the checkpoint policy."""
        if obs.shape[-1] != self.num_observations:
            raise ValueError(
                f"expected a {self.num_observations}-dim observation, got {obs.shape[-1]}. "
                "The checkpoint's input normalization is baked against the exact layout in "
                "isaacsimenvs/scripts/configs/task/Play.yaml obs_list -- a different width means a different env, not a "
                "reshape."
            )
        import torch

        obs = obs.to(self.device)
        expl = torch.full((obs.shape[0], 1), self.expl_coef, device=self.device, dtype=obs.dtype)
        return self.player.get_action(obs=torch.cat([obs, expl], dim=1), is_deterministic=deterministic)

    def reset_rnn(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the LSTM hidden state, for all envs or a subset.

        rl_games stores states as ``(layers, num_envs, units)`` tensors. Resetting per-env matters
        once episodes end at different steps: carrying a finished episode's hidden state into the
        next one is not what the policy saw in training.
        """
        states = getattr(self.player, "states", None)
        if not states:
            return
        for state in states:
            if state is None:
                continue
            if env_ids is None:
                state.zero_()
            else:
                state[:, env_ids] = 0.0
