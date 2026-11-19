"""The quaternion helpers the cloth task's observation, reset and randomization code calls, in
(w, x, y, z).

Isaac Lab's own helpers, bound here once. The other ``mdp`` modules call them as ``quat.<name>``
(a module attribute, looked up at call time), so the Newton backend can rebind them in this one
module: Isaac Lab 3.0 stores and computes quaternions in (x, y, z, w), and
``backends/vbd/newton_support/patches.install_quat_math`` wraps each of these to keep (w, x, y, z)
semantics. ``convert_quat`` is explicit about its orders and is never wrapped; nor is
``quat_from_matrix``, which the backends import from Isaac Lab directly (see
the backend's ``_observed_goal_quat``).
"""

from __future__ import annotations

from isaaclab.utils.math import (  # noqa: F401
    convert_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    random_orientation,
)

__all__ = ["convert_quat", "quat_apply", "quat_from_angle_axis", "quat_mul", "random_orientation"]
