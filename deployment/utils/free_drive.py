"""Free drive gains: no stiffness and light damping, with a soft wall before each joint limit, and a
ramp between free drive and holding (blend_gains) so that the stiffness never comes or goes in one step.

Flexiv's own soft limits (StreamJointTorque's enable_soft_limits) push back hard and suddenly when a
joint gets close to its limit. The wall here starts `margin` before the limit and grows from zero, so
the hand feels it first and the joint does not get that far.
"""

import numpy as np


def wall_gains(q, lower, upper, margin, kp_wall, kd_free, kd_wall):
    """(q_desired, kp, kd, depth) for aioflexiv's joint impedance, tau = kp (q_desired - q) - kd dq.

    Away from the limits: kp 0, kd_free. Within `margin` (rad) of a limit: a spring kp_wall back to
    the wall's start (zero torque where it starts), and damping rising linearly to kd_wall at the limit.
    depth: how far into the wall each joint is, 0 (outside) to 1 (at the limit) and beyond.
    q_desired is q for the joints outside a wall, so going back to holding holds here."""
    q = np.asarray(q, float)
    q_desired = np.clip(q, lower + margin, upper - margin)
    depth = np.abs(q - q_desired) / margin
    kp = np.where(depth > 0, kp_wall, 0.0)
    kd = kd_free + (kd_wall - kd_free) * np.minimum(depth, 1.0)
    return q_desired, kp, kd, depth


def blend_gains(a, hold, free):
    """Gains `a` of the way from `free` (a = 0) to `hold` (a = 1), each (q_desired, kp, kd): the torque
    is exactly a x hold's + (1 - a) x free's, so ramping `a` switches between the two without a step."""
    (q_hold, kp_hold, kd_hold), (q_free, kp_free, kd_free) = hold, free
    kp_hold, kp_free = a * np.asarray(kp_hold, float), (1.0 - a) * np.asarray(kp_free, float)
    kp = kp_hold + kp_free
    q_desired = np.where(kp > 0, (kp_hold * q_hold + kp_free * q_free) / np.where(kp > 0, kp, 1.0), q_free)
    return q_desired, kp, a * np.asarray(kd_hold, float) + (1.0 - a) * np.asarray(kd_free, float)
