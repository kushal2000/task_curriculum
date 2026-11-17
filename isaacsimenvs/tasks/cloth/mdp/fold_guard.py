"""Tensor-only geometry and lifting-work history for rigid-cloth episode resets."""
from __future__ import annotations

import math

import torch

from .geometry import rotations


def table_excursion(vertices_w, table_position, table_quaternion, half_size_xy):
    """Maximum physical-vertex overhang [m], in the tabletop's own XY frame."""
    inverse = table_quaternion * table_quaternion.new_tensor([1., -1., -1., -1.])
    local = rotations.apply(inverse[:, None, None], vertices_w-table_position[:, None, None])
    return (local[..., :2].abs()-half_size_xy).amax(dim=(1, 2, 3)).clamp_min(0)


def robot_opening_work(point, force, position, quaternion, com, velocity,
                       previous_position, previous_quaternion, previous_com, previous_velocity,
                       anchor, axis, opening_sign, dt):
    """Work delivered at the robot contact point, for forces favoring opening.

    A static robot gets no credit for cloth depenetration. A moving side pull
    can earn credit without requiring a vertical fingertip trajectory.
    """
    inverse = quaternion * quaternion.new_tensor([1., -1., -1., -1.])
    local = rotations.apply(inverse, point-position)
    old_point = previous_position+rotations.apply(previous_quaternion, local)
    current_v = velocity[:, :3]+torch.cross(velocity[:, 3:], point-com, dim=-1)
    old_v = previous_velocity[:, :3]+torch.cross(previous_velocity[:, 3:], old_point-previous_com, dim=-1)
    work = (force*(current_v+old_v)*.5).sum(-1)*dt
    moment = (torch.cross(point-anchor, force, dim=-1)*axis).sum(-1)*opening_sign
    return torch.where(moment > 0, work, 0.)


class LiftWorkHistory:
    """Per-half credits persist after release and rearm only on a settled flat half."""

    def __init__(self, num_envs, device, cfg):
        self.cfg = cfg
        self.credit = torch.zeros(num_envs, 2, device=device)
        self.reference_height = torch.zeros_like(self.credit)
        self.previous_gravity = torch.zeros_like(self.credit)
        self.gravity = torch.zeros_like(self.credit)
        self.ready = torch.zeros_like(self.credit, dtype=torch.bool)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids):
        for value in (self.credit, self.reference_height, self.previous_gravity, self.gravity, self.ready, self.initialized):
            value[ids] = 0

    def update(self, *, source_work, height, weight, normal_up, minimum_plate_up,
               maximum_linear_speed, maximum_angular_speed, table_up, opposed_load,
               inside, table_top, dimensions):
        cfg = self.cfg
        self.credit += source_work
        flat = minimum_plate_up > math.cos(math.radians(5))
        flat &= (maximum_linear_speed < .02) & (maximum_angular_speed < .5)
        flat &= height-table_top[:, None] < dimensions[:, 2, None]/2+.003
        flat &= (table_up > 1e-5) & inside[:, None]
        self.ready |= flat
        self.initialized |= flat.all(dim=1)
        self.credit.masked_fill_(flat, 0.)
        self.reference_height = torch.where(flat, height, self.reference_height)
        gravity = weight*(height-self.reference_height)
        previous_gravity = torch.where(flat, torch.zeros_like(gravity), self.previous_gravity)
        gaining_height = gravity-previous_gravity > 1e-8
        self.previous_gravity.copy_(gravity)
        self.gravity = gravity

        base = normal_up > math.cos(math.radians(cfg.maximum_base_tilt_deg))
        base &= height-table_top[:, None] < dimensions[:, 2, None]/2+.015
        base &= opposed_load > cfg.minimum_opposed_load
        rising = (normal_up < math.cos(math.radians(cfg.minimum_lift_angle_deg))) & (normal_up > 0)
        rising &= gaining_height & self.ready & inside[:, None]
        rising &= gravity > weight*torch.maximum(dimensions[:, 0, None]*.05, dimensions.new_tensor(.005))
        missing_input = self.credit < cfg.minimum_lifting_work_fraction*gravity
        return (rising & base.flip(1) & missing_input).any(dim=1)
