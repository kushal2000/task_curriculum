"""Hinge plasticity of the plates: a crease that stays.

A hinge spring (``rigid_cloth.joint_stiffness``) pulls toward a REST angle. Elastic: the rest angle
is 0 (flat) for ever, so a fold springs open unless the folded half's weight holds it. With a yield
angle ``y`` (``rigid_cloth.joint_yield_deg``, DR ``domain_randomization.cloth.hinge_yield_deg``) the
rest angle follows the hinge whenever the hinge is further than ``y`` from it:

    rest <- clamp(rest, q - y, q + y)

so the spring never returns more than ``k * y``, and a hinge folded to 180 deg and let go rests at
``180 - y`` (less whatever gravity closes). The same rule as the VBD sheet's
``plastic_yield_curvature`` (``backends/vbd/plasticity.py``), on one hinge instead of every mesh edge.
``y = 0`` is elastic. The rest angle is the hinge drive's position target, updated once per policy
step and zeroed at reset (a folded start has then yielded like any other fold).
"""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.backends.rigid.pool import to_torch

__all__ = ["HingePlasticity"]


class HingePlasticity:
    """``yield_angle`` (N,) [rad] is each env's current yield angle (0 = elastic); ``rest[s]`` the
    rest angles of pool shape ``s``, one row per instance of its articulation."""

    def __init__(self, b) -> None:
        self.b = b
        r, rd = b.cfg.rigid_cloth, b.cfg.domain_randomization.cloth
        self.range = (float(rd.hinge_yield_deg[0]), float(rd.hinge_yield_deg[1]))
        self.varies = self.range[0] != self.range[1]
        self.enabled = self.varies or float(r.joint_yield_deg) > 0.0
        self.yield_angle = torch.full((b.num_envs,), math.radians(float(r.joint_yield_deg)), device=b.device)
        self.rest = [torch.zeros((a.num_instances, a.num_joints), device=b.device) for a in b.arts]
        #: Per shape, a (rows, J) yield angle [rad] for each hinge on its own, replacing ``yield_angle``
        #: for that shape (the physics-test viewer sets it); None: one angle per env.
        self.joint_yield: list[torch.Tensor | None] = [None for _ in b.arts]
        self._members = [b.pool.members(s) for s in range(len(b.arts))]
        if self.enabled:
            print(
                "[rigid_cloth] hinge plasticity: yield angle "
                + (f"{self.range[0]:g}-{self.range[1]:g} deg per episode" if self.varies
                   else f"{float(r.joint_yield_deg):g} deg"),
                flush=True,
            )

    def reset(self, env_ids: torch.Tensor) -> None:
        """A new episode in ``env_ids``: its yield angle, rest angles from flat."""
        if not self.enabled:
            return
        b = self.b
        ids = torch.as_tensor(env_ids, device=b.device, dtype=torch.long).reshape(-1)
        if self.varies:
            lo, hi = (math.radians(v) for v in self.range)
            self.yield_angle[ids] = torch.empty(ids.numel(), device=b.device).uniform_(lo, hi)
        for s in range(len(b.arts)):
            _keep, local = b.pool_rows(s, ids)
            if local.numel():
                self.rest[s][local] = 0.0
        self.step()

    def step(self) -> None:
        """Move every rest angle to within its yield angle of the hinge; write it as the target."""
        if not self.enabled:
            return
        for s, art in enumerate(self.b.arts):
            q = to_torch(art.data.joint_pos)
            y = self.joint_yield[s]
            if y is None:
                y = self.yield_angle[self._members[s]].unsqueeze(1)
            rest = torch.where(y > 0.0, torch.maximum(torch.minimum(self.rest[s], q + y), q - y), torch.zeros_like(q))
            self.rest[s] = rest
            art.set_joint_position_target(rest)
