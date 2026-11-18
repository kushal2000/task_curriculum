"""Physics-rate table containment and direct lifting-work episode checks.

ClothEnv observes completed substeps through its normal DirectRLEnv hooks. No
physics stepping, robot control, contact material or solver setting changes here.
"""
from __future__ import annotations

import torch

from ...mdp.fold_guard import LiftWorkHistory, robot_opening_work, table_excursion
from ...mdp.geometry import rotations
from .checks import contacts_needed
from .contacts import RigidClothContacts
from .designs import extrusion_mesh


class RigidClothEpisodeGuard:
    def __init__(self, backend):
        self.backend = backend
        self.cfg = backend.cfg.rigid_cloth.episode_checks
        self.n, self.device = backend.num_envs, backend.device
        if self.cfg.press_driven_fold and backend.cfg.task.mode != 'fold':
            raise ValueError('The lifting-work episode check is defined for fold tasks only')
        if self.cfg.table_margin < 0 or not 0 < self.cfg.minimum_lifting_work_fraction <= 1:
            raise ValueError('Invalid table margin or minimum lifting-work fraction')
        self.p = max(len(g.body_ids) for g in backend.groups)
        vertices = max(2*len(p.vertices) for g in backend.groups for p in g.design.plates)
        self.rest_vertices = torch.zeros(self.n, self.p, vertices, 3, device=self.device)
        self.base_mass = torch.zeros(self.n, self.p, device=self.device)
        self.half = torch.full((self.n, self.p), -1, dtype=torch.long, device=self.device)
        self.anchor = torch.zeros(self.n, 3, device=self.device)
        self.axis = torch.zeros_like(self.anchor)
        self.parent = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.opening_sign = torch.zeros(self.n, 2, device=self.device)
        for group in backend.groups:
            ids, dims = group.env_ids, backend._dims[group.env_ids]
            centers = torch.zeros(len(ids), len(group.body_ids), 3, device=self.device)
            self.base_mass[ids, :len(group.body_ids)] = group.mass_base[:, group.body_ids].to(self.device)
            names = [p.name for p in group.design.plates]
            for plate_id, plate in enumerate(group.design.plates):
                points, _ = extrusion_mesh(plate)
                points = torch.as_tensor(points, device=self.device, dtype=torch.float32)
                padded = points[0].expand(vertices, 3).clone()
                padded[:len(points)] = points
                self.rest_vertices[ids, plate_id] = padded[None]*dims[:, None]
                self.half[ids, plate_id] = int(plate.center[0] > 0)
                centers[:, plate_id] = dims*dims.new_tensor(plate.center)
            hinge = group.design.hinges[group.design.fold_hinge]
            self.parent[ids] = names.index(hinge.parent)
            self.anchor[ids] = dims*dims.new_tensor(hinge.anchor)
            axis = dims*dims.new_tensor(hinge.axis)
            self.axis[ids] = axis/axis.norm(dim=-1, keepdim=True)
            for h in (0, 1):
                weight = self.base_mass[ids, :len(names)]*(self.half[ids, :len(names)] == h)
                center = (centers*weight[..., None]).sum(1)/weight.sum(1, keepdim=True)
                self.opening_sign[ids, h] = torch.sign(torch.cross(self.axis[ids], center-self.anchor[ids], dim=-1)[:, 2])
        self.history = LiftWorkHistory(self.n, self.device, self.cfg)
        self.contacts = (RigidClothContacts(backend, capacity=max(16384, self.n*self.cfg.contact_points_per_env))
                         if contacts_needed(self.cfg) else None)
        # Centre-of-mass offsets in each body's own frame are constant (set once when the scene is
        # authored; nothing calls set_coms). Isaac Lab's body_com_pos_w re-reads them from PhysX as a
        # CPU tensor every physics step, which cost ~115 ms per policy step here; cache them once and
        # compose COM_world = link_pos + R(link_quat) * COM_body on the GPU instead (_com_w).
        self._com_b = {'robot': backend.robot.data.body_com_pos_b.clone()}
        for group in backend.groups:
            self._com_b[group.index] = group.articulation.data.body_com_pos_b[:, group.body_ids].clone()
        fraction = float(getattr(self.cfg, 'max_robot_penetration_fraction', 0.0))
        if fraction < 0:
            raise ValueError('max_robot_penetration_fraction must be >= 0')
        # Per-env depth limit [m]: a fraction of that env's panel thickness (dims[:, 2]).
        self.penetration_limit = fraction*backend._dims[:, 2] if fraction > 0 else None
        self.outside = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.pressing = torch.zeros_like(self.outside)
        self.penetrating = torch.zeros_like(self.outside)
        self.nonfinite = torch.zeros_like(self.outside)
        self.fresh = torch.ones_like(self.outside)
        self.excursion = torch.zeros(self.n, device=self.device)
        self.previous_robot = None
        self.last_counter = 0

    @property
    def success_allowed(self):
        if self.cfg.press_driven_fold:
            return self.history.initialized
        return torch.ones_like(self.outside)

    def reset(self, ids):
        self.history.reset(ids)
        for flag in (self.outside, self.pressing, getattr(self, 'penetrating', None), self.nonfinite):
            if flag is None:
                continue
            flag[ids] = False
        self.excursion[ids] = 0
        self.fresh[ids] = True

    def _plate_state(self):
        pos = torch.zeros(self.n, self.p, 3, device=self.device)
        quat = torch.zeros(self.n, self.p, 4, device=self.device)
        quat[..., 0] = 1
        com, velocity = torch.zeros_like(pos), torch.zeros(self.n, self.p, 6, device=self.device)
        for group in self.backend.groups:
            ids, bodies, data = group.env_ids, group.body_ids, group.articulation.data
            link_pos, link_quat = data.body_link_pos_w[:, bodies], data.body_link_quat_w[:, bodies]
            pos[ids, :len(bodies)] = link_pos
            quat[ids, :len(bodies)] = link_quat
            if self.contacts is not None:
                com[ids, :len(bodies)] = self._com_w(group.index, link_pos, link_quat, data, bodies)
                velocity[ids, :len(bodies)] = data.body_com_vel_w[:, bodies]
        return pos, quat, com, velocity

    def _com_w(self, key, link_pos, link_quat, data, bodies=slice(None)):
        """World COM positions from link poses and the cached body-frame COM offsets."""
        com_b = getattr(self, '_com_b', {}).get(key)
        if com_b is None:  # guards built without __init__ (tests) read Isaac Lab's own value
            return data.body_com_pos_w[:, bodies]
        return link_pos+rotations.apply(link_quat, com_b)

    def observe(self, completed_counter):
        """Latch failures once per COMPLETED physics step, including intermediate substeps."""
        if completed_counter <= self.last_counter:
            return
        self.last_counter = completed_counter
        b = self.backend
        pos, quat, com, velocity = self._plate_state()
        vertices = pos[:, :, None]+rotations.apply(quat[:, :, None], self.rest_vertices)
        vertices = torch.where((self.half >= 0)[:, :, None, None], vertices, vertices[:, :1, :1])
        table = b.table.data
        self.excursion = table_excursion(vertices, table.root_pos_w, table.root_quat_w,
                                         pos.new_tensor(b._table_size[:2])/2)
        finite = torch.isfinite(pos).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=(1, 2))
        self.nonfinite |= ~finite
        inside = (self.excursion <= self.cfg.table_margin) & finite
        if self.cfg.outside_table:
            self.outside |= ~inside & finite
        if self.contacts is None:
            return
        limit = getattr(self, 'penetration_limit', None)
        if limit is not None:
            self.penetrating |= self.contacts.robot_penetration(self.n) > limit
        if not self.cfg.press_driven_fold:
            return

        robot = b.robot.data
        link_pos, link_quat = robot.body_link_pos_w, robot.body_link_quat_w
        current = (link_pos, link_quat, self._com_w('robot', link_pos, link_quat, robot), robot.body_com_vel_w)
        if self.previous_robot is None:
            self.previous_robot = tuple(value.clone() for value in current)
        for previous, value in zip(self.previous_robot, current, strict=True):
            previous[self.fresh] = value[self.fresh]
        self.nonfinite |= ~torch.isfinite(velocity).all(dim=(1, 2))
        self.nonfinite |= ~torch.isfinite(current[3]).all(dim=(1, 2))
        self.nonfinite |= ~torch.isfinite(current[0]).all(dim=(1, 2))

        e, p, partner, point, force = self.contacts.raw()
        h = self.half[e, p]
        valid = torch.isfinite(force).all(-1) & torch.isfinite(point).all(-1)
        bad = torch.zeros(self.n, dtype=torch.long, device=self.device).index_add_(0, e, (~valid).long())
        self.nonfinite |= bad > 0
        force, point = force.nan_to_num(), point.nan_to_num()
        robot_contact = partner < self.contacts.num_robot_links
        source = self._robot_work(e, h, partner, point, force, robot_contact, current, pos, quat)
        robot_z = torch.zeros(self.n*2, device=self.device)
        table_up = torch.zeros_like(robot_z)
        robot_z.index_add_(0, e[robot_contact]*2+h[robot_contact], force[robot_contact, 2])
        table_up.index_add_(0, e[~robot_contact]*2+h[~robot_contact], force[~robot_contact, 2].clamp_min(0))
        table_up = table_up.view(self.n, 2)
        opposed = torch.minimum((-robot_z.view(self.n, 2)).clamp_min(0), table_up)
        self._update_history(com, quat, velocity, table_up, opposed, source, inside & ~self.nonfinite)
        for previous, value in zip(self.previous_robot, current, strict=True):
            previous.copy_(value)
        self.fresh.zero_()

    def _robot_work(self, env, half, partner, point, force, keep, robot, pos, quat):
        e, h, j = env[keep], half[keep], partner[keep]
        parent_q = quat[torch.arange(self.n, device=self.device), self.parent]
        anchor = pos[torch.arange(self.n, device=self.device), self.parent]+rotations.apply(parent_q, self.anchor)
        axis = rotations.apply(parent_q, self.axis)
        previous = self.previous_robot
        work = robot_opening_work(
            point[keep], force[keep], robot[0][e, j], robot[1][e, j], robot[2][e, j], robot[3][e, j],
            previous[0][e, j], previous[1][e, j], previous[2][e, j], previous[3][e, j],
            anchor[e], axis[e], self.opening_sign[e, h], self.backend.cfg.sim.dt,
        )
        return torch.zeros(self.n*2, device=self.device).index_add_(0, e*2+h, work.nan_to_num()).view(self.n, 2)

    def _update_history(self, com, quat, velocity, table_up, opposed, source, inside):
        b = self.backend
        mass = self.base_mass*(b._dr_density/float(b.cfg.rigid_cloth.density))[:, None]
        up = 1-2*(quat[..., 1].square()+quat[..., 2].square())
        speed, spin = velocity[..., :3].norm(dim=-1), velocity[..., 3:].norm(dim=-1)
        heights, weights, normals, minimum_up, maximum_speed, maximum_spin = [], [], [], [], [], []
        for h in (0, 1):
            selected = self.half == h
            m = mass*selected
            total = m.sum(1)
            heights.append((com[..., 2]*m).sum(1)/total)
            weights.append(total*9.81)
            normals.append((up*m).sum(1)/total)
            minimum_up.append(torch.where(selected, up, 1.).amin(1))
            maximum_speed.append(torch.where(selected, speed, 0.).amax(1))
            maximum_spin.append(torch.where(selected, spin, 0.).amax(1))
        stack = lambda values: torch.stack(values, dim=1)
        table_top = b.table.data.root_pos_w[:, 2]+b._table_half_height
        # Table colliders use zero rest offset. A positive cloth rest shell
        # intentionally supports its mid-plane above the visible tabletop.
        # Preserve actual COM heights for work/GPE, but include that gap when
        # deciding whether a half has settled flat on its support surface.
        support_surface = table_top+getattr(b, '_dr_rest_offset', b.cfg.rigid_cloth.rest_offset)
        new = self.history.update(
            source_work=source, height=stack(heights), weight=stack(weights), normal_up=stack(normals),
            minimum_plate_up=stack(minimum_up), maximum_linear_speed=stack(maximum_speed),
            maximum_angular_speed=stack(maximum_spin), table_up=table_up, opposed_load=opposed,
            inside=inside, table_top=support_surface, dimensions=b._dims,
        )
        self.pressing |= new

    def failures(self):
        out = {'outside_table': self.outside.clone(), 'press_driven_fold': self.pressing.clone()}
        if getattr(self, 'penetration_limit', None) is not None:  # only when enabled: unchanged keys otherwise
            out['robot_penetration'] = self.penetrating.clone()
        return out
