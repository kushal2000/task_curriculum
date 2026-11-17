"""Opt-in, experiment-only contact diagnostics for the current native VBD cloth.

Record metrics use the latest proxy contact normal/point and the current global
state. They measure overlap with that cached contact plane, NOT exact signed
penetration of the current hand mesh. An8mm particle radius is a collision envelope:
its overlap does not mean the cloth's zero-thickness center surface entered a solid.
Zero depths with zero records do not establish absence of contact/penetration.

The table metric independently queries the actual native collision box against
all cloth nodes. It misses triangle-interior crossings and can miss motion between
samples. No self-intersection claim is made by this module.
"""
from __future__ import annotations

import torch


def _rotate_xyzw(quat, vector):
    uv = 2 * torch.cross(quat[..., :3], vector, dim=-1)
    return vector + quat[..., 3:] * uv + torch.cross(quat[..., :3], uv, dim=-1)


def _compose_xyzw(parent, local):
    xyz = parent[..., :3] + _rotate_xyzw(parent[..., 3:7], local[..., :3])
    a, b = parent[..., 3:7], local[..., 3:7]
    v = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, v, w), dim=-1)


def box_signed_distance(points, poses, half_extents):
    """Exact oriented-box SDF for points(N,P,3), poses(N,7) in native xyz+xyzw."""
    quat = poses[:, None, 3:7].expand(*points.shape[:-1], 4).clone()
    quat[..., :3] *= -1
    local = _rotate_xyzw(quat, points - poses[:, None, :3])
    delta = local.abs() - half_extents[:, None]
    return delta.clamp_min(0).norm(dim=-1) + delta.amax(dim=-1).clamp_max(0)


def contact_depths(points, body_points, normals, radii, margins):
    """Return(envelope overlap, center/contact-plane crossing), both nonnegative metres."""
    distance = ((points-body_points)*normals).sum(-1)
    return (radii+margins-distance).clamp_min(0), (-distance).clamp_min(0)


def reduce_records(world, mask, overlap, crossing, num_envs):
    """Count and per-world maxima; zero for worlds with no selected contact records."""
    counts = overlap.new_zeros(num_envs)
    maxima = overlap.new_zeros(num_envs)
    crossings = overlap.new_zeros(num_envs)
    ids = world[mask]
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
    maxima.scatter_reduce_(0, ids, overlap[mask], reduce='amax', include_self=True)
    crossings.scatter_reduce_(0, ids, crossing[mask], reduce='amax', include_self=True)
    return counts, maxima, crossings


class VbdContactDiagnostics:
    """Cache static native index maps once; read current state/contacts before autoreset."""

    def __init__(self, env):
        import newton
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
        from isaacsimenvs.tasks.cloth.backends.vbd.contacts import _contact_buffer

        self.env, self.num_envs, self.device = env, env.num_envs, env.device
        self._tensor = wp.to_torch
        model = NewtonManager.get_model()
        _, entry = _contact_buffer()
        if entry is None:
            raise RuntimeError('No native cloth solver entry for contact diagnostics')
        self.entry = entry
        if entry.view.particle_count != model.particle_count:
            raise ValueError('Diagnostics require the cloth view to retain all global particle IDs')
        particle_map = entry.particle_local_to_global
        if particle_map is not None:
            actual = wp.to_torch(particle_map).long()
            if not torch.equal(actual, torch.arange(model.particle_count, device=actual.device)):
                raise ValueError('Diagnostics require identity cloth-view particle mapping')
        local_body = wp.to_torch(entry.view.shape_body).long()
        mapping = entry.body_local_to_global
        self.shape_body = local_body.clone()
        if mapping is not None:
            keep = local_body >= 0
            self.shape_body[keep] = wp.to_torch(mapping).long()[local_body[keep]]
        labels = list(model.body_label)
        bodies = self.shape_body.cpu().tolist()
        self.robot = torch.tensor([b >= 0 and '/Robot/' in labels[b] for b in bodies], device=self.device)
        self.table = torch.tensor([b >= 0 and '/Table/' in labels[b] for b in bodies], device=self.device)
        margin = entry.view.shape_margin
        self.margin = (wp.to_torch(margin).float() if margin is not None and len(margin)
                       else torch.zeros(len(bodies), device=self.device))
        self.world = wp.to_torch(model.particle_world).long()
        self.radius = wp.to_torch(model.particle_radius).float()
        self.body_world = wp.to_torch(model.body_world).long()
        shape_type = wp.to_torch(model.shape_type).long()
        table_ids = []
        for e in range(self.num_envs):
            ids = (self.table & (shape_type == int(newton.GeoType.BOX))
                   & (self.body_world[self.shape_body.clamp_min(0)] == e)).nonzero().flatten()
            if ids.numel() != 1:
                raise ValueError(f'Expected one native table collision box in env{e}; found{ids.numel()}')
            table_ids.append(int(ids[0]))
        self.table_ids = torch.tensor(table_ids, device=self.device)
        self.table_bodies = self.shape_body[self.table_ids]
        self.table_local_pose = wp.to_torch(model.shape_transform).view(-1,7)[self.table_ids].clone()
        self.table_half = wp.to_torch(model.shape_scale)[self.table_ids].clone()
        self.particle_rows = torch.stack([(self.world == e).nonzero().flatten() for e in range(self.num_envs)])

    def read(self):
        from isaaclab_newton.physics import NewtonManager
        from isaacsimenvs.tasks.cloth.backends.vbd.contacts import _contact_buffer

        state = NewtonManager.get_state_0()
        points = self._tensor(state.particle_q).view(-1,3)
        bodies = self._tensor(state.body_q).view(-1,7)
        result = {}
        for group in ('robot', 'table'):
            for metric in ('record_count', 'record_envelope_overlap_m', 'record_contact_plane_crossing_m'):
                result[f'{group}_{metric}'] = points.new_zeros(self.num_envs)
        buffer, _ = _contact_buffer()
        if buffer is not None and buffer.soft_contact_max:
            count = int(self._tensor(buffer.soft_contact_count)[0])
            if count > buffer.soft_contact_max:
                raise RuntimeError(f'Contact record count{count} exceeds capacity{buffer.soft_contact_max}')
            if count:
                shape = self._tensor(buffer.soft_contact_shape)[:count].long()
                corners = self._tensor(buffer.soft_contact_indices).view(-1,3)[:count].long()
                bary = self._tensor(buffer.soft_contact_barycentric).view(-1,3)[:count]
                normal = self._tensor(buffer.soft_contact_normal).view(-1,3)[:count]
                body_pos = self._tensor(buffer.soft_contact_body_pos).view(-1,3)[:count]
                valid = (shape >= 0) & (corners[:,0] >= 0)
                shape,corners,bary,normal,body_pos = [a[valid] for a in (shape,corners,bary,normal,body_pos)]
                if len(shape):
                    valid_corner = corners >= 0
                    index = corners.clamp_min(0)
                    point = (points[index]*bary[...,None]*valid_corner[...,None]).sum(1)
                    radius = torch.where(valid_corner,self.radius[index],0.).amax(1)
                    body = self.shape_body[shape]
                    body_pose = bodies[body.clamp_min(0)]
                    rigid_point = _rotate_xyzw(body_pose[:,3:7],body_pos)+body_pose[:,:3]
                    # Static world shapes have body=-1 and store body_pos directly in world space.
                    rigid_point = torch.where((body >= 0)[:,None],rigid_point,body_pos)
                    overlap,crossing = contact_depths(point,rigid_point,normal,radius,self.margin[shape])
                    world = self.world[index[:,0]]
                    for group, mask in (('robot',self.robot[shape]),('table',self.table[shape])):
                        if bool(((world[mask] < 0) | (world[mask] >= self.num_envs)).any()):
                            raise RuntimeError('Contact record references an unexpected cloth world')
                        values = reduce_records(world,mask,overlap,crossing,self.num_envs)
                        for metric,value in zip(('record_count','record_envelope_overlap_m','record_contact_plane_crossing_m'),values):
                            result[f'{group}_{metric}'] = value
        table_pose = _compose_xyzw(bodies[self.table_bodies],self.table_local_pose)
        distance = box_signed_distance(points[self.particle_rows],table_pose,self.table_half)
        result['table_box_center_depth_m'] = (-distance).clamp_min(0).amax(1)
        result['table_box_inside_node_count'] = (distance < 0).to(points.dtype).sum(1)
        if not all(bool(torch.isfinite(v).all()) for v in result.values()):
            raise RuntimeError('Nonfinite native VBD contact diagnostic')
        return result
