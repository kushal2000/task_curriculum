"""Read hand-cloth contact gaps from Newton's cloth solver view.

Particle, edge, and face contacts use global particle/shape IDs and view-local body IDs.
The public environment query delegates here; gaps and environment IDs remain available
on the environment for contact diagnostics.
"""

from __future__ import annotations

import torch


def _contact_buffer():
    """``(contacts, entry)``: the hand-cloth contacts the cloth (VBD) entry steps on, and that
    entry. Under ``SolverCoupledProxy`` they come from the proxy pipeline built on the cloth view
    (``get_proxy_contacts(<rigid entry>, "cloth")``; particle / edge / face records, shape and
    particle ids global, body ids LOCAL to the cloth view). ``(None, entry)`` before the first
    step or without a proxy pipeline."""
    from isaaclab_newton.physics import NewtonManager

    solver = NewtonManager._solver
    cloth = solver.solver("cloth")
    name = next((k for k, e in getattr(solver, "_entries", {}).items() if e.solver is cloth), None)
    if name is None:
        return None, None
    entry = solver._entries[name]
    for src, dst in getattr(solver, "_proxy_collision_configs", {}):
        if dst == name:
            return solver.get_proxy_contacts(src, dst), entry
    return getattr(solver, "_entry_contact_buffers", {}).get(name), entry


def hand_cloth_touching(env) -> torch.Tensor:
    """``(num_envs,)`` bool: any robot shape is in contact with the env's cloth this step
    (``task.release.touch=contact``). Reads the cloth solver's own contact records and evaluates
    each one's gap as VBD's body-particle / edge / face contact does (``-penetration_depth``:
    ``n . (x - bx) - radius - shape_margin``, ``x`` the barycentric point over the record's
    particles, ``bx`` the body-side point at the current body pose); touching = gap <
    ``release_touch_gap``. A non-finite gap reads as touching."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    buffer, entry = _contact_buffer()
    if buffer is None or buffer.soft_contact_max == 0:
        return out
    model, state = NewtonManager._model, NewtonManager._state_0
    if env._touch_cache is None:
        import re

        view, device = entry.view, env.device
        labels = list(model.body_label)
        local_shape_bodies = view.shape_body.numpy()  # body ids local to the cloth view
        body_local_to_global = entry.body_local_to_global.numpy() if entry.body_local_to_global is not None else None
        shape_bodies = [
            int(body_local_to_global[b]) if (body_local_to_global is not None and b >= 0) else int(b)
            for b in local_shape_bodies
        ]
        robot = [b >= 0 and re.search(r"/Robot/", str(labels[b])) is not None for b in shape_bodies]
        if not any(robot):
            raise RuntimeError("task.release.touch=contact: no robot shape found in the cloth view")
        if int(view.particle_count) != int(model.particle_count):
            raise RuntimeError("task.release.touch=contact assumes the cloth view keeps the model's particle ids")
        margin = view.shape_margin
        if margin is not None and margin.shape[0] > 0:
            margin = wp.to_torch(margin).float()
        else:
            margin = torch.zeros(len(shape_bodies), device=device)
        env._touch_cache = {
            "robot": torch.tensor(robot, dtype=torch.bool, device=device),
            "shape_body": torch.tensor(shape_bodies, dtype=torch.long, device=device),  # GLOBAL body ids
            "margin": margin,
            "world": wp.to_torch(model.particle_world).long(),
            "radius": wp.to_torch(model.particle_radius).float(),
        }
        print(f"[cloth] release_touch=contact: {sum(robot)} robot shapes in the cloth view, gap < "
              f"{float(env.cfg.task.release.touch_gap) * 1e3:.1f} mm counts as touching", flush=True)
    cache = env._touch_cache
    capacity = int(buffer.soft_contact_max)
    count = wp.to_torch(buffer.soft_contact_count).long()  # (1,), stays on device
    shape = wp.to_torch(buffer.soft_contact_shape).long()
    selected = (torch.arange(capacity, device=env.device) < count) & (shape >= 0)
    selected &= cache["robot"][shape.clamp(min=0)]
    ids = selected.nonzero(as_tuple=True)[0]
    if ids.numel() == 0:
        return out
    shape = shape[ids]
    corners = wp.to_torch(buffer.soft_contact_indices).view(capacity, 3)[ids].long()
    bary = wp.to_torch(buffer.soft_contact_barycentric).view(capacity, 3)[ids].float()
    normal = wp.to_torch(buffer.soft_contact_normal).view(capacity, 3)[ids].float()
    body_pos = wp.to_torch(buffer.soft_contact_body_pos).view(capacity, 3)[ids].float()
    particle_positions = wp.to_torch(state.particle_q).view(-1, 3).float()
    valid_corners = corners >= 0
    particle_ids = corners.clamp(min=0)
    contact_point = (bary.unsqueeze(-1) * particle_positions[particle_ids] * valid_corners.unsqueeze(-1)).sum(1)
    radius = torch.where(valid_corners, cache["radius"][particle_ids], torch.zeros_like(bary)).amax(1)
    body = cache["shape_body"][shape]
    body_pose = wp.to_torch(state.body_q).view(-1, 7).float()[body.clamp(min=0)]
    # Warp transforms store position followed by an xyzw quaternion.
    body_position = body_pose[:, 0:3]
    quat_vector, quat_scalar = body_pose[:, 3:6], body_pose[:, 6:7]
    cross_term = 2.0 * torch.cross(quat_vector, body_pos, dim=-1)
    body_point = body_pos + quat_scalar * cross_term + torch.cross(quat_vector, cross_term, dim=-1) + body_position
    gap = (normal * (contact_point - body_point)).sum(-1) - radius - cache["margin"][shape]
    env._touch_last_gap = gap  # diagnostics (cloth_touch_probe)
    env._touch_last_env = cache["world"][particle_ids[:, 0]]
    touch = ~(gap >= float(env.cfg.task.release.touch_gap))  # NaN -> touching
    contact_envs = env._touch_last_env
    contact_envs = contact_envs[touch & (contact_envs >= 0) & (contact_envs < env.num_envs)]
    out[contact_envs] = True
    return out
