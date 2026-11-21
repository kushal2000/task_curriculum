"""Measure robot clearance from the VBD proxy contact records."""
import torch


class ReleaseContact:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self._touch_cache = None

    def _cloth_contact_buffer(self):
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

    def hand_cloth_touching(self) -> torch.Tensor:
        """``(num_envs,)`` bool: any robot shape is in contact with the env's cloth this step
        (``cloth.release_touch=contact``). Reads the cloth solver's own contact records and evaluates
        each one's gap as VBD's body-particle / edge / face contact does (``-penetration_depth``:
        ``n . (x - bx) - radius - shape_margin``, ``x`` the barycentric point over the record's
        particles, ``bx`` the body-side point at the current body pose); touching = gap <
        ``release_touch_gap``. A non-finite gap reads as touching."""
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        buf, entry = self._cloth_contact_buffer()
        if buf is None or buf.soft_contact_max == 0:
            raise RuntimeError("Newton cloth contact records are unavailable")
        model, state = NewtonManager._model, NewtonManager._state_0
        if self._touch_cache is None:
            import re

            view, dev = entry.view, self.device
            labels = list(model.body_label)
            sb_local = view.shape_body.numpy()  # body ids local to the cloth view
            l2g = entry.body_local_to_global.numpy() if entry.body_local_to_global is not None else None
            sb = [int(l2g[b]) if (l2g is not None and b >= 0) else int(b) for b in sb_local]
            robot = [b >= 0 and re.search(r"/Robot/", str(labels[b])) is not None for b in sb]
            if not any(robot):
                raise RuntimeError("cloth.release_touch=contact: no robot shape found in the cloth view")
            if int(view.particle_count) != int(model.particle_count):
                raise RuntimeError("cloth.release_touch=contact assumes the cloth view keeps the model's particle ids")
            margin = view.shape_margin
            margin = wp.to_torch(margin).float() if margin is not None and margin.shape[0] > 0 \
                else torch.zeros(len(sb), device=dev)
            self._touch_cache = {
                "robot": torch.tensor(robot, dtype=torch.bool, device=dev),
                "shape_body": torch.tensor(sb, dtype=torch.long, device=dev),  # GLOBAL body ids
                "margin": margin,
                "world": wp.to_torch(model.particle_world).long(),
                "radius": wp.to_torch(model.particle_radius).float(),
            }
            print(f"[cloth] release_touch=contact: {sum(robot)} robot shapes in the cloth view, gap < "
                  f"2.0 mm counts as touching", flush=True)
        tc = self._touch_cache
        m = int(buf.soft_contact_max)
        count = wp.to_torch(buf.soft_contact_count).long()  # (1,), stays on device
        shape = wp.to_torch(buf.soft_contact_shape).long()
        if int(count.item()) > m:
            raise RuntimeError("Newton cloth contact buffer overflowed")
        sel = (torch.arange(m, device=self.device) < count) & (shape >= 0)
        sel &= tc["robot"][torch.where(sel, shape, 0)]
        ids = sel.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return out
        shape = shape[ids]
        corners = wp.to_torch(buf.soft_contact_indices).view(m, 3)[ids].long()
        bary = wp.to_torch(buf.soft_contact_barycentric).view(m, 3)[ids].float()
        normal = wp.to_torch(buf.soft_contact_normal).view(m, 3)[ids].float()
        body_pos = wp.to_torch(buf.soft_contact_body_pos).view(m, 3)[ids].float()
        pq = wp.to_torch(state.particle_q).view(-1, 3).float()
        ok = corners >= 0
        cc = corners.clamp(min=0)
        x = (bary.unsqueeze(-1) * pq[cc] * ok.unsqueeze(-1)).sum(1)
        radius = torch.where(ok, tc["radius"][cc], torch.zeros_like(bary)).amax(1)
        body = tc["shape_body"][shape]
        bq = wp.to_torch(state.body_q).view(-1, 7).float()[body.clamp(min=0)]
        p, qv, qw = bq[:, 0:3], bq[:, 3:6], bq[:, 6:7]  # warp transform: p, then quat xyzw
        t = 2.0 * torch.cross(qv, body_pos, dim=-1)
        bx = body_pos + qw * t + torch.cross(qv, t, dim=-1) + p
        gap = (normal * (x - bx)).sum(-1) - radius - tc["margin"][shape]
        touch = ~(gap >= 0.002)  # NaN -> touching
        env = tc["world"][cc[:, 0]]
        env = env[touch & (env >= 0) & (env < self.num_envs)]
        out[env] = True
        return out
