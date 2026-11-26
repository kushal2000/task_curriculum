"""How hard does the VBD cloth push on the thumb? (diagnostic, 2026-11-24)

Reads the cloth -> robot feedback wrench the coupled MJWarp + VBD solver hands to MuJoCo
(``SolverCoupledProxy._proxy_mappings[*].coupling_forces``: per proxy body, force and torque about
the body's COM, refreshed every coupled substep) and projects it onto each thumb joint's axis:

    tau_j = a_j . sum_{b in subtree(j)} [ (c_b - p_j) x f_b + t_b ]

with ``p_j`` / ``a_j`` the joint anchor and axis in world (from the child body's pose and
``joint_X_c``, so no joint coordinate is needed) and ``c_b`` each downstream body's COM. That is the
torque the cloth applies about the joint, directly comparable with the drive's torque limit.

The sample is the LAST coupled substep of a policy step, so it is a snapshot, not a step maximum.
"""

from __future__ import annotations

import numpy as np

G = 9.81


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (..., 3) by unit quaternions q (..., 4) in xyzw order."""
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


class ThumbForceRecorder:
    """Per policy step: cloth torque about every watched thumb joint and cloth force on every thumb link."""

    def __init__(self, inner, joint_names: list[str], link_pattern: str = "thumb"):
        mgr = inner.sim.physics_manager
        self._mgr = mgr
        model = mgr.get_model()
        solver = getattr(mgr, "_solver", None)
        maps = [m for m in getattr(solver, "_proxy_mappings", []) if getattr(m, "coupling_forces", None) is not None]
        maps = [m for m in maps if m.src_name == "rigid"] or maps
        if not maps:
            raise RuntimeError("no body proxy mapping with coupling_forces on the solver (needs the coupled VBD solver)")
        self._map = maps[0]
        # coupling_forces is indexed by proxy id; the map is keyed by the SOURCE entry's local body index,
        # so go model body -> source-local (entry.body_global_to_local) -> proxy id. -1 = not proxied.
        nb = int(model.body_count)
        src_local_to_proxy = self._map.source_local_to_proxy_global.numpy()
        g2l = solver._entries[self._map.src_name].body_global_to_local.numpy()
        if g2l.shape[0] != nb:
            raise RuntimeError(f"entry {self._map.src_name!r} global->local map has {g2l.shape[0]} bodies, model has {nb}")
        self._s2g = np.full(nb, -1, dtype=np.int64)
        has = (g2l >= 0) & (g2l < src_local_to_proxy.shape[0])
        self._s2g[has] = src_local_to_proxy[g2l[has]]

        jl = [str(x) for x in model.joint_label]
        jp, jc = model.joint_parent.numpy(), model.joint_child.numpy()
        qd0 = model.joint_qd_start.numpy()
        axis_all = model.joint_axis.numpy()
        xc = model.joint_X_c.numpy()
        body_world = model.body_world.numpy()
        bl = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
        self._body_com = model.body_com.numpy()

        children: dict[int, list[int]] = {}
        for j in range(len(jl)):
            if jp[j] >= 0:
                children.setdefault(int(jp[j]), []).append(int(jc[j]))

        def subtree(b: int) -> list[int]:
            out, stack = [], [b]
            while stack:
                x = stack.pop()
                out.append(x)
                stack.extend(children.get(x, []))
            return out

        want = set(joint_names)
        self.joint_names = list(joint_names)
        self.num_envs = int(inner.num_envs)
        rows = []   # (env, joint-name index, child body, X_c, axis)
        pairs_j, pairs_b = [], []
        for j, label in enumerate(jl):
            name = label.rsplit("/", 1)[-1]
            if name not in want or qd0[j + 1] - qd0[j] != 1:
                continue
            env = int(body_world[jc[j]])
            if env < 0:
                continue
            r = len(rows)
            rows.append((env, self.joint_names.index(name), int(jc[j]), xc[j], axis_all[qd0[j]]))
            for b in subtree(int(jc[j])):
                pairs_j.append(r); pairs_b.append(b)
        if not rows:
            raise RuntimeError(f"no 1-DoF Newton joint named any of {joint_names}")
        self._row_env = np.array([r[0] for r in rows]); self._row_name = np.array([r[1] for r in rows])
        self._row_child = np.array([r[2] for r in rows])
        self._row_xc = np.stack([r[3] for r in rows]).astype(np.float64)
        self._row_axis = np.stack([r[4] for r in rows]).astype(np.float64)
        self._pair_j, self._pair_b = np.array(pairs_j), np.array(pairs_b)

        # Thumb links (force magnitude per link name).
        self.link_names = sorted({bl[b] for b in range(nb) if link_pattern in bl[b] and body_world[b] >= 0})
        self._link_b = np.array([b for b in range(nb) if bl[b] in self.link_names and body_world[b] >= 0])
        self._link_env = body_world[self._link_b]
        self._link_name = np.array([self.link_names.index(bl[b]) for b in self._link_b])

        # Contact records per thumb link (newton_support/contact_guards.body_contact_stats), from the cloth
        # entry's VBD solver. Its ModelView numbers bodies LOCALLY (shapes keep the model's numbering), so the
        # thumb links go through the cloth entry's global -> local map; -1 = not in the cloth entry.
        from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import contact_guards

        self._guards = contact_guards
        self._vbd = solver.solver("cloth") if hasattr(solver, "solver") else None
        if self._vbd is not None:
            cg2l = solver._entries["cloth"].body_global_to_local.numpy()
            if cg2l.shape[0] != nb:
                raise RuntimeError(f"cloth entry global->local map has {cg2l.shape[0]} bodies, model has {nb}")
            self._link_local = cg2l[self._link_b].astype(np.int64)
            if (self._link_local < 0).any():
                raise RuntimeError("a thumb link is not in the cloth entry (not proxied into VBD)")
            contact_guards.enable_body_contact_stats(True)

        # Thumb capsules (pierce.py's per-link collision axis + median radius) for location labels measured from
        # the SURFACE that touches the cloth, not the link origins (which sit at the joints).
        import re

        import torch

        from isaacsimenvs.tasks.cloth.backends.vbd.pierce import finger_segments

        caps = finger_segments(model, re.compile(r"/Robot/[^/]*" + link_pattern + r"[^/]*$"))
        cap_names = sorted({c[2] for c in caps})
        cb = np.full((self.num_envs, len(cap_names)), -1, dtype=np.int64)
        cseg = np.zeros((len(cap_names), 2, 3)); crad = np.zeros(len(cap_names))
        for b, w, name, sgm, r in caps:
            if 0 <= w < self.num_envs:
                k = cap_names.index(name); cb[w, k] = b; cseg[k] = sgm; crad[k] = r
        if (cb < 0).any() or not cap_names:
            raise RuntimeError(f"thumb capsules missing in some env: {cap_names}")
        dev = inner.device
        self._cap_b = torch.as_tensor(cb, device=dev)
        self._cap_seg = torch.as_tensor(cseg, device=dev, dtype=torch.float32)
        self._cap_r = torch.as_tensor(crad, device=dev, dtype=torch.float32)
        self.particle_radius = float(np.median(model.particle_radius.numpy()))
        print(f"[thumb_force] thumb capsules {cap_names}, radii {np.round(crad * 1e3, 1).tolist()} mm; cloth particle "
              f"radius {self.particle_radius * 1e3:.1f} mm", flush=True)

        # The cloth's weight per env [N], the scale a real sheet's push should be compared with.
        pm = model.particle_mass.numpy(); pw = model.particle_world.numpy()
        self.cloth_weight_n = float(np.mean([pm[pw == e].sum() * G for e in range(self.num_envs)]))
        print(f"[thumb_force] {len(rows)} joints ({len(self.joint_names)} per env), {len(self.link_names)} thumb links, "
              f"{len(pairs_j)} joint-body pairs; cloth weight {self.cloth_weight_n * 1000:.1f} mN per env", flush=True)

    def sample_records(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(E, L) per thumb link: cloth contact records at the last coupled substep (the force snapshot's),
        the most records in any substep of the step, and the deepest record's penetration into the contact
        distance [mm] over the step (< 0: listed but not touching; nan: no record)."""
        if self._vbd is None:
            return None
        st = self._guards.body_contact_stats(self._vbd, reset=True)
        if st is None:
            return None
        n_last, n_max, depth = st
        shape = (self.num_envs, len(self.link_names))
        a, b, c = np.zeros(shape), np.zeros(shape), np.full(shape, np.nan)
        lb = self._link_local
        a[self._link_env, self._link_name] = n_last[lb]
        b[self._link_env, self._link_name] = n_max[lb]
        d = depth[lb] * 1e3
        c[self._link_env, self._link_name] = np.where(depth[lb] > -0.5, d, np.nan)
        return a, b, c

    def sample_location(self, points_w, table_top_z):
        """Per env, from the thumb capsules' surfaces: (gap to the nearest cloth particle centre [m],
        that touching particle set's lowest centre above the table top [m] (inf if none within 12 mm),
        lowest capsule surface point above the table top [m]). ``points_w`` (E, P, 3), ``table_top_z`` (E,)."""
        import torch
        import warp as wp

        from isaacsimenvs.tasks.cloth.backends.vbd.pierce import quat_rotate_xyzw

        bq = wp.to_torch(self._mgr._state_0.body_q)[self._cap_b]          # (E, K, 7)
        pos, quat = bq[..., :3], bq[..., 3:]
        E, K = self._cap_b.shape
        a = pos + quat_rotate_xyzw(quat, self._cap_seg[None, :, 0].expand(E, -1, -1))
        b = pos + quat_rotate_xyzw(quat, self._cap_seg[None, :, 1].expand(E, -1, -1))
        ab = b - a                                                          # (E, K, 3)
        p = points_w[:, None]                                               # (E, 1, P, 3)
        t = (((p - a[:, :, None]) * ab[:, :, None]).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)[..., None]).clamp(0, 1)
        closest = a[:, :, None] + t[..., None] * ab[:, :, None]
        gap = (p - closest).norm(dim=-1) - self._cap_r[None, :, None]        # (E, K, P) capsule surface to centre
        gmin = gap.amin(dim=(1, 2))
        touching = (gap < self.particle_radius + 0.004).any(dim=1)           # (E, P) particles at the thumb
        ph = points_w[..., 2] - table_top_z[:, None]                         # (E, P) particle centre heights
        sup = torch.where(touching, ph, torch.full_like(ph, float("inf"))).amin(dim=1)
        low = (torch.minimum(a[..., 2], b[..., 2]) - self._cap_r[None]).amin(dim=1) - table_top_z
        return gmin, sup, low

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """(E, J) cloth torque about each watched joint [N m], (E, L) cloth force on each thumb link [N]."""
        F = self._map.coupling_forces.numpy().astype(np.float64)          # (P, 6): force, torque about COM
        bq = self._mgr._state_0.body_q.numpy().astype(np.float64)          # (B, 7): p, q xyzw
        g = self._s2g
        valid = g >= 0
        f_b = np.zeros((bq.shape[0], 3)); t_b = np.zeros((bq.shape[0], 3))
        f_b[valid] = F[g[valid], :3]; t_b[valid] = F[g[valid], 3:]
        pos, rot = bq[:, :3], bq[:, 3:]
        com = pos + _quat_rotate(rot, self._body_com.astype(np.float64))
        cq = rot[self._row_child]
        anchor = pos[self._row_child] + _quat_rotate(cq, self._row_xc[:, :3])
        axis = _quat_rotate(_quat_mul(cq, self._row_xc[:, 3:]), self._row_axis)
        b = self._pair_b; j = self._pair_j
        moment = np.cross(com[b] - anchor[j], f_b[b]) + t_b[b]                # (pairs, 3)
        tau_rows = np.zeros(len(self._row_env))
        np.add.at(tau_rows, j, np.einsum("ij,ij->i", moment, axis[j]))
        tau = np.full((self.num_envs, len(self.joint_names)), np.nan)
        tau[self._row_env, self._row_name] = tau_rows
        links = np.full((self.num_envs, len(self.link_names)), np.nan)
        links[self._link_env, self._link_name] = np.linalg.norm(f_b[self._link_b], axis=1)
        return tau, links
