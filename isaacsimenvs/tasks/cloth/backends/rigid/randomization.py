"""Per-env physics randomisation of the plates, redrawn at every reset (``rigid_cloth.randomization``):
contact margin, hinge damping and stiffness, mass. Friction (``cloth_min``) is drawn once at startup,
in ``contacts.py``."""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.backends.rigid import contacts

__all__ = ["PlateRandomization"]


class PlateRandomization:
    """``sample`` holds each env's current draw (read by the viewer and the logs). Construction
    draws every env once when ``enabled``."""

    def __init__(self, b) -> None:
        self.b = b
        n = b.num_envs
        self.mass_base = [a.root_physx_view.get_masses().clone() for a in b.arts]
        self.inertia_base = [a.root_physx_view.get_inertias().clone() for a in b.arts]
        self.contact_margin = contacts.contact_margin(b)
        self.sample = {
            k: torch.ones(n, device=b.device) for k in ("slat_friction", "fingertip_friction", "hinge_damping", "mass")
        }
        if b.cfg.physx_chain.friction_model == "cloth_min":
            # Drawn once at startup (contacts.py); absolute values here, not factors.
            self.sample["slat_friction"] = b.friction_slat.to(b.device)
            self.sample["fingertip_friction"] = b.friction_robot.to(b.device)
            self.sample["table_friction"] = b.friction_table.to(b.device)
        self.sample["hinge_stiffness"] = torch.zeros(n, device=b.device)
        self.sample["contact_margin"] = torch.full((n,), self.contact_margin, device=b.device)
        self.enabled = bool(b.cfg.rigid_cloth.randomization.enabled)
        if self.enabled:
            self.draw(torch.arange(n, device=b.device))
            rd = b.cfg.rigid_cloth.randomization
            print(
                f"[rigid_cloth] physics randomisation ON, per env at every reset: hinge damping x"
                f"{tuple(rd.hinge_damping_scale)} (log), hinge stiffness {tuple(rd.hinge_stiffness)} "
                f"N m/rad, contact margin {self.contact_margin * 1e3:.1f}"
                f"{rd.contact_margin_offset[0] * 1e3:+.1f}..{rd.contact_margin_offset[1] * 1e3:+.1f} mm, "
                f"mass x{tuple(rd.mass_scale)} (friction: "
                f"{'per env at startup, see the cloth_min line' if b.cfg.physx_chain.friction_model == 'cloth_min' else 'fixed'})",
                flush=True,
            )

    def draw(self, env_ids) -> None:
        """Re-draw the plates' physics for ``env_ids``: only the ranges that are not a point."""
        b = self.b
        rd = b.cfg.rigid_cloth.randomization
        r = b.cfg.rigid_cloth
        ids = torch.as_tensor(env_ids, device=b.device, dtype=torch.long).reshape(-1)
        n = int(ids.numel())
        if n == 0:
            return

        def uni(lo_hi) -> torch.Tensor:
            return torch.empty(n, device=b.device).uniform_(float(lo_hi[0]), float(lo_hi[1]))

        def varies(lo_hi) -> bool:
            return float(lo_hi[0]) != float(lo_hi[1])

        arts = b.arts
        # Per articulation: which of `ids` it exists in, and its own instance rows for them.
        rows = []
        for k in range(len(arts)):
            keep, local = b.pool_rows(k, ids)
            rows.append((keep, local, local.to("cpu", torch.int32)))

        def sel(v: torch.Tensor, keep) -> torch.Tensor:
            return v if keep is None else v[keep.to(v.device)]

        if varies(rd.contact_margin_offset) and self.contact_margin > 0.0:
            margin = (self.contact_margin + uni(rd.contact_margin_offset)).clamp(min=0.0)
            m_cpu = margin.cpu().unsqueeze(1)
            for art, (keep, local, l_cpu) in zip(arts, rows):
                if local.numel() == 0:
                    continue
                view = art.root_physx_view
                rest = view.get_rest_offsets()
                rest[l_cpu.long()] = sel(m_cpu, keep).expand(-1, rest.shape[1])
                # The contact offset is fixed above the whole range (`contacts.contact_offset`).
                view.set_rest_offsets(rest, l_cpu)
            self.sample["contact_margin"][ids] = margin
        if varies(rd.hinge_damping_scale):
            lo, hi = math.log(float(rd.hinge_damping_scale[0])), math.log(float(rd.hinge_damping_scale[1]))
            damp = float(r.joint_damping) * torch.exp(torch.empty(n, device=b.device).uniform_(lo, hi))
            for art, (keep, local, _l) in zip(arts, rows):
                if local.numel():
                    d = sel(damp, keep)
                    art.write_joint_damping_to_sim(d.unsqueeze(1).expand(-1, art.num_joints), env_ids=local)
            self.sample["hinge_damping"][ids] = damp
        if varies(rd.hinge_stiffness):
            stiff = uni(rd.hinge_stiffness)
            for art, (keep, local, _l) in zip(arts, rows):
                if local.numel():
                    k_ = sel(stiff, keep)
                    art.write_joint_stiffness_to_sim(k_.unsqueeze(1).expand(-1, art.num_joints), env_ids=local)
            self.sample["hinge_stiffness"][ids] = stiff
        if varies(rd.mass_scale):
            scale = uni(rd.mass_scale)
            s_cpu = scale.cpu()
            for art, m0, i0, (keep, local, l_cpu) in zip(arts, self.mass_base, self.inertia_base, rows):
                if local.numel() == 0:
                    continue
                view = art.root_physx_view
                masses = view.get_masses()
                inertias = view.get_inertias()
                sc = sel(s_cpu, keep)
                masses[l_cpu.long()] = m0[l_cpu.long()] * sc.unsqueeze(1)
                inertias[l_cpu.long()] = i0[l_cpu.long()] * sc.view(-1, 1, 1)
                view.set_masses(masses, l_cpu)
                view.set_inertias(inertias, l_cpu)
            self.sample["mass"][ids] = scale
