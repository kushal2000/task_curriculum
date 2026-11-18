"""Contact physics of the plates: friction materials (bound before the simulator starts), and the
read-back that refuses to run on a mismatch. See ``physx.py`` for why friction is USD-authored."""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.backends.rigid.physx import (
    DEFAULT_MU,
    bind_material_in_every_env,
    bind_materials,
    collider_paths,
    define_scene_material,
)

__all__ = ["ContactsMixin"]


class ContactsMixin:
    """Friction and contact offsets of the plates, robot and table."""

    def _bind_contact_materials(self) -> None:
        """Two scene-level materials -- slat and fingertip -- bound to those colliders in every env.

        ONE material each, not one per env: PhysX caps a scene at 64K materials, and a material
        authored inside the asset becomes a separate material in every env that references it. At
        12288 envs the pool's five shapes plus the fingertips crossed the cap and the slats silently
        kept the default friction (caught by `_check_contacts`'s read-back).
        """
        from isaaclab.sim.utils import get_current_stage

        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        stage = get_current_stage()
        if pc.friction_model == "cloth_min":
            self._bind_cloth_min_materials(stage)
            return
        slat = "/World/PhysicsMaterials/slat"
        tip = "/World/PhysicsMaterials/fingertip"
        define_scene_material(
            stage, slat, mu=r.friction, combine=pc.slat_friction_combine,
            compliant_stiffness=pc.slat_compliant_stiffness, compliant_damping=pc.slat_compliant_damping,
            compliant_acceleration=pc.slat_compliant_acceleration,
        )
        define_scene_material(stage, tip, mu=r.fingertip_friction, combine=pc.fingertip_friction_combine)
        n = self.num_envs
        slats = 0
        for k, art in enumerate(self._manipuland_arts()):
            asset = art.cfg.prim_path.rsplit("/", 1)[-1]
            envs = self._manipuland_envs(k)
            rels = collider_paths(stage, f"/World/envs/env_{envs[0]}/{asset}")
            if not rels:
                raise RuntimeError(f"[physx] no collider under env_{envs[0]}/{asset}")
            slats += bind_material_in_every_env(stage, n, asset, rels, slat, env_ids=envs)
        tips = tuple(self.robot_spec.fingertip_link_names)
        robot = self.robot.cfg.prim_path.rsplit("/", 1)[-1]
        rels = collider_paths(
            stage, f"/World/envs/env_0/{robot}",
            select=lambda prim: any(f"/{t}/" in str(prim.GetPath()) + "/" for t in tips),
        )
        if not rels:
            raise RuntimeError(f"[physx] no fingertip collider ({tips}) under env_0/{robot}")
        bind_material_in_every_env(stage, n, robot, rels, tip)
        self._fingertip_colliders = len(rels)
        print(
            f"[physx] bound {slats} slat colliders to {slat} and {n * len(rels)} fingertip colliders "
            f"to {tip} ({len(rels)} per env); slat contacts "
            + (
                f"COMPLIANT, stiffness {pc.slat_compliant_stiffness:g} damping {pc.slat_compliant_damping:g} "
                f"({'per unit mass' if pc.slat_compliant_acceleration else 'force'})"
                if pc.slat_compliant_stiffness > 0.0 else "rigid"
            ),
            flush=True,
        )

    def _draw_friction_buckets(self, scale, gen) -> tuple[torch.Tensor, torch.Tensor]:
        """``(bucket multipliers (B,), env -> bucket (N,))`` for one friction, cloth_min model."""
        rd, n = self.cfg.rigid_cloth.randomization, self.num_envs
        lo, hi = float(scale[0]), float(scale[1])
        if not rd.enabled or lo == hi:
            return torch.tensor([lo if rd.enabled else 1.0]), torch.zeros(n, dtype=torch.long)
        b = max(1, int(self.cfg.physx_chain.friction_buckets))
        return torch.linspace(lo, hi, b), torch.randint(0, b, (n,), generator=gen)

    def _bind_cloth_min_materials(self, stage) -> None:
        """``physx_chain.friction_model: cloth_min``: one ``min`` slat material per friction bucket,
        bound to the slats of the envs drawn into it. The robot's and the table's (average-mode)
        frictions are tensor-written once the simulator runs (``_write_cloth_min_frictions``)."""
        r, pc, a = self.cfg.rigid_cloth, self.cfg.physx_chain, self.cfg.assets
        rd = r.randomization
        gen = torch.Generator().manual_seed(int(pc.friction_seed))
        s_mult, s_idx = self._draw_friction_buckets(rd.slat_friction_scale, gen)
        f_mult, f_idx = self._draw_friction_buckets(rd.fingertip_friction_scale, gen)
        t_mult, t_idx = self._draw_friction_buckets(rd.table_friction_scale, gen)
        self._fric_slat = float(r.friction) * s_mult[s_idx]                # (N,) cpu
        self._fric_robot = float(a.finger_tip_friction) * f_mult[f_idx]
        self._fric_table = float(a.table_friction) * t_mult[t_idx]
        paths = []
        for b, m in enumerate(s_mult.tolist()):
            path = f"/World/PhysicsMaterials/slat_{b}"
            define_scene_material(
                stage, path, mu=float(r.friction) * m, combine=pc.slat_friction_combine,
                compliant_stiffness=pc.slat_compliant_stiffness, compliant_damping=pc.slat_compliant_damping,
                compliant_acceleration=pc.slat_compliant_acceleration,
            )
            paths.append(path)
        # Every env's slats to its bucket's material, all pool shapes, in ONE layer (bind_materials).
        s_idx_list = s_idx.tolist()
        bindings = []
        for k, art in enumerate(self._manipuland_arts()):
            asset = art.cfg.prim_path.rsplit("/", 1)[-1]
            envs = self._manipuland_envs(k)
            rels = collider_paths(stage, f"/World/envs/env_{envs[0]}/{asset}")
            if not rels:
                raise RuntimeError(f"[physx] no collider under env_{envs[0]}/{asset}")
            bindings.extend(
                (f"/World/envs/env_{int(i)}/{asset}/{rel}", paths[s_idx_list[int(i)]])
                for i in envs for rel in rels
            )
        slats = bind_materials(stage, bindings, tag="bind_slats_cloth_min")
        self._fingertip_colliders = 0
        print(
            f"[physx] friction model cloth_min: bound {slats} slat colliders to {len(paths)} "
            f"'{pc.slat_friction_combine}' material(s); robot and table average-mode, written at start",
            flush=True,
        )

    def _write_cloth_min_frictions(self) -> None:
        """Every robot shape at its env's robot friction, every table shape at its table friction."""
        ids = torch.arange(self.num_envs, dtype=torch.int64, device="cpu")
        for asset, mu in ((self.robot, self._fric_robot), (self.table, self._fric_table)):
            view = asset.root_physx_view
            mats = view.get_material_properties()
            mats[:, :, 0] = mu.unsqueeze(1)
            mats[:, :, 1] = mu.unsqueeze(1)
            mats[:, :, 2] = 0.0
            view.set_material_properties(mats, ids)

    def _check_cloth_min_frictions(self) -> None:
        """Read every friction back; log the pair distribution."""
        pc = self.cfg.physx_chain
        ids = torch.arange(self.num_envs, device=self.device)
        worst = 0.0
        for k, art in enumerate(self._manipuland_arts()):
            keep, local = self._manipuland_rows(k, ids)
            if local.numel() == 0:
                continue
            env_ids = ids if keep is None else ids[keep.to(ids.device)]
            mats = art.root_physx_view.get_material_properties()[local.cpu().long()]
            want = self._fric_slat[env_ids.cpu()].unsqueeze(1)
            worst = max(worst, float((mats[..., 0] - want).abs().max()))
        for asset, mu in ((self.robot, self._fric_robot), (self.table, self._fric_table)):
            mats = asset.root_physx_view.get_material_properties()
            worst = max(worst, float((mats[..., 0] - mu.unsqueeze(1)).abs().max()))
        if worst > 1e-4:
            raise RuntimeError(f"[physx] cloth_min frictions off by {worst:.4f} from what was written")
        s, f, t = self._fric_slat, self._fric_robot, self._fric_table
        pairs = {
            "cloth-table": torch.minimum(s, t), "cloth-robot": torch.minimum(s, f),
            "robot-table": 0.5 * (f + t),
        }
        capped = float(((s > t) | (s > f)).float().mean())
        txt = ", ".join(
            f"{k} {float(v.mean()):.3f} [{float(v.min()):.3f}, {float(v.max()):.3f}]" for k, v in pairs.items()
        )
        print(
            f"[rigid_cloth] friction (cloth_min): slat {float(s.mean()):.3f} ({pc.slat_friction_combine}), "
            f"robot {float(f.mean()):.3f}, table {float(t.mean()):.3f} (average) -> {txt}; "
            f"slat above robot or table in {capped:.1%} of envs (that pair takes the lower)",
            flush=True,
        )

    def _contact_margin_value(self) -> float:
        return float(self.cfg.rigid_cloth.contact_margin)

    def _contact_offset_value(self) -> float:
        """The slats' PhysX contact offset: above the LARGEST rest offset margin DR can draw.

        Fixed for the run. PhysX refuses a rest offset at or above the contact offset, so a contact
        offset that tracked each draw would have to be rewritten before or after the rest offset
        depending on the direction of the change. It only sets how early a contact is generated,
        not where the surface rests.
        """
        rd = self.cfg.rigid_cloth.randomization
        top = float(rd.contact_margin_offset[1]) if rd.enabled else 0.0
        return self._contact_margin_value() + max(top, 0.0) + float(self.cfg.physx_chain.contact_offset_margin)

    def _check_contacts(self) -> None:
        """Contacts are already authored; read them back and refuse to run on a mismatch."""
        r, pc = self.cfg.rigid_cloth, self.cfg.physx_chain
        m = self._contact_margin_value()
        self._contact_margin = m
        cloth_min = pc.friction_model == "cloth_min"
        if cloth_min:
            self._write_cloth_min_frictions()
            self._check_cloth_min_frictions()
        for art in self._manipuland_arts():
            view = art.root_physx_view
            mats = view.get_material_properties()
            rest = view.get_rest_offsets()
            bad_mu = 0.0 if cloth_min else float((mats[..., 0] - r.friction).abs().max())
            bad_rest = float((rest - m).abs().max())
            if bad_mu > 1e-4 or bad_rest > 1e-6:
                raise RuntimeError(
                    f"[physx] {art.cfg.prim_path}: slat friction off by {bad_mu:.4f} or rest offset off "
                    f"by {bad_rest * 1e3:.3f} mm from what was authored"
                )
        if cloth_min:
            print(
                f"[rigid_cloth] contact margin {m * 1e3:.2f} mm as PhysX rest offset on every slat "
                f"(contact offset {self._contact_offset_value() * 1e3:.1f} mm)",
                flush=True,
            )
            return
        tips = self.robot.root_physx_view.get_material_properties()[0, :, 0]
        n_tip = int(((tips - r.fingertip_friction).abs() < 1e-4).sum())
        n_def = int(((tips - DEFAULT_MU).abs() < 1e-4).sum())
        if n_tip < self._fingertip_colliders or n_tip + n_def != tips.numel():
            raise RuntimeError(
                f"[physx] robot frictions: {n_tip} shapes at {r.fingertip_friction} (bound "
                f"{self._fingertip_colliders} fingertip colliders), {n_def} at {DEFAULT_MU}, of "
                f"{tips.numel()}: {sorted(set(round(float(v), 3) for v in tips))}"
            )
        print(
            f"[rigid_cloth] friction: slat {r.friction:.3f} ({pc.slat_friction_combine}), fingertip "
            f"{r.fingertip_friction:.3f} ({pc.fingertip_friction_combine}, {n_tip} colliders), robot "
            f"links and table {DEFAULT_MU} (average) -> slat-table/link {min(r.friction, DEFAULT_MU):.3f}, "
            f"slat-fingertip {r.fingertip_friction:.3f}, fingertip-table "
            f"{max(r.fingertip_friction, DEFAULT_MU):.3f}\n"
            f"[rigid_cloth] contact margin {m * 1e3:.2f} mm as PhysX rest offset on every slat "
            f"(contact offset {self._contact_offset_value() * 1e3:.1f} mm); slat-slat contact "
            f"{'off (articulation self-collision disabled)' if m > 0.0 else 'on'}",
            flush=True,
        )
