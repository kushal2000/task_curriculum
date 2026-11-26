"""Cloth health per policy step and env (diagnostic, 2026-11-25; checks after Alloy's health flags).

    hand_tracking_probe.py ... --health

Per policy step and env, from the coupled solver's VBD entry:

- ``speed``: the fastest cloth particle [m/s] at the end of the step (a snapshot). Alloy marks a world
  INVALID past 20 m/s (SPEED_FAIL); Newton's cloth_twist test fails past 1.5 m/s per component.
- ``strain``: the most-stretched cloth edge, |x_i - x_j| / rest - 1 (rest = the model's build positions,
  the flat sheet). Alloy's CLOTH_STRETCH.
- ``kick`` / ``cuts``: the largest body-contact exit kick (resolved start depth / dt) over the step's
  substeps, and how many particle velocities the exit-speed cap cut (needs ``env.cloth.exit_speed_measure``
  or ``env.cloth.exit_speed_cap``; ``newton_support/contact_guards.py``). Accumulated on the GPU, so a
  max over every substep, not a snapshot.
- ``sc_overflow``: self-contact primitives whose collision buffer overflowed (VBD drops those pairs
  silently). Alloy's SELF_CONTACT_OVERFLOW. Vertex-triangle and edge-edge, as of the last detection.
"""

from __future__ import annotations

import numpy as np

SPEED_LEVELS = (1.0, 5.0, 20.0)
KICK_LEVELS = (0.5, 1.0, 5.0)
STRAIN_LEVELS = (0.1, 0.25, 0.5)


class ClothHealthRecorder:
    def __init__(self, inner):
        from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import contact_guards

        self._guards = contact_guards
        mgr = inner.sim.physics_manager
        self._mgr = mgr
        coupled = getattr(mgr, "_solver", None)
        if coupled is None or not hasattr(coupled, "solver"):
            raise RuntimeError("cloth health needs the coupled MJWarp + VBD solver")
        self.vbd = coupled.solver("cloth")
        model = self.vbd.model
        self.E = int(inner.num_envs)
        pw = model.particle_world.numpy().astype(np.int64)
        self._pw = pw
        self._ok = (pw >= 0) & (pw < self.E)
        tri = model.tri_indices.numpy().astype(np.int64)
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        q0 = model.particle_q.numpy().astype(np.float64)
        rest = np.linalg.norm(q0[e[:, 0]] - q0[e[:, 1]], axis=1)
        keep = rest > 1e-9
        self._e0, self._e1, self._rest = e[keep, 0], e[keep, 1], rest[keep]
        self._e_world = pw[self._e0]
        self._det = getattr(self.vbd, "trimesh_collision_detector", None)
        self._edge_world = None
        if self._det is not None and getattr(model, "edge_indices", None) is not None and model.edge_count > 0:
            self._edge_world = pw[model.edge_indices.numpy()[:, 2].astype(np.int64)]
        self.rec = {k: [] for k in ("speed", "strain", "kick", "cuts", "sc_overflow")}
        print(f"[cloth_health] {int(self._ok.sum())} cloth particles over {self.E} envs, {self._e0.size} edges "
              f"(rest {self._rest.min() * 1e3:.1f}-{self._rest.max() * 1e3:.1f} mm), self-contact detector "
              f"{'yes' if self._det is not None else 'no'}", flush=True)

    def _per_env_max(self, v: np.ndarray, world: np.ndarray) -> np.ndarray:
        out = np.zeros(self.E)
        ok = (world >= 0) & (world < self.E)
        np.maximum.at(out, world[ok], v[ok])
        return out

    def _per_env_sum(self, v: np.ndarray, world: np.ndarray) -> np.ndarray:
        ok = (world >= 0) & (world < self.E)
        return np.bincount(world[ok], weights=v[ok], minlength=self.E)[: self.E]

    def sample(self) -> None:
        st = self._mgr._state_0
        q = st.particle_q.numpy().astype(np.float64)
        qd = st.particle_qd.numpy().astype(np.float64)
        sp = np.nan_to_num(np.linalg.norm(qd, axis=1), nan=1e9)
        self.rec["speed"].append(self._per_env_max(sp, self._pw))
        ln = np.linalg.norm(q[self._e0] - q[self._e1], axis=1)
        strain = np.nan_to_num(ln / self._rest - 1.0, nan=1e9)
        self.rec["strain"].append(self._per_env_max(strain, self._e_world))
        ex = self._guards.exit_stats(self.vbd, reset=True)
        if ex is not None:
            kick, _cut, capped = ex
            self.rec["kick"].append(self._per_env_max(kick.astype(np.float64), self._pw))
            self.rec["cuts"].append(self._per_env_sum(capped.astype(np.float64), self._pw))
        if self._det is not None:
            d = self._det
            over_v = (d.vertex_colliding_triangles_count.numpy() > d.vertex_colliding_triangles_buffer_sizes.numpy())
            ov = self._per_env_sum(over_v.astype(np.float64), self._pw[: over_v.shape[0]])
            if self._edge_world is not None and getattr(d, "edge_colliding_edges_count", None) is not None:
                over_e = d.edge_colliding_edges_count.numpy() > d.edge_colliding_edges_buffer_sizes.numpy()
                ov = ov + self._per_env_sum(over_e.astype(np.float64), self._edge_world[: over_e.shape[0]])
            self.rec["sc_overflow"].append(ov)

    def summary(self, alive: np.ndarray, finite: np.ndarray, unc_near_cloth: np.ndarray | None) -> dict:
        """``alive`` (S, E) bool per recorded step; ``finite`` (E,); ``unc_near_cloth`` (S-1, E) count of
        uncommanded near-cloth jumps over step t (index t-1), any watched joint."""
        S = len(self.rec["speed"])
        live = alive[:S] & finite[None, :]
        q = lambda x, p: float(np.quantile(x, p)) if x.size else float("nan")  # noqa: E731
        out: dict = {"steps": S, "env_steps": int(live.sum())}

        def block(name, levels, unit):
            if not self.rec[name]:
                return None
            a = np.stack(self.rec[name])[:S]
            v = a[live]
            r = {f"p50_{unit}": q(v, 0.5), f"p99_{unit}": q(v, 0.99), f"max_{unit}": float(v.max()) if v.size else float("nan")}
            for lv in levels:
                r[f"env_steps_over_{lv:g}"] = int((v > lv).sum())
                r[f"envs_ever_over_{lv:g}"] = int(((a > lv) & live).any(axis=0).sum())
            return r, a

        for name, levels, unit in (("speed", SPEED_LEVELS, "mps"), ("strain", STRAIN_LEVELS, "frac"),
                                   ("kick", KICK_LEVELS, "mps")):
            b = block(name, levels, unit)
            if b is None:
                continue
            out[name], a = b
            # Do the uncommanded near-cloth jumps coincide with a spike? (a jump at index t-1 is step t)
            if unc_near_cloth is not None and name in ("speed", "kick"):
                j = unc_near_cloth[: S - 1] > 0
                spike = np.maximum(a[1:S], a[: S - 1])                     # the jump's step or the one before
                lv = levels[1]
                n_j = int((j & live[1:S]).sum())
                out[name][f"uncmd_near_cloth_jumps_with_{name}_over_{lv:g}"] = int((j & live[1:S] & (spike > lv)).sum())
                out[name]["uncmd_near_cloth_jump_steps"] = n_j
                out[name][f"base_rate_{name}_over_{lv:g}"] = float(((spike > lv) & live[1:S]).mean()) if live[1:S].any() else float("nan")
        if self.rec["cuts"]:
            c = np.stack(self.rec["cuts"])[:S]
            out["exit_cap_cuts"] = int(c[live].sum())
        if self.rec["sc_overflow"]:
            o = np.stack(self.rec["sc_overflow"])[:S]
            out["self_contact_overflow"] = {"env_steps_with_overflow": int((o[live] > 0).sum()),
                                            "primitives_max": float(o[live].max()) if live.any() else 0.0}
        print("[cloth_health] " + "; ".join(
            f"{k} p99/max {v.get('p99_mps', v.get('p99_frac', float('nan'))):.3g}/"
            f"{v.get('max_mps', v.get('max_frac', float('nan'))):.3g}"
            for k, v in out.items() if isinstance(v, dict) and ("p99_mps" in v or "p99_frac" in v))
            + f"; cuts {out.get('exit_cap_cuts', '-')}; sc overflow {out.get('self_contact_overflow', '-')}", flush=True)
        return out
