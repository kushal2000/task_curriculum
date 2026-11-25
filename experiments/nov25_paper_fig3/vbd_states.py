"""Staged VBD cloth failures for the paper's Fig. 3 (no robot): a sheet folded by kinematically moved particles,
then released. The grip sets are what makes each fold fail; nothing here comes from a policy rollout.

    .venv/isaaclab3/bin/python experiments/nov25_paper_fig3/vbd_states.py <out_dir>

Writes <out_dir>/<case>.npz: rest (P,3), faces, frames (T,P,3), angle (T,) of the gripped particles [deg].
Material: config/task.yaml's vbd.cloth, at 41x41 particles instead of 7x7.
"""
import sys
from pathlib import Path

import newton
import numpy as np
import warp as wp

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
L, N = 0.05, 41  # half size [m], particles per side
RADIUS, Z0 = 0.0015, 0.002

CASES = {  # gripped particles (x, y in units of L)
    "strip": lambda x, y: (np.abs(x - 0.55) < 0.03),             # a line inside the edge: the outer strip is not folded
    "corner": lambda x, y: (x > 0.85) & (y < 0.25),              # part of the edge: the other corner is left behind
}


def run(name, grip):
    b = newton.ModelBuilder()
    cell = 2 * L / (N - 1)
    b.add_cloth_grid(pos=wp.vec3(-L, -L, Z0), rot=wp.quat_identity(), vel=wp.vec3(0, 0, 0), dim_x=N - 1, dim_y=N - 1,
                     cell_x=cell, cell_y=cell, mass=2.0 * cell * cell, tri_ke=1e2, tri_ka=1e2, tri_kd=1.5e-6,
                     edge_ke=8e-4, edge_kd=1e-3, particle_radius=RADIUS)
    b.add_ground_plane()
    b.color()
    m = b.finalize()
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 5e2, 5e-3, 0.6
    rest = m.particle_q.numpy().copy()
    ids = np.where(grip(rest[:, 0] / L, rest[:, 1] / L))[0]
    base = np.where(rest[:, 0] < -0.1 * L)[0]                    # support half held, as by friction under the palm
    flags0 = m.particle_flags.numpy().copy()
    flags = flags0.copy(); flags[ids] &= ~int(newton.ParticleFlags.ACTIVE); flags[base] &= ~int(newton.ParticleFlags.ACTIVE)
    m.particle_flags = wp.array(flags, dtype=m.particle_flags.dtype)
    solver = newton.solvers.SolverVBD(m, iterations=12, particle_enable_self_contact=True, particle_self_contact_radius=0.0015,
                                      particle_self_contact_margin=0.0025, particle_conservative_bound_relaxation=0.42)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    pipe = newton.CollisionPipeline(m); con = pipe.contacts()
    fps, sub = 60, 24
    dt = 1 / fps / sub
    T_fold, T_settle = 150, 120
    end = np.deg2rad(172)
    frames, angle, rel = [], [], False
    for f in range(T_fold + T_settle):
        if f == T_fold:                                          # release the grip (the support half stays held)
            fl = flags0.copy(); fl[base] &= ~int(newton.ParticleFlags.ACTIVE)
            m.particle_flags = wp.array(fl, dtype=m.particle_flags.dtype); rel = True
            v = s0.particle_qd.numpy(); v[ids] = 0; s0.particle_qd.assign(v)
        for k in range(sub):
            th = end * min(1.0, (f + (k + 1) / sub) / T_fold)
            s0.clear_forces()
            pipe.collide(s0, con)
            if not rel:
                q = s0.particle_q.numpy(); x = rest[ids, 0]
                q[ids, 0] = x * np.cos(th); q[ids, 2] = Z0 + x * np.sin(th) + 0.004 * np.sin(th / 2)
                s0.particle_q.assign(q); s1.particle_q.assign(q)
            solver.step(s0, s1, ctrl, con, dt)
            s0, s1 = s1, s0
        q = s0.particle_q.numpy().copy()
        assert np.isfinite(q).all(), f"{name}: non-finite at frame {f}"
        if f % 5 == 4:
            frames.append(q); angle.append(np.rad2deg(th))
    np.savez(OUT / f"{name}.npz", rest=rest, faces=m.tri_indices.numpy(), frames=np.array(frames), angle=np.array(angle))
    print(name, "grip", len(ids), "frames", len(frames), "z", frames[-1][:, 2].min(), frames[-1][:, 2].max(), flush=True)


for n, g in CASES.items():
    run(n, g)
