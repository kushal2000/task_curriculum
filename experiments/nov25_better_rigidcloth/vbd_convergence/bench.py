"""How many VBD iterations does the cloth need? Hand-free tests of the stable.yaml sheet in Newton's
SolverVBD alone (no robot, no coupler), each swept over the iteration count:

  drop        the flat sheet dropped 3 cm onto the table: does it come to rest (residual speed), how
              stretched is it at rest
  hang        one edge pinned in the air, the rest hanging: steady stretch and residual sway
  cantilever  half the sheet over the table edge: how far the free edge droops (and how still it is)

The sheet is stable.yaml's: 10 cm, 7 x 7, checkerboard cut, 16 mm particles, VBD at 1440 Hz, the same
contact / self-contact / move-limit settings; the material is one of
  stable   vbd.yaml's (tri_ke 100, edge_ke 8e-4, density 2)
  denim    fabric/denim.yaml's (tri_ke 200, edge_ke 6.6e-4, density 0.18; no plasticity here)
  stiff    tri_ke 3000, edge_ke 3e-3, density 0.18 (a realistic, stiffer fabric)

    python bench.py --material stable --test drop --iters 6 12 24 48 96 192 --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

import newton

from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh

MATERIALS = {
    "stable": dict(density=2.0, tri_ke=100.0, tri_ka=100.0, tri_kd=1.5e-6, edge_ke=8.0e-4, edge_kd=1.0e-3),
    "denim": dict(density=0.1822, tri_ke=200.0, tri_ka=200.0, tri_kd=1.5e-6, edge_ke=6.594e-4, edge_kd=9.11e-5),
    "stiff": dict(density=0.1822, tri_ke=3000.0, tri_ka=3000.0, tri_kd=1.5e-6, edge_ke=3.0e-3, edge_kd=1.0e-4),
}
SIZE, RES, THICK = 0.10, 7, 0.016
DT = 1.0 / 1440.0
TABLE_MU, CLOTH_MU = 0.7, 0.25


def build(material: dict, test: str, iters: int):
    b = newton.ModelBuilder()
    r = THICK / 2.0
    cfg = b.default_shape_cfg.copy()
    cfg.mu = TABLE_MU
    # the table: a box whose top is z = 0; for the cantilever its +x edge is at x = 0
    if test in ("drop", "cantilever"):
        cx = -0.15 if test == "cantilever" else 0.0
        b.add_shape_box(body=-1, xform=wp.transform((cx, 0.0, -0.05), wp.quat_identity()), hx=0.15, hy=0.15, hz=0.05, cfg=cfg)
    verts, idx = grid_mesh(SIZE, RES, "alternate")
    z0 = {"drop": r + 0.03, "hang": 0.5, "cantilever": r + 0.0005}[test]
    x0 = 0.0
    b.add_cloth_mesh(pos=wp.vec3(x0, 0.0, z0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                     vertices=[wp.vec3(*v) for v in verts], indices=list(idx), density=material["density"],
                     tri_ke=material["tri_ke"], tri_ka=material["tri_ka"], tri_kd=material["tri_kd"],
                     edge_ke=material["edge_ke"], edge_kd=material["edge_kd"], particle_radius=r)
    P = np.asarray(verts, float)
    if test in ("hang", "cantilever"):
        # hang: the row at the smallest x; cantilever: the half lying on the table (x <= 0), so only the
        # overhang can move -- a pure bending test (an unpinned half-overhanging sheet just slides off)
        pinned = (np.nonzero(np.isclose(P[:, 0], P[:, 0].min()))[0] if test == "hang"
                  else np.nonzero(P[:, 0] <= 1e-9)[0])
        for i in pinned:
            b.particle_mass[i] = 0.0
            b.particle_flags[i] = int(b.particle_flags[i]) & ~int(newton.ParticleFlags.ACTIVE)
    b.color(include_bending=True)
    model = b.finalize()
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 500.0, 5.0e-3, CLOTH_MU
    solver = newton.solvers.SolverVBD(
        model, iterations=iters, particle_enable_self_contact=True, particle_self_contact_radius=0.002,
        particle_self_contact_margin=0.003, particle_conservative_bound_relaxation=0.42,
        rigid_avbd_beta=5.0e5, rigid_contact_k_start=100.0, particle_collision_detection_interval=-1)
    return model, solver, P + np.array([x0, 0.0, z0])


def edge_strain(pts: np.ndarray) -> tuple[float, float]:
    sp = SIZE / (RES - 1)
    s = []
    for i in range(RES):
        for j in range(RES):
            k = i * RES + j
            if j + 1 < RES:
                s.append(np.linalg.norm(pts[k + 1] - pts[k]) / sp)
            if i + 1 < RES:
                s.append(np.linalg.norm(pts[k + RES] - pts[k]) / sp)
    s = np.asarray(s)
    return float(np.median(s)), float(s.max())


def run(material_name: str, test: str, iters: int, seconds: float) -> dict:
    model, solver, rest = build(MATERIALS[material_name], test, iters)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()

    def step():
        nonlocal s0, s1
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, DT)
        s0, s1 = s1, s0

    n = int(round(seconds / DT))
    step()  # warm up (compile)
    step()
    wp.synchronize()
    # 24 steps (1/60 s) as one CUDA graph: the cloth is tiny, so the cost is kernel launches, not work.
    # An even number of steps, so the swapped states are back where they started after each replay.
    graph = None
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as cap:
            for _ in range(24):
                step()
        graph = cap.graph
    t = time.perf_counter()
    speeds, settle = [], None
    for k in range(24, n, 24):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            for _ in range(24):
                step()
        if True:  # every 1/60 s
            v = s0.particle_qd.numpy()
            sp = float(np.linalg.norm(v, axis=1).mean())
            speeds.append(sp)
            if settle is None and k * DT > 0.1 and sp < 1e-3:
                settle = k * DT
    wp.synchronize()
    wall = time.perf_counter() - t
    q = s0.particle_q.numpy()
    med, mx = edge_strain(q)
    out = dict(material=material_name, edge_kd=MATERIALS[material_name]["edge_kd"], test=test, iters=iters, seconds=seconds, wall_s=round(wall, 2),
               ms_per_step=round(1000 * wall / max(n - 1, 1), 3), finite=bool(np.isfinite(q).all()),
               strain_median=round(med, 4), strain_max=round(mx, 4),
               residual_speed_mm_s=round(1000 * float(np.mean(speeds[-12:])), 3), settle_s=settle)
    if test == "cantilever":  # the free edge (largest x): how far below the table top
        free = np.isclose(rest[:, 0], rest[:, 0].max())
        out["droop_mm"] = round(1000 * float(rest[free, 2].mean() - q[free, 2].mean()), 2)
        dx = float(q[free, 0].mean())  # overhang x after sliding (the table edge is x = 0)
        out["droop_deg"] = round(float(np.degrees(np.arctan2(out["droop_mm"] / 1000, max(dx, 1e-6)))), 1)
    if test == "hang":
        out["free_edge_z_mm"] = round(1000 * float(q[np.isclose(rest[:, 0], rest[:, 0].max()), 2].mean()), 2)
    if test == "drop":
        out["height_mm"] = round(1000 * float(np.median(q[:, 2])), 2)
    out["final_q"] = q.round(5).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="stable", choices=list(MATERIALS))
    ap.add_argument("--test", default="drop", choices=("drop", "hang", "cantilever"))
    ap.add_argument("--iters", type=int, nargs="+", default=[6, 12, 24, 48, 96, 192])
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--edge_kd", type=float, default=None, help="override the material's bending damping")
    a = ap.parse_args()
    if a.edge_kd is not None:
        MATERIALS[a.material] = dict(MATERIALS[a.material], edge_kd=a.edge_kd)
    wp.init()
    for it in a.iters:
        r = run(a.material, a.test, it, a.seconds)
        print(json.dumps({k: v for k, v in r.items() if k != "final_q"}), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
