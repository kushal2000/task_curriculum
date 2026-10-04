"""Is the RL task actually SOLVABLE by each rigid approximation? Checked without Isaac Sim.

``rigid_cloth_probe`` answers "does this chain fold". That is not the same question as "can a policy
score a fold", because the score is computed by ``ClothEnv`` from a **particle cloud on the cloth's
own 7x7 grid**, through ``ChainAsClothMesh``, against a tolerance of 0.04 m. Three things sit between
a folded chain and a reward, and every one of them can be wrong on its own:

  1. the grid-to-slat mapping (``rigid_cloth.sheet_grid_slat_map``) -- wrong ordering would attach
     every index set the task computes to the wrong particles, silently;
  2. the Kabsch orientation fit in ``cloth_adapter`` -- it needs the four tracked corners to stay
     RIGID under the fold, which is exactly what ``corner_indices`` was rewritten to guarantee for
     the VBD sheet and has never been checked for a chain;
  3. the fold target's lift -- built from where the plies come to rest, which differs per variant.

So this drives each chain to its folded pose in MuJoCo, lets it settle under gravity, reads the real
``body_pos_w`` / ``body_quat_w`` out of the simulator, and pushes them through **the same adapter and
the same arithmetic the env uses** to produce ``fold_error`` and ``object_rot``. If a physically
settled fold does not score under tolerance here, no amount of RL will make it score there -- and it
would present as a policy that cannot learn, not as a metric that cannot be satisfied.

Reports, per variant:

    fold_err    max over the 4 tracked keypoints of |position - target|, metres. The env's own
                success quantity. Must be < `success_tolerance` (0.04) with margin.
    flat_err    the same at rest, i.e. the score of doing nothing. The gap between the two is the
                signal the policy has to climb; a small gap is a hard credit-assignment problem
                however stable the physics.
    fit_deg     the Kabsch rotation the policy observes as `object_rot`. A completed fold is 180.
    rest_um     FK cloud vs the analytic flat grid, MICROMETRES. A pure self-check of the
                mapping; any non-zero value means the rest of the row is meaningless.

    .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_task_check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# The shared by-path loader; `_rigid_cloth_common` explains why these scripts must not import
# `isaacsimenvs` as a package. `sys.path` first, so the import works both as
# `-m scripts.analysis.<name>` (sys.path[0] is the repo root) and as a plain script path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rigid_cloth_common import REPO_ROOT, load_module  # noqa: E402  (needs the path insert above)

rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py")
cg = load_module("isaacsimenvs/tasks/cloth/utils/cloth_geometry.py")
probe = load_module("scripts/analysis/rigid_cloth_probe.py")


class _FakeArticulationData:
    """The three body-state arrays ``ChainAsClothMesh`` reads, filled from a MuJoCo state.

    A stub rather than the real ``Articulation`` because Isaac Sim needs a GPU and a Kit kernel, and
    the arithmetic under test needs neither. What it must get right is the CONVENTION: Isaac Lab
    hands out ``(num_envs, num_bodies, ...)`` with wxyz quaternions in the world frame, and MuJoCo
    stores ``xpos`` / ``xquat`` per body also as wxyz -- so the only translation is the body
    ordering, which the adapter resolves by name through ``body_names``.
    """

    def __init__(self, pos, quat, lin, ang):
        self.body_pos_w = pos
        self.body_quat_w = quat
        self.body_lin_vel_w = lin
        self.body_ang_vel_w = ang


class _FakeArticulation:
    def __init__(self, body_names, data):
        self.body_names = body_names
        self.data = data


def _mujoco_to_articulation(model, data, spec, device):
    """Read one MuJoCo state out as the body arrays an Isaac Lab articulation would present."""
    import mujoco
    import torch

    names = [rc.slat_link_name(i) for i in range(spec.num_slats)]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    if any(i < 0 for i in ids):
        raise RuntimeError(f"MuJoCo model is missing slat bodies: {names}")

    pos = torch.tensor([[data.xpos[i].copy() for i in ids]], dtype=torch.float32, device=device)
    quat = torch.tensor([[data.xquat[i].copy() for i in ids]], dtype=torch.float32, device=device)
    # cvel is [angular, linear] in the body-CENTRED frame; the adapter wants the link frame, and for
    # this check velocities only have to be finite and consistent, not exact.
    lin = torch.zeros_like(pos)
    ang = torch.zeros_like(pos)
    for k, i in enumerate(ids):
        ang[0, k] = torch.tensor(data.cvel[i][:3].copy(), dtype=torch.float32, device=device)
        lin[0, k] = torch.tensor(data.cvel[i][3:].copy(), dtype=torch.float32, device=device)
    return _FakeArticulation(names, _FakeArticulationData(pos, quat, lin, ang))


def _score(mesh, adapter, spec, args, lift):
    """``fold_error`` and the observed rotation, computed exactly as ``ClothEnv`` computes them."""
    import torch

    cloud = mesh._positions()                       # (1, P, 3) world
    kp_idx = cg.corner_indices(args.resolution, "x")
    rest, _ = cg.grid_mesh(args.size, args.resolution)

    # `ClothEnv.fold_targets_w` is defined RELATIVE TO THE SHEET: it carries each keypoint's folded
    # rest position onto the stationary half's live rigid frame, so the criterion cannot be satisfied
    # by sliding the whole sheet. Reproduced here the same way -- Kabsch on the stationary half.
    st_idx = sorted(set(cg.half_indices(args.resolution, "x", positive=False)))
    st_rest = torch.tensor([rest[i] for i in st_idx], dtype=torch.float32)
    st_cur = cloud[0, st_idx, :]
    p = st_rest - st_rest.mean(dim=0, keepdim=True)
    q = st_cur - st_cur.mean(dim=0, keepdim=True)
    h = p.T @ q + 1e-9 * torch.eye(3)
    u, _, vh = torch.linalg.svd(h)
    v = vh.T
    d = torch.ones(3)
    d[2] = torch.linalg.det(v @ u.T)
    r_st = v @ torch.diag(d) @ u.T                  # rest -> current, stationary half

    folded_rest = []
    for i in kp_idx:
        x, y, z = rest[i]
        folded_rest.append([-x, y, z + lift])       # reflect across the crease, lift onto the ply
    folded_rest = torch.tensor(folded_rest, dtype=torch.float32)
    st_centroid = st_rest.mean(dim=0)
    target = (folded_rest - st_centroid) @ r_st.T + st_cur.mean(dim=0)

    err = (cloud[0, kp_idx, :] - target).norm(dim=-1).amax().item()
    quat = adapter.fit_rotation_wxyz()[0]
    angle = 2.0 * math.degrees(math.acos(min(1.0, abs(float(quat[0])))))
    return err, angle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=float, default=rc.CLOTH_REFERENCE["size"])
    p.add_argument("--resolution", type=int, default=rc.CLOTH_REFERENCE["resolution"])
    p.add_argument("--tolerance", type=float, default=0.04, help="termination.success_tolerance")
    p.add_argument("--density", type=float, default=2.0)
    p.add_argument("--damping", type=float, default=1.0e-5)
    p.add_argument("--friction", type=float, default=rc.CLOTH_REFERENCE["soft_contact_mu"])
    p.add_argument("--ground_friction", type=float, default=rc.CLOTH_REFERENCE["shape_mu"]["table"])
    p.add_argument("--no-pairs", dest="pairs", action="store_false")
    p.set_defaults(pairs=True)
    p.add_argument("--stiffness", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--settle_steps", type=int, default=720)
    p.add_argument("--out_dir", default="/tmp/rigid_cloth_probe")
    p.add_argument(
        "--cloth_lift",
        action="store_true",
        help="score against the CLOTH's 2 mm self-contact radius instead of each chain's own ply "
        "gap, i.e. `rigid_cloth.fold_lift_from_ply_gap=false`",
    )
    args = p.parse_args()
    args.variant = None
    args.num_slats = rc.DEFAULT_NUM_SLATS
    args.thickness = None
    args.shape = "cylinder"
    args.hinge = "mid"
    args.drop_height = 0.05

    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import (
        ChainAsClothMesh,
        ChainAsRigidObject,
    )

    device = torch.device("cpu")
    tol = args.tolerance
    rows = []
    for name in rc.VARIANTS:
        spec = rc.variant(name)
        args.variant = name
        lift = rc.CLOTH_REFERENCE["measured_ply_gap"] if args.cloth_lift else rc.chain_ply_gap(spec)
        if args.cloth_lift:
            lift = rc.CLOTH_REFERENCE["self_contact_radius"]

        rest_verts, _ = cg.grid_mesh(args.size, args.resolution)
        kp_idx = cg.corner_indices(args.resolution, "x")

        def build(angles, steps):
            model, data = probe._build(args, spec)
            probe._place(
                model, data, spec,
                [0.0, 0.0, probe._rest_height(spec) if angles else args.drop_height],
                angles,
            )
            probe._run(model, data, steps)
            art = _mujoco_to_articulation(model, data, spec, device)
            mesh = ChainAsClothMesh(
                art, spec=spec, resolution=args.resolution, num_envs=1, device=device
            )
            adapter = ChainAsRigidObject(
                mesh, articulation=art, spec=spec, num_envs=1, rest_local=rest_verts,
                keypoint_idx=kp_idx, corner_idx=kp_idx,
                corner_rest=[rest_verts[i] for i in kp_idx],
                spawn_z=0.0, mass=args.density * args.size * spec.span, device=device,
            )
            return model, data, mesh, adapter

        # --- rest: the mapping's own self-check, and the score of doing nothing
        _m, _d, mesh, adapter = build(None, args.settle_steps)
        flat_err, flat_deg = _score(mesh, adapter, spec, args, lift)
        # A flat sheet must fit as UNROTATED. A non-zero angle here means the Kabsch fit is reading
        # rotation out of a shape that has none, and every folded number below inherits that error.
        if flat_deg > 1.0:
            raise RuntimeError(
                f"{name}: object_rot reads {flat_deg:.1f} deg on a FLAT sheet -- the corner fit is "
                "degenerate, so the fold numbers below are not trustworthy"
            )
        # Compare the FK cloud against the analytic grid in the sheet's own frame: the chain has
        # settled on the floor, so subtract the mean before comparing shape.
        cloud = mesh._positions()[0]
        want = torch.tensor(rest_verts, dtype=torch.float32)
        rest_err = float((cloud - cloud.mean(0) - (want - want.mean(0))).norm(dim=-1).amax())

        # --- folded: driven to the fold and released, so contact and gravity have their say
        _m, _d, mesh, adapter = build(rc.chain_fold_angles(spec), args.settle_steps)
        fold_err, fold_deg = _score(mesh, adapter, spec, args, lift)

        rows.append((name, spec.num_slats, rest_err, flat_err, fold_err, fold_deg, lift))

    print(
        f"\nfold criterion: max keypoint error < {tol:.3f} m   "
        f"(lift = {'cloth 2.00 mm' if args.cloth_lift else 'each chain ply gap'})\n"
    )
    hdr = f"{'variant':14s} {'bod':>3s} {'rest_um':>9s} {'flat_err':>9s} {'fold_err':>9s} {'fit_deg':>8s} {'lift_mm':>8s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for name, nb, rest_err, flat_err, fold_err, fold_deg, lift in rows:
        ok = "SOLVABLE" if fold_err < tol else "UNREACHABLE"
        margin = "" if fold_err >= tol else f"  margin {(tol - fold_err) * 1e3:.0f} mm"
        print(
            f"{name:14s} {nb:3d} {rest_err * 1e6:7.2f}um {flat_err:9.4f} {fold_err:9.4f} "
            f"{fold_deg:8.1f} {lift * 1e3:8.2f}  {ok}{margin}"
        )
    print(
        "\nrest_um is the grid-to-slat mapping checked against the analytic flat sheet, in "
        "MICROMETRES; anything but ~0 invalidates the rest of the row.\nflat_err is the score of "
        "doing nothing, so fold_err - flat_err is the signal a policy has to find."
    )


if __name__ == "__main__":
    main()
