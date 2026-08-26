"""Drive the cloth through a fold with kinematic ("magic") forces, release it, and record where
it settles.

The question this answers is not "can the policy fold" but "what does a fold physically DO under
the current parameters". Everything the fold criterion asserts is geometric prediction until
something measures it:

  * `fold_targets_w` lifts each folded particle by one `thickness` (16 mm) on the theory that a
    flap rests one particle diameter above the half beneath it. But contact separation is set by
    `self_contact_radius` (2 mm), a different parameter, so the resting gap is unmeasured.
  * `fold_error` at a perfect fold is 16 mm on paper, entirely from the two keypoints that sit on
    the HINGE row and are asked to rise while the crease stays down. A real cloth bulges at the
    crease, so the true floor should be lower -- by how much is unknown.
  * `edge_ke` was cut from 0.5 to 8.0e-4. Whether a crease still holds at that stiffness was
    measured at 0.5, never here.

Method: settle the sheet flat, pin the moving half's particles (`particle_inv_mass = 0` via
`write_nodal_kinematic_target_to_sim_index`) and sweep them along an arc about the live crease
onto the stationary half, then release everything and let it settle under its own dynamics. The
release is the measurement -- a driven fold proves nothing, since kinematic particles ignore the
solver entirely.

    scripts/newton_py -m scripts.analysis.cloth_fold_probe [--resolution 9 --thickness 0.012 ...]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-Cloth-Direct-v0")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle_steps", type=int, default=60, help="let the sheet land and go flat")
    p.add_argument("--fold_steps", type=int, default=90, help="steps to sweep the flap over")
    p.add_argument("--release_steps", type=int, default=180, help="steps to watch after release")
    # Geometry overrides, so this doubles as the 7x7-vs-9x9 comparison.
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--thickness", type=float, default=None)
    p.add_argument("--self_contact_radius", type=float, default=None)
    p.add_argument("--target_lift", type=float, default=None,
                   help="Override the fold-target lift [m]. `_init_fold_targets` uses "
                        "`2 * particle_radius` (a cloth-RIGID offset) as the ply separation, but "
                        "separation is set by `self_contact_radius` (cloth-CLOTH). This tests "
                        "correcting the target without touching the grip radius.")
    p.add_argument("--label", default="probe")
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.assets.handle_head_types = ("hammer",)
        # Newton requires type-homogeneous collision shapes across worlds: without this the
        # multi-asset spawner gives world 0 a different geom type and SolverMuJoCo refuses the
        # scene outright. Every eval entry point pins this the same way.
        disable_randomization(env_cfg)
        use_single_object_variant()
        if args.resolution is not None:
            env_cfg.cloth.resolution = args.resolution
        if args.thickness is not None:
            env_cfg.cloth.thickness = args.thickness
        if args.self_contact_radius is not None:
            env_cfg.cloth.self_contact_radius = args.self_contact_radius
            env_cfg.cloth.self_contact_margin = 1.5 * args.self_contact_radius
        # The guard runs in __post_init__ on the DEFAULTS, not after the overlay, so a bad
        # resolution/thickness pair would otherwise reach the solver silently.
        env_cfg.cloth._check_thickness()
        # A probe wants a NaN to be an error, not a quiet reset.
        env_cfg.cloth.nan_policy = "raise"

        c = env_cfg.cloth
        print(f"[probe] {args.label}: resolution={c.resolution} thickness={c.thickness} "
              f"(radius {c.particle_radius}) spacing={c.spacing:.5f} "
              f"self_contact_radius={c.self_contact_radius} edge_ke={c.edge_ke}", flush=True)

        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped

        if args.target_lift is not None:
            # Rebuild the folded rest positions with a different lift. `_folded_rest` adds
            # `c.thickness` to z; substitute the requested value and re-derive the arrays
            # `fold_targets_w` / `_folded_w` read.
            import torch as _t
            ax = 0 if c.fold_axis == "x" else 1
            rest = _t.tensor(inner._cloth_rest_local, device=inner.device, dtype=_t.float32)
            crease = rest[:, ax].max() - c.size / 2.0
            def _relift(sel):
                f = rest[sel].clone()
                f[:, ax] = 2.0 * crease - f[:, ax]
                f[:, 2] = f[:, 2] + args.target_lift
                return f
            inner._mv_folded_rest = _relift(inner._moving_idx)
            inner._kp_folded_rest = _relift(inner._kp_idx_t)
            print(f"[probe] fold-target lift OVERRIDDEN: {c.thickness*1000:.0f} mm -> "
                  f"{args.target_lift*1000:.0f} mm (grip radius unchanged at "
                  f"{c.particle_radius*1000:.0f} mm)", flush=True)
        dev = inner.device
        zero = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=dev)

        def stats(tag: str, step: int) -> dict:
            parts = inner._particles_w()                      # (N, P, 3)
            mv = parts[:, inner._moving_idx, :]
            st = parts[:, inner._stationary_idx, :]
            hinge = torch.tensor(
                [i for i in range(c.resolution ** 2) if i not in set(
                    inner._moving_idx.tolist()) | set(inner._stationary_idx.tolist())],
                device=dev, dtype=torch.long)
            kp_err = (inner.cloth_keypoints_w() - inner.fold_targets_w()).norm(dim=-1)  # (N,K)
            d = {
                "step": step,
                "fold_err": float(inner.fold_error().mean()),
                "footprint": float(inner.footprint_ratio().mean()),
                "folded": int(inner.is_folded().sum()),
                "z_stat": float(st[..., 2].mean()),
                "z_flap": float(mv[..., 2].mean()),
                "z_hinge": float(parts[:, hinge, 2].mean()),
                "gap_mm": float((mv[..., 2].mean() - st[..., 2].mean()) * 1000),
                "hinge_lift_mm": float((parts[:, hinge, 2].mean() - st[..., 2].mean()) * 1000),
                "kp_err_mm": [round(float(v) * 1000, 1) for v in kp_err.mean(dim=0)],
            }
            print(f"[{tag:7s}] step {step:4d}  fold_err {d['fold_err']:.4f}  "
                  f"footprint {d['footprint']:.3f}  folded {d['folded']}/{inner.num_envs}  "
                  f"z_stat {d['z_stat']:.4f} z_flap {d['z_flap']:.4f}  "
                  f"gap {d['gap_mm']:6.1f} mm  hinge_lift {d['hinge_lift_mm']:5.1f} mm  "
                  f"kp_err {d['kp_err_mm']}", flush=True)
            return d

        # NO TERMINATION during the probe. The first run of this reset mid-fold and the sheet
        # snapped back to EXACTLY the flat state -- `write_root_pose_to_sim` restores the sheet
        # flat, so a reset is indistinguishable from a spring-back unless it is ruled out. A
        # diagnostic wants the physics, not the reset machinery.
        import torch as _t
        _orig_dones = inner._get_dones
        def _no_dones():
            term, trunc = _orig_dones()
            return _t.zeros_like(term), _t.zeros_like(trunc)
        inner._get_dones = _no_dones
        print("[probe] termination SUPPRESSED for the probe (resets would mask the result)",
              flush=True)

        env.reset()
        for i in range(args.settle_steps):
            env.step(zero)
        flat = stats("settled", 0)

        # Sanity-check the env's OWN fold target before driving anything at it. If this is not
        # ~+thickness above the stationary half, the criterion is wrong and nothing downstream
        # means anything.
        _parts = inner._particles_w()
        _end = inner._folded_w(inner._mv_folded_rest)
        _dz = float(_end[..., 2].mean() - _parts[:, inner._stationary_idx, 2].mean()) * 1000
        print(f"[probe] fold TARGET sits {_dz:+.1f} mm above the stationary half "
              f"(thickness = {c.thickness*1000:.0f} mm) -> "
              f"{'OK' if abs(_dz - c.thickness*1000) < 2 else 'UNEXPECTED'}", flush=True)

        # ---- drive the flap over, kinematically -------------------------------------------
        # Endpoints from the env's own definition so the probe cannot disagree with the criterion.
        start = inner._particles_w()[:, inner._moving_idx, :].clone()
        end = inner._folded_w(inner._mv_folded_rest)                       # (N, M, 3)
        # World +Z, deliberately, NOT the fitted sheet normal. The first version used
        # `r[:, :, 2]` and drove the flap DOWNWARD through the table (gap -32.7 mm): the Kabsch
        # frame's third column is only "up" if the fit happens to be oriented that way, and its
        # sign was never checked. The sheet lies flat on a level table, so world +Z is unambiguous.
        normal = torch.tensor([0.0, 0.0, 1.0], device=dev).view(1, 1, 3)
        n_particles = inner._particles_w().shape[1]

        for i in range(args.fold_steps):
            t = (i + 1) / args.fold_steps
            # Straight lerp would drag the flap THROUGH the stationary half. Add a normal-direction
            # bump peaking mid-sweep so it travels over the top, like a real fold.
            arc = torch.sin(torch.tensor(t * 3.14159265, device=dev)) * (0.5 * c.size)
            pos = (1 - t) * start + t * end + arc * normal
            tgt = torch.zeros((inner.num_envs, n_particles, 4), device=dev)
            tgt[..., :3] = inner._particles_w()
            tgt[..., 3] = 1.0                                              # free by default
            tgt[:, inner._moving_idx, :3] = pos
            tgt[:, inner._moving_idx, 3] = 0.0                             # pinned = kinematic
            inner._cloth.write_nodal_kinematic_target_to_sim_index(tgt, None)
            env.step(zero)
        stats("driven", args.fold_steps)

        # ---- release: every particle free again. THIS is the measurement -------------------
        free = torch.zeros((inner.num_envs, n_particles, 4), device=dev)
        free[..., :3] = inner._particles_w()
        free[..., 3] = 1.0
        inner._cloth.write_nodal_kinematic_target_to_sim_index(free, None)

        hist = []
        for i in range(args.release_steps):
            env.step(zero)
            if (i + 1) % 20 == 0 or i < 3:
                hist.append(stats("release", i + 1))

        settled = hist[-1]
        print(f"\n[probe] === {args.label} SETTLED ===", flush=True)
        print(f"  flat-sheet fold_err   : {flat['fold_err']:.4f} m", flush=True)
        print(f"  settled fold_err      : {settled['fold_err']:.4f} m  "
              f"(tolerance {c.keypoint_tolerance})", flush=True)
        print(f"  settled footprint     : {settled['footprint']:.3f}  "
              f"(max {c.max_folded_footprint})", flush=True)
        print(f"  envs scoring folded   : {settled['folded']}/{inner.num_envs}", flush=True)
        print(f"  flap-above-stationary : {settled['gap_mm']:.1f} mm  "
              f"(target assumes {c.thickness * 1000:.0f} mm)", flush=True)
        print(f"  hinge lift            : {settled['hinge_lift_mm']:.1f} mm  "
              f"(idealised model assumes 0)", flush=True)
        print(f"  per-keypoint err (mm) : {settled['kp_err_mm']}", flush=True)
        env.close()

    run()


if __name__ == "__main__":
    main()
