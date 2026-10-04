"""Does the rigid-cloth env actually build, step, and score a fold in the REAL stack?

Everything in ``rigid_cloth_task_check.py`` was measured without Isaac Sim: it drove each chain in
MuJoCo and pushed the resulting body poses through the adapter's arithmetic. That proves the
geometry and the scoring, and proves nothing about the stack the training run uses. Four things
only exist once Isaac Lab, Newton and MJWarp are in the loop, and each of them fails in a way that
looks like something else:

  1. **the joint limits.** They are the reason two plies cannot interpenetrate. Written in the URDF,
     they have to survive URDF -> USD -> Isaac Lab -> Newton -> mjModel. If they are dropped, the
     chain folds to zero thickness and *scores better*, which reads as a policy that solved the task.
  2. **slat-vs-slat self-collision.** Same failure, same disguise. ``self_collision=True`` is set on
     the conversion, but Newton also filters shapes that share a joint, so the surviving pairs have
     to be counted, not assumed.
  3. **the adapter against a live articulation.** ``body_pos_w`` is not in ``patches.TORCH_ATTRS``;
     the fallback to slicing ``body_state_w`` has a unit test and has never met the real solver.
  4. **contact buffer sizing.** ``njmax``/``nconmax`` in ``RigidCloth.yaml`` are inherited from a
     config whose manipuland contributed *particles*. A 17-body chain contributes contacts, and an
     overflow silently drops them.

So this builds the env, prints what the solver actually received, settles the sheet flat, then drives
the hinges to ``chain_fold_angles`` and lets the fold settle under gravity -- and reports
``fold_error`` from the env's own method. The MuJoCo check said every variant lands under 0.004 m
against a 0.04 m tolerance; if this disagrees, the difference is in the stack, not the geometry.

Also times the step loop, which is the number the whole exercise is motivated by. Pass
``--task Isaacsimenvs-Cloth-Direct-v0`` to get the VBD sheet's figure under an identical protocol;
the fold drive is skipped there (it has no hinges) and only the flat + timing sections run.

    scripts/newton_py -m scripts.analysis.rigid_cloth_env_probe --variant box3-mid --num_envs 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _model_report(env, spec) -> dict:
    """What MJWarp was actually handed for the chain: limits, damping, self-collision, friction.

    Read off ``newton.Model`` rather than the config, because every interesting failure here is a
    value that was configured correctly and then lost in translation.
    """
    import numpy as np

    model = env.sim.physics_manager.get_model()
    out: dict = {}

    def short(labels):
        return [str(s).rsplit("/", 1)[-1] for s in labels]

    joint_labels = short(model.joint_label)
    body_labels = short(model.body_label)

    # --- joint limits. Per-DOF arrays; a revolute hinge has one DOF, so index by dof start.
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    damping = model.joint_damping.numpy()
    dof_start = model.joint_qd_start.numpy() if hasattr(model, "joint_qd_start") else None

    hinge_rows = []
    for j, label in enumerate(joint_labels):
        if not label.startswith("slat_joint_"):
            continue
        d = int(dof_start[j]) if dof_start is not None else j
        if d >= len(lower):
            continue
        hinge_rows.append(
            {
                "joint": label,
                "lower_deg": round(math.degrees(float(lower[d])), 2),
                "upper_deg": round(math.degrees(float(upper[d])), 2),
                "damping": float(damping[d]) if d < len(damping) else None,
            }
        )
    out["hinges"] = hinge_rows
    out["num_hinges"] = len(hinge_rows)
    out["expected_hinges"] = spec.num_slats - 1 if spec is not None else None

    # A hinge whose range is the full circle has effectively no limit. That is the silent failure:
    # the URDF said +-90 deg and the plies would pass through each other.
    span = [r["upper_deg"] - r["lower_deg"] for r in hinge_rows]
    out["limit_span_deg_min"] = round(min(span), 2) if span else None
    out["limit_span_deg_max"] = round(max(span), 2) if span else None
    out["unlimited_hinges"] = sum(1 for s in span if s > 359.0)

    # --- self-collision among slats. Count the slat-slat pairs that are NOT filtered, within one
    # env. Newton filters shapes joined by a joint, so adjacent slats are expected to be filtered and
    # i vs i+2 -- the pair that stops the fold -- must not be.
    shape_body = model.shape_body.numpy()
    slat_shapes: dict[int, list[int]] = {}
    for s, b in enumerate(shape_body):
        b = int(b)
        if b < 0 or not body_labels[b].startswith("slat_"):
            continue
        slat_shapes.setdefault(b, []).append(s)

    filtered = set()
    try:
        pairs = model.shape_collision_filter_pairs
        arr = pairs.numpy() if hasattr(pairs, "numpy") else np.asarray(list(pairs))
        for a, b in np.asarray(arr).reshape(-1, 2):
            filtered.add((int(a), int(b)))
            filtered.add((int(b), int(a)))
    except Exception as exc:  # pragma: no cover - diagnostic only
        out["filter_pairs_error"] = repr(exc)

    bodies = sorted(slat_shapes)
    # Only the first env's slats, so the count is per-sheet and comparable across --num_envs.
    per_env = [b for b in bodies if body_labels[b].startswith("slat_")][: spec.num_slats]
    unfiltered_gap2 = 0
    filtered_adjacent = 0
    for k, b in enumerate(per_env):
        for other in per_env[k + 1 :]:
            gap = per_env.index(other) - k
            pairset = [
                (sa, sb) for sa in slat_shapes[b] for sb in slat_shapes[other]
            ]
            any_live = any(p not in filtered for p in pairset)
            if gap == 1 and not any_live:
                filtered_adjacent += 1
            if gap == 2 and any_live:
                unfiltered_gap2 += 1
    out["slat_bodies"] = len(per_env)
    out["slat_shapes"] = sum(len(slat_shapes[b]) for b in per_env)
    out["filtered_adjacent_pairs"] = filtered_adjacent
    out["live_gap2_pairs"] = unfiltered_gap2
    out["num_filter_pairs"] = len(filtered) // 2

    # --- friction actually on the slats.
    mu = model.shape_material_mu.numpy()
    slat_mu = sorted({round(float(mu[s]), 4) for b in per_env for s in slat_shapes[b]})
    out["slat_mu"] = slat_mu

    out["total_bodies"] = len(body_labels)
    out["total_shapes"] = len(shape_body)
    return out


def _frame_check(env, spec) -> dict:
    """Each slat's ACTUAL world frame against what ``chain_frames`` predicts.

    The decisive check on the adapter, and one no unit test can make: ``test_the_adapter_reproduces_
    the_flat_sheet`` builds its fake articulation *from* ``chain_frames``, so it verifies the mapping
    is self-consistent and cannot detect ``chain_frames`` disagreeing with the imported USD. The
    MuJoCo probe reads real body poses but from the URDF directly, which is a different importer.

    A mismatch here misattributes the whole grid: the emulated cloud measured 47 mm across a 100 mm
    sheet, which reads as a crumpled sheet rather than as a mapping error.
    """
    import torch

    from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc
    from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch

    chain = env._chain
    names = list(chain.body_names)
    state = _to_torch(chain.data.body_state_w)
    pos = state[..., 0:3]
    origin = env.scene.env_origins
    want = rc.chain_frames(spec, None)

    rows = []
    worst = 0.0
    print(f"[frames] {'slat':<10} {'want_x_mm':>10} {'got_x_mm':>10} {'err_mm':>8} "
          f"{'want_z_mm':>10} {'got_z_mm':>10}", flush=True)
    for i in range(spec.num_slats):
        name = rc.slat_link_name(i)
        if name not in names:
            continue
        b = names.index(name)
        got = pos[0, b, :] - origin[0]
        wx, wz, _a = want[i]
        err = abs(float(got[0]) - wx)
        worst = max(worst, err)
        rows.append({"slat": name, "want_x": wx, "got_x": float(got[0]), "err": err})
        if i < 6 or i == spec.num_slats - 1:
            print(
                f"[frames] {name:<10} {wx * 1e3:>10.2f} {float(got[0]) * 1e3:>10.2f} "
                f"{err * 1e3:>8.2f} {wz * 1e3:>10.2f} {float(got[2]) * 1e3:>10.2f}",
                flush=True,
            )
    print(f"[frames] worst x error {worst * 1e3:.2f} mm over {len(rows)} slats", flush=True)

    # WHICH accessors exist, and what convention their quaternions are in. The cloud came out 47 mm
    # wide across a 100 mm sheet, and that is exactly what a 180 deg z-rotation of every local offset
    # predicts -- i.e. an identity quaternion read in the wrong component order. Guessing wxyz vs
    # xyzw is how that happened; this prints the raw numbers instead.
    d = chain.data
    have = {
        k: hasattr(d, k)
        for k in ("body_pos_w", "body_quat_w", "body_link_pose_w", "body_state_w", "root_pose_w")
    }
    print(f"[quat] accessors: {have}", flush=True)
    q_state = state[0, 0, 3:7].tolist()
    print(f"[quat] body_state_w[0,0,3:7] = {[round(v, 4) for v in q_state]}", flush=True)
    for k in ("body_quat_w", "body_link_pose_w"):
        if have.get(k):
            v = _to_torch(getattr(d, k))
            v = v[0, 0, -4:] if v.ndim == 3 else v[0, -4:]
            print(f"[quat] {k}[0,0] last4 = {[round(float(x), 4) for x in v]}", flush=True)
    root_b = names.index(rc.slat_link_name(spec.root))
    rp = pos[0, root_b, :] - origin[0]
    print(
        f"[quat] root link {rc.slat_link_name(spec.root)} at x = {float(rp[0]) * 1e3:+.2f} mm; "
        f"chain_frames says {want[spec.root][0] * 1e3:+.2f} mm",
        flush=True,
    )
    return {
        "worst_x_err_mm": round(worst * 1e3, 3),
        "slats": rows,
        "accessors": have,
        "rest_quat_body_state_w": [round(v, 5) for v in q_state],
    }


def _placement_check(env, spec) -> dict:
    """Measure the map from "requested sheet centre" to "where the sheet actually lands".

    Derivation said the reset was right and measurement said it was off by a constant, so this stops
    arguing and measures. It asks the object for a known placement -- sheet centred on the env origin,
    zero yaw -- and reports where the emulated cloud's centroid actually ends up. Anything but zero is
    the correction the root offset is missing, and its sign and magnitude identify the cause.

    Worth having permanently: the reset is the one piece of this adapter whose failure is *silent*.
    A displaced sheet still folds, still scores, and still trains -- against a fold target built
    around the displaced position -- so nothing downstream reports it.
    """
    import torch

    art_obj = env.object
    n = env.num_envs
    want = env.scene.env_origins.clone()
    pose = torch.zeros((n, 7), device=env.device, dtype=torch.float32)
    pose[:, :3] = want
    pose[:, 3] = 1.0  # identity quat -> zero yaw
    art_obj.write_root_pose_to_sim(pose)
    art_obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device))
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)

    # What we asked the articulation for, vs what it reports back. If these agree, the displacement is
    # in the importer's child layout; if they disagree, it is in the root write. Those need different
    # fixes, so the distinction is worth one readback.
    from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch as _t

    want_root_x = float(pose[0, 0]) + float(art_obj._root_offset[0]) - float(env.scene.env_origins[0, 0])
    rp = _t(env._chain.data.root_pose_w)
    got_root_x = float(rp[0, 0]) - float(env.scene.env_origins[0, 0])
    print(
        f"[place] root pose: we wrote x={want_root_x * 1e3:+.2f} mm (centre + offset), "
        f"articulation reports root_pose_w x={got_root_x * 1e3:+.2f} mm, "
        f"quat={[round(float(v), 3) for v in rp[0, 3:7]]}",
        flush=True,
    )

    cloud = env._particles_w()
    centroid = cloud.mean(dim=1)
    delta = (centroid - want)[0]
    print(
        f"[place] requested centre (env frame) 0.00, 0.00 -> got "
        f"{float(delta[0]) * 1e3:+.2f}, {float(delta[1]) * 1e3:+.2f} mm "
        f"(x error {float(delta[0]) * 1e3:+.2f} mm)",
        flush=True,
    )
    fx_root = rc_chain_root_x(spec)
    print(
        f"[place] chain_frames root x = {fx_root * 1e3:+.2f} mm; "
        f"-2 x root = {-2 * fx_root * 1e3:+.2f} mm",
        flush=True,
    )
    return {
        "centre_err_x_mm": round(float(delta[0]) * 1e3, 3),
        "centre_err_y_mm": round(float(delta[1]) * 1e3, 3),
        "root_x_mm": round(fx_root * 1e3, 3),
    }


def rc_chain_root_x(spec) -> float:
    from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc

    return float(rc.chain_frames(spec, None)[spec.root][0])


def _fold_the_chain(env, spec, steps: int) -> None:
    """Drive the hinges to the folded configuration and let gravity settle it.

    Kinematic, not a policy: the question is whether a *physically settled* fold satisfies the env's
    criterion, and a policy that cannot fold would leave that unanswered. The joints are passive
    (``joint_stiffness = 0``), so nothing holds the pose -- it stays folded only because the moving
    ply comes to rest on the stationary one, which is the contact this is here to test.
    """
    import torch

    from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc

    chain = env._chain
    angles = rc.chain_fold_angles(spec, env.cfg.rigid_cloth.joint_limit)
    names = list(chain.joint_names)
    q = torch.zeros((env.num_envs, len(names)), device=env.device)
    hit = 0
    for i in range(spec.num_slats):
        name = rc.slat_joint_name(i)
        if name in names:
            q[:, names.index(name)] = float(angles[i])
            hit += 1
    if hit == 0:
        raise RuntimeError(
            f"none of the chain's joints matched slat_joint_*; joint_names={names[:8]}"
        )
    nonzero = [(rc.slat_joint_name(i), round(math.degrees(a), 1)) for i, a in enumerate(angles) if a]
    print(f"[probe] driving fold: {nonzero}", flush=True)

    # Through `ChainAsRigidObject`, NOT the articulation directly: the reset writes joints through
    # that path, so a fold probe that bypassed it could pass while every reset silently failed.
    env.object.write_joint_positions(q)
    # Track the BEST fold across the settle window rather than reading the end state.
    #
    # Not a refinement -- a correctness fix. A fold that satisfies the criterion TERMINATES the
    # episode, and the env then resets the chain flat. Reading the end state therefore reported the
    # flat sheet (fold_err 0.10002, hinges 0.03 deg) and looked exactly like "the joint write was
    # silently dropped", which sent the hunt into the write API twice. The minimum is also the
    # statistic `eval/episodes.py` reports as `best_fold_err`, so this measures the same thing the
    # evaluation does.
    # Settle with RAW sim steps, not `env.step`. A fold that satisfies the criterion terminates the
    # episode, and `env.step` then resets the chain flat before anything can be sampled -- so every
    # reading came back as the flat sheet (fold_err 0.10002, hinges 0.07 deg) and looked exactly like
    # a joint write that had been silently dropped. This is a physics probe; the episode machinery is
    # measured by the evaluation, not here.
    from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch

    dt = float(env.cfg.sim.dt)
    best = torch.full((env.num_envs,), float("inf"), device=env.device)
    best_hinge = 0.0
    for _ in range(steps):
        env.sim.step(render=False)
        env.scene.update(dt)
        best = torch.minimum(best, env.fold_error())
        best_hinge = max(best_hinge, math.degrees(float(_to_torch(chain.data.joint_pos).abs().max())))
    print(
        f"[probe] fold drive: best fold_err {float(best.mean()):.5f} "
        f"(min {float(best.min()):.5f}), peak hinge {best_hinge:.2f} deg over {steps} steps",
        flush=True,
    )
    return {
        "best_fold_error_mean": round(float(best.mean()), 5),
        "best_fold_error_min": round(float(best.min()), 5),
        "peak_hinge_deg": round(best_hinge, 2),
    }


def _trace(env, inner, act, steps: int) -> None:
    """Per-step chain state, to separate a contact explosion from a bad spawn.

    Reports the ARTICULATION's own quantities rather than the emulated particle cloud, because the
    cloud is a function of them: if ``joint_qd`` is already 1e3 at step 2 the fault is in the solve,
    and if the cloud is wrong while the bodies are sane the fault is in the adapter.
    """
    import torch

    # The adapter's own converter. Newton hands out warp-backed ProxyArrays for some fields and real
    # cuda tensors for others, and hand-rolling the conversion here got it wrong (`.numpy()` on a
    # cuda tensor). One definition, already used on the path under test.
    from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch

    chain = inner._chain
    print(f"[trace] {'step':>4} {'|pos|max':>9} {'z_min':>8} {'|v|max':>9} "
          f"{'|q|max_deg':>10} {'|qd|max':>10} {'finite':>6}", flush=True)
    for k in range(steps):
        d = chain.data
        state = _to_torch(d.body_state_w)
        pos, vel = state[..., 0:3], state[..., 7:10]
        q = _to_torch(d.joint_pos)
        qd = _to_torch(d.joint_vel)
        finite = bool(
            torch.isfinite(pos).all() and torch.isfinite(q).all() and torch.isfinite(qd).all()
        )
        fin = torch.isfinite(pos)
        print(
            f"[trace] {k:>4} {float(pos[fin].abs().max()):>9.3f} "
            f"{float(pos[..., 2][torch.isfinite(pos[..., 2])].min()):>8.4f} "
            f"{float(vel[torch.isfinite(vel)].abs().max()):>9.3f} "
            f"{math.degrees(float(q[torch.isfinite(q)].abs().max())):>10.2f} "
            f"{float(qd[torch.isfinite(qd)].abs().max()):>10.2f} {str(finite):>6}",
            flush=True,
        )
        env.step(act)


def _snapshot(env, label: str) -> dict:
    """The env's own fold quantities, plus the guards that say whether they mean anything."""
    import torch

    parts = env._particles_w()
    fe = env.fold_error()
    fp = env.footprint_ratio()
    obj_z = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
    # The chain's OWN configuration, so "the sheet is crumpled" can be told apart from "the sheet is
    # flat and my metric is wrong". A flat chain has every hinge at 0 and a cloud 100 mm wide.
    extent = parts[..., 0].amax(dim=-1) - parts[..., 0].amin(dim=-1)
    extent_y = parts[..., 1].amax(dim=-1) - parts[..., 1].amin(dim=-1)
    joints = None
    chain = getattr(env, "_chain", None)
    if chain is not None:
        from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch

        joints = _to_torch(chain.data.joint_pos)
    row = {
        "cloud_extent_x_mm": round(float(extent.mean()) * 1e3, 2),
        "cloud_extent_y_mm": round(float(extent_y.mean()) * 1e3, 2),
        "hinge_abs_deg_mean": round(math.degrees(float(joints.abs().mean())), 2)
        if joints is not None
        else None,
        "hinge_abs_deg_max": round(math.degrees(float(joints.abs().max())), 2)
        if joints is not None
        else None,
        "label": label,
        "fold_error_mean": round(float(fe.mean()), 5),
        "fold_error_min": round(float(fe.min()), 5),
        "fold_error_max": round(float(fe.max()), 5),
        "footprint_ratio_mean": round(float(fp.mean()), 4),
        "object_z_mean": round(float(obj_z.mean()), 4),
        "particles_finite": bool(torch.isfinite(parts).all()),
        "particle_z_min": round(float(parts[..., 2].min()), 4),
        "particle_z_max": round(float(parts[..., 2].max()), 4),
    }
    print(
        f"[probe] {label:<8} fold_err {row['fold_error_mean']:.5f} "
        f"(min {row['fold_error_min']:.5f}) footprint {row['footprint_ratio_mean']:.3f} "
        f"obj_z {row['object_z_mean']:.4f} finite={row['particles_finite']} | "
        f"cloud {row['cloud_extent_x_mm']:.1f} x {row['cloud_extent_y_mm']:.1f} mm "
        f"hinges |{row['hinge_abs_deg_mean']}| deg mean, |{row['hinge_abs_deg_max']}| max",
        flush=True,
    )
    return row


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Build, step and fold-test the rigid cloth env.")
    parser.add_argument("--task", default="Isaacsimenvs-RigidCloth-Direct-v0")
    parser.add_argument("--variant", default=None, help="env.rigid_cloth.variant override")
    parser.add_argument("--armature", type=float, default=None, help="rigid_cloth.joint_armature")
    parser.add_argument("--damping", type=float, default=None, help="rigid_cloth.joint_damping")
    parser.add_argument(
        "--start_height",
        type=float,
        default=None,
        help="cloth.start_height override. The cloth's 0.30 is a 15 cm DROP, calibrated so a "
        "particle sheet lands perfectly flat. A limp chain of rigid slats buckles on the way down "
        "instead (measured: footprint 0.47 at rest, i.e. already folded in half), so it needs its "
        "own value. The table surface is at table_half_thickness = 0.150, so 0.151 is a 1 mm drop.",
    )
    parser.add_argument("--stiffness", type=float, default=None, help="rigid_cloth.joint_stiffness")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--settle_steps", type=int, default=60, help="flat settle before folding")
    parser.add_argument("--fold_steps", type=int, default=120, help="settle after driving the fold")
    parser.add_argument("--timing_steps", type=int, default=200)
    parser.add_argument(
        "--trace_steps",
        type=int,
        default=0,
        help="Print the chain's own state for this many steps after reset. The blow-up happens "
        "within 2-7 steps, i.e. during FREE FALL before the sheet reaches the table, so the "
        "per-step numbers are what distinguish a contact explosion from a bad spawn.",
    )
    parser.add_argument(
        "--nan_policy",
        default="raise",
        choices=("raise", "reset"),
        help="`raise` localises a divergence (the guard names the first bad quantity); `reset` "
        "keeps stepping so a trace can run past the first failure.",
    )
    parser.add_argument("--tolerance", type=float, default=0.04)
    # NOT `--device`: `AppLauncher.add_app_launcher_args` adds that itself and raises on a clash.
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.variant:
        hydra_args = hydra_args + [f"env.rigid_cloth.variant={args.variant}"]
    if args.armature is not None:
        hydra_args = hydra_args + [f"env.rigid_cloth.joint_armature={args.armature}"]
    if args.damping is not None:
        hydra_args = hydra_args + [f"env.rigid_cloth.joint_damping={args.damping}"]
    if args.stiffness is not None:
        hydra_args = hydra_args + [f"env.rigid_cloth.joint_stiffness={args.stiffness}"]
    if args.start_height is not None:
        hydra_args = hydra_args + [f"env.cloth.start_height={args.start_height}"]
    sys.argv = [sys.argv[0]] + hydra_args

    try:
        app = AppLauncher(args).app
    except ImportError as exc:
        app = None
        print(f"[probe] kit-less: {exc}", flush=True)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.newton.contact_guard import assert_no_buffer_overflow
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    result: dict = {"task": args.task, "variant": args.variant, "num_envs": args.num_envs}

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.sim_device
        env_cfg.seed = args.seed
        env_cfg.cloth.nan_policy = args.nan_policy
        disable_randomization(env_cfg)
        use_single_object_variant()

        t0 = time.perf_counter()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        result["build_s"] = round(time.perf_counter() - t0, 2)
        print(f"[probe] built in {result['build_s']:.1f} s on {inner.device}", flush=True)

        spec = getattr(inner.cfg, "rigid_cloth", None)
        spec = spec.spec if spec is not None else None
        result["is_chain"] = spec is not None
        result["start_height"] = float(inner.cfg.cloth.start_height)
        if spec is not None:
            result["spec"] = spec.describe()
            result["armature"] = float(inner.cfg.rigid_cloth.joint_armature)
            result["damping"] = float(inner.cfg.rigid_cloth.joint_damping)
            result["stiffness"] = float(inner.cfg.rigid_cloth.joint_stiffness)
            result["model"] = _model_report(inner, spec)
            m = result["model"]
            # Print the COUNTS, not the per-hinge list: the list is 17 entries x 5 lines and
            # truncating the dump hid exactly the numbers this probe exists to report.
            print(
                "[probe] model: "
                f"hinges {m['num_hinges']}/{m['expected_hinges']} "
                f"limits [{m['limit_span_deg_min']}, {m['limit_span_deg_max']}] deg "
                f"unlimited {m['unlimited_hinges']} | "
                f"slat bodies {m['slat_bodies']} shapes {m['slat_shapes']} mu {m['slat_mu']} | "
                f"filter pairs {m['num_filter_pairs']} "
                f"adjacent-filtered {m['filtered_adjacent_pairs']}/{max(m['slat_bodies'] - 1, 0)} "
                f"live-gap2 {m['live_gap2_pairs']} | "
                f"bodies {m['total_bodies']} shapes {m['total_shapes']}",
                flush=True,
            )
            if m["filtered_adjacent_pairs"] < max(m["slat_bodies"] - 1, 0):
                print(
                    "[probe] WARNING adjacent slats are NOT collision-filtered. Two slats sharing a "
                    "hinge touch at the seam with zero gap, so contact pushes them apart while the "
                    "joint pulls them back -- which diverges in a few steps. MuJoCo excludes "
                    "parent-child pairs by default, which is why the MuJoCo probe never saw this.",
                    flush=True,
                )

        env.reset()
        # Contacts that only appear once things touch are invisible to a zero-action warm-up, so this
        # is a floor, not a guarantee; the rollout below is where an overflow would actually show.
        assert_no_buffer_overflow(env)
        env.reset()

        if spec is not None:
            result["frames"] = _frame_check(inner, spec)
            result["placement"] = _placement_check(inner, spec)

        act = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=inner.device)
        if args.trace_steps:
            _trace(env, inner, act, args.trace_steps)
        for _ in range(max(args.settle_steps - args.trace_steps, 0)):
            env.step(act)
        result["flat"] = _snapshot(inner, "flat")
        # A flat sheet must score ~0.100 (the whole sheet width from a fold) at footprint ~1.0. A
        # chain that buckled during the drop reports a SMALLER fold_error while lying crumpled, which
        # would read as partial credit for a fold that never happened -- and would leave the policy a
        # fraction of the intended signal to climb. So this is checked, not just printed.
        result["rests_flat"] = bool(
            result["flat"]["footprint_ratio_mean"] > 0.95
            and result["flat"]["fold_error_mean"] > 0.09
        )
        print(
            f"[probe] rests flat: {result['rests_flat']} "
            f"(footprint {result['flat']['footprint_ratio_mean']:.3f} want >0.95, "
            f"flat fold_err {result['flat']['fold_error_mean']:.4f} want >0.09)",
            flush=True,
        )
        # The stability headline. `_nonfinite_count` is latched by the env's own guard, so it counts
        # every env that diverged at any point, not just those bad right now -- a chain that blows up
        # and is reset looks perfectly healthy in a snapshot.
        result["nonfinite_envs_total"] = int(getattr(inner, "_nonfinite_count", 0))
        result["stable"] = result["nonfinite_envs_total"] == 0
        print(
            f"[probe] stability: {result['nonfinite_envs_total']} env-divergences over "
            f"{args.settle_steps} steps x {args.num_envs} envs "
            f"(armature {result.get('armature')}, damping {result.get('damping')})",
            flush=True,
        )

        if spec is not None:
            result["fold_drive"] = _fold_the_chain(inner, spec, args.fold_steps)
            result["folded"] = _snapshot(inner, "folded")
            # The BEST fold reached, not the state at the end of the window: a successful fold ends
            # the episode and the env resets the chain flat, so the end state is the flat sheet.
            fe = result["fold_drive"]["best_fold_error_mean"]
            result["fold_solvable"] = bool(fe < args.tolerance)
            result["margin_mm"] = round((args.tolerance - fe) * 1e3, 1)
            print(
                f"[probe] fold {'SOLVABLE' if result['fold_solvable'] else 'FAILS'} "
                f"at tolerance {args.tolerance}: margin {result['margin_mm']} mm",
                flush=True,
            )
            env.reset()

        # --- throughput. After a reset so the timing is not paid for the fold's contact pile-up.
        for _ in range(20):
            env.step(act)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(args.timing_steps):
            env.step(act)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall = time.perf_counter() - t0
        result["timing"] = {
            "policy_steps": args.timing_steps,
            "wall_s": round(wall, 3),
            "policy_steps_per_s": round(args.timing_steps / wall, 2),
            "env_steps_per_s": round(args.timing_steps * inner.num_envs / wall, 1),
            "sim_substeps_per_s": round(
                args.timing_steps * inner.num_envs * inner.cfg.decimation / wall, 1
            ),
        }
        print(f"[probe] throughput {result['timing']['env_steps_per_s']:.0f} env-steps/s "
              f"({args.num_envs} envs, zero actions)", flush=True)
        env.close()

    run()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"[probe] wrote {args.out}", flush=True)

    if app is not None:
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
