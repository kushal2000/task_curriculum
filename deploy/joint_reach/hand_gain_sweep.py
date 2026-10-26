"""Measure the Wuji hand's Nm -> A gain conversion: replay the sim's recorded joint targets open loop
(run_real.py --mode replay, no policy) at several --hand_kp_per_nm, and compare each run's hand joint
trajectory with the sim's for the same targets.

The firmware takes kp in A/rad, the sim in Nm/rad, and the hand's torque constant is not identified.
The conversion whose replay best matches the sim trajectory is the one to deploy with.

    # sweep (the arm stays fake by default: only the hand moves)
    python deploy/joint_reach/hand_gain_sweep.py --sequence <seq>.pt --hand <HAND_SN> --kp_per_nm 2 3 4 6 \
        --hand_disabled_joints 3 17 --num_goals 3

    # only analyse existing replay logs
    python deploy/joint_reach/hand_gain_sweep.py --sequence <seq>.pt --logs deploy/logs/jr_replay_*.npz

Per joint, for each run:
  dev   mean |q_real - q_sim| over the run: how far the hand's trajectory is from the sim's (main metric)
  track mean |target - q| one step after the target is sent (sim: ~0.5 deg)
  lag   the shift (policy steps of 16.7 ms) of q_real that best matches q_sim (sim: 0)
kd follows kp so the damping ratio stays the sim's (kd_per_nm = sqrt(2) * sqrt(kp_per_nm / 2)) unless
--kd_per_nm is given.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.joint_reach.core import DEFAULT_SPEC, REPO, Spec  # noqa: E402

MAX_LAG = 30


def analyse(path: Path, spec: Spec, sim_q: np.ndarray, sim_target: np.ndarray) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    a, na = meta["args"], spec.n_arm
    if a["mode"] != "replay":
        print(f"[sweep] warning: {path.name} is a {a['mode']} run; its targets are not the sim's")
    off = set(a.get("hand_disabled_joints") or [])
    joints = [j for j in range(spec.num_joints - na) if j not in off]
    q = d["obs_q"][:, na:]
    tgt = d["target"][:, na:]
    T = len(q) - 1  # real q after target[t] is obs_q[t + 1]
    q_after, sq = q[1:], sim_q[:T, na:]
    dev = np.degrees(np.abs(q_after - sq).mean(0))
    track = np.degrees(np.abs(tgt[:T] - q_after).mean(0))
    lag = np.array([min(range(MAX_LAG), key=lambda s: np.abs(q_after[s:, j] - sq[:T - s, j]).mean())
                    for j in range(q.shape[1])])
    reached = sum(r["reached"] for r in meta["results"])
    return {"path": path, "kp": a["hand_kp_per_nm"], "kd": a["hand_kd_per_nm"], "effort": a["hand_effort_limit"],
            "joints": joints, "dev": dev, "track": track, "lag": lag, "reached": reached,
            "scored": len(meta["results"]), "steps": T,
            "sim_track": np.degrees(np.abs(sim_target[:T, na:] - sq).mean(0))}


def report(runs: list[dict], spec: Spec) -> None:
    na = spec.n_arm
    runs = sorted(runs, key=lambda r: r["kp"])
    joints = sorted(set.intersection(*(set(r["joints"]) for r in runs)))
    names = [spec.joint_names[na + j].removeprefix("l_").replace("_finger", "") for j in joints]
    head = " ".join(f"{'kp x' + format(r['kp'], 'g'):>17s}" for r in runs)
    print("\n[sweep] per joint: dev (mean |q_real - q_sim| deg) / lag (steps); sim's own track error in ()")
    print(f"{'joint':20s} {'(sim)':>6s} {head}")
    for j, name in zip(joints, names):
        cells = " ".join(f"{r['dev'][j]:9.1f} / {r['lag'][j]:2d} st" for r in runs)
        print(f"{name:20s} {runs[0]['sim_track'][j]:6.1f} {cells}")
    print("\n[sweep] summary (hand joints not disabled)")
    print(f"{'kp_per_nm':>9s} {'kd_per_nm':>9s} {'effort A':>8s} {'dev mean':>8s} {'dev max':>7s} "
          f"{'track':>6s} {'lag med':>7s} {'reached':>7s}  log")
    for r in runs:
        dv, tr, lg = r["dev"][joints], r["track"][joints], r["lag"][joints]
        print(f"{r['kp']:9.2f} {r['kd']:9.3f} {r['effort']:8.2f} {dv.mean():8.2f} {dv.max():7.1f} "
              f"{tr.mean():6.2f} {np.median(lg):7.0f} {r['reached']:3d}/{r['scored']:<3d}  {r['path']}")
    best = min(runs, key=lambda r: r["dev"][joints].mean())
    print(f"\n[sweep] closest to sim: --hand_kp_per_nm {best['kp']:g} --hand_kd_per_nm {best['kd']:.3f} "
          f"(dev {best['dev'][joints].mean():.2f} deg; sim track error {runs[0]['sim_track'][joints].mean():.2f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence", required=True, help="real_test_sequence.pt (has the sim trajectory)")
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--logs", nargs="*", default=None, help="analyse these replay logs instead of running")
    p.add_argument("--hand", help="hand serial or 'auto'")
    p.add_argument("--arm", default="fake", help="'fake' (default: the arm is not touched) or the robot serial")
    p.add_argument("--kp_per_nm", type=float, nargs="+", default=[2.0, 3.0, 4.0, 6.0])
    p.add_argument("--kd_per_nm", type=float, nargs="+", default=None,
                   help="one per kp; default keeps the damping ratio: sqrt(2) * sqrt(kp_per_nm / 2)")
    p.add_argument("--hand_effort_limit", type=float, default=0.5)
    p.add_argument("--hand_disabled_joints", type=int, nargs="*", default=[])
    p.add_argument("--num_goals", type=int, default=3)
    p.add_argument("--viser_port", type=int, default=0)
    p.add_argument("--log_dir", default="deploy/logs/hand_gain_sweep")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # interleave with the run_real subprocesses' output

    import torch

    spec = Spec.load(args.spec)
    rec = torch.load(args.sequence, map_location="cpu", weights_only=False)
    sim_q = rec["sim_trajectory"]["q"].double().numpy()
    sim_target = rec["sim_trajectory"]["target"].double().numpy()

    if args.logs is not None:
        paths = [Path(x) for x in args.logs]
    else:
        if not args.hand:
            p.error("--hand is required to run the sweep")
        kd = args.kd_per_nm or [math.sqrt(2.0) * math.sqrt(k / 2.0) for k in args.kp_per_nm]
        if len(kd) != len(args.kp_per_nm):
            p.error("--kd_per_nm needs one value per --kp_per_nm")
        na = spec.n_arm
        print(f"[sweep] {len(kd)} replay runs of {args.num_goals} goals, arm {args.arm}, hand {args.hand}, "
              f"effort limit {args.hand_effort_limit} A")
        for k, c in zip(args.kp_per_nm, kd):
            kp_a, kd_a = spec.stiffness[na:] * k, spec.damping[na:] * c
            print(f"[sweep]   kp x{k:g}: {kp_a.min():.2f}-{kp_a.max():.2f} A/rad, "
                  f"kd x{c:.3f}: {kd_a.min():.4f}-{kd_a.max():.4f} A s/rad")
        live = args.arm != "fake" or args.hand not in ("fake", "dry_run")
        if live and not args.yes and input("[sweep] hardware will move. Type 'go' to start: ").strip() != "go":
            print("[sweep] not started")
            return 1
        out = Path(args.log_dir)
        paths = []
        for k, c in zip(args.kp_per_nm, kd):
            run_dir = out / f"kp{k:g}"
            before = set(run_dir.glob("*.npz"))
            cmd = [sys.executable, str(REPO / "deploy/joint_reach/run_real.py"), "--mode", "replay", "--yes",
                   "--ckpt", "-", "--sequence", args.sequence, "--spec", args.spec,
                   "--arm", args.arm, "--hand", args.hand, "--num_goals", str(args.num_goals),
                   "--hand_kp_per_nm", str(k), "--hand_kd_per_nm", str(c),
                   "--hand_effort_limit", str(args.hand_effort_limit), "--log_dir", str(run_dir),
                   "--hand_disabled_joints", *map(str, args.hand_disabled_joints)]
            if args.viser_port:
                cmd += ["--viser_port", str(args.viser_port)]
            print(f"\n[sweep] ---- kp x{k:g}, kd x{c:.3f}")
            code = subprocess.call(cmd, cwd=REPO)
            new = sorted(set(run_dir.glob("*.npz")) - before)
            if new:
                paths.append(new[-1])
            if code != 0:
                print(f"[sweep] run exited with {code}; stopping the sweep")
                break
    if not paths:
        print("[sweep] no logs")
        return 1
    report([analyse(x, spec, sim_q, sim_target) for x in paths], spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
