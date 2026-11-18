"""Pre-training physics check of a training config: two tests over its whole randomization. Report only.

    python isaacsimenvs/scripts/physics_test/preflight.py \\
        --experiment <recipe> --out_dir <dir> [--launch slurm|local] [hydra overrides ...]

1. RESOLVE the training config exactly as ``train.py`` would (same ``hydra_task_config_with_yaml``
   loader: task YAML + experiment recipe + overrides). Hydra writes it to
   ``<out_dir>/resolved/.hydra/config.yaml``, the same file a training run gets.
2. For every design (at the config's panel thickness), sweep the run's randomization -- every corner of the DR box plus ``--dr_random`` random draws, at
   the corners of the size range -- with both tests:
   * PRESS (``press_test.py``): is the design penetration-free under excessive force? A probe
     calibrated against policy rollouts (0.5 kg) presses every panel centre and hinge at 50 N;
     an env fails if the probe goes through or sinks more than ``--max_depth_mm``.
   * FOLD (``fold_test.py``): can the fold be completed by driving the joints (fold hinge to 180 deg,
     all others held flat, base pinned), and does it stay with no force on it? Fails unless
     ``is_folded`` both at the end of the drive and after release.
3. VIDEO: each sweep's worst case is replayed alone and recorded.
4. REPORT: one table per test (% failing per design and thickness, and the parameter ranges the
   failures share), ``<out_dir>/preflight.json``, exit code 1 if anything failed.

Plain Python: Isaac Sim runs only in child processes (Slurm jobs via experiments/physics_test.sh,
or sequential local runs with --launch local).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = str(ROOT / ".venv/isaacsim/bin/python")
HERE = Path(__file__).resolve().parent

#: Calibrated against policy rollouts (strip_4, density 2.0): this probe fails where the policy
#: penetrated (trained solver settings) and holds where it did not (UWLab settings).
PROBE_MASS_KG = 0.5
PRESS_FORCE_N = 50.0


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="Isaacsimenvs-RigidBodyCloth-Direct-v0")
    p.add_argument("--agent", default="rl_games_sapg_cfg_entry_point")
    p.add_argument("--experiment", help="run recipe, as for train.py --experiment")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--launch", choices=("slurm", "local"), default="slurm")
    p.add_argument("--dr_random", type=int, default=32, help="random DR draws per sweep, besides the corners")
    p.add_argument("--max_depth_mm", type=float, default=2.0)
    p.add_argument("--tests", nargs="+", choices=("press", "fold"), default=("press", "fold"))
    p.add_argument("--resolve_only", action="store_true", help=argparse.SUPPRESS)
    return p.parse_known_args()


def resolve(args, overrides) -> None:
    """Child process: let the training loader build the config; Hydra writes .hydra/config.yaml."""
    from isaaclab.app import AppLauncher

    launcher = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher)
    app = AppLauncher(launcher.parse_args(["--headless"])).app
    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml

    sys.argv = [sys.argv[0], *overrides, f"hydra.run.dir={args.out_dir / 'resolved'}"]

    @hydra_task_config_with_yaml(args.task, args.agent, experiment=args.experiment)
    def run(env_cfg, agent_cfg):
        print(f"[preflight] resolved config written to {args.out_dir / 'resolved/.hydra/config.yaml'}", flush=True)

    run()
    app.close()


def thicknesses(env: dict) -> dict[str, float]:
    """The config's panel thickness (one value per run)."""
    return {"config": float(env["rigid_cloth"]["thickness"])}


class Runner:
    """Runs test commands as Slurm jobs (in batches) or locally, one log per run."""

    def __init__(self, launch: str, out: Path):
        self.launch, self.out, self.jobs = launch, out, []

    def add(self, name: str, test: str, test_args: list[str]) -> None:
        if self.launch == "local":
            print(f"[preflight] {name} (local) ...", flush=True)
            with open(self.out / f"{name}.log", "w") as log:
                subprocess.run([PY, str(HERE / f"{test}_test.py"), "--headless", *test_args], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
            return
        sub = subprocess.run([str(ROOT / "experiments/submit.sh"), f"--job-name=pf_{name}",
                              str(ROOT / "experiments/physics_test.sh"), f"{test}_test",
                              "--experiment_dir", str(self.out), *test_args],
                             cwd=ROOT, check=True, capture_output=True, text=True)
        self.jobs.append(sub.stdout.strip().split()[-1])

    def wait(self) -> None:
        if not self.jobs:
            return
        print(f"[preflight] waiting for {len(self.jobs)} jobs: {','.join(self.jobs)}", flush=True)
        while subprocess.run(["squeue", "-h", "-j", ",".join(self.jobs)], capture_output=True, text=True).stdout.strip():
            time.sleep(20)
        self.jobs = []


def main() -> int:
    args, overrides = parse()
    if args.resolve_only:
        resolve(args, overrides)
        return 0

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = out / "resolved/.hydra/config.yaml"
    print("[preflight] resolving the training config ...", flush=True)
    with open(out / "resolve.log", "w") as log:
        subprocess.run([PY, str(Path(__file__).resolve()), "--resolve_only", "--task", args.task, "--agent", args.agent,
                        "--out_dir", str(out), *(["--experiment", args.experiment] if args.experiment else []),
                        *overrides], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    import yaml

    env = yaml.safe_load(config.read_text())["env"]
    designs = [Path(d).stem for d in env["rigid_cloth"]["designs"]]
    thick = thicknesses(env)
    print(f"[preflight] designs {designs}, thicknesses {thick}, tests {list(args.tests)}", flush=True)

    # 1. Sweeps.
    runner = Runner(args.launch, out)
    sweeps = []
    for test in args.tests:
        for d in designs:
            for label, t in thick.items():
                name = f"{test}_{d}_{label}"
                common = ["--config", str(config), "--design", d, "--thickness", str(t),
                          "--dr_random", str(args.dr_random), "--out", str(out / f"{name}.json")]
                if test == "press":
                    common += ["--forces", str(PRESS_FORCE_N), "--probe_mass", str(PROBE_MASS_KG),
                               "--max_depth_mm", str(args.max_depth_mm)]
                runner.add(name, test, common)
                sweeps.append({"test": test, "design": d, "thickness_label": label, "thickness": t, "name": name})
    runner.wait()

    # 2. Replay each sweep's worst case alone, on video.
    for s in sweeps:
        path = out / f"{s['name']}.json"
        if not path.exists():
            s["status"] = "ERROR"
            continue
        r = json.loads(path.read_text())
        s.update(failed=r["failed_envs"], total=r["num_envs"], failing_ranges=r["failing_ranges"], worst=r["worst_env"],
                 status="FAIL" if r["failed_envs"] else "pass")
        w = r["worst_env"]
        s["video"] = str(out / f"{s['name']}_worst.mp4")
        replay = ["--config", str(config), "--design", s["design"], "--thickness", str(s["thickness"]),
                  "--size", str(w["length"]), str(w["width"]), "--material", json.dumps(w["material"], separators=(",", ":")),
                  "--video", s["video"], "--out", str(out / f"{s['name']}_worst.json")]
        if s["test"] == "press":
            replay += ["--site", w["site"], "--forces", str(w["force_N"]), "--probe_mass", str(PROBE_MASS_KG)]
        runner.add(f"{s['name']}_worst", s["test"], replay)
    runner.wait()

    # 3. Report.
    failed_any = False
    print("\n" + "=" * 110)
    print(f"preflight  {args.experiment or args.task}  overrides {overrides or '-'}")
    for test in args.tests:
        bar = (f"probe {PROBE_MASS_KG} kg at {PRESS_FORCE_N:.0f} N; fail = through or > {args.max_depth_mm} mm"
               if test == "press" else "fail = fold not reached, or open again after release")
        print(f"\n{test.upper()}  ({bar})")
        for s in [s for s in sweeps if s["test"] == test]:
            if s["status"] == "ERROR":
                failed_any = True
                print(f"  ERROR {s['design']:18s} {s['thickness']*1000:5.1f} mm  no result; see {s['name']}.log / its Slurm log")
                continue
            failed_any |= s["failed"] > 0
            pct = 100.0 * s["failed"] / max(1, s["total"])
            w = s["worst"]
            where = f" at {w['site']}" if test == "press" else ""
            print(f"  {s['status']:5s} {s['design']:18s} {s['thickness']*1000:5.1f} mm  {s['failed']:5d}/{s['total']:<5d} "
                  f"({pct:5.1f}%) failing; worst {w['length']*100:.0f}x{w['width']*100:.0f} cm{where}, "
                  f"density {w['material']['density']:.3g}, bending {w['material']['bending_rigidity']:.3g}")
            if s["failed"]:
                limits = [f"{p} {r['failing'][0]:.3g}..{r['failing'][1]:.3g} (of {r['all'][0]:.3g}..{r['all'][1]:.3g})"
                          for p, r in s["failing_ranges"].items() if r["failing"] and r["failing"] != r["all"]]
                if limits:
                    print(f"        failures only within: {'; '.join(limits)}")
            print(f"        video: {s['video']}")
    print(f"\n{'FAILED' if failed_any else 'PASSED'}")
    print("=" * 110, flush=True)
    (out / "preflight.json").write_text(json.dumps(
        {"task": args.task, "experiment": args.experiment, "overrides": overrides, "thicknesses": thick,
         "probe": {"mass_kg": PROBE_MASS_KG, "force_N": PRESS_FORCE_N}, "max_depth_mm": args.max_depth_mm,
         "dr_random": args.dr_random, "sweeps": sweeps, "passed": not failed_any}, indent=2, default=str))
    return 1 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
