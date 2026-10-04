"""Collate the rigid-vs-cloth comparison: training speed, learning, and cross-evaluation.

Three tables, from three sources, because the claim has three parts and they fail independently:

  **speed**      from the training logs' ``fps total`` and the wall-clock each run reported. This is
                 the number the whole exercise is motivated by, and it must be quoted per ARM at the
                 same env count on the same GPU or it is measuring hardware.
  **learning**   from the training logs' ``episode_final/done_fold`` style summaries -- did the policy
                 improve on its own manipuland at a matched env-step budget.
  **transfer**   from ``eval/episodes.py`` JSON: each policy on its own manipuland AND on the VBD
                 cloth. This is the one that decides whether a rigid approximation is *useful*, as
                 opposed to merely trainable.

    .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect.py --job 155018
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("box3-mid", "box-mid-odd", "cyl-mid-odd", "box2-surface", "box-surface", "vbd")

_FPS = re.compile(r"fps total\s*:\s*([\d,]+)")
_EPOCH = re.compile(r"epoch\s*:\s*([\d,]+)\s*/\s*([\d,]+)")
_FRAMES = re.compile(r"frames\s*:\s*([\d,]+)")
_WALL = re.compile(r"^\[train\] arm=\S+ seed=\d+ exit=(\d+) wall_s=(\d+)", re.M)

#: rl_games' own ``fps total`` is OPTIMISTIC for a fast env and must not be the headline.
#: Measured against wall-clock over a 90 s window: the VBD control's reported 3001 fps matched its
#: realised 3003, but ``box3-mid``'s reported 22254 was realised 10923 -- a 2x overstatement, and it
#: turns a true 3.6x speed-up into a claimed 7.4x. Whatever rl_games excludes (logging, resets, the
#: gap between epochs) is a fixed cost, so it is a larger share of a faster env's epoch. The realised
#: rate below is ``frames / wall_s`` from the job's own timing, and it is what the results quote.
_ARM = re.compile(r"^\[train\] arm=(\S+) seed=(\d+)", re.M)


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def read_training(job: str, logs: Path) -> list[dict]:
    rows = []
    for path in sorted(logs.glob(f"rc_train_{job}_*.out")):
        text = path.read_text(errors="replace")
        m = _ARM.search(text)
        if not m:
            continue
        fps = [_int(v) for v in _FPS.findall(text)]
        epochs = _EPOCH.findall(text)
        frames = [_int(v) for v in _FRAMES.findall(text)]
        wall = _WALL.search(text)
        # The MEDIAN fps over the run, not the last value: rl_games' figure swings with whatever the
        # envs are doing that epoch, and a single sample is not a throughput.
        rows.append(
            {
                "arm": m.group(1),
                "fps_median": int(statistics.median(fps)) if fps else None,
                "epoch": _int(epochs[-1][0]) if epochs else 0,
                "epochs_total": _int(epochs[-1][1]) if epochs else 0,
                "frames": frames[-1] if frames else 0,
                "wall_s": int(wall.group(2)) if wall else None,
                "finished": wall is not None,
                "fps_realised": (
                    (frames[-1] / int(wall.group(2)))
                    if (wall and frames and int(wall.group(2)) > 0)
                    else None
                ),
            }
        )
    return rows


def read_evals(results: Path, prefix: str = "rc_eval_") -> list[dict]:
    """Every eval JSON under one label prefix.

    The prefix matters: `rc_eval_*.json` matches all three rounds AND the 1.5 cm re-runs, so an
    unfiltered glob builds a 90-row table in which `tol15_rc3_box-mid-odd_fixdr_s1` is a policy.
    Narrow it to the round under discussion (`--eval_prefix rc_eval_rc3_`) and leave the pooled
    cross-round tables to `rigid_cloth_collect_eval.py`, which is built for them.
    """
    rows = []
    for path in sorted(results.glob(f"{prefix}*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # rc_eval_<label>_c<coef>_s<seed>.json, label = <arm>_s<n>__on_<target>
        stem = path.stem[len(prefix) :]
        label = stem.rsplit("_c", 1)[0]
        policy, _, target = label.partition("__on_")
        # Flatten the nested blocks `episodes.py` writes, so the table below is not guessing at key
        # names -- the first version read `folds_held` / `falls` / `best_fold_err_mean` at the top
        # level, none of which exist, and would have reported every arm as zero.
        reasons = data.get("termination_reasons", {}) or {}
        completed = data.get("completed", {}) or {}
        fold = data.get("fold", {}) or {}
        errs = [v for v in (fold.get("best_fold_err") or []) if v is not None]
        rows.append(
            {
                "policy": policy,
                "target": target,
                "n": int(completed.get("n", 0) or 0),
                "folds_held": int(reasons.get("fold_held", 0) or 0),
                "falls": int(reasons.get("fall", 0) or 0),
                "best_fold_err_mean": (statistics.fmean(errs) if errs else None),
                "episode_len": completed.get("mean_episode_length"),
            }
        )
    return rows


#: Scalars rl_games writes per epoch that say whether the policy is learning THE FOLD, as opposed to
#: collecting shaped reward. `fold_rate` is the fraction of completed episodes that ended in a fold;
#: `fold_err_mean` is how close the sheet got, in metres, and moves even when no fold completes.
_CURVES = ("fold_rate", "fold_err_mean", "rewards/step")


def read_curves(job: str, outputs: Path) -> dict[tuple[str, int], dict]:
    """Per-run learning curves, read from the tensorboard summaries rl_games writes.

    Reported as first-decile vs last-decile means rather than final values: a single epoch's figure
    swings hard at 768 envs, and the question is whether the run IMPROVED, which one endpoint cannot
    answer.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:  # pragma: no cover - reporting aid only
        return {}

    out: dict[tuple[str, int], dict] = {}
    for run in outputs.glob(f"*/*/0_rc_*_{job}"):
        name = run.name[len("0_rc_") :].rsplit(f"_{job}", 1)[0]
        arm, _, seed = name.rpartition("_s")
        events = sorted(run.glob("summaries/events*"))
        if not events:
            continue
        ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:  # pragma: no cover - a run still being written
            continue
        tags = set(ea.Tags()["scalars"])
        row: dict = {}
        for tag in _CURVES:
            if tag not in tags:
                continue
            vals = [e.value for e in ea.Scalars(tag)]
            if not vals:
                continue
            k = max(1, len(vals) // 10)
            row[tag] = (statistics.fmean(vals[:k]), statistics.fmean(vals[-k:]), len(vals))
        if row:
            out[(arm, int(seed))] = row
    return out


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Collate the rigid-cloth comparison.")
    p.add_argument("--job", required=True, help="training array job id")
    p.add_argument("--logs", default="slurm_logs")
    p.add_argument("--results", default="docs/results")
    p.add_argument(
        "--eval_prefix",
        default="rc_eval_",
        help="Label prefix for the cross-evaluation table. The default matches every round and both "
        "fold cutoffs, which is ~660 files and an unreadable table; pass e.g. `rc_eval_rc3_` for one "
        "round. Pooled cross-round tables are `rigid_cloth_collect_eval.py`'s job.",
    )
    args = p.parse_args()

    train = read_training(args.job, REPO_ROOT / args.logs)
    evals = read_evals(REPO_ROOT / args.results, args.eval_prefix)

    print("## Training speed and progress\n")
    print(f"{'arm':<14} {'seeds':<6} {'fps realised':<14} {'fps rl_games':<14} {'x vs vbd':<9} "
          f"{'epoch':<14} {'wall (h)':<9} {'done'}")
    base = None
    by_arm: dict[str, list[dict]] = {}
    for r in train:
        by_arm.setdefault(r["arm"], []).append(r)
    def rate(rows):
        """Realised throughput where the run finished, else rl_games' optimistic figure."""
        real = [r["fps_realised"] for r in rows if r["fps_realised"]]
        if real:
            return statistics.median(real), True
        rep = [r["fps_median"] for r in rows if r["fps_median"]]
        return (statistics.median(rep) if rep else None), False

    vbd = by_arm.get("vbd", [])
    if vbd:
        base, _ = rate(vbd)
    for arm in ARMS:
        rows = by_arm.get(arm)
        if not rows:
            continue
        fps, is_real = rate(rows)
        rep = [r["fps_median"] for r in rows if r["fps_median"]]
        reported = statistics.median(rep) if rep else None
        ep = statistics.median([r["epoch"] for r in rows])
        tot = max(r["epochs_total"] for r in rows)
        walls = [r["wall_s"] for r in rows if r["wall_s"]]
        done = sum(1 for r in rows if r["finished"])
        speedup = f"{fps / base:.1f}x" if (fps and base) else "-"
        wall_h = f"{statistics.median(walls) / 3600:.2f}" if walls else "-"
        shown = f"{int(fps)}" + ("" if is_real else "*") if fps else "-"
        print(f"{arm:<14} {len(rows):<6} {shown:<14} "
              f"{int(reported) if reported else '-':<14} {speedup:<9} "
              f"{int(ep)}/{tot:<9} {wall_h:<9} {done}/{len(rows)}")
    print("\n* still running: rl_games' figure, which overstates a fast env by up to 2x.")

    curves = read_curves(args.job, REPO_ROOT / "outputs")
    if curves:
        print("\n## Learning on its OWN manipuland (first decile -> last decile, mean over seeds)\n")
        print(f"{'arm':<14} {'seeds':<6} {'fold_rate':<20} {'fold_err_mean (m)':<24} "
              f"{'reward/step':<18} {'epochs'}")
        for arm in ARMS:
            rows = [v for (a, _s), v in curves.items() if a == arm]
            if not rows:
                continue
            def span(tag, nd=4):
                have = [r[tag] for r in rows if tag in r]
                if not have:
                    return "-"
                return (f"{statistics.fmean(v[0] for v in have):.{nd}f} -> "
                        f"{statistics.fmean(v[1] for v in have):.{nd}f}")
            n_ep = max((r[t][2] for r in rows for t in r), default=0)
            print(f"{arm:<14} {len(rows):<6} {span('fold_rate', 3):<20} "
                  f"{span('fold_err_mean'):<24} {span('rewards/step', 2):<18} {n_ep}")

    if not evals:
        print("\n(no rc_eval_*.json yet -- run the evaluation matrix)")
        return

    print("\n## Cross-evaluation: held folds out of the episodes run\n")
    print(f"{'policy':<24} {'tested on':<14} {'n':<5} {'held folds':<14} "
          f"{'best_fold_err':<14} {'falls':<9} {'ep len'}")
    agg: dict[tuple[str, str], list[dict]] = {}
    for r in evals:
        agg.setdefault((r["policy"], r["target"]), []).append(r)
    for (policy, target), rows in sorted(agg.items()):
        n = sum(r["n"] for r in rows)
        folds = sum(r["folds_held"] for r in rows)
        errs = [r["best_fold_err_mean"] for r in rows if r["best_fold_err_mean"] is not None]
        falls = sum(r["falls"] for r in rows)
        lens = [r["episode_len"] for r in rows if r["episode_len"] is not None]
        pct = f"{folds}/{n} ({100.0 * folds / n:.0f}%)" if n else "-"
        fall_pct = f"{falls} ({100.0 * falls / n:.0f}%)" if n else "-"
        print(f"{policy:<24} {target:<14} {n:<5} {pct:<14} "
              f"{_fmt(statistics.fmean(errs) if errs else None):<14} {fall_pct:<9} "
              f"{_fmt(statistics.fmean(lens) if lens else None, 0)}")


if __name__ == "__main__":
    main()
