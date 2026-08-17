#!/usr/bin/env python3
"""Summarise cable-eval result JSONs as one table, sorted by goals.

    python3 scripts/cluster/collect_cable_results.py [glob ...]

Reports *total goals* and *max goals in a single episode* alongside the mean, because those are
different questions and conflating them caused a real misreport here: "6 goals" was six separate
envs scoring once each, not one episode scoring six times.

`ejected` and `censored` are shown because both silently corrupt the headline -- an ejected object
registers as lifted, and censored episodes mean the step budget expired mid-flight.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

FIELDS = "tag", "goals", "max/ep", "goals/ep", "lift", "ejec", "cens", "fall", "t/out", "envs"


def row(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    g = d.get("per_env_goals")
    if not g:
        return None
    term = d.get("termination_reasons", {})
    return {
        "tag": path.stem,
        "goals": sum(g),
        "max/ep": max(g),
        "goals/ep": round(d["completed"]["mean_goals_per_episode"], 4),
        "lift": d.get("lift_fraction"),
        "ejec": d.get("ejected", "-"),
        "cens": d.get("censored", "-"),
        "fall": term.get("fall", "-"),
        "t/out": term.get("timeout", "-"),
        "envs": len(g),
    }


def main() -> None:
    patterns = sys.argv[1:] or ["docs/results/*.json"]
    paths = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})
    rows = [r for r in (row(p) for p in paths) if r]
    if not rows:
        print("no result files with per-env goals found")
        return
    rows.sort(key=lambda r: (-r["max/ep"], -r["goals"]))

    width = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in FIELDS}
    print("  ".join(f.ljust(width[f]) for f in FIELDS))
    print("  ".join("-" * width[f] for f in FIELDS))
    for r in rows:
        print("  ".join(str(r[f]).ljust(width[f]) for f in FIELDS))

    best = rows[0]
    print(
        f"\nbest single episode: {best['max/ep']} goals ({best['tag']}); "
        f"{sum(r['goals'] for r in rows)} goals over {len(rows)} runs"
    )


if __name__ == "__main__":
    main()
