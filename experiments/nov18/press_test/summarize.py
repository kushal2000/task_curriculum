"""Summarize press_test JSONs as a markdown table: one row per run, one column per force.

    python experiments/nov18/press_test/summarize.py <label>=<json> [<label>=<json> ...]

Cell = sites through / sites, and the worst steady depth among the sites that held (mm);
'slid' counts sites where the probe ended off the panel.
"""
import json
import math
import sys


def cell(envs):
    through = sum(e["through"] for e in envs)
    held = [e["steady_depth_mm"] for e in envs if not e["through"] and not math.isnan(e["steady_depth_mm"])]
    slid = sum(e["slid_off"] and not e["through"] for e in envs)
    worst = f"{max(held):.1f} mm" if held else "-"
    return f"{through}/{len(envs)} through, held max {worst}" + (f", {slid} slid" if slid else "")


def main():
    rows = [a.split("=", 1) for a in sys.argv[1:]]
    data = {label: json.load(open(path)) for label, path in rows}
    forces = sorted({e["force_N"] for d in data.values() for e in d["per_env"]})
    print("| run | " + " | ".join(f"{f:.0f} N" for f in forces) + " |")
    print("|---|" + "---|" * len(forces))
    for label, d in data.items():
        cells = [cell([e for e in d["per_env"] if e["force_N"] == f]) for f in forces]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
