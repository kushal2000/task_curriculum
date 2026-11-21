"""Print per-fabric rates and an equal-weight seven-fabric average."""
import json
import sys
from pathlib import Path
from statistics import mean

from isaacsimenvs.scripts.evaluation.cloth_standard import summarize

FABRICS = ("denim", "towel", "chiffon", "cotton", "silk", "napkin", "big_napkin")
RATES = ("acquisition_rate", "persistence_rate", "persistence_given_acquisition")


def main(directory):
    fields = ("protocol", "checkpoint", "policy_config", "manifest_sha256", "pinned", "seed", "sapg_expl_coef")
    runs = []
    reference_starts = None
    for fabric in FABRICS:
        chunks = [json.loads(p.read_text()) for p in (directory / fabric).glob("*.json")]
        if not chunks:
            raise ValueError(f"No evaluation results for {fabric}")
        reference = runs[0] if runs else chunks[0]
        starts = []
        for chunk in chunks:
            if chunk["fabric"] != fabric or any(chunk[field] != reference[field] for field in fields):
                raise ValueError(f"Mismatched evaluation for {fabric}")
            starts.extend(zip(chunk["episodes"], chunk["starts"]))
        starts.sort(key=lambda pair: pair[0]["start_id"])
        rows, poses = zip(*starts)
        if [row["start_id"] for row in rows] != list(range(256)):
            raise ValueError(f"Expected all 256 starts for {fabric}")
        if reference_starts is not None and poses != reference_starts:
            raise ValueError(f"Mismatched starts for {fabric}")
        reference_starts = poses
        runs.append(dict(chunks[0], summary=summarize(rows)))
    for tol in ("0.015", "0.04"):
        print(f"{float(tol) * 100:g} cm: acquisition / persistence / persistence given acquisition")
        values = [run["summary"][tol] for run in runs]
        for fabric, entry in zip(FABRICS, values):
            print(fabric, *(fmt(entry[key]) for key in RATES))
        print("macro", *(fmt(mean(v[key] for v in values) if all(v[key] is not None for v in values) else None)
                         for key in RATES))


def fmt(value):
    return "n/a" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    main(Path(sys.argv[1]))
