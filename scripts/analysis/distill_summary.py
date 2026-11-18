"""Table of the point-cloud distillation runs: training log tail + evaluation JSONs.

    python3 scripts/analysis/distill_summary.py [tag ...]      (default: every tag in docs/results/distill)
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RES = Path("docs/results/distill")


def evals():
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(RES.glob("*__on_*.json")):
        m = re.match(r"(.+)__on_(physx|vbd)__(pinned|rand)_s(\d+)\.json", f.name)
        if m:
            out[m.group(1)][f"{m.group(2)}_{m.group(3)}"].append(json.loads(f.read_text()))
    return out


def cell(rs, key=None):
    if not rs:
        return "-"
    if key is None:
        ok = sum(sum(1 for v in r["per_env_success"] if v) for r in rs)
        n = sum(r["n"] for r in rs)
        return f"{100 * ok / max(n, 1):.0f}% ({n})"
    vals = [r for r in rs if key(r) is not None]
    return f"{sum(key(r) for r in vals) / len(vals):.1f}" if vals else "-"


def last_row(tag):
    dirs = sorted(glob.glob(f"outputs/distill/{tag}_s*_*/log.jsonl"))
    if not dirs:
        return None
    lines = Path(dirs[-1]).read_text().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def main():
    ev = evals()
    tags = sys.argv[1:] or sorted(ev)
    print("| run | train iter | train succ | PhysX rand | VBD pinned | VBD rand | aux goal cm (PhysX) | by drag <1 / 1-3 / >3 cm | VBD drag p50 cm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        e, row = ev.get(t, {}), last_row(t)
        px = e.get("physx_rand", [])
        bd = px[0].get("aux_goal_err_cm_by_drag") if px else None
        bd = " / ".join("-" if v is None else f"{v:.1f}" for v in bd.values()) if bd else "-"
        print(f"| {t} | {row['iter'] if row else '-'} | "
              f"{'-' if not row or row['student_success'] is None else f'{100 * row['student_success']:.0f}%'} | "
              f"{cell(px)} | {cell(e.get('vbd_pinned', []))} | {cell(e.get('vbd_rand', []))} | "
              f"{cell(px, lambda r: (r.get('aux_err_cm') or {}).get('keypoints_rel_goal'))} | {bd} | "
              f"{cell(e.get('vbd_pinned', []) + e.get('vbd_rand', []), lambda r: 100 * r['max_drag_xy_m']['p50'] if r['max_drag_xy_m']['p50'] is not None else None)} |")


if __name__ == "__main__":
    main()
