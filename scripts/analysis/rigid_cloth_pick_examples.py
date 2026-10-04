"""Pick one own-chain success, one cloth success and one cloth failure per round-2 policy.

Reads the clips + outcome files `bos14_rigid_cloth_render_examples.sh` writes to
`videos/rigid_cloth_rc2_examples/candidates/` and copies the picks up one level as
`<policy>__{chain_success,cloth_success,cloth_failure}.mp4`, with `picks.json` listing why.

* a success is a held fold (`fold_held`) at the render's tolerance; the cleanest one (lowest best
  fold error) is taken, since every held fold is equally a success and the clearest reads best;
* the clip's final held second (`--end_hold_s`, 1 s at 30 fps) is re-stamped with the scored
  outcome. Clips rendered before `render_newton` did this itself carry a HUD that predates the
  scoring step ("folds 0" on a success), so the HUD plate is covered and redrawn;
* the failure is TYPICAL rather than dramatic: the policy's most frequent cloth failure mode, and
  within it the clip with the median best fold error.

    .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_pick_examples.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaacsimenvs.eval.render_newton import _stamp_hud, stamp_outcome  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "videos/rigid_cloth_rc2_examples"
POLICIES = ("box2-surface", "box2-surface-DR", "box3-mid", "box3-mid-DR")


def load(policy: str, target_is_cloth: bool) -> list[dict]:
    out = []
    for js in sorted((ROOT / "candidates").glob(f"{policy}__on_*_w*.json")):
        on_cloth = "__on_cloth_" in js.name
        if on_cloth != target_is_cloth or not js.with_suffix(".mp4").exists():
            continue
        info = json.loads(js.read_text())
        info["clip"] = js.with_suffix(".mp4")
        out.append(info)
    return out


def err(c: dict) -> float:
    return c.get("min_fold_err") if c.get("min_fold_err") is not None else float("inf")


HOLD_FRAMES = 30


def write_annotated(src: Path, dst: Path, info: dict) -> None:
    """Copy `src` to `dst`, re-stamping the final held second with the scored outcome."""
    reader = iio.get_reader(src)
    fps = reader.get_meta_data().get("fps", 30)
    n = reader.count_frames()
    writer = iio.get_writer(dst, fps=fps, macro_block_size=None)
    held = info["outcome"] in ("fold_held", "fold")
    lines = [
        f"step      {info['steps']:4d}",
        f"best err  {info['min_fold_err'] or 0:6.3f} m  (tol {info['keypoint_tolerance']:.3f})",
        f"episode   {'folded' if held else 'not folded':>10s}    ",
        f"folds     {1 if held else 0}",
    ]
    for i in range(n):
        frame = reader.get_data(i)
        if i >= n - HOLD_FRAMES:
            frame = np.array(frame)
            # cover the stale HUD plate (same geometry as `_stamp_hud`: top-right, 4 lines)
            frame[8:128, frame.shape[1] - 360 :] = (30, 30, 30)
            frame = _stamp_hud(frame, lines, held)
            frame = stamp_outcome(frame, info["outcome"])
        writer.append_data(np.ascontiguousarray(frame))
    writer.close()


def main() -> None:
    picks = {}
    for policy in POLICIES:
        chain, cloth = load(policy, False), load(policy, True)
        chosen = {}
        ok = sorted((c for c in chain if c["outcome"] == "fold_held"), key=err)
        if ok:
            chosen["chain_success"] = ok[0]
        ok = sorted((c for c in cloth if c["outcome"] == "fold_held"), key=err)
        if ok:
            chosen["cloth_success"] = ok[0]
        fails = [c for c in cloth if c["outcome"] not in ("fold_held", "fold", "not_finished")]
        if fails:
            mode = Counter(c["outcome"] for c in fails).most_common(1)[0][0]
            same = sorted((c for c in fails if c["outcome"] == mode), key=err)
            chosen["cloth_failure"] = same[len(same) // 2]

        tally = {
            "chain": dict(Counter(c["outcome"] for c in chain)),
            "cloth": dict(Counter(c["outcome"] for c in cloth)),
        }
        picks[policy] = {"candidates": tally}
        for kind, c in chosen.items():
            dst = ROOT / f"{policy}__{kind}.mp4"
            write_annotated(c["clip"], dst, c)
            picks[policy][kind] = {
                "file": dst.name,
                "from": c["clip"].name,
                "outcome": c["outcome"],
                "steps": c["steps"],
                "best_fold_err_cm": None if err(c) == float("inf") else round(100 * err(c), 2),
            }
        missing = {"chain_success", "cloth_success", "cloth_failure"} - chosen.keys()
        if missing:
            picks[policy]["missing"] = sorted(missing)
        print(f"{policy:16s} chain {tally['chain']}  cloth {tally['cloth']}")
        for kind in ("chain_success", "cloth_success", "cloth_failure"):
            p = picks[policy].get(kind)
            print(f"    {kind:14s} " + (f"{p['file']}  <- {p['from']}  {p['outcome']}, "
                                         f"{p['steps']} steps, best err {p['best_fold_err_cm']} cm"
                                         if p else "NONE FOUND"))
    (ROOT / "picks.json").write_text(json.dumps(picks, indent=2))


if __name__ == "__main__":
    main()
