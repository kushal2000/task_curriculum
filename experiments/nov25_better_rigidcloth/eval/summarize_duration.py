"""duration_probe.py's output: how long penetrations last, and success under persistence-based rules.

    python3 experiments/nov25_better_rigidcloth/eval/summarize_duration.py outputs/eval_nov25_duration/*.json
"""
import json
import sys


def runs(seq, thr):
    """(start, length) of each run of consecutive steps with depth > thr."""
    out, start = [], None
    for i, v in enumerate(seq + [0.0]):
        if v > thr and start is None:
            start = i
        elif v <= thr and start is not None:
            out.append((start, i - start))
            start = None
    return out


def ok_under(ep, thr, n):
    """The episode succeeds under 'fail once depth > thr for n consecutive steps' iff it folded and no such
    run reached n steps before the episode ended (a fold ends it)."""
    if not ep["folded"]:
        return False
    return all(length < n for _s, length in runs(ep["depth_mm"], thr))


for path in sys.argv[1:]:
    d = json.load(open(path))
    eps = [e for e in d["episodes"] if e["outcome"] is not None]
    dt = d["policy_dt"]
    rl = sorted(length for e in eps for _s, length in runs(e["depth_mm"], 0.5))
    pen_eps = [e for e in eps if any(v > 0.5 for v in e["depth_mm"])]
    frac_steps = [sum(v > 0.5 for v in e["depth_mm"]) / max(1, len(e["depth_mm"])) for e in pen_eps]
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else 0
    print(f"\n== {path.rsplit('/', 1)[-1]}  ({len(eps)} episodes, policy step {dt * 1e3:.1f} ms = "
          f"{d['decimation']} physics steps of {d['physics_dt'] * 1e3:.2f} ms)")
    print(f"   folded with the rule off: {100 * sum(e['folded'] for e in eps) / len(eps):.1f}%   "
          f"episodes with any step > 0.5 mm: {100 * len(pen_eps) / len(eps):.1f}%")
    if rl:
        print(f"   penetration runs (> 0.5 mm): {len(rl)}; length in policy steps: median {q(rl, .5)}, p75 {q(rl, .75)}, "
              f"p90 {q(rl, .9)}, max {rl[-1]}; single-step runs {100 * sum(x == 1 for x in rl) / len(rl):.0f}%")
        print(f"   share of an episode's steps over 0.5 mm (episodes that penetrate): median "
              f"{100 * q(sorted(frac_steps), .5):.1f}%, p90 {100 * q(sorted(frac_steps), .9):.1f}%")
    print("   success if the episode fails after N consecutive steps over the threshold:")
    print("     threshold   N=1     N=2     N=3     N=5     N=10    (N=1 at 0.5 mm = the current rule)")
    for thr in (0.5, 1.0, 2.0):
        cells = "  ".join(f"{100 * sum(ok_under(e, thr, n) for e in eps) / len(eps):5.1f}%" for n in (1, 2, 3, 5, 10))
        print(f"     {thr:4.1f} mm    {cells}")
