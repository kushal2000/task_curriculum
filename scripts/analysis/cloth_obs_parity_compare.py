"""Diff two ``cloth_obs_parity.py`` dumps (e.g. Newton vs PhysX), block by observation block.

    python3 scripts/analysis/cloth_obs_parity_compare.py A.pt B.pt [--steps 0,1,10,59]

Prints, for the chosen steps, the worst absolute difference in each observation block over all envs,
and the same for the fold error, the particle cloud and the object / goal quaternions (sign-folded:
q and -q are one rotation). Physics differs between simulators, so differences are expected to GROW
with time; at step 0 -- the pinned start, before anything has moved -- a convention error shows as a
large difference in one block while everything else is near zero.
"""

from __future__ import annotations

import argparse

import torch


def qdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--steps", default="0,1,5,20,59")
    args = p.parse_args()
    A, B = torch.load(args.a), torch.load(args.b)
    assert A["obs_fields"] == B["obs_fields"], (A["obs_fields"], B["obs_fields"])
    steps = [int(s) for s in args.steps.split(",") if int(s) < min(len(A["obs"]), len(B["obs"]))]
    print(f"A = {A['task']}\nB = {B['task']}")
    header = f"{'block':28s}" + "".join(f"{'t=' + str(t):>10s}" for t in steps)
    print(header)
    lo = 0
    for name, size in A["obs_fields"]:
        hi = lo + size
        row = [float((A["obs"][t][:, lo:hi] - B["obs"][t][:, lo:hi]).abs().max()) for t in steps]
        print(f"{name + f' [{lo}:{hi}]':28s}" + "".join(f"{v:10.4f}" for v in row))
        lo = hi
    for key, fn in (
        ("fold_err", lambda a, b: (a - b).abs().max()),
        ("particles", lambda a, b: (a - b).norm(dim=-1).max()),
        ("object_quat", lambda a, b: qdiff(a, b).max()),
        ("goal_quat", lambda a, b: qdiff(a, b).max()),
        ("goal_pos", lambda a, b: (a - b).norm(dim=-1).max()),
    ):
        row = [float(fn(A[key][t], B[key][t])) for t in steps]
        print(f"{key:28s}" + "".join(f"{v:10.4f}" for v in row))
    print("\nenv 0 at step 0:")
    for key in ("object_quat", "goal_quat", "goal_pos"):
        print(f"  {key:12s} A {[round(float(v), 4) for v in A[key][0][0]]}")
        print(f"  {'':12s} B {[round(float(v), 4) for v in B[key][0][0]]}")


if __name__ == "__main__":
    main()
