"""How far the UNTOUCHED plate jumps when a fingertip presses the other one (the PhysX "flick").

    python3 scripts/analysis/flick_probe_summary.py docs/results/fingerprint_physx_on_physx_p*.pt

Reads cloth_obs_parity.py dumps replaying the PhysX policy's recorded actions (bos14_contact_fingerprint.sh
STAGE=replay, task 5). Around step 27 that policy's thumb presses the -x plate near its outer edge;
nothing touches the +x plate. Prints, per dump, the env-median height above the table of the +x
half's outer row and of the -x half's outer row at a few steps: on the VBD cloth both stay at ~8 mm.
"""

from __future__ import annotations

import sys

import torch

TABLE_TOP = 0.53
STEPS = (26, 28, 30, 34, 40)


def main() -> None:
    print(f"{'dump':<52s}" + "".join(f"{f't{s} +x/-x':>14s}" for s in STEPS))
    for path in sys.argv[1:]:
        r = torch.load(path)
        p = r["particles"]  # (T, N, 49, 3)
        g = p.reshape(p.shape[0], p.shape[1], 7, 7, 3)
        row = ""
        for s in STEPS:
            z = (g[s, :, :, :, 2] - TABLE_TOP) * 1e3  # (N, 7, 7); rows run +x -> -x
            hi = z[:, 0].mean(-1).median()
            lo = z[:, -1].mean(-1).median()
            row += f"{float(hi):8.0f}/{float(lo):<5.0f}"
        print(f"{path.split('/')[-1]:<52s}{row}")


if __name__ == "__main__":
    main()
