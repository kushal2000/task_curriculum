"""Build the cross-evaluation matrix: every policy against every manipuland it should be tried on.

The experiment's central question is not "did each policy learn its own env" -- that only says the
approximation is *trainable*. It is **transfer**: a rigid chain is worth using only if a policy
trained on it folds the real VBD sheet. So every trained policy is evaluated twice, on its own
manipuland and on the cloth, and the gap between those two numbers is the result.

Emits a TSV because the consumer is an sbatch array: one line per evaluation, indexed by
``SLURM_ARRAY_TASK_ID``. Generated rather than hand-written so the Hydra run directory (a timestamp
nobody can predict) is resolved by searching for the run NAME, and so a missing checkpoint is a loud
line in this file's output instead of a job that starts, runs for minutes and then cannot find it.

    .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_eval_manifest.py --job 154851
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Index 5 is the control. Keep in step with `scripts/cluster/bos14_rigid_cloth_train.sh`.
ARMS = ("box3-mid", "box-mid-odd", "cyl-mid-odd", "box2-surface", "box-surface", "vbd")

RIGID_TASK = "Isaacsimenvs-RigidCloth-Direct-v0"
CLOTH_TASK = "Isaacsimenvs-Cloth-Direct-v0"


def find_checkpoint(name: str, outputs: Path) -> Path | None:
    """The final checkpoint of the run called ``name``.

    ``train.py`` co-locates rl_games artifacts with the Hydra run directory, so the path contains a
    date and a time this script cannot know. Searching by name is the only stable handle.

    Prefers ``nn/<name>.pth`` -- rl_games' "last" symlink-equivalent -- and falls back to the
    highest-epoch ``last_<name>_ep_<n>_rew_<r>.pth``. Deliberately does NOT fall back to
    ``last/model.pth``: that file is written by a different code path and has been observed to lag
    the ``nn/`` directory by several hundred epochs, which would silently evaluate a younger policy.
    """
    for run_dir in sorted(outputs.rglob(name), reverse=True):
        nn = run_dir / "nn"
        if not nn.is_dir():
            continue
        direct = nn / f"{name}.pth"
        if direct.is_file():
            return direct
        eps = []
        for p in nn.glob(f"last_{name}_ep_*.pth"):
            try:
                eps.append((int(p.name.split("_ep_")[1].split("_")[0]), p))
            except (IndexError, ValueError):
                continue
        if eps:
            return max(eps)[1]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Emit the rigid-cloth cross-evaluation manifest.")
    p.add_argument("--job", required=True, help="the training array's SLURM_ARRAY_JOB_ID")
    p.add_argument("--seeds", type=int, default=3, help="training seeds per arm")
    p.add_argument(
        "--reference",
        default=None,
        help="An existing cloth-trained checkpoint to include as a reference row, evaluated on the "
        "cloth AND on every rigid manipuland. This is the policy the comparison is against; it was "
        "trained for far longer than the matched arms, so it is context, not a matched result.",
    )
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--out", default="docs/results/rc_eval_manifest.tsv")
    args = p.parse_args()

    outputs = REPO_ROOT / args.outputs
    rows: list[tuple[str, ...]] = []
    missing: list[str] = []

    for arm in ARMS:
        for seed in range(1, args.seeds + 1):
            name = f"0_rc_{arm}_s{seed}_{args.job}"
            ckpt = find_checkpoint(name, outputs)
            if ckpt is None:
                missing.append(name)
                continue
            rel = ckpt.relative_to(REPO_ROOT)
            # Own manipuland, then the cloth. For the VBD control the two coincide, so it gets one
            # row on the cloth plus a row on each rigid env -- the reverse transfer direction, which
            # says whether the chains are merely *easier* or actually *different*.
            if arm == "vbd":
                rows.append((f"{arm}_s{seed}__on_vbd", str(rel), CLOTH_TASK, "-"))
                for other in ARMS[:-1]:
                    rows.append((f"{arm}_s{seed}__on_{other}", str(rel), RIGID_TASK, other))
            else:
                rows.append((f"{arm}_s{seed}__on_{arm}", str(rel), RIGID_TASK, arm))
                rows.append((f"{arm}_s{seed}__on_vbd", str(rel), CLOTH_TASK, "-"))

    if args.reference:
        ref = Path(args.reference)
        if not ref.is_absolute():
            ref = REPO_ROOT / ref
        if ref.is_file():
            rel = ref.relative_to(REPO_ROOT) if ref.is_relative_to(REPO_ROOT) else ref
            rows.append(("reference__on_vbd", str(rel), CLOTH_TASK, "-"))
            for other in ARMS[:-1]:
                rows.append((f"reference__on_{other}", str(rel), RIGID_TASK, other))
        else:
            missing.append(f"reference {ref}")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join("\t".join(r) + "\n" for r in rows))

    print(f"wrote {out} with {len(rows)} evaluations (array 0-{len(rows) - 1})")
    for r in rows:
        print("  " + "\t".join(r))
    if missing:
        print(f"\nMISSING {len(missing)} checkpoint(s) -- those arms are omitted from the matrix:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
