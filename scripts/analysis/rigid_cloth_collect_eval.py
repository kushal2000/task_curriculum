"""Collate the rigid-cloth cross-evaluation into the tables `rigid_cloth_rl_comparison.md` prints.

Reads the per-run JSONs that `bos14_rigid_cloth_eval*.sh` writes into `docs/results/`, pools them
over eval seeds, and emits one markdown table plus a per-training-seed breakdown. Every published
table in the write-up is this one program under different flags:

    # round 2 (the two matched chains) against round 1 and the reference, at the 4 cm cutoff
    .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect_eval.py \
        --round rc2 --round 1 --cutoff 4cm

    # the same policies at both cutoffs, dropping the diverged box2-surface fixdr seed
    .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect_eval.py \
        --round rc2 --format rate --exclude_seed box2-surface:fixdr:3

    # round 3 (the three many-slat chains), carrying round 2 into the same table
    .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect_eval.py --round rc3 --round rc2

It replaced three forked copies of itself (`_rc2`, `_rc3`, `_tol15`), which differed by an arm list,
a row label and a percent format while duplicating the pooling -- and had drifted apart: only one of
them knew that the two cutoffs are separate EVALUATION RUNS rather than two scorings of one rollout.
They are: `keypoint_tolerance` gates `ClothEnv.is_success()` and therefore termination, so a 4 cm
episode ends at the first loose fold and the episodes themselves differ. That is why each cutoff is
its own file prefix and never a recomputed column.

`rigid_cloth_collect.py` is a different program and stays separate: it reads training logs and
tensorboard curves to answer "how fast did it learn", not "what does it fold".
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "docs/results"

#: (target, label suffix) per column. "own" is the policy's own manipuland, "vbd" the real cloth;
#: the bare suffix is the pinned start pose and `__rand` the randomised one.
COLS = (("own", ""), ("own", "__rand"), ("vbd", ""), ("vbd", "__rand"))
COL_NAMES = ("own chain, pinned", "own chain, randomised", "cloth, pinned", "cloth, randomised")

#: Fold cutoff -> file prefix. 4 cm is the task's own `keypoint_tolerance`; 1.5 cm (~160 degrees of
#: a 180 degree fold) is the re-run that separates a held fold from a loose one.
CUTOFFS = {"4cm": "rc_eval_", "1.5cm": "rc_eval_tol15_"}

#: Round tag -> how that round's runs are named and labelled. One entry per set of runs that shares
#: a naming scheme, because the scheme is all that distinguishes them: round 1 predates the round
#: prefix and the fix/DR axis, so its files are bare `<arm>_s<n>`, and its 1536-env re-run appends
#: `_e1536`. Adding a round is adding a row here.
ROUNDS: dict[str, dict] = {
    "1": {
        "arms": ("box3-mid", "box2-surface"),
        "label": "{arm}_s{seed}",
        "display": "{arm} round 1 (768 envs)",
        "groups": ("",),
    },
    "1-e1536": {
        "arms": ("box2-surface",),
        "label": "{arm}_s{seed}_e1536",
        "display": "{arm} round 1 (1536 envs, 2 seeds)",
        "groups": ("",),
    },
    "rc2": {
        "arms": ("box3-mid", "box2-surface"),
        "label": "rc2_{arm}_{group}_s{seed}",
        "display": "**{arm} {group}**",
    },
    "rc3": {
        "arms": ("cyl-mid-odd", "box-mid-odd", "box-surface"),
        "label": "rc3_{arm}_{group}_s{seed}",
        "display": "**{arm} {group}**",
    },
}
GROUPS = ("fix", "fixdr")
SEEDS = (1, 2, 3)

#: The cloth-trained policy every arm is read against, as its own label.
REFERENCE = "reference"


def load(prefix: str, label_glob: str) -> list[dict]:
    """Every eval JSON matching one label, over all eval seeds."""
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{prefix}{label_glob}_c0_s*.json"))]


def pooled(runs: list[dict]) -> tuple[int, int, int]:
    """(held folds, falls, episodes) summed over the runs."""
    held = sum(int(r["termination_reasons"]["fold_held"]) for r in runs)
    falls = sum(int(r["termination_reasons"]["fall"]) for r in runs)
    n = sum(int(r["num_envs"]) for r in runs)
    return held, falls, n


def cell(runs: list[dict], fmt: str = "falls") -> str:
    if not runs:
        return "--"
    h, f, n = pooled(runs)
    if fmt == "rate":
        return f"{100 * h / n:.0f}% ({h}/{n})"
    return f"{h}/{n} ({100 * h / n:.0f}%), falls {100 * f / n:.0f}%"


def arm_row(prefix: str, round_tag: str, arm: str, group: str, seeds: str, fmt: str) -> list[str]:
    """One table row's four cells: the policy on its own manipuland and on the cloth, two starts each."""
    seed_glob = "?" if seeds == "?" else f"[{seeds}]"
    label = ROUNDS[round_tag]["label"].format(arm=arm, group=group, seed=seed_glob)
    return [
        cell(load(prefix, f"{label}__on_{arm if target == 'own' else 'vbd'}{suffix}"), fmt)
        for target, suffix in COLS
    ]


def reference_rows(prefix: str, rounds: list[str], fmt: str) -> list[tuple[str, list[str]]]:
    """The reference on the cloth, then on each round's chains, OLD chains before FIXED ones.

    The reference is re-measured per round under its own tag (`reference_rc2`, `reference_rc3`),
    because a fixed chain is a different manipuland from the one the previous round scored it on --
    the same checkpoint, a different question. A bare `reference` is the round-1 measurement, on the
    unfixed chain; the row says which it is rather than leaving the two indistinguishable.
    """
    rows = [(
        "reference, cloth-trained, on cloth",
        [cell(load(prefix, f"{REFERENCE}__on_vbd{s}"), fmt) if t == "own" else "(same)"
         for t, s in COLS],
    )]
    # One row per (chain, which version of it), not per round: the same arm appears in several
    # rounds, and the OLD measurement of it is one measurement however many rounds cite it.
    seen = set()
    for what, tag_of in (("OLD", lambda _r: REFERENCE),
                         ("FIXED", lambda r: f"{REFERENCE}_{r}")):
        for round_tag in rounds:
            for arm in ROUNDS[round_tag]["arms"]:
                tag = tag_of(round_tag)
                if (tag, arm) in seen:
                    continue
                seen.add((tag, arm))
                cells = [cell(load(prefix, f"{tag}__on_{arm}{s}"), fmt) for _, s in COLS[:2]]
                if cells == ["--", "--"]:
                    continue
                rows.append((f"reference on {arm}, {what} chain", cells + ["", ""]))
    return rows


def table(rounds: list[str], cutoffs: list[str], fmt: str, excluded: dict) -> str:
    """One table over the chosen rounds, with a `tol` column iff more than one cutoff is shown."""
    multi = len(cutoffs) > 1
    head = ["policy"] + (["tol"] if multi else []) + list(COL_NAMES)
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    per_seed: dict = defaultdict(dict)

    def emit(name: str, cells_by_cutoff: dict) -> None:
        for tol in cutoffs:
            cells = cells_by_cutoff[tol]
            # A cutoff a row was never measured at is omitted, not printed as a line of dashes:
            # the reference on the OLD chains was only ever run at 4 cm.
            if all(c in ("--", "") for c in cells):
                continue
            lines.append("| " + " | ".join([name] + ([tol] if multi else []) + cells) + " |")

    for round_tag in rounds:
        spec = ROUNDS[round_tag]
        for arm in spec["arms"]:
            for group in spec.get("groups", GROUPS):
                name = spec["display"].format(arm=arm, group=group)
                emit(name + (" (3 seeds)" if group else ""),
                     {tol: arm_row(CUTOFFS[tol], round_tag, arm, group, "?", fmt)
                      for tol in cutoffs})
                # An exclusion ADDS a row rather than replacing it: the pooled row is the honest
                # result and the excluded one is the claim about what the other seeds did, so a
                # table that shows only the second is the one way to mislead with this flag.
                if (seeds := excluded.get((arm, group))):
                    dropped = "".join(sorted(set("123") - set(seeds)))
                    emit(f"{name} excl. failed s{dropped}",
                         {tol: arm_row(CUTOFFS[tol], round_tag, arm, group, seeds, fmt)
                          for tol in cutoffs})
                if not group:
                    continue
                for seed in SEEDS:
                    for tol in cutoffs:
                        per_seed[(arm, group, seed, tol)] = arm_row(
                            CUTOFFS[tol], round_tag, arm, group, str(seed), fmt
                        )

    # The reference rows are the same policy under every round, so they are listed once at the end
    # rather than per round; `reference_rows` keys them by the chain it was measured on.
    ref = {tol: dict(reference_rows(CUTOFFS[tol], rounds, fmt)) for tol in cutoffs}
    for name in ref[cutoffs[0]]:
        emit(name, {tol: ref[tol].get(name, ["--"] * len(COLS)) for tol in cutoffs})

    out = ["\n".join(lines)]
    if per_seed:
        out += ["", "Per training seed (held folds, both eval seeds pooled):"]
        for (arm, group, seed, tol), cells in sorted(per_seed.items()):
            shown = [f"{n}: {c}" for n, c in zip(COL_NAMES, cells) if c != "--"]
            if shown:
                tol_col = f"{tol:>5}  " if len(cutoffs) > 1 else ""
                out.append(f"  {arm:<13} {group:<6} s{seed}  {tol_col}" + "  ".join(shown))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--round",
        dest="rounds",
        action="append",
        choices=sorted(ROUNDS),
        help="Which training round's arms to tabulate; repeat to stack rounds in one table, in the "
        "order given. Default: rc3 then rc2, the two that share the matched-chain physics.",
    )
    parser.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        choices=sorted(CUTOFFS),
        help="Fold cutoff to report; repeat for both, which adds a `tol` column. These are separate "
        "evaluation RUNS, not two scorings of one rollout. Default: both.",
    )
    parser.add_argument(
        "--format",
        choices=("falls", "rate"),
        default="falls",
        help="`falls` (default) is `held/n (pct), falls pct`; `rate` is `pct (held/n)`, which fits "
        "a table that already carries a tol column.",
    )
    parser.add_argument(
        "--exclude_seed",
        action="append",
        default=[],
        metavar="ARM:GROUP:SEED",
        help="Drop one training seed from an arm's pooled row, e.g. `box2-surface:fixdr:3` for a "
        "run that lost the prior in its first epochs. The row is labelled with the exclusion; the "
        "per-seed breakdown still lists every seed, so the dropped one stays visible.",
    )
    args = parser.parse_args()
    rounds = args.rounds or ["rc3", "rc2"]
    cutoffs = args.cutoffs or sorted(CUTOFFS, reverse=True)

    excluded: dict = {}
    for spec in args.exclude_seed:
        arm, group, seed = spec.split(":")
        excluded[(arm, group)] = "".join(s for s in "123" if s != seed)

    print(table(rounds, cutoffs, args.format, excluded))


if __name__ == "__main__":
    main()
