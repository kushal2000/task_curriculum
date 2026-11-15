"""Side-by-side viser viewer: the Flexiv + Wuji with its vendor joint limits vs human-like ones.

Left: ``flexiv_wuji.COMPOSED_URDF``, the robot the PlayFlexivWuji task trains. Right:
``flexiv_wuji.HUMAN_LIKE_URDF``, the same robot with ``HUMAN_LIKE_LIMITS_DEG``: the finger and
thumb joints' backward bend cut, and finger abduction held to +-2 deg. Each stands where the env
puts it, behind its own table.

Each robot has its own sliders, over its own URDF limits: arm joints, hand joints, and a
"hand action (all joints)" slider that sets every hand joint as the Play task maps its hand action
(lower + (a + 1) / 2 * (upper - lower)) -- so the same action value is a different pose on each.
"Copy vendor pose to human-like" puts the right robot in the left one's pose, clamped to its
limits. The readout lists every hand joint where the two differ, in degrees.

    .venv_isaaclab3/bin/python scripts/flexiv_wuji/viser_wuji_limits.py [--port 8080]

On the cluster, forward the port: ``ssh -L 8080:localhost:8080 <node>``, then open
http://localhost:8080.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flexiv_wuji as fw  # noqa: E402
from viser_compare import TABLE_COLOR, JointSliders  # noqa: E402

SETUPS = ("vendor", "human_like")
LABELS = {"vendor": "vendor limits", "human_like": "human-like limits"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    urdfs = {
        "vendor": yourdfpy.URDF.load(str(fw._require(fw.COMPOSED_URDF))),
        "human_like": yourdfpy.URDF.load(str(fw._require(fw.HUMAN_LIKE_URDF))),
    }
    names = list(urdfs["vendor"].actuated_joint_names)
    assert names == list(urdfs["human_like"].actuated_joint_names)
    arm_names = [n for n in names if n in fw.FLEXIV_DEFAULT_ARM_POS]
    hand_names = [n for n in names if n not in arm_names]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)
    roots = {key: server.scene.add_frame(f"/{key}", show_axes=False) for key in SETUPS}
    tables = {
        key: server.scene.add_box(f"/{key}/table", color=TABLE_COLOR, dimensions=fw.FLEXIV_TABLE_SIZE,
                                  position=fw.FLEXIV_TABLE_CENTER)
        for key in SETUPS
    }
    for key, text in LABELS.items():
        server.scene.add_frame(f"/{key}/robot", show_axes=False, position=fw.ROBOT_BASE_POS)
        server.scene.add_label(f"/{key}/label", text, position=(0.0, 0.0, 1.3))
    viser_robots = {key: ViserUrdf(server, urdfs[key], root_node_name=f"/{key}/robot") for key in SETUPS}

    ready = False

    def q(key: str) -> dict[str, float]:
        return {**dict(zip(arms[key].names, arms[key].q())), **dict(zip(hands[key].names, hands[key].q()))}

    def refresh() -> None:
        if not ready:
            return
        sep = layout_sep.value
        for key, sign in (("vendor", -0.5), ("human_like", 0.5)):
            roots[key].position = (sign * sep, 0.0, 0.0)
            tables[key].visible = show_tables.value
        poses = {key: q(key) for key in SETUPS}
        for key in SETUPS:
            viser_robots[key].update_cfg(np.array([poses[key][n] for n in names]))
        rows = [
            f"| `{n}` | {math.degrees(poses['vendor'][n]):6.1f} | {math.degrees(poses['human_like'][n]):6.1f} |"
            for n in hand_names
            if abs(poses["vendor"][n] - poses["human_like"][n]) > 1e-3
        ]
        readout.content = "\n".join(
            ["| hand joint | vendor (deg) | human-like (deg) |", "|---|---:|---:|", *rows]
            if rows else ["The two hands are in the same pose."]
        )

    with server.gui.add_folder("Layout"):
        layout_sep = server.gui.add_slider("separation (m)", 0.0, 2.0, 0.01, 1.2)
        show_tables = server.gui.add_checkbox("tables", True)
    readout = server.gui.add_markdown("")
    copy_button = server.gui.add_button("Copy vendor pose to human-like")
    reset_button = server.gui.add_button("Reset joints")

    arms, hands = {}, {}
    for key in SETUPS:
        arms[key] = JointSliders(server, f"{LABELS[key]}: arm", urdfs[key], fw.FLEXIV_DEFAULT_ARM_POS,
                                 refresh, False, names=arm_names)
        hands[key] = JointSliders(server, f"{LABELS[key]}: hand", urdfs[key], {}, refresh, True,
                                  names=hand_names, hand_action=True)

    @copy_button.on_click
    def _(_) -> None:
        for sliders in (arms, hands):
            lo = {n: s.min for n, s in sliders["human_like"].sliders.items()}
            hi = {n: s.max for n, s in sliders["human_like"].sliders.items()}
            sliders["human_like"].set({
                n: float(np.clip(v, lo[n], hi[n]))
                for n, v in zip(sliders["vendor"].names, sliders["vendor"].q())
            })

    @reset_button.on_click
    def _(_) -> None:
        for sliders in (*arms.values(), *hands.values()):
            sliders.reset()

    for handle in (layout_sep, show_tables):
        handle.on_update(lambda _: refresh())

    ready = True
    refresh()
    print(f"viser on http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
