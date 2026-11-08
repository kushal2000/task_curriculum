"""A stand-in for aiowuji's WujiController when no hand is connected (robot.hand=fake): its joints are
wherever they are told to be. It has only the parts robot.py uses, so robot.py treats both the same.
"""

import numpy as np


class FakeHand:
    dry_run = False       # it "follows" its targets
    error = None          # it never fails

    def __init__(self, q: np.ndarray) -> None:
        self.state = {"qpos": np.array(q, dtype=float)}

    def set_now(self, attr: str, q: np.ndarray) -> None:   # attr: "q_desired"
        self.state = {"qpos": np.array(q, dtype=float)}

    async def move(self, q: np.ndarray, max_velocity: float) -> None:
        self.set_now("q_desired", q)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass
