"""Success-driven thickness adaptation; no prebuilt thickness variants.

Every completed episode counts once, including guard failures and timeouts.
Success means the task's final success (fold AND release when release is enabled).
A geometry change starts a fresh window; unfinished episodes are not evidence.
"""
from collections import deque
import math


class ThicknessCurriculum:
    def __init__(self, cfg):
        self.cfg = cfg
        if not (math.isfinite(cfg.start) and math.isfinite(cfg.target) and 0 < cfg.target <= cfg.start):
            raise ValueError('Thickness curriculum requires 0 < target <= start')
        if not 0 < cfg.reduction < 1 or not 0 < cfg.success_threshold <= 1 or cfg.window_episodes < 1:
            raise ValueError('Invalid thickness curriculum reduction, threshold or episode window')
        self.thickness = float(cfg.start)
        self.recent = deque(maxlen=int(cfg.window_episodes))
        self.advances = 0
        self.last_advance_success_rate = 0.

    @property
    def success_rate(self):
        return sum(self.recent)/len(self.recent) if self.recent else 0.

    def observe(self, successes):
        """Accept completed episodes only; return whether geometry must change."""
        self.recent.extend(bool(value) for value in successes)
        if (len(self.recent) < self.recent.maxlen or self.success_rate < self.cfg.success_threshold
                or self.thickness <= self.cfg.target):
            return False
        self.last_advance_success_rate = self.success_rate
        self.thickness = max(float(self.cfg.target), self.thickness*float(self.cfg.reduction))
        self.advances += 1
        self.recent.clear()
        return True

    def metrics(self):
        return {'curriculum/thickness_mm': self.thickness*1000,
                'curriculum/success_rate': self.success_rate,
                'curriculum/completed_episodes': len(self.recent),
                'curriculum/advances': self.advances,
                'curriculum/last_advance_success_rate': self.last_advance_success_rate}

    def state_dict(self):
        return dict(thickness=self.thickness, recent=list(self.recent), advances=self.advances,
                    last_advance_success_rate=self.last_advance_success_rate)

    def load_state_dict(self, state):
        thickness = float(state['thickness'])
        if not self.cfg.target <= thickness <= self.cfg.start:
            raise ValueError('Checkpoint thickness is outside the curriculum bounds')
        self.thickness = thickness
        self.recent.clear()
        self.recent.extend(bool(value) for value in state['recent'])
        self.advances = int(state['advances'])
        self.last_advance_success_rate = float(state['last_advance_success_rate'])
