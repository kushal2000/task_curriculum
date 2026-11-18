"""Wuji Hand 2 sysid recordings -> 2 s replay windows, exactly as aiowuji sysid_fit.py cuts them.

Pure numpy (no Isaac Sim), shared by sysid_replay_physx.py (the PhysX rollouts) and
sysid_replay_report.py (the scoring / plots).

Frames are the hand's 1 ms telemetry frames (seq - seq[0]); rows with a repeated seq are dropped,
missing frames are NaN. The target the firmware used at frame f is q_cmd at the last host tick that
had seen frame f - LAG (2 ms command lag, firmware_law.lag_steps).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNS = {  # run dir -> (mount, gravity in the vendor MJCF wrist frame, m/s^2)
    "sysid-20260930T173832Z": ("fingers-up", (0.0, 0.0, 9.81)),
    "sysid-20261009T211407Z": ("palm-down", (0.0, 9.81, 0.0)),
}
JOINT_NAMES = [f"{f}_S{i}" for f in ("thumb", "index", "middle", "ring", "pinky") for i in range(1, 5)]
LAG = 2
WINDOW_FRAMES = 2000  # 2 s at 1 kHz
FRAME_DT = 1e-3


@dataclass(eq=False)
class Trial:
    file: str            # "<run>/trial_NNNN.npz" (the fit JSON's key)
    run: str
    mount: str
    gravity: np.ndarray  # (3,) wrist frame
    finger: str
    joints: list         # (4,) excited columns (0..19)
    kp: np.ndarray       # (20,) A/rad as sent
    kd: np.ndarray       # (20,) A s/rad
    effort_limit: float  # A
    q: np.ndarray        # (frames, 4) rad, NaN where no frame
    qd: np.ndarray       # (frames, 4)
    q_all: np.ndarray    # (frames, 20) rad, NaN where no frame
    qd_all: np.ndarray   # (frames, 20)
    cmd: np.ndarray      # (frames, 20) the target in use at each frame (lag applied)
    q_cmd_sent: np.ndarray  # (20,) the first sent target (non-excited joints hold it)


def load_trials(data_dir: str | Path) -> list[Trial]:
    data_dir = Path(data_dir)
    trials = []
    for run, (mount, gravity) in RUNS.items():
        manifest = json.loads((data_dir / run / "manifest.json").read_text())
        for entry in manifest["trials"]:
            d = np.load(data_dir / run / entry["file"])
            joints = list(entry["joints"])
            seq = d["seq"].astype(np.int64) - int(d["seq"][0])
            fresh = np.r_[True, np.diff(seq) > 0]
            n = int(seq[-1]) + 1
            arrays = {}
            for key in ("q", "qd"):
                arrays[key] = np.full((n, 20), np.nan)
                arrays[key][seq[fresh]] = d[key][fresh]
            tick = np.searchsorted(seq, np.arange(n), side="right") - 1
            cmd = d["q_cmd"][tick[np.maximum(np.arange(n) - LAG, 0)]].astype(float)
            trials.append(Trial(
                file=f"{run}/{entry['file']}", run=run, mount=mount, gravity=np.array(gravity),
                finger=entry["finger"], joints=joints, kp=d["kp"].astype(float), kd=d["kd"].astype(float),
                effort_limit=float(manifest["effort_limit"]),
                q=arrays["q"][:, joints], qd=arrays["qd"][:, joints], q_all=arrays["q"], qd_all=arrays["qd"],
                cmd=cmd, q_cmd_sent=d["q_cmd"][0].astype(float)))
    return trials


@dataclass
class Window:
    trial: int   # index into the trial list
    start: int   # first frame (a recorded one)
    stop: int    # last frame compared (exclusive of nothing: frames start..stop)


def make_windows(trials: list[Trial], length: int = WINDOW_FRAMES) -> list[Window]:
    """sysid_fit.windows(): consecutive `length`-frame windows, each moved to its first recorded frame."""
    out = []
    for i, t in enumerate(trials):
        frames = len(t.q)
        measured = np.isfinite(t.q[:, 0])
        for start in range(0, max(frames - length, 1), length):
            start = int(np.flatnonzero(measured[start:])[0]) + start
            stop = min(start + length, frames - 1)
            out.append(Window(i, start, stop))
    return out


def interp_frames(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """x (frames, k) sampled at integer frames (NaN = missing), read at fractional frames f -> (len(f), k).
    Linear between the two neighbouring frames; NaN when either is missing or out of range."""
    f = np.asarray(f, float)
    lo = np.floor(f).astype(int)
    hi = lo + 1
    w = (f - lo)[:, None]
    ok = (lo >= 0) & (hi < len(x))
    lo_c, hi_c = np.clip(lo, 0, len(x) - 1), np.clip(hi, 0, len(x) - 1)
    a, b = x[lo_c], x[hi_c]
    exact = np.isclose(w[:, 0], 0.0)
    out = np.where(exact[:, None], a, (1 - w) * a + w * b)
    out[~ok & ~(exact & (lo >= 0) & (lo < len(x)))] = np.nan
    return out
