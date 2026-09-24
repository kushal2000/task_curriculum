"""G1 + Wuji Hand 2 models and joint trajectories, shared by the MuJoCo and Isaac Sim replayers.

Pure numpy + stdlib on purpose: the Isaac Sim venv has no ``mujoco``, and the MuJoCo side has no
Isaac, so anything both replayers need is computed here from the model files directly.

Trajectory file format (``.npz``), written by :func:`save_trajectory`:

    joint_names  (J,)    str    joint names as in the MJCF / URDF, any subset, any order
    q            (T, J)  float  joint positions, radians
    fps          ()      float  frame rate
    root_pos     (T, 3)  float  optional: pelvis world position, selects the floating-base model
    root_quat    (T, 4)  float  optional: pelvis world orientation, (w, x, y, z)

Joints a trajectory leaves out are held at the model's ``stand`` keyframe.

The wuji-hand-teleop clip layout (a directory holding ``clip.json``, ``arm_q.npz`` and
``hand_q20.npz``) loads too, so the upstream clips replay unchanged.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "g1_wuji2_description"

VARIANTS = (29, 23)

#: Pelvis height the upstream fixed-base models weld at. Also where the Isaac Sim replayer
#: places the fixed-base URDF root, so both simulators show the robot at the same height.
FIXED_PELVIS_HEIGHT = 0.79


def model_path(variant: int, kind: str) -> Path:
    """Path to one model file. ``kind``: ``fixed`` / ``floating`` MJCF, their ``scene_*``
    wrappers (``scene_fixed`` / ``scene_floating``), or ``urdf``."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
    stem = f"g1_{variant}_wuji2"
    name = {
        "fixed": f"{stem}_fixed.xml",
        "floating": f"{stem}.xml",
        "scene_fixed": f"scene_{stem}_fixed.xml",
        "scene_floating": f"scene_{stem}.xml",
        "urdf": f"{stem}.urdf",
    }[kind]
    path = ASSET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/g1_wuji/fetch_assets.sh"
        )
    return path


def mjcf_joint_names(variant: int) -> list[str]:
    """Hinge joints of the fixed-base MJCF, in qpos order (document order of named joints)."""
    root = ET.parse(model_path(variant, "fixed")).getroot()
    return [
        j.get("name") for j in root.find("worldbody").iter("joint") if j.get("name")
    ]


def stand_pose(variant: int) -> dict[str, float]:
    """The ``stand`` keyframe of the fixed-base MJCF, as joint name -> position."""
    root = ET.parse(model_path(variant, "fixed")).getroot()
    keys = [k for k in root.iter("key") if k.get("name") == "stand"]
    if not keys:
        raise ValueError("fixed-base MJCF has no 'stand' keyframe")
    qpos = [float(x) for x in keys[0].get("qpos").split()]
    names = mjcf_joint_names(variant)
    if len(qpos) != len(names):
        raise ValueError(
            f"stand keyframe has {len(qpos)} values for {len(names)} joints"
        )
    return dict(zip(names, qpos))


@dataclass
class Trajectory:
    joint_names: list[str]
    q: np.ndarray  # (T, J)
    fps: float
    root_pos: np.ndarray | None = None  # (T, 3)
    root_quat: np.ndarray | None = None  # (T, 4), wxyz

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float64)
        if self.q.ndim != 2 or self.q.shape[1] != len(self.joint_names):
            raise ValueError(
                f"q has shape {self.q.shape}, expected (T, {len(self.joint_names)})"
            )
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names has duplicates")
        if (self.root_pos is None) != (self.root_quat is None):
            raise ValueError("root_pos and root_quat must be given together")
        if self.root_pos is not None:
            self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
            self.root_quat = np.asarray(self.root_quat, dtype=np.float64)
            if self.root_pos.shape != (len(self), 3) or self.root_quat.shape != (
                len(self),
                4,
            ):
                raise ValueError("root_pos must be (T, 3) and root_quat (T, 4)")
            self.root_quat = self.root_quat / np.linalg.norm(
                self.root_quat, axis=1, keepdims=True
            )

    def __len__(self) -> int:
        return self.q.shape[0]

    @property
    def floating(self) -> bool:
        return self.root_pos is not None

    def full(self, joint_order: list[str], default: dict[str, float]) -> np.ndarray:
        """(T, len(joint_order)) positions in ``joint_order``; absent joints take ``default``."""
        unknown = sorted(set(self.joint_names) - set(joint_order))
        if unknown:
            raise KeyError(f"trajectory joints not in the model: {unknown}")
        out = np.tile([default[n] for n in joint_order], (len(self), 1))
        index = {n: i for i, n in enumerate(joint_order)}
        out[:, [index[n] for n in self.joint_names]] = self.q
        return out


def save_trajectory(path, joint_names, q, fps, root_pos=None, root_quat=None) -> None:
    """Write a trajectory in the ``.npz`` format this module documents."""
    traj = Trajectory(list(joint_names), q, float(fps), root_pos, root_quat)
    arrays = {
        "joint_names": np.array(traj.joint_names),
        "q": traj.q,
        "fps": np.float64(traj.fps),
    }
    if traj.floating:
        arrays.update(root_pos=traj.root_pos, root_quat=traj.root_quat)
    np.savez(path, **arrays)


def load_trajectory(path) -> Trajectory:
    """Load a ``.npz`` trajectory, or a wuji-hand-teleop clip directory."""
    path = Path(path)
    if path.is_dir():
        return _load_teleop_clip(path)
    z = np.load(path, allow_pickle=False)
    root = (z["root_pos"], z["root_quat"]) if "root_pos" in z.files else (None, None)
    return Trajectory(
        [str(n) for n in z["joint_names"]], z["q"], float(z["fps"]), *root
    )


def _load_teleop_clip(clip_dir: Path) -> Trajectory:
    """wuji-hand-teleop ``prepare_clip`` output: arms 7+7, hands 20+20, at ``rate_hz``.

    Clip joint names are the short forms (``left_elbow``, ``l_thumb_mcp``); the composed model
    names them ``left_elbow_joint`` and ``left_wuji_l_thumb_mcp``.
    """
    meta = json.loads((clip_dir / "clip.json").read_text())
    arm = np.load(clip_dir / "arm_q.npz")
    hand = np.load(clip_dir / "hand_q20.npz")
    names, cols = [], []
    for side in ("left", "right"):
        names += [f"{n}_joint" for n in meta["arm_joint_names"][side]]
        cols.append(arm[side])
    for side in ("left", "right"):
        names += [f"{side}_wuji_{n}" for n in meta["hand_joint_names"][side]]
        cols.append(hand[side])
    return Trajectory(names, np.concatenate(cols, axis=1), float(meta["rate_hz"]))
