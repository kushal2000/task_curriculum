"""Pose-only interactive HTML capture for Play training.

This module deliberately does not use Isaac cameras, RTX sensors, Replicator, or
the Isaac viewport.  It samples one environment's state tensors and writes a
Three.js/URDF HTML viewer that can be opened locally or logged to WandB.
"""

from __future__ import annotations

import time
import urllib.request
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gymnasium as gym
import numpy as np

from isaacsimenvs.utils.interactive_viewer import create_html, make_embedded_robot, make_url_robot

from .utils.scene_utils import JOINT_NAMES_CANONICAL


REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/kushal2000/depthbasedRL/main/"
ROBOT_URDF_RELATIVE_PATH = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
TABLE_URDF_PATH = REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _quat_wxyz_to_xyzw(quat) -> np.ndarray:
    quat_np = _to_numpy(quat)
    return quat_np[[1, 2, 3, 0]]


def _pose_xyzw(pos, quat_wxyz) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = _to_numpy(pos)
    pose[3:] = _quat_wxyz_to_xyzw(quat_wxyz)
    return pose


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _github_slug_from_remote(remote_url: str | None) -> str | None:
    if not remote_url:
        return None

    remote = remote_url.strip()
    if remote.endswith(".git"):
        remote = remote[:-4]

    prefixes = (
        "git@github.com:",
        "ssh://git@github.com/",
        "https://github.com/",
        "http://github.com/",
    )
    for prefix in prefixes:
        if remote.startswith(prefix):
            slug = remote[len(prefix):]
            parts = slug.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
    return None


def _derive_github_raw_base() -> str:
    remote_url = _git_output(["remote", "get-url", "origin"])
    slug = _github_slug_from_remote(remote_url)
    if slug is None:
        print(
            f"[pose_viewer] Could not derive GitHub origin from {remote_url!r}; "
            f"falling back to {FALLBACK_GITHUB_RAW_BASE}",
            flush=True,
        )
        return FALLBACK_GITHUB_RAW_BASE

    ref = _git_output(["branch", "--show-current"])
    if not ref:
        ref = _git_output(["rev-parse", "HEAD"])
    if not ref:
        print(
            f"[pose_viewer] Could not derive git branch/commit; falling back to {FALLBACK_GITHUB_RAW_BASE}",
            flush=True,
        )
        return FALLBACK_GITHUB_RAW_BASE

    raw_base = f"https://raw.githubusercontent.com/{slug}/{quote(ref, safe='')}/"
    print(f"[pose_viewer] GitHub raw base: {raw_base}", flush=True)
    return raw_base


def _normalize_raw_base(github_raw_base: str | None) -> str:
    base = github_raw_base or _derive_github_raw_base()
    return base if base.endswith("/") else base + "/"


def _check_url(url: str, url_check: str) -> None:
    if url_check == "skip":
        return
    print(f"[pose_viewer] URL check ({url_check}) -> {url}", flush=True)
    start = time.monotonic()
    try:
        request = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(request, timeout=10)
        print(f"[pose_viewer]   PASSED ({time.monotonic() - start:.2f}s)", flush=True)
    except Exception as exc:
        message = f"[pose_viewer]   FAILED ({time.monotonic() - start:.2f}s): {exc}"
        if url_check == "error":
            raise RuntimeError(message) from exc
        print(message, flush=True)


def _raw_url_for_repo_path(path: Path, raw_base: str) -> str | None:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return raw_base + quote(rel.as_posix(), safe="/")


def _rewrite_embedded_urdf_mesh_urls(
    urdf_text: str,
    *,
    source_urdf_path: Path,
    raw_base: str,
) -> str:
    """Make relative mesh filenames loadable from a standalone HTML document."""

    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text

    changed = False
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename:
            continue
        if filename.startswith(("http://", "https://", "data:")):
            continue

        if filename.startswith("package://"):
            rel_name = filename[len("package://") :]
            mesh_path = REPO_ROOT / rel_name
        else:
            candidate = Path(filename)
            if candidate.is_absolute():
                mesh_path = candidate
            elif filename.startswith("assets/"):
                mesh_path = REPO_ROOT / filename
            else:
                mesh_path = source_urdf_path.parent / filename

        mesh_url = _raw_url_for_repo_path(mesh_path, raw_base)
        if mesh_url is None:
            continue
        mesh_elem.set("filename", mesh_url)
        changed = True

    if not changed:
        return urdf_text
    return ET.tostring(root, encoding="unicode")


def object_urdf_for_env(env, env_id: int) -> tuple[str, Path]:
    """Return the procedural object URDF text assigned to one env."""

    urdf_paths = getattr(env, "_object_urdf_paths", None)
    asset_indices = getattr(env, "_object_asset_index_per_env", None)
    if not urdf_paths or asset_indices is None:
        raise RuntimeError(
            "Play env does not expose object URDF mapping. "
            "Expected _object_urdf_paths and _object_asset_index_per_env."
        )

    asset_index = int(asset_indices[env_id].detach().cpu().item())
    urdf_path = Path(urdf_paths[asset_index])
    return urdf_path.read_text(encoding="utf-8"), urdf_path


def table_urdf_for_env(env, env_id: int) -> tuple[str, Path]:
    """Return the table URDF text assigned to one env: its scaled variant if the task made them,
    else the task's own ``assets.table_urdf`` (the Flexiv tasks' is not Play's)."""

    table_paths = getattr(env, "_table_urdf_paths", None)
    if table_paths:
        urdf_path = Path(table_paths[env_id % len(table_paths)])
        return urdf_path.read_text(encoding="utf-8"), urdf_path
    cfg_path = getattr(getattr(getattr(env, "cfg", None), "assets", None), "table_urdf", None)
    urdf_path = TABLE_URDF_PATH
    if cfg_path:
        urdf_path = Path(cfg_path)
        if not urdf_path.is_absolute():
            urdf_path = REPO_ROOT / urdf_path
    return urdf_path.read_text(encoding="utf-8"), urdf_path


def hole_urdf_for_env(env, env_id: int) -> tuple[str, Path] | tuple[None, None]:
    """Return the peg-in-hole receptive URDF text when the env exposes one."""

    hole_paths = getattr(env, "_hole_urdf_paths", None)
    if not hole_paths:
        return None, None
    urdf_path = Path(hole_paths[env_id % len(hole_paths)])
    return urdf_path.read_text(encoding="utf-8"), urdf_path


def capture_pose_viewer_frame(env, env_id: int) -> dict[str, Any]:
    """Capture one env-local frame from a live PlayEnv."""

    if env_id < 0 or env_id >= env.num_envs:
        raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={env.num_envs}")

    origin = env.scene.env_origins[env_id]

    if hasattr(env, "_perm_lab_to_canon"):
        joint_pos = env.robot.data.joint_pos[env_id, env._perm_lab_to_canon]
        spec = getattr(env, "robot_spec", None)
        joint_names = list(spec.joint_names_canonical if spec else JOINT_NAMES_CANONICAL)
    else:
        joint_pos = env.robot.data.joint_pos[env_id]
        joint_names = list(env.robot.data.joint_names)

    robot_root_pos = env.robot.data.root_pos_w[env_id] - origin
    table_pos = env.table.data.root_pos_w[env_id] - origin
    hole = getattr(env, "hole", None)

    frame = {
        "env_id": int(env_id),
        "robot_joint_names": joint_names,
        "robot_joint_pos": _to_numpy(joint_pos),
        "robot_base_pose": _pose_xyzw(robot_root_pos, env.robot.data.root_quat_w[env_id]),
        "table_pose": _pose_xyzw(table_pos, env.table.data.root_quat_w[env_id]),
    }
    # The cloth envs have no rigid manipuland or goal marker: the sheet is drawn as deformables.
    if hasattr(env, "object") and hasattr(env, "goal_viz"):
        object_pos = env.object.data.root_pos_w[env_id] - origin
        goal_pos = env.goal_viz.data.root_pos_w[env_id] - origin
        frame["object_pose"] = _pose_xyzw(object_pos, env.object.data.root_quat_w[env_id])
        frame["goal_pose"] = _pose_xyzw(goal_pos, env.goal_viz.data.root_quat_w[env_id])
    if hole is not None:
        hole_pos = hole.data.root_pos_w[env_id] - origin
        frame["hole_pose"] = _pose_xyzw(hole_pos, hole.data.root_quat_w[env_id])
    return frame


def build_pose_viewer_html(
    *,
    frames: list[dict[str, Any]],
    object_urdf_text: str,
    table_urdf_text: str,
    hole_urdf_text: str | None = None,
    object_urdf_path: Path | None = None,
    table_urdf_path: Path | None = None,
    hole_urdf_path: Path | None = None,
    github_raw_base: str | None = None,
    url_check: str = "skip",
    deformables: dict | None = None,
    draw_rigid_object: bool = True,
    robot_urdf_relative_path: str = ROBOT_URDF_RELATIVE_PATH,
) -> str:
    """Build a self-contained-ish viewer HTML string from captured frames.

    Object and table URDFs are embedded.  The robot URDF is URL-backed because
    its hand references mesh files that the browser must fetch, so
    ``robot_urdf_relative_path`` (repo-relative) and its meshes must be pushed.
    """

    if not frames:
        raise ValueError("Cannot build pose viewer from zero frames.")

    raw_base = _normalize_raw_base(github_raw_base)
    robot_urdf_url = raw_base + robot_urdf_relative_path
    _check_url(robot_urdf_url, url_check)
    if object_urdf_path is not None:
        object_urdf_text = _rewrite_embedded_urdf_mesh_urls(
            object_urdf_text,
            source_urdf_path=object_urdf_path,
            raw_base=raw_base,
        )
    if table_urdf_path is not None:
        table_urdf_text = _rewrite_embedded_urdf_mesh_urls(
            table_urdf_text,
            source_urdf_path=table_urdf_path,
            raw_base=raw_base,
        )
    if hole_urdf_text is not None and hole_urdf_path is not None:
        hole_urdf_text = _rewrite_embedded_urdf_mesh_urls(
            hole_urdf_text,
            source_urdf_path=hole_urdf_path,
            raw_base=raw_base,
        )

    timestamps = np.arange(len(frames), dtype=np.float32) / 60.0
    robots = [
        make_url_robot(name="robot", urdf_url=robot_urdf_url, animated=True),
        make_embedded_robot(name="table", urdf_text=table_urdf_text),
    ]
    object_poses = {
        "table": np.stack([frame["table_pose"] for frame in frames]),
    }
    # `draw_rigid_object=False` is for tasks whose manipuland is NOT the spawned rigid tool.
    # Cloth is one: the sheet replaces the spawned object after the scene is built, but the tool
    # pool is still constructed, so `object_urdf_text` is a hammer. Drawn unconditionally, that
    # puts a hammer at `object_pose` and a GREEN hammer at `goal_pose` -- and the green is close
    # enough to the fold ghost's green to read as part of the goal. Both are meaningless for
    # cloth: the tool is inert, and `goal_viz` is a pure rendering artefact that
    # `ClothEnv._drive_goal_marker` overwrites every step. The sheet supplies its own object and
    # goal as deformable channels.
    if draw_rigid_object:
        robots.append(make_embedded_robot(name="object", urdf_text=object_urdf_text))
        robots.append(
            make_embedded_robot(
                name="goal",
                urdf_text=object_urdf_text,
                color_override=(0.20, 0.72, 0.31),
            )
        )
        object_poses["object"] = np.stack([frame["object_pose"] for frame in frames])
        object_poses["goal"] = np.stack([frame["goal_pose"] for frame in frames])
    if hole_urdf_text is not None and all("hole_pose" in frame for frame in frames):
        robots.insert(2, make_embedded_robot(name="hole", urdf_text=hole_urdf_text))
        object_poses["hole"] = np.stack([frame["hole_pose"] for frame in frames])

    return create_html(
        joint_names=frames[0]["robot_joint_names"],
        robot_joint_positions=np.stack([frame["robot_joint_pos"] for frame in frames]),
        robots=robots,
        object_poses=object_poses,
        robot_base_poses=np.stack([frame["robot_base_pose"] for frame in frames]),
        timestamps=timestamps,
        deformables=deformables,
    )


class PlayPoseViewerWrapper(gym.Wrapper):
    """Gym wrapper that periodically writes pose-only interactive HTML rollouts."""

    def __init__(
        self,
        env: gym.Env,
        *,
        output_dir: str | Path,
        capture_len: int,
        capture_interval: int,
        env_id: int = 0,
        wandb_key: str = "interactive_viewer",
        github_raw_base: str | None = None,
        url_check: str = "skip",
    ) -> None:
        super().__init__(env)
        if capture_len <= 0:
            raise ValueError(f"capture_viewer_len must be > 0, got {capture_len}")
        if url_check not in {"skip", "warn", "error"}:
            raise ValueError(f"capture_viewer_url_check must be skip/warn/error, got {url_check}")

        inner = self.env.unwrapped
        if env_id < 0 or env_id >= inner.num_envs:
            raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={inner.num_envs}")

        self.output_dir = Path(output_dir)
        self.capture_len = int(capture_len)
        self.capture_interval = int(capture_interval)
        self.env_id = int(env_id)
        self.wandb_key = wandb_key
        self.github_raw_base = github_raw_base
        self.url_check = url_check
        self._object_urdf_text, self._object_urdf_path = self._object_urdf(inner)
        self._table_urdf_text, self._table_urdf_path = table_urdf_for_env(
            inner, self.env_id
        )
        self._hole_urdf_text, self._hole_urdf_path = hole_urdf_for_env(inner, self.env_id)

        self._step = 0
        self._capture_index = 0
        self._frames: list[dict[str, Any]] | None = []
        # Per-capture buffers of (C, H, W) student-camera frames (numpy, [0, 1] floats).
        # _depth_frames is the policy's actual input (noise-on path).
        # _depth_frames_clean is the same view without depth augmentation,
        # useful for A/B-comparing the noise pipeline on the same camera pose.
        # Both stay empty when the env doesn't expose `get_student_obs`.
        self._depth_frames: list = []
        self._depth_frames_clean: list = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[pose_viewer] enabled: "
            f"env_id={self.env_id} len={self.capture_len} interval={self.capture_interval} "
            f"output_dir={self.output_dir}",
            flush=True,
        )

    def step(self, action):
        result = self.env.step(action)
        self._step += 1

        if self._frames is None and self.capture_interval > 0 and self._step % self.capture_interval == 0:
            self._frames = []
            self._depth_frames = []
            self._depth_frames_clean = []

        if self._frames is not None:
            self._frames.append(self._capture_frame())
            depth_frame = self._capture_student_image()
            if depth_frame is not None:
                self._depth_frames.append(depth_frame)
            clean_frame = self._capture_student_image_clean()
            if clean_frame is not None:
                self._depth_frames_clean.append(clean_frame)
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()

        return result

    def _capture_student_image(self):
        """Pull the env_id slice of the student's input image, if the env exposes one.

        Returns a (C, H, W) numpy array in [0, 1] (depth after env-side
        ``window_normalize`` preproc), or ``None`` when student_obs isn't
        configured. Called once per pose-viewer-captured step, so cost is
        capped at one extra `get_student_obs()` per (capture_len * frequency).
        """
        inner = self.env.unwrapped
        if not hasattr(inner, "get_student_obs"):
            return None
        student_cfg = getattr(getattr(inner, "cfg", None), "student_obs", None)
        if student_cfg is not None and not getattr(student_cfg, "enabled", True):
            return None
        try:
            student_obs = inner.get_student_obs()
        except Exception as exc:
            print(f"[pose_viewer] get_student_obs failed: {exc}", flush=True)
            return None
        image = student_obs.get("image") if isinstance(student_obs, dict) else None
        if image is None:
            return None
        return image[self.env_id].detach().cpu().numpy()

    def _capture_student_image_clean(self):
        """Same env_id slice as the noisy capture, but with depth noise bypassed.

        Reads the post-preprocess, post-crop tensor stashed by
        `scene_utils.read_student_camera_image` so the two videos differ
        ONLY in whether `_apply_depth_noise` was applied. When
        `use_depth_aug=False` this equals the noisy capture.

        Returns a (1, H, W) numpy array in [0, 1], or `None` when the env
        hasn't run a depth read yet.
        """
        inner = self.env.unwrapped
        clean = getattr(inner, "_last_student_image_clean", None)
        if clean is None:
            return None
        return clean[self.env_id].detach().cpu().numpy()

    def close(self) -> None:
        if self._frames:
            self._finalize_capture(partial=True)
        return self.env.close()

    def _object_urdf(self, inner):
        """The manipuland's URDF ``(text, path)``; Play's procedural tool."""
        return object_urdf_for_env(inner, self.env_id)

    def _capture_frame(self) -> dict:
        """One frame of viewer state. Subclasses extend this to add task-specific geometry."""
        return capture_pose_viewer_frame(self.env.unwrapped, self.env_id)

    def _build_html(self, frames: list[dict]) -> str:
        """Render captured frames to HTML. Subclasses override to pass extra channels."""
        return build_pose_viewer_html(
            frames=frames,
            object_urdf_text=self._object_urdf_text,
            table_urdf_text=self._table_urdf_text,
            hole_urdf_text=self._hole_urdf_text,
            object_urdf_path=self._object_urdf_path,
            table_urdf_path=self._table_urdf_path,
            hole_urdf_path=self._hole_urdf_path,
            github_raw_base=self.github_raw_base,
            url_check=self.url_check,
            robot_urdf_relative_path=self.env.unwrapped.cfg.assets.robot_urdf,
        )

    def _finalize_capture(self, *, partial: bool = False) -> None:
        assert self._frames is not None
        frames = self._frames
        if not frames:
            self._frames = None
            return

        suffix = "partial" if partial else f"step_{self._step:09d}"
        html_path = self.output_dir / f"pose_viewer_{suffix}_{self._capture_index:04d}.html"
        html_text = self._build_html(frames)
        html_path.write_text(html_text, encoding="utf-8")
        print(f"[pose_viewer] wrote {len(frames)} frames to {html_path}", flush=True)
        self._log_wandb(html_text)

        self._capture_index += 1
        self._frames = None
        self._depth_frames = []
        self._depth_frames_clean = []

    def _log_wandb(self, html_text: str) -> None:
        try:
            import wandb
        except Exception:
            return

        if wandb.run is None:
            return

        try:
            wandb.log({self.wandb_key: wandb.Html(html_text)})
            print(f"[pose_viewer] logged WandB Html key={self.wandb_key} step={self._step}", flush=True)
        except Exception as exc:
            print(f"[pose_viewer] WandB log failed: {exc}", flush=True)

        if not self._depth_frames and not self._depth_frames_clean:
            return

        import numpy as np

        def _to_uint8_grayscale_rgb(frames: list) -> np.ndarray:
            video = np.stack(frames, axis=0)                        # (T, C, H, W) float in [0, 1]
            video = (np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, axis=1)                 # (T, 3, H, W) for wandb.Video single-video tiling
            return video

        for label, frames in (("depth", self._depth_frames), ("depth_clean", self._depth_frames_clean)):
            if not frames:
                continue
            try:
                video = _to_uint8_grayscale_rgb(frames)
                key = f"{self.wandb_key}_{label}"
                wandb.log({key: wandb.Video(video, fps=30, format="mp4")})
                print(f"[pose_viewer] logged WandB Video key={key} "
                      f"shape={video.shape} dtype={video.dtype}", flush=True)
            except Exception as exc:
                print(f"[pose_viewer] WandB {label} video log failed: {exc}", flush=True)


# ==================================================================================================
# The cloth tasks' viewer -- Interactive HTML capture for cloth-folding training, logged to wandb.
#
#
# Extends the Play pose viewer with the one thing a rigid-pose viewer cannot show: the sheet. A cloth
# has no root transform -- its vertex positions *are* its state -- so it travels as a deformable
# channel (static triangle indices plus a per-frame vertex trajectory) rather than as a (T, 7) pose.
#
# **Nothing is uploaded to wandb as an asset.** The sheet is a procedural grid, so its geometry is
# plain numbers in the payload: 81 vertices x 3 floats per frame, order 1 KB. The robot's ~41 MB of
# STLs are fetched by the browser from the public repo via `--capture_viewer_github_raw_base`, exactly
# as the Play viewer already does.
#
# Three things are drawn that the Play viewer has no notion of:
#
# * the moving half, in the colour the renders use for it
# * the stationary half, likewise
# * the fold target as a translucent ghost -- the same `folded_half_w()` the reward and the success
#   criterion are computed from, so what is on screen is what is being optimised
#
# and two things are deliberately NOT drawn. The Play viewer renders the spawned rigid tool at
# `object_pose` and a green copy of it at `goal_pose`. For cloth the tool pool is still built
# (`Cloth.yaml` pins `handle_head_types: ["hammer"]`) but the sheet replaces the spawned object, so
# both are hammers: one inert, one a green hammer sitting at a `goal_viz` marker that
# `ClothEnv._drive_goal_marker` overwrites every step. `draw_rigid_object=False` suppresses them.
# This is the same confusion the offscreen renderer had before `_hide_goal_viz`.
# ==================================================================================================

#: Matches the offscreen renderer (`scripts/visualization/render_newton.py`) so a wandb clip and an mp4 of the same
#: run are directly comparable. Nothing here may be blue -- the table, floor and sky all are.
MOVING_COLOR = (0.95, 0.30, 0.10)      # vermilion: the half that must travel
STATIONARY_COLOR = (0.98, 0.78, 0.15)  # gold: the half that must not
GOAL_COLOR = (0.15, 0.85, 0.35)        # green: where the moving half must end up
GOAL_OPACITY = 0.45


def capture_cloth_viewer_frame(env, env_id: int) -> dict[str, Any]:
    """One frame: everything the Play viewer captures, plus the sheet and its fold target."""
    frame = capture_pose_viewer_frame(env, env_id)
    origin = env.scene.env_origins[env_id]

    parts = env._particles_w()[env_id] - origin
    frame["cloth_vertices"] = parts.detach().cpu().numpy().tolist()
    # The ghost is the goal, over every particle of the moving half rather than the four scored
    # keypoints -- `folded_half_w` shares `_folded_w` with `fold_targets_w`, so the drawing cannot
    # drift from the criterion.
    ghost = env.folded_half_w()[env_id] - origin
    frame["goal_vertices"] = ghost.detach().cpu().numpy().tolist()
    # The rigid-cloth pool: draw its actual plates, not the particle grid they carry (a grid drawn as
    # one sheet stretches across every plate boundary as the plates rotate).
    if hasattr(env, "viewer_plates"):
        shape, verts, tris, moving = env.viewer_plates(env_id)
        frame["plate_shape"] = int(shape)
        frame["plate_vertices"] = (verts - origin.detach().cpu().numpy()).tolist()
        frame["plate_tris"] = tris.tolist()
        frame["plate_moving"] = moving.tolist()
    # Reported so the HTML caption can carry the same numbers as the mp4 HUD.
    frame["fold_err"] = float(env.fold_error()[env_id])
    frame["footprint"] = float(env.footprint_ratio()[env_id])
    return frame


class ClothPoseViewerWrapper(PlayPoseViewerWrapper):
    """Robot + table, plus the cloth sheet and its fold target (no rigid manipuland)."""

    def _object_urdf(self, inner):
        return "", None

    def _capture_frame(self):  # pragma: no cover - exercised in training
        return capture_cloth_viewer_frame(self.env.unwrapped, self.env_id)

    def _build_html(self, frames: list[dict[str, Any]]) -> str:
        import numpy as np

        from isaacsimenvs.tasks.cloth.utils import geometry

        inner = self.env.unwrapped
        res = int(inner._sheet_geom.resolution)
        sheet_tris = geometry.grid_triangles(res)
        ghost_tris = geometry.half_triangles(res, reflected=False)

        sheet = np.asarray([f["cloth_vertices"] for f in frames], dtype=float)
        ghost = np.asarray([f["goal_vertices"] for f in frames], dtype=float)
        # Rigid-cloth pool: the plates themselves, when every frame shows the same shape (a capture
        # spanning a reset onto another shape falls back to the particle grid).
        plates = None
        if all("plate_vertices" in f for f in frames) and len({f["plate_shape"] for f in frames}) == 1:
            plates = (
                np.asarray([f["plate_vertices"] for f in frames], dtype=float),
                np.asarray(frames[0]["plate_tris"], dtype=int).reshape(-1, 3),
                np.asarray(frames[0]["plate_moving"], dtype=bool),
            )

        # Split the sheet by which half each triangle belongs to, by MAJORITY vertex. An
        # all-three test leaves every triangle touching the excluded hinge row unclassified --
        # a quarter of the sheet -- which reads as a third stripe rather than two halves.
        moving = set(geometry.half_indices(res, positive=True))
        tri = np.asarray(sheet_tris, dtype=int).reshape(-1, 3)
        in_moving = np.isin(tri, list(moving)).sum(axis=1) >= 2

        return build_pose_viewer_html(
            frames=frames,
            object_urdf_text=self._object_urdf_text,
            table_urdf_text=self._table_urdf_text,
            hole_urdf_text=self._hole_urdf_text,
            object_urdf_path=self._object_urdf_path,
            table_urdf_path=self._table_urdf_path,
            hole_urdf_path=self._hole_urdf_path,
            github_raw_base=self.github_raw_base,
            url_check=self.url_check,
            # The task's own robot (the robot overlay sets it); the default is the Kuka + Sharpa.
            robot_urdf_relative_path=inner.cfg.assets.robot_urdf,
            # The sheet is the manipuland, not the spawned rigid tool -- see the note in
            # `build_pose_viewer_html`. Without this the viewer draws a hammer at `object_pose`
            # and a green hammer at `goal_pose`, on top of the three cloth channels below.
            draw_rigid_object=False,
            deformables={
                "cloth_moving": {
                    "indices": (plates[1][plates[2]] if plates else tri[in_moving]).reshape(-1).tolist(),
                    "vertices": plates[0] if plates else sheet,
                    "color": MOVING_COLOR,
                },
                "cloth_stationary": {
                    "indices": (plates[1][~plates[2]] if plates else tri[~in_moving]).reshape(-1).tolist(),
                    "vertices": plates[0] if plates else sheet,
                    "color": STATIONARY_COLOR,
                },
                "fold_goal": {
                    "indices": ghost_tris,
                    "vertices": ghost,
                    "color": GOAL_COLOR,
                    "opacity": GOAL_OPACITY,
                },
            },
        )
