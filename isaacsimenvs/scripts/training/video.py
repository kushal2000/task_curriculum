"""Ship `gym.wrappers.RecordVideo` output to wandb.

`RecordVideo` writes .mp4 files into the run directory and stops there — nothing in
play2perfect's stack picks them up, because its video story is the pose-based
`PlayPoseViewerWrapper`, which renders the Kuka URDF in the browser and has no idea what
a cartpole is. So tasks with their own articulation get no visual output at all.

This wrapper closes that gap the generic way: watch the video folder, and log each
finished .mp4 to wandb as it appears. It works for any task because it never looks at
the scene — only at the files the recorder produces.

Wrap *outside* `RecordVideo` so the mp4 is fully written by the time we see it:

    env = gym.wrappers.RecordVideo(env, video_folder=..., ...)
    env = WandbVideoUploader(env, video_folder=...)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

__all__ = ["WandbVideoUploader"]


class WandbVideoUploader(gym.Wrapper):
    """Upload newly-written .mp4 files to wandb under `key`."""

    def __init__(
        self,
        env: gym.Env,
        video_folder: str | Path,
        *,
        key: str = "video",
        fps: int = 30,
        check_every: int = 200,
        max_videos: int | None = None,
    ) -> None:
        super().__init__(env)
        self.video_folder = Path(video_folder)
        self.key = key
        self.fps = fps
        self.check_every = max(1, check_every)
        self.max_videos = max_videos

        self._seen: set[Path] = set()
        self._uploaded = 0
        self._steps = 0

    def _flush(self) -> None:
        if self.max_videos is not None and self._uploaded >= self.max_videos:
            return
        if not self.video_folder.is_dir():
            return

        try:
            import wandb
        except ImportError:  # pragma: no cover - wandb is optional
            return
        if wandb.run is None:
            # --capture_video without --wandb_activate: the files are still on disk.
            return

        for path in sorted(self.video_folder.glob("*.mp4")):
            if path in self._seen:
                continue
            # RecordVideo writes incrementally; a zero-byte file is still being encoded,
            # and uploading it would produce a broken artifact.
            if path.stat().st_size == 0:
                continue
            self._seen.add(path)
            try:
                wandb.log({self.key: wandb.Video(str(path), fps=self.fps, format="mp4")})
                self._uploaded += 1
            except Exception as exc:  # noqa: BLE001 - a bad upload must not kill training
                print(f"[video_utils] failed to upload {path.name}: {exc}", flush=True)
            if self.max_videos is not None and self._uploaded >= self.max_videos:
                return

    def step(self, action):
        result = self.env.step(action)
        self._steps += 1
        if self._steps % self.check_every == 0:
            self._flush()
        return result

    def close(self) -> None:
        # RecordVideo finalises the current clip in its own close(), so flush after it.
        self.env.close()
        self._flush()
