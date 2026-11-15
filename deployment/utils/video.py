"""H.264 mp4 files: VideoWriter (RGB frames -> ffmpeg) and VideoReader (frame k / the frame at time t).

recorder.py writes the camera to <recording>.camera.mp4 with VideoWriter, one frame per camera frame,
and puts each frame's time in the .jsonl (topic camera_frame: seq, t, frame = index in the mp4).
playback.py reads it back with VideoReader. ffmpeg comes from imageio-ffmpeg (pip install
imageio-ffmpeg); its build has NVENC (the GPU's video encoder, no CPU load) and libx264.

Codecs (record.camera_codec) and quality (record.camera_quality, lower = better and bigger):
    h264_nvenc  GPU encoder, constant quality -cq (0-51, ~19-23 looks like the original)
    libx264     CPU encoder, -crf (same scale); use it on a machine without an NVIDIA GPU
"""

import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


ENCODERS = {
    "h264_nvenc": lambda q: ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(q),
                             "-b:v", "0"],
    "libx264": lambda q: ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(q)],
}


class VideoWriter:
    """RGB frames (H, W, 3) uint8, all the same size, -> an H.264 mp4 at a nominal fps."""

    def __init__(self, path: Path, width: int, height: int, fps: float, codec: str = "h264_nvenc",
                 quality: int = 21) -> None:
        if codec not in ENCODERS:
            raise ValueError(f"codec {codec}: {' | '.join(ENCODERS)}")
        self.path, self.width, self.height, self.n = Path(path), width, height, 0
        cmd = [ffmpeg_exe(), "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
               *ENCODERS[codec](quality), "-pix_fmt", "yuv420p", "-g", str(int(round(2 * fps))),
               "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        """Raises BrokenPipeError if ffmpeg has stopped (its error is on stderr)."""
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"frame {rgb.shape}, the video is {self.height} x {self.width} x 3")
        self.proc.stdin.write(np.ascontiguousarray(rgb, np.uint8).data)
        self.n += 1

    def close(self) -> int:
        """Finish the file; ffmpeg's exit code (0 = ok)."""
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        return self.proc.wait()


class VideoReader:
    """Frames of an mp4 by index, or by time given each frame's time (times[k] = frame k's t).
    Reads forward; seeks only when asked for an earlier frame or one far ahead."""

    def __init__(self, path: Path, times: np.ndarray | None = None) -> None:
        import cv2
        self.cv2, self.path = cv2, Path(path)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.times = times
        self.k, self.rgb = -1, None   # the frame last read

    def frame(self, k: int) -> np.ndarray | None:
        """Frame k as RGB, or None past the end."""
        k = min(k, self.n - 1)
        if k == self.k:
            return self.rgb
        if k < self.k or k > self.k + 60:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, k)
            self.k = k - 1
        while self.k < k:
            ok, bgr = self.cap.read()
            if not ok:
                return self.rgb
            self.k += 1
            self.rgb = bgr[:, :, ::-1]
        return self.rgb

    def index_at(self, t: float) -> int:
        """The newest frame at or before t (-1: none yet)."""
        return int(np.searchsorted(self.times, t, side="right")) - 1

    def at(self, t: float) -> np.ndarray | None:
        k = self.index_at(t)
        return None if k < 0 else self.frame(k)
