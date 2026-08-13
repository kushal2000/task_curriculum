"""Interactive HTML viewer for the multi-link cartpole, logged to wandb.

Same idea and same machinery as play2perfect's `PlayPoseViewerWrapper` — capture a
window of poses, render them into the three.js page in
`isaacsimenvs/utils/interactive_viewer/index.template.html`, and log it with
`wandb.Html` — but much simpler, because this articulation is boxes and cylinders.

Play's viewer has to point `URDFLoader` at GitHub raw URLs so the browser can fetch the
Kuka and SHARPA STL meshes, which is why it needs `--capture_viewer_github_raw_base` and
a URL reachability check. The cartpole URDF references no mesh files at all, so it is
embedded verbatim and the page is genuinely self-contained: no network, no repo, no
branch to keep alive.

Cheap enough to leave on during training: it reads one env's joint vector per step and
touches neither the renderer nor a camera, so it costs nothing like `--capture_video`
(which needs `--enable_cameras` and the RTX pipeline).
"""

from __future__ import annotations

import re
from pathlib import Path

import gymnasium as gym
import numpy as np

from isaacsimenvs.utils.interactive_viewer.viewer_api import create_html, make_embedded_robot

__all__ = ["CartpolePoseViewerWrapper"]

# One colour per rigid segment, base -> tip. Links fused by a locked joint share a
# colour, so a colour boundary in the rendered pole is exactly a FREE joint — the
# morphology is readable straight off the geometry without a legend.
_SEGMENT_PALETTE = [
    (0.95, 0.35, 0.25),   # red
    (0.30, 0.70, 0.95),   # blue
    (0.55, 0.85, 0.35),   # green
    (0.98, 0.75, 0.20),   # amber
    (0.75, 0.45, 0.90),   # violet
    (0.20, 0.85, 0.80),   # teal
    (0.95, 0.55, 0.75),   # pink
    (0.65, 0.65, 0.70),   # grey
]


def _recolour_pole_links(urdf_text: str, free_mask: np.ndarray) -> str:
    """Tint each pole link by the rigid segment it belongs to.

    `free_mask[j]` is whether pole joint j is free. Links accumulate into a segment
    until the next free joint opens a new one, which is the same `cumsum(free) - 1`
    grouping `difficulty_math.segment_geometry` uses — so the picture and the
    observation always agree.
    """
    seg_id = np.cumsum(free_mask.astype(int)) - 1
    out = urdf_text
    for link_idx, seg in enumerate(seg_id):
        r, g, b = _SEGMENT_PALETTE[int(seg) % len(_SEGMENT_PALETTE)]
        out = re.sub(
            rf'(<material name="pole_{link_idx}_color">\s*<color rgba=")[^"]*(")',
            rf'\g<1>{r} {g} {b} 1.0\g<2>',
            out,
        )
    return out


def _caption_html(free_mask: np.ndarray, seg_lengths: list[float]) -> str:
    """A small overlay naming the active joints and the resulting segment lengths."""
    n_max = len(free_mask)
    free_idx = [j for j in range(n_max) if free_mask[j]]
    chips = "".join(
        f'<span style="display:inline-block;min-width:1.6em;text-align:center;'
        f'margin:0 2px;padding:2px 4px;border-radius:3px;'
        f'background:{"#2e7d32" if free_mask[j] else "#37474f"};'
        f'color:#fff;font-weight:{"700" if free_mask[j] else "400"};">{j}</span>'
        for j in range(n_max)
    )
    seg_txt = ", ".join(f"{v:.3f}" for v in seg_lengths)
    swatches = "".join(
        f'<span style="display:inline-block;width:14px;height:14px;margin-right:4px;'
        f'vertical-align:middle;background:rgb({int(255*r)},{int(255*g)},{int(255*b)});'
        f'border-radius:2px;"></span>'
        for r, g, b in (_SEGMENT_PALETTE[i % len(_SEGMENT_PALETTE)] for i in range(len(seg_lengths)))
    )
    return f"""
<div style="position:fixed;left:12px;bottom:12px;z-index:9999;
            font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
            background:rgba(18,20,24,0.86);color:#e8eaed;padding:10px 12px;
            border-radius:6px;border:1px solid rgba(255,255,255,0.12);max-width:60ch;">
  <div style="font-weight:700;margin-bottom:4px;">
    {len(free_idx)} free joint{"" if len(free_idx) == 1 else "s"} of {n_max}
  </div>
  <div style="margin-bottom:4px;">joints (green = free): {chips}</div>
  <div>segments: {swatches}<span style="vertical-align:middle;">{seg_txt} m</span></div>
  <div style="opacity:0.65;margin-top:4px;">
    a colour change along the pole is a free joint; one colour = one rigid segment
  </div>
</div>
"""


class CartpolePoseViewerWrapper(gym.Wrapper):
    """Periodically capture one env's joint trajectory and log it as an interactive page."""

    def __init__(
        self,
        env: gym.Env,
        output_dir: str | Path,
        *,
        capture_len: int = 300,
        capture_interval: int = 3000,
        env_id: int = 0,
        wandb_key: str = "interactive_viewer",
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_len = capture_len
        self.capture_interval = capture_interval
        self.env_id = env_id
        self.wandb_key = wandb_key

        self._step = 0
        self._frames: list[np.ndarray] | None = None
        self._mask: np.ndarray | None = None
        self._seg_lengths: list[float] = []
        self._captures = 0

        base = env.unwrapped
        if env_id < 0 or env_id >= base.num_envs:
            raise ValueError(f"env_id={env_id} out of range for num_envs={base.num_envs}")
        self._joint_names = list(base.cartpole.data.joint_names)
        self._dt = float(base.step_dt)

        urdf_path = getattr(base, "_cartpole_urdf_path", None)
        if urdf_path is None:
            raise RuntimeError(
                "env has no _cartpole_urdf_path; scene_utils.setup_scene must record it "
                "before the viewer can embed the articulation."
            )
        self._urdf_text = Path(urdf_path).read_text()

    def step(self, action):
        result = self.env.step(action)
        self._step += 1

        if self._frames is None and self.capture_interval > 0 and self._step % self.capture_interval == 0:
            self._frames = []

        if self._frames is not None:
            base = self.env.unwrapped
            mask = base._free_mask[self.env_id].detach().cpu().numpy().astype(bool)
            if self._mask is None:
                self._mask = mask
                self._seg_lengths = [
                    v for v in base._seg_lengths[self.env_id].detach().cpu().tolist() if v > 0.0
                ]
            elif not np.array_equal(mask, self._mask):
                # The env reset mid-capture and drew a new morphology. Cut the clip here
                # rather than render frames from two different pole structures under one
                # colouring, which would make the highlight a lie.
                self._finalize_capture()
                return result

            self._frames.append(base.cartpole.data.joint_pos[self.env_id].detach().cpu().numpy())
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()

        return result

    def _finalize_capture(self) -> None:
        frames_list, mask, seg_lengths = self._frames, self._mask, self._seg_lengths
        self._frames = None
        self._mask = None
        self._seg_lengths = []
        if not frames_list or mask is None:
            return
        frames = np.asarray(frames_list, dtype=float)  # (T, J)
        self._captures += 1

        try:
            html = create_html(
                joint_names=self._joint_names,
                robot_joint_positions=frames,
                robots=[
                    make_embedded_robot(
                        name="cartpole",
                        # Tint per rigid segment so the free joints are visible in the
                        # geometry itself, not just the caption.
                        urdf_text=_recolour_pole_links(self._urdf_text, mask),
                        animated=True,
                    )
                ],
                dt=self._dt,
                robot_name="cartpole",
            )
            caption = _caption_html(mask, seg_lengths)
            html = (
                html.replace("</body>", caption + "</body>", 1)
                if "</body>" in html
                else html + caption
            )
        except Exception as exc:  # noqa: BLE001 - a viewer failure must not kill training
            print(f"[cartpole/pose_viewer] failed to build HTML: {exc}", flush=True)
            return

        path = self.output_dir / f"viewer_step{self._step:09d}.html"
        path.write_text(html)
        self._log_wandb(html)

    def _log_wandb(self, html_text: str) -> None:
        try:
            import wandb
        except ImportError:  # pragma: no cover - wandb is optional
            return
        if wandb.run is None:
            return
        try:
            wandb.log({self.wandb_key: wandb.Html(html_text)})
            print(
                f"[cartpole/pose_viewer] logged WandB Html key={self.wandb_key} step={self._step}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cartpole/pose_viewer] wandb log failed: {exc}", flush=True)

    def close(self) -> None:
        # Flush a partial window so a short run still produces something to look at.
        if self._frames:
            self._finalize_capture()
        self.env.close()
