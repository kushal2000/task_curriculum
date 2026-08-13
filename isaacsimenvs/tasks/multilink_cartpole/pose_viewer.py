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

from pathlib import Path

import gymnasium as gym
import numpy as np

from isaacsimenvs.utils.interactive_viewer.viewer_api import create_html, make_embedded_robot

__all__ = ["CartpolePoseViewerWrapper"]


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
            self._frames.append(base.cartpole.data.joint_pos[self.env_id].detach().cpu().numpy())
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()

        return result

    def _finalize_capture(self) -> None:
        frames = np.asarray(self._frames, dtype=float)  # (T, J)
        self._frames = None
        self._captures += 1

        try:
            html = create_html(
                joint_names=self._joint_names,
                robot_joint_positions=frames,
                robots=[
                    make_embedded_robot(
                        name="cartpole",
                        urdf_text=self._urdf_text,
                        animated=True,
                    )
                ],
                dt=self._dt,
                robot_name="cartpole",
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
