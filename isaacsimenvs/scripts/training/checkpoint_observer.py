"""What training does at its checkpoints, beyond rl_games' own saving.

* Every ``rollout_every`` epochs (default 500, a multiple of the recipes' ``snapshot_every`` of 100, so
  each rollout sits on a ``snap_*`` checkpoint) it asks the wandb viewer for one rollout: an HTML
  episode on wandb.
* At the end of training it uploads the best and the last checkpoint to wandb as versions of one
  artifact, ``ckpt-<run name>``, aliased ``best`` and ``last``.

Main rank only, like the viewer and the wandb run.
"""

from __future__ import annotations

from pathlib import Path

from rl_games.common.algo_observer import AlgoObserver

from isaacsimenvs.scripts.evaluation.checkpoints import find_checkpoint

DEFAULT_ROLLOUT_EVERY = 500


class CheckpointObserver(AlgoObserver):
    def __init__(self, *, viewer=None, rollout_every: int = DEFAULT_ROLLOUT_EVERY) -> None:
        super().__init__()
        self.viewer = viewer
        if int(rollout_every) <= 0:
            raise ValueError(f"rollout_every must be > 0, got {rollout_every}")
        self.rollout_every = int(rollout_every)
        self.algo = None

    def after_init(self, algo) -> None:
        self.algo = algo
        if self.viewer is not None:
            print(f"[checkpoints] a viewer rollout every {self.rollout_every} epochs", flush=True)

    def after_print_stats(self, frame, epoch_num, total_time) -> None:
        if self.viewer is not None and epoch_num % self.rollout_every == 0:
            self.viewer.request_capture(epoch=epoch_num, global_step=frame)

    def upload_checkpoints(self) -> None:
        """Log the best and the last checkpoint as wandb artifact versions (no-op without a run)."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None or self.algo is None:
            return
        name = self.algo.config["name"]
        train_dir = Path(self.algo.train_dir)
        for alias in ("best", "last"):
            path = find_checkpoint(name, train_dir, alias)
            if path is None:
                print(f"[checkpoints] no {alias} checkpoint of {name} to upload", flush=True)
                continue
            artifact = wandb.Artifact(f"ckpt-{name}", type="model", metadata={"select": alias, "file": path.name})
            artifact.add_file(str(path.resolve()), name="model.pth")
            wandb.run.log_artifact(artifact, aliases=[alias])
            print(f"[checkpoints] uploaded {alias}: {path}", flush=True)
