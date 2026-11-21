"""One-rollout acquisition and release scoring for VBD cloth."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

import torch

TOLERANCES = (0.015, 0.04)
FAILURES = ("nonfinite", "fall", "hand_far")


class FoldScore:
    def __init__(self, n, device, step_dt):
        self.required = math.ceil(3.0 / step_dt - 1e-6)
        self.dt = step_dt
        self.step = 0
        self.live = torch.ones(n, dtype=torch.bool, device=device)
        self.end = torch.zeros(n, dtype=torch.long, device=device)
        self.streak = torch.zeros((n, 2), dtype=torch.long, device=device)
        self.release_streak = torch.zeros_like(self.streak)
        self.acquired = torch.zeros_like(self.streak)
        self.released = torch.zeros_like(self.streak)
        self.persisted = torch.zeros_like(self.streak)
        self.lost = torch.zeros_like(self.streak, dtype=torch.bool)
        self.tolerances = torch.tensor(TOLERANCES, device=device)

    def update(self, error, footprint, touching, reasons, bad_reward):
        self.step += 1
        code = torch.where(reasons["timeout"], 1, 0)
        for i, reason in reversed(list(enumerate(FAILURES, 2))):
            code = torch.where(reasons[reason], i, code)
        invalid = bad_reward | ~torch.isfinite(error) | ~torch.isfinite(footprint)
        code = torch.where(invalid, 2, code)
        valid = self.live[:, None] & (code[:, None] <= 1)
        folded = valid & (error[:, None] < self.tolerances) & (footprint[:, None] < 0.65)
        self.streak = torch.where(folded, self.streak + 1, 0)
        previously_acquired = self.acquired > 0
        self.acquired = torch.where(~previously_acquired & (self.streak >= 10), self.step, self.acquired)
        clear_fold = folded & previously_acquired & ~touching[:, None]
        self.released = torch.where(clear_fold & (self.released == 0), self.step, self.released)
        self.lost |= valid & ~folded & (self.released > 0)
        self.release_streak = torch.where(clear_fold, self.release_streak + 1, 0)
        self.persisted = torch.where(
            (self.persisted == 0) & (self.release_streak >= self.required), self.step, self.persisted
        )
        self.end = torch.where(self.live & (code > 0), code, self.end)
        self.live &= code == 0

    def results(self, ids):
        names = ("censored", "timeout", *FAILURES)
        rows = []
        for start, end, acq, rel, per, lost in zip(
            ids, self.end.tolist(), self.acquired.tolist(), self.released.tolist(),
            self.persisted.tolist(), self.lost.tolist(),
        ):
            scores = {}
            for j, tol in enumerate(TOLERANCES):
                def time(step):
                    return step * self.dt if step else None

                scores[str(tol)] = dict(
                    acquisition_s=time(acq[j]), release_s=time(rel[j]), persistence_s=time(per[j]),
                    lost_after_release=lost[j],
                )
            rows.append(dict(start_id=start, outcome=names[end], scores=scores))
        return rows


def summarize(rows):
    summary = {}
    for tol in map(str, TOLERANCES):
        scores = [row["scores"][tol] for row in rows]
        acquired = [s for s in scores if s["acquisition_s"] is not None]
        released = [s for s in acquired if s["release_s"] is not None]
        persisted = sum(s["persistence_s"] is not None for s in released)
        summary[tol] = dict(
            acquisition_rate=len(acquired) / len(scores),
            persistence_rate=persisted / len(scores),
            persistence_given_acquisition=persisted / len(acquired) if acquired else None,
            median_acquisition_s=median(s["acquisition_s"] for s in acquired) if acquired else None,
            median_release_s=median(s["release_s"] for s in released) if released else None,
            acquired_without_release=len(acquired) - len(released),
            lost_after_release=sum(s["lost_after_release"] and s["persistence_s"] is None for s in released),
            timed_out_after_acquisition=sum(
                row["outcome"] == "timeout" and s["acquisition_s"] is not None and s["persistence_s"] is None
                for row, s in zip(rows, scores)
            ),
        )
    return summary


def run_standard(env, player, args):
    from isaacsimenvs.scripts.evaluation.contact_guard import capture_fd_output, raise_if_overflowed
    from isaacsimenvs.tasks.cloth.backends.vbd.release_contact import ReleaseContact

    inner = env.unwrapped
    n = inner.num_envs
    path = Path(args.cloth_standard)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest["version"] != 1:
        raise ValueError("Unsupported cloth start manifest version")
    offset = args.cloth_start_offset
    starts = manifest["starts"][offset:offset + n]
    if offset < 0 or len(starts) != n:
        raise ValueError("Requested starts fall outside the cloth manifest")
    if args.cloth_pinned:
        starts = [[0., 0., 0.]] * n
    poses = []
    for x, y, yaw in starts:
        angle = math.radians(yaw) / 2
        z = inner.cfg.reset.table_reset_z + inner.cfg.reset.table_object_z_offset
        poses.append([x, y, z, math.cos(angle), 0., 0., math.sin(angle)])
    inner.task.evaluation_start_poses = torch.tensor(poses, device=inner.device)
    score = FoldScore(n, inner.device, inner.step_dt)
    contact = ReleaseContact(n, inner.device)

    def observe(view, reasons, bad_reward):
        score.update(view.fold_error, view.footprint, contact.hand_cloth_touching(), reasons, bad_reward)

    obs, _ = env.reset()
    inner.evaluation_step_observer = observe
    try:
        with torch.inference_mode(), capture_fd_output() as read_so_far:
            for step in range(inner.max_episode_length):
                action = player.get_action(obs["policy"], deterministic=True)
                obs, *_ = env.step(action.to(inner.device))
                if (step + 1) % 60 == 0:
                    print(f"[cloth-standard] step {step + 1}/{inner.max_episode_length}", flush=True)
                    raise_if_overflowed(read_so_far(), n, f"standardized cloth rollout step {step + 1}")
                if not bool(score.live.any()):
                    break
            captured = read_so_far()
        raise_if_overflowed(captured, n, "standardized cloth rollout")
    finally:
        del inner.evaluation_step_observer
    rows = score.results(range(offset, offset + n))
    if any(row["outcome"] == "censored" for row in rows):
        raise RuntimeError("VBD standard evaluation ended with unfinished starts")
    result = dict(
        protocol="vbd-fold-v1", checkpoint=args.checkpoint, policy_config=args.policy_config,
        fabric=args.cloth_fabric, manifest_sha256=hashlib.sha256(raw).hexdigest(),
        pinned=args.cloth_pinned, seed=args.seed, sapg_expl_coef=args.sapg_expl_coef,
        config=inner.cfg.to_dict(), starts=starts, episodes=rows, summary=summarize(rows),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
