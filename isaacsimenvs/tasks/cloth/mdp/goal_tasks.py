"""The pool's one-fold tasks as tensors, and the fold error against a goal state
(``fold_pool.goal_task``).

``geometry/fold_states.py`` enumerates, per shape, the reachable folded states, the single folds
between them and which states look alike from above. ``GoalTasks`` stacks every shape's tasks into
one table (a task = a global row), holds each env's current task (``task``, drawn by the backend
at every reset) and answers, for a batch of envs each on its own task:

* where the goal sheet is now (``goal_particles``): the goal state carried on the CURRENT pose of
  the particles the fold leaves in place -- the same construction as the stage-1 fold target, so
  the goal follows the sheet when it is dragged or turned;
* the designated fold error: the farthest any moved particle is from its place in that goal;
* the look-alike fold error (``accept_lookalike``): the minimum of that over the goal states that look the same from
  above. Only the designated goal and the ONE alternative whose hinge angles are nearest to the
  sheet's are fitted each step (two batched 3x3 SVDs, whatever the class size).

``FoldGeometry`` (``fold.py``) reads it to derive the fold. Torch only; lengths in metres; grids in the root plate's frame at the nominal sheet size, scaled
in-plane by each env's own factor at use.
"""

from __future__ import annotations

import math

import torch

from .geometry import fold_states as ft

__all__ = ["GoalTasks", "rigid_fit"]


def rigid_fit(src: torch.Tensor, cur: torch.Tensor, w: torch.Tensor):
    """Weighted least-squares rotation taking ``src`` ``(N, P, 3)`` onto ``cur`` ``(N, P, 3)``
    (weights ``w`` ``(N, P)``): ``(R (N, 3, 3), src centroid, cur centroid)``. A degenerate or
    non-finite fit falls back to the identity, never to NaN or a mirror."""
    w = w.to(src.dtype).unsqueeze(-1)
    total = w.sum(dim=1).clamp(min=1.0)
    cs, cc = (src * w).sum(dim=1) / total, (cur * w).sum(dim=1) / total
    h = torch.einsum("npi,npj->nij", (src - cs.unsqueeze(1)) * w, cur - cc.unsqueeze(1))
    h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    d = torch.ones((src.shape[0], 3), device=src.device, dtype=src.dtype)
    d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
    r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
    bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
    if bool(bad.any()):
        r = r.clone()
        r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
    return r, cs, cc


class GoalTasks:
    """See the module doc. ``mode``: ``half_fold`` or ``any_fold`` (``GoalTaskCfg.mode``).
    ``joint_names[s]``: shape ``s``'s hinge joints in the simulator's order, each
    ``joint_prefix + <hinge child>``. ``read_hinge_abs``: ``() -> (N, J)`` |hinge angle| of each
    env's active shape (zero-padded), needed with ``accept_lookalike``."""

    def __init__(
        self, shapes, resolution: int, mode: str, joint_names, joint_prefix: str, device, *,
        accept_lookalike: bool = False, read_hinge_abs=None,
    ) -> None:
        if mode not in ("half_fold", "any_fold"):
            raise ValueError(f"goal tasks: mode must be half_fold or any_fold, got {mode!r}")
        self.device, self.mode = device, mode
        #: ``any_fold``: the fold is scored against the goal state (keypoints, targets, frame and
        #: error all follow the task); ``half_fold`` keeps the stage-1 construction.
        self.any_fold = mode == "any_fold"
        self.accept_lookalike = bool(accept_lookalike)
        self.read_hinge_abs = read_hinge_abs
        #: ``(N,)`` each env's task (a row of the tables); ``None`` until the backend's first draw.
        self.task: torch.Tensor | None = None
        tasks, self.shape_rows, self.names = [], [], []
        self.sample_tbl, self.sample_cnt = [], []
        max_j = max(len(j) for j in joint_names)
        report = []
        for s, shape in enumerate(shapes):
            st = ft.enumerate_tasks(shape)
            rows = ft.task_arrays(st, resolution, only_half=(mode == "half_fold"))
            if not rows:
                raise ValueError(f"[goal_task] {shape.name}: no one-fold task on a {resolution}x{resolution} grid")
            order = [joint_names[s].index(f"{joint_prefix}{h.child}") for h in shape.hinges]
            first = len(tasks)
            for row in rows:
                q = torch.zeros(max_j)
                q[order] = torch.as_tensor(row["angles"], dtype=torch.float32)
                row = dict(row, shape=s, q=q)
                row["member_q"] = []
                for j, _grid, _frame in row["members"]:
                    g = torch.zeros(max_j)
                    g[order] = torch.as_tensor(st.states[j].angles, dtype=torch.float32)
                    row["member_q"].append(g)
                tasks.append(row)
            self.shape_rows.append((first, len(tasks)))
            starts = sorted({r["start"] for r in rows})
            per = [[first + k for k, r in enumerate(rows) if r["start"] == a] for a in starts]
            width = max(len(p) for p in per)
            tbl = torch.zeros((len(per), width), dtype=torch.long)
            for i, p in enumerate(per):
                tbl[i, : len(p)] = torch.tensor(p)
            self.sample_tbl.append(tbl.to(device))
            self.sample_cnt.append(torch.tensor([len(p) for p in per], device=device))
            sizes = [len(r["members"]) for r in rows]
            report.append(f"{shape.name}: {len(st.states)} states, {len(rows)} tasks from {len(starts)} starts, "
                          f"look-alike goals <= {max(sizes)}")
        self.num_tasks = t = len(tasks)
        p = resolution * resolution
        c = max(len(r["members"]) for r in tasks)
        f32 = dict(device=device, dtype=torch.float32)
        self.q_start = torch.stack([r["q"] for r in tasks]).to(device)                     # (T, J) signed
        self.p_start = torch.tensor([r["p_start"].tolist() for r in tasks], **f32)         # (T, P, 3)
        self.p_goal = torch.tensor([r["p_goal"].tolist() for r in tasks], **f32)
        self.moving = torch.tensor([r["moving"].tolist() for r in tasks], device=device, dtype=torch.bool)
        self.kp = torch.tensor([r["kp"].tolist() for r in tasks], device=device, dtype=torch.long)   # (T, 4)
        self.rot = torch.tensor([r["rot"].tolist() for r in tasks], **f32)                 # (T, 3, 3)
        self.direction = torch.tensor([r["direction"] for r in tasks], **f32)
        self.task_shape = torch.tensor([r["shape"] for r in tasks], device=device, dtype=torch.long)
        # Look-alike goals: member 0 is the task's own goal.
        self.m_goal = torch.zeros((t, c, p, 3), **f32)
        self.m_frame = torch.zeros((t, c, p), device=device, dtype=torch.bool)
        self.m_q = torch.zeros((t, c, max_j), **f32)
        self.m_q_signed = torch.zeros((t, c, max_j), **f32)   # the hinge angles that give each goal
        self.m_ok = torch.zeros((t, c), device=device, dtype=torch.bool)
        for i, r in enumerate(tasks):
            for k, ((_j, grid, frame), q) in enumerate(zip(r["members"], r["member_q"])):
                self.m_goal[i, k] = torch.as_tensor(grid, dtype=torch.float32)
                self.m_frame[i, k] = torch.as_tensor(frame)
                self.m_q[i, k] = q.abs()
                self.m_q_signed[i, k] = q
                self.m_ok[i, k] = True
        self.names = [f"{shapes[r['shape']].name}:{r['start']}->{r['goal']}({r['name']})" for r in tasks]
        print(f"[goal_task] mode {mode}: {t} tasks; " + "; ".join(report), flush=True)

    # ------------------------------------------------------------------ draws

    def draw(self, shape_ids: torch.Tensor) -> torch.Tensor:
        """A task per env: a start state uniformly among the shape's starts, then one of its folds."""
        out = torch.zeros_like(shape_ids)
        for s, (tbl, cnt) in enumerate(zip(self.sample_tbl, self.sample_cnt)):
            on = shape_ids == s
            n = int(on.sum())
            if n == 0:
                continue
            a = torch.randint(tbl.shape[0], (n,), device=self.device)
            k = (torch.rand(n, device=self.device) * cnt[a]).long().clamp(max=tbl.shape[1] - 1)
            k = torch.minimum(k, cnt[a] - 1)
            out[on] = tbl[a, k]
        return out

    # ------------------------------------------------------------------ each env's task

    def draw_envs(self, env_ids: torch.Tensor, shape_ids: torch.Tensor) -> None:
        """Draw the task of ``env_ids``, whose active shapes are ``shape_ids``. The first call
        (the backend's start) covers every env."""
        drawn = self.draw(shape_ids)
        if self.task is None:
            if drawn.shape[0] != env_ids.shape[0]:
                raise ValueError("goal tasks: the first draw must cover every env")
            self.task = drawn.clone()
        else:
            self.task[env_ids] = drawn

    def keypoint_idx(self) -> torch.Tensor:
        """``(N, 4)`` grid indices of each env's tracked points: on the piece its fold moves."""
        return self.kp[self.task]

    def keypoint_rest(self) -> torch.Tensor:
        """``(N, 4, 3)`` those points in the task's START state, nominal size (root plate frame)."""
        return torch.gather(self.p_start[self.task], 1, self.keypoint_idx().unsqueeze(-1).expand(-1, -1, 3))

    def lookalike_error(self, s3: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """``(N,)`` fold error against the look-alike goal nearest the sheet's hinge state; +inf
        where the task's goal has no look-alike."""
        task = self.task
        member, has = self.nearest_alternative(task, self.read_hinge_abs())
        err = self.error(task, s3, parts, member)
        return torch.where(has, err, torch.full_like(err, float("inf")))

    # ------------------------------------------------------------------ geometry

    def goal_particles(self, task: torch.Tensor, s3: torch.Tensor, parts: torch.Tensor, member=None):
        """``(N, P, 3)`` the goal sheet now, and the mask of particles that goal moves. ``member``
        ``(N,)``: which look-alike goal (default the task's own)."""
        if member is None:
            goal, frame = self.p_goal[task], ~self.moving[task]
        else:
            goal, frame = self.m_goal[task, member], self.m_frame[task, member]
        goal = goal * s3
        r, cs, cc = rigid_fit(goal, parts, frame)
        return torch.einsum("nij,npj->npi", r, goal - cs.unsqueeze(1)) + cc.unsqueeze(1), ~frame

    def frame(self, task: torch.Tensor, s3: torch.Tensor, parts: torch.Tensor):
        """``(R, centroid now)`` of the particles the designated fold leaves in place, measured from
        the task's START state."""
        r, _cs, cc = rigid_fit(self.p_start[task] * s3, parts, ~self.moving[task])
        return r, cc

    def error(self, task, s3, parts, member=None) -> torch.Tensor:
        goal, moved = self.goal_particles(task, s3, parts, member)
        d = (parts - goal).norm(dim=-1)
        return torch.where(moved, d, torch.zeros_like(d)).amax(dim=1)

    def nearest_alternative(self, task: torch.Tensor, q_abs: torch.Tensor):
        """``(member (N,), has (N,))``: the look-alike goal other than the task's own whose hinge
        state is nearest to the sheet's ``q_abs`` ``(N, J)`` (|angle|)."""
        d = (self.m_q[task] - q_abs.unsqueeze(1)).abs().sum(dim=-1)                 # (N, C)
        ok = self.m_ok[task].clone()
        ok[:, 0] = False
        d = torch.where(ok, d, torch.full_like(d, float("inf")))
        return d.argmin(dim=1), ok.any(dim=1)

    def sector_yaw(self, task: torch.Tensor, width_deg: float, centre_deg: float, margin_deg: float) -> torch.Tensor:
        """A yaw that puts the designated fold's travel direction in the sector."""
        half = max(0.5 * float(width_deg) - float(margin_deg), 0.0)
        u = (2.0 * torch.rand(task.shape[0], device=self.device) - 1.0) * math.radians(half)
        yaw = math.radians(float(centre_deg)) + u - self.direction[task]
        return torch.atan2(torch.sin(yaw), torch.cos(yaw))
