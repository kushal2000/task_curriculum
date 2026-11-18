"""Rigid-cloth training throughput under one config + overrides, driven by a real policy.

    .venv/isaacsim/bin/python experiments/nov18/throughput/bench.py --headless \
        --config <run>/.hydra/config.yaml --checkpoint <ckpt> --num_envs 12288 --out x.json [key=value ...]

Builds the env and policy exactly like eval_cloth.py (same loader, clipping, autoresets), warms up,
then times --steps policy steps. Reports env-steps/s for the full step (policy inference + env.step,
what training logs as "fps step + policy inference") and for env.step alone, with CUDA synchronised
around every timed call.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from isaacsimenvs.scripts.evaluation.eval_cloth import add_common_args, build, launch  # noqa: E402


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    add_common_args(p, num_envs=12288)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--torch_profile", type=int, default=0,
                   help="profile this many steps after warm-up with torch.profiler; print the top ops")
    p.add_argument("--check_com", action="store_true",
                   help="after warm-up, compare the guard's cached-offset COMs with Isaac Lab's body_com_pos_w")
    p.add_argument("--profile_observe", action="store_true",
                   help="time each section of RigidClothEpisodeGuard.observe (instrumented copy)")
    p.add_argument("--profile_guard", action="store_true",
                   help="time the episode guard's parts (CUDA-synchronised; adds sync overhead itself)")
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = launch(args)
    import torch

    b = build(args)
    env, n = b.env, b.inner.num_envs
    timers = {}
    if args.profile_observe:
        import types
        from isaacsimenvs.tasks.cloth.backends.rigid import episode_guard as eg
        guard = b.inner._episode_guard

        def lap(state, label):
            torch.cuda.synchronize(); now = time.perf_counter()
            calls, total = timers.get(label, (0, 0.0))
            timers[label] = (calls + 1, total + now - state[0]); state[0] = now

        def observe(self, completed_counter):
            if completed_counter <= self.last_counter:
                return
            self.last_counter = completed_counter
            torch.cuda.synchronize(); st = [time.perf_counter()]
            b_ = self.backend
            pos, quat, com, velocity = self._plate_state(); lap(st, "a plate_state")
            vertices = pos[:, :, None]+eg.rotations.apply(quat[:, :, None], self.rest_vertices)
            vertices = torch.where((self.half >= 0)[:, :, None, None], vertices, vertices[:, :1, :1])
            table = b_.table.data
            self.excursion = eg.table_excursion(vertices, table.root_pos_w, table.root_quat_w,
                                                pos.new_tensor(b_._table_size[:2])/2); lap(st, "b vertices+table_excursion")
            finite = torch.isfinite(pos).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=(1, 2))
            self.nonfinite |= ~finite
            inside = (self.excursion <= self.cfg.table_margin) & finite
            if self.cfg.outside_table:
                self.outside |= ~inside & finite
            lap(st, "c finite+outside latch")
            if self.contacts is None:
                return
            limit = getattr(self, 'penetration_limit', None)
            if limit is not None:
                self.penetrating |= self.contacts.robot_penetration(self.n) > limit
            lap(st, "d penetration query")
            if not self.cfg.press_driven_fold:
                return
            robot = b_.robot.data
            current = (robot.body_link_pos_w, robot.body_link_quat_w, robot.body_com_pos_w, robot.body_com_vel_w)
            lap(st, "e robot state read")
            if self.previous_robot is None:
                self.previous_robot = tuple(value.clone() for value in current)
            for previous, value in zip(self.previous_robot, current, strict=True):
                previous[self.fresh] = value[self.fresh]
            lap(st, "f previous[fresh] masked update")
            self.nonfinite |= ~torch.isfinite(velocity).all(dim=(1, 2))
            self.nonfinite |= ~torch.isfinite(current[3]).all(dim=(1, 2))
            self.nonfinite |= ~torch.isfinite(current[0]).all(dim=(1, 2))
            lap(st, "g robot finite checks")
            e, p_, partner, point, force = self.contacts.raw(); lap(st, "h contacts.raw")
            h = self.half[e, p_]
            valid = torch.isfinite(force).all(-1) & torch.isfinite(point).all(-1)
            bad = torch.zeros(self.n, dtype=torch.long, device=self.device).index_add_(0, e, (~valid).long())
            self.nonfinite |= bad > 0
            force, point = force.nan_to_num(), point.nan_to_num()
            robot_contact = partner < self.contacts.num_robot_links
            lap(st, "i contact post-processing")
            source = self._robot_work(e, h, partner, point, force, robot_contact, current, pos, quat); lap(st, "j robot_work")
            robot_z = torch.zeros(self.n*2, device=self.device)
            table_up = torch.zeros_like(robot_z)
            robot_z.index_add_(0, e[robot_contact]*2+h[robot_contact], force[robot_contact, 2])
            table_up.index_add_(0, e[~robot_contact]*2+h[~robot_contact], force[~robot_contact, 2].clamp_min(0))
            table_up = table_up.view(self.n, 2)
            opposed = torch.minimum((-robot_z.view(self.n, 2)).clamp_min(0), table_up)
            lap(st, "k force sums")
            self._update_history(com, quat, velocity, table_up, opposed, source, inside & ~self.nonfinite); lap(st, "l update_history")
            for previous, value in zip(self.previous_robot, current, strict=True):
                previous.copy_(value)
            self.fresh.zero_(); lap(st, "m previous copy")

        guard.observe = types.MethodType(observe, guard)
    if args.profile_guard:
        guard = b.inner._episode_guard

        def timed(obj, name, label):
            fn = getattr(obj, name)

            def wrapper(*a, **k):
                torch.cuda.synchronize(); t = time.perf_counter()
                out = fn(*a, **k)
                torch.cuda.synchronize()
                calls, total = timers.get(label, (0, 0.0))
                timers[label] = (calls + 1, total + time.perf_counter() - t)
                return out
            setattr(obj, name, wrapper)

        timed(guard, "observe", "guard.observe (total)")
        if guard.contacts is not None:
            timed(guard.contacts, "raw", "contacts.raw (readout)")
            timed(guard.contacts, "robot_penetration", "contacts.robot_penetration")
        timed(guard, "_robot_work", "guard._robot_work")
        timed(guard, "_update_history", "guard._update_history")
        timed(guard, "_plate_state", "guard._plate_state")
    obs, _ = env.reset()
    b.player.reset_rnn()
    t_full = t_env = 0.0
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile
        with torch.inference_mode():
            for _ in range(args.warmup):
                obs, _, te, tr, _ = env.step(b.act(obs))
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
                for _ in range(args.torch_profile):
                    obs, _, te, tr, _ = env.step(b.act(obs))
                torch.cuda.synchronize()
        table = prof.key_averages(group_by_stack_n=4).table(sort_by="cpu_time_total", row_limit=45, max_name_column_width=70)
        print(table, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".profile.txt").write_text(table)
        prof.export_stacks(str(args.out.with_suffix(".stacks.txt")), "self_cpu_time_total")
        env.close()
        if app is not None:
            app.close()
        return
    with torch.inference_mode():
        for i in range(args.warmup + args.steps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            action = b.act(obs)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(action)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            ended = terminated | truncated
            if bool(ended.any()):
                b.player.reset_rnn(ended.nonzero(as_tuple=True)[0])
            if i >= args.warmup:
                t_full += t2 - t0
                t_env += t2 - t1
            if i == args.warmup - 1:
                timers.clear()
                if args.check_com:
                    g = b.inner._episode_guard
                    r = b.inner.robot.data
                    diff = {"robot": float((g._com_w("robot", r.body_link_pos_w, r.body_link_quat_w, r) - r.body_com_pos_w).abs().max())}
                    for grp in b.inner.backend.groups:
                        d = grp.articulation.data
                        mine = g._com_w(grp.index, d.body_link_pos_w[:, grp.body_ids], d.body_link_quat_w[:, grp.body_ids], d, grp.body_ids)
                        diff[f"cloth_{grp.index}"] = float((mine - d.body_com_pos_w[:, grp.body_ids]).abs().max())
                    print("[bench] com max abs diff (m):", diff, flush=True)
                    timers["com_check"] = (0, 0.0)
                    com_diff = diff
    result = {"overrides": list(args.overrides), "num_envs": n, "steps": args.steps,
              "com_max_abs_diff_m": locals().get("com_diff"),
              "fps_full": round(n * args.steps / t_full), "fps_env_step": round(n * args.steps / t_env),
              "ms_per_step_full": round(1000 * t_full / args.steps, 2),
              "ms_per_step_env": round(1000 * t_env / args.steps, 2),
              "guard_ms_per_policy_step": {k: {"calls_per_step": round(c / args.steps, 2),
                                               "ms": round(1000 * t / args.steps, 2)} for k, (c, t) in sorted(timers.items())}}
    print("[bench]", json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
