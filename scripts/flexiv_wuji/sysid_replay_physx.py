"""Replay the real Wuji Hand 2 sysid trials in PhysX on a Flexiv + right Wuji robot (RightReal).

    PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/sysid_replay_physx.py \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --headless \
        --data_dir <dir holding sysid-2026*/> --dt 0.00833333333 --out replay_dt120.npz

One env per (variant, 2 s window), windows cut exactly as aiowuji sysid_fit.py does (each starts from
the recorded q, qd). The SIM is stepped directly (no env actions). Per env:
  * hand drive = the firmware PD: stiffness kp_scale kp Kt, damping kd_scale kd Kt (kp, kd as recorded,
    the excited finger's and the held fingers' kp 8 / kd 0.15), max force effort_limit (0.5 A) x Kt;
    Kt = the fit's gear. kp_scale / kd_scale = the fit's firmware_law (1.966 / 1.944).
  * targets: the frame-wise target with the 2 ms lag, zero-order held per sim step, sampled at the step
    midpoint (hold=mid) or start (hold=start). At dt 1 ms both are exactly sysid_fit's.
  * friction / armature: PhysX's RightReal values scaled per variant (RightReal already carries the
    per-type friction scale, so 1 = the spec). spring=1 adds the fit's passive spring by folding it
    into the drive (stiffness kp+ks, target (kp q_cmd + ks springref)/(kp+ks); the spring is then
    clipped with the PD, which the real one is not); spring=2 applies it as the env does, a joint
    effort ks (ref - q) from action_utils.hand_spring_effort every step, outside the PD's clip.
  * gravity: on for the hand links only (the robot USD disables it), the scene gravity vector set so
    that it is the recorded mount's in the vendor MJCF wrist frame (fingers-up +z, palm-down +y). The
    two mounts run as two phases. The table and the cloth chain are moved 100 m below.
Variants: --variants "name:static_scale:dynamic_scale:armature_scale:spring:hold,..."
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0")
parser.add_argument("--data_dir", required=True)
parser.add_argument("--fit_json", default="assets/flexiv_wuji/wuji_hand2_right_sysid_fit_20261009.json")
parser.add_argument("--dt", type=float, default=1.0 / 120.0)
parser.add_argument("--variants", default="V1:1:1:1:0:mid,V2:0:0:1:0:mid,V3:0.6:0.6:1:0:mid,V4:0.6:0.6:3:0:mid")
parser.add_argument("--vendor_limits", type=int, default=1, help="hand joint limits = vendor MJCF ranges")
parser.add_argument("--max_windows", type=int, default=0, help="per mount, for a smoke run (0 = all)")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.eval.protocol import disable_randomization  # noqa: E402
from isaacsimenvs.tasks.play.utils.action_utils import hand_spring_effort  # noqa: E402
from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sysid_replay_data import FRAME_DT, JOINT_NAMES, WINDOW_FRAMES, load_trials, make_windows  # noqa: E402

MOUNTS = ("fingers-up", "palm-down")
FINGER_PREFIXES = ("r_thumb", "r_index", "r_middle", "r_ring", "r_pinky")
ARM_GAIN = 10.0


def parse_variants(text):
    out = []
    for item in text.split(","):
        name, fs, fd, arm, spring, hold = item.split(":")
        assert hold in ("mid", "start"), hold
        out.append({"name": name, "fs": per_joint(fs), "fd": per_joint(fd), "arm": float(arm),
                    "spring": int(spring), "hold": hold})
    return out


def per_joint(text):
    """'0.8' (every hand joint), 'a|b|c|d' (S1..S4, fingers and thumb) or 'a|b|c|d|e|f|g|h' (finger S1..S4,
    thumb S1..S4) -> 20 scales in JOINT_NAMES order."""
    v = [float(x) for x in text.split("|")]
    if len(v) == 1:
        v = v * 4
    finger, thumb = (v[:4], v[4:]) if len(v) == 8 else (v, v)
    return thumb + finger * 4


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@hydra_task_config_with_yaml(args_cli.task, "rl_games_sapg_cfg_entry_point")
def run(env_cfg, agent_cfg) -> None:
    t_start = time.time()
    variants = parse_variants(args_cli.variants)
    trials = load_trials(args_cli.data_dir)
    windows = make_windows(trials)
    by_mount = {m: [i for i, w in enumerate(windows) if trials[w.trial].mount == m] for m in MOUNTS}
    if args_cli.max_windows:
        by_mount = {m: v[: args_cli.max_windows] for m, v in by_mount.items()}
    n_slot = max(len(v) for v in by_mount.values())
    V = len(variants)
    N = V * n_slot
    dt = float(args_cli.dt)
    n_steps = int(round(WINDOW_FRAMES * FRAME_DT / dt))
    print(f"[replay] {len(trials)} trials, windows {[len(v) for v in by_mount.values()]}, {V} variants, "
          f"{N} envs, dt {dt:.6g}, {n_steps} steps/window", flush=True)

    fit = json.loads(Path(args_cli.fit_json).read_text())
    law = fit["firmware_law"]
    kp_scale, kd_scale = float(law["kp_scale"]), float(law["kd_scale"])
    fj = [fit["joints"][n] for n in JOINT_NAMES]
    gear = np.array([j["gear"] for j in fj])
    ks = np.array([j["stiffness"] for j in fj])
    sref = np.array([j["springref"] for j in fj])

    env_cfg.scene.num_envs = N
    env_cfg.seed = 0
    env_cfg.sim.dt = dt
    disable_randomization(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    robot, device = inner.robot, inner.device
    view = robot.root_physx_view
    names = list(robot.data.joint_names)
    bodies = list(robot.body_names)
    env.reset()
    assert abs(float(inner.physics_dt) - dt) < 1e-9, (inner.physics_dt, dt)
    cols = [names.index(j["mjcf_joint"]) for j in fj]  # data column -> sim joint
    cols_t = torch.tensor(cols, device=device)
    report = {"task": args_cli.task, "robot": inner.robot_spec.name, "dt": dt, "n_steps": n_steps, "N": N,
              "variants": variants, "kp_scale": kp_scale, "kd_scale": kd_scale}

    all_idx_cpu = torch.arange(N, dtype=torch.int32)
    kp0 = robot.data.joint_stiffness.clone()
    kd0 = robot.data.joint_damping.clone()
    lim0 = robot.data.joint_effort_limits.clone()
    fr0 = view.get_dof_friction_properties().clone()  # (N, J, 3) cpu
    arm0 = view.get_dof_armatures().clone()
    home = robot.data.default_joint_pos.clone()
    # Hand joint limits -> the vendor MJCF ranges (RightReal's human-like URDF holds finger abduction to
    # +-2 deg, MCP back-bend to -20 deg and PIP to 100 deg; the recordings go beyond). Arm drive x ARM_GAIN.
    lim_rr = robot.data.joint_pos_limits[0, cols_t].cpu().numpy().copy()
    mjcf = ET.parse(Path(args_cli.data_dir) / "mjbatch_sysid/aiowuji/models/wuji_hand2/mjcf/right.xml").getroot()
    rng = {j.get("name"): [float(v) for v in j.get("range").split()] for j in mjcf.iter("joint") if j.get("range")}
    if args_cli.vendor_limits:
        pl = robot.data.joint_pos_limits.clone()
        for k, c in enumerate(cols):
            pl[:, c, 0], pl[:, c, 1] = rng[fj[k]["mjcf_joint"]]
        robot.write_joint_position_limit_to_sim(pl, warn_limit_violation=False)
    arm_cols = [i for i in range(len(names)) if i not in cols]
    kp0[:, arm_cols] *= ARM_GAIN
    kd0[:, arm_cols] *= ARM_GAIN
    report["hand_limits_rightreal"] = {JOINT_NAMES[k]: lim_rr[k].round(4).tolist() for k in range(20)}
    report["hand_limits_replay"] = {JOINT_NAMES[k]: robot.data.joint_pos_limits[0, c].cpu().numpy().round(4).tolist()
                                    for k, c in enumerate(cols)}
    report["arm_gain_scale"] = ARM_GAIN
    report["readback_vs_fit"] = {
        JOINT_NAMES[k]: {"static": round(float(fr0[0, c, 0]), 6), "dynamic": round(float(fr0[0, c, 1]), 6),
                         "viscous": float(fr0[0, c, 2]), "armature": float(arm0[0, c]),
                         "fit_frictionloss": fj[k]["frictionloss"], "fit_armature": fj[k]["armature"],
                         "fit_damping": fj[k]["damping"]}
        for k, c in enumerate(cols)}

    def step(n=1, rec=None, tgt_seq=None, spring_mask=None):
        for k in range(n):
            if tgt_seq is not None:
                tgt = home.clone()
                tgt[:, cols_t] = tgt_seq[:, k]
                robot.set_joint_position_target(tgt)
            if spring_mask is not None:
                robot.set_joint_effort_target(hand_spring_effort(inner) * spring_mask)
            robot.write_data_to_sim()
            inner.sim.step(render=False)
            robot.update(dt)
            if rec is not None:
                rec[:, k + 1] = robot.data.joint_pos[:, cols_t]

    def set_state(q, qd=None):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q) if qd is None else qd)
        robot.set_joint_position_target(q)
        robot.set_joint_velocity_target(torch.zeros_like(q))
        robot.set_joint_effort_target(torch.zeros_like(q))

    def set_params(stiff, damp, limit, fs, fd, arm_scale):
        """stiff/damp/limit (N, 20) for the hand; fs/fd (N, 20) or (N,), arm_scale (N,) scales."""
        fs, fd = np.asarray(fs, float), np.asarray(fd, float)
        fs = fs if fs.ndim == 2 else np.repeat(fs[:, None], 20, 1)
        fd = fd if fd.ndim == 2 else np.repeat(fd[:, None], 20, 1)
        K, D, L = kp0.clone(), kd0.clone(), lim0.clone()
        K[:, cols_t] = torch.as_tensor(stiff, dtype=K.dtype, device=device)
        D[:, cols_t] = torch.as_tensor(damp, dtype=D.dtype, device=device)
        L[:, cols_t] = torch.as_tensor(limit, dtype=L.dtype, device=device)
        robot.write_joint_stiffness_to_sim(K)
        robot.write_joint_damping_to_sim(D)
        robot.write_joint_effort_limit_to_sim(L)
        fr = fr0.clone()
        fr[:, cols, 0] *= torch.as_tensor(fs, dtype=fr.dtype, device=fr.device)
        fr[:, cols, 1] *= torch.as_tensor(fd, dtype=fr.dtype, device=fr.device)
        view.set_dof_friction_properties(fr, indices=all_idx_cpu.to(fr.device))
        a = arm0.clone()
        a[:, cols] *= torch.as_tensor(arm_scale, dtype=a.dtype, device=a.device)[:, None]
        view.set_dof_armatures(a, indices=all_idx_cpu.to(a.device))

    # ------------------------------------------------ scene: table / chain away, gravity on the hand
    tp = inner.table.data.root_state_w[:, :7].clone()
    tp[:, 2] -= 100.0
    inner.table.write_root_pose_to_sim(tp)
    chain = getattr(inner, "_chain", None)
    if chain is not None:
        cp = chain.data.root_state_w[:, :7].clone()
        cp[:, 2] -= 100.0
        chain.write_root_pose_to_sim(cp)
        chain.write_root_velocity_to_sim(torch.zeros(N, 6, device=device))
        try:
            cv = chain.root_physx_view
            dgc = cv.get_disable_gravities()
            dgc[:] = 1
            cv.set_disable_gravities(dgc, indices=torch.arange(N, dtype=torch.int32, device=dgc.device))
            report["chain_gravity_disabled"] = True
        except Exception as e:  # noqa: BLE001
            report["chain_gravity_disabled"] = f"failed: {e}"
    hand_body = [i for i, b in enumerate(bodies) if b.startswith(FINGER_PREFIXES)]
    dg = view.get_disable_gravities()
    report["robot_disable_gravity_before"] = int(dg[0].sum())
    dg[:, hand_body] = 0
    view.set_disable_gravities(dg, indices=torch.arange(N, dtype=torch.int32, device=dg.device))
    dg2 = view.get_disable_gravities()
    report["gravity_on_bodies"] = [bodies[i] for i in range(len(bodies)) if int(dg2[0, i]) == 0]

    # ------------------------------------------------ wrist frame (hand at zero, gravity still -z)
    q = home.clone()
    q[:, cols_t] = 0.0
    set_params(kp0[:, cols_t].cpu().numpy(), kd0[:, cols_t].cpu().numpy(), lim0[:, cols_t].cpu().numpy(),
               np.ones(N), np.ones(N), np.ones(N))
    sim_view(inner).set_gravity(_float3((0.0, 0.0, 0.0)))
    set_state(q)
    step(2)
    set_state(q)
    step(1)
    urdf = ET.parse(env_cfg.assets.robot_urdf).getroot()
    rots = {}
    for j in urdf.iter("joint"):
        if j.find("parent").get("link") != "r_wrist":
            continue
        child = j.find("child").get("link")
        o = j.find("origin")
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        r_world_child = quat_wxyz_to_mat(robot.data.body_quat_w[0, bodies.index(child)].cpu().numpy().astype(float))
        rots[child] = r_world_child @ rpy_to_mat(*rpy).T
    r_wrist = np.mean(list(rots.values()), axis=0)
    u, _, vt = np.linalg.svd(r_wrist)
    r_wrist = u @ vt
    spread = max(math.degrees(math.acos(np.clip((np.trace(r_wrist.T @ r) - 1) / 2, -1, 1))) for r in rots.values())
    # Fingers should extend along -z of the wrist frame: index tip direction from its MCP, at zero.
    p_mcp = robot.data.body_pos_w[0, bodies.index("r_index_finger_proximal")].cpu().numpy()
    p_tip = robot.data.body_pos_w[0, bodies.index("r_index_finger_distal")].cpu().numpy()
    tip_dir_wrist = r_wrist.T @ ((p_tip - p_mcp) / np.linalg.norm(p_tip - p_mcp))
    report["wrist_frame"] = {"R_world_wrist": r_wrist.round(5).tolist(), "finger_frames_spread_deg": round(spread, 4),
                             "from_bodies": sorted(rots), "index_dir_in_wrist": tip_dir_wrist.round(4).tolist()}
    print("[replay] wrist", json.dumps(report["wrist_frame"]), flush=True)

    rec = torch.zeros(N, n_steps + 1, 20, device=device)
    sim_q = np.full((V, len(windows), n_steps + 1, 4), np.nan, dtype=np.float32)
    held_dev = np.full((V, len(windows)), np.nan, dtype=np.float32)
    arm_dev = np.zeros(len(MOUNTS))
    checks = {}
    for mi, mount in enumerate(MOUNTS):
        slots = by_mount[mount]
        g_wrist = np.array(trials[windows[slots[0]].trial].gravity)
        g_world = r_wrist @ g_wrist
        sim_view(inner).set_gravity(_float3(g_world))
        g_read = sim_view(inner).get_gravity()
        # ---- sag check: index finger limp (no drive, no Coulomb friction), others held, 0.25 s
        jm, jp = JOINT_NAMES.index("index_S1"), JOINT_NAMES.index("index_S3")
        q = home.clone()
        q[:, cols_t] = 0.0
        if mount == "fingers-up":
            q[:, cols[jm]] = 0.4
        stiff = kp0[:, cols_t].cpu().numpy().copy()
        damp = kd0[:, cols_t].cpu().numpy().copy()
        for c in range(4, 8):
            stiff[:, c] = 0.0
            damp[:, c] = 0.0
        set_params(stiff, damp, lim0[:, cols_t].cpu().numpy(), np.zeros(N), np.zeros(N), np.ones(N))
        set_state(q)
        q_before = robot.data.joint_pos[0, cols_t].cpu().numpy().copy()
        step(int(round(0.25 / dt)))
        q_after = robot.data.joint_pos[0, cols_t].cpu().numpy()
        checks[mount] = {"g_wrist": g_wrist.tolist(), "g_world": g_world.round(4).tolist(),
                         "g_readback": [round(float(v), 4) for v in g_read],
                         "index_dir_dot_ghat": round(float(tip_dir_wrist @ g_wrist / 9.81), 4),
                         "limp_index_dq_rad_0.25s": {n: round(float(q_after[k] - q_before[k]), 4)
                                                     for n, k in (("S1", jm), ("S2", jm + 1), ("S3", jp), ("S4", jp + 1))},
                         "expect": "S1, S3 flex (dq > 0) under gravity toward the palm side"}
        print(f"[replay] {mount} check", json.dumps(checks[mount]), flush=True)

        # ---- the replay
        stiff = np.zeros((N, 20)); damp = np.zeros((N, 20)); limit = np.zeros((N, 20))
        fs = np.zeros((N, 20)); fd = np.zeros((N, 20)); arm_s = np.zeros(N)
        tgt = np.zeros((N, n_steps, 20), dtype=np.float32)
        q0 = home.clone(); qd0 = torch.zeros_like(q0)
        q0_np = np.zeros((N, 20)); qd0_np = np.zeros((N, 20))
        excited = np.zeros((N, 4), dtype=int)
        spring_eff = torch.zeros(N, 1, device=device)  # 1 = spring as a joint effort (spring=2)
        for vi, var in enumerate(variants):
            off = 0.5 if var["hold"] == "mid" else 0.0
            for s in range(n_slot):
                e = vi * n_slot + s
                w = windows[slots[s % len(slots)]]
                t = trials[w.trial]
                kp_n = kp_scale * t.kp * gear
                kd_n = kd_scale * t.kd * gear
                frames = np.minimum(np.floor(w.start + (np.arange(n_steps) + off) * dt / FRAME_DT + 1e-9).astype(int),
                                    len(t.cmd) - 1)
                cmd = t.cmd[frames]  # (n_steps, 20)
                if var["spring"] == 2:
                    spring_eff[e] = 1.0
                if var["spring"] == 1:
                    tgt[e] = (kp_n * cmd + ks * sref) / (kp_n + ks)
                    stiff[e] = kp_n + ks
                else:
                    tgt[e] = cmd
                    stiff[e] = kp_n
                damp[e] = kd_n
                limit[e] = t.effort_limit * gear
                fs[e], fd[e], arm_s[e] = var["fs"], var["fd"], var["arm"]
                q0_np[e] = t.q_all[w.start]
                qd0_np[e] = np.nan_to_num(t.qd_all[w.start])
                excited[e] = t.joints
        assert np.isfinite(q0_np).all()
        set_params(stiff, damp, limit, fs, fd, arm_s)
        q0[:, cols_t] = torch.as_tensor(q0_np, dtype=q0.dtype, device=device)
        qd0[:, cols_t] = torch.as_tensor(qd0_np, dtype=q0.dtype, device=device)
        tgt_t = torch.as_tensor(tgt, device=device)
        robot.write_joint_state_to_sim(q0, qd0)
        robot.set_joint_velocity_target(torch.zeros_like(q0))
        robot.set_joint_effort_target(torch.zeros_like(q0))
        rec[:] = 0.0
        rec[:, 0] = q0[:, cols_t]
        arm_ids = [i for i in range(len(names)) if i not in cols]
        t0 = time.time()
        arm_max = 0.0
        for k in range(n_steps):
            step(1, None, tgt_t[:, k:k + 1], spring_eff if bool(spring_eff.any()) else None)
            rec[:, k + 1] = robot.data.joint_pos[:, cols_t]
            if k % 10 == 0:
                arm_max = max(arm_max, float((robot.data.joint_pos[:, arm_ids] - home[:, arm_ids]).abs().max()))
        arm_dev[mi] = arm_max
        print(f"[replay] {mount}: {n_steps} steps in {time.time() - t0:.1f} s", flush=True)
        rec_np = rec.cpu().numpy()
        for vi in range(V):
            for s in range(len(slots)):
                e = vi * n_slot + s
                wi = slots[s]
                sim_q[vi, wi] = rec_np[e][:, excited[e]]
                other = [c for c in range(20) if c not in excited[e]]
                held_dev[vi, wi] = np.abs(rec_np[e][:, other] - q0_np[e][other]).max()
    report["checks"] = checks
    report["arm_max_dev_rad"] = arm_dev.tolist()
    report["held_fingers_max_dev_rad"] = {v["name"]: float(np.nanmax(held_dev[i])) for i, v in enumerate(variants)}
    report["seconds"] = round(time.time() - t_start, 1)
    print("[replay] report", json.dumps({k: report[k] for k in ("arm_max_dev_rad", "held_fingers_max_dev_rad",
                                                               "gravity_on_bodies", "seconds")}), flush=True)
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, sim_q=sim_q, held_dev=held_dev, dt=dt, n_steps=n_steps,
        win_trial=np.array([w.trial for w in windows]), win_start=np.array([w.start for w in windows]),
        win_stop=np.array([w.stop for w in windows]), trial_files=np.array([t.file for t in trials]),
        variants=json.dumps(variants), report=json.dumps(report))
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[replay] saved {out}", flush=True)
    import os
    os._exit(0)


def sim_view(inner):
    v = getattr(inner.sim, "physics_sim_view", None)
    if v is None:
        v = getattr(inner.sim, "_physics_sim_view", None)
    if v is None:
        import omni.physics.tensors as tensors

        v = tensors.create_simulation_view("torch")
    return v


def _float3(v):
    import carb

    return carb.Float3(float(v[0]), float(v[1]), float(v[2]))


if __name__ == "__main__":
    run()
