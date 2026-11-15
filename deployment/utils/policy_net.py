"""What every task's policy node shares: the trained network, its action -> joint targets, waiting for
the robot and the object before the first step, and the 60 Hz step clock. Needs torch.

    net = load_network(checkpoint, config, expl_coef, num_actions)
    state, obj = wait_for_inputs(sub_state, sub_object)
    targets = action_to_targets(spec, prev_targets, action)
    t_next = time.perf_counter(); ... t_next = wait_step(t_next, dt, step)   # once per step, at its end
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from .config import REPO
from .link import recv_latest
from .observation import NUM_OBS

OBJECT_LOST_BELOW_TABLE = 0.05   # m: object this far below the table top = fell off


def load_network(checkpoint: Path, config: Path, expl_coef: float, num_actions: int):
    """The trained policy: an LSTM (rl_games SAPG checkpoint), one robot, on the CPU.

    Use: net.get_action(obs (1, 134) tensor, deterministic=True) -> (1, 27) actions in [-1, 1], once
    per step, in order (the LSTM remembers past steps); net.reset_rnn() clears that memory.

    Built through the training repo's isaacsimenvs/eval/player.py (rl_games reads
    pretrained_policy/config.yaml). SAPG trains several policy "blocks" that differ only in one extra
    input, the exploration coefficient; expl_coef 0.0 is the one trained on the task reward only.
    """
    # One CPU thread. The net is tiny (0.6 ms on 8 threads, 2 ms on 1, idle), but torch's default
    # thread pool spins on every core: with FoundationPose, the recorder and the viewer busy on the
    # same PC, it took 21 ms median, 50 ms p99, and the loop fell to 30 Hz. 1 thread under that load: 2.3 ms.
    torch.set_num_threads(1)
    # load player.py by path: importing the isaacsimenvs package would pull in Isaac Lab
    path = REPO / "isaacsimenvs" / "eval" / "player.py"
    module_spec = importlib.util.spec_from_file_location("_play_player", path)
    player = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(player)
    return player.PretrainedPlayer(config_path=config, checkpoint_path=checkpoint, num_envs=1, device="cpu",
                                   num_observations=NUM_OBS, num_actions=num_actions, expl_coef=expl_coef)


def action_to_targets(spec: dict, prev_targets: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Policy action (27 numbers in [-1, 1]) -> joint position targets (rad), as the training env
    (action_utils.apply_action_pipeline, no delay):

    Arm (first 7): a velocity command, target = prev + dof_speed_scale * dt * action.
    Hand (last 20): an absolute position, -1 -> lower limit, +1 -> upper limit.
    Both then smoothed, new = 0.1 * target + 0.9 * prev, and clamped to the joint limits.
    """
    n = spec["num_arm_joints"]
    lo, hi = spec["joint_lower"], spec["joint_upper"]
    arm = np.clip(prev_targets[:n] + spec["dof_speed_scale"] * spec["policy_dt"] * action[:n], lo[:n], hi[:n])
    hand = lo[n:] + 0.5 * (action[n:] + 1.0) * (hi[n:] - lo[n:])
    a_arm, a_hand = spec["arm_moving_average"], spec["hand_moving_average"]
    arm = a_arm * arm + (1.0 - a_arm) * prev_targets[:n]
    hand = a_hand * hand + (1.0 - a_hand) * prev_targets[n:]
    return np.clip(np.concatenate([arm, hand]), lo, hi)


def wait_for_inputs(sub_state, sub_object, timeout_s: float = 30.0) -> tuple[dict, dict]:
    """Block until robot.py is ready (hand home, forwarding targets) and an object_pose has arrived.
    Starting earlier takes prev_targets from a hand still moving home: the first targets then jump."""
    state = obj = None
    t_end = time.time() + timeout_s
    while state is None or obj is None:
        if time.time() > t_end:
            missing = [name for name, m in (("ready joint_state", state), ("object_pose", obj)) if m is None]
            raise SystemExit(f"[policy] no {' and no '.join(missing)} after {timeout_s:.0f} s: is the publisher running?")
        new = recv_latest(sub_state)
        if new is not None and new.get("ready"):
            state = new
        obj = recv_latest(sub_object) or obj
        time.sleep(0.01)
    return state, obj


def wait_step(t_next: float, dt: float, step: int, name: str = "policy") -> float:
    """Keep the loop at one step per dt: sleep until t_next + dt and return that as the next t_next.
    More than a whole step behind: say so and start again from now instead of catching up.
    Start with t_next = time.perf_counter() before the loop."""
    t_next += dt
    late = time.perf_counter() - t_next
    if late < 0:
        time.sleep(-late)
    elif late > dt:   # more than a whole step behind: don't try to catch up
        print(f"[{name}] step {step} {1000 * late:.0f} ms late", flush=True)
        t_next = time.perf_counter()
    return t_next
