"""Newton only: which robot shape pairs are in contact at the reset pose (hand held at its default
pose)? Lists contact pairs (body labels), their penetration, and the shapes' margin / gap.

    .venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/freespace/diag_selfcontact.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import FIXED_OVERRIDES, RECIPE  # noqa: E402

TASK = "Isaacsimenvs-RigidClothNewton-Direct-v0"


def main() -> None:
    sys.argv = [sys.argv[0], *FIXED_OVERRIDES, "env.scene.num_envs=1", *sys.argv[1:]]
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaaclab_newton.physics import NewtonManager

    from isaacsimenvs.scripts.evaluation.protocol import disable_randomization
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(TASK, "", experiment=RECIPE)
    def run(env_cfg, _a):
        env_cfg.scene.num_envs = 1
        disable_randomization(env_cfg)
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        lay = inner.robot_layout
        canon = list(lay.spec.joint_names_canonical)
        l2c = lay.perm_lab_to_canon
        lower = lay.joint_pos_limits[0, :, 0][l2c]
        upper = lay.joint_pos_limits[0, :, 1][l2c]
        default = lay.default_joint_pos[0][l2c]
        hold = torch.zeros(1, len(canon), device=inner.device)
        hold[0, 7:] = 2 * (default[7:] - lower[7:]) / (upper[7:] - lower[7:]) - 1
        env.reset()
        model = NewtonManager._model
        labels = list(model.body_label) if hasattr(model, "body_label") else list(model.body_key)
        shape_body = model.shape_body.numpy()
        margin = model.shape_margin.numpy()
        gap = model.shape_gap.numpy()
        for step in range(3):
            env.step(hold)
            c = NewtonManager._contacts
            n = int(c.rigid_contact_count.numpy()[0])
            s0 = c.rigid_contact_shape0.numpy()[:n]
            s1 = c.rigid_contact_shape1.numpy()[:n]
            pairs = Counter()
            for a, b in zip(s0, s1):
                la = labels[shape_body[a]].split("/")[-1] if shape_body[a] >= 0 else "world"
                lb = labels[shape_body[b]].split("/")[-1] if shape_body[b] >= 0 else "world"
                if la.startswith(("r_", "link")) and lb.startswith(("r_", "link")):
                    pairs[tuple(sorted((la, lb)))] += 1
            print(f"[diag] step {step}: {n} contacts; robot-robot pairs: {dict(pairs)}", flush=True)
        q = inner.robot.data.joint_pos
        q = (q.torch if hasattr(q, "torch") else q)[0][l2c].cpu().numpy()
        print("[diag] q after 3 steps (hand):", {canon[i]: round(float(q[i]), 4) for i in range(7, len(canon))})
        robot_shapes = [i for i in range(len(shape_body)) if shape_body[i] >= 0 and labels[shape_body[i]].split("/")[-1].startswith("r_")]
        print("[diag] hand shapes margin", np.unique(margin[robot_shapes]), "gap", np.unique(gap[robot_shapes]))
        env.close()

    run()


if __name__ == "__main__":
    main()
