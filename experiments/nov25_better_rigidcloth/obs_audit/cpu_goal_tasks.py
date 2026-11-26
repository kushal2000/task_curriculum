"""CPU check: the box2-mid half_fold goal task's moving half and goal grid vs VBD's stage-1 goal."""
import numpy as np, json
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_states as ft, fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh, half_indices, corner_indices
sh = fs.shape("box2-mid")
st = ft.enumerate_tasks(sh)
rows = ft.task_arrays(st, 7, only_half=True)
print("shape size", sh.size, "n half tasks", len(rows))
v, _ = grid_mesh(sh.size, 7)
v = np.array(v)
for r in rows:
    mv = np.nonzero(r["moving"])[0]
    print("name", r["name"], "moving idx", mv.tolist())
    print("kp", r["kp"].tolist(), "direction deg", np.degrees(r["direction"]))
    print("p_start - vbd rest max err", np.abs(r["p_start"] - v).max())
    print("moving x mean", r["p_start"][mv, 0].mean())
    print("p_goal z range", r["p_goal"][:, 2].min(), r["p_goal"][:, 2].max())
print("VBD moving (+x) idx", sorted(half_indices(7, "x", True)))
print("VBD keypoints", corner_indices(7, "x"))
