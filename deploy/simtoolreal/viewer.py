"""The viewer node: a live viser page of everything on the bus. Only listens, never publishes, so it
can be started and stopped at any time without touching the run.

    .venv_deploy/bin/python deploy/simtoolreal/viewer.py      # then open http://<robot PC>:8091

Subscribes  joint_state  -> the robot, at its measured joint angles
            joint_target -> a green see-through robot at the policy's targets
            object_pose  -> the object mesh
            goal_pose    -> an orange see-through object mesh
            camera_image -> the camera image in the side panel (if perception sends it)

The scene is in the env frame (the training env's): table and robot base where training had them.
The status panel lists each topic's age and rate, and the object -> goal distance against the
success threshold. replay.py draws recordings with the same build_scene / draw.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from utils.config import REPO, load_config, repo_path                                 # noqa: E402
from utils.goals import keypoint_distance                                             # noqa: E402
from utils.link import TOPIC_PORTS, age, decode_jpeg, open_subscriber, recv_latest    # noqa: E402
from utils.spec import base_to_env, load_spec                                         # noqa: E402

MESH_DIR = REPO / "assets" / "flexiv_wuji"   # the URDF's mesh paths are relative to it (fetch_assets.sh)


def build_scene(server, spec, mesh_path: Path) -> dict:
    """Table, robot, target ghost, object, goal, and the side panel. Returns the handles."""
    import trimesh
    import yourdfpy
    from viser.extras import ViserUrdf

    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=2.0, height=2.0, cell_size=0.25)
    server.scene.add_box("/table", dimensions=spec["table_size"], position=spec["table_pos_env"],
                         color=(0.55, 0.45, 0.35))
    root = server.scene.add_frame("/robot_root", position=spec["robot_root_pos_env"],
                                  wxyz=spec["robot_root_quat_wxyz"], show_axes=False)

    def load_urdf():
        return yourdfpy.URDF.load(spec["urdf_path"], mesh_dir=str(MESH_DIR), build_collision_scene_graph=False)

    robot = ViserUrdf(server, load_urdf(), root_node_name=root.name + "/measured")
    ghost = ViserUrdf(server, load_urdf(), root_node_name=root.name + "/targets",
                      mesh_color_override=(0.15, 0.85, 0.35, 0.3))
    mesh = trimesh.load(mesh_path, force="mesh")
    obj = server.scene.add_mesh_trimesh("/object", mesh)
    server.scene.add_frame("/object/axes", axes_length=0.06, axes_radius=0.002)
    goal = server.scene.add_mesh_simple("/goal", vertices=mesh.vertices, faces=mesh.faces,
                                        color=(0.95, 0.6, 0.1), opacity=0.4)
    server.scene.add_frame("/goal/axes", axes_length=0.06, axes_radius=0.002)
    return {
        "robot": robot, "ghost": ghost, "object": obj, "goal": goal,
        # viser's joint order -> index in the spec's joint order
        "order": [list(spec["joint_names"]).index(j) for j in robot.get_actuated_joint_names()],
        "show_targets": server.gui.add_checkbox("show joint targets (green)", True),
        "status": server.gui.add_markdown("waiting for messages ..."),
        "image": server.gui.add_image(np.zeros((48, 64, 3), np.uint8), label="camera"),
    }


def draw(server, scene: dict, spec, msgs: dict) -> None:
    """Put the robot, target ghost, object and goal where these messages say (topic -> newest message;
    missing or None topics are skipped). The camera image is not drawn here (decoding costs ~ms)."""
    order = scene["order"]
    with server.atomic():
        if msgs.get("joint_state") is not None:
            scene["robot"].update_cfg(msgs["joint_state"]["q"][order])
        show = scene["show_targets"].value and msgs.get("joint_target") is not None
        scene["ghost"].show_visual = show
        if show:
            scene["ghost"].update_cfg(msgs["joint_target"]["targets"][order])
        for topic in ("object_pose", "goal_pose"):
            if msgs.get(topic) is not None:
                handle = scene["object" if topic == "object_pose" else "goal"]
                handle.position, handle.wxyz = base_to_env(spec, msgs[topic]["pos"], msgs[topic]["quat_wxyz"])


def goal_text(spec, last: dict, tolerance: float) -> str:
    """"object -> goal ..." line, or "" without both poses."""
    o, g = last.get("object_pose"), last.get("goal_pose")
    if o is None or g is None:
        return ""
    d = keypoint_distance(spec, *base_to_env(spec, o["pos"], o["quat_wxyz"]),
                          *base_to_env(spec, g["pos"], g["quat_wxyz"]))
    return (f"object -> goal (keypoints, worst): **{100 * d:.1f} cm** "
            f"(success below {100 * tolerance:.1f} cm) · goal {g['goal_index'] + 1}")


def status_text(spec, last: dict, rates: dict, stale_s: float, tolerance: float) -> str:
    """Markdown: one line per topic, then the object -> goal distance."""
    lines = []
    for topic, m in last.items():
        if m is None:
            lines.append(f"- `{topic}`: nothing yet")
        else:
            a = age(m)
            flag = " **STALE**" if a > stale_s else ""
            lines.append(f"- `{topic}`: {rates[topic]:.0f} Hz, {1000 * a:.0f} ms old{flag}")
    state = last.get("joint_state")
    if state is not None and state.get("tripped"):
        lines.insert(0, f"**SAFETY TRIP: {state['tripped']}**\n")
    return "\n".join(lines) + "\n\n" + goal_text(spec, last, tolerance)


def success_tolerance(cfg, spec) -> float:
    """m: the keypoint distance under which the object counts as at the goal."""
    return (cfg.goals.success_tolerance or spec["success_tolerance_final"]) * spec["keypoint_scale"]


def main() -> None:
    import viser

    cfg = load_config()
    vc = cfg.viewer
    spec = load_spec(repo_path(cfg.spec_path))
    tolerance = success_tolerance(cfg, spec)

    server = viser.ViserServer(host="0.0.0.0", port=vc.port)
    scene = build_scene(server, spec, repo_path(cfg.objects[cfg.object.name].mesh))
    subs = {topic: open_subscriber(cfg.bus.host, cfg.bus[port], topic) for topic, port in TOPIC_PORTS.items()}
    last = dict.fromkeys(TOPIC_PORTS)                  # newest message per topic
    counts = dict.fromkeys(TOPIC_PORTS, 0)             # messages since t_rate
    t_rate = time.time()
    print(f"[viewer] http://localhost:{vc.port}", flush=True)

    while True:
        t0 = time.perf_counter()
        new = {}
        for topic, sub in subs.items():
            m = recv_latest(sub)
            if m is not None:
                new[topic] = last[topic] = m
                counts[topic] += 1
        if new:
            draw(server, scene, spec, last)
        if "camera_image" in new:
            scene["image"].image = decode_jpeg(new["camera_image"]["jpeg"])

        now = time.time()
        if now - t_rate >= 1.0:
            rates = {k: v / (now - t_rate) for k, v in counts.items()}
            counts, t_rate = dict.fromkeys(TOPIC_PORTS, 0), now
            scene["status"].content = status_text(spec, last, rates, vc.stale_s, tolerance)

        time.sleep(max(0.0, 1.0 / vc.hz - (time.perf_counter() - t0)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
