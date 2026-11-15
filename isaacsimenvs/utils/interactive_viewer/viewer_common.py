from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "index.template.html"
DEFAULT_SCENE_PATH = REPO_ROOT / "scene" / "scene_chain_local.json"
DEFAULT_TRAJECTORY_PATH = REPO_ROOT / "trajectory" / "trajectory_chain_demo.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated" / "viewer"


def render_template(template_path: Path, scene_payload: dict) -> str:
    return template_path.read_text(encoding="utf-8").replace(
        "__SCENE_JSON__",
        json.dumps(scene_payload, separators=(",", ":")),
    )


def build_robot_payload(robot_entry: dict) -> dict:
    robot_payload = {
        "name": robot_entry["name"],
        "position": robot_entry.get("position", [0.0, 0.0, 0.0]),
        "rpy": robot_entry.get("rpy", [0.0, 0.0, 0.0]),
        "animated": robot_entry.get("animated", False),
    }

    urdf_url = robot_entry.get("urdf_url")
    urdf_file = robot_entry.get("urdf_file")

    if urdf_url:
        robot_payload["urdf_path"] = urdf_url
    elif urdf_file:
        urdf_path = REPO_ROOT / "robot" / urdf_file
        if not urdf_path.exists():
            raise ValueError(f"Referenced URDF file does not exist: {urdf_file}")
        robot_payload["urdf_file"] = urdf_file
        robot_payload["urdf_path"] = f"robot/{urdf_file}"
    else:
        raise ValueError(f'Scene robot "{robot_entry["name"]}" must define either "urdf_file" or "urdf_url".')

    return robot_payload

