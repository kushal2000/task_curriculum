from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def config_path() -> Path:
    override = os.environ.get("AIOFLEXIV_CONFIG")
    if override:
        return Path(override).expanduser()

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "aioflexiv" / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid aioflexiv config at {path}: expected a JSON object")
    return data


def get_last_robot_sn() -> str | None:
    value = load_config().get("last_robot_sn")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def save_last_robot_sn(robot_sn: str) -> None:
    serial = robot_sn.strip()
    if not serial:
        raise ValueError("robot_sn must not be empty")

    path = config_path()
    config = load_config()
    config["last_robot_sn"] = serial

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def resolve_robot_sn(robot_sn: str | None) -> str:
    if robot_sn and robot_sn.strip():
        return robot_sn.strip()

    saved = get_last_robot_sn()
    if saved:
        return saved

    raise RuntimeError(
        "No robot serial number provided and no saved serial number found at "
        f"{config_path()}. Run `aioflexiv status <ROBOT_SN>` once to save one."
    )
