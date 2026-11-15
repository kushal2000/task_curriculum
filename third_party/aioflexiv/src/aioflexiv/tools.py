from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


FLANGE_TCP_LOCATION = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class ToolPayload:
    """Payload parameters from Flexiv's tool model."""

    name: str
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]
    tcp_location: tuple[float, float, float, float, float, float, float]

    @property
    def is_empty(self) -> bool:
        return self.mass <= 0.0


def _float_tuple(values: Iterable[Any], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}, got {len(result)}")
    return result


def _payload_value(source: Any, names: tuple[str, ...], default: Iterable[float]):
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    return default


def tool_payload_from_params(params: Any, *, name: str = "") -> ToolPayload:
    """Convert Flexiv RDK ToolParams-like objects into a stable payload object."""

    payload_name = str(
        _payload_value(params, ("name",), name if name else "tool")
    )
    mass = float(_payload_value(params, ("mass",), 0.0))
    com = _float_tuple(
        _payload_value(params, ("com", "CoM", "center_of_mass"), (0.0, 0.0, 0.0)),
        3,
        "com",
    )
    inertia = _float_tuple(
        _payload_value(params, ("inertia",), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        6,
        "inertia",
    )
    tcp_location = _float_tuple(
        _payload_value(
            params,
            ("tcp_location", "tcp"),
            FLANGE_TCP_LOCATION,
        ),
        7,
        "tcp_location",
    )
    return ToolPayload(
        name=payload_name,
        mass=mass,
        com=com,  # type: ignore[arg-type]
        inertia=inertia,  # type: ignore[arg-type]
        tcp_location=tcp_location,  # type: ignore[arg-type]
    )


def coerce_tool_payload(payload: Any) -> ToolPayload | None:
    if payload is None:
        return None
    if isinstance(payload, ToolPayload):
        return payload
    return tool_payload_from_params(payload)


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.split(".", 1)[-1] if "." in text else text


def call_any_method(obj: Any, names: Iterable[str], *args):
    for name in names:
        method = getattr(obj, name, None)
        if method is not None:
            return method(*args)
    joined = ", ".join(names)
    raise AttributeError(f"{type(obj).__name__} has none of these methods: {joined}")


def switch_robot_to_idle(robot: Any, flexivrdk: Any) -> None:
    mode = getattr(robot, "mode", None)
    if mode is not None and _enum_name(mode()).upper() == "IDLE":
        return

    switch_mode = getattr(robot, "SwitchMode", None)
    idle_mode = getattr(getattr(flexivrdk, "Mode", None), "IDLE", None)
    if switch_mode is not None and idle_mode is not None:
        switch_mode(idle_mode)
        return

    stop = getattr(robot, "Stop", None)
    if stop is not None:
        stop()


def switch_active_tool(
    robot_sn: str,
    tool_name: str,
    *,
    network_interface_whitelist: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Switch the robot's active Flexiv tool using the official Python RDK."""
    if not str(tool_name).strip():
        raise ValueError("tool_name must not be empty")

    import flexivrdk

    robot = flexivrdk.Robot(
        robot_sn,
        network_interface_whitelist or [],
        verbose,
        False,
    )
    tool = flexivrdk.Tool(robot)
    current = str(tool.name())
    if current == tool_name:
        return current

    switch_robot_to_idle(robot, flexivrdk)
    call_any_method(tool, ("Switch", "switch"), tool_name)
    return str(tool.name())


def read_active_tool_payload(
    robot_sn: str,
    *,
    network_interface_whitelist: list[str] | None = None,
    verbose: bool = False,
) -> ToolPayload:
    """Read the currently active Flexiv tool payload through the Python RDK."""

    import flexivrdk

    robot = flexivrdk.Robot(
        robot_sn,
        network_interface_whitelist or [],
        verbose,
        False,
    )
    tool = flexivrdk.Tool(robot)
    name = str(tool.name())
    return tool_payload_from_params(tool.params(), name=name)
