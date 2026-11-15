from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Sequence

from .config import config_path, resolve_robot_sn, save_last_robot_sn
from .tools import FLANGE_TCP_LOCATION, call_any_method, switch_robot_to_idle

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _IS_TTY else ""


BOLD = _c("\033[1m")
DIM = _c("\033[2m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
RED = _c("\033[31m")
RST = _c("\033[0m")


def _get_version() -> str:
    try:
        return version("aioflexiv")
    except PackageNotFoundError:
        return "?"


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.split(".", 1)[-1] if "." in text else text


def _human_enum(value: Any) -> str:
    return _enum_name(value).replace("_", " ").lower()


def _yes_no(value: bool, *, good_when: bool = True) -> str:
    color = GREEN if bool(value) is good_when else RED
    return f"{color}{'yes' if value else 'no'}{RST}"


def _ok_bad(ok: bool, ok_text: str, bad_text: str) -> str:
    return f"{GREEN}{ok_text}{RST}" if ok else f"{RED}{bad_text}{RST}"


def _line(label: str, value: Any) -> None:
    dots = "." * max(2, 18 - len(label))
    print(f"    {label} {dots} {value}")


def _section(name: str) -> None:
    print()
    print(f"  {BOLD}{name}{RST}")


def _safe_call(label: str, func, default: Any = None) -> Any:
    try:
        return func()
    except Exception as exc:
        return default if default is not None else f"unavailable ({exc})"


def _vec(values: Iterable[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _max_abs_line(values: Sequence[float], limits: Sequence[float] | None = None) -> str:
    if not values:
        return "unavailable"
    idx, val = max(enumerate(values), key=lambda item: abs(float(item[1])))
    joint = f"A{idx + 1}"
    text = f"{abs(float(val)):.4f} @ {joint}"
    if limits is not None and idx < len(limits):
        limit = float(limits[idx])
        color = GREEN if abs(float(val)) <= limit else RED
        text = f"{color}{text}{RST} {DIM}(limit {limit:.4f}){RST}"
    return text


def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _tool_params_to_dict(params: Any) -> dict[str, Any]:
    return {
        "mass": float(params.mass),
        "CoM": [float(v) for v in params.CoM],
        "inertia": [float(v) for v in params.inertia],
        "tcp_location": [float(v) for v in params.tcp_location],
    }


def _print_tool_params(params: Any) -> None:
    _line("Mass", f"{float(params.mass):.3f} kg")
    _line("CoM", f"{_vec(params.CoM)} m")

    inertia = list(params.inertia)
    if len(inertia) >= 6:
        ixx, iyy, izz, ixy, ixz, iyz = [float(v) for v in inertia[:6]]
        _line("Inertia", f"[{ixx:.4f}, {ixy:.4f}, {ixz:.4f}] kg*m^2")
        print(f"                       [{ixy:.4f}, {iyy:.4f}, {iyz:.4f}]")
        print(f"                       [{ixz:.4f}, {iyz:.4f}, {izz:.4f}]")

    tcp = list(params.tcp_location)
    if len(tcp) >= 7:
        _line("TCP offset", f"{_vec(tcp[:3])} m")
        _line("TCP quat", _vec(tcp[3:7]))
        _line("TCP RPY", f"{_vec(_quat_to_rpy(*[float(v) for v in tcp[3:7]]))} rad")


def _print_tool(robot: Any, flexivrdk: Any) -> None:
    _section("Tool")
    try:
        tool = flexivrdk.Tool(robot)
        name = tool.name()
        params = tool.params()
    except Exception as exc:
        _line("Status", f"{DIM}unavailable ({exc}){RST}")
        return

    _line("Name", name)
    _print_tool_params(params)


def _print_devices(robot: Any, flexivrdk: Any) -> None:
    try:
        device = flexivrdk.Device(robot)
        names = list(device.list())
    except Exception:
        return
    if not names:
        return

    _section("Devices")
    for name in names:
        enabled = _safe_call("enabled", lambda n=name: device.enabled(n), default=None)
        connected = _safe_call("connected", lambda n=name: device.connected(n), default=None)
        bits = []
        if connected is not None:
            bits.append(_ok_bad(bool(connected), "connected", "disconnected"))
        if enabled is not None:
            bits.append(_ok_bad(bool(enabled), "enabled", "disabled"))
        _line(name, ", ".join(bits) if bits else "present")


def _print_events(robot: Any, count: int) -> None:
    if count <= 0:
        return
    events = _safe_call("event_log", robot.event_log, default=[])
    if not events:
        return

    _section(f"Recent Events")
    for event in list(events)[-count:]:
        level = _enum_name(getattr(event, "level", "UNKNOWN")).lower()
        color = RED if level in {"error", "critical"} else YELLOW if level == "warning" else DIM
        event_id = getattr(event, "id", "?")
        desc = getattr(event, "description", "")
        _line(f"#{event_id}", f"{color}{level}{RST} {desc}")
        action = getattr(event, "recommended_actions", "")
        if action:
            print(f"                       {DIM}{action}{RST}")


def _connect_robot_for_command(args: argparse.Namespace):
    robot_sn = resolve_robot_sn(getattr(args, "robot_sn", None))

    import flexivrdk

    robot = flexivrdk.Robot(
        robot_sn,
        getattr(args, "network_interface", None) or [],
        bool(getattr(args, "verbose", False)),
        False,
    )
    info = robot.info()
    serial = getattr(info, "serial_num", robot_sn) or robot_sn
    save_last_robot_sn(serial)
    return flexivrdk, robot, serial


def _tool_exists(tool: Any, name: str) -> bool:
    return bool(call_any_method(tool, ("exist", "Exist"), name))


def _switch_tool(tool: Any, name: str) -> None:
    call_any_method(tool, ("Switch", "switch"), name)


def _update_tool(tool: Any, name: str, params: Any) -> None:
    call_any_method(tool, ("Update", "update"), name, params)


def _add_tool(tool: Any, name: str, params: Any) -> None:
    call_any_method(tool, ("Add", "add"), name, params)


def _calibrate_payload(tool: Any, tool_mounted: bool) -> Any:
    return call_any_method(
        tool,
        ("CalibratePayloadParams", "calibrate_payload_params"),
        bool(tool_mounted),
    )


def _confirm(prompt: str, *, assume_yes: bool = False, default: bool = False) -> bool:
    if assume_yes:
        _line("Confirm", "yes")
        return True
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"  {prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _wait_for_user(prompt: str, *, assume_yes: bool = False) -> None:
    if assume_yes:
        _line("Step", prompt)
        return
    input(f"  {prompt} Press Enter when ready. ")


def _print_command_error(section: str, exc: Exception, *, status: int = 1) -> int:
    _section(section)
    _line("Status", f"{RED}failed{RST}")
    _line("Error", exc)
    print()
    return status


def _print_tool_command_error(exc: Exception, *, status: int = 1) -> int:
    return _print_command_error("Tool", exc, status=status)


def _tool_error_status(exc: Exception) -> int:
    return 2 if "No robot serial number" in str(exc) else 1


def cmd_tool_status(args: argparse.Namespace) -> int:
    try:
        flexivrdk, robot, serial = _connect_robot_for_command(args)
        tool = flexivrdk.Tool(robot)
        name = tool.name()
        params = tool.params()
    except Exception as exc:
        return _print_tool_command_error(exc, status=_tool_error_status(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "robot_sn": serial,
                    "active_tool": name,
                    "params": _tool_params_to_dict(params),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _section("Tool")
    _line("Robot", serial)
    _line("Name", name)
    _print_tool_params(params)
    print()
    return 0


def cmd_tool_list(args: argparse.Namespace) -> int:
    try:
        flexivrdk, robot, serial = _connect_robot_for_command(args)
        tool = flexivrdk.Tool(robot)
        active = str(tool.name())
        names = list(tool.list())
        entries = [
            {
                "name": str(name),
                "active": str(name) == active,
                "params": _tool_params_to_dict(tool.params(str(name))),
            }
            for name in names
        ]
    except Exception as exc:
        return _print_tool_command_error(exc, status=_tool_error_status(exc))

    if args.json:
        print(
            json.dumps(
                {"robot_sn": serial, "active_tool": active, "tools": entries},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _section("Tools")
    _line("Robot", serial)
    _line("Active", active)
    if not entries:
        _line("Saved", "none")
    for entry in entries:
        marker = "*" if entry["active"] else "-"
        print(f"\n    {marker} {entry['name']}")
        _print_tool_params(tool.params(entry["name"]))
    print()
    return 0


def cmd_tool_load(args: argparse.Namespace) -> int:
    try:
        flexivrdk, robot, serial = _connect_robot_for_command(args)
        tool = flexivrdk.Tool(robot)
        if not _tool_exists(tool, args.name):
            raise RuntimeError(f"Tool {args.name!r} does not exist")
        switch_robot_to_idle(robot, flexivrdk)
        _switch_tool(tool, args.name)
        active = tool.name()
    except Exception as exc:
        return _print_tool_command_error(exc, status=_tool_error_status(exc))

    _section("Tool")
    _line("Robot", serial)
    _line("Loaded", active)
    print()
    return 0


def _tcp_location_for_calibration(
    args: argparse.Namespace,
    existing_params: Any | None,
) -> list[float]:
    if args.tcp_location is not None:
        return [float(v) for v in args.tcp_location]
    if existing_params is not None:
        return [float(v) for v in existing_params.tcp_location]
    return FLANGE_TCP_LOCATION.copy()


def _set_tcp_location(params: Any, tcp_location: Sequence[float]) -> None:
    values = [float(v) for v in tcp_location]
    try:
        params.tcp_location = values
        return
    except (AttributeError, TypeError):
        pass

    current = params.tcp_location
    for idx, value in enumerate(values):
        current[idx] = value


def cmd_tool_calibrate(args: argparse.Namespace) -> int:
    if args.name.strip().lower() == "flange":
        return _print_tool_command_error(
            RuntimeError("Cannot calibrate the reserved Flange tool; choose a tool name")
        )

    try:
        flexivrdk, robot, serial = _connect_robot_for_command(args)
        tool = flexivrdk.Tool(robot)
        info = robot.info()
        has_ft_sensor = bool(getattr(info, "has_FT_sensor", False))
        exists = _tool_exists(tool, args.name)
        existing_params = tool.params(args.name) if exists else None
        tcp_location = _tcp_location_for_calibration(args, existing_params)
    except Exception as exc:
        return _print_tool_command_error(exc, status=_tool_error_status(exc))

    _section("Payload Calibration")
    _line("Robot", serial)
    _line("Tool", args.name)
    _line("Existing", "yes" if exists else "no")
    _line("FT sensor", "yes" if has_ft_sensor else "no")
    tcp_source = (
        "provided"
        if args.tcp_location is not None
        else "existing tool"
        if exists
        else "flange default"
    )
    _line("TCP source", tcp_source)
    _line("TCP", _vec(tcp_location))
    print()
    print("  Stop torque control and clear the robot workspace before continuing.")
    print("  Flexiv requires IDLE mode for tool changes and payload calibration.")
    print(
        "  The calibrated TCP returned by Flexiv is invalid, "
        "so aioflexiv preserves/sets TCP separately."
    )

    if not _confirm("Continue with payload calibration?", assume_yes=args.yes):
        _line("Status", "cancelled")
        print()
        return 130

    try:
        switch_robot_to_idle(robot, flexivrdk)
        _switch_tool(tool, "Flange")
        _wait_for_user(
            f"Mount {args.name!r} rigidly on the flange.",
            assume_yes=args.yes,
        )
        params = _calibrate_payload(tool, True)

        run_unmounted = bool(args.unmounted_pass)
        if not has_ft_sensor and not args.skip_unmounted_pass and not run_unmounted:
            if args.yes:
                _line("Unmounted pass", "skipped")
            else:
                run_unmounted = _confirm(
                    "No FT sensor detected. Run the optional unmounted second pass?",
                    assume_yes=False,
                    default=False,
                )
        if run_unmounted:
            _wait_for_user(
                f"Unmount {args.name!r} from the flange.",
                assume_yes=args.yes,
            )
            params = _calibrate_payload(tool, False)

        _set_tcp_location(params, tcp_location)
        if exists:
            _update_tool(tool, args.name, params)
            action = "updated"
        else:
            _add_tool(tool, args.name, params)
            action = "added"
        if args.load:
            _switch_tool(tool, args.name)
    except Exception as exc:
        return _print_tool_command_error(exc)

    _section("Tool")
    _line("Robot", serial)
    _line("Saved", f"{args.name} ({action})")
    _line("Loaded", "yes" if args.load else "no")
    _print_tool_params(params)
    print()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        robot_sn = resolve_robot_sn(args.robot_sn)
    except Exception as exc:
        _section("Connection")
        _line("Status", f"{RED}failed{RST}")
        _line("Error", exc)
        print()
        return 2

    ver = _get_version()
    print(f"\n  {BOLD}aioflexiv{RST} {DIM}v{ver}{RST}  {DIM}|{RST}  {robot_sn}")

    import flexivrdk

    try:
        robot = flexivrdk.Robot(
            robot_sn, args.network_interface or [], args.verbose, False
        )
    except Exception as exc:
        _section("Connection")
        _line("Status", f"{RED}failed{RST}")
        _line("Error", exc)
        print()
        return 1

    info = robot.info()
    serial = getattr(info, "serial_num", robot_sn) or robot_sn
    try:
        save_last_robot_sn(serial)
    except Exception as exc:
        _section("Config")
        _line("Warning", f"Could not save last serial to {config_path()} ({exc})")

    states = _safe_call("states", robot.states, default=None)

    op_status = robot.operational_status()
    op_name = _enum_name(op_status)
    operational = robot.operational()
    op_color = GREEN if operational else RED if "FAULT" in op_name else YELLOW

    _section("System")
    _line("Connected", _yes_no(robot.connected()))
    _line("Serial", serial)
    _line("Model", getattr(info, "model_name", "?") or "?")
    _line("Software", getattr(info, "software_ver", "?") or "?")
    _line("License", getattr(info, "license_type", "?") or "?")
    _line("Operational", f"{op_color}{_human_enum(op_status)}{RST}")
    _line("Mode", _human_enum(robot.mode()))

    _section("Robot")
    _line("Fault", _yes_no(robot.fault(), good_when=False))
    _line("Recovery", _yes_no(robot.recovery(), good_when=False))
    _line("Reduced", _yes_no(robot.reduced(), good_when=False))
    _line("Stopped", _yes_no(robot.stopped()))
    _line("Busy", _yes_no(robot.busy(), good_when=False))
    _line("E-Stop released", _yes_no(robot.estop_released()))
    _line("Enable button", _ok_bad(robot.enabling_button_pressed(), "pressed", "released"))
    _line("Timeliness limit", _yes_no(robot.reached_timeliness_failure_limit(), good_when=False))
    _line("DoF", f"{info.DoF} total, {info.DoF_m} manipulator, {info.DoF_e} external")
    _line("FT sensor", _yes_no(bool(info.has_FT_sensor)))

    q_min = list(info.q_min)
    q_max = list(info.q_max)
    dq_max = list(info.dq_max)
    tau_max = list(info.tau_max)
    if q_min and q_max:
        _line("q min", f"{_vec(q_min, 3)} rad")
        _line("q max", f"{_vec(q_max, 3)} rad")
    if dq_max:
        _line("dq max", f"{_vec(dq_max, 3)} rad/s")
    if tau_max:
        _line("tau max", f"{_vec(tau_max, 2)} Nm")

    if states is not None:
        q = list(states.q)
        dq = list(states.dq)
        dtheta = list(states.dtheta)
        tau = list(states.tau)
        tau_ext = list(states.tau_ext)
        _section("State")
        _line("Timestamp", tuple(states.timestamp))
        _line("q", f"{_vec(q)} rad")
        _line("dq", f"{_vec(dq)} rad/s")
        _line("dtheta", f"{_vec(dtheta)} rad/s")
        _line("Max |dq|", _max_abs_line(dq, dq_max))
        _line("Max |dtheta|", _max_abs_line(dtheta, dq_max))
        _line("tau", f"{_vec(tau, 3)} Nm")
        _line("tau_ext", f"{_vec(tau_ext, 3)} Nm")
        if hasattr(states, "temperature"):
            _line("Temperature", f"{_vec(states.temperature, 1)} C")

        _section("TCP")
        _line("Pose", f"{_vec(states.tcp_pose)} [m, quat]")
        _line("Velocity", f"{_vec(states.tcp_vel)} [m/s, rad/s]")
        _line("Flange pose", f"{_vec(states.flange_pose)} [m, quat]")
        _line("Ext wrench TCP", f"{_vec(states.ext_wrench_in_tcp, 3)} [N, Nm]")
        _line("Ext wrench world", f"{_vec(states.ext_wrench_in_world, 3)} [N, Nm]")

    _print_tool(robot, flexivrdk)
    _print_devices(robot, flexivrdk)
    _print_events(robot, args.events)
    print()
    return 0


def _float_in_range(low: float, high: float = math.inf):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid number: {text!r}") from None
        if not (math.isfinite(value) and low <= value <= high):
            bounds = f"at least {low:g}" if math.isinf(high) else f"from {low:g} to {high:g}"
            raise argparse.ArgumentTypeError(f"must be {bounds}, got {text}")
        return value

    return parse


def _gravcomp_robot(args: argparse.Namespace):
    from .robot import FlexivRobotInterface, MujocoRobotInterface, is_mujoco_robot_sn

    if is_mujoco_robot_sn(args.robot_sn):
        return MujocoRobotInterface()
    return FlexivRobotInterface(
        args.robot_sn,
        tool=args.tool,
        friction_comp_scale=args.friction_comp_scale,
        network_interface_whitelist=args.network_interface,
        verbose=args.verbose,
    )


async def _serve_qpos(
    controller: Any, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        request_line = await reader.readline()
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass  # skip the headers
        parts = request_line.split()
        if len(parts) > 1 and parts[1] == b"/qpos":
            state = controller.state
            qpos = None if state is None else [round(float(v), 6) for v in state["qpos"]]
            status, body = "200 OK", {"qpos": qpos}
        else:
            status, body = "404 Not Found", {"error": "not found, try GET /qpos"}
        payload = json.dumps(body).encode()
        header = (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n\r\n"
        )
        writer.write(header.encode() + payload)
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


async def _run_gravcomp(controller: Any, *, http_port: int = 0) -> None:
    """Run the controller's torque loop until SIGINT or SIGTERM, then stop the robot."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    signals = (signal.SIGINT, signal.SIGTERM)
    # Installed before the robot starts, so that Ctrl+C always ends in controller.stop().
    for sig in signals:
        loop.add_signal_handler(sig, stop.set)
    server = None
    try:
        if http_port:
            server = await asyncio.start_server(
                lambda reader, writer: _serve_qpos(controller, reader, writer),
                "0.0.0.0",
                http_port,
            )
        task = await controller.start()
        print(f"\n  {GREEN}Running{RST} {DIM}- move the robot by hand. Ctrl+C stops.{RST}", flush=True)
        stopped = asyncio.create_task(stop.wait())
        await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
        stopped.cancel()
    finally:
        try:
            # Raises the torque loop's error if the loop ended on its own.
            await controller.stop()
        finally:
            if server is not None:
                server.close()
            for sig in signals:
                loop.remove_signal_handler(sig)


def cmd_gravcomp(args: argparse.Namespace) -> int:
    from .controller import FlexivController

    try:
        robot = _gravcomp_robot(args)
    except Exception as exc:
        return _print_command_error("Gravity Compensation", exc, status=2)

    controller = FlexivController(robot)
    controller.kp = 0.0
    controller.kd = args.damping

    print(f"\n  {BOLD}aioflexiv{RST} {DIM}v{_get_version()}{RST}  {DIM}|{RST}  {robot.robot_sn}")
    _section("Gravity Compensation")
    if args.tool:
        _line("Tool", args.tool)
    _line("Damping", f"kd = {args.damping:g} Nm*s/rad")
    _line("Friction comp", f"{args.friction_comp_scale:g}%")
    if args.http_port:
        _line("HTTP", f"GET /qpos on port {args.http_port}")
    print()
    print("  Keep clear of the robot until it is running.", flush=True)

    try:
        asyncio.run(_run_gravcomp(controller, http_port=args.http_port))
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        return _print_command_error("Gravity Compensation", exc)

    print(f"\n  {GREEN}Stopped{RST}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aioflexiv command line tools")
    parser.add_argument("--version", action="version", version=f"aioflexiv {_get_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_robot_arg(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "robot_sn",
            nargs="?",
            help=(
                "Robot serial number. If omitted, use the latest serial saved in "
                f"{config_path()}."
            ),
        )

    def add_connection_options(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "--network-interface",
            action="append",
            help=(
                "Whitelist a local IPv4 interface while searching for the specified robot; "
                "may be repeated."
            ),
        )
        cmd.add_argument(
            "--verbose",
            action="store_true",
            help="Enable Flexiv RDK verbose output.",
        )

    status = subparsers.add_parser("status", help="Show Flexiv robot status")
    add_robot_arg(status)
    add_connection_options(status)
    status.add_argument("--events", type=int, default=3, help="Number of recent events to show.")
    status.set_defaults(func=cmd_status)

    gravcomp = subparsers.add_parser(
        "gravcomp",
        help="Gravity compensation mode (foreground, Ctrl+C to stop)",
        description=(
            "Run the torque loop with zero stiffness so the robot can be moved by hand: "
            "Flexiv compensates gravity and the active tool's payload. Ctrl+C stops the "
            "robot and switches it to IDLE."
        ),
    )
    gravcomp.add_argument(
        "robot_sn",
        nargs="?",
        help=(
            "Robot serial number, or 'mujoco' to run the MuJoCo simulation. If omitted, "
            f"use the latest serial saved in {config_path()}."
        ),
    )
    add_connection_options(gravcomp)
    gravcomp.add_argument(
        "--damping",
        type=_float_in_range(0.0),
        default=0.0,
        help="Joint velocity damping (kd) per joint, in Nm*s/rad (default: 0).",
    )
    gravcomp.add_argument(
        "--http-port",
        type=int,
        default=0,
        help="Serve GET /qpos on this port (e.g. 8080).",
    )
    gravcomp.add_argument(
        "--tool",
        help=(
            "Load this saved Flexiv tool before starting, so that gravity compensation "
            "includes its payload. Defaults to the active tool."
        ),
    )
    gravcomp.add_argument(
        "--friction-comp-scale",
        type=_float_in_range(0.0, 100.0),
        default=100.0,
        help=(
            "Percentage of joint friction that Flexiv compensates, 0 to 100 (default: 100). "
            "Lower values add natural damping."
        ),
    )
    gravcomp.set_defaults(func=cmd_gravcomp)

    tool = subparsers.add_parser("tool", help="Manage Flexiv robot tool payloads")
    tool_subparsers = tool.add_subparsers(dest="tool_command", required=True)

    tool_status = tool_subparsers.add_parser("status", help="Show the active tool")
    add_robot_arg(tool_status)
    add_connection_options(tool_status)
    tool_status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    tool_status.set_defaults(func=cmd_tool_status)

    tool_list = tool_subparsers.add_parser("list", help="List saved tools and parameters")
    add_robot_arg(tool_list)
    add_connection_options(tool_list)
    tool_list.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    tool_list.set_defaults(func=cmd_tool_list)

    tool_load = tool_subparsers.add_parser("load", help="Load an existing tool on the robot")
    tool_load.add_argument("name", help="Name of the saved Flexiv tool to load.")
    add_robot_arg(tool_load)
    add_connection_options(tool_load)
    tool_load.set_defaults(func=cmd_tool_load)

    tool_calibrate = tool_subparsers.add_parser(
        "calibrate",
        help="Interactively calibrate payload parameters and save a Flexiv tool",
    )
    tool_calibrate.add_argument("name", help="Name of the Flexiv tool to add or update.")
    add_robot_arg(tool_calibrate)
    add_connection_options(tool_calibrate)
    tool_calibrate.add_argument(
        "--tcp-location",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help=(
            "TCP pose in flange frame. Defaults to the existing tool TCP, or flange TCP "
            "for a new tool."
        ),
    )
    second_pass = tool_calibrate.add_mutually_exclusive_group()
    second_pass.add_argument(
        "--unmounted-pass",
        action="store_true",
        help="Run Flexiv's optional second calibration pass with the tool unmounted.",
    )
    second_pass.add_argument(
        "--skip-unmounted-pass",
        action="store_true",
        help="Skip the optional unmounted pass even when no FT sensor is detected.",
    )
    tool_calibrate.add_argument(
        "--load",
        action="store_true",
        help="Switch to the calibrated tool after saving it.",
    )
    tool_calibrate.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Accept prompts and use defaults for non-interactive calibration.",
    )
    tool_calibrate.set_defaults(func=cmd_tool_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
