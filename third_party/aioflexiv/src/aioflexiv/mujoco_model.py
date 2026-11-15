from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .tools import ToolPayload, coerce_tool_payload


DEFAULT_MUJOCO_SITE_NAME = "attachment_site"
MUJOCO_TOOL_PAYLOAD_BODY_NAME = "aioflexiv_tool_payload"


def _model_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / "models",
        Path.cwd() / "models",
        Path(sys.prefix) / "share" / "aioflexiv" / "models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_mujoco_model_path(model_name: str | None = None) -> Path:
    normalized = "" if model_name is None else model_name.lower().replace(" ", "")
    model_dir = _model_root()

    if "rizon4s" in normalized or "a02ls" in normalized:
        return model_dir / "flexiv_rizon4s" / "flexiv_rizon4s.xml"
    if "rizon4" in normalized or "a02l" in normalized:
        return model_dir / "flexiv_rizon4" / "flexiv_rizon4.xml"
    return model_dir / "flexiv_rizon4" / "flexiv_rizon4.xml"


def default_mujoco_scene_path(model_name: str | None = None) -> Path:
    model_path = default_mujoco_model_path(model_name)
    scene_path = model_path.with_name("scene.xml")
    return scene_path if scene_path.exists() else model_path


def _format_vec(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _resolve_xml_path(path: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    return (path.parent / candidate).resolve()


def _adjust_compiler_paths(root: ET.Element, path: Path) -> None:
    for compiler in root.findall("compiler"):
        meshdir = compiler.get("meshdir")
        if meshdir:
            mesh_path = Path(meshdir)
            if not mesh_path.is_absolute():
                compiler.set("meshdir", str((path.parent / mesh_path).resolve()))


def _inertia_matrix(inertia: tuple[float, float, float, float, float, float]):
    ixx, iyy, izz, ixy, ixz, iyz = inertia
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


def _validate_tool_payload(payload: ToolPayload) -> None:
    if not np.isfinite(payload.mass) or payload.mass < 0.0:
        raise ValueError("Tool payload mass must be finite and non-negative")
    if payload.is_empty:
        return

    com = np.asarray(payload.com, dtype=float)
    inertia = _inertia_matrix(payload.inertia)
    if not np.all(np.isfinite(com)):
        raise ValueError("Tool payload CoM must contain finite values")
    if not np.all(np.isfinite(inertia)):
        raise ValueError("Tool payload inertia must contain finite values")
    eigs = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    if eigs[0] <= 0.0:
        raise ValueError(
            "Tool payload inertia must be positive definite when mass is positive"
        )


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    return None


def _remove_existing_payload_body(parent: ET.Element) -> None:
    for child in list(parent):
        if child.tag == "body" and child.get("name") == MUJOCO_TOOL_PAYLOAD_BODY_NAME:
            parent.remove(child)


def _insert_payload_body(
    root: ET.Element,
    *,
    payload: ToolPayload,
    site_name: str | None,
    body_name: str,
) -> bool:
    parent = None
    payload_body_attrs: dict[str, str] = {
        "name": MUJOCO_TOOL_PAYLOAD_BODY_NAME,
        "gravcomp": "1",
    }

    if site_name is not None:
        site = _find_named(root, "site", site_name)
        if site is None:
            return False
        parent = _parent_map(root).get(site)
        if parent is None or parent.tag != "body":
            raise ValueError(f"MuJoCo site {site_name!r} is not attached to a body")
        payload_body_attrs["pos"] = site.get("pos", "0 0 0")
        if site.get("quat") is not None:
            payload_body_attrs["quat"] = site.get("quat", "1 0 0 0")
    else:
        parent = _find_named(root, "body", body_name)
        if parent is None:
            return False
        payload_body_attrs["pos"] = "0 0 0"

    _remove_existing_payload_body(parent)
    payload_body = ET.SubElement(parent, "body", payload_body_attrs)
    ET.SubElement(
        payload_body,
        "inertial",
        {
            "pos": _format_vec(payload.com),
            "mass": f"{payload.mass:.10g}",
            "fullinertia": _format_vec(payload.inertia),
        },
    )
    return True


def _rewrite_payload_xml(
    source_path: Path,
    dest_dir: Path,
    *,
    payload: ToolPayload,
    site_name: str | None,
    body_name: str,
) -> Path | None:
    tree = ET.parse(source_path)
    root = tree.getroot()
    _adjust_compiler_paths(root, source_path)

    include_paths: list[tuple[ET.Element, Path]] = []
    for include in root.findall(".//include"):
        filename = include.get("file")
        if filename:
            include_paths.append((include, _resolve_xml_path(source_path, filename)))

    if _insert_payload_body(
        root,
        payload=payload,
        site_name=site_name,
        body_name=body_name,
    ):
        for include, include_path in include_paths:
            include.set("file", str(include_path))
        dest_path = dest_dir / source_path.name
        tree.write(dest_path, encoding="unicode")
        return dest_path

    for include, include_path in include_paths:
        patched_include = _rewrite_payload_xml(
            include_path,
            dest_dir,
            payload=payload,
            site_name=site_name,
            body_name=body_name,
        )
        if patched_include is None:
            include.set("file", str(include_path))
            continue
        for other_include, other_include_path in include_paths:
            other_include.set(
                "file",
                patched_include.name
                if other_include is include
                else str(other_include_path),
            )
        dest_path = dest_dir / source_path.name
        tree.write(dest_path, encoding="unicode")
        return dest_path

    return None


def _write_payload_model_xml(
    source_path: Path,
    dest_dir: Path,
    *,
    payload: ToolPayload,
    site_name: str | None,
    body_name: str,
) -> Path:
    patched_path = _rewrite_payload_xml(
        source_path,
        dest_dir,
        payload=payload,
        site_name=site_name,
        body_name=body_name,
    )
    if patched_path is None:
        frame = f"site {site_name!r}" if site_name is not None else f"body {body_name!r}"
        raise ValueError(f"Could not find MuJoCo {frame} to attach tool payload")
    return patched_path


class MujocoModelBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        site_name: str | None = DEFAULT_MUJOCO_SITE_NAME,
        body_name: str = "link7",
        tool_payload: ToolPayload | dict | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo model backend requires the `mujoco` Python package."
            ) from exc

        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo model XML not found: {path}")

        self._mujoco = mujoco
        self.path = path
        self.tool_payload = coerce_tool_payload(tool_payload)
        if self.tool_payload is not None:
            _validate_tool_payload(self.tool_payload)
        self._payload_xml_dir: tempfile.TemporaryDirectory[str] | None = None
        compile_path = path
        if self.tool_payload is not None and not self.tool_payload.is_empty:
            self._payload_xml_dir = tempfile.TemporaryDirectory(
                prefix="aioflexiv_mujoco_"
            )
            compile_path = _write_payload_model_xml(
                path,
                Path(self._payload_xml_dir.name),
                payload=self.tool_payload,
                site_name=site_name,
                body_name=body_name,
            )
        self.model = mujoco.MjModel.from_xml_path(str(compile_path))
        self.data = mujoco.MjData(self.model)
        self._gravity_data = mujoco.MjData(self.model)
        # MuJoCo 3.10 changed mj_fullM(m, dst, qM) to mj_fullM(m, d, dst), and 3.11 removed
        # MjData.qM.
        major, minor = (int(part) for part in mujoco.__version__.split(".")[:2])
        self._full_mass_matrix_takes_data = (major, minor) >= (3, 10)

        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError(
                f"Expected a 7-DoF Flexiv model, got nq={self.model.nq}, "
                f"nv={self.model.nv} from {path}"
            )

        self.site_name = site_name
        self.body_name = body_name
        if site_name is not None:
            self.frame_type = "site"
            self.frame_name = site_name
            self.frame_id = self.model.site(site_name).id
        else:
            self.frame_type = "body"
            self.frame_name = body_name
            self.frame_id = self.model.body(body_name).id

    def state(self, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, np.ndarray]:
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos must have shape {(self.model.nq,)}, got {qpos.shape}"
            )
        if qvel.shape != (self.model.nv,):
            raise ValueError(
                f"qvel must have shape {(self.model.nv,)}, got {qvel.shape}"
            )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, self.data)

        ee = self._ee()
        jac = self._jacobian()
        mm = self._mass_matrix(self.data)
        bias = np.asarray(self.data.qfrc_bias, dtype=float).copy()
        gravity = self._gravity(qpos)

        return {
            "ee": ee,
            "jac": jac,
            "mm": mm,
            "coriolis": bias - gravity,
            "gravity": gravity,
            "mujoco_model_path": np.asarray([str(self.path)], dtype=object),
            "mujoco_frame_type": np.asarray([self.frame_type], dtype=object),
            "mujoco_frame_name": np.asarray([self.frame_name], dtype=object),
            "mujoco_tool_name": np.asarray([self.tool_name], dtype=object),
            "mujoco_tool_mass": np.asarray([self.tool_mass], dtype=float),
        }

    @property
    def tool_name(self) -> str:
        return "" if self.tool_payload is None else self.tool_payload.name

    @property
    def tool_mass(self) -> float:
        return 0.0 if self.tool_payload is None else float(self.tool_payload.mass)

    def gravity(self, qpos: np.ndarray) -> np.ndarray:
        return self._gravity(np.asarray(qpos, dtype=float))

    def _ee(self) -> np.ndarray:
        ee = np.eye(4)
        if self.frame_type == "site":
            frame = self.data.site(self.frame_id)
        else:
            frame = self.data.body(self.frame_id)
        ee[:3, :3] = frame.xmat.reshape(3, 3)
        ee[:3, 3] = frame.xpos
        return ee

    def _jacobian(self) -> np.ndarray:
        jac = np.zeros((6, self.model.nv))
        if self.frame_type == "site":
            self._mujoco.mj_jacSite(
                self.model, self.data, jac[:3], jac[3:], self.frame_id
            )
        else:
            self._mujoco.mj_jacBody(
                self.model, self.data, jac[:3], jac[3:], self.frame_id
            )
        return jac

    def _mass_matrix(self, data) -> np.ndarray:
        mm = np.zeros((self.model.nv, self.model.nv))
        if self._full_mass_matrix_takes_data:
            self._mujoco.mj_fullM(self.model, data, mm)
        else:
            self._mujoco.mj_fullM(self.model, mm, data.qM)
        return mm

    def _gravity(self, qpos: np.ndarray) -> np.ndarray:
        self._gravity_data.qpos[:] = qpos
        self._gravity_data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self._gravity_data)
        return np.asarray(self._gravity_data.qfrc_bias, dtype=float).copy()
