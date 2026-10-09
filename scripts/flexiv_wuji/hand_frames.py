"""The canonical hand frame both left hands are compared in (numpy + yourdfpy).

Built the same way from each hand's knuckles: origin at the middle-finger MCP, x from the hand
base towards it (along the fingers), y from the pinky MCP towards the index MCP, z = x cross y
(palm side). The two vendors' link frames differ; this frame does not.
"""

from __future__ import annotations

import numpy as np

#: Links the frame is built from: (hand base, index MCP, middle MCP, pinky MCP).
HAND_KEYPOINTS = {
    "kuka_sharpa": (
        "left_hand_C_MC", "left_index_MCP_VL", "left_middle_MCP_VL", "left_pinky_MCP_VL",
    ),
    "flexiv_wuji": (
        "l_wrist", "l_index_finger_proximal", "l_middle_finger_proximal", "l_pinky_proximal",
    ),
}


def hand_frame(points) -> np.ndarray:
    """Canonical hand frame (4x4) from (base, index MCP, middle MCP, pinky MCP) positions."""
    base, index, middle, pinky = (np.asarray(p, dtype=float) for p in points)
    x = middle - base
    x /= np.linalg.norm(x)
    y = index - pinky
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    T[:3, 3] = middle
    return T


def hand_points(urdf, links, T_world_root=np.eye(4)) -> list[np.ndarray]:
    """World positions of ``links`` on a posed yourdfpy URDF whose root sits at ``T_world_root``."""
    return [
        (T_world_root @ urdf.get_transform(link, urdf.base_link))[:3, 3] for link in links
    ]
