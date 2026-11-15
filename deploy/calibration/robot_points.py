"""Points on the robot's visual meshes at a joint configuration, the scene's large planes (the table the
camera calibration levels to).

    model = RobotSurface(urdf, joint_names)
    pts, nrm = model.sample(q, links=ARM_LINKS)            # base frame
    planes = table_planes(cam_pts)                          # (a, b, c, d) per plane, camera frame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ARM_LINKS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7")


class RobotSurface:
    def __init__(self, urdf: str | Path, joint_names: list[str], spacing: float = 0.004) -> None:
        import yourdfpy

        self.urdf = yourdfpy.URDF.load(str(urdf), build_scene_graph=True, load_meshes=True)
        self.names = list(joint_names)
        self._order = [self.names.index(j) for j in self.urdf.actuated_joint_names]
        g = self.urdf.scene.graph
        # per visual geometry node: (link, points and normals in the node frame)
        self.nodes: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for node in g.nodes_geometry:
            link = g.transforms.parents[node]
            mesh = self.urdf.scene.geometry[g[node][1]]
            n = max(int(mesh.area / spacing ** 2), 50)
            pts, face = mesh.sample(n, return_index=True)
            self.nodes.append((node, link, np.asarray(pts), np.asarray(mesh.face_normals[face])))

    def sample(self, q: np.ndarray, links=None, exclude=None) -> tuple[np.ndarray, np.ndarray]:
        """Surface points and normals in the base frame at q (canonical joint order)."""
        self.urdf.update_cfg(np.asarray(q)[self._order])
        g = self.urdf.scene.graph
        P, N = [], []
        for node, link, pts, nrm in self.nodes:
            if (links is not None and link not in links) or (exclude is not None and link in exclude):
                continue
            T = g.get(node)[0]
            P.append(pts @ T[:3, :3].T + T[:3, 3])
            N.append(nrm @ T[:3, :3].T)
        return np.concatenate(P), np.concatenate(N)


def arm_tree(model: RobotSurface, q: np.ndarray) -> cKDTree:
    """KD-tree of the arm's surface points at q, for distances of base-frame points to the arm."""
    return cKDTree(model.sample(q, links=ARM_LINKS)[0])


def table_planes(pts: np.ndarray, max_planes: int = 4, min_frac: float = 0.05) -> list[np.ndarray]:
    """The scene's large planes as (a, b, c, d), a x + b y + c z + d = 0 with the unit normal toward the
    camera (the camera sees their front), one per orientation and offset: candidates for the table the
    robot stands on."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64))).voxel_down_sample(0.01)
    n0, planes = len(pc.points), []
    for _ in range(max_planes):
        if len(pc.points) < max(min_frac * n0, 3):
            break
        plane, inl = pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        if len(inl) < min_frac * n0:
            break
        plane = np.asarray(plane) / np.linalg.norm(plane[:3])
        # RANSAC's normal (from 3 points) is off by ~0.5 deg: least squares on all the inliers
        P = np.asarray(pc.points)
        for _ in range(3):
            on = P[np.abs(P @ plane[:3] + plane[3]) < 0.01]
            c = on.mean(0)
            n = np.linalg.svd(on - c, full_matrices=False)[2][2]
            plane = np.append(n, -n @ c)
        plane = plane if plane[3] > 0 else -plane  # camera (the origin) on the positive side
        if all(plane[:3] @ p[:3] < np.cos(np.radians(5)) or abs(plane[3] - p[3]) > 0.05 for p in planes):
            planes.append(plane)
        pc = pc.select_by_index(inl, invert=True)
    return planes
