"""Conservative surface clearance between every robot collider and the rigid cloth.

Actual collision geometry is enclosed by body-local boxes, then by world-axis
boxes after each body rotation. Their separation is a lower bound on physical
surface distance. Passing a 10 cm gate therefore guarantees at least that much
clearance; a rotated or curved shape can conservatively delay release.
"""
from __future__ import annotations

import numpy as np
import torch

from ...mdp.geometry import rotations


def world_bounds(center, extent, position, quaternion):
    """Enclose link-local boxes in world-axis boxes; tensors broadcast over envs."""
    center, extent = center.expand_as(position), extent.expand_as(position)
    rotation = rotations.to_matrix(quaternion)
    return position + rotations.apply(quaternion, center), torch.einsum('...ij,...j->...i', rotation.abs(), extent)


def box_clearance(robot_center, robot_extent, cloth_center, cloth_extent, chunk_size=16):
    """Minimum box-to-box surface separation per env, including every shape pair.

    Nonfinite geometry fails closed with zero clearance. Shape chunks bound the
    temporary tensor size for large RL batches.
    """
    result = robot_center.new_full((robot_center.shape[0],), float('inf'))
    for first in range(0, robot_center.shape[1], chunk_size):
        center, extent = robot_center[:, first:first+chunk_size], robot_extent[:, first:first+chunk_size]
        gap = ((center[:, :, None] - cloth_center[:, None]).abs()
               - extent[:, :, None] - cloth_extent[:, None]).clamp_min(0)
        squared = gap.square().sum(dim=-1).amin(dim=(1, 2))
        result = torch.minimum(result, squared)
    return torch.nan_to_num(result.sqrt(), nan=0., posinf=0., neginf=0.)


def robot_collision_bounds(robot):
    """Read enabled USD collision shapes once, expressed in their body-link frames."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = get_current_stage()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False, ignoreVisibility=True,
    )
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    body_ids, centers, extents = [], [], []
    for body_id, path in enumerate(robot.root_physx_view.link_paths[0]):
        body = stage.GetPrimAtPath(path)
        if not body.IsValid():
            raise RuntimeError(f'Robot body missing from USD stage: {path}')
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            # A nested articulated body owns its own collision shapes.
            ancestor = prim
            while ancestor != body and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor = ancestor.GetParent()
            if ancestor != body:
                continue
            bounds = cache.ComputeRelativeBound(prim, body).ComputeAlignedBox()
            if bounds.IsEmpty():
                # Importers can omit extents or mark collision meshes as guide
                # geometry. Read actual points if USD's bound cache cannot help.
                if prim.IsA(UsdGeom.Mesh):
                    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                else:
                    boundable = UsdGeom.Boundable(prim)
                    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, Usd.TimeCode.Default())
                    if extent is None or len(extent) != 2:
                        raise RuntimeError(f'Robot collider has no geometric bounds: {prim.GetPath()}')
                    low, high = map(np.asarray, extent)
                    points = np.array([[x, y, z] for x in (low[0], high[0])
                                       for y in (low[1], high[1]) for z in (low[2], high[2])])
                if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                    raise RuntimeError(f'Robot collider has no geometric points: {prim.GetPath()}')
                relative = np.asarray(transforms.GetLocalToWorldTransform(prim)
                                      * transforms.GetLocalToWorldTransform(body).GetInverse())
                points = points @ relative[:3, :3] + relative[3, :3]
                low, high = points.min(axis=0), points.max(axis=0)
            else:
                low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
            center, extent = (low+high)/2, (high-low)/2
            if not np.isfinite(center).all() or not np.isfinite(extent).all():
                raise RuntimeError(f'Robot collider has nonfinite bounds: {prim.GetPath()}')
            body_ids.append(body_id)
            centers.append(center)
            extents.append(extent)
    if not body_ids:
        raise RuntimeError('Whole-robot release requires at least one enabled robot collision shape')
    return body_ids, np.asarray(centers), np.asarray(extents)


class RobotClothClearance:
    """All arm, base, palm and finger colliders against all real cloth plates."""

    def __init__(self, backend):
        self.backend = backend
        ids, centers, extents = robot_collision_bounds(backend.robot)
        self.body_ids = torch.tensor(ids, device=backend.device, dtype=torch.long)
        self.centers = torch.as_tensor(centers, device=backend.device, dtype=torch.float32)
        self.extents = torch.as_tensor(extents, device=backend.device, dtype=torch.float32)
        self.plates = []
        for group in backend.groups:
            centers, extents = [], []
            for plate in group.design.plates:
                vertices = np.asarray(plate.vertices)
                low, high = vertices.min(0), vertices.max(0)
                centers.append([*(low+high)/2, 0.])
                extents.append([*(high-low)/2, plate.thickness/2])
            dimensions = backend._dims[group.env_ids, None]
            center = torch.as_tensor(centers, device=backend.device, dtype=dimensions.dtype)[None] * dimensions
            extent = torch.as_tensor(extents, device=backend.device, dtype=dimensions.dtype)[None] * dimensions
            self.plates.append((group, center, extent))
        print(f'[cloth] release clearance: {len(ids)} robot collision bounds against every cloth plate', flush=True)

    def distance(self):
        backend = self.backend
        robot = backend.robot.data
        centers, extents = world_bounds(self.centers[None], self.extents[None],
                                        robot.body_link_pos_w[:, self.body_ids],
                                        robot.body_link_quat_w[:, self.body_ids])
        distance = centers.new_empty(backend.num_envs)
        for group, local_center, local_extent in self.plates:
            data, bodies = group.articulation.data, group.body_ids
            cloth_center, cloth_extent = world_bounds(local_center, local_extent,
                                                       data.body_link_pos_w[:, bodies],
                                                       data.body_link_quat_w[:, bodies])
            distance[group.env_ids] = box_clearance(centers[group.env_ids], extents[group.env_ids],
                                                    cloth_center, cloth_extent)
        return distance
