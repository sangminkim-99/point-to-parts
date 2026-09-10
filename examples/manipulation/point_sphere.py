"""Mesh-free observed-surface clearance, with optional cuRobo FK input.

Positive clearance is outside the inflated sampled surface, NOT proof of free
space. Unknown regions and object interiors are not represented by these points.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class PartCloud:
    part_id: int
    points: torch.Tensor  # [N,3], fixed part-local coordinates, metres
    pose: torch.Tensor    # [4,4], part -> robot base coordinates
    padding: float = .005 # sampling + estimation allowance, not a safety guarantee


def point_sphere_clearance(spheres, parts, chunk_size=2048):
    """Return [..., S, P] clearance; each part keeps its own identity.

    Spheres are [...,S,4] in robot-base frame. Radii <=0 are disabled (cuRobo
    convention). Empty clouds return +inf (no evidence), not verified free space.
    Piecewise differentiable in sphere centers/poses; no dense mesh or SDF.
    """
    if spheres.ndim < 2 or spheres.shape[-1] != 4 or chunk_size < 1:
        raise ValueError('expected [...,S,4] spheres and positive chunk_size')
    if not torch.isfinite(spheres).all():
        raise ValueError('sphere coordinates must be finite')
    if len({p.part_id for p in parts}) != len(parts):
        raise ValueError('part IDs must be unique')
    centers, radii = spheres[..., :3], spheres[..., 3]
    output = []
    for part in parts:
        pts, T = part.points, part.pose
        if pts.ndim != 2 or pts.shape[-1] != 3 or T.shape != (4, 4):
            raise ValueError('expected [N,3] points and [4,4] rigid pose')
        if not math.isfinite(part.padding) or part.padding < 0 or not torch.isfinite(pts).all() or not torch.isfinite(T).all():
            raise ValueError('finite geometry and nonnegative padding required')
        if pts.device != spheres.device or T.device != spheres.device or pts.dtype != spheres.dtype or T.dtype != spheres.dtype:
            raise ValueError('geometry and spheres must share device and dtype')
        eye = torch.eye(3, device=T.device, dtype=T.dtype)
        if not torch.allclose(T[:3, :3].T @ T[:3, :3], eye, atol=1e-4) or not torch.isclose(torch.linalg.det(T[:3, :3]), T.new_tensor(1.), atol=1e-4) or not torch.allclose(T[3], T.new_tensor([0., 0, 0, 1])):
            raise ValueError('part pose must be a rigid SE(3) transform')
        local = (centers - T[:3, 3]) @ T[:3, :3]
        distance = torch.full_like(radii, float('inf'))
        for chunk in pts.split(chunk_size):
            if len(chunk):
                d = torch.linalg.vector_norm(local.unsqueeze(-2)-chunk, dim=-1).amin(-1)
                distance = torch.minimum(distance, d)
        clearance = distance - radii - part.padding
        output.append(torch.where(radii > 0, clearance, torch.full_like(clearance, float('inf'))))
    return torch.stack(output, -1) if output else spheres.new_empty((*radii.shape, 0))


def curobo_clearance(robot_model, q, parts, chunk_size=2048):
    """Use cuRobo CudaRobotModel FK, then our point/sphere query.

    This is not an installed cuRobo MotionGen world-collision backend.
    """
    spheres = robot_model.get_state(q).link_spheres_tensor
    if spheres is None:
        raise ValueError('cuRobo robot configuration must include collision spheres')
    return point_sphere_clearance(spheres, parts, chunk_size)
