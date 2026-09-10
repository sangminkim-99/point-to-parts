"""Step 6: write the discovered object out as a URDF with collision meshes.

Each part's Gaussian centres become a Poisson mesh for collision, and each
fitted joint contributes its type, axis, origin and observed limits.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np


def part_mesh(points, colors=None, depth=8, trim=0.20, voxel=0.004,
              margin=0.01):
    """Poisson surface from a part's Gaussian centres, low-density faces cut.

    Poisson closes an open cloud by inventing surface where it has no evidence,
    and the invention is outward. Measured on ikeasmall02, trimming only the
    bottom 2% of density left meshes up to 1.85x the true extent of their own
    points -- a drawer whose collision mesh was larger than the cabinet it
    slides in. Trimming the bottom 20% and then cropping to the points' own
    bounds brings every part to within 10% of its true size and still keeps
    three quarters of the triangles.
    """
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    if len(pc.points) < 100:
        return None
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.03, 30))
    pc.orient_normals_consistent_tangent_plane(20)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth)
    d = np.asarray(dens)
    if d.size:
        mesh.remove_vertices_by_mask(d < np.quantile(d, trim))
    # Poisson may still bulge past the evidence; the points bound the part.
    if margin >= 0:
        q = np.asarray(pc.points)
        box = o3d.geometry.AxisAlignedBoundingBox(q.min(0) - margin,
                                                  q.max(0) + margin)
        mesh = mesh.crop(box)
    mesh.compute_vertex_normals()
    return mesh if len(mesh.triangles) else None


def _xyz(e, v):
    e.set("xyz", " ".join(f"{float(x):.9g}" for x in v))



def joint_frames(jm):
    """Factor A(q) = origin @ motion(axis,q) @ child_offset.

    JointModel stores public axis/point in parent coordinates. URDF expresses
    the axis in its joint frame. A fixed helper link preserves the physical
    child's canonical mesh frame, including an off-origin hinge pivot.
    """
    A0 = np.asarray(jm.at(0.), dtype=float)
    axis = A0[:3, :3].T @ np.asarray(jm.axis, float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(A0).all() or not np.isfinite(axis).all() or norm < 1e-8:
        raise ValueError("invalid joint frame or axis")
    axis /= norm
    point = np.zeros(3)
    if jm.kind == 'revolute' and jm.point is not None:
        point = A0[:3, :3].T @ (np.asarray(jm.point) - A0[:3, 3])
    origin, offset = A0.copy(), np.eye(4)
    origin[:3, 3] += A0[:3, :3] @ point
    offset[:3, 3] = -point
    return origin, axis, offset


def _origin(joint, T):
    from scipy.spatial.transform import Rotation
    e = ET.SubElement(joint, 'origin')
    _xyz(e, T[:3, 3])
    e.set('rpy', ' '.join(f'{v:.9g}' for v in Rotation.from_matrix(T[:3, :3]).as_euler('xyz')))


def _tree(parts):
    if not parts:
        raise ValueError('cannot export an empty object')
    parents = [int(getattr(p, 'parent', j)) for j, p in enumerate(parts)]
    roots = [j for j, par in enumerate(parents) if j == par]
    if len(roots) != 1:
        raise ValueError('URDF requires one connected rooted tree')
    for j, par in enumerate(parents):
        if not 0 <= par < len(parts):
            raise ValueError('invalid parent index')
        if par == j:
            continue
        jm = getattr(parts[j], 'joint', None)
        if jm is None or jm.kind not in ('rigid', 'revolute', 'prismatic'):
            raise ValueError(f'part{j} has no supported fitted joint; cannot export a connected URDF')
        seen, k = set(), j
        while k != roots[0]:
            if k in seen:
                raise ValueError('cyclic part graph')
            seen.add(k)
            k = parents[k]
            if not 0 <= k < len(parts):
                raise ValueError('invalid parent index')
    return parents

def export(path, parts, points_of, colors_of=None, name="discovered",
           mesh_dir=None, mass=0.2):
    """Write <path>.urdf plus one collision mesh per part.

    `parts` are objects with .parent and .joint (a fitted JointModel); the joint
    axis and pivot are expressed in the parent frame. Physical link mesh frames
    are preserved using fixed joint-frame adapters.
    """
    path = os.fspath(path)
    parents = _tree(parts)
    mesh_dir = mesh_dir or os.path.join(os.path.dirname(path) or ".", "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    root = ET.Element("robot", {"name": name})
    written = {}
    for j, p in enumerate(parts):
        link = ET.SubElement(root, "link", {"name": f"part{j}"})
        pts = points_of(j)
        if pts is None or len(pts) < 100:
            # sparse tracks alone cannot carry a surface; step 4 supplies it
            print(f"[urdf] part{j}: {0 if pts is None else len(pts)} points, "
                  f"too few for a mesh (run with the gaussian model on)")
            continue
        m = part_mesh(pts, None if colors_of is None else colors_of(j))
        if m is None:
            continue
        rel = f"part{j}.obj"
        import open3d as o3d
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, rel), m)
        written[j] = rel
        c = np.asarray(pts).mean(0)
        for tag in ("visual", "collision"):
            v = ET.SubElement(link, tag)
            g = ET.SubElement(v, "geometry")
            ET.SubElement(g, "mesh", {"filename": os.path.relpath(os.path.join(mesh_dir, rel), os.path.dirname(path) or ".")})
        ine = ET.SubElement(link, "inertial")
        _xyz(ET.SubElement(ine, "origin"), c)
        ET.SubElement(ine, "mass", {"value": f"{mass}"})
        ET.SubElement(ine, "inertia", {"ixx": "1e-3", "iyy": "1e-3", "izz": "1e-3",
                                       "ixy": "0", "ixz": "0", "iyz": "0"})

    # Preserve the fitted tree. Re-rooting requires transforming complete
    # relative models, not just negating a parent-frame axis.
    for j, p in enumerate(parts):
        par = parents[j]
        if par == j:
            continue
        jm = p.joint
        if jm.kind == 'rigid':
            jt = ET.SubElement(root, 'joint', name=f'j{par}_{j}', type='fixed')
            ET.SubElement(jt, 'parent', link=f'part{par}')
            ET.SubElement(jt, 'child', link=f'part{j}')
            _origin(jt, jm.at(0.))
            continue
        origin, axis, offset = joint_frames(jm)
        helper = f'joint_frame{j}'
        ET.SubElement(root, 'link', name=helper)
        jt = ET.SubElement(root, 'joint', name=f'j{par}_{j}', type=jm.kind)
        ET.SubElement(jt, 'parent', link=f'part{par}')
        ET.SubElement(jt, 'child', link=helper)
        _origin(jt, origin)
        _xyz(ET.SubElement(jt, 'axis'), axis)
        vs = [jm.value_of(A) for A in jm.A] if jm.A else [0.]
        ET.SubElement(jt, 'limit', lower=f'{min(vs):.9g}', upper=f'{max(vs):.9g}',
                      effort='50', velocity='1.0')
        fixed = ET.SubElement(root, 'joint', name=f'j{j}_mesh_frame', type='fixed')
        ET.SubElement(fixed, 'parent', link=helper)
        ET.SubElement(fixed, 'child', link=f'part{j}')
        _origin(fixed, offset)
    ET.indent(root)
    out = path if path.endswith(".urdf") else path + ".urdf"
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out, written
