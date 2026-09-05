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
    e.set("xyz", " ".join(f"{float(x):.6f}" for x in v))


def export(path, parts, points_of, colors_of=None, name="discovered",
           mesh_dir=None, mass=0.2):
    """Write <path>.urdf plus one collision mesh per part.

    `parts` are objects with .parent and .joint (a fitted JointModel); the joint
    frame is the parent's frame, which is how the joint was estimated.
    """
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
            ET.SubElement(g, "mesh", {"filename": f"meshes/{rel}"})
        ine = ET.SubElement(link, "inertial")
        _xyz(ET.SubElement(ine, "origin"), c)
        ET.SubElement(ine, "mass", {"value": f"{mass}"})
        ET.SubElement(ine, "inertia", {"ixx": "1e-3", "iyy": "1e-3", "izz": "1e-3",
                                       "ixy": "0", "ixz": "0", "iyz": "0"})

    # The tracker's root is part 0, which is simply the part that existed
    # first, and that is often a drawer rather than the body it slides in. A
    # URDF rooted on a drawer is kinematically fine and reads as nonsense: you
    # hold the drawer still and push the cabinet. Measured on ikeasmall02, the
    # axis was within 1 degree of the truth and pointed the opposite way for
    # exactly this reason. So the biggest part becomes the root here, and any
    # relation that has to be reversed is inverted rather than re-fitted --
    # the child moves by +axis q relative to the parent, so the parent moves
    # by -axis q relative to the child, and the observed range flips with it.
    def _extent(j):
        q = points_of(j)
        if q is None or len(q) < 4:
            return 0.0
        q = np.asarray(q)
        return float(np.linalg.norm(q.max(0) - q.min(0)))
    # by size, not by point count: on ikeasmall02 the body has the FEWEST
    # gaussians of the three parts, 5,510 against 11,000 for each drawer,
    # because the drawers face the camera and the body is mostly behind them
    hub = max(range(len(parts)), key=_extent) if len(parts) else 0
    for j, p in enumerate(parts):
        jm = getattr(p, "joint", None)
        par = getattr(p, "parent", 0)
        if jm is None or jm.kind is None or par == j or par >= len(parts):
            continue
        vs = [jm.value_of(A) for A in jm.A] if jm.A else [0.0]
        axis, lo, hi = np.asarray(jm.axis, float), min(vs), max(vs)
        if j == hub:                    # this relation points the wrong way
            par, j, axis, lo, hi = j, par, -axis, -hi, -lo
        jt = ET.SubElement(root, "joint", {"name": f"j{par}_{j}",
                                           "type": jm.kind})
        ET.SubElement(jt, "parent", {"link": f"part{par}"})
        ET.SubElement(jt, "child", {"link": f"part{j}"})
        _xyz(ET.SubElement(jt, "origin"),
             jm.point if (jm.kind == "revolute" and jm.point is not None)
             else np.zeros(3))
        _xyz(ET.SubElement(jt, "axis"), axis)
        # only the range that was actually observed is claimed as the limit
        ET.SubElement(jt, "limit", {"lower": f"{lo:.4f}", "upper": f"{hi:.4f}",
                                    "effort": "50", "velocity": "1.0"})
    ET.indent(root)
    out = path if path.endswith(".urdf") else path + ".urdf"
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out, written
