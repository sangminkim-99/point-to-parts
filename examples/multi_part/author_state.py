"""Immutable demo snapshots and mesh-free kinematic export. No camera/GPU I/O."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np


def snapshot(stream):
    parts = [SimpleNamespace(part_id=p.part_id, parent=p.parent,
                            pose=p.pose.copy(), observed=bool(p.observed),
                            joint=copy.deepcopy(p.joint)) for p in stream.parts]
    if stream.model is not None:
        points = stream.model.cloud.means.detach().cpu().numpy().copy()
        colors = stream.model.cloud.colors.detach().cpu().numpy().copy()
        labels = stream.model.labels.copy()
    else:
        points = np.concatenate([stream.anchor_xyz[p.idx] for p in stream.parts])
        labels = np.concatenate([np.full(len(p.idx), j) for j,p in enumerate(stream.parts)])
        colors = np.full_like(points, .7)
    return SimpleNamespace(parts=parts, points=points, colors=colors,
                           labels=labels, root=int(stream.root),
                           frame=int(stream.n - 1), K=stream.K.copy())


def union_mask(seg, background=255):
    """Binary UNION mask from a renderer part-index seg image.

    ``scripts/sim/render_partnet_sequence.py`` writes seg as a uint8 part index
    with 255 = background, so the object (all parts, no per-part labels) is
    ``seg != 255``. The naive ``seg > 0`` grabs the background instead and leaves
    the tracker with zero valid points -- keep this rule in one tested place.
    """
    return (np.asarray(seg) != background).astype(np.uint8)


def cloud_center_radius(points, pct=95):
    """Centroid and a robust (percentile) radius of a display-frame cloud."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros(3), 0.0
    c = p.mean(0)
    d = np.linalg.norm(p - c, axis=1)
    return c, float(np.percentile(d, pct))


# Oblique viewing direction (camera relative to look-at), consistent with the
# scene's -y up. A pure viewing choice; it never touches model coordinates.
VIEW_DIR = np.array([0., -0.28, -0.96])

# Single source of truth for the per-part display palette (RGB 0-255) and names,
# shared by the 3D cloud and the legend so a swatch and its label always agree.
PALETTE = np.array([[239, 155, 56], [65, 193, 163], [155, 115, 232], [72, 163, 230]])
PALETTE_NAMES = ['orange', 'teal', 'purple', 'blue']


def geometry_counts(labels, n_parts):
    """Points assigned to each part vs unassigned (label not in [0, n_parts)).

    Unassigned geometry (dense label -1, or any label outside the current parts)
    has no part pose; the viewer never gives it a fabricated one, so counting it
    is how the user sees it exists at all.
    """
    labels = np.asarray(labels)
    per_part = [int((labels == j).sum()) for j in range(n_parts)]
    assigned = int(sum(per_part))
    return dict(per_part=per_part, assigned=assigned,
                unassigned=int(labels.size - assigned))


def legend_markdown(part_ids, counts):
    """A legend that names each part's colour and its point count, and states
    the unassigned (hidden, un-posed) count explicitly."""
    lines = ['**Legend** — part colors when Color by part is enabled:', '']
    for k, pid in enumerate(part_ids):
        name = PALETTE_NAMES[pid % len(PALETTE_NAMES)]
        lines.append(f'- {name} — part {pid}: {counts["per_part"][k]:,} pts')
    lines.append(f'- unassigned (hidden, no part pose): {counts["unassigned"]:,} pts')
    return '\n'.join(lines)


def frame_view(center, radius, factor=3.2, min_dist=0.3, max_dist=1.1,
               direction=VIEW_DIR, up=(0., -1., 0.)):
    """Viser camera pose that frames a cloud of the given centroid/radius.

    Centering on the cloud centroid keeps the object filling the view even when
    the reference origin (the initial camera pose) sits far from the geometry.
    """
    center = np.asarray(center, float).reshape(3)
    dist = float(np.clip(factor * max(radius, 1e-3), min_dist, max_dist))
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.)
    return dict(position=tuple(center + d * dist),
                look_at=tuple(center), up=tuple(up))


def displayed_poses(state, root_view=True, joint_index=None, fraction=0.):
    """Preview a copied joint configuration; never modify the tracked state."""
    poses = [p.pose.copy() for p in state.parts]
    if joint_index is not None:
        p = state.parts[joint_index]
        if p.joint is not None and p.joint.kind in ('revolute', 'prismatic'):
            limits = p.joint.limits()
            if limits is not None and np.isfinite(limits).all():
                q = limits[0] + np.clip(fraction, 0, 1) * (limits[1] - limits[0])
                new_pose = poses[p.parent] @ p.joint.at(q)
                delta = new_pose @ np.linalg.inv(poses[joint_index])
                for j in range(len(poses)):
                    k, seen = j, set()
                    while k not in seen:
                        if k == joint_index:
                            poses[j] = delta @ poses[j]
                            break
                        seen.add(k)
                        k = state.parts[k].parent
                poses[joint_index] = new_pose
    if root_view:
        C = np.linalg.inv(state.parts[state.root].pose)
        poses = [C @ p for p in poses]
    return poses


def save_snapshot(state, directory, source=None):
    """Clouds always save; URDF only for a supported fitted tree."""
    from examples.multi_part.urdf_export import _tree, _origin, _xyz, joint_frames
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'model.npz', means=state.points,
                        colors=state.colors, labels=state.labels,
                        poses=np.stack([p.pose for p in state.parts]),
                        part_ids=[p.part_id for p in state.parts],
                        parents=[p.parent for p in state.parts], K=state.K,
                        frame=state.frame,
                        coordinate_frame='per_part_anchor; poses map anchor to camera')
    meta = dict(frame=state.frame, reference_part_id=state.parts[state.root].part_id,
                observed=[p.observed for p in state.parts],
                capture_source=source,
                status='user-saved estimate; not validated for robot execution',
                collision='external part-local point clouds; no URDF collision geometry',
                limits='observed ranges, not verified mechanical limits')
    try:
        parents = _tree(state.parts)
        robot = ET.Element('robot', name='authored_object')
        for j in range(len(state.parts)):
            ET.SubElement(robot, 'link', name=f'part{j}')
        for j, p in enumerate(state.parts):
            par = parents[j]
            if par == j:
                continue
            jm = p.joint
            if jm.kind == 'rigid':
                joint = ET.SubElement(robot, 'joint', name=f'j{par}_{j}', type='fixed')
                ET.SubElement(joint, 'parent', link=f'part{par}')
                ET.SubElement(joint, 'child', link=f'part{j}')
                _origin(joint, jm.at(0.))
                continue
            origin, axis, offset = joint_frames(jm)
            limits = jm.limits()
            if limits is None or not np.isfinite(limits).all():
                raise ValueError(f'part {j} has no finite observed range')
            ET.SubElement(robot, 'link', name=f'joint_frame{j}')
            joint = ET.SubElement(robot, 'joint', name=f'j{par}_{j}', type=jm.kind)
            ET.SubElement(joint, 'parent', link=f'part{par}')
            ET.SubElement(joint, 'child', link=f'joint_frame{j}')
            _origin(joint, origin)
            _xyz(ET.SubElement(joint, 'axis'), axis)
            ET.SubElement(joint, 'limit', lower=str(limits[0]), upper=str(limits[1]),
                          effort='1', velocity='1')
            fixed = ET.SubElement(robot, 'joint', name=f'offset{j}', type='fixed')
            ET.SubElement(fixed, 'parent', link=f'joint_frame{j}')
            ET.SubElement(fixed, 'child', link=f'part{j}')
            _origin(fixed, offset)
        ET.indent(robot)
        ET.ElementTree(robot).write(directory / 'object.urdf', encoding='utf-8',
                                    xml_declaration=True)
        meta['urdf'] = 'object.urdf'
    except ValueError as exc:
        meta['urdf'] = None
        meta['urdf_reason'] = str(exc)
    (directory / 'metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
    return meta
