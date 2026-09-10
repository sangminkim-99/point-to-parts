"""Offline oracle diagnostic: which GT surfaces support a saved recovery pose?

GT labels are read here only, never by the online recovery search.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.multi_part.surface_memory import match_joint_surface
from point2pose.io.sources.dataset.sapien_reader import SapienReader


def support_labels(points, pose, depth, K, part_map, tolerance, mask=None):
    q = points @ pose[:3, :3].T + pose[:3, 3]
    z = q[:, 2]
    uv = q @ K.T
    xy = np.rint(uv[:, :2] / np.maximum(z[:, None], 1e-6)).astype(int)
    h, w = depth.shape
    valid = np.isfinite(q).all(1) & (z > .05) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    u, v = np.clip(xy[:, 0], 0, w-1), np.clip(xy[:, 1], 0, h-1)
    valid &= (depth[v, u] > .05) & (np.abs(z-depth[v, u]) < tolerance) & (part_map[v, u] >= 0)
    if mask is not None:
        valid &= mask[v, u] > 0
    labels, counts = np.unique(part_map[v[valid], u[valid]], return_counts=True)
    return {int(k): int(n) for k, n in zip(labels, counts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshots', type=Path, required=True)
    ap.add_argument('--seq-dir', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    reader = SapienReader(args.seq_dir)
    rows = []
    for path in sorted(args.snapshots.glob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            result = match_joint_surface(d['points'], d['poses'], d['depth'], d['mask'], d['K'],
                tolerance=float(d['tolerance']), min_fraction=float(d['min_fraction']), margin=float(d['margin']))
            row = {'snapshot': str(path), 'frame': int(d['frame']), 'part_id': int(d['part_id']),
                   'distinct_match': result is not None}
            if result is not None:
                counts = support_labels(d['points'], result['pose'], d['depth'], d['K'],
                    reader.render_part_index_map(row['frame']), float(d['tolerance']), d['mask'])
                row.update(index=result['index'], score=result['score'], support=result['support'],
                    gt_support_counts={reader.object_names[k]: n for k, n in counts.items()})
            rows.append(row)
    if not rows:
        ap.error('no snapshot files found')
    result = {'scope': 'offline GT-label diagnostic, not online input or an ablation', 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
