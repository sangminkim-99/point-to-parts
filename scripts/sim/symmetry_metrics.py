"""Evaluation-only surface displacement under explicitly supplied symmetries.

Inputs are absolute link-to-camera poses and object-link-local surface samples.
Never infer symmetries from predictions, and never feed these into tracking.
"""
import numpy as np


def surface_pose_errors(predicted, truth, points, symmetries=()):
    """Return strict and symmetry-aware RMS surface errors in metres.

    Each symmetry maps the object link frame to itself. Identity is always
    included. Symmetry alternatives use truth @ S, preserving off-origin
    symmetry centers. The complete transform is scored jointly, not independent
    minima for rotation and translation from different alternatives.
    """
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    points = np.asarray(points, float)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[1:] != (4,4):
        raise ValueError('poses must have matching (frames,4,4) shapes')
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('need nonempty (points,3) object-local surface samples')
    choices = [np.eye(4)] + [np.asarray(s, float) for s in symmetries]
    for T in (predicted, truth, np.array(choices)):
        if not np.isfinite(T).all() or T.shape[-2:] != (4,4):
            raise ValueError('invalid transforms')
        if not np.allclose(T[...,3,:], [0,0,0,1]):
            raise ValueError('invalid homogeneous row')
        R = T[...,:3,:3]
        if not np.allclose(R.swapaxes(-1,-2) @ R, np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(R),1,atol=1e-5):
            raise ValueError('transforms must be proper rigid motions')
    if not np.isfinite(points).all():
        raise ValueError('invalid surface samples')
    predicted_points = points @ predicted[:,:3,:3].transpose(0,2,1) + predicted[:,None,:3,3]
    errors=[]
    for S in choices:
        equivalent = truth @ S
        target = points @ equivalent[:,:3,:3].transpose(0,2,1) + equivalent[:,None,:3,3]
        errors.append(np.sqrt(np.mean(np.sum((predicted_points-target)**2,axis=-1),axis=-1)))
    errors=np.array(errors)
    return {'strict_rms_m': errors[0], 'symmetry_rms_m': errors.min(axis=0),
            'symmetry_index': errors.argmin(axis=0)}


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description='Evaluate keyframe_baseline trace; all geometry and symmetry are evaluation-only.')
    ap.add_argument('trace', type=Path)
    ap.add_argument('--points', required=True, type=Path, help='NPY object-link-local surface samples')
    ap.add_argument('--symmetries', type=Path, help='NPY (N,4,4), explicitly verified object-local symmetries')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('output already exists; preserve previous evaluations')
    with np.load(args.trace, allow_pickle=False) as z:
        truth, observed = z['gt_poses'], z['observed'].astype(bool)
        # Keyframe baseline maps initial camera coordinates to current camera.
        # Geometry/symmetries live in the physical object link frame.
        predicted = z['poses'] @ truth[0]
    points = np.load(args.points, allow_pickle=False)
    symmetries = () if args.symmetries is None else np.load(args.symmetries, allow_pickle=False)
    values = surface_pose_errors(predicted, truth, points, symmetries)
    result = {'trace': str(args.trace), 'points': str(args.points),
              'symmetries': str(args.symmetries) if args.symmetries else None,
              'scope': 'evaluation only; supplied symmetries must preserve object appearance and geometry',
              'frames': len(truth), 'observed_frames': int(observed.sum())}
    for name in ('strict_rms_m', 'symmetry_rms_m'):
        result[name] = {}
        for label, keep in (('all', np.ones(len(truth), bool)), ('observed', observed), ('held', ~observed)):
            x = values[name][keep]
            result[name][label] = {'count': len(x), 'median': float(np.median(x)) if len(x) else None,
                                  'max': float(np.max(x)) if len(x) else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
